import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import time # เพิ่มไลบรารี time สำหรับระบบ Cooldown ของปุ่มล่องหน

sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# ฟังก์ชันพิเศษ: สำหรับวางรูปภาพ PNG (มีพื้นหลังใส) ทับบนวิดีโอ
# ==========================================
def overlay_transparent(background, overlay, x, y):
    bg_h, bg_w, _ = background.shape
    ov_h, ov_w, ov_c = overlay.shape

    # เช็คว่ารูปภาพล้นจอหรือไม่ ถ้าล้นให้ข้ามไปเพื่อป้องกัน Error
    if x >= bg_w or y >= bg_h or x + ov_w <= 0 or y + ov_h <= 0:
        return background
    if x < 0 or y < 0 or x + ov_w > bg_w or y + ov_h > bg_h:
        return background # เพื่อความชัวร์ จะไม่วาดถ้ารูปล้นขอบจอ

    if ov_c == 4: # ถ้ารูปมี Alpha Channel (โปร่งใส)
        alpha = overlay[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha
        for c in range(3):
            background[y:y+ov_h, x:x+ov_w, c] = (alpha * overlay[:, :, c] + alpha_inv * background[y:y+ov_h, x:x+ov_w, c])
    else:
        background[y:y+ov_h, x:x+ov_w] = overlay # ถ้ารูปไม่มี Alpha ก็แปะทับทื่อๆ เลย
        
    return background

# ==========================================
# 1. ตั้งค่า MediaPipe (AI จับท่าทาง)
# ==========================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
# ใช้ Model Complexity = 1 (กำลังดีสำหรับ WebCam/Jetson)
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6, model_complexity=1)

# ==========================================
# 2. ตั้งค่าระบบควบคุม (Control Variables)
# ==========================================
# 2.1 ตัวแปรสำหรับระบบกันกระชาก (EMA Filter)
# Alpha คือความไว: ค่าเข้าใกล้ 1 จะตามมือเร็วแต่สั่นง่าย / ค่าเข้าใกล้ 0 จะสมูทมากแต่ตามมือช้า
EMA_ALPHA = 0.15  
current_smoothed_pos = np.array([0.0, 0.0, 0.0])
current_smoothed_gripper = 0.0 # เพิ่มตัวแปรกริปเปอร์
is_first_frame = True

# 2.2 ตัวแปรสำหรับ Mark ตำแหน่ง
marked_position = None
is_robot_moving_to_mark = False
is_holding = False # ตัวแปรใหม่สำหรับสถานะ Hold Position

# 2.3 ตัวแปรสำหรับ Gesture มือซ้าย (Cooldown ป้องกันลั่น)
last_hold_toggle_time = 0
last_mark_time = 0

# ==========================================
# 3. โหลดไฟล์ตั้งค่า 3D (ถ้ามีกล้อง 2 ตัว)
# ==========================================
STEREO_MODE = False
P1, P2 = None, None

if os.path.exists("stereo_params.npz"):
    try:
        params = np.load("stereo_params.npz")
        P1, P2 = params['P1'], params['P2']
        STEREO_MODE = True
        print("✅ โหลดไฟล์ Stereo สำเร็จ: ทำงานในโหมด 3D เต็มรูปแบบ")
    except Exception as e:
        print(f"⚠️ โหลดไฟล์ Stereo ไม่สำเร็จ ({e})")

if not STEREO_MODE:
    print("⚠️ ไม่พบไฟล์ stereo_params.npz (หรือพัง)")
    print("👉 ทำงานในโหมดทดสอบ 'กล้องเดี่ยว (2D)' เพื่อทดสอบระบบกันกระชากและ Mark ตำแหน่ง")

# โหลดไฟล์รูปภาพไอคอน (ถ้ามี) สำหรับแสดงด้านล่างจอ
# แนะนำให้ใช้รูปขนาดประมาณ 60x60 - 80x80 pixels
icon_hold = cv2.imread("icon_hold.png", cv2.IMREAD_UNCHANGED)
icon_mark = cv2.imread("icon_mark.png", cv2.IMREAD_UNCHANGED)

if icon_hold is None or icon_mark is None:
    print("⚠️ ไม่พบไฟล์รูปภาพ 'icon_hold.png' หรือ 'icon_mark.png' ระบบจะแสดงผลเป็นข้อความแทน")

# ==========================================
# 4. ฟังก์ชันสำหรับหาจุด 3D (Triangulation)
# ==========================================
def get_3d_point(p_left, p_right, proj1, proj2):
    # รับพิกัด [x, y] ของกล้องซ้ายและขวา มาคำนวณจุดตัด 3 มิติ
    pt_l = np.array([[p_left[0]], [p_left[1]]], dtype=np.float32)
    pt_r = np.array([[p_right[0]], [p_right[1]]], dtype=np.float32)
    
    pt_4d = cv2.triangulatePoints(proj1, proj2, pt_l, pt_r)
    # แปลงกลับเป็น 3D (X, Y, Z)
    pt_3d = pt_4d[:3] / pt_4d[3]
    return pt_3d.flatten()

# ==========================================
# 5. เปิดกล้องและรันลูปหลัก
# ==========================================
# หมายเหตุ: ถ้ามีกล้องตัวเดียวใช้ cv2.VideoCapture(0)
cam_left = cv2.VideoCapture(0)
cam_right = cv2.VideoCapture(1) if STEREO_MODE else None

# --- ทำให้ภาพแสดงผลเป็นหน้าต่างปกติ (สามารถใช้เมาส์ย่อ-ขยายได้) ---
cv2.namedWindow("Vision Tracking (Robot Target)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Vision Tracking (Robot Target)", 1024, 768) # กำหนดขนาดหน้าต่างเริ่มต้น (ปรับเปลี่ยนได้)

print("==========================================")
print("คีย์ลัดสำหรับควบคุม:")
print(" - กด [H] : เปิด/ปิด โหมด Hold (หยุดหุ่นชั่วคราว)")
print(" - กด [M] : เพื่อ Mark ตำแหน่ง (จุดปลอดภัย)")
print(" - กด [R] : สั่งหุ่นยนต์กลับไปจุด Mark (Return)")
print(" - กด [Q] : ออกจากโปรแกรม")
print("==========================================")

while True:
    ret_l, frame_l = cam_left.read()
    if not ret_l:
        break
    
    # สลับซ้ายขวา (Mirror) ของภาพซ้าย เพื่อให้ควบคุมง่ายขึ้นเหมือนส่องกระจก
    frame_l = cv2.flip(frame_l, 1)
    
    frame_r = None
    if STEREO_MODE and cam_right is not None:
        ret_r, frame_r = cam_right.read()
        if ret_r:
            # สลับซ้ายขวา (Mirror) ของภาพขวาด้วยเช่นกัน
            frame_r = cv2.flip(frame_r, 1)

    # 5.1 นำภาพให้ AI ประมวลผล (ใช้ภาพซ้ายเป็นหลัก)
    rgb_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB)
    results_l = pose.process(rgb_l)
    
    # วาดโครงกระดูกลงบนภาพ
    if results_l.pose_landmarks:
        mp_drawing.draw_landmarks(frame_l, results_l.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 5.2 หาตำแหน่ง TCP (Tool Center Point) แทนข้อมือ
    raw_target_pos = None
    raw_gripper_percent = None # เพิ่มตัวรับค่ากริปเปอร์ดิบ

    if results_l.pose_landmarks:
        # --- ส่วนที่ 1: ตรวจจับท่าทางมือซ้าย (ทำเป็นปุ่มล่องหน) ---
        left_wrist = results_l.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        left_ear = results_l.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR]
        right_shoulder = results_l.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # เช็คว่ากล้องเห็นข้อมือซ้ายชัดเจนไหม
        if left_wrist.visibility > 0.6:
            h, w, _ = frame_l.shape
            
            # หาระยะห่าง 3D สำหรับท่าทางใหม่ (อ้างอิงกับสรีระร่างกาย)
            dist_to_ear = np.sqrt((left_wrist.x - left_ear.x)**2 + 
                                   (left_wrist.y - left_ear.y)**2 + 
                                   (left_wrist.z - left_ear.z)**2)
            
            dist_to_shoulder = np.sqrt((left_wrist.x - right_shoulder.x)**2 + 
                                       (left_wrist.y - right_shoulder.y)**2 + 
                                       (left_wrist.z - right_shoulder.z)**2)
            
            current_time = time.time()
            
            # ท่าที่ 1: เอามือซ้าย แตะหูซ้าย = เปิด/ปิด Hold (มี Cooldown 1 วินาที)
            if dist_to_ear < 0.15 and (current_time - last_hold_toggle_time > 1.0):
                is_holding = not is_holding
                last_hold_toggle_time = current_time
                state_text = "ON" if is_holding else "OFF"
                print(f"✋ LEFT HAND COMMAND: Toggle Hold -> {state_text}")
                # วาดวงกลมแฟลชสีส้มเตือนว่ากดติดแล้ว
                cv2.circle(frame_l, (int(left_wrist.x * w), int(left_wrist.y * h)), 40, (0, 165, 255), -1)
            
            # ท่าที่ 2: เอามือซ้าย แตะไหล่ขวา (ท่าไขว้แขน) = Mark ตำแหน่ง (มี Cooldown 2 วินาที)
            if dist_to_shoulder < 0.15 and (current_time - last_mark_time > 2.0):
                marked_position = current_smoothed_pos.copy() if not is_first_frame else np.array([0,0,0])
                last_mark_time = current_time
                print(f"📌 LEFT HAND COMMAND: Mark Position -> {marked_position}")
                # วาดวงกลมแฟลชสีฟ้าเตือนว่ากดติดแล้ว
                cv2.circle(frame_l, (int(left_wrist.x * w), int(left_wrist.y * h)), 40, (255, 255, 0), -1)

        # --- ส่วนที่ 2: ดึงข้อมูลมือขวา (ควบคุมเป้าหมายหุ่นยนต์) ---
        # ดึงข้อมูลนิ้วชี้ (Landmark 20) และนิ้วโป้ง (Landmark 22)
        index_finger = results_l.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
        thumb_finger = results_l.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB]
        
        # เช็คว่ากล้องเห็นมือชัดเจนหรือไม่ (ใช้ค่าเฉลี่ยความมั่นใจของ 2 นิ้ว)
        if (index_finger.visibility + thumb_finger.visibility) / 2 > 0.6:
            h, w, _ = frame_l.shape
            
            # คำนวณหาจุดกึ่งกลางระหว่างนิ้วชี้และนิ้วโป้ง (ตำแหน่งตรงกลาง Gripper)
            tcp_x = int(((index_finger.x + thumb_finger.x) / 2) * w)
            tcp_y = int(((index_finger.y + thumb_finger.y) / 2) * h)
            
            px_l = [tcp_x, tcp_y]
            
            if STEREO_MODE and frame_r is not None:
                # โหมด 3D: ต้องรัน AI ภาพขวาด้วย แล้วทำ Triangulate
                pass 
            else:
                # โหมดทดสอบกล้องเดี่ยว: จำลองพิกัด 3D โดยให้ Z เป็น 0
                raw_target_pos = np.array([px_l[0], px_l[1], 0.0])
                
                # วาดวงกลมสีแดงที่ TCP (จุดหนีบของ Gripper)
                cv2.circle(frame_l, (px_l[0], px_l[1]), 10, (0, 0, 255), -1)
                
                # วาดเส้นเชื่อมระหว่างนิ้วชี้กับนิ้วโป้ง (จำลองปาก Gripper)
                idx_x, idx_y = int(index_finger.x * w), int(index_finger.y * h)
                thb_x, thb_y = int(thumb_finger.x * w), int(thumb_finger.y * h)
                cv2.line(frame_l, (idx_x, idx_y), (thb_x, thb_y), (0, 255, 255), 2)

                # --- ส่วนคำนวณระยะกริปเปอร์ ---
                # ใช้ระยะห่าง 3D เพื่อไม่ให้ค่าเพี้ยนเวลาคนยื่นมือเข้าใกล้หรือออกห่างกล้อง
                dist_3d = np.sqrt((index_finger.x - thumb_finger.x)**2 + 
                                  (index_finger.y - thumb_finger.y)**2 + 
                                  (index_finger.z - thumb_finger.z)**2)
                
                # แปลงระยะห่างเป็นเปอร์เซ็นต์การหนีบ (0% = ปิดสนิท, 100% = กางสุด)
                # !! อาจต้องปรับ MIN_DIST และ MAX_DIST ให้เข้ากับขนาดมือคนคุม !!
                MIN_DIST = 0.02 # ระยะตอนปลายนิ้วชนกัน
                MAX_DIST = 0.12 # ระยะตอนกางนิ้วสุด
                raw_gripper_percent = np.clip((dist_3d - MIN_DIST) / (MAX_DIST - MIN_DIST) * 100, 0, 100)

    # 5.3 ระบบกันกระชาก (EMA Filter) & อัปเดตพิกัดส่งให้หุ่น
    if raw_target_pos is not None and raw_gripper_percent is not None:
        if is_first_frame:
            current_smoothed_pos = raw_target_pos
            current_smoothed_gripper = raw_gripper_percent
            is_first_frame = False
        elif not is_holding: # อัปเดตตำแหน่งเฉพาะตอนที่ไม่ได้ Hold อยู่
            # สมการ EMA: จุดใหม่ = (ความไว * จุดดิบจากกล้อง) + ((1 - ความไว) * จุดเก่า)
            current_smoothed_pos = (EMA_ALPHA * raw_target_pos) + ((1.0 - EMA_ALPHA) * current_smoothed_pos)
            current_smoothed_gripper = (EMA_ALPHA * raw_gripper_percent) + ((1.0 - EMA_ALPHA) * current_smoothed_gripper)
            
        # วาดวงกลมสีเขียว (จุดที่ผ่านการ Smooth แล้ว ซึ่งจะวิ่งตามข้อมือแบบนุ่มๆ)
        cv2.circle(frame_l, (int(current_smoothed_pos[0]), int(current_smoothed_pos[1])), 8, (0, 255, 0), 2)
        
        # !! ตรงนี้คือจุดที่คุณจะแพ็กข้อมูล X,Y,Z ส่งผ่าน MQTT/Serial ไปให้ Raspberry Pi !!
        # print(f"ส่งข้อมูล -> X:{current_smoothed_pos[0]:.1f}, Y:{current_smoothed_pos[1]:.1f}, Gripper:{current_smoothed_gripper:.1f}%")

    # ==========================================
    # 6. วาด UI (User Interface) บนภาพ
    # ==========================================
    h, w, _ = frame_l.shape

    # สร้างกรอบพื้นหลังสีดำโปร่งแสงสำหรับ UI สถานะ (มุมซ้ายบน)
    overlay = frame_l.copy()
    cv2.rectangle(overlay, (10, 10), (320, 120), (0, 0, 0), -1)
    
    # --- เพิ่ม: กรอบสีดำโปร่งแสงด้านล่าง สำหรับแสดง Guide ท่าทาง ---
    bottom_bar_height = 100
    cv2.rectangle(overlay, (0, h - bottom_bar_height), (w, h), (0, 0, 0), -1)
    
    cv2.addWeighted(overlay, 0.6, frame_l, 0.4, 0, frame_l)

    if is_holding:
        status_text = "HOLDING (Manual)"
        color = (0, 165, 255) # เปลี่ยน UI เป็นสีส้มเมื่อถูก Hold
    else:
        status_text = "Tracking TCP" if raw_target_pos is not None else "Vision Lost! (Holding Pos)"
        color = (0, 255, 0) if raw_target_pos is not None else (0, 0, 255)
        
    # ย่อขนาดตัวอักษรลง (0.6) และจัดตำแหน่งให้อยู่ในกรอบ
    cv2.putText(frame_l, f"Status: {status_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # แสดงเปอร์เซ็นต์ Gripper ที่หน้าจอ
    gripper_text = f"Gripper: {current_smoothed_gripper:.0f}%" if raw_target_pos is not None else "Gripper: --"
    cv2.putText(frame_l, gripper_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    if marked_position is not None:
        cv2.putText(frame_l, f"Marked: {marked_position[0]:.0f}, {marked_position[1]:.0f}", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # --- เพิ่ม: วาดไอคอนและข้อความ Guide ด้านล่าง ---
    center_x = int(w / 2)
    y_pos = h - bottom_bar_height + 20
    cv2.putText(frame_l, "Gesture Guide:", (20, y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # วาด Guide สำหรับ Hold (แตะหู)
    hold_guide_x = 250
    if icon_hold is not None:
        frame_l = overlay_transparent(frame_l, icon_hold, hold_guide_x, y_pos - 10)
        cv2.putText(frame_l, "Touch Left Ear = HOLD", (hold_guide_x + 80, y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    else:
        cv2.putText(frame_l, "[ICON] Touch Left Ear = HOLD", (hold_guide_x, y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # วาด Guide สำหรับ Mark (แตะไหล่)
    mark_guide_x = 650
    if icon_mark is not None:
        frame_l = overlay_transparent(frame_l, icon_mark, mark_guide_x, y_pos - 10)
        cv2.putText(frame_l, "Touch Right Shoulder = MARK", (mark_guide_x + 80, y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    else:
        cv2.putText(frame_l, "[ICON] Touch Right Shoulder = MARK", (mark_guide_x, y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


    cv2.imshow("Vision Tracking (Robot Target)", frame_l)

    # ==========================================
    # 7. รับคำสั่งคีย์บอร์ด (เอาไว้เป็นปุ่มสำรอง)
    # ==========================================
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key == ord('h') or key == ord('H'):
        is_holding = not is_holding
        state_text = "ON (หุ่นยนต์หยุดนิ่ง)" if is_holding else "OFF (หุ่นยนต์ขยับตาม)"
        print(f"⏸️ HOLD POSITION: {state_text}")
    elif key == ord('m') or key == ord('M'):
        marked_position = current_smoothed_pos.copy()
        print(f"📌 MARK POSITION: บันทึกพิกัด {marked_position} เรียบร้อยแล้ว")
    elif key == ord('r') or key == ord('R'):
        if marked_position is not None:
            print("🚀 สั่งหุ่นยนต์: RETURN TO MARK!")
            # ตรงนี้สามารถสั่งลัดให้ Raspberry Pi วิ่งกลับไปพิกัด marked_position ได้ทันที
        else:
            print("⚠️ ยังไม่ได้ Mark ตำแหน่ง! กรุณากด M ก่อน")

cam_left.release()
if cam_right is not None:
    cam_right.release()
cv2.destroyAllWindows()