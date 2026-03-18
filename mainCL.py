import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# ฟังก์ชันพิเศษ: สำหรับวางรูปภาพ PNG (มีพื้นหลังใส) ทับบนวิดีโอ
# ==========================================
def overlay_transparent(background, overlay, x, y):
    bg_h, bg_w, _ = background.shape
    ov_h, ov_w, ov_c = overlay.shape

    if x >= bg_w or y >= bg_h or x + ov_w <= 0 or y + ov_h <= 0:
        return background
    if x < 0 or y < 0 or x + ov_w > bg_w or y + ov_h > bg_h:
        return background

    if ov_c == 4:
        alpha = overlay[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha
        for c in range(3):
            background[y:y+ov_h, x:x+ov_w, c] = (alpha * overlay[:, :, c] + alpha_inv * background[y:y+ov_h, x:x+ov_w, c])
    else:
        background[y:y+ov_h, x:x+ov_w] = overlay

    return background

# ==========================================
# 1. ตั้งค่า MediaPipe (AI จับท่าทาง)
# ==========================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6, model_complexity=1)

# ==========================================
# 2. ตั้งค่าระบบควบคุม (Control Variables)
# ==========================================
# 2.1 ตัวแปรสำหรับระบบกันกระชาก (EMA Filter)
EMA_ALPHA = 0.15
current_smoothed_pos = np.array([0.0, 0.0, 0.0])
current_smoothed_gripper = 0.0
is_first_frame = True

# 2.2 ตัวแปรสำหรับ Mark ตำแหน่ง
marked_position = None
is_robot_moving_to_mark = False
is_holding = False

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

# โหลดไฟล์รูปภาพไอคอน (ถ้ามี)
icon_hold = cv2.imread("icon_hold.png", cv2.IMREAD_UNCHANGED)
icon_mark = cv2.imread("icon_mark.png", cv2.IMREAD_UNCHANGED)

if icon_hold is None or icon_mark is None:
    print("⚠️ ไม่พบไฟล์รูปภาพ 'icon_hold.png' หรือ 'icon_mark.png' ระบบจะแสดงผลเป็นข้อความแทน")

# ==========================================
# 4. ฟังก์ชันสำหรับหาจุด 3D (Triangulation)
# ==========================================
def get_3d_point(p_left, p_right, proj1, proj2):
    pt_l = np.array([[p_left[0]], [p_left[1]]], dtype=np.float32)
    pt_r = np.array([[p_right[0]], [p_right[1]]], dtype=np.float32)
    pt_4d = cv2.triangulatePoints(proj1, proj2, pt_l, pt_r)
    pt_3d = pt_4d[:3] / pt_4d[3]
    return pt_3d.flatten()

# ==========================================
# 5. เปิดกล้องและรันลูปหลัก
# ==========================================
cam_left = cv2.VideoCapture(0)
cam_right = cv2.VideoCapture(1) if STEREO_MODE else None

cv2.namedWindow("Vision Tracking (Robot Target)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Vision Tracking (Robot Target)", 1024, 768)

print("==========================================")
print("คีย์ลัดสำหรับควบคุม:")
print(" - กด [H] : เปิด/ปิด โหมด Hold (หยุดหุ่นชั่วคราว)")
print(" - กด [M] : เพื่อ Mark ตำแหน่ง (จุดปลอดภัย)")
print(" - กด [R] : สั่งหุ่นยนต์กลับไปจุด Mark (Return)")
print(" - กด [Q] : ออกจากโปรแกรม")
print("==========================================")
print("ท่าทางมือซ้าย (ปุ่มล่องหน):")
print(" - ยกมือซ้ายเหนือไหล่ซ้าย  → เปิด/ปิด HOLD (Cooldown 1 วิ)")
print(" - กำหมัดมือซ้าย            → MARK ตำแหน่ง (Cooldown 2 วิ)")
print("==========================================")

while True:
    ret_l, frame_l = cam_left.read()
    if not ret_l:
        break

    frame_l = cv2.flip(frame_l, 1)

    frame_r = None
    if STEREO_MODE and cam_right is not None:
        ret_r, frame_r = cam_right.read()
        if ret_r:
            frame_r = cv2.flip(frame_r, 1)

    # 5.1 นำภาพให้ AI ประมวลผล
    rgb_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB)
    results_l = pose.process(rgb_l)

    if results_l.pose_landmarks:
        mp_drawing.draw_landmarks(frame_l, results_l.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 5.2 หาตำแหน่ง TCP และควบคุม Gripper
    raw_target_pos = None
    raw_gripper_percent = None

    if results_l.pose_landmarks:
        # ==========================================
        # ส่วนที่ 1: ตรวจจับท่าทางมือซ้าย (ปุ่มล่องหน)
        # ==========================================
        left_wrist    = results_l.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        left_shoulder = results_l.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_index    = results_l.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX]
        left_thumb    = results_l.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB]

        if left_wrist.visibility > 0.6:
            h, w, _ = frame_l.shape
            current_time = time.time()

            # --- ท่าที่ 1: ยกมือซ้ายเหนือไหล่ซ้าย = เปิด/ปิด HOLD ---
            # (y น้อยกว่า = อยู่สูงกว่าในภาพ)
            is_hand_raised = left_wrist.y < left_shoulder.y - 0.05

            if is_hand_raised and (current_time - last_hold_toggle_time > 1.0):
                is_holding = not is_holding
                last_hold_toggle_time = current_time
                state_text = "ON" if is_holding else "OFF"
                print(f"✋ GESTURE: Raise Left Hand → Hold {state_text}")
                cv2.circle(frame_l,
                           (int(left_wrist.x * w), int(left_wrist.y * h)),
                           40, (0, 165, 255), -1)

            # --- ท่าที่ 2: กำหมัดมือซ้าย = MARK ตำแหน่ง ---
            # (นิ้วชี้ซ้ายและนิ้วโป้งซ้ายชิดกัน)
            dist_left_fingers = np.sqrt(
                (left_index.x - left_thumb.x)**2 +
                (left_index.y - left_thumb.y)**2 +
                (left_index.z - left_thumb.z)**2
            )
            is_fist = dist_left_fingers < 0.04  # ปรับค่านี้ถ้า trigger ไม่ติด

            if is_fist and (current_time - last_mark_time > 2.0):
                marked_position = current_smoothed_pos.copy() if not is_first_frame else np.array([0, 0, 0])
                last_mark_time = current_time
                print(f"📌 GESTURE: Left Fist → Mark Position {marked_position}")
                cv2.circle(frame_l,
                           (int(left_wrist.x * w), int(left_wrist.y * h)),
                           40, (255, 255, 0), -1)

        # ==========================================
        # ส่วนที่ 2: ดึงข้อมูลมือขวา (ควบคุมเป้าหมายหุ่นยนต์)
        # ==========================================
        index_finger = results_l.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
        thumb_finger  = results_l.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB]

        if (index_finger.visibility + thumb_finger.visibility) / 2 > 0.6:
            h, w, _ = frame_l.shape

            tcp_x = int(((index_finger.x + thumb_finger.x) / 2) * w)
            tcp_y = int(((index_finger.y + thumb_finger.y) / 2) * h)
            px_l = [tcp_x, tcp_y]

            if STEREO_MODE and frame_r is not None:
                pass  # โหมด 3D: รัน AI ภาพขวาแล้วทำ Triangulate
            else:
                raw_target_pos = np.array([px_l[0], px_l[1], 0.0])

                cv2.circle(frame_l, (px_l[0], px_l[1]), 10, (0, 0, 255), -1)

                idx_x, idx_y = int(index_finger.x * w), int(index_finger.y * h)
                thb_x, thb_y = int(thumb_finger.x * w), int(thumb_finger.y * h)
                cv2.line(frame_l, (idx_x, idx_y), (thb_x, thb_y), (0, 255, 255), 2)

                # คำนวณระยะกริปเปอร์ (ใช้ระยะ 3D เพื่อความแม่นยำ)
                dist_3d = np.sqrt(
                    (index_finger.x - thumb_finger.x)**2 +
                    (index_finger.y - thumb_finger.y)**2 +
                    (index_finger.z - thumb_finger.z)**2
                )
                MIN_DIST = 0.02  # ระยะตอนปลายนิ้วชนกัน
                MAX_DIST = 0.12  # ระยะตอนกางนิ้วสุด
                raw_gripper_percent = np.clip(
                    (dist_3d - MIN_DIST) / (MAX_DIST - MIN_DIST) * 100, 0, 100
                )

    # 5.3 ระบบกันกระชาก (EMA Filter) & อัปเดตพิกัดส่งให้หุ่น
    if raw_target_pos is not None and raw_gripper_percent is not None:
        if is_first_frame:
            current_smoothed_pos = raw_target_pos
            current_smoothed_gripper = raw_gripper_percent
            is_first_frame = False
        elif not is_holding:
            current_smoothed_pos     = (EMA_ALPHA * raw_target_pos)      + ((1.0 - EMA_ALPHA) * current_smoothed_pos)
            current_smoothed_gripper = (EMA_ALPHA * raw_gripper_percent) + ((1.0 - EMA_ALPHA) * current_smoothed_gripper)

        cv2.circle(frame_l,
                   (int(current_smoothed_pos[0]), int(current_smoothed_pos[1])),
                   8, (0, 255, 0), 2)

        # !! ส่งข้อมูลผ่าน MQTT/Serial ไปให้ Raspberry Pi ที่นี่ !!
        # print(f"ส่งข้อมูล → X:{current_smoothed_pos[0]:.1f}, Y:{current_smoothed_pos[1]:.1f}, Gripper:{current_smoothed_gripper:.1f}%")

    # ==========================================
    # 6. วาด UI (User Interface) บนภาพ
    # ==========================================
    h, w, _ = frame_l.shape

    overlay = frame_l.copy()
    cv2.rectangle(overlay, (10, 10), (320, 120), (0, 0, 0), -1)

    bottom_bar_height = 100
    cv2.rectangle(overlay, (0, h - bottom_bar_height), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame_l, 0.4, 0, frame_l)

    if is_holding:
        status_text = "HOLDING (Manual)"
        color = (0, 165, 255)
    else:
        status_text = "Tracking TCP" if raw_target_pos is not None else "Vision Lost! (Holding Pos)"
        color = (0, 255, 0) if raw_target_pos is not None else (0, 0, 255)

    cv2.putText(frame_l, f"Status: {status_text}", (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    gripper_text = f"Gripper: {current_smoothed_gripper:.0f}%" if raw_target_pos is not None else "Gripper: --"
    cv2.putText(frame_l, gripper_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if marked_position is not None:
        cv2.putText(frame_l, f"Marked: {marked_position[0]:.0f}, {marked_position[1]:.0f}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # --- Guide ท่าทางด้านล่าง ---
    y_pos = h - bottom_bar_height + 20
    cv2.putText(frame_l, "Gesture Guide:", (20, y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    hold_guide_x = 250
    if icon_hold is not None:
        frame_l = overlay_transparent(frame_l, icon_hold, hold_guide_x, y_pos - 10)
        cv2.putText(frame_l, "Raise Left Hand = HOLD",
                    (hold_guide_x + 80, y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    else:
        cv2.putText(frame_l, "[ICON] Raise Left Hand = HOLD",
                    (hold_guide_x, y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    mark_guide_x = 650
    if icon_mark is not None:
        frame_l = overlay_transparent(frame_l, icon_mark, mark_guide_x, y_pos - 10)
        cv2.putText(frame_l, "Left Fist = MARK",
                    (mark_guide_x + 80, y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    else:
        cv2.putText(frame_l, "[ICON] Left Fist = MARK",
                    (mark_guide_x, y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Vision Tracking (Robot Target)", frame_l)

    # ==========================================
    # 7. รับคำสั่งคีย์บอร์ด (ปุ่มสำรอง)
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
            # สั่งให้ Raspberry Pi วิ่งกลับไปพิกัด marked_position ได้ทันที
        else:
            print("⚠️ ยังไม่ได้ Mark ตำแหน่ง! กรุณากด M ก่อน")

# ==========================================
# 8. ปิดกล้องและทำความสะอาด
# ==========================================
cam_left.release()
if cam_right is not None:
    cam_right.release()
cv2.destroyAllWindows()