import cv2
import mediapipe as mp
import json
import math
import time

# เริ่มต้นใช้งาน MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic # เพิ่ม Holistic สำหรับจับโครงสร้างมือโดยเฉพาะ

cap = cv2.VideoCapture(0)

# ปรับขนาดภาพที่รับจากกล้องให้ใหญ่ขึ้น (HD 720p)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ตั้งค่าให้หน้าต่าง OpenCV เปิดขึ้นมามีขนาดใหญ่และสามารถใช้เมาส์ลากปรับขนาดได้
cv2.namedWindow('Robot Arm Vision Tracking', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Robot Arm Vision Tracking', 1024, 768)

# เปลี่ยนมาใช้ Holistic แทน Pose
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:

    print("Starting camera... Press 'q' to exit")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("ไม่สามารถรับภาพจากกล้องได้")
            break

        image = cv2.flip(image, 1)
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # ประมวลผลด้วย Holistic
        results = holistic.process(image_rgb)
        image.flags.writeable = True
        
        overlay = image.copy()
        cv2.rectangle(overlay, (5, 5), (750, 230), (0, 0, 0), -1) # ขยายกรอบดำให้ใหญ่ขึ้น
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        # วาดเส้นโครงร่างกาย (Pose)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # วาดเส้นมือขวา (Hand) เพื่อให้เห็นข้อนิ้วชัดๆ
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS)

        # ตรวจสอบว่ากล้องมองเห็นทั้ง ไหล่ (จาก Pose) และ ปลายนิ้ว (จาก Hand)
        if results.pose_landmarks and results.right_hand_landmarks:
            
            # ดึงข้อมูล ข้อมือขวา (16) และ ไหล่ขวา (12)
            right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # ดึงข้อมูล ปลายนิ้วชี้ (8) และ ปลายนิ้วโป้ง (4) จาก Hand Landmarks โดยตรง
            right_index_tip = results.right_hand_landmarks.landmark[8]
            right_thumb_tip = results.right_hand_landmarks.landmark[4]

            if right_wrist.visibility > 0.5 and right_shoulder.visibility > 0.5:
                
                scale_factor = 1000.0 # จำลองความยาวแขนให้หน่วยเป็นมิลลิเมตร
                
                # --- คำนวณพิกัดสัมพัทธ์ (Relative to Shoulder) ---
                # ให้ ไหล่ขวา เป็นจุด (0, 0, 0) หรือเปรียบเสมือนฐานหุ่นยนต์ (Robot Base)
                
                # แกน X: ซ้าย-ขวา (กางแขนออกขวาเป็นบวก)
                rel_x = (right_wrist.x - right_shoulder.x) * scale_factor
                
                # แกน Y: บน-ล่าง (ปกติภาพ Y ลงล่างคือบวก เรากลับสมการให้ "ยกแขนขึ้นบนเป็นบวก")
                rel_y = (right_shoulder.y - right_wrist.y) * scale_factor
                
                # แกน Z: ใกล้-ไกล (ยื่นมือเข้าหากล้องเป็นบวก)
                rel_z = (right_shoulder.z - right_wrist.z) * scale_factor 

                target_x = round(rel_x, 2)
                target_y = round(rel_y, 2)
                target_z = round(rel_z, 2)

                # --- คำนวณองศาการหนีบ (Gripper Pinch Angle) ---
                # หาความห่างระหว่าง ปลายนิ้วชี้ และ ปลายนิ้วโป้ง
                dx = right_index_tip.x - right_thumb_tip.x
                dy = right_index_tip.y - right_thumb_tip.y
                pinch_dist = math.sqrt(dx**2 + dy**2)
                
                # กำหนดค่า Min/Max ของระยะหนีบ (สเกลของ Hand จะแคบกว่า Pose เล็กน้อย)
                min_pinch = 0.02 # นิ้วชนกันสนิท
                max_pinch = 0.12 # กางนิ้วชี้กับโป้ง
                
                # แปลงระยะให้เป็นสัดส่วน (0.0 ถึง 1.0)
                pinch_percent = (pinch_dist - min_pinch) / (max_pinch - min_pinch)
                pinch_percent = max(0.0, min(1.0, pinch_percent)) # บังคับค่าให้อยู่ในช่วง 0-1 เสมอ
                
                # แปลงสัดส่วนเป็นองศา (เช่น 0 = หนีบปิดสนิท, 90 = เปิดอ้าสุด)
                gripper_angle = int(pinch_percent * 90)

                # วาดเส้นเชื่อมระหว่างไหล่กับข้อมือเพื่อให้เห็นภาพจุดอ้างอิง
                h, w, c = image.shape
                shoulder_px = (int(right_shoulder.x * w), int(right_shoulder.y * h))
                wrist_px = (int(right_wrist.x * w), int(right_wrist.y * h))
                cv2.line(image, shoulder_px, wrist_px, (0, 255, 255), 3)
                cv2.circle(image, shoulder_px, 8, (0, 0, 255), -1) # จุดแดงคือจุด Origin (0,0)

                # วาดเส้นสีม่วงเชื่อมระหว่างปลายนิ้วชี้กับปลายนิ้วโป้ง
                thumb_px = (int(right_thumb_tip.x * w), int(right_thumb_tip.y * h))
                index_px = (int(right_index_tip.x * w), int(right_index_tip.y * h))
                cv2.line(image, thumb_px, index_px, (255, 0, 255), 2)
                cv2.circle(image, thumb_px, 6, (255, 0, 255), -1)
                cv2.circle(image, index_px, 6, (255, 0, 255), -1)

                payload = {
                    "x": target_x,
                    "y": target_y,
                    "z": target_z,
                    "rx": 0.0,
                    "ry": 90.0,
                    "rz": 0.0,
                    "gripper": gripper_angle
                }
                json_data = json.dumps(payload)

                cv2.putText(image, "Right Arm Data (Origin = Shoulder):", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, f"X (Right/Left): {target_x} mm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(image, f"Y (Up/Down)  : {target_y} mm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(image, f"Z (Fwd/Back) : {target_z} mm", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(image, f"Gripper Angle: {gripper_angle} deg", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2) # เพิ่มโชว์องศาหนีบ
                cv2.putText(image, f"JSON: {json_data}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        else:
            cv2.putText(image, "Right Arm/Hand NOT Visible!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(image, "Please show both shoulder and hand clearly.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Robot Arm Vision Tracking', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()