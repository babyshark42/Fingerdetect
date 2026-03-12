import cv2
import mediapipe as mp
import json
import math
import time

# เริ่มต้นใช้งาน MediaPipe Pose สำหรับตรวจจับโครงสร้างร่างกาย
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# เปิดกล้องเว็บแคม (เลข 0 คือกล้องตัวแรกของเครื่อง)
cap = cv2.VideoCapture(0)

# กำหนดค่าเริ่มต้นสำหรับ MediaPipe Pose
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    print("Starting camera... Press 'q' to exit")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("ไม่สามารถรับภาพจากกล้องได้")
            break

        # พลิกภาพ (Mirror) เพื่อให้ซ้าย-ขวาตรงกับความเป็นจริงตอนมองจอ
        image = cv2.flip(image, 1)
        
        # แปลงสีภาพจาก BGR (ที่ OpenCV ใช้) เป็น RGB (ที่ MediaPipe ต้องการ)
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # ประมวลผลหา Pose (จุดข้อต่อต่างๆ)
        results = pose.process(image_rgb)

        # วาดผลลัพธ์ลงบนภาพเดิม
        image.flags.writeable = True
        
        if results.pose_landmarks:
            # วาดเส้นโครงร่าง (Skeleton) บนตัวคน
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # ดึงข้อมูลจุดที่ 16 (ข้อมือขวา - RIGHT_WRIST)
            # อ้างอิงจุด: 11=ไหล่ขวา, 13=ศอกขวา, 15=ข้อมือขวา (MediaPipe นับสลับซ้ายขวาตอนไม่ flip)
            # เนื่องจากเรา flip ภาพแล้ว ข้อมือขวาของเราบนจอคือจุดที่ 16
            right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            
            # วาดพื้นหลังสีดำโปร่งแสงเพื่อให้ตัวหนังสืออ่านง่ายขึ้น
            overlay = image.copy()
            cv2.rectangle(overlay, (5, 5), (650, 180), (0, 0, 0), -1)
            alpha = 0.5 # ความโปร่งแสง
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

            # ตรวจสอบว่ากล้องมองเห็นข้อมือชัดเจนหรือไม่ (Visibility > 50%)
            if right_wrist.visibility > 0.5:
                # ค่าที่ได้จะเป็นค่า Normalized (0.0 - 1.0) เราต้องแปลงให้เป็นสเกลจำลอง (เช่น มิลลิเมตร)
                # สมมติให้กรอบภาพกว้าง 1000mm สูง 1000mm และลึก 1000mm
                scale_factor = 1000.0
                
                # คำนวณพิกัด X, Y, Z (จำลองสเกล)
                target_x = round(right_wrist.x * scale_factor, 2)
                target_y = round(right_wrist.y * scale_factor, 2)
                target_z = round(right_wrist.z * scale_factor, 2) # Z คือความลึก (ใกล้-ไกลจากกล้อง)

                # จำลองค่า Roll, Pitch, Yaw (ในระบบจริงต้องคำนวณจากเวกเตอร์ของข้อมือและฝ่ามือ)
                # ตรงนี้ใส่ค่าสมมติไปก่อนเพื่อให้เห็นโครงสร้างข้อมูล
                target_rx = 0.0
                target_ry = 90.0
                target_rz = 0.0

                # สร้างชุดข้อมูล JSON จำลองเพื่อเตรียมส่งให้ Raspberry Pi
                payload = {
                    "x": target_x,
                    "y": target_y,
                    "z": target_z,
                    "rx": target_rx,
                    "ry": target_ry,
                    "rz": target_rz,
                    "gripper": "open" # จำลองสถานะกริปเปอร์
                }
                
                # แปลง Dict เป็น JSON String
                json_data = json.dumps(payload)

                # แสดงผลข้อความบนหน้าจอ OpenCV
                cv2.putText(image, "Right Wrist Data:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, f"X: {target_x} mm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(image, f"Y: {target_y} mm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(image, f"Z: {target_z} mm", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(image, f"JSON: {json_data}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # พิมพ์ลง Console ด้วย (เพื่อให้เห็นชัดๆ ว่าจะส่งอะไรผ่าน Network)
                # print(f"Sending -> {json_data}")
            else:
                # ถ้ามองไม่เห็นข้อมือ ให้ขึ้นข้อความเตือนสีแดง
                cv2.putText(image, "Right Wrist NOT Visible!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(image, "Please raise your right arm into the camera view.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # แสดงภาพผลลัพธ์
        cv2.imshow('Robot Arm Vision Tracking', image)

        # กด 'q' เพื่อออกจากโปรแกรม
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# ปิดการเชื่อมต่อกล้องและหน้าต่างทั้งหมด
cap.release()
cv2.destroyAllWindows()