import cv2
import mediapipe as mp
import json
import math
import time
# import rclpy # ปิดไว้ก่อนสำหรับทดสอบ
# from std_msgs.msg import String # ปิดไว้ก่อนสำหรับทดสอบ

# เริ่มต้นใช้งาน MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow('Robot Arm Vision Tracking', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Robot Arm Vision Tracking', 1024, 768)

# --- ตั้งค่า ROS2 Node (ปิดไว้ก่อน) ---
# rclpy.init()
# node = rclpy.create_node('vision_arm_tracker')
# publisher = node.create_publisher(String, '/robot_arm/target_cmd', 10)

# --- ตัวแปรสำหรับจำค่าล่าสุด และทำสมูททิ่ง (Smoothing) ---
prev_x, prev_y, prev_z = 0.0, 0.0, 0.0
prev_gripper = 0
prev_rx, prev_ry, prev_rz = 0.0, 90.0, 0.0 # จำค่าองศาการหมุน

# ค่า Smoothing (0.0 ถึง 1.0) 
# ยิ่งน้อย ยิ่งสมูท/ลดการสั่น แต่จะตามมือช้าลงนิดนึง (Lag)
# ยิ่งมาก ยิ่งตอบสนองไว แต่ถ้ากล้องหลุดจะแกว่งง่าย
alpha_pos = 0.3 
alpha_grip = 0.4
alpha_rot = 0.3 # Smoothing สำหรับองศาการหมุนของข้อมือ

# --- ตั้งค่าระยะแขนสูงสุด (Workspace Limit) ---
# ป้องกันไม่ให้ส่งค่าพิกัดที่ไกลเกินไปจนหุ่นยนต์พยายามยืดตัวจนมอเตอร์พัง (หน่วย mm)
MAX_REACH = 500.0 

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
        
        results = holistic.process(image_rgb)
        image.flags.writeable = True
        
        overlay = image.copy()
        cv2.rectangle(overlay, (5, 5), (750, 230), (0, 0, 0), -1)
        alpha_bg = 0.5
        cv2.addWeighted(overlay, alpha_bg, image, 1 - alpha_bg, 0, image)

        # สถานะการมองเห็น
        arm_visible = False
        hand_visible = False
        robot_enabled = False # สถานะ Safety Switch
        
        target_rx, target_ry = prev_rx, prev_ry # ดึงค่าเก่ามาเตรียมไว้ก่อน

        # --- 0. ประมวลผล Safety Switch (มือซ้าย) ---
        if results.pose_landmarks:
            left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            
            # ถ้ายกมือซ้ายสูงกว่าไหล่ซ้าย (ค่า Y น้อยกว่า) = Enable ให้หุ่นยนต์ขยับตาม
            # เป็น Safety ป้องกันหุ่นขยับเองเวลาเราเผลอเดินผ่านกล้อง
            if left_wrist.y < left_shoulder.y and left_wrist.visibility > 0.5:
                robot_enabled = True

        # --- 1. ประมวลผลส่วน แขน (Pose) ---
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

            # ถ้าระดับความมั่นใจว่าเห็นไหล่และข้อมือ มากกว่า 50%
            if right_wrist.visibility > 0.5 and right_shoulder.visibility > 0.5:
                arm_visible = True
                scale_factor = 1000.0 
                
                # พิกัดดิบที่ได้จากกล้องเฟรมนี้
                raw_x = (right_wrist.x - right_shoulder.x) * scale_factor
                raw_y = (right_shoulder.y - right_wrist.y) * scale_factor
                raw_z = (right_shoulder.z - right_wrist.z) * scale_factor 

                # เข้าสมการ Smoothing (ลดการแกว่ง)
                target_x = (alpha_pos * raw_x) + ((1 - alpha_pos) * prev_x)
                target_y = (alpha_pos * raw_y) + ((1 - alpha_pos) * prev_y)
                target_z = (alpha_pos * raw_z) + ((1 - alpha_pos) * prev_z)

                # --- ตรวจสอบ Workspace Limit (Clamping) ---
                arm_length = math.sqrt(target_x**2 + target_y**2 + target_z**2)
                if arm_length > MAX_REACH:
                    scale = MAX_REACH / arm_length
                    target_x *= scale
                    target_y *= scale
                    target_z *= scale

                # จำค่าไว้ใช้รอบหน้า
                prev_x, prev_y, prev_z = target_x, target_y, target_z

                # วาดเส้นอ้างอิง
                h, w, c = image.shape
                shoulder_px = (int(right_shoulder.x * w), int(right_shoulder.y * h))
                wrist_px = (int(right_wrist.x * w), int(right_wrist.y * h))
                cv2.line(image, shoulder_px, wrist_px, (0, 255, 255), 3)
                cv2.circle(image, shoulder_px, 8, (0, 0, 255), -1)

        # --- 2. ประมวลผลส่วน มือและนิ้ว (Hand) ---
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            right_index_tip = results.right_hand_landmarks.landmark[8]
            right_thumb_tip = results.right_hand_landmarks.landmark[4]
            hand_visible = True

            dx = right_index_tip.x - right_thumb_tip.x
            dy = right_index_tip.y - right_thumb_tip.y
            pinch_dist = math.sqrt(dx**2 + dy**2)
            
            min_pinch = 0.02 
            max_pinch = 0.12 
            pinch_percent = (pinch_dist - min_pinch) / (max_pinch - min_pinch)
            pinch_percent = max(0.0, min(1.0, pinch_percent))
            
            raw_gripper = int(pinch_percent * 90)

            # เข้าสมการ Smoothing กริปเปอร์
            gripper_angle = int((alpha_grip * raw_gripper) + ((1 - alpha_grip) * prev_gripper))
            prev_gripper = gripper_angle

            # --- คำนวณทิศทางการหันข้อมือ (Orientation - Pitch, Yaw) ---
            # ใช้จุดข้อมือ (0) และโคนนิ้วกลาง (9) สร้างเวกเตอร์ทิศทางปลายนิ้ว
            wrist_3d = results.right_hand_landmarks.landmark[0]
            mid_mcp_3d = results.right_hand_landmarks.landmark[9] 
            
            vx = mid_mcp_3d.x - wrist_3d.x
            vy = mid_mcp_3d.y - wrist_3d.y
            vz = mid_mcp_3d.z - wrist_3d.z
            
            # คำนวณมุม Pitch (ก้ม-เงยมือ) และ Yaw (หันซ้าย-ขวา) จากเวกเตอร์ 3D
            raw_pitch = math.degrees(math.atan2(-vy, math.sqrt(vx**2 + vz**2)))
            raw_yaw = math.degrees(math.atan2(vx, vz))
            
            target_rx = (alpha_rot * raw_pitch) + ((1 - alpha_rot) * prev_rx)
            target_ry = (alpha_rot * raw_yaw) + ((1 - alpha_rot) * prev_ry)
            prev_rx, prev_ry = target_rx, target_ry

            # วาดเส้นกริปเปอร์
            h, w, c = image.shape
            thumb_px = (int(right_thumb_tip.x * w), int(right_thumb_tip.y * h))
            index_px = (int(right_index_tip.x * w), int(right_index_tip.y * h))
            cv2.line(image, thumb_px, index_px, (255, 0, 255), 2)
            cv2.circle(image, thumb_px, 6, (255, 0, 255), -1)
            cv2.circle(image, index_px, 6, (255, 0, 255), -1)

        # --- 3. จัดการกรณีที่มองไม่เห็นบางส่วน (Occlusion Handling) ---
        # ถ้ามองไม่เห็น ให้ดึงค่าล่าสุดมาใช้ (Last Known State)
        if not arm_visible:
            target_x, target_y, target_z = prev_x, prev_y, prev_z
            cv2.putText(image, "Arm Occluded: Holding Position", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
        if not hand_visible:
            gripper_angle = prev_gripper
            cv2.putText(image, "Hand Occluded: Holding Gripper", (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # --- 4. แพ็กข้อมูลและแสดงผล ---
        payload = {
            "x": round(target_x, 2),
            "y": round(target_y, 2),
            "z": round(target_z, 2),
            "rx": round(target_rx, 2),
            "ry": round(target_ry, 2),
            "rz": 0.0, # Roll จากกล้องมักจะแกว่งมาก ขอฟิกซ์เป็น 0 ไว้ก่อนเพื่อความเสถียร
            "gripper": int(gripper_angle),
            "enabled": robot_enabled
        }
        json_data = json.dumps(payload)

        # --- 5. ส่งข้อมูล (จำลองแทน ROS2 ไปก่อน) ---
        if robot_enabled:
            # msg = String()
            # msg.data = json_data
            # publisher.publish(msg)
            cv2.putText(image, "STATUS: READY TO SEND", (450, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # หากอยากเห็นค่าไหลรัวๆ ในหน้าจอ Console ให้เอา # บรรทัดล่างนี้ออกครับ
            # print(f"Output -> {json_data}")
        else:
            cv2.putText(image, "STATUS: STANDBY (Raise Left Hand to Enable)", (450, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # ประมวลผลคิวของ ROS2 (ปิดไว้ก่อน)
        # rclpy.spin_once(node, timeout_sec=0)

        cv2.putText(image, "Right Arm Data (Origin = Shoulder):", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"X (Right/Left): {round(target_x, 2)} mm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, f"Y (Up/Down)  : {round(target_y, 2)} mm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, f"Z (Fwd/Back) : {round(target_z, 2)} mm", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(image, f"Gripper Angle: {gripper_angle} deg", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(image, f"JSON: {json_data}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow('Robot Arm Vision Tracking', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# ปิดการทำงาน ROS2 เมื่อจบโปรแกรม (ปิดไว้ก่อน)
# node.destroy_node()
# rclpy.shutdown()