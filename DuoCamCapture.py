import cv2
import os

# 1. ตั้งค่าโฟลเดอร์สำหรับเก็บรูป
output_dir_left = "calibration_data/left"
output_dir_right = "calibration_data/right"

# สร้างโฟลเดอร์ถ้ายังไม่มี
os.makedirs(output_dir_left, exist_ok=True)
os.makedirs(output_dir_right, exist_ok=True)

print("กำลังเชื่อมต่อกล้อง...")
# 2. เชื่อมต่อกล้อง (เลข 0 และ 1 คือ ID ของกล้อง)
# หมายเหตุ: ใน Windows การใส่ cv2.CAP_DSHOW จะช่วยให้เปิดกล้องได้เร็วและไม่ดีเลย์
cam_left = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam_right = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# ตั้งค่าความละเอียดกล้อง (แนะนำ 640x480 หรือ 1280x720)
# ต้องตั้งให้เท่ากันทั้ง 2 กล้อง และตอนใช้งานจริงก็ต้องใช้ความละเอียดนี้
cam_left.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cam_left.isOpened() or not cam_right.isOpened():
    print("❌ ไม่สามารถเปิดกล้องได้ กรุณาตรวจสอบการเชื่อมต่อ USB หรือเปลี่ยนเลข ID กล้อง")
    exit()

print("✅ เชื่อมต่อกล้องสำเร็จ!")
print("========================================")
print("วิธีใช้งาน:")
print("- กดปุ่ม [Spacebar] เพื่อถ่ายรูปคู่ (ซ้าย-ขวา)")
print("- กดปุ่ม [Q] หรือ [ESC] เพื่อออกจากโปรแกรม")
print("========================================")

img_count = 0

while True:
    # 3. ดึงภาพจากกล้องทั้ง 2 ตัว
    ret_l, frame_l = cam_left.read()
    ret_r, frame_r = cam_right.read()

    if not ret_l or not ret_r:
        print("เกิดข้อผิดพลาดในการดึงภาพจากกล้อง")
        break

    # 4. แสดงผลภาพ (รวมเป็นหน้าต่างเดียวเพื่อดูง่าย)
    # นำภาพซ้ายและขวามาต่อกันแนวนอน
    combined_frame = cv2.hconcat([frame_l, frame_r])
    
    # เพิ่มข้อความบอกจำนวนรูปที่ถ่ายไปแล้ว
    cv2.putText(combined_frame, f"Captured: {img_count}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Stereo Camera (Left | Right)", combined_frame)

    # 5. รอรับคำสั่งคีย์บอร์ด
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == 27: # กด 'q' หรือ 'ESC' เพื่อออก
        print("ปิดโปรแกรม...")
        break
    
    elif key == 32: # กด 'Spacebar' เพื่อถ่ายรูป
        # ตั้งชื่อไฟล์ให้ตรงกัน เช่น img_001.png
        filename = f"img_{img_count:03d}.png"
        
        path_l = os.path.join(output_dir_left, filename)
        path_r = os.path.join(output_dir_right, filename)
        
        cv2.imwrite(path_l, frame_l)
        cv2.imwrite(path_r, frame_r)
        
        print(f"📸 ถ่ายรูปคู่ที่ {img_count:03d} สำเร็จ! เซฟลงโฟลเดอร์แล้ว")
        img_count += 1

# 6. คืนทรัพยากร
cam_left.release()
cam_right.release()
cv2.destroyAllWindows()