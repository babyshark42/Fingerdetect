import cv2
import cv2.aruco as aruco
import sys  # 1. เพิ่มบรรทัดนี้

# บังคับให้ Terminal พิมพ์ภาษาไทยได้
sys.stdout.reconfigure(encoding='utf-8')
# 1. กำหนดรูปแบบ Dictionary (ใช้แบบ 4x4 จำนวน 50 ลาย ซึ่งสแกนง่ายและแม่นยำ)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# 2. กำหนดจำนวนช่องตาราง (กว้าง 5 ช่อง, สูง 7 ช่อง)
squares_x = 5
squares_y = 7

# ฟังก์ชันสำหรับสร้างและบันทึกภาพ รองรับ OpenCV ทุกเวอร์ชั่น
def generate_board(filename, square_length, marker_length, img_width, img_height):
    try:
        # สำหรับ OpenCV เวอร์ชั่นเก่า (ต่ำกว่า 4.7 - มักเจอบน Jetson Nano)
        board = aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, aruco_dict)
        img = board.draw((img_width, img_height))
    except AttributeError:
        # สำหรับ OpenCV เวอร์ชั่น 4.7 ขึ้นไป (มักเจอบน PC ที่เพิ่งลง Python ใหม่)
        board = aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)
        img = board.generateImage((img_width, img_height))
        
    cv2.imwrite(filename, img)
    print(f"✅ สร้างไฟล์ {filename} สำเร็จ (ขนาด {img_width}x{img_height} px)")

print("กำลังสร้างไฟล์ ChArUco Board...")

# ==========================================
# 📄 สเปคสำหรับกระดาษ A4
# ==========================================
# ตารางกว้าง 3.5 ซม. (0.035 ม.), Marker กว้าง 2.6 ซม. (0.026 ม.)
# ความละเอียด 2000x2800 พิกเซล (อัตราส่วนพอดีกับ 5x7 ช่อง)
generate_board('charuco_A4_print_me.png', 0.035, 0.026, 2000, 2800)

# ==========================================
# 📄 สเปคสำหรับกระดาษ A3
# ==========================================
# ตารางกว้าง 5.0 ซม. (0.050 ม.), Marker กว้าง 3.8 ซม. (0.038 ม.)
# ความละเอียด 3000x4200 พิกเซล 
# generate_board('charuco_A3_print_me.png', 0.050, 0.038, 3000, 4200)

print("\n⚠️ ข้อควรระวังตอนสั่ง Print:")
print("- ห้ามเลือก 'Fit to Page' หรือ 'Scale to Fit' เด็ดขาด")
print("- ให้ตั้งค่า Scale เป็น '100%' หรือ 'Actual Size' เท่านั้นครับ")