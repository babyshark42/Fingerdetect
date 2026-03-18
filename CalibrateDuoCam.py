import cv2
import cv2.aruco as aruco
import numpy as np
import os
import glob

# ==========================================
# 1. ตั้งค่าพารามิเตอร์ของกระดาน ChArUco (ต้องให้ตรงกับของจริงที่วัดได้!)
# ==========================================
SQUARES_X = 5         # จำนวนช่องตารางแนวนอน
SQUARES_Y = 7         # จำนวนช่องตารางแนวตั้ง
SQUARE_LENGTH = 0.035 # ความยาวช่องตาราง 1 ช่อง (หน่วยเป็น เมตร) -> เปลี่ยนตามที่วัดจริง!
MARKER_LENGTH = 0.026 # ความยาว Marker สีดำด้านใน (หน่วยเป็น เมตร) -> เปลี่ยนตามที่วัดจริง!

# โฟลเดอร์ที่เก็บรูปถ่าย
DIR_LEFT = "calibration_data/left"
DIR_RIGHT = "calibration_data/right"
OUTPUT_FILE = "stereo_params.npz"

# ==========================================
# 2. สร้างออบเจกต์ Board ตามเวอร์ชันของ OpenCV
# ==========================================
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

try:
    # สำหรับ OpenCV ต่ำกว่า 4.7
    board = aruco.CharucoBoard_create(SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
    detectorParams = aruco.DetectorParameters_create()
except AttributeError:
    # สำหรับ OpenCV 4.7 ขึ้นไป
    board = aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
    detectorParams = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, detectorParams)

# ==========================================
# 3. ฟังก์ชันสำหรับหาจุดตัด ChArUco ในภาพ 1 ภาพ
# ==========================================
def extract_charuco_corners(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ตรวจจับ ArUco Markers
    try:
        # OpenCV < 4.7
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=detectorParams)
    except AttributeError:
        # OpenCV >= 4.7
        corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None and len(ids) > 0:
        # หาจุดตัดตาราง (ChArUco Corners) จาก Markers ที่เจอ
        try:
            _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)
        except AttributeError:
            # OpenCV >= 4.7 (อาจเรียกผ่าน cv2.aruco โดยตรง)
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            
        if charuco_corners is not None and len(charuco_corners) >= 4:
            return charuco_corners, charuco_ids, gray.shape[::-1] # return shape เป็น (width, height)
            
    return None, None, None

# ==========================================
# 4. เริ่มกระบวนการจับคู่ภาพซ้าย-ขวา
# ==========================================
left_images = sorted(glob.glob(f"{DIR_LEFT}/*.png"))
right_images = sorted(glob.glob(f"{DIR_RIGHT}/*.png"))

if len(left_images) == 0 or len(left_images) != len(right_images):
    print("❌ จำนวนรูปซ้ายและขวาไม่เท่ากัน หรือไม่พบรูปภาพ กรุณาตรวจสอบโฟลเดอร์")
    exit()

print(f"🔍 พบรูปภาพทั้งหมด {len(left_images)} คู่ กำลังเริ่มค้นหาจุดตัดตาราง...")

all_obj_points = []
all_img_points_L = []
all_img_points_R = []
image_size = None

valid_pairs = 0

for img_L_path, img_R_path in zip(left_images, right_images):
    # หาจุดตัดของภาพซ้ายและขวา
    corners_L, ids_L, size_L = extract_charuco_corners(img_L_path)
    corners_R, ids_R, size_R = extract_charuco_corners(img_R_path)
    
    if image_size is None and size_L is not None:
        image_size = size_L # บันทึกขนาดภาพไว้ใช้ตอน Calibrate

    if corners_L is not None and corners_R is not None:
        # === หาจุดพิกัด "ที่เจอร่วมกัน" ทั้งสองกล้อง ===
        # (เพราะบางรูปกล้องซ้ายอาจจะเห็น 20 จุด แต่กล้องขวาเห็น 15 จุด เราต้องเอาเฉพาะจุดที่เห็นตรงกัน)
        common_ids, idx_L, idx_R = np.intersect1d(ids_L, ids_R, return_indices=True)
        
        if len(common_ids) >= 4: # ต้องเจออย่างน้อย 4 จุดร่วมกันถึงจะคำนวณ 3D ได้
            matched_corners_L = corners_L[idx_L]
            matched_corners_R = corners_R[idx_R]
            
            # หาพิกัด 3D อ้างอิงบนกระดาน (Object Points)
            try:
                obj_pts = board.chessboardCorners[common_ids]
            except AttributeError:
                 # OpenCV >= 4.7
                 obj_pts = board.getChessboardCorners()[common_ids]

            all_obj_points.append(obj_pts)
            all_img_points_L.append(matched_corners_L)
            all_img_points_R.append(matched_corners_R)
            valid_pairs += 1
            print(f"✅ ภาพ {os.path.basename(img_L_path)}: จับคู่จุดตัดได้ {len(common_ids)} จุด")
        else:
            print(f"⚠️ ภาพ {os.path.basename(img_L_path)}: จุดตัดร่วมกันน้อยเกินไป (ข้าม)")
    else:
        print(f"❌ ภาพ {os.path.basename(img_L_path)}: มองไม่เห็นกระดาน (ข้าม)")

print("-" * 50)
print(f"📊 สรุป: นำไปคำนวณได้ {valid_pairs} คู่ จากทั้งหมด {len(left_images)} คู่")

if valid_pairs < 10:
    print("⚠️ คำเตือน: จำนวนคู่ภาพที่ใช้งานได้น้อยกว่า 10 คู่ อาจทำให้การคำนวณพิกัด 3D ไม่แม่นยำ")
    print("แนะนำให้ถ่ายรูปเพิ่มในมุมที่หลากหลายกว่านี้")

# ==========================================
# 5. รันสมการคำนวณ Stereo Calibration
# ==========================================
print("⚙️ กำลังประมวลผลเมทริกซ์ 3 มิติ (อาจใช้เวลาสักครู่)...")

# 5.1 Calibrate แยกทีละกล้องก่อน (เพื่อให้ได้ Intrinsic ที่แม่นยำขึ้น)
ret_L, mtx_L, dist_L, _, _ = cv2.calibrateCamera(all_obj_points, all_img_points_L, image_size, None, None)
ret_R, mtx_R, dist_R, _, _ = cv2.calibrateCamera(all_obj_points, all_img_points_R, image_size, None, None)

# 5.2 Calibrate รวม 2 กล้อง (หาความสัมพันธ์ ซ้าย-ขวา)
flags = cv2.CALIB_FIX_INTRINSIC # ล็อกค่าเลนส์ไว้ เพราะหาแยกมาแล้ว
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

ret_stereo, mtx_L, dist_L, mtx_R, dist_R, R, T, E, F = cv2.stereoCalibrate(
    all_obj_points, all_img_points_L, all_img_points_R,
    mtx_L, dist_L, mtx_R, dist_R, image_size, criteria=criteria, flags=flags)

print(f"🎯 Stereo Calibration Error (RMS): {ret_stereo:.4f} px")
if ret_stereo < 1.0:
    print("   -> ดีมาก! ค่า Error ต่ำกว่า 1 พิกเซล การจับพิกัด 3D จะแม่นยำมาก")
elif ret_stereo < 3.0:
    print("   -> ใช้งานได้ (แต่ถ้าอยากเป๊ะระดับมิลลิเมตร ควรตั้งกล้องให้แน่นและถ่าย Calibration ใหม่ให้แสงสว่างกว่านี้)")
else:
    print("   -> แย่! ค่า Error สูงเกินไป การหันแขนหุ่นยนต์อาจจะเพี้ยนหรือกระตุกได้")

# 5.3 หา Projection Matrices (P1, P2) ซึ่งเป็นหัวใจหลักที่จะเอาไปใช้ในขั้นตอน Triangulation
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    mtx_L, dist_L, mtx_R, dist_R, image_size, R, T)

# ==========================================
# 6. บันทึกผลลัพธ์ทั้งหมด
# ==========================================
np.savez(OUTPUT_FILE, 
         mtx_L=mtx_L, dist_L=dist_L, 
         mtx_R=mtx_R, dist_R=dist_R, 
         R=R, T=T, P1=P1, P2=P2)

print("-" * 50)
print(f"💾 บันทึกไฟล์ตั้งค่ากล้องสำเร็จที่: {OUTPUT_FILE}")
print("พร้อมสำหรับการนำไปรัน AI จับพิกัดแขน 3 มิติ (Triangulation) แล้ว!")