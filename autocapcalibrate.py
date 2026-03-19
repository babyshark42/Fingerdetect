import cv2
import cv2.aruco as aruco
import numpy as np
import os
import sys
import time
import tkinter as tk
from tkinter import ttk
import subprocess

sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# 1. ตั้งค่า ChArUco Board (สเปคของคุณ)
# ==========================================
CHARUCO_COLS    = 5
CHARUCO_ROWS    = 7
SQUARE_LENGTH   = 0.04   # 4 ซม.
MARKER_LENGTH   = 0.03   # 3 ซม.
ARUCO_DICT_ID   = aruco.DICT_4X4_50
TARGET_IMAGES   = 30     # จำนวนรูปที่ต้องการเก็บก่อนเริ่มคำนวณ

# โฟลเดอร์แบ็คอัพรูปภาพ
DIR_L = "auto_calib_data/left"
DIR_R = "auto_calib_data/right"
os.makedirs(DIR_L, exist_ok=True)
os.makedirs(DIR_R, exist_ok=True)

aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT_ID)
try:
    # สำหรับ OpenCV ต่ำกว่า 4.7
    board = aruco.CharucoBoard_create(CHARUCO_COLS, CHARUCO_ROWS, SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
    detectorParams = aruco.DetectorParameters_create()
except AttributeError:
    # สำหรับ OpenCV 4.7 ขึ้นไป
    board = aruco.CharucoBoard((CHARUCO_COLS, CHARUCO_ROWS), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
    detectorParams = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, detectorParams)

# ==========================================
# 2. ฟังก์ชัน GUI เลือกกล้อง
# ==========================================
def get_camera_list():
    cam_names = []
    if os.name == 'nt':
        try:
            cmd = 'powershell -Command "Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -eq \'Image\' -or $_.PNPClass -eq \'Camera\' } | Select-Object -ExpandProperty Caption"'
            result = subprocess.run(cmd, capture_output=True, text=True, creationflags=0x08000000)
            if result.returncode == 0:
                cam_names = [line.strip() for line in result.stdout.split('\n') if line.strip()]
        except: pass
    return [f"{i} - {cam_names[i] if i < len(cam_names) else 'Unknown Camera'}" for i in range(5)]

def show_camera_gui():
    root = tk.Tk()
    root.title("Auto Stereo Calibration")
    root.geometry("450x250")
    
    selected = {"l": 0, "r": 1, "start": False}
    cam_opts = get_camera_list()

    tk.Label(root, text="ตั้งค่ากล้องสำหรับ Auto Calibration", font=('Helvetica', 12, 'bold')).pack(pady=10)
    
    tk.Label(root, text="กล้องซ้าย:").pack()
    l_cb = ttk.Combobox(root, values=cam_opts, state="readonly", width=40)
    l_cb.set(cam_opts[0])
    l_cb.pack()

    tk.Label(root, text="กล้องขวา:").pack()
    r_cb = ttk.Combobox(root, values=cam_opts, state="readonly", width=40)
    r_cb.set(cam_opts[1] if len(cam_opts)>1 else cam_opts[0])
    r_cb.pack()

    def on_start():
        selected["l"] = int(l_cb.get().split(" - ")[0])
        selected["r"] = int(r_cb.get().split(" - ")[0])
        selected["start"] = True
        root.destroy()

    tk.Button(root, text="▶ เปิดกล้อง", command=on_start, bg="green", fg="white").pack(pady=20)
    root.mainloop()
    return selected["l"], selected["r"], selected["start"]

# ==========================================
# 3. ฟังก์ชันดึงจุด ChArUco
# ==========================================
def extract_corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=detectorParams)
    except AttributeError:
        corners, ids, _ = detector.detectMarkers(gray)

    annotated = frame.copy()
    if ids is not None and len(ids) > 0:
        aruco.drawDetectedMarkers(annotated, corners, ids)
        try:
            _, ch_corn, ch_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)
        except AttributeError:
            _, ch_corn, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            
        if ch_corn is not None and len(ch_corn) >= 4:
            aruco.drawDetectedCornersCharuco(annotated, ch_corn, ch_ids)
            return annotated, ch_corn, ch_ids, True
    return annotated, None, None, False

# ==========================================
# 4. เริ่มระบบเปิดกล้อง
# ==========================================
l_id, r_id, is_started = show_camera_gui()
if not is_started: sys.exit()

cam_l = cv2.VideoCapture(l_id, cv2.CAP_DSHOW)
cam_r = cv2.VideoCapture(r_id, cv2.CAP_DSHOW)

for cam in (cam_l, cam_r):
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cam_l.isOpened() or not cam_r.isOpened():
    print("❌ เปิดกล้องไม่สำเร็จ")
    sys.exit()

# ตัวแปรเก็บข้อมูลสำหรับคำนวณ
all_obj_pts, all_img_pts_L, all_img_pts_R = [], [], []
img_size = None
img_count = 0
auto_mode = False
last_cap_time = time.time()
flash_timer = 0

print("="*50)
print("โหมด AUTO CALIBRATION พร้อมใช้งาน!")
print("  - กด [S] เพื่อเปิด/ปิด โหมด Auto-Capture")
print("  - กด [C] เพื่อบังคับเริ่มคำนวณ (กรณีไม่อยากรอครบ 30 รูป)")
print("  - กด [Q] เพื่อออก")
print("="*50)

while True:
    ret_l, frame_l = cam_l.read()
    ret_r, frame_r = cam_r.read()
    if not ret_l or not ret_r: break
    if img_size is None: img_size = frame_l.shape[:2][::-1]

    ann_l, c_cor_l, c_ids_l, ok_l = extract_corners(frame_l)
    ann_r, c_cor_r, c_ids_r, ok_r = extract_corners(frame_r)

    current_time = time.time()
    
    # เงื่อนไข Auto-Capture
    if auto_mode and ok_l and ok_r and (current_time - last_cap_time > 1.0):
        # หาจุดที่ 2 กล้องมองเห็นตรงกัน
        com_ids, idx_l, idx_r = np.intersect1d(c_ids_l, c_ids_r, return_indices=True)
        
        # 🌟 ต้องเห็นร่วมกันอย่างน้อย 12 จุด ถึงจะยอมถ่ายให้ (ป้องกันรูปขยะ)
        if len(com_ids) >= 12:
            try:
                obj_pts = board.chessboardCorners[com_ids]
            except AttributeError:
                obj_pts = board.getChessboardCorners()[com_ids]

            all_obj_pts.append(obj_pts)
            all_img_pts_L.append(c_cor_l[idx_l])
            all_img_pts_R.append(c_cor_r[idx_r])
            
            # Save รูปเป็น Backup (เผื่อนำไปเช็คทีหลัง)
            cv2.imwrite(f"{DIR_L}/img_{img_count:03d}.png", frame_l)
            cv2.imwrite(f"{DIR_R}/img_{img_count:03d}.png", frame_r)

            img_count += 1
            last_cap_time = current_time
            flash_timer = 10 # สั่งกระพริบจอ 10 เฟรม
            print(f"📸 Auto-Captured: รูปที่ {img_count}/{TARGET_IMAGES} (พบจุดร่วม {len(com_ids)} จุด)")

    # วาด UI แถบสถานะ
    display = cv2.hconcat([ann_l, ann_r])
    if flash_timer > 0:
        cv2.rectangle(display, (0,0), (1280, 480), (255,255,255), 15)
        flash_timer -= 1

    hud_color = (0, 255, 0) if auto_mode else (0, 0, 255)
    hud_text = f"AUTO MODE: ON | Captured: {img_count}/{TARGET_IMAGES}" if auto_mode else "AUTO MODE: OFF (Press 'S' to start)"
    cv2.putText(display, hud_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, hud_color, 2)
    
    cv2.imshow("Auto Stereo Calibrator", display)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        sys.exit()
    elif key in (ord('s'), ord('S')):
        auto_mode = not auto_mode
    
    # 🌟 เมื่อเก็บครบ 30 รูป หรือบังคับกด C จะเข้าสู่โหมดคำนวณอัตโนมัติ 🌟
    if img_count >= TARGET_IMAGES or key in (ord('c'), ord('C')):
        if img_count < 10:
            print("⚠️ มีรูปน้อยเกินไป (ควรมีอย่างน้อย 10 รูป) ให้เก็บรูปต่อ...")
            continue
        
        print("\n" + "="*50)
        print(f"กำลังคำนวณสมการ 3 มิติจากรูปทั้งหมด {img_count} รูป...")
        print("กรุณารอสักครู่ (อาจใช้เวลา 10-20 วินาที)...")
        print("="*50)
        
        # ปิดหน้าต่างกล้องไปก่อน
        cv2.destroyAllWindows()
        
        # 1. Calibrate กล้องเดี่ยวเพื่อหา Intrinsic Matrix
        _, m_l, d_l, _, _ = cv2.calibrateCamera(all_obj_pts, all_img_pts_L, img_size, None, None)
        _, m_r, d_r, _, _ = cv2.calibrateCamera(all_obj_pts, all_img_pts_R, img_size, None, None)

        # 2. Stereo Calibrate หาความสัมพันธ์ ซ้าย-ขวา
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        rms, m_l, d_l, m_r, d_r, R, T, E, F = cv2.stereoCalibrate(
            all_obj_pts, all_img_pts_L, all_img_pts_R,
            m_l, d_l, m_r, d_r, img_size, criteria=criteria, flags=flags)

        # 3. Stereo Rectify หาค่า R1, R2, P1, P2 (สำคัญมาก!)
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(m_l, d_l, m_r, d_r, img_size, R, T)

        # 4. บันทึกผลลัพธ์ลงไฟล์ 
        np.savez("stereo_params.npz", 
                 mtx_L=m_l, dist_L=d_l, mtx_R=m_r, dist_R=d_r, 
                 R=R, T=T, R1=R1, R2=R2, P1=P1, P2=P2)

        print("\n🎉 สร้างไฟล์ stereo_params.npz สำเร็จแล้ว!")
        print(f"🎯 ค่าความคลาดเคลื่อน (RMS Error): {rms:.4f} pixels")
        
        if rms < 1.0:
            print("   -> 🌟 ยอดเยี่ยมมาก! ค่า RMS สวยงาม หุ่นยนต์จะแม่นยำมาก")
        elif rms < 2.0:
            print("   -> 👍 พอใช้ได้ ลุยต่อได้เลย")
        else:
            print("   -> ⚠️ ค่าคลาดเคลื่อนสูงไปนิดนึง ถ้าเอาไปใช้แล้ว 3D Skeleton ยังเบี้ยว แนะนำให้รัน Calibration ใหม่อีกรอบครับ")
            
        print("\nนำไฟล์ไปรันกับโปรแกรม main_tracker.py ต่อได้เลยครับ!")
        break

cam_l.release()
cam_r.release()
cv2.destroyAllWindows()