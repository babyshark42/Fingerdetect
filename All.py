import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import time
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess

sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# 1. การตั้งค่าสเปค ChArUco Board และระบบ
# ==========================================
CHARUCO_COLS    = 5
CHARUCO_ROWS    = 7
SQUARE_LENGTH   = 0.04   # 4 ซม.
MARKER_LENGTH   = 0.03   # 3 ซม.
ARUCO_DICT_ID   = aruco.DICT_4X4_50
TARGET_IMAGES   = 30     # รูปสำหรับ Stereo Calib

# โฟลเดอร์เก็บข้อมูลชั่วคราว
os.makedirs("auto_calib_data/left", exist_ok=True)
os.makedirs("auto_calib_data/right", exist_ok=True)

# ตั้งค่า ArUco
aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT_ID)
try:
    board = aruco.CharucoBoard_create(CHARUCO_COLS, CHARUCO_ROWS, SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
    detectorParams = aruco.DetectorParameters_create()
    IS_CV_OLD = True
except AttributeError:
    board = aruco.CharucoBoard((CHARUCO_COLS, CHARUCO_ROWS), SQUARE_LENGTH, MARKER_LENGTH, aruco_dict)
    detectorParams = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, detectorParams)
    IS_CV_OLD = False

# ==========================================
# 2. ฟังก์ชันช่วยเหลือต่างๆ (Helper Functions)
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

def extract_corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if IS_CV_OLD:
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=detectorParams)
    else:
        corners, ids, _ = detector.detectMarkers(gray)

    annotated = frame.copy()
    if ids is not None and len(ids) > 0:
        aruco.drawDetectedMarkers(annotated, corners, ids)
        if IS_CV_OLD:
            _, ch_corn, ch_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)
        else:
            _, ch_corn, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            
        if ch_corn is not None and len(ch_corn) >= 4:
            aruco.drawDetectedCornersCharuco(annotated, ch_corn, ch_ids)
            return annotated, ch_corn, ch_ids, True
    return annotated, None, None, False

def get_marker_center(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if IS_CV_OLD:
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=detectorParams)
    else:
        corners, ids, _ = detector.detectMarkers(gray)
        
    if ids is not None and len(ids) > 0:
        c = corners[0][0]
        center_x = (c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4
        center_y = (c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4
        cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
        aruco.drawDetectedMarkers(frame, corners, ids)
        return frame, np.array([[center_x, center_y]], dtype=np.float32)
    return frame, None

# ==========================================
# 3. โมดูล 1: Auto Stereo Calibration
# ==========================================
def run_stereo_calibration(l_id, r_id):
    cam_l = cv2.VideoCapture(l_id, cv2.CAP_DSHOW)
    cam_r = cv2.VideoCapture(r_id, cv2.CAP_DSHOW)
    for cam in (cam_l, cam_r):
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cam_l.isOpened() or not cam_r.isOpened():
        messagebox.showerror("Error", "เปิดกล้องไม่สำเร็จ กรุณาเช็ค ID กล้อง")
        return

    all_obj_pts, all_img_pts_L, all_img_pts_R = [], [], []
    img_size, img_count, auto_mode, last_cap_time = None, 0, False, time.time()
    flash_timer = 0

    print("\n--- เริ่มโหมด Stereo Calibration ---")
    print("กด 'S' เปิด/ปิด Auto-Capture, 'C' บังคับคำนวณ, 'Q' ออก")

    while True:
        ret_l, frame_l = cam_l.read()
        ret_r, frame_r = cam_r.read()
        if not ret_l or not ret_r: break
        if img_size is None: img_size = frame_l.shape[:2][::-1]

        ann_l, c_cor_l, c_ids_l, ok_l = extract_corners(frame_l)
        ann_r, c_cor_r, c_ids_r, ok_r = extract_corners(frame_r)

        current_time = time.time()
        if auto_mode and ok_l and ok_r and (current_time - last_cap_time > 1.0):
            com_ids, idx_l, idx_r = np.intersect1d(c_ids_l, c_ids_r, return_indices=True)
            if len(com_ids) >= 12:
                if IS_CV_OLD: obj_pts = board.chessboardCorners[com_ids]
                else: obj_pts = board.getChessboardCorners()[com_ids]

                all_obj_pts.append(obj_pts)
                all_img_pts_L.append(c_cor_l[idx_l])
                all_img_pts_R.append(c_cor_r[idx_r])
                img_count += 1
                last_cap_time = current_time
                flash_timer = 10
                print(f"📸 Captured: {img_count}/{TARGET_IMAGES} (จุดร่วม {len(com_ids)})")

        display = cv2.hconcat([ann_l, ann_r])
        if flash_timer > 0:
            cv2.rectangle(display, (0,0), (1280, 480), (255,255,255), 15)
            flash_timer -= 1

        hud_color = (0, 255, 0) if auto_mode else (0, 0, 255)
        hud_text = f"AUTO: ON | Captured: {img_count}/{TARGET_IMAGES}" if auto_mode else "AUTO: OFF (Press 'S')"
        cv2.putText(display, hud_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, hud_color, 2)
        cv2.imshow("1. Auto Stereo Calibrator", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27): break
        elif key in (ord('s'), ord('S')): auto_mode = not auto_mode

        if img_count >= TARGET_IMAGES or key in (ord('c'), ord('C')):
            if img_count < 10:
                print("⚠️ มีรูปน้อยเกินไป (<10)")
                continue
            
            print("\n⏳ กำลังคำนวณสมการ Stereo...")
            cv2.destroyAllWindows()
            _, m_l, d_l, _, _ = cv2.calibrateCamera(all_obj_pts, all_img_pts_L, img_size, None, None)
            _, m_r, d_r, _, _ = cv2.calibrateCamera(all_obj_pts, all_img_pts_R, img_size, None, None)

            flags = cv2.CALIB_FIX_INTRINSIC
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
            rms, m_l, d_l, m_r, d_r, R, T, E, F = cv2.stereoCalibrate(
                all_obj_pts, all_img_pts_L, all_img_pts_R,
                m_l, d_l, m_r, d_r, img_size, criteria=criteria, flags=flags)

            R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(m_l, d_l, m_r, d_r, img_size, R, T)
            np.savez("stereo_params.npz", mtx_L=m_l, dist_L=d_l, mtx_R=m_r, dist_R=d_r, 
                     R=R, T=T, R1=R1, R2=R2, P1=P1, P2=P2)

            messagebox.showinfo("Success", f"สร้างไฟล์ stereo_params.npz สำเร็จ!\nRMS Error: {rms:.4f} pixels")
            break

    cam_l.release(); cam_r.release()
    cv2.destroyAllWindows()

# ==========================================
# 4. โมดูล 2: Floor Calibration
# ==========================================
def run_floor_calibration(l_id):
    if not os.path.exists("stereo_params.npz"):
        messagebox.showerror("Error", "ต้องทำ Stereo Calibration ก่อนครับ!")
        return

    stereo_data = np.load("stereo_params.npz")
    mtx_L, dist_L = stereo_data['mtx_L'], stereo_data['dist_L']

    cam_l = cv2.VideoCapture(l_id, cv2.CAP_DSHOW)
    cam_l.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam_l.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # เพิ่มการตรวจสอบว่าเปิดกล้องสำเร็จหรือไม่
    if not cam_l.isOpened():
        messagebox.showerror("Error", "เปิดกล้องไม่สำเร็จ! กรุณาเช็คการเชื่อมต่อกล้องหรือรีสตาร์ทโปรแกรม")
        return

    print("\n--- เริ่มโหมด Floor Calibration ---")
    print("วาง ChArUco บนพื้น แล้วกด 'C' เพื่อบันทึกพิกัดพื้น")

    while True:
        ret, frame = cam_l.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if IS_CV_OLD: corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=detectorParams)
        else: corners, ids, _ = detector.detectMarkers(gray)

        rvec_floor, tvec_floor = None, None

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(frame, corners, ids)
            if IS_CV_OLD: _, ch_corn, ch_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)
            else: _, ch_corn, ch_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
                
            # แก้ไขจาก 4 เป็น 6
            if ch_corn is not None and len(ch_corn) >= 6:
                aruco.drawDetectedCornersCharuco(frame, ch_corn, ch_ids)
                
                # เพิ่มตัวแปรเริ่มต้นป้องกันโปรแกรมค้างเวลาเห็นมุมไม่ครบ
                ret_pnp = False 
                
                if IS_CV_OLD:
                    ret_pnp, rvec_floor, tvec_floor = aruco.estimatePoseCharucoBoard(ch_corn, ch_ids, board, mtx_L, dist_L, None, None)
                else:
                    objPoints, imgPoints = board.matchImagePoints(ch_corn, ch_ids)
                    # แก้ไขจาก 4 เป็น 6
                    if objPoints is not None and len(objPoints) >= 6:
                        ret_pnp, rvec_floor, tvec_floor = cv2.solvePnP(objPoints, imgPoints, mtx_L, dist_L)

                if ret_pnp:
                    cv2.drawFrameAxes(frame, mtx_L, dist_L, rvec_floor, tvec_floor, 0.15)
                    cv2.putText(frame, "Floor Detected! Press 'C' to Capture", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("2. Floor Calibrator (Left Cam)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27): break
        elif key in (ord('c'), ord('C')) and rvec_floor is not None:
            np.savez("floor_params.npz", rvec=rvec_floor, tvec=tvec_floor)
            messagebox.showinfo("Success", "บันทึกระนาบพื้นสำเร็จ! (floor_params.npz)")
            break

    cam_l.release()
    cv2.destroyAllWindows()

# ==========================================
# 5. โมดูล 3: Real-time 3D Tracking
# ==========================================
def run_3d_tracker(l_id, r_id):
    if not os.path.exists("stereo_params.npz") or not os.path.exists("floor_params.npz"):
        messagebox.showerror("Error", "ต้องทำ Calibration ให้ครบทั้ง Stereo และ Floor ก่อนครับ!")
        return

    stereo = np.load("stereo_params.npz")
    P1 = stereo['mtx_L'] @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = stereo['mtx_R'] @ np.hstack((stereo['R'], stereo['T']))

    floor = np.load("floor_params.npz")
    R_floor, _ = cv2.Rodrigues(floor['rvec'])
    R_floor_inv = np.linalg.inv(R_floor)
    tvec_floor = floor['tvec']

    cam_l = cv2.VideoCapture(l_id, cv2.CAP_DSHOW)
    cam_r = cv2.VideoCapture(r_id, cv2.CAP_DSHOW)

    plt.ion()
    fig = plt.figure("3D Viewer", figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    history_x, history_y, history_z = [], [], []

    print("\n--- เริ่ม 3D Tracking ---")
    print("กด 'Q' ที่หน้าต่างกล้องเพื่อออก")

    while True:
        ret_l, frame_l = cam_l.read()
        ret_r, frame_r = cam_r.read()
        if not ret_l or not ret_r: break

        frame_l, pt_l = get_marker_center(frame_l)
        frame_r, pt_r = get_marker_center(frame_r)

        if pt_l is not None and pt_r is not None:
            pt_4d = cv2.triangulatePoints(P1, P2, pt_l.T, pt_r.T)
            pt_3d_cam = pt_4d[:3] / pt_4d[3]
            pt_world = R_floor_inv @ (pt_3d_cam - tvec_floor)
            X, Y, Z = pt_world.flatten()
            
            history_x.append(X)
            history_y.append(Y)
            history_z.append(Z)
            
            ax.clear()
            ax.set_xlim([-0.5, 0.5])
            ax.set_ylim([-0.5, 0.5])
            ax.set_zlim([0, 1.0])
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
            ax.set_title("Real-time 3D Tracker")
            
            ax.scatter(X, Y, Z, color='red', s=100)
            if len(history_x) > 50:
                history_x.pop(0); history_y.pop(0); history_z.pop(0)
            ax.plot(history_x, history_y, history_z, color='blue', alpha=0.5)
            
            plt.draw()
            plt.pause(0.001)

        display = cv2.hconcat([cv2.resize(frame_l, (400, 300)), cv2.resize(frame_r, (400, 300))])
        cv2.imshow("3. Tracking Cameras", display)

        if cv2.waitKey(1) & 0xFF in (ord('q'), 27): break

    cam_l.release(); cam_r.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close()

# ==========================================
# 6. Main GUI Menu
# ==========================================
class TrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AIO 3D Stereo Tracker")
        self.root.geometry("450x380")
        
        # ค้นหากล้อง
        cam_opts = get_camera_list()
        
        # Header
        tk.Label(root, text="🚀 All-in-One 3D Camera Setup", font=('Helvetica', 14, 'bold')).pack(pady=10)

        # Camera Selectors
        frame_cams = tk.Frame(root)
        frame_cams.pack(pady=5)
        
        tk.Label(frame_cams, text="Left Camera ID:").grid(row=0, column=0, padx=5)
        self.cb_left = ttk.Combobox(frame_cams, values=cam_opts, state="readonly", width=30)
        self.cb_left.set(cam_opts[0])
        self.cb_left.grid(row=0, column=1, pady=2)

        tk.Label(frame_cams, text="Right Camera ID:").grid(row=1, column=0, padx=5)
        self.cb_right = ttk.Combobox(frame_cams, values=cam_opts, state="readonly", width=30)
        self.cb_right.set(cam_opts[1] if len(cam_opts)>1 else cam_opts[0])
        self.cb_right.grid(row=1, column=1, pady=2)

        # Menu Buttons
        tk.Label(root, text="--- ขั้นตอนการทำงาน ---", fg="gray").pack(pady=10)
        
        btn_style = {'width': 35, 'height': 2, 'font': ('Helvetica', 10, 'bold')}
        
        tk.Button(root, text="1. รัน Stereo Calibration (หาค่ากล้อง)", bg="#add8e6", 
                  command=self.btn_stereo, **btn_style).pack(pady=2)
                  
        tk.Button(root, text="2. รัน Floor Calibration (หาพิกัดพื้น)", bg="#90ee90", 
                  command=self.btn_floor, **btn_style).pack(pady=2)
                  
        tk.Button(root, text="3. เริ่ม Real-time 3D Tracker", bg="#ffb6c1", 
                  command=self.btn_track, **btn_style).pack(pady=2)

    def get_ids(self):
        return int(self.cb_left.get().split(" - ")[0]), int(self.cb_right.get().split(" - ")[0])

    def btn_stereo(self):
        l, r = self.get_ids()
        self.root.withdraw() # ซ่อนหน้าต่างหลัก
        run_stereo_calibration(l, r)
        self.root.deiconify() # โชว์หน้าต่างหลักกลับมา

    def btn_floor(self):
        l, _ = self.get_ids()
        self.root.withdraw()
        run_floor_calibration(l)
        self.root.deiconify()

    def btn_track(self):
        l, r = self.get_ids()
        self.root.withdraw()
        run_3d_tracker(l, r)
        self.root.deiconify()

if __name__ == "__main__":
    root = tk.Tk()
    app = TrackerApp(root)
    root.mainloop()