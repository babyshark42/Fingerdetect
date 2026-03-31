import cv2
import cv2.aruco as aruco
import numpy as np
import sys
import os
import time
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import mediapipe as mp
import asyncio
import websockets
import json
import threading
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# 🌟 CLASS ใหม่: ดึงภาพจากกล้องแบบแยก Thread (แก้ปัญหาคอขวด USB)
# ==========================================
class CameraStream:
    def __init__(self, src=0):
        # ตัด CAP_DSHOW ออก เพื่อให้รันบน Linux (Jetson) ได้
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        
    def start(self):
        # เริ่ม Thread แยกสำหรับการดึงภาพโดยเฉพาะ
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.grabbed, self.frame.copy() if self.grabbed else None

    def stop(self):
        self.stopped = True
        self.stream.release()

# ==========================================
# 1. การตั้งค่าสเปค ChArUco Board และระบบ
# ==========================================
CHARUCO_COLS    = 5
CHARUCO_ROWS    = 7
SQUARE_LENGTH   = 0.04   # 4 ซม.
MARKER_LENGTH   = 0.03   # 3 ซม.
ARUCO_DICT_ID   = aruco.DICT_4X4_50
TARGET_IMAGES   = 30     # รูปสำหรับ Stereo Calib

os.makedirs("auto_calib_data/left", exist_ok=True)
os.makedirs("auto_calib_data/right", exist_ok=True)

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
    # 🌟 แก้ให้รองรับทั้ง Windows และ Linux (Jetson Nano)
    cam_names = []
    if os.name == 'nt':
        try:
            cmd = 'powershell -Command "Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -eq \'Image\' -or $_.PNPClass -eq \'Camera\' } | Select-Object -ExpandProperty Caption"'
            result = subprocess.run(cmd, capture_output=True, text=True, creationflags=0x08000000)
            if result.returncode == 0:
                cam_names = [line.strip() for line in result.stdout.split('\n') if line.strip()]
        except: pass
        return [f"{i} - {cam_names[i] if i < len(cam_names) else 'Unknown Camera'}" for i in range(5)]
    else:
        # สำหรับ Linux หาไฟล์ใน /dev/video*
        import glob
        cams = glob.glob('/dev/video*')
        cams.sort()
        return [f"{c.replace('/dev/video','')}" for c in cams][:5] if cams else ["0", "1"]

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
            
        if ch_corn is not None and len(ch_corn) >= 6:
            aruco.drawDetectedCornersCharuco(annotated, ch_corn, ch_ids)
            return annotated, ch_corn, ch_ids, True
    return annotated, None, None, False

# ==========================================
# 3. โมดูล 1: Auto Stereo Calibration
# ==========================================
def run_stereo_calibration(l_id, r_id):
    cam_l = CameraStream(l_id).start()
    cam_r = CameraStream(r_id).start()
    time.sleep(1.0) # รอให้กล้องวอร์มอัพ

    if not cam_l.grabbed or not cam_r.grabbed:
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

    cam_l.stop(); cam_r.stop()
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

    cam_l = CameraStream(l_id).start()
    time.sleep(1.0)
    
    if not cam_l.grabbed:
        messagebox.showerror("Error", "เปิดกล้องไม่สำเร็จ! กรุณาเช็คการเชื่อมต่อกล้อง")
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
                
            if ch_corn is not None and len(ch_corn) >= 6:
                aruco.drawDetectedCornersCharuco(frame, ch_corn, ch_ids)
                ret_pnp = False 
                
                if IS_CV_OLD:
                    ret_pnp, rvec_floor, tvec_floor = aruco.estimatePoseCharucoBoard(ch_corn, ch_ids, board, mtx_L, dist_L, None, None)
                else:
                    objPoints, imgPoints = board.matchImagePoints(ch_corn, ch_ids)
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

    cam_l.stop()
    cv2.destroyAllWindows()

# ==========================================
# 4.5 ระบบ WebSocket Server
# ==========================================
latest_skeleton_data = {"persons": [], "timestamp": 0}

async def skeleton_ws_handler(websocket):
    print("🌐 Web Client Connected!")
    try:
        while True:
            await websocket.send(json.dumps(latest_skeleton_data))
            await asyncio.sleep(1/30) # อัปเดตที่ 30Hz
    except websockets.exceptions.ConnectionClosed:
        print("🌐 Web Client Disconnected!")

async def main_ws():
    print("🚀 กำลังเปิด WebSocket Server ที่พอร์ต 8767...")
    try:
        async with websockets.serve(skeleton_ws_handler, "0.0.0.0", 8767):
            print("✅ WebSocket Server รันสำเร็จแล้ว!")
            await asyncio.Future()
    except OSError as e:
        print(f"❌ [ERROR] WebSocket เกิดข้อผิดพลาด/พอร์ตชนกัน: {e}")

def start_ws_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main_ws())

# ==========================================
# 5. โมดูล 3: Real-time 3D Skeleton Tracking
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

    TARGET_LANDMARKS = [0, 27, 28, 11, 12, 23, 24, 13, 14, 15, 16]
    SKELETON_CONNECTIONS = [
        (11, 12), (12, 24), (24, 23), (23, 11),
        (11, 13), (13, 15), (12, 14), (14, 16),
        (0, 11), (0, 12), (23, 27), (24, 28)
    ]

    mp_pose = mp.solutions.pose
    pose_l = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose_r = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # 🌟 ใช้ Threading ดึงภาพกล้อง เพื่อไม่ให้เป็นคอขวด 🌟
    cam_l = CameraStream(l_id).start()
    cam_r = CameraStream(r_id).start()
    time.sleep(1.0)

    if not cam_l.grabbed or not cam_r.grabbed:
        messagebox.showerror("Error", "เปิดกล้องไม่สำเร็จ!")
        cam_l.stop(); cam_r.stop()
        return

    ws_thread = threading.Thread(target=start_ws_server, daemon=True)
    ws_thread.start()

    def get_target_keypoints(frame, pose):
        if frame is None: return None, {}
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        keypoints = {}
        h, w, _ = frame.shape
        
        if results.pose_landmarks:
            for idx in TARGET_LANDMARKS:
                lm = results.pose_landmarks.landmark[idx]
                if lm.visibility > 0.5:
                    px, py = int(lm.x * w), int(lm.y * h)
                    keypoints[idx] = (px, py)
                    cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
                    cv2.putText(frame, str(idx), (px+5, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
        return frame, keypoints

    cv_window_name = "3. Tracking Cameras (L | R)"
    cv_3d_name = "3D Skeleton Views"
    cv2.namedWindow(cv_window_name)
    cv2.namedWindow(cv_3d_name)

    global latest_skeleton_data
    
    # 🌟 ใช้ ThreadPoolExecutor ให้ MediaPipe รันขนานกัน 2 กล้อง 🌟
    executor = ThreadPoolExecutor(max_workers=2)

    def map_range(val, in_min, in_max, out_min, out_max):
        return int((val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

    prev_time = time.time()

    while True:
        ret_l, frame_l = cam_l.read()
        ret_r, frame_r = cam_r.read()
        if not ret_l or not ret_r: continue

        # 🌟 รัน MediaPipe ซ้าย-ขวา พร้อมกัน (Parallel) 🌟
        future_l = executor.submit(get_target_keypoints, frame_l, pose_l)
        future_r = executor.submit(get_target_keypoints, frame_r, pose_r)
        
        frame_l, kps_l = future_l.result()
        frame_r, kps_r = future_r.result()

        pts_3d_world = {}

        for idx in TARGET_LANDMARKS:
            if idx in kps_l and idx in kps_r:
                pt_l = np.array([[kps_l[idx][0], kps_l[idx][1]]], dtype=np.float32)
                pt_r = np.array([[kps_r[idx][0], kps_r[idx][1]]], dtype=np.float32)
                
                pt_4d = cv2.triangulatePoints(P1, P2, pt_l.T, pt_r.T)
                pt_3d_cam = pt_4d[:3] / pt_4d[3] 
                
                pt_world = R_floor_inv @ (pt_3d_cam - tvec_floor)
                xw, yw, zw = pt_world.flatten()
                pts_3d_world[idx] = [xw, yw, -zw]

        if pts_3d_world:
            keypoints_data = {}
            for idx, pt in pts_3d_world.items():
                keypoints_data[str(idx)] = {"x": float(pt[0]), "y": float(pt[2]), "z": float(pt[1]), "visibility": 1.0}
            
            latest_skeleton_data = {
                "persons": [{"id": 0, "keypoints": keypoints_data}],
                "timestamp": int(time.time() * 1000)
            }
        else:
            latest_skeleton_data = {"persons": [], "timestamp": int(time.time() * 1000)}

        # คำนวณ FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # วาด 3D View
        canvas_3d = np.zeros((400, 1200, 3), dtype=np.uint8)
        
        def draw_view(pts, view_type, offset_x):
            cv2.rectangle(canvas_3d, (offset_x, 0), (offset_x + 400, 400), (50, 50, 50), 1)
            def get_pt(pt_3d):
                x, y, z = pt_3d
                if view_type == 'front': 
                    px, py = map_range(x, -1.0, 1.0, 50, 350), map_range(z, 2.0, -0.5, 50, 350)
                elif view_type == 'top':   
                    px, py = map_range(x, -1.0, 1.0, 50, 350), map_range(y, 1.0, -1.0, 50, 350)
                elif view_type == 'side':  
                    px, py = map_range(y, -1.0, 1.0, 50, 350), map_range(z, 2.0, -0.5, 50, 350)
                return (px + offset_x, py)

            for (idx1, idx2) in SKELETON_CONNECTIONS:
                if idx1 in pts and idx2 in pts:
                    cv2.line(canvas_3d, get_pt(pts[idx1]), get_pt(pts[idx2]), (255, 100, 100), 2)
                    
            for idx, pt in pts.items():
                p2d = get_pt(pt)
                cv2.circle(canvas_3d, p2d, 5, (0, 0, 255), -1)
                if view_type == 'front': 
                    cv2.putText(canvas_3d, str(idx), (p2d[0]+5, p2d[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            titles = {'front': "Front View (X-Z)", 'top': "Top View (X-Y)", 'side': "Side View (Y-Z)"}
            cv2.putText(canvas_3d, titles[view_type], (offset_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        draw_view(pts_3d_world, 'front', 0)
        draw_view(pts_3d_world, 'top', 400)
        draw_view(pts_3d_world, 'side', 800)

        # 🌟 ลบหน้าต่าง JSON ออก เพื่อลดภาระ CPU ใช้ Print แทน 🌟
        print(f"\rFPS: {fps:.1f} | Sending Data: {len(pts_3d_world)} points", end="")

        cv2.imshow(cv_3d_name, canvas_3d)
        
        cv2.putText(frame_l, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        display = cv2.hconcat([cv2.resize(frame_l, (400, 300)), cv2.resize(frame_r, (400, 300))])
        cv2.imshow(cv_window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27): break
        if cv2.getWindowProperty(cv_window_name, cv2.WND_PROP_VISIBLE) < 1: break

    cam_l.stop(); cam_r.stop()
    cv2.destroyAllWindows()
    executor.shutdown()

# ==========================================
# 6. Main GUI Menu
# ==========================================
class TrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AIO 3D Stereo Tracker (Optimized)")
        self.root.geometry("450x380")
        
        cam_opts = get_camera_list()
        tk.Label(root, text="🚀 All-in-One 3D Camera Setup", font=('Helvetica', 14, 'bold')).pack(pady=10)

        frame_cams = tk.Frame(root)
        frame_cams.pack(pady=5)
        
        tk.Label(frame_cams, text="Left Camera ID:").grid(row=0, column=0, padx=5)
        self.cb_left = ttk.Combobox(frame_cams, values=cam_opts, state="readonly", width=30)
        if cam_opts: self.cb_left.set(cam_opts[0])
        self.cb_left.grid(row=0, column=1, pady=2)

        tk.Label(frame_cams, text="Right Camera ID:").grid(row=1, column=0, padx=5)
        self.cb_right = ttk.Combobox(frame_cams, values=cam_opts, state="readonly", width=30)
        if len(cam_opts)>1: self.cb_right.set(cam_opts[1])
        elif cam_opts: self.cb_right.set(cam_opts[0])
        self.cb_right.grid(row=1, column=1, pady=2)

        tk.Label(root, text="--- ขั้นตอนการทำงาน ---", fg="gray").pack(pady=10)
        btn_style = {'width': 35, 'height': 2, 'font': ('Helvetica', 10, 'bold')}
        
        tk.Button(root, text="1. รัน Stereo Calibration", bg="#add8e6", command=self.btn_stereo, **btn_style).pack(pady=2)
        tk.Button(root, text="2. รัน Floor Calibration", bg="#90ee90", command=self.btn_floor, **btn_style).pack(pady=2)
        tk.Button(root, text="3. เริ่ม Real-time 3D Skeleton", bg="#ffb6c1", command=self.btn_track, **btn_style).pack(pady=2)

    def get_ids(self):
        val_l = self.cb_left.get().split(" - ")[0]
        val_r = self.cb_right.get().split(" - ")[0]
        # รองรับ Linux (Jetson) ที่ /dev/video* อาจจะเป็นแค่ตัวเลข หรือไม่มีขีด
        try: l = int(val_l)
        except: l = 0
        try: r = int(val_r)
        except: r = 1
        return l, r

    def btn_stereo(self):
        l, r = self.get_ids()
        self.root.withdraw()
        run_stereo_calibration(l, r)
        self.root.deiconify()

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