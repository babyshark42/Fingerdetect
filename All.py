import cv2
import cv2.aruco as aruco
import numpy as np
import sys
import os
import time
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import mediapipe as mp # นำเข้า MediaPipe สำหรับจับ Skeleton
import asyncio
import websockets
import json
import threading

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
            
        if ch_corn is not None and len(ch_corn) >= 6:
            aruco.drawDetectedCornersCharuco(annotated, ch_corn, ch_ids)
            return annotated, ch_corn, ch_ids, True
    return annotated, None, None, False

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

    cam_l.release()
    cv2.destroyAllWindows()

# ==========================================
# 4.5 ระบบ WebSocket Server (ส่งข้อมูลไป Web Browser)
# ==========================================
latest_skeleton_data = {
    "persons": [],
    "timestamp": 0
}

async def skeleton_ws_handler(websocket):
    print("🌐 Web Client Connected!")
    try:
        while True:
            # ส่งข้อมูลล่าสุดไปให้ Browser ทุกๆ ~15fps (0.066 วินาที)
            await websocket.send(json.dumps(latest_skeleton_data))
            await asyncio.sleep(1/15)
    except websockets.exceptions.ConnectionClosed:
        print("🌐 Web Client Disconnected!")

# 🌟 เพิ่มการดักจับ Error กรณีพอร์ตชนกัน (อัปเดตใหม่ให้คลุม Windows)
async def main_ws():
    print("🚀 กำลังเปิด WebSocket Server ที่พอร์ต 8767...")
    try:
        async with websockets.serve(skeleton_ws_handler, "0.0.0.0", 8767):
            print("✅ WebSocket Server รันสำเร็จแล้ว!")
            await asyncio.Future()  # สั่งให้รันลูปค้างไว้ตลอดไป
    except OSError as e:
        # เช็คทั้งรหัส 10048 (Windows) และ 98 (Linux/Mac) หรือข้อความที่มี 10048
        if e.errno in (98, 10048) or "10048" in str(e):
            print("\n" + "="*50)
            print("❌ [ERROR] ไม่สามารถเปิด WebSocket ได้: พอร์ต 8767 ถูกใช้งานอยู่")
            print("❌ [วิธีแก้] กรุณาพิมพ์คำสั่งนี้ใน Terminal ทิ้งไว้ 1 รอบ:")
            print("   👉  npx kill-port 8767")
            print("="*50 + "\n")
        else:
            print(f"❌ [ERROR] WebSocket เกิดข้อผิดพลาด: {e}")

def start_ws_server():
    # สร้าง Event Loop สำหรับรัน WebSockets ใน Thread แยกด้วยวิธีใหม่
    asyncio.run(main_ws())

# ==========================================
# 5. โมดูล 3: Real-time 3D Skeleton Tracking
# ==========================================
def run_3d_tracker(l_id, r_id):
    if not os.path.exists("stereo_params.npz") or not os.path.exists("floor_params.npz"):
        messagebox.showerror("Error", "ต้องทำ Calibration ให้ครบทั้ง Stereo และ Floor ก่อนครับ!")
        return

    # โหลดค่า Parameter
    stereo = np.load("stereo_params.npz")
    P1 = stereo['mtx_L'] @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = stereo['mtx_R'] @ np.hstack((stereo['R'], stereo['T']))

    floor = np.load("floor_params.npz")
    R_floor, _ = cv2.Rodrigues(floor['rvec'])
    R_floor_inv = np.linalg.inv(R_floor)
    tvec_floor = floor['tvec']

    # 11 จุดสำคัญที่คุณต้องการ (Height, Torso, Arms)
    TARGET_LANDMARKS = [0, 27, 28, 11, 12, 23, 24, 13, 14, 15, 16]

    # จับคู่เส้นที่จะใช้วาด (กระดูก) ในรูปแบบ (จุดเริ่มต้น, จุดสิ้นสุด)
    SKELETON_CONNECTIONS = [
        (11, 12), (12, 24), (24, 23), (23, 11),  # Torso (กรอบสี่เหลี่ยมลำตัว)
        (11, 13), (13, 15),                      # Left Arm (แขนซ้าย)
        (12, 14), (14, 16),                      # Right Arm (แขนขวา)
        (0, 11), (0, 12),                        # Head to Torso (จมูกไปไหล่)
        (23, 27), (24, 28)                       # Torso to Ankles (สะโพกไปข้อเท้า)
    ]

    # ตั้งค่า MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose_l = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose_r = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # เปิดกล้อง
    cam_l = cv2.VideoCapture(l_id, cv2.CAP_DSHOW)
    cam_r = cv2.VideoCapture(r_id, cv2.CAP_DSHOW)

    # 🌟 เพิ่มการเช็คว่าเปิดกล้องสำเร็จหรือไม่ 🌟
    if not cam_l.isOpened() or not cam_r.isOpened():
        messagebox.showerror("Error", "เปิดกล้องไม่สำเร็จ!\nกรุณาเช็ค ID กล้อง หรือตรวจสอบว่ามีโปรแกรมอื่นใช้งานกล้องอยู่หรือไม่")
        cam_l.release()
        cam_r.release()
        return

    # 🌟 สตาร์ท WebSocket Server แบบ Background Thread 🌟
    ws_thread = threading.Thread(target=start_ws_server, daemon=True)
    ws_thread.start()

    # ฟังก์ชันสกัดจุด 2D จากภาพ
    def get_target_keypoints(frame, pose):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        keypoints = {}
        h, w, _ = frame.shape
        
        if results.pose_landmarks:
            for idx in TARGET_LANDMARKS:
                lm = results.pose_landmarks.landmark[idx]
                if lm.visibility > 0.5: # เอาเฉพาะจุดที่มองเห็นชัดเจน (มั่นใจเกิน 50%)
                    px, py = int(lm.x * w), int(lm.y * h)
                    keypoints[idx] = (px, py)
                    cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
                    cv2.putText(frame, str(idx), (px+5, py-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
        return frame, keypoints

    print("\n--- เริ่ม 3D Skeleton Tracking ---")
    print("กด 'Q' ที่หน้าต่างกล้องเพื่อออก หรือกดกากบาท (X) เพื่อปิด")
    
    cv_window_name = "3. Tracking Cameras (Left | Right)"
    cv_3d_name = "3D Skeleton Views (Real-time OpenCV)"
    cv_data_name = "WebSocket Data Stream" # 🌟 ชื่อหน้าต่างใหม่
    
    cv2.namedWindow(cv_window_name)
    cv2.namedWindow(cv_3d_name)
    cv2.namedWindow(cv_data_name) # 🌟 สร้างหน้าต่าง

    global latest_skeleton_data # เรียกใช้ตัวแปรโกลบอลเพื่อส่งข้อมูล

    # ฟังก์ชันช่วยแปลงพิกัด 3D ให้อยู่ในสเกลของหน้าจอ (Pixel)
    def map_range(val, in_min, in_max, out_min, out_max):
        return int((val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

    while True:
        ret_l, frame_l = cam_l.read()
        ret_r, frame_r = cam_r.read()
        if not ret_l or not ret_r: break

        # 1. หาจุดโครงกระดูก 2D ของแต่ละกล้อง
        frame_l, kps_l = get_target_keypoints(frame_l, pose_l)
        frame_r, kps_r = get_target_keypoints(frame_r, pose_r)

        pts_3d_world = {}

        # 2. Triangulate ทีละจุด
        for idx in TARGET_LANDMARKS:
            if idx in kps_l and idx in kps_r:
                pt_l = np.array([[kps_l[idx][0], kps_l[idx][1]]], dtype=np.float32)
                pt_r = np.array([[kps_r[idx][0], kps_r[idx][1]]], dtype=np.float32)
                
                # คำนวณ 3D
                pt_4d = cv2.triangulatePoints(P1, P2, pt_l.T, pt_r.T)
                pt_3d_cam = pt_4d[:3] / pt_4d[3] 
                
                # นำไปอิงระนาบพื้นโลก
                pt_world = R_floor_inv @ (pt_3d_cam - tvec_floor)
                xw, yw, zw = pt_world.flatten()
                
                # เก็บค่า X, Y, Z (ปรับ Z ให้ชี้ขึ้นฟ้า)
                pts_3d_world[idx] = [xw, yw, -zw]

        # ==========================================
        # 🌟 เตรียมข้อมูล JSON Format ส่งเข้าเว็บ (ตามโครงสร้างของคุณ)
        # ==========================================
        if pts_3d_world:
            keypoints_data = {}
            for idx, pt in pts_3d_world.items():
                # 🌟 สลับแกนให้ตรงกับ Three.js (Web 3D) 🌟
                # Web แกน Y คือความสูง (เอา pt[2] มาใส่)
                # Web แกน Z คือความลึก (เอา pt[1] มาใส่) 
                # (ถ้าในเว็บคนเดินถอยหลัง ให้ลองใส่เครื่องหมายลบเป็น -float(pt[1]) ดูครับ)
                keypoints_data[str(idx)] = {
                    "x": float(pt[0]),
                    "y": float(pt[2]), 
                    "z": float(pt[1]), 
                    "visibility": 1.0 # (สมมติว่าถ้าหาเจอ = 1.0)
                }
            
            latest_skeleton_data = {
                "persons": [{
                    "id": 0,
                    "keypoints": keypoints_data
                }],
                "timestamp": int(time.time() * 1000)
            }
        else:
            latest_skeleton_data = {
                "persons": [],
                "timestamp": int(time.time() * 1000)
            }

        # ==========================================
        # 3. วาดกราฟด้วย OpenCV ล้วนๆ (รับประกันความลื่น 60 FPS)
        # ==========================================
        # สร้างกระดานดำแนวยาว ขนาด 1200x400
        canvas_3d = np.zeros((400, 1200, 3), dtype=np.uint8)
        
        def draw_view(pts, view_type, offset_x):
            # ตีเส้นขอบหน้าต่างย่อย
            cv2.rectangle(canvas_3d, (offset_x, 0), (offset_x + 400, 400), (50, 50, 50), 1)
            
            # แปลง 3D เป็น 2D Pixel
            def get_pt(pt_3d):
                x, y, z = pt_3d
                if view_type == 'front':   # มองจากด้านหน้า (X-Z)
                    px = map_range(x, -1.0, 1.0, 50, 350)
                    py = map_range(z, 2.0, -0.5, 50, 350) # Z สูงสุดอยู่ขอบบน
                elif view_type == 'top':   # มองจากด้านบน (X-Y)
                    px = map_range(x, -1.0, 1.0, 50, 350)
                    py = map_range(y, 1.0, -1.0, 50, 350)
                elif view_type == 'side':  # มองจากด้านข้าง (Y-Z)
                    px = map_range(y, -1.0, 1.0, 50, 350)
                    py = map_range(z, 2.0, -0.5, 50, 350)
                return (px + offset_x, py)

            # วาดเส้นกระดูก
            for (idx1, idx2) in SKELETON_CONNECTIONS:
                if idx1 in pts and idx2 in pts:
                    cv2.line(canvas_3d, get_pt(pts[idx1]), get_pt(pts[idx2]), (255, 100, 100), 2)
                    
            # วาดจุดข้อต่อ
            for idx, pt in pts.items():
                p2d = get_pt(pt)
                cv2.circle(canvas_3d, p2d, 5, (0, 0, 255), -1)
                # ใส่ตัวเลขเฉพาะจอหน้า
                if view_type == 'front': 
                    cv2.putText(canvas_3d, str(idx), (p2d[0]+5, p2d[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # ใส่หัวข้อ
            titles = {'front': "Front View (X-Z)", 'top': "Top View (X-Y)", 'side': "Side View (Y-Z)"}
            cv2.putText(canvas_3d, titles[view_type], (offset_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # วาดทั้ง 3 มุมมองลงไปบนกระดานดำ
        draw_view(pts_3d_world, 'front', 0)
        draw_view(pts_3d_world, 'top', 400)
        draw_view(pts_3d_world, 'side', 800)

        # นำภาพ 3D เสมือนขึ้นจอ
        cv2.imshow(cv_3d_name, canvas_3d)

        # วิดีโอสตรีม
        display = cv2.hconcat([cv2.resize(frame_l, (400, 300)), cv2.resize(frame_r, (400, 300))])
        cv2.imshow(cv_window_name, display)

        # ==========================================
        # 🌟 4. สร้างหน้าต่างสำหรับดูข้อมูล JSON (Data Stream)
        # ==========================================
        # สร้างกระดานสีดำ ขนาด 400x600 (กว้างxสูง)
        data_canvas = np.zeros((600, 400, 3), dtype=np.uint8)
        cv2.putText(data_canvas, "Sending to WebSocket (Port 8767):", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # จัด Format JSON ให้สวยงาม (มี Indent)
        json_str = json.dumps(latest_skeleton_data, indent=2)
        
        # OpenCV ไม่รองรับการขึ้นบรรทัดใหม่ (\n) ต้อง Loop ปริ้นท์ทีละบรรทัด
        y_offset = 60
        for line in json_str.split('\n'):
            # ปริ้นท์ข้อความสีขาว
            cv2.putText(data_canvas, line[:60], (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
            if y_offset > 580: # ถ้าข้อมูลยาวเกินหน้าจอ ให้แสดง ...
                cv2.putText(data_canvas, "...", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                break
                
        cv2.imshow(cv_data_name, data_canvas)

        # ==========================================
        # เงื่อนไขการออกจากโปรแกรม
        # ==========================================
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27): 
            break
        if cv2.getWindowProperty(cv_window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.getWindowProperty(cv_3d_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        # 🌟 เช็คว่าหน้าต่าง Data โดนกดปิดไหม
        if cv2.getWindowProperty(cv_data_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cam_l.release(); cam_r.release()
    cv2.destroyAllWindows()

# ==========================================
# 6. Main GUI Menu
# ==========================================
class TrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AIO 3D Stereo Tracker")
        self.root.geometry("450x380")
        
        cam_opts = get_camera_list()
        tk.Label(root, text="🚀 All-in-One 3D Camera Setup", font=('Helvetica', 14, 'bold')).pack(pady=10)

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

        tk.Label(root, text="--- ขั้นตอนการทำงาน ---", fg="gray").pack(pady=10)
        btn_style = {'width': 35, 'height': 2, 'font': ('Helvetica', 10, 'bold')}
        
        tk.Button(root, text="1. รัน Stereo Calibration (หาค่ากล้อง)", bg="#add8e6", 
                  command=self.btn_stereo, **btn_style).pack(pady=2)
                  
        tk.Button(root, text="2. รัน Floor Calibration (หาพิกัดพื้น)", bg="#90ee90", 
                  command=self.btn_floor, **btn_style).pack(pady=2)
                  
        tk.Button(root, text="3. เริ่ม Real-time 3D Skeleton", bg="#ffb6c1", 
                  command=self.btn_track, **btn_style).pack(pady=2)

    def get_ids(self):
        return int(self.cb_left.get().split(" - ")[0]), int(self.cb_right.get().split(" - ")[0])

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