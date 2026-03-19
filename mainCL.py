import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import time
import math
import tkinter as tk
from tkinter import ttk
import subprocess 

sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# ฟังก์ชันพิเศษ: วางรูปภาพ PNG (มีพื้นหลังใส) ทับบนวิดีโอ
# ==========================================
def overlay_transparent(background, overlay, x, y):
    bg_h, bg_w, _ = background.shape
    ov_h, ov_w, ov_c = overlay.shape
    if x >= bg_w or y >= bg_h or x + ov_w <= 0 or y + ov_h <= 0:
        return background
    if x < 0 or y < 0 or x + ov_w > bg_w or y + ov_h > bg_h:
        return background
    if ov_c == 4:
        alpha = overlay[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha
        for c in range(3):
            background[y:y+ov_h, x:x+ov_w, c] = (
                alpha * overlay[:, :, c] +
                alpha_inv * background[y:y+ov_h, x:x+ov_w, c]
            )
    else:
        background[y:y+ov_h, x:x+ov_w] = overlay
    return background

# ==========================================
# ฟังก์ชันพิเศษ: วาดโครงกระดูก 3 มิติลงบนอีกหน้าต่าง
# ==========================================
def render_3d_skeleton(points_3d_dict, w=640, h=640):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    if len(points_3d_dict) < 5:
        cv2.putText(img, "Waiting for Stereo Vision...", (100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img

    # 1. พลิกแกน Y ทันที! (เพราะสมการภาพ Y ลงล่าง แต่ 3D ทั่วไป Y ต้องขึ้นบน)
    keys = list(points_3d_dict.keys())
    pts_list = []
    for idx in keys:
        pt = points_3d_dict[idx]
        pts_list.append([pt[0], -pt[1], pt[2]]) # 🔥 ใส่เครื่องหมายลบเพื่อกลับด้าน Y
        
    pts_array = np.array(pts_list)
    center = np.mean(pts_array, axis=0)
    centered_pts = pts_array - center

    # 2. มุมมองกล้อง
    theta_y = math.radians(-30) 
    theta_x = math.radians(15)  

    Ry = np.array([[math.cos(theta_y), 0, math.sin(theta_y)],
                   [0, 1, 0],
                   [-math.sin(theta_y), 0, math.cos(theta_y)]])
    
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(theta_x), -math.sin(theta_x)],
                   [0, math.sin(theta_x), math.cos(theta_x)]])

    max_dist = np.max(np.linalg.norm(centered_pts, axis=1))
    scale = (w / 2.5) / max_dist if max_dist > 0 else 1.0

    def project_pt(pt):
        pt_rot = Rx @ Ry @ pt
        px = int(pt_rot[0] * scale + w / 2)
        py = int(-pt_rot[1] * scale + h / 2) # Screen Y ลงล่าง เลยต้องสลับ
        return px, py

    # 3. สร้างพื้น (ตอนนี้ Y พุ่งขึ้นบนแล้ว เท้าเลยเป็นจุดที่ค่า Y ต่ำที่สุด/min)
    floor_y = np.min(centered_pts[:, 1]) - (max_dist * 0.1) 
    
    grid_size = max_dist * 1.5
    grid_step = grid_size / 5
    
    # วาดตารางพื้นสีเทา
    for z in np.arange(-grid_size, grid_size + grid_step, grid_step):
        p1 = np.array([-grid_size, floor_y, z])
        p2 = np.array([grid_size, floor_y, z])
        cv2.line(img, project_pt(p1), project_pt(p2), (40, 40, 40), 1) 
        
    for x in np.arange(-grid_size, grid_size + grid_step, grid_step):
        p1 = np.array([x, floor_y, -grid_size])
        p2 = np.array([x, floor_y, grid_size])
        cv2.line(img, project_pt(p1), project_pt(p2), (40, 40, 40), 1)
        
    # 4. วาดแกน X(แดง), Y(เขียว), Z(น้ำเงิน)
    axis_len = grid_size * 0.6
    p_org = project_pt(np.array([0, floor_y, 0]))
    p_x = project_pt(np.array([axis_len, floor_y, 0]))
    p_y = project_pt(np.array([0, floor_y + axis_len, 0])) # Y บวก พุ่งขึ้นบน!
    p_z = project_pt(np.array([0, floor_y, axis_len]))

    cv2.line(img, p_org, p_x, (0, 0, 255), 2) # X = แดง
    cv2.line(img, p_org, p_y, (0, 255, 0), 2) # Y = เขียว
    cv2.line(img, p_org, p_z, (255, 0, 0), 2) # Z = น้ำเงิน

    cv2.putText(img, "X", (p_x[0]+5, p_x[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(img, "Y", (p_y[0]+5, p_y[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img, "Z", (p_z[0]+5, p_z[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # 5. วาดจุด Skeleton
    projected = {}
    for i, idx in enumerate(keys):
        projected[idx] = project_pt(centered_pts[i])

    for connection in mp.solutions.pose.POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx in projected and end_idx in projected:
            cv2.line(img, projected[start_idx], projected[end_idx], (255, 255, 255), 2)

    for idx, (px, py) in projected.items():
        color = (0, 255, 0) 
        if idx in [15, 17, 19, 21]: color = (0, 0, 255) 
        if idx in [16, 18, 20, 22]: color = (255, 0, 0) 
        cv2.circle(img, (px, py), 4, color, -1)
        
    cv2.putText(img, "Live 3D Skeleton (Fixed Upright)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return img

# ==========================================
# 1. ตั้งค่า MediaPipe
# ==========================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=1
)

# ==========================================
# 2. ตัวแปรควบคุมระบบ EMA + Deadzone + Origin
# ==========================================
EMA_ALPHA_POS = 0.02   
EMA_ALPHA_RPY = 0.02   
EMA_ALPHA_GRP = 0.04   
DEADZONE_POS  = 8.0    
DEADZONE_RPY  = 4.0    
RESET_TIMEOUT = 2.0    

HOLD_COOLDOWN   = 2.0  
MARK_COOLDOWN   = 3.0  
HOLD_MARGIN     = 0.12 
FIST_THRESHOLD  = 0.03 

current_smoothed_pos     = np.array([0.0, 0.0, 0.0])
current_smoothed_rpy     = np.array([0.0, 0.0, 0.0])
current_smoothed_gripper = 0.0
is_first_frame           = True
last_valid_time          = time.time()

marked_position          = None
is_holding               = False
last_hold_toggle_time    = 0.0
last_mark_time           = 0.0

global_origin = np.zeros(3)
is_origin_set = False

# 🔥 ตัวแปรสำหรับแก้ภาพสั่น (Skeleton Filter)
smoothed_3d_skeleton = {}
SKELETON_EMA_ALPHA = 0.15 # ค่าน้อย = นิ่งมากแต่ขยับตามช้า, ค่ามาก = ไวแต่สั่น

# ==========================================
# 3. โหลดไฟล์ Stereo Calibration
# ==========================================
STEREO_MODE = False
P1, P2 = None, None
mtx_L, dist_L, mtx_R, dist_R = None, None, None, None
R1, R2 = None, None
map1_l, map2_l, map1_r, map2_r = None, None, None, None
CAM_W, CAM_H = 640, 480

if os.path.exists("stereo_params.npz"):
    try:
        params   = np.load("stereo_params.npz")
        mtx_L, dist_L = params['mtx_L'], params['dist_L']
        mtx_R, dist_R = params['mtx_R'], params['dist_R']
        
        if 'R1' in params and 'R2' in params:
            R1, R2 = params['R1'], params['R2']
            P1, P2 = params['P1'], params['P2']
        else:
            R, T = params['R'], params['T']
            R1, R2, P1_new, P2_new, _, _, _ = cv2.stereoRectify(mtx_L, dist_L, mtx_R, dist_R, (CAM_W, CAM_H), R, T)
            P1, P2 = P1_new, P2_new

        map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_L, dist_L, R1, P1, (CAM_W, CAM_H), cv2.CV_32FC1)
        map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_R, dist_R, R2, P2, (CAM_W, CAM_H), cv2.CV_32FC1)

        STEREO_MODE = True
        print("✅ โหลดไฟล์ Stereo สำเร็จ: ทำงานในโหมด 3D (Rectified)")
    except Exception as e:
        print(f"⚠️ โหลดไฟล์ Stereo ไม่สำเร็จ ({e})")

if not STEREO_MODE:
    print("⚠️ ไม่พบ stereo_params.npz → ทำงานในโหมดกล้องเดี่ยว (2D)")

icon_hold = cv2.imread("icon_hold.png", cv2.IMREAD_UNCHANGED)
icon_mark = cv2.imread("icon_mark.png", cv2.IMREAD_UNCHANGED)

# ==========================================
# 4. ฟังก์ชันคณิตศาสตร์
# ==========================================
def get_3d_point(p_left, p_right, proj1, proj2):
    pt_l  = np.array([[p_left[0]],  [p_left[1]]],  dtype=np.float64)
    pt_r  = np.array([[p_right[0]], [p_right[1]]], dtype=np.float64)
    pt_4d = cv2.triangulatePoints(proj1, proj2, pt_l, pt_r)
    pt_3d = pt_4d[:3] / pt_4d[3]
    return pt_3d.flatten()

def calculate_rpy(wrist, index, pinky):
    dx = index.x - wrist.x
    dy = index.y - wrist.y
    dz = index.z - wrist.z

    yaw   = math.degrees(math.atan2(dx, dz))
    pitch = math.degrees(math.atan2(-dy, math.sqrt(dx**2 + dz**2)))

    dx_side = pinky.x - wrist.x
    dy_side = pinky.y - wrist.y
    roll    = math.degrees(math.atan2(dy_side, dx_side))

    return np.array([roll, pitch, yaw])

def lm_px(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)

# ==========================================
# 4.5 ฟังก์ชันดึงชื่อกล้องและเปิดหน้าต่าง GUI
# ==========================================
def get_camera_list():
    cam_names = []
    if os.name == 'nt': 
        try:
            cmd = 'powershell -Command "Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -eq \'Image\' -or $_.PNPClass -eq \'Camera\' } | Select-Object -ExpandProperty Caption"'
            result = subprocess.run(cmd, capture_output=True, text=True, creationflags=0x08000000)
            if result.returncode == 0:
                cam_names = [line.strip() for line in result.stdout.split('\n') if line.strip()]
        except Exception as e:
            pass
    
    options = []
    for i in range(5): 
        name = cam_names[i] if i < len(cam_names) else f"Unknown Camera"
        options.append(f"{i} - {name}")
    return options

def show_camera_selection_gui():
    root = tk.Tk()
    root.title("ตั้งค่ากล้อง (Main Tracker)")
    
    window_width = 420
    window_height = 280
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    
    selected_cams = {"left": 0, "right": 1, "start": False}
    tk.Label(root, text="ตั้งค่ากล้องก่อนเริ่มระบบ Vision", font=('Helvetica', 12, 'bold')).pack(pady=15)

    cam_options = get_camera_list()

    tk.Label(root, text="กล้องซ้าย (Master - แกนหลัก):").pack()
    left_var = tk.StringVar(value=cam_options[0])
    left_cb = ttk.Combobox(root, textvariable=left_var, values=cam_options, state="readonly", width=45)
    left_cb.pack(pady=5)

    tk.Label(root, text="กล้องขวา (Stereo Depth - เพื่อหาความลึก):").pack()
    right_options = ["None (ใช้กล้องเดียว)"] + cam_options
    default_right = cam_options[1] if len(cam_options) > 1 else "None (ใช้กล้องเดียว)"
    right_var = tk.StringVar(value=default_right)
    right_cb = ttk.Combobox(root, textvariable=right_var, values=right_options, state="readonly", width=45)
    right_cb.pack(pady=5)

    def on_start():
        selected_cams["left"] = int(left_var.get().split(" - ")[0])
        r_val = right_var.get()
        if r_val == "None (ใช้กล้องเดียว)":
            selected_cams["right"] = None
        else:
            selected_cams["right"] = int(r_val.split(" - ")[0])
        selected_cams["start"] = True
        root.destroy()

    tk.Button(root, text="▶ เริ่มทำงาน (Start Tracking)", command=on_start, bg="green", fg="white", font=('Helvetica', 10, 'bold')).pack(pady=20)
    root.mainloop()
    return selected_cams["left"], selected_cams["right"], selected_cams["start"]

# ==========================================
# 5. เปิดกล้อง
# ==========================================
left_cam_id, right_cam_id, is_started = show_camera_selection_gui()

if not is_started:
    sys.exit()

cam_left  = cv2.VideoCapture(left_cam_id, cv2.CAP_DSHOW)
cam_right = None
if right_cam_id is not None:
    _cam_right_test = cv2.VideoCapture(right_cam_id, cv2.CAP_DSHOW)
    if _cam_right_test.isOpened():
        cam_right = _cam_right_test
    else:
        _cam_right_test.release()

for cam in filter(None, [cam_left, cam_right]):
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

cv2.namedWindow("Vision Tracking (Robot Target)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Vision Tracking (Robot Target)", 1280, 960)

if STEREO_MODE and cam_right is not None:
    cv2.namedWindow("3D Skeleton (Stereo Merged)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("3D Skeleton (Stereo Merged)", 640, 640)

# ==========================================
# 6. ลูปหลัก
# ==========================================
while True:
    ret_l, frame_l = cam_left.read()
    if not ret_l: break

    ret_r, frame_r = False, None
    if cam_right is not None:
        ret_r, frame_r = cam_right.read()

    raw_l = frame_l.copy()
    raw_r = frame_r.copy() if ret_r else np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)

    if STEREO_MODE:
        frame_l = cv2.remap(frame_l, map1_l, map2_l, cv2.INTER_LINEAR)
        if ret_r: frame_r = cv2.remap(frame_r, map1_r, map2_r, cv2.INTER_LINEAR)

    h, w = frame_l.shape[:2]

    rgb_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB)
    results_l = pose.process(rgb_l)

    results_r = None
    if ret_r:
        rgb_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB)
        results_r = pose.process(rgb_r)

    if results_l.pose_landmarks: mp_drawing.draw_landmarks(frame_l, results_l.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    if results_r and results_r.pose_landmarks: mp_drawing.draw_landmarks(frame_r, results_r.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # ==========================================
    # 🔥 สร้าง Full 3D Skeleton + กรองค่าแก้สั่น
    # ==========================================
    full_3d_skeleton = {}
    
    if STEREO_MODE and results_l.pose_landmarks and results_r and results_r.pose_landmarks:
        for i in range(33):
            lm_l = results_l.pose_landmarks.landmark[i]
            lm_r = results_r.pose_landmarks.landmark[i]
            
            if lm_l.visibility > 0.4 and lm_r.visibility > 0.4:
                px_l = lm_px(lm_l, w, h)
                px_r = lm_px(lm_r, w, h)
                
                unflipped_x_l = w - px_l[0]
                unflipped_x_r = w - px_r[0]
                
                pt3d = get_3d_point([unflipped_x_l, px_l[1]], [unflipped_x_r, px_r[1]], P1, P2)
                
                # นำจุด 3D เข้าสมการ Filter เพื่อแก้การสั่น
                if i not in smoothed_3d_skeleton:
                    smoothed_3d_skeleton[i] = pt3d
                else:
                    smoothed_3d_skeleton[i] = (SKELETON_EMA_ALPHA * pt3d) + ((1.0 - SKELETON_EMA_ALPHA) * smoothed_3d_skeleton[i])
                
                full_3d_skeleton[i] = smoothed_3d_skeleton[i]

        img_3d_view = render_3d_skeleton(full_3d_skeleton, 640, 640)
        cv2.imshow("3D Skeleton (Stereo Merged)", img_3d_view)
    else:
        # ถ้าระบบหลุดโฟกัส ให้รีเซ็ตค่าสมูท จะได้ไม่ค้าง
        smoothed_3d_skeleton.clear()

    # ==========================================
    # 6.1 หาตำแหน่ง TCP และควบคุม Gripper
    # ==========================================
    raw_target_pos      = None
    raw_target_rpy      = None
    raw_gripper_percent = None
    vis_score_l         = 0.0

    if results_l.pose_landmarks and results_l.pose_world_landmarks:
        lm   = results_l.pose_landmarks
        wlm  = results_l.pose_world_landmarks

        left_wrist    = lm.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        left_shoulder = lm.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_index    = lm.landmark[mp_pose.PoseLandmark.LEFT_INDEX]
        left_thumb    = lm.landmark[mp_pose.PoseLandmark.LEFT_THUMB]

        if left_wrist.visibility > 0.6:
            current_time = time.time()
            wx_l, wy_l   = lm_px(left_wrist, w, h)

            is_hand_raised = left_wrist.y < left_shoulder.y - HOLD_MARGIN
            if is_hand_raised and (current_time - last_hold_toggle_time > HOLD_COOLDOWN):
                is_holding = not is_holding
                last_hold_toggle_time = current_time
                cv2.circle(frame_l, (wx_l, wy_l), 40, (0, 165, 255), -1)

            dist_lf = math.sqrt((left_index.x - left_thumb.x)**2 + (left_index.y - left_thumb.y)**2 + (left_index.z - left_thumb.z)**2)
            is_fist = dist_lf < FIST_THRESHOLD
            if is_fist and (current_time - last_mark_time > MARK_COOLDOWN):
                marked_position = current_smoothed_pos.copy() - global_origin
                last_mark_time  = current_time
                cv2.circle(frame_l, (wx_l, wy_l), 40, (255, 255, 0), -1)

        right_wrist  = lm.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        index_finger = lm.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
        thumb_finger = lm.landmark[mp_pose.PoseLandmark.RIGHT_THUMB]
        pinky_finger = lm.landmark[mp_pose.PoseLandmark.RIGHT_PINKY]

        w_right_wrist  = wlm.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        w_index_finger = wlm.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
        w_pinky_finger = wlm.landmark[mp_pose.PoseLandmark.RIGHT_PINKY]

        vis_score_l = (index_finger.visibility + thumb_finger.visibility) / 2

        if vis_score_l > 0.1:
            ix_l, iy_l = lm_px(index_finger, w, h)
            tx_l, ty_l = lm_px(thumb_finger, w, h)
            tcp_x_l    = (ix_l + tx_l) // 2
            tcp_y_l    = (iy_l + ty_l) // 2

            raw_target_rpy = calculate_rpy(w_right_wrist, w_index_finger, w_pinky_finger)

            if STEREO_MODE and results_r and results_r.pose_landmarks and results_r.pose_world_landmarks:
                lm_r  = results_r.pose_landmarks
                wlm_r = results_r.pose_world_landmarks

                idx_r   = lm_r.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
                thb_r   = lm_r.landmark[mp_pose.PoseLandmark.RIGHT_THUMB]
                pnk_r   = lm_r.landmark[mp_pose.PoseLandmark.RIGHT_PINKY]
                wrist_r = lm_r.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

                w_idx_r   = wlm_r.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
                w_thb_r   = wlm_r.landmark[mp_pose.PoseLandmark.RIGHT_THUMB]
                w_pnk_r   = wlm_r.landmark[mp_pose.PoseLandmark.RIGHT_PINKY]
                w_wrist_r = wlm_r.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

                vis_score_r = (idx_r.visibility + thb_r.visibility) / 2

                ix_r, iy_r = lm_px(idx_r, w, h)
                tx_r, ty_r = lm_px(thb_r, w, h)
                tcp_x_r    = (ix_r + tx_r) // 2
                tcp_y_r    = (iy_r + ty_r) // 2

                unflipped_tcp_x_l = w - tcp_x_l
                unflipped_tcp_x_r = w - tcp_x_r

                pt_3d = get_3d_point([unflipped_tcp_x_l, tcp_y_l], [unflipped_tcp_x_r, tcp_y_r], P1, P2)
                raw_target_pos = pt_3d

                if vis_score_l < 0.3 and vis_score_r > 0.5:
                    raw_target_rpy = calculate_rpy(w_wrist_r, w_idx_r, w_pnk_r)
                    idx_3d = get_3d_point([w - ix_r, iy_r], [w - ix_l, iy_l], P1, P2)
                    thb_3d = get_3d_point([w - tx_r, ty_r], [w - tx_l, ty_l], P1, P2)
                    dist_mm = np.linalg.norm(idx_3d - thb_3d) * 1000
                else:
                    idx_3d  = get_3d_point([w - ix_l, iy_l], [w - ix_r, iy_r], P1, P2)
                    thb_3d  = get_3d_point([w - tx_l, ty_l], [w - tx_r, ty_r], P1, P2)
                    dist_mm = np.linalg.norm(idx_3d - thb_3d) * 1000

                raw_gripper_percent = float(np.clip((dist_mm - 20) / (120 - 20) * 100, 0, 100))
                cv2.circle(frame_r, (tcp_x_r, tcp_y_r), 10, (0, 0, 255), -1)

            else:
                raw_target_pos = np.array([
                    float(tcp_x_l),
                    float(tcp_y_l),
                    current_smoothed_pos[2] if not is_first_frame else 0.0
                ])
                dist_norm = math.sqrt((index_finger.x - thumb_finger.x)**2 + (index_finger.y - thumb_finger.y)**2 + (index_finger.z - thumb_finger.z)**2)
                raw_gripper_percent = float(np.clip((dist_norm - 0.02) / (0.12 - 0.02) * 100, 0, 100))

            target_color = (0, 0, 255) if vis_score_l > 0.4 else (0, 165, 255)
            cv2.circle(frame_l, (tcp_x_l, tcp_y_l), 10, target_color, -1)
            cv2.line(frame_l, lm_px(index_finger, w, h), lm_px(thumb_finger, w, h), target_color, 2)

    # ==========================================
    # 6.2 EMA Filter (ส่วนควบคุมหุ่น)
    # ==========================================
    if raw_target_pos is not None and raw_target_rpy is not None and raw_gripper_percent is not None:
        last_valid_time = time.time()

        if is_first_frame:
            current_smoothed_pos     = raw_target_pos.copy()
            current_smoothed_rpy     = raw_target_rpy.copy()
            current_smoothed_gripper = raw_gripper_percent
            is_first_frame           = False
        elif not is_holding:
            delta_pos = np.linalg.norm(raw_target_pos - current_smoothed_pos)
            delta_rpy = np.linalg.norm(raw_target_rpy - current_smoothed_rpy)

            if delta_pos > DEADZONE_POS:
                current_smoothed_pos = (EMA_ALPHA_POS * raw_target_pos + (1.0 - EMA_ALPHA_POS) * current_smoothed_pos)
            if delta_rpy > DEADZONE_RPY:
                current_smoothed_rpy = (EMA_ALPHA_RPY * raw_target_rpy + (1.0 - EMA_ALPHA_RPY) * current_smoothed_rpy)
            current_smoothed_gripper = (EMA_ALPHA_GRP * raw_gripper_percent + (1.0 - EMA_ALPHA_GRP) * current_smoothed_gripper)
    else:
        if time.time() - last_valid_time > RESET_TIMEOUT:
            is_first_frame = True

    relative_pos = current_smoothed_pos - global_origin

    # ==========================================
    # 7. วาด UI
    # ==========================================
    disp_l = cv2.flip(frame_l, 1)

    overlay_buf = disp_l.copy()
    bottom_bar_h = 80
    cv2.rectangle(overlay_buf, (10, 10), (430, 200), (0, 0, 0), -1)
    cv2.rectangle(overlay_buf, (0, h - bottom_bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay_buf, 0.6, disp_l, 0.4, 0, disp_l)

    if is_holding:
        status_text, s_color = "HOLDING (Manual)", (0, 165, 255)
    elif raw_target_pos is not None:
        if vis_score_l > 0.4:
            status_text, s_color = "Tracking 6-DOF (Clear)", (0, 255, 0)
        else:
            status_text, s_color = "Tracking (Occluded)", (0, 255, 255)
    else:
        status_text, s_color = "Vision Lost!", (0, 0, 255)

    cv2.putText(disp_l, f"Status: {status_text}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, s_color, 2)
    cv2.putText(disp_l, f"P(X,Y,Z): {relative_pos[0]:.1f}, {relative_pos[1]:.1f}, {relative_pos[2]:.1f}",
                (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if is_origin_set:
        cv2.putText(disp_l, "[Origin SET]", (260, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.putText(disp_l, f"O(R,P,Y): {current_smoothed_rpy[0]:.1f}, {current_smoothed_rpy[1]:.1f}, {current_smoothed_rpy[2]:.1f}",
                (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(disp_l, f"Gripper: {current_smoothed_gripper:.0f}%",
                (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
                
    if marked_position is not None:
        cv2.putText(disp_l, f"Marked: {marked_position[0]:.1f}, {marked_position[1]:.1f}, {marked_position[2]:.1f}",
                    (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    yg = h - bottom_bar_h + 20
    cv2.putText(disp_l, "Gestures:", (20, yg + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(disp_l, "Raise Left Hand=HOLD", (140, yg + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    cv2.putText(disp_l, "Left Fist=MARK", (370, yg + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(disp_l, "[O] Key=Zero Origin", (520, yg + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # ==========================================
    # 8. แสดงผล 4 จอหลัก
    # ==========================================
    TW, TH = 640, 480

    raw_l_disp = cv2.resize(cv2.flip(raw_l, 1), (TW, TH))
    cv2.putText(raw_l_disp, "1. RAW CAM 1 (Left)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    raw_r_disp = cv2.resize(cv2.flip(raw_r, 1), (TW, TH))
    cv2.putText(raw_r_disp, "2. RAW CAM 2 (Right)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    calib_l_disp = cv2.resize(disp_l, (TW, TH))
    cv2.putText(calib_l_disp, "3. CAM 1 (Processed)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if ret_r and frame_r is not None:
        disp_r = frame_r.copy()
        if results_r and results_r.pose_landmarks:
            mp_drawing.draw_landmarks(disp_r, results_r.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        disp_r = cv2.flip(disp_r, 1)
        calib_r_disp = cv2.resize(disp_r, (TW, TH))
        label_r = "4. CAM 2 (Stereo)" if STEREO_MODE else "4. CAM 2 (No Stereo File)"
    else:
        calib_r_disp = np.zeros((TH, TW, 3), dtype=np.uint8)
        label_r = "4. CAM 2 (Not Connected)"
    cv2.putText(calib_r_disp, label_r, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    top_row      = cv2.hconcat([raw_l_disp, raw_r_disp])
    bottom_row   = cv2.hconcat([calib_l_disp, calib_r_disp])
    display_frame = cv2.vconcat([top_row, bottom_row])

    cv2.imshow("Vision Tracking (Robot Target)", display_frame)

    # ==========================================
    # 9. คีย์บอร์ด
    # ==========================================
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break
    elif key in (ord('h'), ord('H')):
        is_holding = not is_holding
    elif key in (ord('m'), ord('M')):
        marked_position = current_smoothed_pos.copy() - global_origin
    elif key in (ord('o'), ord('O'), ord('z'), ord('Z')):
        global_origin = current_smoothed_pos.copy()
        is_origin_set = True
    elif key in (ord('c'), ord('C')):
        global_origin = np.zeros(3)
        is_origin_set = False

# ==========================================
# 10. ปิดกล้อง
# ==========================================
cam_left.release()
if cam_right is not None:
    cam_right.release()
cv2.destroyAllWindows()