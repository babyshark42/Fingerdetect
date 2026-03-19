import cv2
import mediapipe as mp
import numpy as np
import os
import sys
import time
import math

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
# 2. ตัวแปรควบคุมระบบ EMA + Deadzone
# ==========================================
EMA_ALPHA_POS = 0.02   # หน่วงมากขึ้น — ค่าต่ำ = smooth กว่า แต่ตามช้ากว่า
EMA_ALPHA_RPY = 0.02   # RPY หน่วงสูงสุด
EMA_ALPHA_GRP = 0.04   # gripper หน่วงขึ้น
DEADZONE_POS  = 8.0    # mm — ไม่อัปเดตถ้าขยับน้อยกว่านี้ (กันสั่น)
DEADZONE_RPY  = 4.0    # องศา — กัน RPY กระตุก
RESET_TIMEOUT = 2.0    # วินาที — reset EMA ถ้า vision หายนาน

# --- Gesture thresholds ---
HOLD_COOLDOWN   = 2.0  # วินาที ระหว่าง toggle hold (เดิม 1.0 — ลั่นง่าย)
MARK_COOLDOWN   = 3.0  # วินาที ระหว่าง mark (เดิม 2.0)
HOLD_MARGIN     = 0.12 # มือต้องสูงกว่าไหล่ N หน่วย (เดิม 0.05 — trigger ง่ายเกิน)
FIST_THRESHOLD  = 0.03 # ระยะนิ้วชี้-หัวแม่มือถือว่ากำ (เดิม 0.04 — กำง่ายเกิน)

current_smoothed_pos     = np.array([0.0, 0.0, 0.0])
current_smoothed_rpy     = np.array([0.0, 0.0, 0.0])
current_smoothed_gripper = 0.0
is_first_frame           = True
last_valid_time          = time.time()

marked_position          = None
is_holding               = False
last_hold_toggle_time    = 0.0
last_mark_time           = 0.0

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
        P1, P2   = params['P1'], params['P2']
        R1, R2   = params['R1'], params['R2']
        mtx_L, dist_L = params['mtx_L'], params['dist_L']
        mtx_R, dist_R = params['mtx_R'], params['dist_R']

        # สร้าง remap maps ล่วงหน้า (คำนวณครั้งเดียว ไม่ต้องทำในลูป)
        map1_l, map2_l = cv2.initUndistortRectifyMap(
            mtx_L, dist_L, R1, P1, (CAM_W, CAM_H), cv2.CV_32FC1)
        map1_r, map2_r = cv2.initUndistortRectifyMap(
            mtx_R, dist_R, R2, P2, (CAM_W, CAM_H), cv2.CV_32FC1)

        STEREO_MODE = True
        print("✅ โหลดไฟล์ Stereo สำเร็จ: ทำงานในโหมด 3D (Rectified)")
    except Exception as e:
        print(f"⚠️ โหลดไฟล์ Stereo ไม่สำเร็จ ({e})")

if not STEREO_MODE:
    print("⚠️ ไม่พบ stereo_params.npz → ทำงานในโหมดกล้องเดี่ยว (2D)")

# โหลดไอคอน
icon_hold = cv2.imread("icon_hold.png", cv2.IMREAD_UNCHANGED)
icon_mark = cv2.imread("icon_mark.png", cv2.IMREAD_UNCHANGED)
if icon_hold is None or icon_mark is None:
    print("⚠️ ไม่พบ icon_hold.png / icon_mark.png → แสดงเป็นข้อความแทน")

# ==========================================
# 4. ฟังก์ชันคณิตศาสตร์
# ==========================================
def get_3d_point(p_left, p_right, proj1, proj2):
    """Triangulate จุดเดียว คืนค่า [X, Y, Z] หน่วยเดียวกับ T ใน stereo calibration"""
    pt_l  = np.array([[p_left[0]],  [p_left[1]]],  dtype=np.float64)
    pt_r  = np.array([[p_right[0]], [p_right[1]]], dtype=np.float64)
    pt_4d = cv2.triangulatePoints(proj1, proj2, pt_l, pt_r)
    pt_3d = pt_4d[:3] / pt_4d[3]
    return pt_3d.flatten()

def calculate_rpy(wrist, index, pinky):
    """
    คำนวณ Roll, Pitch, Yaw จาก world_landmarks (หน่วยเมตร)
    - wrist / index / pinky คือ landmark objects ที่มี .x .y .z จริง
    """
    dx = index.x - wrist.x
    dy = index.y - wrist.y
    dz = index.z - wrist.z

    # ✅ แก้: ไม่ abs(dz) เพื่อให้ทิศทาง Yaw ถูกต้อง
    yaw   = math.degrees(math.atan2(dx, dz))
    pitch = math.degrees(math.atan2(-dy, math.sqrt(dx**2 + dz**2)))

    dx_side = pinky.x - wrist.x
    dy_side = pinky.y - wrist.y
    roll    = math.degrees(math.atan2(dy_side, dx_side))

    return np.array([roll, pitch, yaw])

def lm_px(landmark, w, h):
    """แปลง normalized landmark → pixel บนภาพดิบ (ก่อน flip)"""
    return int(landmark.x * w), int(landmark.y * h)

# ==========================================
# 5. เปิดกล้อง
# ==========================================
cam_left  = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# เปิดกล้องขวาเสมอเพื่อตรวจสอบก่อน แล้วค่อยตัดสินใจ STEREO_MODE
_cam_right_test = cv2.VideoCapture(2, cv2.CAP_DSHOW)
if _cam_right_test.isOpened():
    cam_right = _cam_right_test
    if not STEREO_MODE:
        print("ℹ️  พบกล้องขวา แต่ไม่มีไฟล์ stereo_params.npz → แสดงภาพกล้องขวาแต่ไม่ triangulate")
else:
    _cam_right_test.release()
    cam_right = None
    print("ℹ️  ไม่พบกล้องขวา (ID=1) → โหมดกล้องเดี่ยว")

for cam in filter(None, [cam_left, cam_right]):
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

cv2.namedWindow("Vision Tracking (Robot Target)", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Vision Tracking (Robot Target)", 1280, 960)

print("==========================================")
print("เริ่มระบบ Vision Tracking...")
print("  [H] Toggle Hold   [M] Mark Position   [Q/ESC] ออก")
print("==========================================")

# ==========================================
# 6. ลูปหลัก
# ==========================================
while True:
    ret_l, frame_l = cam_left.read()
    if not ret_l:
        break

    ret_r, frame_r = False, None
    if cam_right is not None:
        ret_r, frame_r = cam_right.read()

    # --- เก็บ raw ก่อน remap (เพื่อแสดงแถวบน) ---
    raw_l = frame_l.copy()
    raw_r = frame_r.copy() if ret_r else np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)

    # ==========================================
    # ✅ แก้จุดสำคัญ #3: Rectify ด้วย remap (ไม่ใช่แค่ undistort)
    # ==========================================
    if STEREO_MODE:
        frame_l = cv2.remap(frame_l, map1_l, map2_l, cv2.INTER_LINEAR)
        if ret_r:
            frame_r = cv2.remap(frame_r, map1_r, map2_r, cv2.INTER_LINEAR)

    h, w = frame_l.shape[:2]

    # ==========================================
    # ✅ แก้จุดสำคัญ #1: MediaPipe ประมวลผลภาพดิบ (ไม่ flip ก่อน)
    # ==========================================
    rgb_l    = cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB)
    results_l = pose.process(rgb_l)

    results_r = None
    if ret_r:
        rgb_r    = cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB)
        results_r = pose.process(rgb_r)

    # วาด skeleton บนภาพก่อน flip (พิกัดตรงกัน)
    if results_l.pose_landmarks:
        mp_drawing.draw_landmarks(frame_l, results_l.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    if results_r and results_r.pose_landmarks:
        mp_drawing.draw_landmarks(frame_r, results_r.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # ==========================================
    # 6.1 หาตำแหน่ง TCP และควบคุม Gripper
    # ==========================================
    raw_target_pos      = None
    raw_target_rpy      = None
    raw_gripper_percent = None
    vis_score_l         = 0.0

    if results_l.pose_landmarks and results_l.pose_world_landmarks:
        lm   = results_l.pose_landmarks        # normalized (pixel)
        wlm  = results_l.pose_world_landmarks  # ✅ world (เมตร) ใช้คำนวณ RPY

        # --- มือซ้าย (Gesture ควบคุม) ---
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
                print(f"✋ GESTURE: Toggle Hold → {'ON' if is_holding else 'OFF'}")
                cv2.circle(frame_l, (wx_l, wy_l), 40, (0, 165, 255), -1)

            dist_lf = math.sqrt(
                (left_index.x - left_thumb.x)**2 +
                (left_index.y - left_thumb.y)**2 +
                (left_index.z - left_thumb.z)**2
            )
            is_fist = dist_lf < FIST_THRESHOLD
            if is_fist and (current_time - last_mark_time > MARK_COOLDOWN):
                marked_position = current_smoothed_pos.copy() if not is_first_frame else np.zeros(3)
                last_mark_time  = current_time
                print(f"📌 GESTURE: Mark Position → {marked_position}")
                cv2.circle(frame_l, (wx_l, wy_l), 40, (255, 255, 0), -1)

        # --- มือขวา (ควบคุม TCP) ---
        right_wrist  = lm.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        index_finger = lm.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
        thumb_finger = lm.landmark[mp_pose.PoseLandmark.RIGHT_THUMB]
        pinky_finger = lm.landmark[mp_pose.PoseLandmark.RIGHT_PINKY]

        # ✅ world landmarks สำหรับ RPY
        w_right_wrist  = wlm.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        w_index_finger = wlm.landmark[mp_pose.PoseLandmark.RIGHT_INDEX]
        w_pinky_finger = wlm.landmark[mp_pose.PoseLandmark.RIGHT_PINKY]

        vis_score_l = (index_finger.visibility + thumb_finger.visibility) / 2

        if vis_score_l > 0.1:
            ix_l, iy_l = lm_px(index_finger, w, h)
            tx_l, ty_l = lm_px(thumb_finger, w, h)
            tcp_x_l    = (ix_l + tx_l) // 2
            tcp_y_l    = (iy_l + ty_l) // 2

            # ✅ แก้จุดสำคัญ #4: RPY จาก world_landmarks + สูตร yaw ที่ถูกต้อง
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

                # ✅ แก้จุดสำคัญ #1: ภาพ rectified แล้ว ไม่ต้อง un-mirror ซ้ำ
                # ส่งพิกัดตรงๆ เข้า triangulate
                pt_3d       = get_3d_point([tcp_x_l, tcp_y_l], [tcp_x_r, tcp_y_r], P1, P2)
                raw_target_pos = pt_3d

                # Smart Failover: ถ้ากล้องซ้ายโดนบัง ใช้ข้อมูลกล้องขวาแทน
                if vis_score_l < 0.3 and vis_score_r > 0.5:
                    raw_target_rpy = calculate_rpy(w_wrist_r, w_idx_r, w_pnk_r)
                    # ✅ แก้จุดสำคัญ #5: Gripper จากระยะ 3D จริง
                    idx_3d = get_3d_point([ix_r, iy_r], [ix_l, iy_l], P1, P2)
                    thb_3d = get_3d_point([tx_r, ty_r], [tx_l, ty_l], P1, P2)
                    dist_mm = np.linalg.norm(idx_3d - thb_3d) * 1000
                else:
                    idx_3d  = get_3d_point([ix_l, iy_l], [ix_r, iy_r], P1, P2)
                    thb_3d  = get_3d_point([tx_l, ty_l], [tx_r, ty_r], P1, P2)
                    dist_mm = np.linalg.norm(idx_3d - thb_3d) * 1000

                raw_gripper_percent = float(np.clip((dist_mm - 20) / (120 - 20) * 100, 0, 100))

                # วาดจุดเป้าหมายกล้องขวา
                cv2.circle(frame_r, (tcp_x_r, tcp_y_r), 10, (0, 0, 255), -1)

            else:
                # โหมดกล้องเดี่ยว (2D)
                raw_target_pos = np.array([
                    float(tcp_x_l),
                    float(tcp_y_l),
                    current_smoothed_pos[2] if not is_first_frame else 0.0
                ])
                dist_norm = math.sqrt(
                    (index_finger.x - thumb_finger.x)**2 +
                    (index_finger.y - thumb_finger.y)**2 +
                    (index_finger.z - thumb_finger.z)**2
                )
                raw_gripper_percent = float(np.clip((dist_norm - 0.02) / (0.12 - 0.02) * 100, 0, 100))

            # วาดเป้าหมายกล้องซ้าย
            target_color = (0, 0, 255) if vis_score_l > 0.4 else (0, 165, 255)
            cv2.circle(frame_l, (tcp_x_l, tcp_y_l), 10, target_color, -1)
            cv2.line(frame_l,
                     lm_px(index_finger, w, h),
                     lm_px(thumb_finger, w, h),
                     target_color, 2)

    # ==========================================
    # 6.2 EMA Filter + Deadzone + Reset timeout
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
                current_smoothed_pos = (EMA_ALPHA_POS * raw_target_pos +
                                        (1.0 - EMA_ALPHA_POS) * current_smoothed_pos)
            if delta_rpy > DEADZONE_RPY:
                current_smoothed_rpy = (EMA_ALPHA_RPY * raw_target_rpy +
                                        (1.0 - EMA_ALPHA_RPY) * current_smoothed_rpy)
            current_smoothed_gripper = (EMA_ALPHA_GRP * raw_gripper_percent +
                                        (1.0 - EMA_ALPHA_GRP) * current_smoothed_gripper)
    else:
        # ✅ แก้จุดสำคัญ #6: Reset EMA ถ้า vision หายนาน
        if time.time() - last_valid_time > RESET_TIMEOUT:
            is_first_frame = True

    # ==========================================
    # 7. วาด UI — flip ภาพ content ก่อน แล้ววาง text ทีหลัง (text จะตรงเสมอ)
    # ==========================================
    # flip frame_l ก่อน เพื่อให้ skeleton ตรงกับ mirror
    disp_l = cv2.flip(frame_l, 1)

    # วาด semi-transparent overlay
    overlay_buf = disp_l.copy()
    bottom_bar_h = 80
    cv2.rectangle(overlay_buf, (10, 10), (430, 200), (0, 0, 0), -1)
    cv2.rectangle(overlay_buf, (0, h - bottom_bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay_buf, 0.6, disp_l, 0.4, 0, disp_l)

    # Status text (วาดบน flipped image → ตัวหนังสือตรงปกติ)
    if is_holding:
        status_text, s_color = "HOLDING (Manual)", (0, 165, 255)
    elif raw_target_pos is not None:
        if vis_score_l > 0.4:
            status_text, s_color = "Tracking 6-DOF (Clear)", (0, 255, 0)
        else:
            status_text, s_color = "Tracking (Occluded/Inferred)", (0, 255, 255)
    else:
        status_text, s_color = "Vision Lost!", (0, 0, 255)

    cv2.putText(disp_l, f"Status: {status_text}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, s_color, 2)
    cv2.putText(disp_l,
                f"P(X,Y,Z): {current_smoothed_pos[0]:.1f}, {current_smoothed_pos[1]:.1f}, {current_smoothed_pos[2]:.1f}",
                (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(disp_l,
                f"O(R,P,Y): {current_smoothed_rpy[0]:.1f}, {current_smoothed_rpy[1]:.1f}, {current_smoothed_rpy[2]:.1f}",
                (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(disp_l, f"Gripper: {current_smoothed_gripper:.0f}%",
                (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
    if marked_position is not None:
        cv2.putText(disp_l,
                    f"Marked: {marked_position[0]:.1f}, {marked_position[1]:.1f}, {marked_position[2]:.1f}",
                    (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Gesture guide
    yg = h - bottom_bar_h + 20
    cv2.putText(disp_l, "Gestures:", (20, yg + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(disp_l, "Raise Left Hand=HOLD",
                (160, yg + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    cv2.putText(disp_l, "Left Fist=MARK",
                (420, yg + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # ==========================================
    # 8. แสดงผล 4 จอ — disp_l flipped แล้ว, ภาพอื่น flip + วาง label หลัง resize
    # ==========================================
    TW, TH = 640, 480

    # จอ 1: RAW ซ้าย
    raw_l_disp = cv2.resize(cv2.flip(raw_l, 1), (TW, TH))
    cv2.putText(raw_l_disp, "1. RAW CAM 1 (Left)",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # จอ 2: RAW ขวา
    raw_r_disp = cv2.resize(cv2.flip(raw_r, 1), (TW, TH))
    cv2.putText(raw_r_disp, "2. RAW CAM 2 (Right)",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # จอ 3: CAM 1 Processed (disp_l flipped แล้ว, วาง label ตรงๆ)
    calib_l_disp = cv2.resize(disp_l, (TW, TH))
    cv2.putText(calib_l_disp, "3. CAM 1 (Processed)",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # จอ 4: CAM 2
    if ret_r and frame_r is not None:
        disp_r = frame_r.copy()
        if results_r and results_r.pose_landmarks:
            mp_drawing.draw_landmarks(disp_r, results_r.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        disp_r = cv2.flip(disp_r, 1)   # flip หลัง draw skeleton
        calib_r_disp = cv2.resize(disp_r, (TW, TH))
        label_r = "4. CAM 2 (Stereo)" if STEREO_MODE else "4. CAM 2 (No Stereo File)"
    else:
        calib_r_disp = np.zeros((TH, TW, 3), dtype=np.uint8)
        label_r = "4. CAM 2 (Not Connected)"
    cv2.putText(calib_r_disp, label_r,
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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
        print(f"⌨️  Hold toggled → {'ON' if is_holding else 'OFF'}")
    elif key in (ord('m'), ord('M')):
        marked_position = current_smoothed_pos.copy()
        print(f"⌨️  Mark → {marked_position}")

# ==========================================
# 10. ปิดกล้อง
# ==========================================
cam_left.release()
if cam_right is not None:
    cam_right.release()
cv2.destroyAllWindows()