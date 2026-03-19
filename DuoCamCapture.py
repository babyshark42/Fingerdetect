import cv2
import os
import cv2.aruco as aruco
import numpy as np

# ============================================================
# 0. ตั้งค่า ChArUco Board
# ============================================================
CHARUCO_COLS    = 5          # จำนวนช่องแนวนอน (squares)
CHARUCO_ROWS    = 7          # จำนวนช่องแนวตั้ง (squares)
SQUARE_LENGTH   = 0.04       # ขนาดช่องสี่เหลี่ยม (เมตร) — ปรับตามบอร์ดจริง
MARKER_LENGTH   = 0.03       # ขนาด ArUco marker (เมตร) — ปรับตามบอร์ดจริง
ARUCO_DICT_ID   = aruco.DICT_4X4_50

aruco_dict  = aruco.getPredefinedDictionary(ARUCO_DICT_ID)
charuco_board = aruco.CharucoBoard(
    (CHARUCO_COLS, CHARUCO_ROWS),
    SQUARE_LENGTH,
    MARKER_LENGTH,
    aruco_dict
)
detector_params = aruco.DetectorParameters()
charuco_params  = aruco.CharucoParameters()
charuco_detector = aruco.CharucoDetector(charuco_board, charuco_params, detector_params)


def detect_and_draw_charuco(frame):
    """
    ตรวจจับ ChArUco บน frame แล้ว overlay ผลลัพธ์ลงบนภาพสำเนา
    คืนค่า (annotated_frame, charuco_corners, charuco_ids, detected_ok)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    annotated = frame.copy()

    charuco_corners, charuco_ids, marker_corners, marker_ids = \
        charuco_detector.detectBoard(gray)

    detected_ok = (
        charuco_ids is not None and
        len(charuco_ids) >= 4          # ต้องเจอ corner อย่างน้อย 4 จุดจึงจะ valid
    )

    # --- วาด ArUco markers ทุกตัวที่เจอ ---
    if marker_ids is not None:
        aruco.drawDetectedMarkers(annotated, marker_corners, marker_ids)

    # --- วาด ChArUco corners + แสดง ID ---
    if charuco_ids is not None:
        aruco.drawDetectedCornersCharuco(annotated, charuco_corners, charuco_ids)

        # แสดง corner ID แต่ละตัว
        for i, corner in enumerate(charuco_corners):
            x, y = int(corner[0][0]), int(corner[0][1])
            cid  = int(charuco_ids[i][0])
            cv2.putText(
                annotated, str(cid), (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA
            )

    # --- แถบสถานะ detect ได้/ไม่ได้ (ขอบสีซ้าย-ขวา) ---
    status_color = (0, 255, 0) if detected_ok else (0, 0, 255)  # เขียว / แดง
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], annotated.shape[0]),
                  status_color, 4)

    # ข้อความสรุปจำนวน corners
    n_found = len(charuco_ids) if charuco_ids is not None else 0
    n_total = (CHARUCO_COLS - 1) * (CHARUCO_ROWS - 1)
    status_text = f"ChArUco: {n_found}/{n_total}" if detected_ok else f"ChArUco: NOT DETECTED ({n_found}/{n_total})"
    cv2.putText(
        annotated, status_text, (10, annotated.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA
    )

    return annotated, charuco_corners, charuco_ids, detected_ok


# ============================================================
# 1. ตั้งค่าโฟลเดอร์สำหรับเก็บรูป
# ============================================================
output_dir_left  = "calibration_data/left"
output_dir_right = "calibration_data/right"
os.makedirs(output_dir_left,  exist_ok=True)
os.makedirs(output_dir_right, exist_ok=True)

print("กำลังเชื่อมต่อกล้อง...")
cam_left  = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cam_right = cv2.VideoCapture(2, cv2.CAP_DSHOW)

for cam in (cam_left, cam_right):
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cam_left.isOpened() or not cam_right.isOpened():
    print("❌ ไม่สามารถเปิดกล้องได้ กรุณาตรวจสอบการเชื่อมต่อ USB หรือเปลี่ยนเลข ID กล้อง")
    exit()

print("✅ เชื่อมต่อกล้องสำเร็จ!")
print("=" * 50)
print("วิธีใช้งาน:")
print("  [Spacebar]  ถ่ายรูปคู่ (บันทึกเฉพาะเมื่อ detect สำเร็จทั้ง 2 กล้อง)")
print("  [Q / ESC]   ออกจากโปรแกรม")
print("=" * 50)

img_count = 0

while True:
    ret_l, frame_l = cam_left.read()
    ret_r, frame_r = cam_right.read()

    if not ret_l or not ret_r:
        print("เกิดข้อผิดพลาดในการดึงภาพจากกล้อง")
        break

    # ตรวจจับ ChArUco ทั้ง 2 ฝั่ง
    ann_l, ch_corners_l, ch_ids_l, ok_l = detect_and_draw_charuco(frame_l)
    ann_r, ch_corners_r, ch_ids_r, ok_r = detect_and_draw_charuco(frame_r)

    # รวมภาพแสดงผล
    combined = cv2.hconcat([ann_l, ann_r])

    # แถบ HUD บนสุด
    both_ok = ok_l and ok_r
    hud_color  = (0, 255, 0) if both_ok else (0, 165, 255)
    hud_text   = f"Captured: {img_count}   |   {'✔ BOTH OK — press Space to save' if both_ok else '✖ Waiting for valid detection...'}"
    cv2.putText(combined, hud_text, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, hud_color, 2, cv2.LINE_AA)

    cv2.imshow("Stereo ChArUco Calibration (Left | Right)", combined)

    key = cv2.waitKey(1) & 0xFF

    if key in (ord('q'), 27):
        print("ปิดโปรแกรม...")
        break

    elif key == 32:  # Spacebar
        if both_ok:
            filename = f"img_{img_count:03d}.png"
            cv2.imwrite(os.path.join(output_dir_left,  filename), frame_l)
            cv2.imwrite(os.path.join(output_dir_right, filename), frame_r)
            print(f"📸 บันทึกรูปคู่ที่ {img_count:03d} สำเร็จ "
                  f"(L:{len(ch_ids_l)} corners, R:{len(ch_ids_r)} corners)")
            img_count += 1
        else:
            sides = []
            if not ok_l: sides.append("ซ้าย")
            if not ok_r: sides.append("ขวา")
            print(f"⚠️  detect ไม่สำเร็จ ({', '.join(sides)}) — ไม่บันทึก")

cam_left.release()
cam_right.release()
cv2.destroyAllWindows()