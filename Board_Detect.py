import os
import time
import ctypes
import cv2
import numpy as np
import win32gui, win32ui, win32con
import pygetwindow as gw

# ========== CẤU HÌNH ==========
CHROME_TITLE_KEYWORD = "chess.com"
CHROME_WND_TITLE     = "Google Chrome"
FRAME_WINDOW_NAME    = "Real-time Chessboard"

# ===== Hàm chụp ảnh của cửa sổ Chrome =====
PW_RENDERFULLCONTENT = 0x00000002

def capture_chrome(hwnd):
    """
    Chụp toàn bộ nội dung cửa sổ (bao gồm phần cuộn) của hwnd.
    Trả về một ảnh BGR (numpy array).
    """
    l, t, r, b = win32gui.GetClientRect(hwnd)
    w, h = r - l, b - t
    hdc = win32gui.GetWindowDC(hwnd)
    mfc = win32ui.CreateDCFromHandle(hdc)
    save_dc = mfc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(mfc, w, h)
    save_dc.SelectObject(bmp)
    ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), PW_RENDERFULLCONTENT)
    bits = bmp.GetBitmapBits(True)
    img = np.frombuffer(bits, np.uint8).reshape((h, w, 4))
    win32gui.DeleteObject(bmp.GetHandle())
    save_dc.DeleteDC(); mfc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hdc)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

# ===== Phát hiện vùng bàn cờ trên ảnh =====
def detect_board_region(frame):
    """
    Phát hiện vùng vuông gần tỉ lệ 1:1 lớn nhất sau khi tìm contours.
    Trả về bounding box (x1, y1, x2, y2) hoặc None nếu không tìm thấy.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_bbox = None
    max_area = 0

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        ar = w / float(h)
        area = w * h
        if 0.9 < ar < 1.1 and area > max_area and w > 100:
            best_bbox = (x, y, x + w, y + h)
            max_area = area

    return best_bbox

# ===== Tạo danh sách ô và kiểm tra xoay ngược bằng so sánh patch centered của A1 và H8 =====
def make_grid_with_orientation(frame, bbox):
    """
    Tạo danh sách các ô (x1, y1, x2, y2, square_name) từ bounding box,
    kiểm tra bàn cờ có bị xoay 180° hay không bằng cách so sánh
    độ đen (grayscale trung bình thấp hơn = tối hơn) của patch nhỏ (1/7 kích thước ô)
    nằm giữa ô A1 và H8, rồi gán tên ô tương ứng.
    """
    x1, y1, x2, y2 = bbox
    size = min(x2 - x1, y2 - y1)
    cell = size // 8

    # Tọa độ các đường lưới
    vs = [x1 + i * cell for i in range(9)]
    hs = [y1 + i * cell for i in range(9)]

    # Tạo list toạ độ từng ô
    coords = []
    for r in range(8):
        for c in range(8):
            sx1, sy1 = vs[c], hs[r]
            sx2, sy2 = vs[c + 1], hs[r + 1]
            coords.append((sx1, sy1, sx2, sy2))

    # Hàm tính giá trị grayscale trung bình của patch nhỏ centered trong ô
    def patch_gray_center(idx):
        sx1, sy1, sx2, sy2 = coords[idx]
        w = sx2 - sx1
        h = sy2 - sy1
        patch_size = cell // 7

        # Tọa độ trung tâm ô
        cx = (sx1 + sx2) // 2
        cy = (sy1 + sy2) // 2

        # Tọa độ patch centered
        px1 = cx - patch_size // 2
        py1 = cy - patch_size // 2
        px2 = px1 + patch_size
        py2 = py1 + patch_size

        # Giới hạn trong khung hình
        px1 = max(px1, 0); py1 = max(py1, 0)
        px2 = min(px2, frame.shape[1]); py2 = min(py2, frame.shape[0])

        roi = frame[py1:py2, px1:px2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray_roi))

    # Chỉ mục:
    # - H8: r=0, c=7 => index = 7
    # - A1: r=7, c=0 => index = 56
    idx_h8 = 7
    idx_a1 = 56

    gray_h8 = patch_gray_center(idx_h8)
    gray_a1 = patch_gray_center(idx_a1)

    # Trong chessboard chuẩn: H8 là dark (patch_gray_center thấp hơn), A1 là light
    # Nếu patch ở A1 tối hơn patch ở H8 (gray_a1 < gray_h8), tức A1 thực sự là dark => tức bàn cờ bị xoay
    # flipped = True khi A1 tối hơn H8
    flipped = gray_a1 < gray_h8

    squares = []
    for r in range(8):
        for c in range(8):
            sx1, sy1, sx2, sy2 = coords[r * 8 + c]
            if not flipped:
                file = chr(ord('a') + c)
                rank = 8 - r
            else:
                file = chr(ord('a') + (7 - c))
                rank = r + 1
            name = f"{file}{rank}"
            squares.append((sx1, sy1, sx2, sy2, name))

    return squares, flipped

# Vẽ lưới và tên ô lên ảnh
def annotate_board(frame, squares, flipped):
    """
    Vẽ bounding box và tên ô lên frame, ghi chú xoay nếu cần.
    """
    for sx1, sy1, sx2, sy2, name in squares:
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
        cv2.putText(frame, name, (sx1 + 2, sy2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    note = "Flipped" if flipped else "Standard"
    cv2.putText(frame, note, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame

# Xoá tất cả cửa sổ ô trước khi hiển thị lại 
def destroy_square_windows(squares):
    for _, _, _, _, name in squares:
        try:
            cv2.destroyWindow(name)
        except cv2.error:
            pass

# Main 
if __name__ == "__main__":
    # Tìm cửa sổ Chrome có chứa "chess.com"
    hwnd = None
    for w in gw.getAllWindows():
        if CHROME_WND_TITLE in w.title and CHROME_TITLE_KEYWORD in w.title.lower():
            hwnd = w._hWnd
            print(f"Found Chrome window: {w.title}")
            break

    if not hwnd:
        print("Không tìm thấy cửa sổ Chrome chứa chess.com.")
        exit(1)

    # Tạo cửa sổ chính hiển thị bàn cờ
    cv2.namedWindow(FRAME_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(FRAME_WINDOW_NAME, 800, 800)

    prev_squares = []  # lưu lại grid để đóng cửa sổ cũ khi thay đổi

    print("Bắt đầu realtime. Nhấn 'q' để thoát.")
    while True:
        frame = capture_chrome(hwnd)
        board_bbox = detect_board_region(frame)

        if board_bbox is not None:
            squares, flipped = make_grid_with_orientation(frame, board_bbox)

            # Nếu grid thay đổi, đóng các cửa sổ ô cũ
            if squares != prev_squares:
                destroy_square_windows(prev_squares)
                prev_squares = squares.copy()

            annotated = annotate_board(frame.copy(), squares, flipped)
            cv2.imshow(FRAME_WINDOW_NAME, annotated)
        else:
            cv2.imshow(FRAME_WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    destroy_square_windows(prev_squares)
    cv2.destroyWindow(FRAME_WINDOW_NAME)
    cv2.destroyAllWindows()
