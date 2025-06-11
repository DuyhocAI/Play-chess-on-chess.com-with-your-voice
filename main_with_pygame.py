import sys
import os
import threading
import time

# Thêm đường dẫn chứa Board_Detect.py vào sys.path
board_detect_dir = r"D:\Bao_Duy\Autochess"
if board_detect_dir not in sys.path:
    sys.path.append(board_detect_dir)

# Giờ có thể import các hàm từ Board_Detect.py
from Board_Detect import (
    capture_chrome,
    detect_board_region,
    make_grid_with_orientation,
    annotate_board
)

import cv2
import numpy as np
import win32gui, win32ui, win32con
import pygetwindow as gw
import pygame
import pyautogui

CHROME_TITLE_KEYWORD = "chess.com"
CHROME_WND_TITLE     = "Google Chrome"
FRAME_UPDATE_RATE    = 30  # FPS cho pygame window

# Hàm tìm Chrome hwnd 
def find_chess_chrome():
    """
    Trả về hwnd của cửa sổ Chrome đang mở chess.com, hoặc None nếu không tìm thấy.
    """
    for w in gw.getAllWindows():
        if CHROME_WND_TITLE in w.title and CHROME_TITLE_KEYWORD in w.title.lower():
            return w._hWnd
    return None

# Hàm chuyển OpenCV image -> Pygame surface 
def cv2_to_pygame(frame_bgr):
    """
    Chuyển một ảnh BGR (OpenCV) thành Pygame Surface.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = frame_rgb.shape
    return pygame.image.frombuffer(frame_rgb.tobytes(), (w, h), "RGB")

# Luồng detection
class BoardDetector(threading.Thread):
    """
    Chạy trong thread riêng để capture và detect bàn cờ liên tục.
    Kết quả: self.annotated_image chứa ảnh đã vẽ lưới (OpenCV BGR).
    Lưu self.squares để mapping tọa độ.
    """
    def __init__(self, hwnd):
        super().__init__()
        self.hwnd = hwnd
        self.running = True
        self.latest_frame = None         # Ảnh gốc BGR
        self.annotated_image = None      # Ảnh sau khi annotate BGR
        self.flipped = False
        self.squares = []                # Danh sách squares [(sx1,sy1,sx2,sy2,name), ...]
        self.bbox = None                 # Bounding box bàn cờ

    def run(self):
        while self.running:
            try:
                frame = capture_chrome(self.hwnd)
                self.latest_frame = frame

                bbox = detect_board_region(frame)
                self.bbox = bbox
                if bbox is not None:
                    squares, flipped = make_grid_with_orientation(frame, bbox)
                    self.squares = squares
                    self.flipped = flipped
                    annotated = annotate_board(frame.copy(), squares, flipped)
                    self.annotated_image = annotated
                else:
                    # Nếu không detect được, chỉ show ảnh gốc
                    self.annotated_image = frame.copy()

            except Exception as e:
                # Trong trường hợp lỗi, giữ khung trước đó
                print(f"[Detector] Lỗi khi capture/detect: {e}")
            time.sleep(1.0 / FRAME_UPDATE_RATE)

    def stop(self):
        self.running = False

# Chức năng move trên Chrome 
def move_on_chrome(hwnd, squares, src, dst):
    """
    Thực hiện click chuyển quân: click vào center của ô src, rồi ô dst.
    squares: list (sx1,sy1,sx2,sy2,name)
    src, dst: string như "e2", "e4"
    """
    # Lấy vị trí client (góc trên-trái của vùng content) trên màn hình
    client_left, client_top = win32gui.ClientToScreen(hwnd, (0, 0))

    # Tìm tọa độ center từng ô trong danh sách squares
    coord_map = {name.lower(): (sx1, sy1, sx2, sy2) for (sx1, sy1, sx2, sy2, name) in squares}

    src = src.lower()
    dst = dst.lower()
    if src not in coord_map or dst not in coord_map:
        print(f"[Move] Ô không hợp lệ: {src} hoặc {dst}")
        return

    sx1, sy1, sx2, sy2 = coord_map[src]
    dx1, dy1, dx2, dy2 = coord_map[dst]

    # Tính center trong client coordinates
    src_cx = (sx1 + sx2) // 2
    src_cy = (sy1 + sy2) // 2
    dst_cx = (dx1 + dx2) // 2
    dst_cy = (dy1 + dy2) // 2

    # Chuyển sang screen coordinates
    screen_src_x = client_left + src_cx
    screen_src_y = client_top + src_cy
    screen_dst_x = client_left + dst_cx
    screen_dst_y = client_top + dst_cy

    # Click source, rồi click destination
    pyautogui.moveTo(screen_src_x, screen_src_y)
    pyautogui.click()
    time.sleep(0.1)
    pyautogui.moveTo(screen_dst_x, screen_dst_y)
    pyautogui.click()
    print(f"[Move] Đã di chuyển từ {src.upper()} đến {dst.upper()}")

# ===== Main =====
def main():
    # 1. Tìm cửa sổ Chrome chứa chess.com
    hwnd = find_chess_chrome()
    if hwnd is None:
        print("Không tìm thấy cửa sổ Chrome chứa chess.com. Vui lòng mở chess.com trong Chrome rồi chạy lại.")
        return

    # 2. Khởi tạo luồng detector
    detector = BoardDetector(hwnd)
    detector.start()

    # 3. Khởi tạo Pygame
    pygame.init()
    pygame.display.set_caption("Chessboard Detector & Move Command")

    # Đợi lần đầu detector có ảnh đầu tiên
    print("Đang chờ luồng detector chạy...")
    while detector.annotated_image is None:
        time.sleep(0.1)

    # Lấy kích thước ảnh để tạo cửa sổ Pygame
    init_frame = detector.annotated_image
    h, w = init_frame.shape[:2]
    screen = pygame.display.set_mode((w, h + 80))  
    # +80 px để dành chỗ nhập text và hướng dẫn

    font = pygame.font.SysFont("Arial", 20)
    input_font = pygame.font.SysFont("Arial", 24)
    clock = pygame.time.Clock()

    # 4. Vòng lặp chính Pygame
    running = True
    user_message = "Nhập lệnh (ví dụ \"e2 e4\"), ENTER để gửi. Q để thoát."
    input_text = ""
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]

                elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                    # Khi nhấn Enter, parse input_text
                    parts = input_text.strip().split()
                    if len(parts) == 2:
                        src, dst = parts
                        # Thực hiện move trên Chrome
                        if detector.squares:  # Đảm bảo đã detect
                            move_on_chrome(hwnd, detector.squares, src, dst)
                        else:
                            print("[Move] Chưa detect được bàn cờ.")
                    else:
                        print("[Move] Lệnh không đúng định dạng. Ví dụ: e2 e4")
                    input_text = ""

                else:
                    # Chỉ nhận ký tự chữ/ số và space
                    char = event.unicode
                    if char.isalnum() or char == ' ':
                        input_text += char

        # 5. Lấy ảnh annotate mới nhất
        annotated = detector.annotated_image
        if annotated is not None:
            surface = cv2_to_pygame(annotated)
            screen.blit(surface, (0, 0))

        # 6. Vẽ khung nhập lệnh ở dưới
        #    Một khung background tối + văn bản hướng dẫn + input hiện tại
        pygame.draw.rect(screen, (40, 40, 40), (0, h, w, 80))
        # Hướng dẫn
        guide_surf = font.render(user_message, True, (230, 230, 230))
        screen.blit(guide_surf, (10, h + 5))
        # Nội dung input
        input_surf = input_font.render(input_text, True, (200, 200, 0))
        screen.blit(input_surf, (10, h + 40))

        pygame.display.flip()
        clock.tick(FRAME_UPDATE_RATE)

    # Khi thoát, dừng detector và Pygame
    detector.stop()
    detector.join()
    pygame.quit()

if __name__ == "__main__":
    main()
