import os
import sys
import threading
import time
import math
import requests  # Thêm thư viện requests để gọi API

from stockfish import Stockfish
import cv2
import numpy as np
import win32gui
import pygetwindow as gw
import pygame
import pyautogui
from ultralytics import YOLO

# Add path to Board_Detect.py
BOARD_DETECT_DIR = r"D:\Bao_Duy\Autochess"
if BOARD_DETECT_DIR not in sys.path:
    sys.path.append(BOARD_DETECT_DIR)

from Board_Detect import (
    capture_chrome,
    detect_board_region,
    make_grid_with_orientation,
    annotate_board
)

# ----- CONFIGURATION -----
CHROME_TITLE_KEYWORD = "chess.com"
CHROME_WND_TITLE     = "Google Chrome"
FRAME_RATE           = 30
YOLO_WEIGHTS         = r"D:\Bao_Duy\Autochess\runs_final_8\train\chess_enhanced_20250710_1522\weights\last.pt"
CONF_THRESH          = 0.25
IGNORE_CLASS         = "board"
STOCKFISH_PATH       = r"D:\Bao_Duy\Autochess\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
EMPTY_SQUARE_THRESHOLD = 15  # Color difference threshold for empty squares

PIECE_TYPES = {
    'p': 'pawn',
    'r': 'rook',
    'n': 'knight',
    'b': 'bishop',
    'q': 'queen',
    'k': 'king'
}

PIECE_TO_FEN = {
    'white_pawn': 'P', 'black_pawn': 'p',
    'white_rook': 'R', 'black_rook': 'r',
    'white_knight': 'N', 'black_knight': 'n',
    'white_bishop': 'B', 'black_bishop': 'b',
    'white_queen': 'Q', 'black_queen': 'q',
    'white_king': 'K', 'black_king': 'k'
}

def find_chess_chrome():
    for w in gw.getAllWindows():
        if CHROME_WND_TITLE in w.title and CHROME_TITLE_KEYWORD in w.title.lower():
            return w._hWnd
    return None

def cv2_to_pygame(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = frame_rgb.shape
    return pygame.image.frombuffer(frame_rgb.tobytes(), (w, h), "RGB")

def convert_to_black_and_white(frame):
    """Chuyển đổi hình ảnh sang đen trắng."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Chuyển sang xám
    _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)  # Ngưỡng để chuyển sang nhị phân
    return bw

def convert_bw_to_rgb(bw_image):
    """Chuyển đổi hình ảnh đen trắng thành hình ảnh RGB."""
    return cv2.cvtColor(bw_image, cv2.COLOR_GRAY2BGR)

def validate_chess_square(square):
    if len(square) != 2:
        return False
    col, row = square[0].lower(), square[1]
    return col in 'abcdefgh' and row in '12345678'

def choose_player_color():
    print("\n" + "="*50)
    print("🎯 CHỌN MÀU QUÂN CỜ")
    print("="*50)
    while True:
        choice = input("Bạn chơi quân gì? (1=trắng/white, 2=đen/black): ").strip()
        if choice in ['1', 'white', 'trắng']:
            print("Bạn chọn quân TRẮNG")
            return 'white'
        elif choice in ['2', 'black', 'đen']:
            print("Bạn chọn quân ĐEN")
            return 'black'
        else:
            print("Vui lòng nhập '1' (trắng) hoặc '2' (đen)")

def is_square_empty(square_region, grid_size=10):
    """Check if square is empty by comparing colors of the center with the upper right corner."""
    h, w = square_region.shape[:2]
    cell_size = h // grid_size
    
    # Center region (middle of the square)
    center = square_region[(grid_size//2-1)*cell_size:(grid_size//2)*cell_size,
                          (grid_size//2-1)*cell_size:(grid_size//2)*cell_size]
    
    # Upper right corner region (column 8, row 2 in 10x10 grid)
    corner = square_region[1*cell_size:2*cell_size, 
                          7*cell_size:8*cell_size]
    
    # Calculate average color for both regions
    center_avg = np.mean(center, axis=(0,1))
    corner_avg = np.mean(corner, axis=(0,1))
    
    # Calculate color difference
    color_diff = np.linalg.norm(center_avg - corner_avg)
    
    return color_diff < EMPTY_SQUARE_THRESHOLD

def compare_queens(new_piece, existing_piece):
    """So sánh hai quân Queen."""
    return new_piece == existing_piece

class BoardDetector(threading.Thread):
    def __init__(self, hwnd):
        super().__init__()
        self.hwnd = hwnd
        self.running = True
        self.annotated_image = None
        self.squares = []
        self.flipped = False
        self.model = YOLO(YOLO_WEIGHTS)
        self.current_board_state = {}
        self.board_updated = False
        self.lock = threading.Lock()

    def run(self):
        while self.running:
            frame = capture_chrome(self.hwnd)
            bw_frame = convert_to_black_and_white(frame)  # Chuyển đổi sang đen trắng
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Định nghĩa biến gray ở đây
            bbox = detect_board_region(frame)

            if not bbox:
                self.annotated_image = frame.copy()
                time.sleep(1/FRAME_RATE)
                continue

            x1, y1, x2, y2 = bbox
            crop = bw_frame[y1:y2, x1:x2]  # Sử dụng hình ảnh đen trắng
            crop_rgb = convert_bw_to_rgb(crop)  # Chuyển đổi sang RGB để đưa vào mô hình
            grid, self.flipped = make_grid_with_orientation(frame, bbox)
            annotated = annotate_board(frame.copy(), grid, self.flipped)

            new_state = {}
            
            for sx1, sy1, sx2, sy2, sq in grid:
                square_region = bw_frame[sy1:sy2, sx1:sx2]  # Sử dụng hình ảnh đen trắng
                
                # Kiểm tra xem ô có trống không
                if is_square_empty(square_region):
                    new_state[sq.lower()] = '.'  # Đánh dấu ô trống
                    cv2.putText(annotated, f"{sq}: empty", (sx1+4, sy1+16),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1, cv2.LINE_AA)
                    continue
                
                # Nếu không trống thì dùng YOLO để nhận diện quân cờ
                det = self.model(crop_rgb, conf=CONF_THRESH, imgsz=640)[0]  # Sử dụng hình ảnh RGB
                for box in det.boxes:
                    cls_id = int(box.cls[0])
                    lbl = det.names[cls_id]
                    if lbl == IGNORE_CLASS:
                        continue

                    bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                    fx1, fy1 = bx1 + x1, by1 + y1
                    fx2, fy2 = bx2 + x1, by2 + y1
                    cx, cy = (fx1+fx2)//2, (fy1+fy2)//2

                    if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
                        w_cell, h_cell = sx2-sx1, sy2-sy1
                        sub_w = max(1, int(w_cell/math.sqrt(8)))
                        sub_h = max(1, int(h_cell/math.sqrt(8)))
                        center_x, center_y = (sx1+sx2)//2, (sy1+sy2)//2
                        rx1 = max(sx1, center_x-sub_w//2)
                        ry1 = max(sy1, center_y-sub_h//2)
                        rx2 = min(sx2, rx1+sub_w)
                        ry2 = min(sy2, ry1+sub_h)
                        mean_int = float(np.mean(gray[ry1:ry2, rx1:rx2])) if gray.size else 0
                        color = 'white' if mean_int > 128 else 'black'

                        piece_name = f"{color}_{PIECE_TYPES[lbl.lower()]}"
                        
                        # Kiểm tra nếu quân cờ đã có trên ô
                        existing_piece_name = new_state.get(sq.lower(), None)
                        if existing_piece_name:
                            if not compare_queens(piece_name, existing_piece_name):
                                print(f"[Warning] Quân cờ không giống nhau: {existing_piece_name} vs {piece_name}")
                                continue  # Nếu không giống nhau, bỏ qua và không cập nhật

                        fen = PIECE_TO_FEN[piece_name]
                        new_state[sq.lower()] = fen

                        cv2.putText(annotated, f"{sq}: {piece_name}", (sx1+4, sy1+16),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1, cv2.LINE_AA)

            with self.lock:
                if new_state != self.current_board_state:
                    self.current_board_state = new_state
                    self.board_updated = True

            self.squares = grid
            self.annotated_image = annotated
            time.sleep(1/FRAME_RATE)

    def get_fen_from_detected_pieces(self, turn='w'):
        with self.lock:
            board = [['.' for _ in range(8)] for _ in range(8)]
            for sq, p in self.current_board_state.items():
                if len(sq) == 2:
                    col = ord(sq[0]) - ord('a')
                    row = 8 - int(sq[1])
                    if 0 <= col < 8 and 0 <= row < 8:
                        board[row][col] = p
            rows = []
            for r in board:
                count = 0
                fen_r = ''
                for c in r:
                    if c == '.':
                        count += 1
                    else:
                        if count > 0:
                            fen_r += str(count)
                            count = 0
                        fen_r += c
                if count > 0:
                    fen_r += str(count)
                rows.append(fen_r)
            board_fen = '/'.join(rows)
            return f"{board_fen} {turn} KQkq - 0 1"

    def is_board_updated(self):
        with self.lock:
            return self.board_updated

    def mark_board_processed(self):
        with self.lock:
            self.board_updated = False

    def get_board_info(self):
        with self.lock:
            return dict(self.current_board_state)

    def stop(self):
        self.running = False

def move_on_chrome(hwnd, squares, src, dst, callback=None):
    try:
        left, top = win32gui.ClientToScreen(hwnd, (0, 0))
        coord = {n.lower(): (x1, y1, x2, y2) for x1, y1, x2, y2, n in squares}
        s, d = src.lower(), dst.lower()
        if s not in coord or d not in coord:
            print(f"[Move] Không tìm thấy ô {src} hoặc {dst}")
            return False
        x1, y1, x2, y2 = coord[s]
        x1b, y1b, x2b, y2b = coord[d]
        sx, sy = (x1 + x2) // 2, (y1 + y2) // 2
        dx, dy = (x1b + x2b) // 2, (y1b + y2b) // 2
        px1, py1 = left + sx, top + sy
        px2, py2 = left + dx, top + dy
        pyautogui.click(px1, py1)
        time.sleep(0.2)
        pyautogui.click(px2, py2)
        time.sleep(0.5)
        if callback: callback()
        return True
    except Exception as e:
        print(f"[Move] Lỗi: {e}")
        return False

class StockfishManager:
    def __init__(self, path):
        try:
            self.stockfish = Stockfish(path=path, depth=12, parameters={"Threads": 2, "Hash": 512})
            print("[Stockfish] Engine initialized successfully")
            self.piece_positions = {}  # Dictionary to track piece positions {'e2': 'white_pawn', ...}
            self.move_history = []      # List to track move history ['e2e4', 'e7e5', ...]
            self._initialize_piece_positions()
        except Exception as e:
            print(f"[Stockfish] Error initializing: {e}")
            self.stockfish = None
        
        self.current_fen = None
        self.current_turn = 'w'
        self.last_evaluation = None
        self.all_moves_analysis = []
        self.is_valid = self.stockfish is not None

    def _initialize_piece_positions(self):
        """Initialize piece positions from standard chess starting position"""
        initial_position = [
            ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
            ['p']*8,
            ['.']*8,
            ['.']*8,
            ['.']*8,
            ['.']*8,
            ['P']*8,
            ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
        ]
        
        for row in range(8):
            for col in range(8):
                piece = initial_position[row][col]
                if piece != '.':
                    square = f"{chr(ord('a') + col)}{8 - row}"
                    color = 'black' if piece.islower() else 'white'
                    piece_type = PIECE_TYPES[piece.lower()]
                    self.piece_positions[square] = f"{color}_{piece_type}"

    def update_after_move(self, move_notation):
        """Update piece positions after a move"""
        if len(move_notation) != 4:
            print(f"[Error] Invalid move notation: {move_notation}")
            return False
            
        src = move_notation[:2].lower()
        dst = move_notation[2:].lower()
        
        # Check if there is a piece at the source position
        if src not in self.piece_positions:
            print(f"[Error] No piece found at {src}")
            return False
            
        # Move the piece
        moved_piece = self.piece_positions[src]
        
        # If the destination is empty, move the piece there
        if dst not in self.piece_positions:
            self.piece_positions[dst] = moved_piece
            del self.piece_positions[src]
        else:
            # If the destination has a piece, it means we are capturing
            print(f"[Warning] Destination {dst} is occupied. Overwriting the piece.")
            del self.piece_positions[dst]  # Remove the captured piece
            self.piece_positions[dst] = moved_piece
            del self.piece_positions[src]
        
        # Handle pawn promotion (simplified - assumes promotion to queen)
        if moved_piece.endswith('_pawn') and dst[1] in ('1', '8'):
            color = moved_piece.split('_')[0]
            self.piece_positions[dst] = f"{color}_queen"
            
        # Save the move in history
        self.move_history.append(move_notation)
        return True

    def reset_position(self):
        """Reset the board to the initial state (does not affect memory)"""
        try:
            self.stockfish.set_position([])
            self.move_history.clear()
            self.current_turn = 'w'
            self.last_evaluation = None
            self.all_moves_analysis = []
            # Do not reset piece_positions, keep the current state
            return True
        except Exception as e:
            print(f"[Stockfish] Error resetting: {e}")
            return False

    def update_memory(self, detected_positions):
        """Update memory based on detected positions from the camera"""
        # Compare detected positions with current piece positions
        for square, piece in detected_positions.items():
            if square not in self.piece_positions:
                # If the square is empty in memory but has a piece detected, update it
                self.piece_positions[square] = piece
            elif self.piece_positions[square] != piece:
                # If the piece has changed, update the memory
                self.piece_positions[square] = piece

    def print_current_position(self):
        """Print current board position in human-readable format"""
        board = [['.' for _ in range(8)] for _ in range(8)]
        
        for square, piece in self.piece_positions.items():
            col = ord(square[0]) - ord('a')
            row = 8 - int(square[1])
            if 0 <= row < 8 and 0 <= col < 8:
                piece_char = PIECE_TO_FEN[piece]
                board[row][col] = piece_char
        
        print("\nCurrent Board:")
        for row in board:
            print(' '.join(row))
        print()

    def set_turn(self, turn):
        self.current_turn = turn
        print(f"[Stockfish] Turn set to: {'White' if turn == 'w' else 'Black'}")

    def toggle_turn(self):
        self.current_turn = 'b' if self.current_turn == 'w' else 'w'
        return self.current_turn

    def update_position_from_detection(self, fen):
        if not self.is_valid:
            print("[Stockfish] Engine not available")
            return False
            
    def update_position_from_detection(self, fen):
        if not self.is_valid:
            print("[Stockfish] Engine not available")
            return False
            
        try:
            print(f"[Stockfish] Trying to set FEN: {fen}")
            
            # Check if FEN is valid before setting
            if not self._is_valid_fen(fen):
                print(f"[Stockfish] Invalid FEN format: {fen}")
                return False
                
            self.stockfish.set_fen_position(fen)
            self.current_fen = fen
            print(f"[Stockfish] Position updated successfully")
            return True
            
        except Exception as e:
            print(f"[Stockfish] Error updating position: {e}")
            # Try to reset and set to starting position
            try:
                print("[Stockfish] Attempting to reset to starting position...")
                self.stockfish.set_position([])
                self.current_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                return True
            except:
                return False

    def _is_valid_fen(self, fen):
        """Check if FEN is valid"""
        try:
            parts = fen.split()
            if len(parts) != 6:
                return False
            
            # Check board position
            board_part = parts[0]
            ranks = board_part.split('/')
            if len(ranks) != 8:
                return False
                
            for rank in ranks:
                count = 0
                for char in rank:
                    if char.isdigit():
                        count += int(char)
                    elif char in 'rnbqkpRNBQKP':
                        count += 1
                    else:
                        return False
                if count != 8:
                    return False
                    
            return True
        except:
            return False

    def _analyze_current_position(self):
        """Analyze current position and all possible moves"""
        if not self.is_valid:
            return
            
        try:
            print("[Stockfish] Analyzing current position...")
            
            # Get evaluation of current position
            self.last_evaluation = self.stockfish.get_evaluation()
            print(f"[Stockfish] Evaluation: {self.last_evaluation}")
            
            # Get all valid moves and analyze
            self.all_moves_analysis = self.stockfish.get_top_moves(7)  # Calculate the top 7 moves
            print(f"[Stockfish] Found {len(self.all_moves_analysis)} top moves")
            
            # Ensure there is always a best move
            if not self.all_moves_analysis:
                print("[Stockfish] No top moves found, trying get_best_move...")
                best_move = self.stockfish.get_best_move()
                if best_move:
                    print(f"[Stockfish] Best move found: {best_move}")
                    self.all_moves_analysis = [{'Move': best_move, 'Centipawn': 0, 'Mate': None}]
                else:
                    print("[Stockfish] No best move available")
                    
        except Exception as e:
            print(f"[Stockfish] Error analyzing position: {e}")
            self.all_moves_analysis = []

    def get_best_move_advanced(self, time_ms=2000):
        """Get best move with detailed analysis"""
        if not self.is_valid:
            print("[Stockfish] Engine not available")
            return None, None, []
            
        try:
            print(f"[Stockfish] Getting best move with {time_ms}ms thinking time...")
            
            # Analyze current position
            self._analyze_current_position()
            
            # Try multiple ways to get best move
            best_move = None
            
            # Method 1: Get from analyzed top moves
            if self.all_moves_analysis:
                best_move = self.all_moves_analysis[0]['Move']
                print(f"[Stockfish] Best move from analysis: {best_move}")
            
            # Method 2: Get best move directly
            if not best_move:
                best_move = self.stockfish.get_best_move()
                print(f"[Stockfish] Best move direct: {best_move}")
            
            # Method 3: Get best move with time
            if not best_move:
                best_move = self.stockfish.get_best_move_time(time_ms)
                print(f"[Stockfish] Best move with time: {best_move}")
            
            return best_move, self.last_evaluation, self.all_moves_analysis[:5]  # Top 5 moves
            
        except Exception as e:
            print(f"[Stockfish] Error getting best move: {e}")
            return None, None, []

    def get_all_moves_analysis(self):
        """Return analysis of all possible moves"""
        return self.all_moves_analysis

    def reset_position(self):
        try:
            self.stockfish.set_position([])
            self.move_history.clear()
            self.current_turn = 'w'
            self.last_evaluation = None
            self.all_moves_analysis = []
            return True
        except Exception as e:
            print(f"[Stockfish] Error resetting: {e}")
            return False

def get_best_moves_from_chess_com(fen):
    """Gọi API Chess.com để lấy các nước đi tốt nhất cho FEN hiện tại"""
    url = "https://api.chess.com/pub/engine"
    params = {
        "fen": fen
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get('bestMoves', [])
    else:
        print(f"[Chess.com API] Error: {response.status_code}")
        return []

def main():
    player_color = choose_player_color()
    player_turn = 'w' if player_color == 'white' else 'b'

    if not os.path.isfile(STOCKFISH_PATH):
        print("Không tìm thấy Stockfish tại", STOCKFISH_PATH)
        return
    
    print("[System] Initializing Stockfish...")
    sfm = StockfishManager(STOCKFISH_PATH)
    
    if not sfm.is_valid:
        print("[System] ERROR: Could not initialize Stockfish engine!")
        return
    
    sfm.set_turn(player_turn)

    print("[System] Looking for Chrome with chess.com...")
    hwnd = find_chess_chrome()
    if not hwnd:
        print("Không tìm thấy Chrome mở chess.com")
        return

    print("[System] Starting board detector...")
    detector = BoardDetector(hwnd)
    detector.start()
    
    pygame.init()
    pygame.display.set_caption(f"Chess Assistant - {player_color.upper()}")
    
    print("[System] Waiting for board detection...")
    while detector.annotated_image is None:
        time.sleep(0.1)

    h, w = detector.annotated_image.shape[:2]
    screen = pygame.display.set_mode((w, h+200))
    font = pygame.font.SysFont("Arial", 14)
    clock = pygame.time.Clock()

    user_input = ""
    thinking = False
    last_analysis = "Press 's' then Enter to analyze position"
    move_status = "Ready - Enter your move or press 's' to analyze"
    current_hint = None
    top_moves = []
    show_all_moves = False

    def analyze_position():
        nonlocal thinking, last_analysis, current_hint, top_moves
        thinking = True
        print("[Analysis] Starting position analysis...")
        
        turn = sfm.current_turn
        fen = detector.get_fen_from_detected_pieces(turn)
        print(f"[Analysis] Generated FEN: {fen}")
        print(f"[Analysis] Current turn: {turn}")
        
        board_info = detector.get_board_info()
        if not board_info:
            print("[Analysis] No pieces detected on board")
            last_analysis = "No pieces detected - please ensure board is visible"
            thinking = False
            return
        
        print(f"[Analysis] Detected pieces: {len(board_info)}")
        
        if sfm.update_position_from_detection(fen):
            print("[Analysis] Position updated successfully, getting best move...")
            best, eval_, tops = sfm.get_best_move_advanced(2000)
            
            # Gọi API Chess.com để lấy các nước đi tốt nhất
            best_moves = get_best_moves_from_chess_com(fen)
            current_hint = best_moves[0] if best_moves else None
            top_moves = best_moves
            
            turn_txt = "WHITE" if turn=='w' else "BLACK"
            
            if best:
                # Format evaluation
                eval_text = ""
                if eval_:
                    if eval_['type'] == 'cp':
                        eval_text = f"{eval_['value']}cp"
                    elif eval_['type'] == 'mate':
                        eval_text = f"Mate in {eval_['value']}"
                
                # Format alternative moves
                alt_moves = []
                if tops and len(tops) > 1:
                    for move_data in tops[1:4]:  # Show next 3 best moves
                        move_str = move_data['Move']
                        if move_data.get('Mate'):
                            alt_moves.append(f"{move_str}(M{move_data['Mate']})")
                        else:
                            alt_moves.append(f"{move_str}({move_data.get('Centipawn', 0)}cp)")
                
                alt_text = f" | Alt: {', '.join(alt_moves)}" if alt_moves else ""
                last_analysis = f"{turn_txt} → {best} ({eval_text}){alt_text}"
            else:
                last_analysis = f"{turn_txt} → No valid moves found"
        else:
            last_analysis = "Error: Could not update Stockfish position"
        
        thinking = False

    def execute_move(inp):
        nonlocal move_status
        parts = inp.strip().split()
        if len(parts) != 2:
            move_status = "Error: enter 2 squares (e.g. e2 e4)"
            return
        src, dst = parts
        if not validate_chess_square(src) or not validate_chess_square(dst):
            move_status = "Error: invalid square"
            return
        
        if move_on_chrome(hwnd, detector.squares, src, dst):
            move_notation = f"{src}{dst}"
            if sfm.update_after_move(move_notation):
                move_status = f"✓ {src}->{dst}"
                # Cập nhật lại bộ nhớ từ camera 
                sfm.update_memory(detector.get_board_info())
            else:
                move_status = f"✗ {src}->{dst}"
        else:
            move_status = f"✗ {src}->{dst}"

    print("[System] Ready!")
    print("Commands: s=Analyze, z=Reset, x=Show All Moves, v=Change Turn, ESC=Quit")
    print("Type moves: e2 e4 (then Enter)")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_BACKSPACE:
                    user_input = user_input[:-1]
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    if user_input.strip() == "s":
                        if not thinking:
                            threading.Thread(target=analyze_position, daemon=True).start()
                    else:
                        execute_move(user_input)
                    user_input = ""
                else:
                    char = event.unicode
                    if char and (char.isalnum() or char == ' '):
                        # Handle special commands
                        if char.lower() == 'z' and not user_input:  # Reset
                            sfm.reset_position()
                            current_hint = None
                            move_status = "Stockfish reset"
                        elif char.lower() == 'x' and not user_input:  # Toggle all moves
                            show_all_moves = not show_all_moves
                            move_status = f"Show all moves: {'ON' if show_all_moves else 'OFF'}"
                        elif char.lower() == 'v' and not user_input:  # Change turn
                            new_turn = sfm.toggle_turn()
                            move_status = f"Turn: {'WHITE' if new_turn=='w' else 'BLACK'}"
                        else:
                            user_input += char

        # Draw board and overlay
        if detector.annotated_image is not None:
            screen.blit(cv2_to_pygame(detector.annotated_image), (0, 0))

        # Draw hint arrows for best moves if available
        if top_moves and detector.squares:
            for move in top_moves:
                src = move[:2]
                dst = move[2:4]
                centers = {square.lower(): ((x1+x2)//2, (y1+y2)//2) 
                          for x1, y1, x2, y2, square in detector.squares}
                if src in centers and dst in centers:
                    src_x, src_y = centers[src]
                    dst_x, dst_y = centers[dst]
                    arrow_color = (0, 255, 0)  # Màu xanh cho nước đi
                    pygame.draw.line(screen, arrow_color, (src_x, src_y), (dst_x, dst_y), 5)
                    pygame.draw.circle(screen, arrow_color, (dst_x, dst_y), 8)

        # Draw UI panel
        pygame.draw.rect(screen, (30, 30, 30), (0, h, w, 200))
        y_pos = h + 5

        # Status line
        status_color = (255, 255, 0) if thinking else (100, 255, 100)
        screen.blit(font.render(move_status, True, status_color), (10, y_pos))
        y_pos += 20

        # Analysis line
        if last_analysis:
            screen.blit(font.render(last_analysis, True, (200, 200, 100)), (10, y_pos))
            y_pos += 20

        # User input
        screen.blit(font.render(f"Your input: {user_input}", True, (200, 200, 200)), (10, y_pos))
        y_pos += 20

        # Show all moves if enabled
        if show_all_moves and sfm.all_moves_analysis:
            moves_text = "Top moves: "
            for i, move in enumerate(sfm.all_moves_analysis[:8]):
                move_str = move['Move']
                if move.get('Mate'):
                    moves_text += f"{move_str}(M{move['Mate']}) "
                else:
                    moves_text += f"{move_str}({move.get('Centipawn', 0)}cp) "
            screen.blit(font.render(moves_text, True, (150, 150, 200)), (10, y_pos))
            y_pos += 20

        # Instructions
        instructions = [
            "s=Analyze, z=Reset, x=Toggle Moves, v=Change Turn, ESC=Quit",
            "Enter moves like: e2 e4 then press Enter"
        ]
        for instruction in instructions:
            screen.blit(font.render(instruction, True, (180, 180, 180)), (10, y_pos))
            y_pos += 20

        pygame.display.flip()
        clock.tick(FRAME_RATE)

    # Clean up
    detector.stop()
    detector.join()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
