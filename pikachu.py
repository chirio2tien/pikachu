import itertools
import pyautogui
import numpy as np
import cv2
import os
import time
import heapq
import re
from collections import deque
import itertools

# Tọa độ góc trên trái và dưới phải của bảng game (CẦN CHỈNH THEO MÀN HÌNH CỦA BẠN)
x1, y1, x2, y2 = 525, 100, 1365, 685
rows, cols = 9, 16  # Kích thước lưới game

def capture_screen():
    """Chụp màn hình khu vực chứa bảng game."""
    screenshot = pyautogui.screenshot(region=(x1, y1, x2 - x1, y2 - y1))
   
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    cv2.imwrite("screenshot.png",screenshot)
    print("Ảnh chụp màn hình đã lưu.")
    return screenshot

def extract_tiles(grid_img):
    h, w = grid_img.shape[:2]
    tile_h, tile_w = h / rows, w / cols  # Giữ giá trị float để tính chính xác hơn
    tiles = []
    
    os.makedirs("tiles", exist_ok=True)
    
    for i in range(rows):
        row = []
        for j in range(cols):
            x, y = int(j * tile_w), int(i * tile_h)
            tile = grid_img[y:int(y + tile_h), x:int(x + tile_w)]  # Cắt chính xác theo float

            cv2.imwrite(f"tiles/tile_{i}_{j}.png", tile)
            row.append(tile)
    
    print("✅ Đã trích xuất các ô hình với nền trong suốt!")
    return tiles

def load_tiles_from_folder(folder="tiles", rows=9, cols=16):
    """
    Load các tệp ảnh theo định dạng tile_i_j.png và sắp xếp thành ma trận.
    """
    matrix = [[None for _ in range(cols)] for _ in range(rows)]
    pattern = re.compile(r"tile_(\d+)_(\d+)\.png")
    
    if not os.path.exists(folder):
        print(f"❌ Thư mục '{folder}' không tồn tại!")
        return matrix

    files = sorted(os.listdir(folder))
    
    for filename in files:
        match = pattern.match(filename)
        if match:
            i = int(match.group(1))
            j = int(match.group(2))
            path = os.path.join(folder, filename)
            img = cv2.imread(path)
            if img is not None:
                matrix[i][j] = img
    return matrix

def get_tile_positions():
    """Tính toán vị trí trung tâm của từng ô để click."""
    tile_positions = []
    tile_w = (x2 - x1) / cols
    tile_h = (y2 - y1) / rows

    for i in range(rows):
        row = []
        for j in range(cols):
            center_x = x1 + j * tile_w + tile_w / 2
            center_y = y1 + i * tile_h + tile_h / 2
            row.append((center_x, center_y))
        tile_positions.append(row)

    return tile_positions

def click_tile(position):
    """Click vào một ô trên màn hình."""
    x, y = position
    pyautogui.moveTo(x, y, duration=0.1)
    pyautogui.click()
    time.sleep(0.1)

def load_tiles_from_folder(folder="tiles"):
    """Tải các ô hình từ thư mục."""
    matrix = [[None for _ in range(cols)] for _ in range(rows)]
    pattern = re.compile(r"tile_(\d+)_(\d+)\.png")
    
    files = sorted(os.listdir(folder))
    
    for filename in files:
        match = pattern.match(filename)
        if match:
            i, j = int(match.group(1)), int(match.group(2))
            path = os.path.join(folder, filename)
            img = cv2.imread(path)
            if img is not None:
                matrix[i][j] = img

    return matrix

def group_tiles(matrix, threshold=0.05):
    """Nhóm các ô hình giống nhau bằng so sánh XOR."""
    labels = [[0 for _ in range(cols)] for _ in range(rows)]
    reps = []
    rep_labels = []
    next_label = 1

    for i in range(rows):
        for j in range(cols):
            tile = matrix[i][j]
            if tile is None:
                continue
            gray_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            assigned = False

            for rep, label in zip(reps, rep_labels):
                diff = cv2.bitwise_xor(gray_tile, rep)
                diff_ratio = np.sum(diff) / (gray_tile.shape[0] * gray_tile.shape[1] * 255)

                if diff_ratio < threshold:
                    labels[i][j] = label
                    assigned = True
                    break

            if not assigned:
                labels[i][j] = next_label
                reps.append(gray_tile)
                rep_labels.append(next_label)
                next_label += 1

    return labels


def can_connect_bfs(x1, y1, x2, y2, matrix, max_turns=2):
    """
    Kiểm tra xem hai ô có thể kết nối với nhau với tối đa số lần rẽ cho phép không.
    """
    ROWS, COLS = len(matrix), len(matrix[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Lên, Xuống, Trái, Phải
    queue = deque([(x1, y1, -1, 0)])  # (x, y, hướng trước, số lần rẽ)
    visited = set()
    
    while queue:
        x, y, prev_dir, turns = queue.popleft()
        if (x, y) == (x2, y2):
            return True  # Tìm thấy đường đi hợp lệ
        
        if (x, y, prev_dir) in visited:
            continue
        visited.add((x, y, prev_dir))
        
        for d, (dx, dy) in enumerate(directions):
            nx, ny = x + dx, y + dy
            new_turns = turns + (1 if d != prev_dir and prev_dir != -1 else 0)
            
            if 0 <= nx < ROWS and 0 <= ny < COLS and new_turns <= max_turns:
                if (nx, ny) == (x2, y2) or matrix[nx][ny] == 0:
                    queue.append((nx, ny, d, new_turns))
    
    return False

def find_valid_pairs(matrix):
    """
    Tìm tất cả các cặp ô có thể kết nối hợp lệ.
    """
    ROWS, COLS = len(matrix), len(matrix[0])
    pairs = []
    positions = {}
    
    # Gom nhóm các ô có cùng giá trị
    for x in range(ROWS):
        for y in range(COLS):
            if matrix[x][y] != 0:
                positions.setdefault(matrix[x][y], []).append((x, y))
    
    # Duyệt từng nhóm và tìm cặp hợp lệ
    for value, cells in positions.items():
        for (x1, y1), (x2, y2) in itertools.combinations(cells, 2):
            if can_connect_bfs(x1, y1, x2, y2, matrix):
                pairs.append(((x1, y1), (x2, y2)))
    
    return pairs



def add_border(matrix):
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    
    # Tạo ma trận mới có viền 0
    new_matrix = [[0] * (cols + 2)]  # Hàng trên cùng toàn 0
    
    for row in matrix:
        new_matrix.append([0] + row + [0])  # Thêm 0 vào đầu và cuối mỗi hàng
    
    new_matrix.append([0] * (cols + 2))  # Hàng dưới cùng toàn 0
    
    return new_matrix
def apply_mask(matrix_a, matrix_b):
    """Thay đổi giá trị của matrix_a nếu matrix_b có giá trị 0 tại cùng vị trí"""
    rows = len(matrix_a)
    cols = len(matrix_a[0]) if rows > 0 else 0
    
    result = [[0 if matrix_b[i][j] == 0 else matrix_a[i][j] for j in range(cols)] for i in range(rows)]
    return result


def auto_play():
    #con full 1
    matrix = [[1 for _ in range(18)] for _ in range(11)]
    """Bot tự động chơi Pikachu."""
    print("⏳ Chuẩn bị chơi game trong 5 giây...")
    time.sleep(5)
    matrix = add_border(matrix)
    while True:
        img = capture_screen()
        extract_tiles(img)
        tiles = load_tiles_from_folder()
        labels = group_tiles(tiles)


        labels = add_border(labels)
        labels = apply_mask(labels, matrix)
        for row in labels:
            print(row)
        valid_moves = find_valid_pairs(labels)
        if not valid_moves:
                print("game win")
                break
        while valid_moves:  
            
            positions = get_tile_positions()

            for move in valid_moves:
                pos1, pos2 = move
                if(matrix[pos1[0]][pos1[1]] == 0):
                    continue
                if(matrix[pos2[0]][pos2[1]] == 0):
                    continue
                matrix[pos1[0]][pos1[1]] = 0
                matrix[pos2[0]][pos2[1]] = 0
                click_tile(positions[pos1[0]-1][pos1[1]-1])
                click_tile(positions[pos2[0]-1][pos2[1]-1])
                time.sleep(0.1)
            
            for row in labels:
                print(row)
            labels = apply_mask(labels, matrix)
            valid_moves = find_valid_pairs(labels)
        
        coordinates = [ (1178, 239),(1549, 965)]
        for coord in coordinates:
            pyautogui.moveTo(coord[0], coord[1], duration=0.5)
            pyautogui.click()
        time.sleep(5)
        


auto_play()