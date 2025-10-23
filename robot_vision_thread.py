# ファイル名: robot_vision_threaded.py
# (main_control_threaded.py と同じフォルダに保存してください)

import cv2
import numpy as np
import math
import time
# threading をインポート (multiprocessing は不要)

# --- パラメータ (軽量化のため調整) ---
RESIZE_WIDTH = 240
MIN_NOISE_AREA = 80 

# ===================================================================
# スレッド1: 操舵用 (line_detect ベース)
# ===================================================================
def steering_thread_func(camera_index, shared_state, lock):
    """
    【スレッド版】
    操舵（消失点）を検出し、ズレ量(float)を共有辞書に書き込み続ける関数。
    """
    
    # --- 操舵用パラメータ ---
    CANNY_THRESHOLD1 = 100
    CANNY_THRESHOLD2 = 150
    HOUGH_THRESHOLD = 35 
    HOUGH_MIN_LINE_LENGTH = 35
    HOUGH_MAX_LINE_GAP = 10
    CLIP_LIMIT = 15.0
    TILE_GRID_SIZE = (4, 4)

    print(f"[操舵スレッド]: カメラ({camera_index})の起動を試みます...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[操舵スレッド] エラー: カメラ ({camera_index}) を開けません。")
        with lock:
            shared_state['stop'] = True # エラーならメインスレッドに停止を通知
        return
    print(f"[操舵スレッド]: カメラ({camera_index}) 起動完了。")

    # --- メインループ ---
    # shared_state['stop'] をチェックしてループを制御
    while True:
        with lock:
            if shared_state['stop']:
                break
                
        ret, frame = cap.read()
        if not ret:
            print("[操舵スレッド] エラー: フレームを取得できません。")
            time.sleep(0.5)
            continue
            
        # --- 1. リサイズ ---
        try:
            orig_height, orig_width = frame.shape[:2]
            aspect_ratio = orig_height / orig_width
            resize_height = int(RESIZE_WIDTH * aspect_ratio)
            resized_frame = cv2.resize(frame, (RESIZE_WIDTH, resize_height), interpolation=cv2.INTER_AREA)
            height, width = resized_frame.shape[:2]
        except Exception as e:
            print(f"[操舵スレッド] リサイズエラー: {e}")
            time.sleep(0.5)
            continue

        # --- 2. 前処理 ---
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)
        adjusted = clahe.apply(gray)
        blurred_again = cv2.GaussianBlur(adjusted, (7, 7), 0)
        edges = cv2.Canny(blurred_again, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
        
        # --- 3. ノイズ除去 ---
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)
        cleaned_edges = np.zeros_like(edges)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > MIN_NOISE_AREA:
                cleaned_edges[labels == i] = 255
        
        # --- 4. ハフ変換 & 消失点計算 ---
        lines = cv2.HoughLinesP(cleaned_edges, 1, np.pi/180,
                                threshold=HOUGH_THRESHOLD,
                                minLineLength=HOUGH_MIN_LINE_LENGTH,
                                maxLineGap=HOUGH_MAX_LINE_GAP)
        
        diagonal_lines = []
        vp_x = width // 2
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle_rad = math.atan2(y2 - y1, x2 - x1)
                angle_deg = math.degrees(angle_rad)
                abs_angle_deg = abs(angle_deg)

                is_horizontal = (abs_angle_deg <= 10) or (abs_angle_deg >= 175)
                is_vertical = (50 <= abs_angle_deg <= 130)
                if is_horizontal or is_vertical or x1 == x2:
                    continue
                
                m = (y2 - y1) / (x2 - x1)
                c = y1 - m * x1
                diagonal_lines.append((m, c))

        intersection_points = []
        if len(diagonal_lines) >= 2:
            for i in range(len(diagonal_lines)):
                for j in range(i + 1, len(diagonal_lines)):
                    m1, c1 = diagonal_lines[i]
                    m2, c2 = diagonal_lines[j]
                    if abs(m1 - m2) < 1e-5 or m1*m2 >0 : continue
                    x = (c2 - c1) / (m1 - m2)
                    y = m1 * x + c1
                    if -width < x < width * 2 and -height < y < height * 2:
                        intersection_points.append((x, y))

        if intersection_points:
            x_coords = [p[0] for p in intersection_points]
            vp_x = int(np.median(x_coords))

        # --- 5. ズレ量を計算 ---
        image_center_x = width / 2
        x_difference = vp_x - image_center_x
        
        # --- 6. 共有辞書へ書き込み (ロックを使用) ---
        with lock:
            shared_state['steering_value'] = x_difference
        
        time.sleep(0.5) # 元のコードのスリープを維持

    cap.release()
    print("[操舵スレッド]: カメラを解放しました。")


# ===================================================================
# スレッド2: 壁検出用 (wall_line ベース)
# ===================================================================
def wall_thread_func(camera_index, shared_state, lock):
    """
    【スレッド版】
    壁（垂直線）を検出し、結果(0 or 1)を共有辞書に書き込み続ける関数。
    """

    # --- 壁検出用パラメータ ---
    CANNY_THRESHOLD1 = 90 
    CANNY_THRESHOLD2 = 150
    HOUGH_THRESHOLD = 30 
    HOUGH_MIN_LINE_LENGTH = 30
    HOUGH_MAX_LINE_GAP = 10
    CLIP_LIMIT = 15.0
    TILE_GRID_SIZE = (12, 12)

    print(f"[壁検出スレッド]: カメラ({camera_index})の起動を試みます...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[壁検出スレッド] エラー: カメラ ({camera_index}) を開けません。")
        with lock:
            shared_state['stop'] = True # エラーならメインスレッドに停止を通知
        return
    print(f"[壁検出スレッド]: カメラ({camera_index}) 起動完了。")

    # --- メインループ ---
    # shared_state['stop'] をチェックしてループを制御
    while True:
        with lock:
            if shared_state['stop']:
                break

        ret, frame = cap.read()
        if not ret:
            print("[壁検出スレッド] エラー: フレームを取得できません。")
            time.sleep(0.5)
            continue
            
        # --- 1. リサイズ ---
        try:
            orig_height, orig_width = frame.shape[:2]
            aspect_ratio = orig_height / orig_width
            resize_height = int(RESIZE_WIDTH * aspect_ratio)
            resized_frame = cv2.resize(frame, (RESIZE_WIDTH, resize_height), interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"[壁検出スレッド] リサイズエラー: {e}")
            time.sleep(0.5)
            continue

        # --- 2. 前処理 ---
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)
        adjusted = clahe.apply(gray)
        blurred_again = cv2.GaussianBlur(adjusted, (7, 7), 0)
        edges = cv2.Canny(blurred_again, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
        
        # --- 3. ノイズ除去 ---
        """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)
        cleaned_edges = np.zeros_like(edges)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > MIN_NOISE_AREA:
                cleaned_edges[labels == i] = 255
        """
        # --- 4. ハフ変換 ---
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                threshold=HOUGH_THRESHOLD,
                                minLineLength=HOUGH_MIN_LINE_LENGTH,
                                maxLineGap=HOUGH_MAX_LINE_GAP)
        
        local_wall_detected = 0 # 0: 未検出
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle_deg = math.degrees(math.atan2(y2 - y1, x2 - x1))
                abs_angle_deg = abs(angle_deg)
                
                is_vertical = (80 <= abs_angle_deg <= 100)
                if is_vertical:
                    local_wall_detected = 1 # 1: 検出
                    break 
        
        # --- 5. 共有辞書へ書き込み (ロックを使用) ---
        with lock:
            shared_state['wall_detected'] = local_wall_detected
            
        time.sleep(0.2) # 元のコードのスリープを維持

    cap.release()
    print("[壁検出スレッド]: カメラを解放しました。")