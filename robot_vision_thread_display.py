# ファイル名: robot_vision_thread.py
# (main_control_thread_display.py と同じフォルダに保存してください)

import cv2
import numpy as np
import math
import time

# --- パラメータ (軽量化のため調整) ---
RESIZE_WIDTH = 240
MIN_NOISE_AREA = 80 

# ===================================================================
# スレッド1: 操舵用 (変更なし)
# ===================================================================
def steering_thread_func(camera_index, shared_state, lock):
    """
    【スレッド版】
    操舵（消失点）を検出し、ズレ量(float)と処理済みフレームを共有辞書に書き込む。
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
        
        # --- 5b. デバッグ描画 ---
        cv2.circle(resized_frame, (vp_x, height // 2), 10, (0, 0, 255), -1) 
        cv2.line(resized_frame, (width // 2, 0), (width // 2, height), (255, 0, 0), 1)

        # --- 6. 共有辞書へ書き込み (ロックを使用) ---
        with lock:
            shared_state['steering_value'] = x_difference
            shared_state['steering_frame'] = resized_frame.copy() 
        
        time.sleep(0.5) 

    cap.release()
    print("[操舵スレッド]: カメラを解放しました。")


# ===================================================================
# スレッド2: 壁検出用 ( ★★★ ここから大幅に変更 ★★★ )
# ===================================================================
def wall_thread_func(camera_index, shared_state, lock):
    """
    【スレッド版・360度カメラ対応】
    フレームを「右(上半分)」「左(下半分)」に分割し、
    *どちらか*で垂直線を検出したらフラグを立て、
    処理済みフレームを結合して共有辞書に書き込む。
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
            shared_state['stop'] = True 
        return
    print(f"[壁検出スレッド]: カメラ({camera_index}) 起動完了。")

    # --- メインループ ---
    while True:
        with lock:
            if shared_state['stop']:
                break

        ret, frame = cap.read()
        if not ret or frame is None:
            print("[壁検出スレッド] エラー: フレームを取得できません。")
            time.sleep(0.5)
            continue
        
        # --- ★ 1. フレームを上下（右と左）に分割 ★ ---
        try:
            orig_height, orig_width = frame.shape[:2]
            half_height = orig_height // 2
            
            # 上半分 = 右側
            frame_right = frame[0:half_height, :] 
            # 下半分 = 左側
            frame_left = frame[half_height:, :]
            
            fragments = [frame_right, frame_left]
            processed_frames = []
            local_wall_detected = 0 # 0: 未検出
        
        except Exception as e:
            print(f"[壁検出スレッド] フレーム分割エラー: {e}")
            time.sleep(0.5)
            continue
            
        # --- ★ 2. 分割した画像を個別に処理 ★ ---
        for img_fragment in fragments:
            
            fragment_detected = 0
            resized_frame = None # 初期化

            try:
                # --- 2a. リサイズ ---
                orig_h_frag, orig_w_frag = img_fragment.shape[:2]
                if orig_h_frag == 0 or orig_w_frag == 0:
                    raise ValueError("Fragment is empty")
                    
                aspect_ratio = orig_h_frag / orig_w_frag
                resize_height = int(RESIZE_WIDTH * aspect_ratio)
                resized_frame = cv2.resize(img_fragment, (RESIZE_WIDTH, resize_height), interpolation=cv2.INTER_AREA)

                # --- 2b. 前処理 ---
                gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)
                adjusted = clahe.apply(gray)
                blurred_again = cv2.GaussianBlur(adjusted, (7, 7), 0)
                edges = cv2.Canny(blurred_again, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
                
                # --- 2c. ノイズ除去 (元コードでコメントアウト) ---
                # ... 
                
                # --- 2d. ハフ変換 ---
                lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                                        threshold=HOUGH_THRESHOLD,
                                        minLineLength=HOUGH_MIN_LINE_LENGTH,
                                        maxLineGap=HOUGH_MAX_LINE_GAP)
                
                # --- 2e. 検出ロジック ---
                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        angle_deg = math.degrees(math.atan2(y2 - y1, x2 - x1))
                        abs_angle_deg = abs(angle_deg)
                        
                        is_vertical = (80 <= abs_angle_deg <= 100)
                        if is_vertical:
                            fragment_detected = 1 # 1: 検出
                            local_wall_detected = 1 # ★ どちらかで検出したらメインフラグを立てる
                            break 
            
            except Exception as e:
                print(f"[壁検出スレッド] Fragment処理エラー: {e}")
                # エラーが発生した場合、ダミーの黒画像を作成
                # (vstackが失敗するのを防ぐため)
                # 仮の高さを設定 (アスペクト比 4:3 を想定)
                dummy_height = int(RESIZE_WIDTH * (3/4)) 
                resized_frame = np.zeros((dummy_height, RESIZE_WIDTH, 3), dtype=np.uint8)
                cv2.putText(resized_frame, "ERROR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


            # --- 2f. 描画 ---
            if fragment_detected == 1:
                cv2.putText(resized_frame, "WALL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            processed_frames.append(resized_frame)

        # --- ★ 3. 処理済みフレームを結合して共有 ★ ---
        try:
            # 2つのフレームを縦に結合 (上: Right, 下: Left)
            combined_frame = np.vstack((processed_frames[0], processed_frames[1]))
            
            # --- 5. 共有辞書へ書き込み (ロックを使用) ---
            with lock:
                shared_state['wall_detected'] = local_wall_detected
                shared_state['wall_frame'] = combined_frame.copy() 
        
        except Exception as e:
            print(f"[壁検出スレッド] フレーム結合エラー: {e}")
            # vstackに失敗した場合 (サイズ不一致など)
            with lock:
                 # 検出結果だけは更新する
                shared_state['wall_detected'] = local_wall_detected
                # フレームは更新しない (あるいはエラー画像を送る)
                # shared_state['wall_frame'] = ... 

            
        time.sleep(0.2) # 元のコードのスリープを維持

    cap.release()
    print("[壁検出スレッド]: カメラを解放しました。")