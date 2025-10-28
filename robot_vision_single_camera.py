# ファイル名: robot_vision_single_camera.py
# (main_control_single_camera.py と同じフォルダに保存してください)

import cv2
import numpy as np
import math
import time

# --- パラメータ (軽量化のため調整) ---
RESIZE_WIDTH = 240
MIN_NOISE_AREA = 45 

# ===================================================================
# スレッド1: 操舵用 (描画・フレーム共有・動画保存を有効化)
# ===================================================================
def steering_thread_func(camera_index, shared_state, lock):
    """
    【スレッド版・GUI・動画保存対応】
    操舵（消失点）を検出し、ズレ量(float)と描画フレームを共有辞書に書き込む。
    また、処理結果を 'output_steering.avi' に保存する。
    """
    
    # --- 操舵用パラメータ ---
    CANNY_THRESHOLD1 = 100
    CANNY_THRESHOLD2 = 150
    HOUGH_THRESHOLD = 35 
    HOUGH_MIN_LINE_LENGTH = 35
    HOUGH_MAX_LINE_GAP = 10
    CLIP_LIMIT = 15.0
    TILE_GRID_SIZE = (4, 4)
    TARGET_FPS = 2  
    INTERVAL = 1.0 / TARGET_FPS

    print(f"[操舵スレッド]: カメラ({camera_index})の起動を試みます...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[操舵スレッド] エラー: カメラ ({camera_index}) を開けません。")
        with lock:
            shared_state['stop'] = True 
        return
    print(f"[操舵スレッド]: カメラ({camera_index}) 起動完了。")
    
    last_processed_time = time.time()
    
    # --- ★★★ 動画保存用の設定 ★★★ ---
    out_video_writer = None
    output_filename = 'output_steering.avi'
    
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
        
        current_time = time.time()
        
        if(current_time - last_processed_time) >= INTERVAL:
            
            last_processed_time = current_time
        
            # --- 1. リサイズ ---
            try:
                orig_height, orig_width = frame.shape[:2]
                aspect_ratio = orig_height / orig_width
                resize_height = int(RESIZE_WIDTH * aspect_ratio)
                resized_frame = cv2.resize(frame, (RESIZE_WIDTH, resize_height), interpolation=cv2.INTER_AREA)
                height, width = resized_frame.shape[:2]
                
                # --- ★★★ 動画ライターの初期化 (初回のみ) ★★★ ---
                if out_video_writer is None:
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        # 処理後のフレームサイズ (width, height) と ターゲットFPS で初期化
                        out_video_writer = cv2.VideoWriter(output_filename, fourcc, TARGET_FPS, (width, height))
                        if not out_video_writer.isOpened():
                             print(f"[操舵スレッド] 警告: 動画ファイル ({output_filename}) を開けません。保存はスキップされます。")
                             out_video_writer = None # 失敗したら None のままにする
                        else:
                             print(f"[操舵スレッド]: 動画保存を開始します ({output_filename})")
                    except Exception as e:
                        print(f"[操舵スレッド] 動画ライター初期化エラー: {e}")
                        out_video_writer = None # エラー時も None に


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

                        # ======================================================
                        # --- ★★★ (描画処理 1: 検出線) [変更箇所] ★★★ ---
                        # y = mx + c から、y=0 (画面上端) と y=height (画面下端) の
                        # x座標を計算し、画面いっぱいに線を引く
                        # 画面端まで線を延長して描画
                        points = []
                        if m != 0:
                            x_at_y0 = -c / m
                            if 0 <= x_at_y0 <= width: points.append((int(x_at_y0), 0))
                            x_at_y_height = (height - c) / m
                            if 0 <= x_at_y_height <= width: points.append((int(x_at_y_height), height))
                        y_at_x0 = c
                        if 0 <= y_at_x0 <= height: points.append((0, int(y_at_x0)))
                        y_at_x_width = m * width + c
                        if 0 <= y_at_x_width <= height: points.append((width, int(y_at_x_width)))
                    
                        if len(points) >= 2:
                            cv2.line(resized_frame, points[0], points[1], (0, 255, 0), 2)
                        """
                        try:
                            if m != 0:
                                # 画面上端 (y=0) での x 座標
                                x_at_y0 = int((0 - c) / m)
                                # 画面下端 (y=height) での x 座標
                                x_at_y_height = int((height - c) / m)
                                
                                # 検出した緑の線を描画
                                # (座標が画面外でもcv2.lineが自動でクリッピングしてくれます)
                                cv2.line(resized_frame, (x_at_y0, 0), (x_at_y_height, height), (0, 255, 0), 2)
                                
                        except (OverflowError, ValueError):
                             # m が 0 に極めて近い場合や、値が大きすぎる場合にエラーになるのを防ぐ
                             pass
                        """ 
                        # --- (描画処理 1 ここまで) ---
                        # ======================================================


                intersection_points = []
                if len(diagonal_lines) >= 2:
                    for i in range(len(diagonal_lines)):
                        for j in range(i + 1, len(diagonal_lines)):
                            m1, c1 = diagonal_lines[i]
                            m2, c2 = diagonal_lines[j]
                            # 傾きが似ているか、符号が同じ（同じ方向）場合は除外
                            if abs(m1 - m2) < 1e-5 or m1*m2 > 0 : continue 
                            
                            x = (c2 - c1) / (m1 - m2)
                            y = m1 * x + c1
                            # 画面内外の広めの範囲で交点を許容
                            if -width < x < width * 2 and -height < y < height * 2:
                                intersection_points.append((x, y))

                if intersection_points:
                    x_coords = [p[0] for p in intersection_points]
                    # 中央値を使って外れ値に強い消失点X座標を決定
                    vp_x = int(np.median(x_coords))

                # --- 5. ズレ量を計算 ---
                image_center_x = width / 2
                x_difference = vp_x - image_center_x
                
                # --- ★★★ (描画処理 2: 消失点と中心線) ★★★ ---
                # 消失点 (赤丸) を描画
                cv2.circle(resized_frame, (vp_x, height // 2), 10, (0, 0, 255), -1) 
                # 画像中心 (青線) を描画
                cv2.line(resized_frame, (width // 2, 0), (width // 2, height), (255, 0, 0), 1)

                # --- 6. 共有辞書へ書き込み (ロックを使用) ---
                with lock:
                    shared_state['steering_value'] = x_difference
                    # ★★★ GUI用にフレームを共有 ★★★
                    shared_state['steering_frame'] = resized_frame.copy() 
                
                # --- ★★★ 7. 動画ファイルに書き込み ★★★ ---
                if out_video_writer:
                    try:
                        out_video_writer.write(resized_frame)
                    except Exception as e:
                        print(f"[操舵スレッド] 動画フレームの書き込みエラー: {e}")

            except Exception as e:
                    print(f"[操舵スレッド] 処理中に予期せぬエラー: {e}")
                    pass
        
        time.sleep(0.001) 

    cap.release()
    # --- ★★★ 動画ファイルを閉じる ★★★ ---
    if out_video_writer:
        out_video_writer.release()
        print(f"[操舵スレッド]: 動画ファイル ({output_filename}) を保存・解放しました。")
        
    print("[操舵スレッド]: カメラを解放しました。")


# ===================================================================
# スレッド2: 壁検出用 (★★ 使用しないためコメントアウト ★★)
# ===================================================================
# def wall_thread_func(camera_index, shared_state, lock):
#     """
#     【スレッド版・360度カメラ対応・ヘッドレス】
#     壁を検出し、フラグのみを共有辞書に書き込む。
#     """
#
#     # --- 壁検出用パラメータ ---
#     ( ... 省略 ... )
#    
#     print(f"[壁検出スレッド]: カメラ({camera_index})の起動を試みます...")
#     ( ... 省略 ... )
#     print("[壁検出スレッド]: カメラを解放しました。")
    

# ===================================================================
# スレッド3: 重心検出用 (★★ 使用しないためコメントアウト ★★)
# ===================================================================
# def gravity_thread_func(camera_index, shared_state, lock):
#     """
#     【スレッド版・ヘッドレス】
#     最も暗い部分の重心を検出し、ズレ量(float)のみを共有辞書に書き込む。
#     """
#     
#     # --- 重心検出用パラメータ ---
#     ( ... 省略 ... )
#
#     print(f"[重心スレッド]: カメラ({camera_index})の起動を試みます...")
#     ( ... 省略 ... )
#     print("[重心スレッド]: カメラを解放しました。")