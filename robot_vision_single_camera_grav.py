# ファイル名: robot_vision_single_camera.py
# (main_control_single_camera.py と同じフォルダに保存してください)

import cv2
import numpy as np
import math
import time

# --- パラメータ (軽量化のため調整) ---
RESIZE_WIDTH = 240
MIN_NOISE_AREA = 45 # (重心検出では使われない)

# ===================================================================
# スレッド1: 操舵用 (★★ 使用しないためコメントアウト ★★)
# ===================================================================
# def steering_thread_func(camera_index, shared_state, lock):
#     """
#     【スレッド版・GUI・動画保存対応】
#     操舵（消失点）を検出し、ズレ量(float)と描画フレームを共有辞書に書き込む。
#     また、処理結果を 'output_steering.avi' に保存する。
#     """
#     
#     # --- 操舵用パラメータ ---
#     ( ... 中身をすべてコメントアウト ... )
#
#     print("[操舵スレッド]: カメラを解放しました。")


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
#     print("[壁検出スレッド]: カメラを解放しました。")
    

# ===================================================================
# スレッド3: 重心検出用 (★★★ 有効化 ★★★)
# ===================================================================
def gravity_thread_func(camera_index, shared_state, lock):
    """
    【スレッド版・GUI・動画保存対応】
    最も暗い部分の重心を検出し、ズレ量(float)と描画フレームを共有辞書に書き込む。
    また、処理結果を 'output_gravity.avi' に保存する。
    """
    
    # --- 重心検出用パラメータ ---
    TARGET_FPS = 5  
    INTERVAL = 1.0 / TARGET_FPS

    print(f"[重心スレッド]: カメラ({camera_index})の起動を試みます...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[重心スレッド] エラー: カメラ ({camera_index}) を開けません。")
        with lock:
            shared_state['stop'] = True 
        return
    print(f"[重心スレッド]: カメラ({camera_index}) 起動完了。")
    
    last_processed_time = time.time()
    
    # --- ★★★ 動画保存用の設定 ★★★ ---
    out_video_writer = None
    output_filename = 'output_gravity.avi' # 保存ファイル名を変更
    
    # --- メインループ ---
    while True:
        with lock:
            if shared_state['stop']:
                break
                
        ret, frame = cap.read()
        if not ret:
            print("[重心スレッド] エラー: フレームを取得できません。")
            time.sleep(0.5)
            continue
        
        current_time = time.time()
        
        if(current_time - last_processed_time) >= INTERVAL:
            
            last_processed_time = current_time
        
            try:
                # --- 1. リサイズ ---
                orig_height, orig_width = frame.shape[:2]
                aspect_ratio = orig_height / orig_width
                resize_height = int(RESIZE_WIDTH * aspect_ratio)
                resized_frame = cv2.resize(frame, (RESIZE_WIDTH, resize_height), interpolation=cv2.INTER_AREA)
                height, width = resized_frame.shape[:2]

                # --- ★★★ 動画ライターの初期化 (初回のみ) ★★★ ---
                if out_video_writer is None:
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        out_video_writer = cv2.VideoWriter(output_filename, fourcc, TARGET_FPS, (width, height))
                        if not out_video_writer.isOpened():
                             print(f"[重心スレッド] 警告: 動画ファイル ({output_filename}) を開けません。保存はスキップされます。")
                             out_video_writer = None
                        else:
                             print(f"[重心スレッド]: 動画保存を開始します ({output_filename})")
                    except Exception as e:
                        print(f"[重心スレッド] 動画ライター初期化エラー: {e}")
                        out_video_writer = None

                # --- 2. グレースケールに変換 ---
                gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

                # --- 3. 重心計算 ---
                # 白(255)を0、黒(0)を255に反転させ、暗い部分ほど重みを大きくする
                inverted_array = 255 - gray_frame
                total_weight = np.sum(inverted_array)
                
                center_x = width / 2 # デフォルト値 (真っ白な画像など)
                
                if total_weight > 0:
                    # 各X座標の重みの合計を計算
                    x_coords = np.arange(width)
                    weighted_sum_x = np.sum(x_coords * np.sum(inverted_array, axis=0))
                    
                    # 重心 (Weighted Average) を計算
                    center_x = weighted_sum_x / total_weight

                # --- 4. ズレ量を計算 ---
                image_center_x = width / 2
                x_difference = center_x - image_center_x
                
                # --- 5. デバッグ描画 (★★★ 有効化 ★★★) ---
                # 画像中心 (水色)
                cv2.line(resized_frame, (width // 2, 0), (width // 2, height), (255, 255, 0), 2)
                # 重心位置 (赤丸)
                cv2.circle(resized_frame, (int(center_x), height // 2), 10, (0, 0, 255), -1)

                # --- 6. 共有辞書へ書き込み (ロックを使用) ---
                with lock:
                    shared_state['gravity_value'] = x_difference
                    #shared_state['gravity_frame'] = resized_frame.copy() # ★★★ GUI用に有効化 ★★★
            
                # --- ★★★ 7. 動画ファイルに書き込み ★★★ ---
                if out_video_writer:
                    try:
                        out_video_writer.write(resized_frame)
                    except Exception as e:
                        print(f"[重心スレッド] 動画フレームの書き込みエラー: {e}")

            except Exception as e:
                    print(f"[重心スレッド] 処理中に予期せぬエラー: {e}")
                    pass
        
        time.sleep(0.001) 

    cap.release()
    
    # --- ★★★ 動画ファイルを閉じる ★★★ ---
    if out_video_writer:
        out_video_writer.release()
        print(f"[重心スレッド]: 動画ファイル ({output_filename}) を保存・解放しました。")
        
    print("[重心スレッド]: カメラを解放しました。")