import pyrealsense2 as rs
import numpy as np
import cv2
import sys

print("RealSenseカメラを起動します。録画と距離測定を開始します。")
print("映像ウィンドウをアクティブにした状態で 'q' を押すか、")
print("このターミナルで Ctrl+C を押すと録画を終了します。")

# --- 1. RealSenseのセットアップ ---
pipeline = rs.pipeline()
config = rs.config()

# 解像度とFPSを設定 (録画設定と合わせる)
WIDTH = 640
HEIGHT = 480
FPS = 30 # RealSenseとVideoWriterのFPSを合わせる

config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

# (重要) 深度とカラーの位置合わせ（アライメント）設定
align_to = rs.stream.color
align = rs.align(align_to)

# ストリーミング開始
try:
    profile = pipeline.start(config)
except RuntimeError as e:
    print(f"エラー: RealSenseカメラの起動に失敗しました。 {e}")
    sys.exit()

# --- 2. 録画のための設定 (cam.py から流用) ---

# size はRealSenseの設定値を使用
size = (WIDTH, HEIGHT)
output_filename = 'output.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# fps はRealSenseの設定値 (FPS) を使用
fps_video = float(FPS) 

# VideoWriter オブジェクトを作成
try:
    out = cv2.VideoWriter(output_filename, fourcc, fps_video, size)
    print(f"録画を開始しました。保存先: {output_filename}")
except Exception as e:
    print(f"エラー: VideoWriterの作成に失敗しました: {e}")
    pipeline.stop()
    sys.exit()

# --- 3. メインループ (距離測定 + 録画) ---
try:
    while True:
        # --- RealSenseのフレーム取得 ---
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        # OpenCVで扱えるNumpy配列に変換
        color_image = np.asanyarray(color_frame.get_data())
        
        # --- 距離の測定 (NEW) ---
        center_x = WIDTH // 2
        center_y = HEIGHT // 2
        distance = depth_frame.get_distance(center_x, center_y)
        
        # --- 映像への描画 (NEW) ---
        # 十字線
        cv2.drawMarker(color_image, (center_x, center_y), (0, 0, 255), 
                       markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        
        # 距離テキスト
        if distance > 0:
            distance_text = f"{distance:.2f} m"
        else:
            distance_text = "N/A" # 測定不可
            
        cv2.putText(color_image, 
                    distance_text, 
                    (center_x - 50, center_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.0, 
                    (0, 255, 0), # 緑色
                    2)

        # --- 録画処理 (cam.py から流用) ---
        # 描画済みの「color_image」をファイルに書き込む
        out.write(color_image)
        # ---------------------
            
        # 画面に表示
        cv2.imshow('RealSense Recording (Press "q" to quit)', color_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q' キーが押されたため終了します。")
            break

except KeyboardInterrupt:
    print("\nCtrl+C が押されたため終了します。")

finally:
    # 終了処理
    print("録画を終了し、ファイルを保存します。")
    pipeline.stop()   # <--- NEW: RealSenseを停止
    cap.release() if 'cap' in locals() and cap.isOpened() else None # 元コードの名残(念の為)
    out.release()     # <--- 動画ファイルを解放
    cv2.destroyAllWindows()
    print(f"動画ファイル '{output_filename}' が保存されました。")