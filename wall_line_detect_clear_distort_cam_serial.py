import cv2
import numpy as np
import math
import os
import sys
from clear_distortion import ImageUndistortion
import serial
import time

# --- パラメータ設定 ---
# Cannyエッジ検出の低閾値　数字を小さくするとエッジを多く拾える
CANNY_THRESHOLD1 = 90
# Cannyエッジ検出の高閾値　数字を小さくするとはっきりしたものしかエッジとして検出されない
CANNY_THRESHOLD2 = 150
# ハフ変換の投票数の閾値　直線を検出するために最低どれくらいの点が必要なのかを決めている
HOUGH_THRESHOLD = 60
# 検出する線の最小長　この値を小さくすると短い線も検出される
HOUGH_MIN_LINE_LENGTH = 50
# 線上の点と見なすための最大間隔
HOUGH_MAX_LINE_GAP = 10
# コントラストの強さを調整
CLIP_LIMIT = 15.0
TILE_GRID_SIZE = (20, 20)
RESIZE_WIDTH = 640
PLAYBACK_SPEED_MS = 30
#ロボットとの通信設定
ENABLE_SERIAL_COMMUNICATION = True
# 使用するカメラの番号 (通常は0か1)
CAMERA_INDEX = 0


# --- 動画ファイルがあるフォルダのパスを指定 ---
VIDEO_FOLDER_PATH = "/Users/shigemitsuhiroki/vscode/sewage_movie/left_side_movie"

def process_frame(frame,undistorter):
    """
    動画の1フレームを処理して、結果画像とエッジ画像を返す関数
    """
    #--- 0. 歪み補正 ---
    frame = undistorter.undistort_image(frame)
    
    # --- 1. リサイズ ---
    orig_height, orig_width = frame.shape[:2]
    aspect_ratio = orig_height / orig_width
    resize_height = int(RESIZE_WIDTH * aspect_ratio)
    resized_frame = cv2.resize(frame, (RESIZE_WIDTH, resize_height), interpolation=cv2.INTER_AREA)
    height, width = resized_frame.shape[:2]

    # --- 2. 前処理 ---
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)
    adjusted = clahe.apply(blurred)
    blurred_again = cv2.GaussianBlur(adjusted, (7, 7), 0)

    # --- 3. Cannyエッジ検出 ---
    edges = cv2.Canny(blurred_again, CANNY_THRESHOLD1, CANNY_THRESHOLD2)

    # --- 4. 確率的ハフ変換 ---
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=HOUGH_THRESHOLD,
                            minLineLength=HOUGH_MIN_LINE_LENGTH,
                            maxLineGap=HOUGH_MAX_LINE_GAP)

    # --- ▼▼▼【修正点】描画処理のロジックを修正 ▼▼▼ ---
    # 描画用のカラー画像とカウンターを初期化
    line_image = np.copy(resized_frame)
    drawn_lines_count = 0

    # --- 5. 線の描画 ---
    if lines is not None:
        # 検出した全ての線に対してループ処理を行う
        for line in lines:
            x1, y1, x2, y2 = line[0]

            angle_deg = math.degrees(math.atan2(y2 - y1, x2 - x1))
            abs_angle_deg = abs(angle_deg)

            # 垂直な線(90°±10°)かどうかを判定
            is_vertical = (80 <= abs_angle_deg <= 100)

            # 垂直な線でなければ、この線は処理しない
            if not is_vertical:
                continue

            # 画面の端から端まで線を延長して描画する
            start_point, end_point = None, None
            if x1 == x2:  # 完全な垂直線の場合
                start_point = (x1, 0)
                end_point = (x1, height)
            else:  # 少し傾いた垂直線の場合
                m = (y2 - y1) / (x2 - x1)
                c = y1 - m * x1
                y_at_x0 = c
                y_at_x_width = m * width + c
                # 線が画像の上下を突き抜けるように始点と終点を設定
                start_point = (0, int(y_at_x0))
                end_point = (width, int(y_at_x_width))

            # 線を描画
            cv2.line(line_image, start_point, end_point, (255, 0, 0), 2) # 青色で描画
            drawn_lines_count += 1
    # --- ▲▲▲【修正点】ここまで ▲▲▲ ---

    # --- 6. 表示用に画像を結合 ---
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # 高さが違う場合のみパディング（念のため）
    h1, w1 = line_image.shape[:2]
    h2, w2 = edges_colored.shape[:2]
    if h1 != h2:
        max_height = max(h1, h2)
        line_image = cv2.copyMakeBorder(line_image, 0, max_height - h1, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        edges_colored = cv2.copyMakeBorder(edges_colored, 0, max_height - h2, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
    combined_image = np.hstack((line_image, edges_colored))

    return combined_image

def main():
    try:
        camera_matrix, dist_coeffs = ImageUndistortion.get_camera_parameters()
        undistorter = ImageUndistortion(camera_matrix, dist_coeffs)
        print("カメラの歪み補正の準備が完了しました。")
    except Exception as e:
        print(f"歪み補正の初期化中にエラーが発生しました: {e}")
        return
    
    # --- シリアルポートの準備 ---
    ser = None
    if ENABLE_SERIAL_COMMUNICATION:
        try:
            # ★★★ ポート名は環境に合わせて変更してください ★★★
            port_name = '/dev/tty.usbmodem14101' # 例: Windows 'COM3', Mac '/dev/tty.usbmodem...'
            ser = serial.Serial(port_name, 115200, timeout=1)
            time.sleep(2)
            print(f"シリアルポート ({port_name}) を開きました。")
        except serial.SerialException as e:
            print(f"シリアルポートを開けませんでした: {e}")
            ser = None
    
    # --- カメラの起動 ---
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"エラー: カメラ (インデックス: {CAMERA_INDEX}) を開けません。")
        return
    print("カメラを起動しました。'q'キーを押すと終了します。")

    """    
    # --- 1. 処理する動画ファイルを選択させる ---
    try:
        video_files = [f for f in os.listdir(VIDEO_FOLDER_PATH) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        if not video_files:
            print(f"エラー: フォルダ '{VIDEO_FOLDER_PATH}' に動画ファイルが見つかりません。")
            return
        
        print("--- 処理する動画を選択してください ---")
        for i, filename in enumerate(video_files):
            print(f"  {i}: {filename}")
        
        choice = int(input("番号を入力してください: "))
        selected_video = video_files[choice]
        video_path = os.path.join(VIDEO_FOLDER_PATH, selected_video)
        print(f"'{selected_video}' を処理します。")
        
    except (FileNotFoundError, IndexError, ValueError) as e:
        print(f"エラー: 動画の選択に失敗しました。({e})")
        return

    # --- 2. 動画の読み込みと情報表示 ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"エラー: 動画ファイル '{video_path}' を開けません。")
        return
    
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_duration_sec = 0

    if fps > 0:
        video_duration_sec = frame_count / fps
        minutes = int(video_duration_sec // 60)
        seconds = int(video_duration_sec % 60)
        print(f"動画の長さ: {minutes}分{seconds}秒 ({fps:.2f} FPS)")
    else:
        print("動画の長さを取得できませんでした。")
    
    # --- 3. ユーザーから再生開始時間を取得 ---
    start_frame = 0
    while True:
        try:
            start_min_str = input("再生を開始する時間（分）を入力してください (例: 1): ")
            start_sec_str = input("再生を開始する時間（秒）を入力してください (例: 30): ")
            start_min = int(start_min_str)
            start_sec = int(start_sec_str)
            total_input_seconds = start_min * 60 + start_sec

            if total_input_seconds >= video_duration_sec:
                print(f"エラー: 入力された時間は動画の長さを超えています。再度入力してください。")
                continue
            
            start_frame = int(total_input_seconds * fps)
            print(f"{start_min}分{start_sec}秒（{start_frame}フレーム目）から再生を開始します。")
            break

        except ValueError:
            print("エラー: 半角数字で入力してください。")
        except Exception as e:
            print(f"予期せぬエラーが発生しました: {e}")
            return
        
    # --- 4. メインループで動画を再生・処理 ---
    print("\nビデオの処理を開始します。")
    print("  - qキー: 終了")
    print("  - スペースキー: 一時停止 / 再生")
    """
    #paused = False
    try:
        #cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while True:
            #if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\nビデオが終了しました。")
                break
            
            processed_and_combined_frame = process_frame(frame,undistorter)            
            cv2.imshow('Processed Video | Canny Edges', processed_and_combined_frame)
            
            # シリアル通信が有効ならコマンドを送信
            #if ser and ser.is_open:
            #    ser.write(command.encode())
            #    print(f"送信コマンド: {command.strip()} (壁の平均X座標: {avg_x})")
            key = cv2.waitKey(PLAYBACK_SPEED_MS) & 0xFF
            
            if key == ord('q'):
                print("\n処理を中断しました。")
                break
            #elif key == ord(' '):
            #    paused = not paused
    finally:
        # --- 5. 終了処理 ---
        if ser and ser.is_open:
            ser.close()
            print("シリアルポートを閉じました。")
        cap.release()
        cv2.destroyAllWindows()
        print("ビデオを解放し、ウィンドウを閉じました。")

if __name__ == '__main__':
    main()