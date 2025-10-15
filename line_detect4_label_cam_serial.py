import cv2
import numpy as np
import math
import os # フォルダ操作のために追加
import serial
import time

# --- パラメータ設定 ---
# Cannyエッジ検出の低閾値
CANNY_THRESHOLD1 = 100
# Cannyエッジ検出の高閾値
CANNY_THRESHOLD2 = 150
# ハフ変換の投票数の閾値
HOUGH_THRESHOLD = 90
# 検出する線の最小長
HOUGH_MIN_LINE_LENGTH = 100
# 線上の点と見なすための最大間隔
HOUGH_MAX_LINE_GAP = 10
# コントラストの強さを調整
CLIP_LIMIT = 15.0
# CLAHEの分割数
TILE_GRID_SIZE = (20, 20)
# 画像のリサイズ後の幅
RESIZE_WIDTH = 640
# ▼▼▼【追加】ノイズとみなす最小面積（ピクセル数）▼▼▼
MIN_NOISE_AREA = 80
# 動画の再生速度（ミリ秒）
PLAYBACK_SPEED_MS = 30 # 30ミリ秒ごとに1フレーム進める
#ロボットとの通信設定
ENABLE_SERIAL_COMMUNICATION = True
# 使用するカメラの番号 (通常は0か1)
CAMERA_INDEX = 0
# ロボットのズレを判定する閾値 (ピクセル単位)
THRESHOLD = 20


def process_frame(frame):
    """
    動画の1フレームを処理して、結果画像とエッジ画像を返す関数
    """
    # --- 1. リサイズ ---
    orig_height, orig_width = frame.shape[:2]
    aspect_ratio = orig_height / orig_width
    resize_height = int(RESIZE_WIDTH * aspect_ratio)
    resized_frame = cv2.resize(frame, (RESIZE_WIDTH, resize_height), interpolation=cv2.INTER_AREA)

    # リサイズ後の高と幅を取得
    height, width = resized_frame.shape[:2]

    # --- 変数の初期化 ---
    vp_x = width // 2
    vp_y = height // 2
    x_difference = 0.0
    command = "S \n" # コマンドも初期化

    # --- 2. 前処理 ---
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_GRID_SIZE)
    adjusted = clahe.apply(blurred)
    blurred_again = cv2.GaussianBlur(adjusted, (7, 7), 0)

    # --- 3. Cannyエッジ検出 ---
    edges = cv2.Canny(blurred_again, CANNY_THRESHOLD1, CANNY_THRESHOLD2)

    # --- ▼▼▼【統合】ラベリングによるノイズ除去 ▼▼▼ ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)
    cleaned_edges = np.zeros_like(edges)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > MIN_NOISE_AREA:
            cleaned_edges[labels == i] = 255
    # --- ▲▲▲【統合】ここまで ▲▲▲ ---

    # --- 4. 確率的ハフ変換 ---
    # 【変更】ノイズ除去後の `cleaned_edges` を使用
    lines = cv2.HoughLinesP(cleaned_edges, 1, np.pi/180,
                            threshold=HOUGH_THRESHOLD,
                            minLineLength=HOUGH_MIN_LINE_LENGTH,
                            maxLineGap=HOUGH_MAX_LINE_GAP)

    # 描画用のカラー画像を作成
    line_image = np.copy(resized_frame)
    diagonal_lines = []

    # --- 5. 線の描画と消失点の計算 ---
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle_rad = math.atan2(y2 - y1, x2 - x1)
            angle_deg = math.degrees(angle_rad)
            abs_angle_deg = abs(angle_deg)

            is_horizontal = (abs_angle_deg <= 10) or (abs_angle_deg >= 175)
            is_vertical = (80 <= abs_angle_deg <= 100)
            if is_horizontal or is_vertical:
                continue

            if x1 == x2: continue
            else:
                m = (y2 - y1) / (x2 - x1)
                c = y1 - m * x1
            diagonal_lines.append((m, c))

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
                cv2.line(line_image, points[0], points[1], (0, 255, 0), 2)

    intersection_points = []
    if len(diagonal_lines) >= 2:
        for i in range(len(diagonal_lines)):
            for j in range(i + 1, len(diagonal_lines)):
                m1, c1 = diagonal_lines[i]
                m2, c2 = diagonal_lines[j]
                if abs(m1 - m2) < 1e-5: continue
                x = (c2 - c1) / (m1 - m2)
                y = m1 * x + c1
                if -width < x < width * 2 and -height < y < height * 2:
                    intersection_points.append((x, y))

    if intersection_points:
        x_coords = [p[0] for p in intersection_points]
        y_coords = [p[1] for p in intersection_points]
        vp_x = int(np.median(x_coords))
        vp_y = int(np.median(y_coords))

        if 0 <= vp_y < height and 0 <= vp_x < width:
            bgr_value = resized_frame[vp_y, vp_x]
            gray_value = gray[vp_y, vp_x]
            print(f"交点座標: ({vp_x}, {vp_y}), BGR値: {bgr_value}, 輝度値: {gray_value}")
        else:
            print(f"交点座標: ({vp_x}, {vp_y}) は画像範囲外です。")

    cv2.circle(line_image, (vp_x, vp_y), 10, (0, 0, 255), -1)

    # 画像中心とのズレを計算し、コマンドを生成
    image_center_x = width / 2
    x_difference = vp_x - image_center_x
    command = ''
    if abs(x_difference) > THRESHOLD:
        if x_difference > 0:
            command = f"R {x_difference:.2f}\n" # 右にズレ
        else:
            command = f"L {abs(x_difference):.2f}\n" # 左にズレ
    else:
        command = "S \n" # ほぼ中央

    # --- 6. 表示用に画像を結合 ---
    # 【変更】ノイズ除去後のエッジ画像を表示
    edges_colored = cv2.cvtColor(cleaned_edges, cv2.COLOR_GRAY2BGR)

    h1, w1 = line_image.shape[:2]
    h2, w2 = edges_colored.shape[:2]
    max_height = max(h1, h2)
    if h1 < max_height:
        line_image = cv2.copyMakeBorder(line_image, 0, max_height - h1, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    if h2 < max_height:
        edges_colored = cv2.copyMakeBorder(edges_colored, 0, max_height - h2, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    combined_image = np.hstack((line_image, edges_colored))

    return combined_image, command, vp_x, x_difference


def main():
    ser = None
    if ENABLE_SERIAL_COMMUNICATION:
        try:
            # ★★★ ポート名は環境に合わせて変更してください ★★★
            port_name = '/dev/tty.usbmodem14101' # (例: Windows 'COM3', Mac '/dev/tty.usbmodem...')
            ser = serial.Serial(port_name, 115200, timeout=1)
            time.sleep(2)
            print(f"シリアルポート ({port_name}) を開きました。")
        except serial.SerialException as e:
            print(f"シリアルポートを開けませんでした: {e}")
            ser = None

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"エラー: カメラ (インデックス: {CAMERA_INDEX}) を開けません。")
        return
    print("カメラを起動しました。'q'キーを押すと終了します。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("エラー: フレームをキャプチャできません。")
                break

            processed_frame, command, vp_x, diff = process_frame(frame)
            cv2.imshow('Processed Video | Canny Edges', processed_frame)

            if ser and ser.is_open:
                ser.write(command.encode())
                print(f"送信コマンド: {command.strip()} (ズレ: {diff:.2f})") # 送信内容をコンソールに表示

            key = cv2.waitKey(PLAYBACK_SPEED_MS) & 0xFF
            if key == ord('q'):
                print("\n処理を中断しました。")
                break
    finally:
        if ser and ser.is_open:
            ser.write("S \n".encode())
            ser.close()
            print("シリアルポートを閉じました。")
        cap.release()
        cv2.destroyAllWindows()
        print("ビデオを解放し、ウィンドウを閉じました。")

if __name__ == '__main__':
    main()