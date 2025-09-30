import numpy as np
import cv2  # OpenCVをインポート
import serial
import time

# --- パラメータ設定 ---
# 実際にロボットと通信する場合は、ここをTrueにしてください
ENABLE_SERIAL_COMMUNICATION = False
# 使用するカメラの番号 (通常は0か1)
CAMERA_INDEX = 0 
# ロボットのズレを判定する閾値 (ピクセル単位)
THRESHOLD = 20

# --- 関数定義 ---
def process_image_and_get_command(image_array):
    """
    1フレームの画像配列を受け取り、重心を計算して制御コマンドを返す関数
    """
    # 画像の高さと幅を取得
    height, width = image_array.shape

    # 画素値の反転 (暗い部分を重くするため)
    inverted_array = 255 - image_array

    # 重心の計算
    total_weight = np.sum(inverted_array)

    if total_weight == 0:
        # 画像が真っ白の場合 (ゼロ除算エラーを防止)
        center_x = width / 2
    else:
        # x座標の重心を計算 (y座標は今回は不要)
        x_coords = np.arange(width)
        center_x = np.sum(x_coords * np.sum(inverted_array, axis=0)) / total_weight

    # 画像中心とのズレを計算 (中心を0とする座標系)
    image_center_x = width / 2
    x_difference = center_x - image_center_x

    # ズレの大きさに基づいて状態を判断し、コマンドを生成
    command = ''
    if abs(x_difference) > THRESHOLD:
        if x_difference > 0:
            # 右にズレている
            command = f"R {x_difference:.2f}\n"
        else:
            # 左にズレている
            command = f"L {abs(x_difference):.2f}\n"
    else:
        # ほぼ中央
        command = "S \n"
        
    return command, center_x, x_difference

# --- メイン処理 ---
def main():
    ser = None
    # 1. シリアルポートの準備
    if ENABLE_SERIAL_COMMUNICATION:
        try:
            # ★★★ ポート名は環境に合わせて変更してください ★★★
            port_name = '/dev/tty.usbmodem14101' # (例: Windowsなら 'COM3')
            ser = serial.Serial(port_name, 115200, timeout=1)
            time.sleep(2) # 接続が安定するまで待機
            print(f"Cugoとのシリアルポート ({port_name}) を開きました。")
        except serial.SerialException as e:
            print(f"シリアルポートを開けませんでした: {e}")
            print("シリアル通信なしで続行します。")
            
    # 2. カメラの準備
    cap = cv2.VideoCapture(CAMERA_INDEX) # カメラを開く,カメラが一台なら0
    if not cap.isOpened():
        print(f"エラー: カメラ (インデックス: {CAMERA_INDEX}) を開けません。")
        return

    print("カメラを起動しました。'q'キーを押すと終了します。")

    try:
        # 3. 無限ループで映像を処理
        while True:
            # カメラから1フレーム読み込む，retは読み取りに成功したらTrue
            ret, frame = cap.read()
            if not ret:
                print("エラー: フレームをキャプチャできません。")
                break

            # 画像をグレースケールに変換
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 画像を処理してコマンドを取得
            command, centroid_x, diff = process_image_and_get_command(gray_frame)
            
            # コマンドを送信
            if ser and ser.is_open: #.is_opnenは開いているかどうかを判定する
                ser.write(command.encode())
            
            # 画面に状態を表示
            print(f"ズレ: {diff:6.2f} | 送信コマンド: {command.strip()}", end="\r")

            # --- 結果を映像に描画して表示 ---
            height, width, _ = frame.shape
            # 画像中心の線 (水色)
            cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 255, 0), 2)
            # 計算された重心の位置 (赤丸)
            cv2.circle(frame, (int(centroid_x), height // 2), 10, (0, 0, 255), -1)
            # プレビューウィンドウを表示
            cv2.imshow('Cugo Camera View', frame)

            # 'q'キーが押されたらループを抜ける
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n終了します。")
                break
    finally:
        # 4. 終了処理
        if ser and ser.is_open:
            # ロボットを停止させるコマンドを送信
            ser.write("S \n".encode()) 
            ser.close()
            print("シリアルポートを閉じました。")
        
        cap.release()
        cv2.destroyAllWindows()
        print("カメラを解放し、ウィンドウを閉じました。")

if __name__ == '__main__':
    main()