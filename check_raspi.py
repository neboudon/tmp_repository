import cv2
import threading
import time

# VideoCaptureを別スレッドで実行し、常に最新のフレームだけを保持するクラス
class ThreadedCamera:
    def __init__(self, src=0):
        # GoProのカメラINDEXを指定 (環境によって0や1に変わる可能性があります)
        self.capture = cv2.VideoCapture(src)
        
        # GoPro本体で設定した解像度をリクエスト (1280x720 = 720p)
        # これにより、意図しない高解像度でのデータ受信を防ぎます
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # FPSも低めにリクエスト (30fps)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        
        # バッファサイズを最小にし、遅延の蓄積を防ぐ
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 実際のカメラ設定を取得して表示
        width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        print(f"--- カメラ情報 ---")
        print(f"解像度: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"--------------------")

        if not self.capture.isOpened():
            print("エラー: カメラを開けませんでした。")
            return

        self.status, self.frame = self.capture.read()
        
        # フレームをバックグラウンドで読み込み続けるスレッドを開始
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        
    def update(self):
        while True:
            if self.capture.isOpened():
                # 常に最新のフレームを読み込み、古いフレームは破棄される
                self.status, self.frame = self.capture.read()
            time.sleep(.01) # CPU負荷を少し下げるため

    def read(self):
        # メインスレッドは、保持されている最新のフレームを返すだけ
        return self.status, self.frame

    def release(self):
        self.capture.release()

# --- メイン処理 ---
if __name__ == '__main__':
    CAMERA_INDEX = 0  # GoProが認識されているカメラ番号
    
    print("カメラの初期化を開始します...")
    threaded_camera = ThreadedCamera(CAMERA_INDEX)
    
    # カメラの初期化には少し時間がかかる場合がある
    time.sleep(2)
    print("映像の表示を開始します。'q'キーで終了します。")

    while True:
        try:
            status, frame = threaded_camera.read()
            
            # フレームが正常に取得できた場合のみ表示
            if status:
                cv2.imshow("GoPro Live Feed (Optimized for RPi)", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("終了します。")
                break
        except KeyboardInterrupt:
            print("終了します。")
            break
            
    threaded_camera.release()
    cv2.destroyAllWindows()