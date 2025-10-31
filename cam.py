import cv2
import sys

print("カメラを起動します。録画を開始します。")
print("映像ウィンドウをアクティブにした状態で 'q' を押すか、")
print("このターミナルで Ctrl+C を押すと録画を終了します。")

# カメラを開く
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("エラー: カメラを開けませんでした。")
    sys.exit()

# --- 録画のための設定 (ここから NEW) ---

# 1. 保存する動画の解像度を取得
#    書き込むフレームとサイズが合っていないとエラーになるため
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (frame_width, frame_height)

# 2. 保存するファイル名
#    拡張子を .avi にします (後述)
output_filename = 'output.avi'

# 3. 動画のコーデック (FourCC) を指定
#    'XVID' は .avi 形式で一般的によく使われるコーデック
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# 4. フレームレート (FPS) を指定
#    cap.get(cv2.CAP_PROP_FPS) は0を返すことがあるため、
#    ここでは 20.0 のような固定値を指定するのが安全です。
fps = 20.0

# 5. VideoWriter オブジェクトを作成
try:
    out = cv2.VideoWriter(output_filename, fourcc, fps, size)
    print(f"録画を開始しました。保存先: {output_filename}")
except Exception as e:
    print(f"エラー: VideoWriterの作成に失敗しました: {e}")
    cap.release()
    sys.exit()

# --- 録画のための設定 (ここまで NEW) ---


try:
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("エラー: フレームの取得に失敗しました。")
            break
            
        # --- 録画処理 (NEW) ---
        # 読み込んだフレームをファイルに書き込む
        out.write(frame)
        # ---------------------
            
        # 画面に表示
        cv2.imshow('Camera Feed (Press "q" to quit)', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("'q' キーが押されたため終了します。")
            break

except KeyboardInterrupt:
    print("\nCtrl+C が押されたため終了します。")

finally:
    # 終了処理
    print("録画を終了し、ファイルを保存します。")
    cap.release()     # カメラを解放
    out.release()     # <--- NEW: 動画ファイルを解放 (ここでファイルが正しく保存されます)
    cv2.destroyAllWindows()
    print(f"動画ファイル '{output_filename}' が保存されました。")