import asyncio
import cv2  # OpenCVライブラリをインポート
from open_gopro import WiredGoPro

# GoProのプレビューストリームを受信するアドレス
PREVIEW_STREAM_URL = "udp://127.0.0.1:8554"

async def main():
    print("有線接続でGoProを探しています...")
    gopro = None
    cap = None
    try:
        # WiredGoPro を使って直接USB接続を試みる
        gopro = WiredGoPro()
        await gopro.open()
        print("GoProに有線で接続しました！")

        # 1. 録画を開始
        print("録画を開始します...")
        await gopro.http_command.set_shutter(shutter=True)

        # 2. GoProにプレビューストリームを開始させる
        print("プレビューストリームを開始します...")
        await gopro.http_command.start_preview()

        # 3. OpenCVでストリームを受信する準備
        cap = cv2.VideoCapture(PREVIEW_STREAM_URL)
        if not cap.isOpened():
            raise ConnectionError(f"OpenCVがストリーム({PREVIEW_STREAM_URL})を開けませんでした。")

        print("\nプレビューウィンドウが表示されます。'q'キーを押すと録画を停止して終了します。")

        # 4. 'q'キーが押されるまで映像を表示し続ける
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ストリームからフレームを読み取れませんでした。")
                break

            # ウィンドウにフレームを表示
            cv2.imshow("GoPro Live Recording", frame)

            # 'q'キーが押されたらループを抜ける
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n'q'キーが押されました。")
                break

    except Exception as e:
        print(f"処理中にエラーが発生しました: {e}")

    finally:
        # 5. 終了処理
        print("\nクリーンアップ処理を実行します...")
        if gopro:
            print("録画を停止します...")
            await gopro.http_command.set_shutter(shutter=False)
            print("プレビューストリームを停止します...")
            await gopro.http_command.stop_preview()
            await gopro.close()
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        print("リソースを解放し、終了しました。")

if __name__ == "__main__":
    asyncio.run(main())