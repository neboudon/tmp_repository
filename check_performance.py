import cv2
import time

# --- ⚙️ 設定項目 ---
# 使用するカメラのインデックス番号
CAMERA_INDEX = 0
# 1つの設定あたり何秒間テストするか
TEST_DURATION_SECONDS = 10
# テストしたい解像度(幅, 高さ)とFPSの組み合わせリスト
# このリストを編集して、試したい設定を追加・削除してください。
SETTINGS_TO_TEST = [
    (1920, 1080, 30),  # Full HD @ 30 FPS
    (1280, 720, 60),   # HD @ 60 FPS
    (1280, 720, 30),   # HD @ 30 FPS
    (640, 480, 30),    # VGA @ 30 FPS
    (320, 240, 30),    # QVGA @ 30 FPS
]
# --------------------

def run_test(width_req, height_req, fps_req):
    """指定された設定でカメラのパフォーマンステストを実行する関数"""
    print("-" * 50)
    print(f"🚀 テスト開始: 要求設定 = {width_req}x{height_req} @ {fps_req} FPS")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("エラー: カメラを開けませんでした。")
        return

    # カメラに設定を要求
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width_req)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_req)
    cap.set(cv2.CAP_PROP_FPS, fps_req)

    # 実際に設定された値を取得
    time.sleep(1) # 設定反映待ち
    width_actual = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_actual = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_actual = cap.get(cv2.CAP_PROP_FPS)
    print(f"✅ 実際のカメラ設定: {width_actual}x{height_actual} @ {fps_actual:.2f} FPS")

    frame_count = 0
    start_time = time.time()
    
    # 指定された秒数だけループを回す
    while time.time() - start_time < TEST_DURATION_SECONDS:
        ret, frame = cap.read()
        if not ret:
            print("フレームの取得に失敗しました。")
            break
        
        frame_count += 1

        # 画面に現在の情報を描画
        elapsed_time = time.time() - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        info_text = f"Actual: {width_actual}x{height_actual} @ {current_fps:.1f} FPS"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Performance Test", frame)
        
        # 'q'キーでテストを中断
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("テストを中断しました。")
            break
    
    end_time = time.time()
    total_time = end_time - start_time
    average_fps = frame_count / total_time if total_time > 0 else 0

    print(f"\n--- 結果 ---")
    print(f"処理時間: {total_time:.2f} 秒")
    print(f"処理フレーム数: {frame_count} フレーム")
    print(f"平均処理性能: {average_fps:.2f} FPS")
    print("-" * 50 + "\n")

    cap.release()
    cv2.destroyAllWindows()
    time.sleep(2) # 次のテストの前に少し待つ

def main():
    for w, h, f in SETTINGS_TO_TEST:
        run_test(w, h, f)
    print("全てのテストが完了しました。")

if __name__ == '__main__':
    main()