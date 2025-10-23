# ファイル名: main_control_thread_display.py
# (robot_vision_thread.py と同じフォルダに保存してください)

# import serial # シリアル通信はしない
import time
import threading 
import cv2
# robot_vision_thread.py からスレッド用の関数をインポート
from robot_vision_thread_display import steering_thread_func, wall_thread_func 

### --- ステートマシンの状態定義 ---
STATE_DRIVING = 0
STATE_STOPPED = 1

### --- 設定 ---
# 停止・クールダウン
STOP_DURATION_SEC = 3.0    
STOP_COOLDOWN_SEC = 10.0   

# ロボット制御
STEERING_THRESHOLD = 20 

# カメラ設定
CAMERA_INDEX_STEERING = 0 # ★★★ 操舵用カメラの番号 ★★★
CAMERA_INDEX_WALL = 1     # ★★★ 壁検出用カメラの番号 ★★★

# メインループの周期 (ミリ秒)
# time.sleep(0.2) の代わりに cv2.waitKey(50) を使う
MAIN_LOOP_WAIT_MS = 50 # 50ms (約20FPSでGUIを更新)


def main():
    
    # --- 1. スレッド間共有変数の初期化 ---
    shared_state = {
        'steering_value': 0.0,
        'wall_detected': 0,
        'stop': False, 
        'steering_frame': None, # ★★★ 追加: 操舵フレーム用 ★★★
        'wall_frame': None      # ★★★ 追加: 壁検出フレーム用 ★★★
    }
    
    lock = threading.Lock()

    # --- 2. シリアルポートの準備 (削除) ---
            
    # --- 3. 画像処理スレッドを起動 ---
    t_steering = threading.Thread(target=steering_thread_func, 
                                 args=(CAMERA_INDEX_STEERING, shared_state, lock))
    
    t_wall = threading.Thread(target=wall_thread_func, 
                             args=(CAMERA_INDEX_WALL, shared_state, lock))
    
    print("[メイン]: 操舵スレッドと壁検出スレッドを起動します...")
    t_steering.start()
    t_wall.start()
    
    # --- 4. メイン制御ループ (ステートマシン) ---
    current_state = STATE_DRIVING
    stop_timer_end_time = 0.0
    stop_cooldown_end_time = 0.0
    
    print("[メイン]: 制御ループを開始します。'q'キーで終了します。")
    
    # フレーム取得用の変数をループ外で初期化
    frame_steering = None
    frame_wall = None

    try:
        while True:
            current_time = time.time()
            
            # --- 4-1. 共有変数を安全に読み出す ---
            with lock:
                if shared_state['stop']:
                    print("[メイン]: スレッドからの停止要求を検出。ループを終了します。")
                    break
                    
                current_steering_diff = shared_state['steering_value']
                is_wall_detected = (shared_state['wall_detected'] == 1)
                
                # ★★★ 変更: フレームを取得 (.copy() してロックをすぐ解放) ★★★
                if shared_state['steering_frame'] is not None:
                    frame_steering = shared_state['steering_frame'].copy()
                if shared_state['wall_frame'] is not None:
                    frame_wall = shared_state['wall_frame'].copy()
            
            # --- 4-2. 操舵コマンドを生成 ---
            steering_command = "S" 
            if abs(current_steering_diff) > STEERING_THRESHOLD:
                if current_steering_diff > 0:
                    steering_command = f"R {current_steering_diff:.2f}" 
                else:
                    steering_command = f"L {abs(current_steering_diff):.2f}" 
            
            # --- 4-3. ステートマシンによるコマンド決定 ---
            final_command = steering_command 

            if current_state == STATE_DRIVING:
                if is_wall_detected and current_time > stop_cooldown_end_time:
                    print(f"[メイン]: !!! 壁を検出！ {STOP_DURATION_SEC}秒間停止します。 !!!")
                    current_state = STATE_STOPPED
                    stop_timer_end_time = current_time + STOP_DURATION_SEC
                    stop_cooldown_end_time = current_time + STOP_COOLDOWN_SEC
                    final_command = "H" 
                else:
                    final_command = steering_command

            elif current_state == STATE_STOPPED:
                if current_time > stop_timer_end_time:
                    print("[メイン]: --- 停止時間終了。運転を再開します。---")
                    current_state = STATE_DRIVING
                    final_command = steering_command
                else:
                    final_command = "H"
            
            # --- 4-4. シリアル通信 (削除済み) ---
            
            # --- 4-5. 状態の表示 (標準出力) ---
            state_text = "DRIVING" if current_state == STATE_DRIVING else "STOPPED"
            print(f"状態: {state_text}, "
                  f"壁: {is_wall_detected}, "
                  f"ズレ(数値): {current_steering_diff:.2f}, "
                  f"コマンド: {final_command}")
            
            # --- 4-6. 画像の表示 (★★★ 追加 ★★★) ---
            if frame_steering is not None:
                cv2.imshow('Steering Camera', frame_steering)
            
            if frame_wall is not None:
                cv2.imshow('Wall Camera', frame_wall)

            # --- 4-7. メインループの待機 (★★★ 変更 ★★★) ---
            # time.sleep() の代わりに cv2.waitKey() を使用
            key = cv2.waitKey(MAIN_LOOP_WAIT_MS) & 0xFF
            if key == ord('q'):
                print("\n[メイン]: 'q'キーを検出。全スレッドを停止します。")
                break # ループを抜ける

    except KeyboardInterrupt:
        print("\n[メイン]: Ctrl+Cを検出。全スレッドを停止します。")
    finally:
        # --- 5. 終了処理 ---
        print("[メイン]: 終了処理中...")
        
        with lock:
            shared_state['stop'] = True
        
        t_steering.join()
        t_wall.join()
        print("[メイン]: 両スレッドが終了しました。")
            
        cv2.destroyAllWindows() # ★★★ GUIウィンドウを閉じる ★★★
        print("[メイン]: プログラムを終了します。")


if __name__ == '__main__':
    main()