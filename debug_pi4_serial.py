# ファイル名: debug_pi4_serial.py

import serial
import time

# ★★★ここが重要★★★
# UbuntuでGP14/GP15を使うための設定(cmdline.txt編集)を行ったため、
# ポート名は '/dev/ttyS0' になります。
SERIAL_PORT = '/dev/ttyS0'
SERIAL_BAUDRATE = 115200

ser = None

try:
    # シリアルポートを開く
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=1)
    print(f"--- シリアルポート ({SERIAL_PORT}) を開きました。 ---")
    
    while True:
        # --- 1. Pi 4 から Pico へデータを送信 ---
        message_to_send = "PING_FROM_PI\n" # 改行コード(\n)が重要
        
        try:
            ser.write(message_to_send.encode('utf-8'))
            print(f"Pi 4 Sent ->: {message_to_send.strip()}")
        except serial.SerialException as e:
            print(f"!!! 送信エラー: {e}")
            break

        # --- 2. Pico からの返信を待機 ---
        # ser.readline()は、改行コードが来るまで待機 (timeout=1秒)
        try:
            response = ser.readline()
            if response:
                print(f"Pico Replied <-: {response.decode('utf-8').strip()}")
            else:
                print("Pico Replied <-: (タイムアウト: 返信なし)")
        except serial.SerialException as e:
            print(f"!!! 受信エラー: {e}")
            break

        time.sleep(2) # 2秒待機

except serial.SerialException as e:
    print(f"\n!!! エラー: {SERIAL_PORT} を開けません。{e}")
    print("Ubuntuのステップ1(cmdline.txt編集)が正しく完了しているか、")
    print("または 'sudo usermod -aG dialout $USER' (要再起動) が必要か確認してください。")
except KeyboardInterrupt:
    print("\n--- ユーザーにより中断されました。 ---")
finally:
    if ser and ser.is_open:
        ser.close()
        print(f"--- シリアルポート ({SERIAL_PORT}) を閉じました。 ---")