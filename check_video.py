import cv2

# カメラを初期化 (0はデフォルトのカメラ)
cap = cv2.VideoCapture(0)

# カメラが正常に開けたかを確認
if not cap.isOpened():
    print("エラー: カメラを開けませんでした。")
    exit()

# --- カメラ設定を行う ---
# 幅を1280ピクセルに設定
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# 高さを720ピクセルに設定
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# フレームレートを30に設定
cap.set(cv2.CAP_PROP_FPS, 30)

# 設定が反映されたか確認するために、現在の設定値を取得して表示
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"現在の解像度: {int(width)} x {int(height)}, FPS: {fps}")


# whileループでカメラのフレームを1枚ずつ読み込む
while True:
    # フレームを1枚読み込む
    ret, frame = cap.read()

    # フレームが正しく読み込めなかった場合は、ループを抜ける
    if not ret:
        print("エラー: フレームを読み込めませんでした。")
        break

    # 'Camera Feed' という名前のウィンドウにフレームを表示
    cv2.imshow('Camera Feed', frame)

    # 'q'キーが押されたらループを抜ける
    # cv2.waitKey(1) は1ミリ秒キー入力を待つ
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# 使用が終わったら、カメラを解放し、すべてのウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()