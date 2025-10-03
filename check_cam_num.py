import cv2

def find_camera_index():
    """
    利用可能なカメラのインデックスを順番に試し、映像を表示する関数。
    'q'キーで次のカメラへ、ウィンドウを閉じると終了します。
    """
    index = 0
    while True:
        # 指定したインデックスでカメラを開こうと試みる
        cap = cv2.VideoCapture(index)

        # カメラが開けなかった場合、これ以上カメラはないと判断して終了
        if not cap.isOpened():
            print(f"インデックス {index} のカメラは見つかりませんでした。")
            print("利用可能なカメラの確認を終了します。")
            break

        print(f"インデックス {index} のカメラをテスト中... ('q'キーを押すと次に進みます)")
        
        # カメラが開けたら、その映像を表示し続ける
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # ウィンドウに映像を表示
            cv2.imshow(f'Camera Index Test: {index}', frame)

            # 'q'キーが押されたら、このカメラのテストを終了して次へ
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 後片付け
        cap.release()
        cv2.destroyAllWindows()
        
        # 次のインデックスへ
        index += 1

if __name__ == '__main__':
    find_camera_index()