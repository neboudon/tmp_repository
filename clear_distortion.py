import cv2
import numpy as np
 
class ImageUndistortion:
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs #カメラがどのように歪んでいるかを示す
 
    def undistort_image(self, image):#カメラの歪みを取り除く
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 0.1, (w, h))
        #roiは有効なピクセル領域を示す矩形
        dst = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
        #undistort関数は画像の歪みを補正するために使用される
        x, y, w, h = roi
        if all(v > 0 for v in [x, y, w, h]):
            dst = dst[y:y+h, x:x+w]
        return dst
    
    def process_image(self, image=None, image_path=None):
        """
        画像を処理して、その歪みを補正します。

        :param image: 入力する歪んだ画像（NumPy配列形式）。
        :param image_path: 歪んだ画像のファイルパス（任意）。
        :param save_path: 補正後の画像を保存するパス。
        """
        if image is None and image_path is not None:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image from {image_path}")
                return
 
        if image is None:
            print("Error: No image provided")
            return
 
        undistorted_image = self.undistort_image(image)
 
        return undistorted_image
    @staticmethod
    def get_camera_parameters():
        """
        すでに計算されたカメラ行列と歪み係数を返す
        """
        camera_matrix = np.array([[1.51874119e+03, 0.00000000e+00, 1.91845852e+03],
                                [0.00000000e+00, 1.51911954e+03, 1.06247899e+03],
                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        
        dist_coeffs = np.array([-0.2724813, 0.09954433, 0.00044363, -0.00102376, -0.01633367])
        return camera_matrix, dist_coeffs
 
if __name__ == "__main__":
    # Provided camera matrix and distortion coefficients
    camera_matrix = np.array([[1.51874119e+03, 0.00000000e+00, 1.91845852e+03],
                              [0.00000000e+00, 1.51911954e+03, 1.06247899e+03],
                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    
    dist_coeffs = np.array([-0.2724813, 0.09954433, 0.00044363, -0.00102376, -0.01633367])
    
    # Create an instance of the ImageUndistortion class
    undistorter = ImageUndistortion(camera_matrix, dist_coeffs)
    
    # Load the distorted image
    #image_path = '/Users/lukaszbaldyga/Desktop/Mac_Research/17_line_detection_and_unwrappingp/line-detection/subfile/1656.png'
    image_path = '/Users/lukaszbaldyga/Desktop/Mac_Research/25_jikkenno3/K3front/exported_photos_GX08/1656.png'
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
    else:
        # Perform undistortion
        undistorted_image = undistorter.undistort_image(image)
        
        # Display the original and undistorted images side by side
        h, w = image.shape[:2]
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        comparison[:, :w] = image
        comparison[:, w:] = cv2.resize(undistorted_image, (w, h))
        
        # Add labels to the images
        cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "Undistorted", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Save the undistorted image and comparison
        cv2.imwrite("undistorted_image.jpg", undistorted_image)
        cv2.imwrite("comparison_image.jpg", comparison)
        print("Undistorted image saved as 'undistorted_image.jpg'")
        print("Comparison image saved as 'comparison_image.jpg'")
        
        # Show the comparison
        cv2.imshow("Original vs Undistorted", cv2.resize(comparison, (1600, 600)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()