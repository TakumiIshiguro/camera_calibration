import cv2
import numpy as np
import glob
import os

CHESSBOARD_SIZE = (10, 7)
SQUARE_SIZE = 25.0

objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

obj_points = []  
img_points = [] 

folder_path = "/home/takumi/Pictures/Webcam"  

images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

if len(images) < 10:
    print(f"画像が不足しています。10枚の画像が必要ですが、{len(images)}枚しか見つかりませんでした。")
    exit()

for image_path in images:
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
    if ret:
        obj_points.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                                     criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        img_points.append(corners2)

        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
        cv2.imshow('Detected Corners', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

if len(obj_points) > 0:
    print("開始")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    if ret:
        print("成功")
        print("カメラ行列:")
        print(camera_matrix)
        print("歪み係数:")
        print(dist_coeffs)

        np.savez("calibration_data.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        print("保存")
    else:
        print("失敗")
else:
    print("データが悪い")

