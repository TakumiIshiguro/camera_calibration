import cv2
import numpy as np
import os

calibration_file = "calibration_data.npz"
if not os.path.exists(calibration_file):
    exit()

calibration_data = np.load(calibration_file)
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

folder_path = "/home/takumi/Pictures/Webcam"

images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

for image_path in images:
    img = cv2.imread(image_path)
    if img is None:
        print(f"失敗")
        continue

    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    cv2.imshow('Original Image', img)
    cv2.imshow('Undistorted Image', undistorted_img)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

