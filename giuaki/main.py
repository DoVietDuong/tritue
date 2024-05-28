import cv2
import numpy as np

# Đọc ảnh từ tập tin
image_path = "C:\\Users\\ADMIN\\Desktop\\trituenhantao\\giuaki\\im0.jpeg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Tìm contours của vùng màu trắng
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Tính diện tích vùng màu trắng
white_area = cv2.countNonZero(image)
print("Diện tích vùng màu trắng:", white_area)

# Tính trọng tâm của vùng màu trắng
moments = cv2.moments(contours[0])
centroid_x = int(moments["m10"] / moments["m00"])
centroid_y = int(moments["m01"] / moments["m00"])
print("Vị trí trọng tâm (x, y):", centroid_x, centroid_y)

# Tính moment trung tâm
central_moments = cv2.moments(contours[0], True)
print("Moment trung tâm:")
print("m11:", central_moments["mu11"])
print("m20:", central_moments["mu20"])
print("m02:", central_moments["mu02"])
print("m30:", central_moments["mu30"])
print("m03:", central_moments["mu03"])
print("m21:", central_moments["mu21"])
print("m12:", central_moments["mu12"])

# Tính moment chuẩn hoá
normalized_moments = cv2.moments(contours[0], True)
print("Moment chuẩn hoá:")
print("M11:", normalized_moments["nu11"])
print("M20:", normalized_moments["nu20"])
print("M02:", normalized_moments["nu02"])
print("M30:", normalized_moments["nu30"])
print("M03:", normalized_moments["nu03"])
print("M21:", normalized_moments["nu21"])
print("M12:", normalized_moments["nu12"])

# Tính moment Hu
hu_moments = cv2.HuMoments(moments)
print("Moment Hu:")
print("S1:", hu_moments[0][0])
print("S2:", hu_moments[1][0])
print("S3:", hu_moments[2][0])
print("S4:", hu_moments[3][0])
print("S5:", hu_moments[4][0])
print("S6:", hu_moments[5][0])
print("S7:", hu_moments[6][0])
