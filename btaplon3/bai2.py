import cv2
import numpy as np
from sklearn.cluster import KMeans

# Đọc ảnh
img_path = "C:\\Users\\ADMIN\\Desktop\\trituenhantao\\btaplon3\\a.png"
image = cv2.imread(img_path, 1)  # Use 1 for loading the image in color

# Reshape hình ảnh thành ma trận 2D
pixels = image.reshape(-1, 3)

# Sử dụng K-means để phân cụm các pixel thành 3 nhóm màu
kmeans = KMeans(n_clusters=3)
kmeans.fit(pixels)

# Lấy ra các nhãn cho từng pixel
labels = kmeans.labels_

# Tìm label của nhóm có giá trị trung bình cao nhất 
counts = np.bincount(labels)
dominant_color_label = np.argmax(counts)

# Tạo mask cho các pixel thuộc nhóm có label giống nhãn của nhóm màu chính
mask = (labels == dominant_color_label).reshape(image.shape[0], image.shape[1]).astype(np.uint8)

# Áp dụng mask để giữ lại bông hoa, còn lại là nền đen
flower_only = cv2.bitwise_and(image, image, mask=mask)

# Hiển thị hình ảnh bông hoa và nền đen
cv2.imshow('Flower Only', flower_only)
cv2.waitKey(0)
cv2.destroyAllWindows()
