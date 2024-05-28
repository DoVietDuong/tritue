from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from skimage import io
target_color = [181, 43, 19, 255]


def find_regions_with_color(image, color):
    regions = []
    rows, cols, _ = image.shape
    visited = np.zeros((rows, cols), dtype=bool)

    for i in range(rows):
        for j in range(cols):
            if not visited[i][j] and np.array_equal(image[i][j], color):
                # Tìm vùng chứa màu target_color bằng DFS
                top_left = (j, i)
                bottom_right = (j, i)
                stack = [(i, j)]
                while stack:
                    y, x = stack.pop()
                    visited[y][x] = True
                    top_left = (min(top_left[0], x), min(top_left[1], y))
                    bottom_right = (
                        max(bottom_right[0], x), max(bottom_right[1], y))

                    # Kiểm tra 8 hướng xung quanh
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < cols and 0 <= ny < rows and not visited[ny][nx] and np.array_equal(image[ny][nx], color):
                                stack.append((ny, nx))

                regions.append((top_left, bottom_right))

    return regions


def check_overlap(rect1, rect2):
    # rect1 và rect2 là các tuple (top_left, bottom_right)
    top_left1, bottom_right1 = rect1
    top_left2, bottom_right2 = rect2

    # Kiểm tra xem rect1 có nằm bên trái hoặc bên phải rect2 không
    if bottom_right1[0] < top_left2[0] or top_left1[0] > bottom_right2[0]:
        return False

    # Kiểm tra xem rect1 có nằm phía trên hoặc phía dưới rect2 không
    if bottom_right1[1] < top_left2[1] or top_left1[1] > bottom_right2[1]:
        return False

    # Nếu không thỏa mãn điều kiện trên, tức là hai hình chữ nhật chồng lấn nhau
    return True


def find_nearest_color(pixel, color_array):
    # Tìm màu gần nhất trong color_array với pixel
    min_distance = float('inf')
    nearest_color = None
    for color in color_array:
        distance = np.linalg.norm(pixel - color)
        if distance < min_distance:
            min_distance = distance
            nearest_color = color
    return nearest_color


url = "C:\\Users\\ADMIN\\Desktop\\trituenhantao\\baitap4\\cachua.png"  # first image
img = io.imread(url)

plt.figure(figsize=(15, 15))
plt.subplot(221)
plt.imshow(img)

img_init = img.copy()
img_ori = img.copy()

img = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
k = 5
# "pick out" the K-means tool from our collection of algorithms
clt = KMeans(n_clusters=k)

clt.fit(img)  # apply the model to our data, the image

colorArray = []
for color in clt.cluster_centers_:
    colorArray.append(color.astype("uint8").tolist())

print(colorArray)

for i in range(len(img_init)):
    for j in range(len(img_init[0])):
        pixel = img_init[i][j]
        nearest_color = find_nearest_color(pixel, colorArray)
        img_init[i][j] = nearest_color

# Hiển thị ảnh đã được quantize màu
regions = find_regions_with_color(img_init, target_color)
# Sử dụng hàm combine_overlapping_rectangles với danh sách các hình chữ nhật

combined_rectangle = []

for i, (rect1) in enumerate(regions):
    for j, (rect2) in enumerate(regions[i + 1:]):
        if check_overlap(rect1, rect2):
            # Hai hình chữ nhật chồng lấn nhau
            # Tạo hình chữ nhật lớn hơn bao bọc cả hai
            combined_top_left = (
                min(rect1[0][0], rect2[0][0]), min(rect1[0][1], rect2[0][1]))
            combined_bottom_right = (
                max(rect1[1][0], rect2[1][0]), max(rect1[1][1], rect2[1][1]))
            combined_rectangle.append(
                (combined_top_left, combined_bottom_right))


# Vẽ khung hình chữ nhật cho các vùng đã tìm thấy
for top_left, bottom_right in combined_rectangle:
    cv2.rectangle(img_ori, top_left, bottom_right, (0, 0, 255), thickness=3)

# Hiển thị ảnh
plt.subplot(222)
plt.imshow(img_init)
plt.subplot(223)
plt.imshow(img_ori)
plt.show()