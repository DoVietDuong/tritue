#############################################################################
#                                                                           #
# Author: Team 5                                                            #
#                                                                           #
# Description:                                                              #
#  - File này đọc ảnh gốc từ thư mục "Leafs/original".                      #
#  - Thực hiện chuyển ảnh sang dạng ảnh xám, và ảnh trắng đen.              #
#  - Đầu ra được lưu tại thư mục "Leafs/gray" và "Leafs/binary".            #
#                                                                           #
#############################################################################

import os
import cv2

# Đường dẫn tới thư mục chứa ảnh gốc
original_folder_path = "Leafs\\original"

# Tạo thư mục "gray" và "binary" nếu chưa tồn tại
gray_folder_path = os.path.join(original_folder_path, "..\\gray")
binary_folder_path = os.path.join(original_folder_path, "..\\binary")
os.makedirs(gray_folder_path, exist_ok=True)
os.makedirs(binary_folder_path, exist_ok=True)

# Lặp qua từng thư mục trong thư mục gốc
for folder_name in os.listdir(original_folder_path):
    folder_path = os.path.join(original_folder_path, folder_name)
    if os.path.isdir(folder_path):
        # Lặp qua từng file trong thư mục con
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            # Đọc ảnh
            image = cv2.imread(image_path)
            if image is not None:
                # Chuyển ảnh sang ảnh xám
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Lưu ảnh xám vào thư mục "gray"
                gray_image_path = os.path.join(
                    gray_folder_path, folder_name, filename)
                os.makedirs(os.path.dirname(gray_image_path), exist_ok=True)
                cv2.imwrite(gray_image_path, gray_image)
                # Chuyển ảnh xám sang ảnh nhị phân
                _, binary_image = cv2.threshold(
                    gray_image, 128, 255, cv2.THRESH_BINARY)
                binary_image = cv2.bitwise_not(binary_image)
                # Lưu ảnh nhị phân vào thư mục "binary"
                binary_image_path = os.path.join(
                    binary_folder_path, folder_name, filename)
                os.makedirs(os.path.dirname(binary_image_path), exist_ok=True)
                cv2.imwrite(binary_image_path, binary_image)
