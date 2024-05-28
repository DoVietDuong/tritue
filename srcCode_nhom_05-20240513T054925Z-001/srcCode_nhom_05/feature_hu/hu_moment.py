#############################################################################
#                                                                           #
# Author: Team 5                                                            #
#                                                                           #
# Description:                                                              #
#  - File này tính toán đặc trưng Humoments trên ảnh nhị phân.              #
#  - Tính toán và gắn nhãn cho từng vector đặc trưng theo tên thư mục mà    #
#   ảnh đó thuộc về.                                                        #
#  - Đầu ra được lưu tại file HumomentsFeature_Nhom_05.csv                  #
#                                                                           #
#############################################################################

import cv2
import os
import csv


def calculate_humoments(image):
    # Tính toán các đặc trưng humoments
    _, image = cv2.threshold(
        image, 128, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(image)
    humoments = cv2.HuMoments(moments).flatten()

    return humoments


def process_images_in_folder(folder_path, label, csv_writer):
    # Lặp qua tất cả các tệp trong thư mục
    for filename in os.listdir(folder_path):
        # Đường dẫn đầy đủ đến tệp
        file_path = os.path.join(folder_path, filename)

        # Đọc ảnh từ tệp
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        _, image = cv2.threshold(
            image, 128, 255, cv2.THRESH_BINARY)

        height, width = image.shape[:2]
        image = cv2.resize(image, (width // 3, height // 3))
        # cv2.imshow('Resized Image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # Tính toán humoments cho ảnh
        humoments = calculate_humoments(image)

        # Ghi dòng vào tệp CSV
        row = [filename, label] + list(humoments)
        csv_writer.writerow(row)


# Đường dẫn tới thư mục Leaf/binary (thư mục ảnh trắng đen)
leaf_folder = "CuoiKy\\Leafs\\binary"

# Tên của tệp CSV để lưu kết quả
csv_file = "CuoiKy\\feature_hu\\HumomentsFeature_Nhom_05.csv"

# Mở tệp CSV để ghi
with open(csv_file, 'w', newline='') as file:
    csv_writer = csv.writer(file)
    # Ghi tiêu đề cho các cột
    csv_writer.writerow(["Image", "Label", "Hu1", "Hu2",
                        "Hu3", "Hu4", "Hu5", "Hu6", "Hu7"])

    # Lặp qua các thư mục con trong thư mục Leaf
    for folder_name in ["12", "14", "15"]:
        folder_path = os.path.join(leaf_folder, folder_name)
        # Xử lý ảnh trong thư mục và ghi kết quả vào tệp CSV
        process_images_in_folder(folder_path, folder_name, csv_writer)
