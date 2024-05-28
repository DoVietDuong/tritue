#############################################################################
#                                                                           #
# Author: Team 5                                                            #
#                                                                           #
# Description:                                                              #
#  - File này tính toán đặc trưng HOG trên ảnh xám.                         #
#  - Tính toán và gắn nhãn cho từng vector đặc trưng theo tên thư mục mà    #
#   ảnh đó thuộc về.                                                        #
#  - Đầu ra được lưu tại file hogFeature_Nhom_05.csv                        #
#                                                                           #
#############################################################################

import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from os import listdir
from os.path import isfile, join

# HOG parameters
cell_size = (200, 200)
block_size = (2, 2)
nbins = 9

# Function to compute HOG features for an image


def compute_hog(image):
    hog_features, _ = hog(image, orientations=nbins, pixels_per_cell=cell_size,
                          cells_per_block=block_size, block_norm='L2-Hys',
                          visualize=True, transform_sqrt=True)
    return hog_features


# Path to the Leaf folder containing subfolders 12, 14, 15
leaf_folder = "CuoiKy\\Leafs\\gray"
feature_dimension = 7
# Function to process images in a folder


def process_images_in_folder(folder_path, label):
    image_files = [f for f in listdir(
        folder_path) if isfile(join(folder_path, f))]
    data = []
    for image_file in image_files:
        image_path = join(folder_path, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        height, width = image.shape[:2]
        image = cv2.resize(image, (width // 3, height // 3))

        print("image size:", image.shape)
        hog_features = compute_hog(image)
        row = [image_file, label] + hog_features[:feature_dimension].tolist()
        data.append(row)
    return data

# Main function to process all subfolders


def main():
    all_data = []
    for subfolder in ["12", "14", "15"]:
        subfolder_path = join(leaf_folder, subfolder)
        label = subfolder
        data = process_images_in_folder(subfolder_path, label)
        all_data.extend(data)

    # Create DataFrame
    df = pd.DataFrame(all_data, columns=[
                      "Image_Name", "Label"] + [f"HOG_{i}" for i in range(1, feature_dimension+1)])

    # Save DataFrame to CSV
    df.to_csv("CuoiKy\\feature_hog\\hogFeatures_nhom_05.csv", index=False)


if __name__ == "__main__":
    main()
