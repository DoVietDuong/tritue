import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Bước 1: Thu thập dữ liệu hình ảnh của lá cây từ các thư mục khác nhau
def load_images_from_folders(folders):
    images = []
    labels = []
    for label, folder in enumerate(folders):
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
                labels.append(label)
    return images, labels

# Đường dẫn đến các thư mục chứa hình ảnh của các loại cây
folder1 = "C:\\Users\\ADMIN\\Desktop\\trituenhantao\\cuoiki\\12-20240426T135217Z-001\\12"
folder2 = "C:\\Users\\ADMIN\\Desktop\\trituenhantao\\cuoiki\\14-20240426T135155Z-001\\14"
folder3 = "C:\\Users\\ADMIN\\Desktop\\trituenhantao\\cuoiki\\15-20240426T135132Z-001\\15"
folders = [folder1, folder2, folder3]

images, labels = load_images_from_folders(folders)

# Bước 2: Chuẩn bị dữ liệu và trích xuất đặc trưng HOG
def preprocess_image(image, target_size=(100, 100)):
    image = cv2.resize(image, target_size)
    return image

def extract_hog_features(images):
    features = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
        features.append(hog_features)
    return features

preprocessed_images = [preprocess_image(img) for img in images]
hog_features = extract_hog_features(preprocessed_images)

# Bước 3: Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(hog_features, labels, test_size=0.2, random_state=42)

# Bước 4: Huấn luyện mô hình
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Bước 5: Đánh giá mô hình và hiển thị kết quả
y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# Tạo một từ điển ánh xạ từ nhãn sang tên của cây
label_to_tree = {
    0: "Pittosporum tobira",
    1: "Chimonanthus praecox",
    2: "Cinnamomum camphora"
}

# Hiển thị dự đoán của mô hình cùng với tên của cây dự đoán
for pred_label, true_label in zip(y_pred, y_test):
    predicted_tree = label_to_tree[pred_label]
    true_tree = label_to_tree[true_label]
    print("Predicted:", predicted_tree, " - True:", true_tree)

print("=> Accuracy:", accuracy)