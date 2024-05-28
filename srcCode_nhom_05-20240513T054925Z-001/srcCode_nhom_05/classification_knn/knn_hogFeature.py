import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Đọc dữ liệu từ file CSV
data = pd.read_csv('D:\\HoangCode_ChieuT2\\CuoiKy\\Buoc3\\feature_hog\\HogFeatures_nhom_05.csv')

# Loại bỏ cột không phù hợp (tên file ảnh)
data = data.drop('Image_Name', axis=1)

# Tách features và labels
X = data.iloc[:, 1:]  # Bỏ cột Label để lấy features
y = data.iloc[:, 0]   # Cột Label

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Số lượng fold cho cross-validation và số láng giềng
num_folds = 5
n_neighbors = 3

# Khởi tạo model KNN
knn = KNeighborsClassifier(n_neighbors=n_neighbors)

# Khởi tạo k-fold cross-validation
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Lists để lưu trữ các điểm đánh giá
accuracy_list = []
precision_list = []
recall_list = []
f1_score_list = []

# Confusion matrix trung bình qua các fold
conf_matrix_sum = np.zeros((len(np.unique(y)), len(np.unique(y))))

# Dự đoán kết quả sử dụng k-fold cross-validation
for fold, (train_index, test_index) in enumerate(kf.split(X_scaled)):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit model trên tập huấn luyện
    knn.fit(X_train, y_train)

    # Dự đoán trên tập kiểm tra
    y_pred = knn.predict(X_test)

    # Tính toán confusion matrix
    conf_matrix_fold = confusion_matrix(y_test, y_pred)
    conf_matrix_sum += conf_matrix_fold

    # Tính toán các điểm đánh giá
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Lưu trữ các điểm đánh giá vào list
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_score_list.append(f1)

    # In kết quả dự đoán cho mỗi lần dự đoán trong fold
    print(f'Fold {fold+1} Predictions:')
    for i in range(len(y_pred)):
        print(f'Predicted label: {y_pred[i]} - True label: {y_test.iloc[i]}')

    print(f'Cross-Validation Results (Accuracy): {accuracy:.2f}')
    print(f'Cross-Validation Results (Precision): {precision:.2f}')
    print(f'Cross-Validation Results (Recall): {recall:.2f}')
    print(f'Cross-Validation Results (F1-score): {f1:.2f}')
    print("________________________________________ \n")

# In các điểm đánh giá trung bình qua các fold
print(f'Mean Accuracy: {np.mean(accuracy_list):.2f}')
print(f'Mean Precision: {np.mean(precision_list):.2f}')
print(f'Mean Recall: {np.mean(recall_list):.2f}')
print(f'Mean F1 Score: {np.mean(f1_score_list):.2f}')


# Biểu đồ hiệu suất của mô hình trên các fold
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Performance Metrics Across Folds', fontsize=16)

sns.lineplot(x=range(1, num_folds+1), y=accuracy_list, ax=ax[0, 0], marker='o')
ax[0, 0].set_title('Accuracy per Fold')
ax[0, 0].set_xlabel('Fold Number')
ax[0, 0].set_ylabel('Accuracy')

sns.lineplot(x=range(1, num_folds+1), y=precision_list, ax=ax[0, 1], marker='o')
ax[0, 1].set_title('Precision per Fold')
ax[0, 1].set_xlabel('Fold Number')
ax[0, 1].set_ylabel('Precision')

sns.lineplot(x=range(1, num_folds+1), y=recall_list, ax=ax[1, 0], marker='o')
ax[1, 0].set_title('Recall per Fold')
ax[1, 0].set_xlabel('Fold Number')
ax[1, 0].set_ylabel('Recall')

sns.lineplot(x=range(1, num_folds+1), y=f1_score_list, ax=ax[1, 1], marker='o')
ax[1, 1].set_title('F1 Score per Fold')
ax[1, 1].set_xlabel('Fold Number')
ax[1, 1].set_ylabel('F1 Score')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])


conf_matrix_avg = conf_matrix_sum / num_folds

# # Biểu đồ confusion matrix trung bình
# fig, ax = plt.subplots(figsize=(8, 6))
# plt.imshow(conf_matrix_avg, cmap='Blues')
# plt.colorbar()
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.show()

# Trung bình confusion matrix và hiển thị bằng Seaborn
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix_avg, annot=True, fmt=".2f", cmap="Blues", cbar=False,
            xticklabels=[f'Predicted {label}' for label in np.unique(y)],
            yticklabels=[f'True {label}' for label in np.unique(y)])
plt.title('Average Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


