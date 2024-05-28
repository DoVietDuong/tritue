import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Đọc dữ liệu từ file CSV
data = pd.read_csv('CuoiKy\\feature_hu\\HumomentsFeature_Nhom_05.csv')

# Tách features và labels
X = data.iloc[:, 2:].values  # Lấy các cột từ cột thứ 3 trở đi làm features
y = data['Label'].values     # Lấy cột label làm nhãn

# Số lượng fold cho cross-validation
num_folds = 10
n_neighbors = 3
# Khởi tạo model KNN với số lân cận là 3 (có thể điều chỉnh)
knn = KNeighborsClassifier(n_neighbors)

# Khởi tạo k-fold cross-validation
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

all_predictions = []
all_indices = []

# Confusion matrix trung bình qua các fold
conf_matrix_sum = np.zeros((len(np.unique(y)), len(np.unique(y))))
# Dự đoán kết quả sử dụng k-fold cross-validation
for fold, indices in enumerate(kf.split(X)):

    train_index, test_index = indices
    # Tách dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Chuẩn hóa Z-score
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # Fit model trên tập huấn luyện
    knn.fit(X_train_std, y_train)

    # Dự đoán trên tập kiểm tra
    y_pred = knn.predict(X_test_std)

    # Tính toán confusion matrix
    conf_matrix_fold = confusion_matrix(
        y_test, y_pred, labels=np.unique(y))
    conf_matrix_sum += conf_matrix_fold

    # In kết quả dự đoán cho mỗi lần dự đoán trong fold
    print(f'Fold {fold+1} Predictions:')
    for i in range(len(y_pred)):
        print(f'Predicted label: {y_pred[i]} - True label: {y_test[i]}')

     # Lưu kết quả dự đoán và chỉ số tương ứng
    all_predictions.extend(y_pred)
    all_indices.extend(test_index)

    # Tính toán và in độ chính xác cho fold hiện tại
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Fold accuracy: {accuracy:.4f}")
    print("____________________ \n\n")

conf_matrix_avg = conf_matrix_sum
plt.imshow(conf_matrix_avg, cmap='Blues')
plt.colorbar()
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(ticks=np.arange(len(np.unique(y))), labels=np.unique(y))
plt.yticks(ticks=np.arange(len(np.unique(y))), labels=np.unique(y))
for i in range(conf_matrix_avg.shape[0]):
    for j in range(conf_matrix_avg.shape[1]):
        plt.text(j, i, format(conf_matrix_avg[i, j], '.0f'),
                 ha="center", va="center", color="white" if conf_matrix_avg[i, j] > conf_matrix_avg.max() / 2. else "black")
plt.show()

# Sắp xếp lại predictions theo thứ tự ban đầu
sorted_predictions = [x for _, x in sorted(zip(all_indices, all_predictions))]
# In Classification Report từ tất cả các dự đoán

print(classification_report(y, sorted_predictions))  # Sử dụng all_predictions
