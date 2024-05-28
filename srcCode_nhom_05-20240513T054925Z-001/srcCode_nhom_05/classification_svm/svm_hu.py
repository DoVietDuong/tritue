import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np


# Đọc dữ liệu từ file CSV
data = pd.read_csv("CuoiKy\\feature_hu\\HumomentsFeature_Nhom_05.csv")

# Gộp 3 loại thành 2 loại
# Giả sử bạn muốn gộp nhãn 12 và 14 thành một nhãn mới, ví dụ: 1
# data['Label'] = data['Label'].replace({15: 12})
# data = data[data['Label'] != 14]
# Tách features và labels
X = data.iloc[:, 2:].values
y = data.iloc[:, 1].values

# Số folds cho cross-validation
k = 10

# Tạo đối tượng KFold
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Lưu kết quả dự đoán cho mỗi lần lặp
all_predictions = []
all_indices = []

conf_matrix_sum = np.zeros((len(np.unique(y)), len(np.unique(y))))
# Duyệt qua các folds
for fold, indices in enumerate(kf.split(X)):

    train_index, test_index = indices
    # Tách dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Chuẩn hóa Z-score
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # Tạo mô hình SVM
    svm = SVC(kernel='linear')

    # Huấn luyện mô hình
    svm.fit(X_train_std, y_train)

    # Dự đoán trên tập kiểm tra
    y_pred = svm.predict(X_test_std)

    # Tính toán confusion matrix
    conf_matrix_fold = confusion_matrix(
        y_test, y_pred, labels=np.unique(y))
    conf_matrix_sum += conf_matrix_fold

    # In kết quả dự đoán cho mỗi mẫu trong tập kiểm tra
    print(f'Fold {fold+1} Predictions:')

    for i in range(len(test_index)):
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
