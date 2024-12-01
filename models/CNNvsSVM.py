# Import các thư viện cần thiết
import matplotlib.pyplot as plt  # Thư viện vẽ biểu đồ
from sklearn.metrics import accuracy_score  # Hàm tính độ chính xác
from keras.models import load_model  # Hàm load model đã lưu
import pickle  # Thư viện để lưu và tải các đối tượng Python
import numpy as np  # Thư viện xử lý mảng số học
from keras.datasets import mnist  # Bộ dữ liệu MNIST được tích hợp sẵn

# 1. Tải dữ liệu MNIST
# Hàm load_data() tự động chia dữ liệu thành tập huấn luyện và kiểm tra
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Chuẩn bị dữ liệu kiểm tra:
# - Reshape lại để phù hợp với input của model (thêm chiều cho kênh màu)
# - Chuyển sang kiểu float32 và chuẩn hóa giá trị pixel về khoảng 0-1
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Chuyển nhãn sang dạng one-hot encoding (vector nhị phân)
y_test_one_hot = np.eye(10)[y_test]

# 2. Tải và đánh giá model CNN
# Load model CNN đã được huấn luyện trước đó
cnn_model = load_model('Models/OptimizationMnistCNN.keras')

# Dự đoán nhãn cho tập kiểm tra
cnn_predictions = cnn_model.predict(x_test)

# Tính độ chính xác của mô hình CNN
# np.argmax để chuyển từ one-hot về nhãn gốc để so sánh
cnn_accuracy = accuracy_score(y_test, np.argmax(cnn_predictions, axis=1))

# 3. Tải và đánh giá model SVM
# Mở file model SVM đã lưu bằng pickle
with open('Models/svm_mnist.pkl', 'rb') as file:
    svm_model = pickle.load(file)

# Chuẩn bị dữ liệu cho SVM:
# Flatten ảnh từ 2D (28x28) thành 1D để phù hợp với input SVM
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# Dự đoán nhãn bằng mô hình SVM
svm_predictions = svm_model.predict(x_test_flat)

# Tính độ chính xác của mô hình SVM
svm_accuracy = accuracy_score(y_test, svm_predictions)

# 4. In kết quả so sánh độ chính xác
print(f"Độ chính xác của CNN: {cnn_accuracy * 100:.2f}%")
print(f"Độ chính xác của SVM: {svm_accuracy * 100:.2f}%")

# 5. Vẽ biểu đồ so sánh độ chính xác
# Định nghĩa nhãn và giá trị độ chính xác
labels = ['CNN', 'SVM']
accuracies = [cnn_accuracy, svm_accuracy]

# Tạo biểu đồ cột so sánh
plt.bar(labels, accuracies, color=['blue', 'green'])
plt.xlabel('Model')  # Nhãn trục x
plt.ylabel('Accuracy')  # Nhãn trục y
plt.title('Comparison of Model Accuracy')  # Tiêu đề biểu đồ
plt.ylim(0, 1)  # Giới hạn trục y từ 0 đến 1

# Thêm nhãn giá trị độ chính xác lên từng cột
plt.text(0, cnn_accuracy, f"{cnn_accuracy*100:.2f}%", ha='center', va='bottom')
plt.text(1, svm_accuracy, f"{svm_accuracy*100:.2f}%", ha='center', va='bottom')

# Lưu biểu đồ thành file ảnh
plt.savefig('accuracy_comparison.png')

# Hiển thị biểu đồ
plt.show()