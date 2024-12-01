# Import các thư viện cần thiết
import numpy as np  # Thư viện xử lý mảng và tính toán số học
import gzip  # Thư viện giải nén các tệp nén
import tensorflow as tf  # Thư viện học máy và deep learning của Google
from tensorflow.keras.models import Sequential  # Mô hình tuần tự để xây dựng neural network
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input  # Các lớp để xây dựng mạng neural
import sys  # Thư viện tương tác với hệ thống
import io  # Thư viện quản lý đầu vào/đầu ra
import matplotlib.pyplot as plt  # Thư viện vẽ biểu đồ

# Cấu hình encoding cho stdout để hỗ trợ hiển thị tiếng Việt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Định nghĩa đường dẫn tới các tệp dữ liệu MNIST
train_images_path = r'D:/Coding/lab8-mnist/mnist_dataset/train-images-idx3-ubyte.gz'  # Tệp ảnh huấn luyện
train_labels_path = r'D:/Coding/lab8-mnist/mnist_dataset/train-labels-idx1-ubyte.gz'  # Tệp nhãn huấn luyện
test_images_path = r'D:/Coding/lab8-mnist/mnist_dataset/t10k-images-idx3-ubyte.gz'  # Tệp ảnh kiểm tra
test_labels_path = r'D:/Coding/lab8-mnist/mnist_dataset/t10k-labels-idx1-ubyte.gz'  # Tệp nhãn kiểm tra

# Hàm tải ảnh từ tệp nén
def load_images(filepath):
    with gzip.open(filepath, 'rb') as f:
        f.read(16)  # Bỏ qua 16 byte đầu tiên của tệp (phần header)
        buffer = f.read()  # Đọc toàn bộ dữ liệu còn lại
        images = np.frombuffer(buffer, dtype=np.uint8)  # Chuyển dữ liệu thành mảng numpy
        images = images.reshape(-1, 28, 28)  # Định dạng lại thành ma trận 2D (28x28)
    return images

# Hàm tải nhãn từ tệp nén
def load_labels(filepath):
    with gzip.open(filepath, 'rb') as f:
        f.read(8)  # Bỏ qua 8 byte đầu tiên của tệp (phần header)
        buffer = f.read()  # Đọc toàn bộ dữ liệu còn lại
        labels = np.frombuffer(buffer, dtype=np.uint8)  # Chuyển dữ liệu thành mảng numpy
    return labels

# Tải và chuẩn hóa dữ liệu (chia giá trị pixel cho 255 để đưa về khoảng 0-1)
train_images = load_images(train_images_path) / 255.0
train_labels = load_labels(train_labels_path)
test_images = load_images(test_images_path) / 255.0
test_labels = load_labels(test_labels_path)

# Xây dựng mô hình Convolutional Neural Network (CNN)
model = Sequential([
    Input(shape=(28, 28, 1)),  # Lớp đầu vào với kích thước ảnh 28x28 pixel, 1 kênh (ảnh xám)
    Conv2D(32, kernel_size=(3, 3), activation='relu'),  # Lớp tích chập thứ nhất, 32 bộ lọc, kích thước 3x3
    MaxPooling2D(pool_size=(2, 2)),  # Lớp max pooling giảm kích thước đặc trưng
    Conv2D(64, kernel_size=(3, 3), activation='relu'),  # Lớp tích chập thứ hai, 64 bộ lọc
    MaxPooling2D(pool_size=(2, 2)),  # Lớp max pooling thứ hai
    Flatten(),  # Chuyển đổi ma trận thành vector 1 chiều
    Dense(128, activation='relu'),  # Lớp kết nối đầy đủ với 128 nút
    Dense(10, activation='softmax')  # Lớp đầu ra với 10 nút (0-9), hàm softmax để phân loại
])

# Biên dịch mô hình với:
# - Bộ tối ưu hóa Adam
# - Hàm mất mát categorical crossentropy
# - Độ đo là accuracy
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình với 40 epoch, sử dụng dữ liệu kiểm tra để đánh giá
history = model.fit(train_images, train_labels, epochs=40, validation_data=(test_images, test_labels))

# Lưu mô hình đã huấn luyện
model.save('./Models/MnistCNN.keras')

# Đánh giá mô hình trên tập kiểm tra
model_loss, model_accuracy = model.evaluate(test_images, test_labels)
print(f"\nmodel Loss: {model_loss}")
print(f"model Accuracy: {model_accuracy}")

# Trích xuất các giá trị accuracy và loss từ quá trình huấn luyện
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Tạo biểu đồ để trực quan hóa kết quả huấn luyện
plt.figure(figsize=(12, 6))

# Biểu đồ độ chính xác
plt.subplot(1, 2, 1)
plt.plot(range(1, 41), accuracy, label='Training Accuracy')
plt.plot(range(1, 41), val_accuracy, label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Biểu đồ loss
plt.subplot(1, 2, 2)
plt.plot(range(1, 41), loss, label='Training Loss')
plt.plot(range(1, 41), val_loss, label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Điều chỉnh bố cục và hiển thị biểu đồ
plt.tight_layout()
plt.show()