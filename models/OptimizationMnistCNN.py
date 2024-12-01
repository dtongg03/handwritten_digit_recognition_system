# Import các thư viện cần thiết
import numpy as np  # Thư viện xử lý mảng và tính toán số học
import gzip  # Thư viện giải nén các tệp nén
import tensorflow as tf  # Thư viện học máy và deep learning của Google
from tensorflow.keras.models import Sequential  # Mô hình tuần tự để xây dựng neural network
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input, Dropout, BatchNormalization  # Các lớp để xây dựng mạng neural
from tensorflow.keras.callbacks import EarlyStopping  # Kỹ thuật dừng sớm để tránh overfitting
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Tăng cường dữ liệu (data augmentation)
import sys  # Thư viện tương tác với hệ thống
import io  # Thư viện quản lý đầu vào/đầu ra
import matplotlib.pyplot as plt  # Thư viện vẽ biểu đồ

# Cấu hình encoding cho stdout để hỗ trợ hiển thị tiếng Việt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Định nghĩa đường dẫn tới các tệp dữ liệu MNIST
train_images_path = 'D:/Coding/lab8-mnist/mnist_dataset/train-images-idx3-ubyte.gz'  # Tệp ảnh huấn luyện
train_labels_path = 'D:/Coding/lab8-mnist/mnist_dataset/train-labels-idx1-ubyte.gz'  # Tệp nhãn huấn luyện
test_images_path = 'D:/Coding/lab8-mnist/mnist_dataset/t10k-images-idx3-ubyte.gz'  # Tệp ảnh kiểm tra
test_labels_path = 'D:/Coding/lab8-mnist/mnist_dataset/t10k-labels-idx1-ubyte.gz'  # Tệp nhãn kiểm tra

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

# Tạo bộ tăng cường dữ liệu (Data Augmentation)
# Các phép biến đổi này giúp tăng tính đa dạng của dữ liệu huấn luyện
datagen = ImageDataGenerator(
    rotation_range=10,  # Xoay ảnh trong khoảng ±10 độ
    width_shift_range=0.1,  # Dịch chuyển ngang ±10% chiều rộng
    height_shift_range=0.1,  # Dịch chuyển dọc ±10% chiều cao
    zoom_range=0.1  # Phóng to/thu nhỏ ±10%
)
# Chuẩn bị bộ sinh dữ liệu
datagen.fit(train_images.reshape(-1, 28, 28, 1))

# Xây dựng mô hình CNN với các kỹ thuật cải tiến
model = Sequential([
    Input(shape=(28, 28, 1)),  # Lớp đầu vào với kích thước ảnh 28x28 pixel, 1 kênh (ảnh xám)

    # Khối tích chập đầu tiên
    Conv2D(64, kernel_size=(3, 3), activation='relu'),  # 64 bộ lọc, kích thước 3x3
    BatchNormalization(),  # Chuẩn hóa batch để ổn định quá trình huấn luyện
    MaxPooling2D(pool_size=(2, 2)),  # Giảm kích thước đặc trưng
    Dropout(0.25),  # Ngăn chặn overfitting bằng cách loại bỏ ngẫu nhiên 25% nút

    # Khối tích chập thứ hai
    Conv2D(128, kernel_size=(3, 3), activation='relu'),  # 128 bộ lọc
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Khối tích chập thứ ba
    Conv2D(256, kernel_size=(3, 3), activation='relu'),  # 256 bộ lọc
    BatchNormalization(),
    Flatten(),  # Chuyển đổi ma trận thành vector 1 chiều

    # Lớp kết nối đầy đủ
    Dense(256, activation='relu'),  # 256 nút
    Dropout(0.5),  # Loại bỏ 50% nút để giảm overfitting
    Dense(10, activation='softmax')  # Lớp đầu ra với 10 nút (0-9), hàm softmax để phân loại
])

# Biên dịch mô hình với:
# - Bộ tối ưu hóa Adam
# - Hàm mất mát categorical crossentropy
# - Độ đo là accuracy
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Thiết lập kỹ thuật dừng sớm (Early Stopping)
# Theo dõi giá trị loss trên tập validation
# Nếu không cải thiện sau 15 epoch, dừng huấn luyện
# Khôi phục trọng số tốt nhất
early_stopping = EarlyStopping(monitor='val_loss', patience=15,
                               restore_best_weights=True)

# Huấn luyện mô hình
# Sử dụng data augmentation để sinh thêm dữ liệu
# Batch size 64 để cân bằng giữa tốc độ và bộ nhớ
history = model.fit(
    datagen.flow(train_images.reshape(-1, 28, 28, 1), train_labels, batch_size=64),
    epochs=40,
    validation_data=(test_images.reshape(-1, 28, 28, 1), test_labels),
    callbacks=[early_stopping]
)

# Lưu mô hình đã huấn luyện
model.save('./Models/OptimizationMnistCNN.keras')

# Đánh giá mô hình trên tập kiểm tra
model_loss, model_accuracy = model.evaluate(test_images.reshape(-1, 28, 28, 1), test_labels)
print(f"\nModel Loss: {model_loss}")
print(f"Model Accuracy: {model_accuracy}")

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