# Import các thư viện cần thiết
import numpy as np  # Thư viện hỗ trợ thao tác với mảng số học
import tensorflow as tf  # Thư viện xây dựng và huấn luyện mô hình học sâu
from tensorflow.keras.models import load_model  # Hàm để tải mô hình đã huấn luyện
import matplotlib.pyplot as plt  # Thư viện để vẽ biểu đồ và đồ thị
from sklearn.metrics import confusion_matrix  # Hàm tính ma trận nhầm lẫn
import seaborn as sns  # Thư viện vẽ biểu đồ nâng cao (heatmap)
import gzip  # Thư viện hỗ trợ giải nén file định dạng .gz

# Thiết lập để hiển thị tiếng Việt trên biểu đồ
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# Hàm tải và chuẩn bị dữ liệu
def load_and_prepare_data(train_images_path, train_labels_path, test_images_path, test_labels_path):
    # Hàm tải ảnh từ file .gz
    def load_images(filepath):
        with gzip.open(filepath, 'rb') as f:
            f.read(16)  # Bỏ qua phần header dài 16 byte
            buffer = f.read()  # Đọc toàn bộ dữ liệu
            images = np.frombuffer(buffer, dtype=np.uint8)  # Chuyển đổi dữ liệu thành mảng uint8
            images = images.reshape(-1, 28, 28)  # Định dạng lại thành ảnh 28x28
        return images

    # Hàm tải nhãn từ file .gz
    def load_labels(filepath):
        with gzip.open(filepath, 'rb') as f:
            f.read(8)  # Bỏ qua phần header dài 8 byte
            buffer = f.read()  # Đọc toàn bộ dữ liệu
            labels = np.frombuffer(buffer, dtype=np.uint8)  # Chuyển đổi dữ liệu thành mảng uint8
        return labels

    # Tải và chuẩn hóa dữ liệu ảnh và nhãn
    test_images = load_images(test_images_path) / 255.0  # Chia 255 để chuẩn hóa ảnh về [0, 1]
    test_labels = load_labels(test_labels_path)

    return test_images.reshape(-1, 28, 28, 1), test_labels  # Định dạng lại dữ liệu ảnh

# Hàm tính toán các chỉ số đánh giá thủ công
def calculate_metrics_manual(model, test_images, test_labels):
    # Lấy dự đoán từ mô hình
    predictions = model.predict(test_images)  # Dự đoán xác suất các lớp
    predicted_labels = np.argmax(predictions, axis=1)  # Chuyển xác suất thành nhãn dự đoán

    # Tổng số lượng mẫu
    total_samples = len(test_labels)

    # Độ chính xác (Accuracy)
    correct_predictions = np.sum(predicted_labels == test_labels)  # Số dự đoán đúng
    accuracy = correct_predictions / total_samples  # Tính độ chính xác

    # Tạo ma trận nhầm lẫn
    num_classes = len(np.unique(test_labels))  # Số lớp
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)  # Ma trận nhầm lẫn rỗng
    for true_label, predicted_label in zip(test_labels, predicted_labels):
        confusion_matrix[true_label, predicted_label] += 1  # Tăng số lượng tương ứng trong ma trận

    # Precision, Recall, F1-score (Weighted)
    precision_list, recall_list, f1_list = [], [], []  # Danh sách lưu chỉ số của từng lớp
    for i in range(num_classes):
        TP = confusion_matrix[i, i]  # True Positives
        FP = np.sum(confusion_matrix[:, i]) - TP  # False Positives
        FN = np.sum(confusion_matrix[i, :]) - TP  # False Negatives

        # Tránh chia cho 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precision_list.append(precision)  # Thêm precision của lớp i
        recall_list.append(recall)  # Thêm recall của lớp i
        f1_list.append(f1)  # Thêm F1-score của lớp i

    # Tính trung bình có trọng số
    weights = [np.sum(test_labels == i) / total_samples for i in range(num_classes)]  # Trọng số
    weighted_precision = np.sum(np.array(precision_list) * weights)
    weighted_recall = np.sum(np.array(recall_list) * weights)
    weighted_f1 = np.sum(np.array(f1_list) * weights)

    return {
        'accuracy': accuracy,  # Độ chính xác
        'precision': weighted_precision,  # Precision trung bình
        'recall': weighted_recall,  # Recall trung bình
        'f1': weighted_f1  # F1-score trung bình
    }

# Hàm vẽ biểu đồ so sánh các chỉ số
def plot_metrics_comparison(metrics_original, metrics_optimized):
    metrics = ['Độ chính xác (Accuracy)', 'Độ chuẩn xác (Precision)', 'Độ nhạy (Recall)', 'Điểm F1 (F1-Score)']
    original_values = [metrics_original['accuracy'], metrics_original['precision'], metrics_original['recall'], metrics_original['f1']]
    optimized_values = [metrics_optimized['accuracy'], metrics_optimized['precision'], metrics_optimized['recall'], metrics_optimized['f1']]

    x = np.arange(len(metrics))  # Tọa độ x
    width = 0.35  # Độ rộng các cột

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width / 2, original_values, width, label='Mô hình gốc', color='skyblue')  # Cột của mô hình gốc
    rects2 = ax.bar(x + width / 2, optimized_values, width, label='Mô hình tối ưu', color='lightgreen')  # Cột của mô hình tối ưu

    ax.set_ylabel('Điểm số')  # Nhãn trục y
    ax.set_title('So sánh hiệu suất giữa hai mô hình')  # Tiêu đề
    ax.set_xticks(x)  # Tọa độ trục x
    ax.set_xticklabels(metrics)  # Nhãn trục x
    ax.legend()  # Hiển thị chú thích

    # Ghi giá trị lên từng cột
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',  # Giá trị được hiển thị
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)  # Ghi giá trị cột mô hình gốc
    autolabel(rects2)  # Ghi giá trị cột mô hình tối ưu

    plt.tight_layout()
    plt.show()

# Hàm vẽ ma trận nhầm lẫn
def plot_confusion_matrices(model_original, model_optimized, test_images, test_labels):
    pred_original = np.argmax(model_original.predict(test_images), axis=1)  # Dự đoán của mô hình gốc
    pred_optimized = np.argmax(model_optimized.predict(test_images), axis=1)  # Dự đoán của mô hình tối ưu

    cm_original = confusion_matrix(test_labels, pred_original)  # Ma trận nhầm lẫn - mô hình gốc
    cm_optimized = confusion_matrix(test_labels, pred_optimized)  # Ma trận nhầm lẫn - mô hình tối ưu

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))  # Tạo 2 biểu đồ cạnh nhau

    sns.heatmap(cm_original, annot=True, fmt='d', ax=ax1, cmap='Blues')  # Heatmap mô hình gốc
    ax1.set_title('Ma trận nhầm lẫn - Mô hình gốc')
    ax1.set_xlabel('Dự đoán')
    ax1.set_ylabel('Thực tế')

    sns.heatmap(cm_optimized, annot=True, fmt='d', ax=ax2, cmap='Blues')  # Heatmap mô hình tối ưu
    ax2.set_title('Ma trận nhầm lẫn - Mô hình tối ưu')
    ax2.set_xlabel('Dự đoán')
    ax2.set_ylabel('Thực tế')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_images, test_labels = load_and_prepare_data(
        'D:/Coding/lab8-mnist/mnist_dataset/train-images-idx3-ubyte.gz',
        'D:/Coding/lab8-mnist/mnist_dataset/train-labels-idx1-ubyte.gz',
        'D:/Coding/lab8-mnist/mnist_dataset/t10k-images-idx3-ubyte.gz',
        'D:/Coding/lab8-mnist/mnist_dataset/t10k-labels-idx1-ubyte.gz'
    )

    model_original = load_model('./Models/MnistCNN.keras')
    model_optimized = load_model('./Models/OptimizationMnistCNN.keras')

    metrics_original = calculate_metrics_manual(model_original, test_images, test_labels)
    metrics_optimized = calculate_metrics_manual(model_optimized, test_images, test_labels)

    print("\nChỉ số đánh giá mô hình gốc:", metrics_original)
    print("\nChỉ số đánh giá mô hình tối ưu:", metrics_optimized)

    plot_metrics_comparison(metrics_original, metrics_optimized)
    plot_confusion_matrices(model_original, model_optimized, test_images, test_labels)
