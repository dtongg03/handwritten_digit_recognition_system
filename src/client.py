import numpy as np
import warnings
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import *
from PIL import ImageGrab, Image, ImageDraw
import io

warnings.filterwarnings("ignore")

# Kích thước cửa sổ và tên ảnh
width = 800
height = 600
image_size = (28, 28)


class HandwritingRecognizer:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.setup_gui()
        self.drawing = False
        self.last_x = None
        self.last_y = None
        # Tạo ảnh trong bộ nhớ thay vì lưu file
        self.image = Image.new('RGB', (400, 400), 'black')
        self.draw = ImageDraw.Draw(self.image)

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Handwriting Recognition")

        # Canvas tổng quát
        self.main_canvas = Canvas(self.root, width=width, height=height, bg="#AFBFF5")
        self.main_canvas.pack()

        # Khung vẽ
        self.frame_draw = Frame(self.root)
        self.frame_draw.place(relx=0.05, rely=0.3, relwidth=0.4, relheight=0.53)

        # Canvas vẽ với nền đen
        self.draw_canvas = Canvas(self.frame_draw, bg="black")
        self.draw_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)

        # Bind các sự kiện chuột
        self.draw_canvas.bind("<Button-1>", self.start_drawing)
        self.draw_canvas.bind("<B1-Motion>", self.draw)
        self.draw_canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        # Khung hiển thị kết quả
        self.frame_output = Frame(self.root)
        self.frame_output.place(relx=0.55, rely=0.3, relwidth=0.4, relheight=0.53)

        # Các nút và nhãn
        self.setup_buttons()
        self.setup_labels()

    def setup_buttons(self):
        classify_button = Button(self.root, text="Classify", command=self.collect_image, bg="#D2D2D2")
        classify_button.place(relx=0.05, rely=0.2, relwidth=0.2, relheight=0.1)

        clear_button = Button(self.root, text="Clear", command=self.clear_frame, bg="#D2D2D2")
        clear_button.place(relx=0.25, rely=0.2, relwidth=0.2, relheight=0.1)

    def setup_labels(self):
        label_prediction = Label(self.root, text="Model Prediction", bg="#D2D2D2")
        label_prediction.place(relx=0.55, rely=0.2, relwidth=0.4, relheight=0.1)

        label_title = Label(self.root, text="HTR system by Team 6", bg="#AFBFF5")
        label_title.config(font=("Times", 20))
        label_title.place(relx=0.25, rely=0.05, relwidth=0.5, relheight=0.1)

    def start_drawing(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def draw(self, event):
        if self.drawing:
            pen_color = "#FFFFFF"  # Màu trắng
            pen_size = 15  # Tăng kích thước bút
            x1, y1 = (event.x - pen_size), (event.y - pen_size)
            x2, y2 = (event.x + pen_size), (event.y + pen_size)

            # Vẽ trên canvas
            self.draw_canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                         fill=pen_color, width=pen_size * 2, capstyle=tk.ROUND, smooth=True)

            # Vẽ trên ảnh trong bộ nhớ
            self.draw.line([self.last_x, self.last_y, event.x, event.y],
                           fill="white", width=pen_size * 2)

            self.last_x = event.x
            self.last_y = event.y

    def stop_drawing(self, event):
        self.drawing = False

    def preprocess_image(self):
        # Lấy kích thước thực của canvas
        canvas_width = self.draw_canvas.winfo_width()
        canvas_height = self.draw_canvas.winfo_height()

        # Tạo ảnh mới với kích thước thực của canvas
        img = Image.new('RGB', (canvas_width, canvas_height), 'black')
        draw = ImageDraw.Draw(img)

        # Sao chép tất cả các đối tượng từ canvas sang ảnh mới
        for item in self.draw_canvas.find_all():
            coords = self.draw_canvas.coords(item)
            if len(coords) >= 4:  # Nếu là line hoặc oval
                draw.line(coords, fill='white', width=15)

        # Chuyển đổi sang ảnh xám và resize
        img = img.convert('L')
        img = img.resize(image_size, Image.Resampling.LANCZOS)

        # Chuyển đổi thành mảng numpy và tiền xử lý
        image_array = np.array(img)
        image_array = image_array / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)

        return image_array

    def collect_image(self):
        processed_image = self.preprocess_image()
        predicted_digit = self.prediction(processed_image)
        self.print_prediction(predicted_digit)

    def prediction(self, image_matrix):
        pred = self.model.predict(image_matrix)[0]
        return np.argmax(pred)

    def print_prediction(self, prediction):
        for widget in self.frame_output.winfo_children():
            widget.destroy()
        mapping = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                   10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
                   20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
                   30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'c', 39: 'd',
                   40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j', 46: 'k', 47: 'l', 48: 'm', 49: 'n',
                   50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u', 57: 'v', 58: 'w', 59: 'x',
                   60: 'y', 61: 'z', 62: '`', 63: '~', 64: '!', 65: '@', 66: '#', 67: '$', 68: '%', 69: '^',
                   70: '&', 71: '*', 72: '(', 73: ')', 74: '-', 75: '_', 76: '+', 77: '=', 78: '[', 79: ']',
                   80: '{', 81: '}', 82: ';', 83: ':', 84: '"', 85: "'", 86: ',', 87: '.', 88: '/', 89: '?',
                   90: '\\', 91: '|'}
        predicted_char = mapping[prediction]
        myLabel = Label(self.frame_output, text=predicted_char)
        myLabel.config(font=("Times", 200))
        myLabel.place(relx=0.2, rely=0.15, relwidth=0.7, relheight=0.6)

    def clear_frame(self):
        self.draw_canvas.delete("all")
        self.image = Image.new('RGB', (400, 400), 'black')
        self.draw = ImageDraw.Draw(self.image)

    def run(self):
        self.root.mainloop()


# Khởi tạo và chạy ứng dụng
app = HandwritingRecognizer('D:/Coding/Handwriten_recognition/.venv/models/Models/OptimizationMnistCNN.keras')
app.run()