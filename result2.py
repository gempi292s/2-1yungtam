import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
from tensorflow.keras.models import load_model

class ImageSelectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Selection App")

        # 메인 프레임 생성
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack()

        # 왼쪽 프레임 (원본 이미지 및 캔버스)
        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # 이미지 파일 선택 버튼
        self.select_button = tk.Button(self.left_frame, text="Select Image", command=self.load_image)
        self.select_button.pack(pady=10)

        # Canvas 설정
        self.canvas = tk.Canvas(self.left_frame, bg="white", width=400, height=300)
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self.start_selection)
        self.canvas.bind("<B1-Motion>", self.draw_selection)
        self.canvas.bind("<ButtonRelease-1>", self.end_selection)

        # 오른쪽 프레임 (재구성된 이미지 및 버튼)
        self.right_frame = tk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # 이미지 라벨 설정
        self.image_label = tk.Label(self.right_frame)
        self.image_label.pack(pady=10)

        # 모델 불러오기
        self.model = load_model('autoencoder_model.h5')

        # 복원된 이미지 출력 버튼
        self.reconstruct_button = tk.Button(self.right_frame, text="Reconstruct Image", command=self.reconstruct_image)
        self.reconstruct_button.pack(pady=10)

        # 선택한 영역 초기화
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.selection_coords = None

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image_path = file_path
            self.image = Image.open(file_path)
            self.image = self.image.resize((400, 300))  # 이미지 크기 조정
            self.photo = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def start_selection(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def draw_selection(self, event):
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline="red")

    def end_selection(self, event):
        self.selection_coords = (self.start_x, self.start_y, event.x, event.y)

    def get_selected_region(self):
        if self.selection_coords:
            x0, y0, x1, y1 = self.selection_coords
            x0, y0 = min(x0, x1), min(y0, y1)
            x1, y1 = max(x0, x1), max(y0, y1)

            if x0 < x1 and y0 < y1:
                selected_region = self.image.crop((x0, y0, x1, y1))
                return selected_region, (x0, y0, x1, y1)
        return None, None

    def preprocess_image(self, image):
        # 선택한 이미지가 흰색 배경에 arial 폰트로 변환된 텍스트 이미지가 될 수 있도록 처리
        new_image = Image.new('L', (200, 50), 'white')
        draw = ImageDraw.Draw(new_image)

        # Arial 글꼴 사용
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except IOError:
            font = ImageFont.load_default()

        # 이미지를 200x50 크기로 리사이즈하여 중앙에 배치
        width, height = image.size
        image = image.resize((200, 50), Image.LANCZOS)

        new_image.paste(image, (0, 0))

        # 이미지를 numpy 배열로 변환 및 정규화
        return np.array(new_image) / 255.0

    def reconstruct_image(self):
        selected_region, region_coords = self.get_selected_region()
        if selected_region is not None:
            processed_image = self.preprocess_image(selected_region)
            processed_image = np.expand_dims(processed_image, axis=0)  # 배치 차원 추가
            processed_image = np.expand_dims(processed_image, axis=-1)  # 채널 차원 추가

            reconstructed_region = self.model.predict(processed_image)
            reconstructed_region = np.squeeze(reconstructed_region)  # 배치 및 채널 차원 제거
            reconstructed_image = Image.fromarray((reconstructed_region * 255).astype(np.uint8))

            # 선택한 영역 크기로 리사이즈
            width, height = region_coords[2] - region_coords[0], region_coords[3] - region_coords[1]
            reconstructed_image = reconstructed_image.resize((width, height), Image.LANCZOS)

            # 원본 이미지에 복원된 영역을 다시 넣기
            original_image = self.image.copy()
            original_image.paste(reconstructed_image, region_coords)

            # 이미지를 Tkinter PhotoImage로 변환하여 표시
            self.reconstructed_photo = ImageTk.PhotoImage(original_image)
            self.image_label.config(image=self.reconstructed_photo)
        else:
            messagebox.showinfo("Error", "No region selected. Please select a region.")

# Tkinter GUI 실행
root = tk.Tk()
app = ImageSelectionApp(root)

root.mainloop()
