import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model

# 모델 불러오기
model = load_model('autoencoder_model.h5')

# 이미지를 모델 입력에 맞게 전처리하는 함수
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # 이미지를 그레이스케일로 변환
    img = img.resize((200, 50))  # 이미지 크기 조정
    img_array = np.array(img) / 255.0  # 이미지 배열로 변환 및 정규화
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
    img_array = np.expand_dims(img_array, axis=-1)  # 채널 차원 추가
    return img_array

# 이미지를 예측하고 결과를 표시하는 함수
def predict_image(image_path):
    img_array = preprocess_image(image_path)
    reconstructed_img = model.predict(img_array)
    reconstructed_img = np.squeeze(reconstructed_img)  # 배치 및 채널 차원 제거
    return reconstructed_img

# 파일 선택 대화상자를 통해 이미지 파일을 선택하고 예측한 이미지를 표시하는 함수
def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        predicted_img = predict_image(file_path)
        show_predicted_image(predicted_img, file_path)

# 예측 결과 이미지를 표시하는 함수
def show_predicted_image(image_array, file_path):
    img = Image.fromarray((image_array * 255).astype(np.uint8))
    img = img.resize((400, 100))  # 이미지 크기 조정
    img_tk = ImageTk.PhotoImage(img)
    
    # 이미지를 표시하는 라벨 업데이트
    predicted_label.config(image=img_tk)
    predicted_label.image = img_tk
    
    # 파일 경로 표시
    file_label.config(text="Selected File: " + file_path)

# Tkinter GUI 생성
root = tk.Tk()
root.title("Image Reconstruction Demo")

# 이미지 표시를 위한 라벨
predicted_label = tk.Label(root)
predicted_label.pack()

# 파일 경로 표시를 위한 라벨
file_label = tk.Label(root, text="Selected File: ")
file_label.pack()

# 이미지 선택 버튼
select_button = tk.Button(root, text="Select Image", command=select_image)
select_button.pack()

# GUI 실행
root.mainloop()