import os
from PIL import Image, ImageDraw
import numpy as np
import random

# 기존 이미지 폴더 경로
original_folder = "word image folder"

# 새로운 이미지 폴더 경로
hidden_folder = "word image folder hidden"

# 새로운 폴더 생성
if not os.path.exists(hidden_folder):
    os.makedirs(hidden_folder)

# 기존 이미지 폴더 내 모든 이미지 파일을 읽어오기
image_files = [f for f in os.listdir(original_folder) if f.endswith('.png')]

# 검정색 부분을 가리는 함수
def hide_black_box(image_path, save_path):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # 이미지를 넘파이 배열로 변환
    img_array = np.array(img)
    
    # 검정색 부분의 좌표 찾기
    black_indices = np.argwhere(img_array == 0)
    
    # 검정색 부분이 없으면 종료
    if len(black_indices) == 0:
        img.save(save_path)
        return
    
    # 랜덤하게 검정색 부분 선택
    black_index = random.choice(black_indices)
    start_x, start_y = black_index[1], black_index[0]
    
    # 랜덤한 크기로 사각형 그리기
    box_width = random.randint(10, 50)  # 가리는 사각형의 너비
    box_height = random.randint(10, 50)  # 가리는 사각형의 높이
    
    draw.rectangle([start_x, start_y, start_x + box_width, start_y + box_height], fill="white")
    
    # 변경된 이미지 저장
    img.save(save_path)

# 각 이미지에 대해 검정색 부분을 가리기
for image_file in image_files:
    original_path = os.path.join(original_folder, image_file)
    hidden_path = os.path.join(hidden_folder, image_file)
    hide_black_box(original_path, hidden_path)