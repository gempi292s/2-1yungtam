import os
import random
from PIL import Image, ImageDraw, ImageFont

def create_word_images(word, folder):
    # 이미지 저장 폴더 생성
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # 단어를 이미지로 변환
    img = Image.new('L', (200, 50), color=255)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    draw.text((10, 10), word, font=font, fill=0)
    img.save(f"{folder}/{word}.png")
    
    # 가려진 문자 데이터 생성
    masked_img = img.copy()
    masked_draw = ImageDraw.Draw(masked_img)
    
    # 단어 중 일부 문자를 가립니다.
    num_chars_to_mask = random.randint(1, len(word))
    chars_to_mask = random.sample(range(len(word)), num_chars_to_mask)
    for char_index in chars_to_mask:
        # 문자 크기를 구하기 위해 getbbox 사용
        char_bbox = font.getbbox(word[char_index])
        char_width = char_bbox[2] - char_bbox[0]
        char_height = char_bbox[3] - char_bbox[1]
        
        x = 10 + sum(font.getbbox(word[i])[2] - font.getbbox(word[i])[0] for i in range(char_index))
        y = 10
        masked_draw.rectangle([x, y, x + char_width, y + char_height], fill=0)
    
    masked_img.save(f"{folder}/{word}_masked.png")

# 사용자로부터 단어 입력 받기
word = input("단어를 입력하세요: ")

# 이미지 저장 폴더 설정
folder = "newword"

# 단어 이미지 생성 및 저장
create_word_images(word, folder)
