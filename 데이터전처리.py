import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os
import random

# 데이터 전처리
df = pd.read_csv('english Dictionary.csv')
df.drop_duplicates(subset=['word'], keep='last', inplace=True)
df = df.dropna(subset=['word'])
df['word'] = df['word'].astype(str)
df = df[~df['word'].str.contains(r'[^a-zA-Z]')]
df['word'] = df['word'].str.lower()


words = df['word'].tolist()

# 이미지 저장 폴더 생성
original_folder = "word_image_folder"
masked_folder = "masked_word_image_folder"
if not os.path.exists(original_folder):
    os.makedirs(original_folder)
if not os.path.exists(masked_folder):
    os.makedirs(masked_folder)

# 단어를 이미지로 변환
for word in words:
    if word.startswith('c'):
        continue
    img = Image.new('L', (200, 50), color=255)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    draw.text((10, 10), word, font=font, fill=0)
    img.save(f"{original_folder}/{word}.png")


    
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
    
    masked_img.save(f"{masked_folder}/{word}_masked.png")