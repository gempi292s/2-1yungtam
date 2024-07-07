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

# 'a'로 시작하는 단어 필터링
words = df[df['word'].str.startswith('a')]['word'].tolist()

# 이미지 저장 폴더 생성
if not os.path.exists("word image folder"):
    os.makedirs("word image folder")

# 유명한 폰트 목록
fonts = [
    "arial.ttf",
    "times.ttf",
    "cour.ttf",
    "verdana.ttf",
    "helvetica.ttf"
]

# 랜덤 이미지 크기 범위
min_width, max_width = 200, 800
min_height, max_height = 50, 200

# 글자 크기
font_size = 24

# 단어를 이미지로 변환
for word in words:
    for font_path in fonts:
        # 랜덤 이미지 크기 선택
        width = random.randint(min_width, max_width)
        height = random.randint(min_height, max_height)
        
        img = Image.new('L', (width, height), color=255)
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            font = ImageFont.load_default()
        
        # 텍스트 크기 계산
        bbox = draw.textbbox((0, 0), word, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 중앙에 텍스트 배치
        text_x = (width - text_width) // 2
        text_y = (height - text_height) // 2
        
        draw.text((text_x, text_y), word, font=font, fill=0)
        
        # 글꼴과 이미지 크기를 포함한 파일명 생성
        font_name = os.path.splitext(os.path.basename(font_path))[0]
        img.save(f"word image folder/{word}_{font_name}_{width}x{height}.png")