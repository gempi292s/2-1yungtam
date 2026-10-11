"""단어 이미지 합성 데이터셋.

2024 버전(데이터전처리.py)과 달라진 점
- 이미지를 디스크에 미리 저장하지 않고 학습 중에 매번 새로 생성 (무한 데이터, 매 에폭 다른 마스크)
- 글자 위치를 실제 렌더링 좌표(textlength/textbbox)로 계산 → 가림 상자가 글자와 정확히 일치
- 여러 폰트/크기/위치/노이즈로 증강해서 실제 이미지에도 강하게
- 가려진 영역을 알려주는 마스크 채널을 함께 제공
- 단어 단위로 train/val/test 를 분리해서, 처음 보는 단어로 성능을 측정
"""
import os
import random
import zlib

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

IMG_H, IMG_W = 50, 200
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
CHAR2IDX = {c: i + 1 for i, c in enumerate(ALPHABET)}  # 0 은 CTC blank

# 윈도우/맥/리눅스에서 흔한 폰트 후보. 없는 폰트는 자동으로 건너뜀
FONT_CANDIDATES = [
    "arial.ttf", "Arial.ttf", "times.ttf", "verdana.ttf", "calibri.ttf", "georgia.ttf",
    "DejaVuSans.ttf", "DejaVuSerif.ttf", "DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    "/usr/share/fonts/opentype/inter/Inter-Regular.otf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
]


def available_fonts():
    fonts = []
    seen = set()
    for name in FONT_CANDIDATES:
        try:
            f = ImageFont.truetype(name, 20)
        except OSError:
            continue
        key = f.getname()
        if key not in seen:
            seen.add(key)
            fonts.append(name)
    return fonts


def load_font(name, size):
    if name is None:
        return ImageFont.load_default(size=size)  # Pillow 내장 폰트 (어디서나 동작)
    return ImageFont.truetype(name, size)


def load_words(csv_path="english Dictionary.csv"):
    """2024 버전과 같은 규칙: 알파벳으로만 된 단어, 소문자, 중복 제거."""
    df = pd.read_csv(csv_path, usecols=["word"])
    w = df["word"].dropna().astype(str)
    w = w[~w.str.contains(r"[^a-zA-Z]")].str.lower()
    w = w[w.str.len() <= 18]
    return sorted(set(w))


def split_words(words):
    """단어 해시로 고정 분할 (train 90% / val 5% / test 5%)."""
    split = {"train": [], "val": [], "test": []}
    for w in words:
        h = zlib.crc32(w.encode()) % 100
        split["train" if h < 90 else "val" if h < 95 else "test"].append(w)
    return split


def render_word(word, font, x0, y0, ink=0, bg=255):
    img = Image.new("L", (IMG_W, IMG_H), color=bg)
    draw = ImageDraw.Draw(img)
    draw.text((x0, y0), word, font=font, fill=ink)
    # 글자별 실제 상자: 앞 글자들의 진짜 advance 폭으로 위치 계산
    boxes = []
    for i, ch in enumerate(word):
        x = x0 + draw.textlength(word[:i], font=font)
        boxes.append(draw.textbbox((x, y0), ch, font=font))
    return img, boxes


def fit_font(word, font_name, max_size, rng):
    size = max_size
    while size > 10:
        font = load_font(font_name, size)
        l, t, r, b = font.getbbox(word)
        if r - l <= IMG_W - 8 and b - t <= IMG_H - 6:
            return font, (l, t, r, b)
        size -= 1
    font = load_font(font_name, size)
    return font, font.getbbox(word)


def make_sample(word, rng, fonts, augment=True):
    """(가려진 이미지, 마스크, 원본 이미지, 단어) 를 반환. 값은 0~1 float32."""
    if augment:
        font_name = rng.choice(fonts + [None])
        font, (l, t, r, b) = fit_font(word, font_name, rng.randint(18, 28), rng)
        x0 = rng.randint(2, max(2, IMG_W - (r - l) - 2)) - l
        y0 = rng.randint(2, max(2, IMG_H - (b - t) - 2)) - t
        ink = rng.randint(0, 60)
        bg = rng.randint(200, 255)
    else:
        # 2024 버전과 같은 고정 배치 (x=10, y=10, 크기 24)
        font, (l, t, r, b) = fit_font(word, fonts[0] if fonts else None, 24, rng)
        x0, y0, ink, bg = 10, 10, 0, 255
    img, boxes = render_word(word, font, x0, y0, ink, bg)
    if augment and rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.8)))

    clean = np.asarray(img, dtype=np.float32) / 255.0
    if augment and rng.random() < 0.5:
        clean = np.clip(clean + np.random.default_rng(rng.randint(0, 2**31)).normal(
            0, rng.uniform(0.0, 0.04), clean.shape).astype(np.float32), 0, 1)

    mask = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    mode = rng.random() if augment else 0.0
    if mode < 0.7:
        # 2024 버전과 같은 방식: 글자 단위로 가리기 (단, 전부는 가리지 않음)
        k = rng.randint(1, max(1, (len(word) + 1) // 2))
        for i in rng.sample(range(len(word)), k):
            x1, y1, x2, y2 = boxes[i]
            pad = rng.randint(0, 2) if augment else 0
            mask[max(0, int(y1) - pad):int(np.ceil(y2)) + pad,
                 max(0, int(x1) - pad):int(np.ceil(x2)) + pad] = 1
    else:
        # 임의 사각형 (GUI 에서 사용자가 대충 드래그하는 경우 대비)
        for _ in range(rng.randint(1, 3)):
            w = rng.randint(6, 40)
            h = rng.randint(10, IMG_H)
            x = rng.randint(0, IMG_W - w)
            y = rng.randint(0, IMG_H - h)
            mask[y:y + h, x:x + w] = 1

    fill = 0.0 if (not augment or rng.random() < 0.6) else rng.choice([1.0, 0.5])
    masked = np.where(mask > 0, fill, clean).astype(np.float32)
    return masked, mask, clean, word


def encode_word(word):
    return [CHAR2IDX[c] for c in word]


class WordInpaintDataset:
    """torch IterableDataset 처럼 쓰는 무한 생성기 (worker 별 시드 분리)."""

    def __init__(self, words, fonts, augment=True, seed=0):
        self.words, self.fonts, self.augment, self.seed = words, fonts, augment, seed

    def __iter__(self):
        import torch
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = random.Random(self.seed * 1000 + wid + os.getpid())
        while True:
            yield make_sample(rng.choice(self.words), rng, self.fonts, self.augment)


def fixed_set(words, fonts, n, seed, augment):
    """평가용 고정 샘플 집합."""
    rng = random.Random(seed)
    np.random.seed(seed)
    picked = rng.sample(words, min(n, len(words)))
    return [make_sample(w, rng, fonts, augment) for w in picked]


def collate(batch):
    import torch
    masked = torch.from_numpy(np.stack([b[0] for b in batch]))[:, None]
    mask = torch.from_numpy(np.stack([b[1] for b in batch]))[:, None]
    clean = torch.from_numpy(np.stack([b[2] for b in batch]))[:, None]
    words = [b[3] for b in batch]
    targets = torch.tensor([i for w in words for i in encode_word(w)], dtype=torch.long)
    lengths = torch.tensor([len(w) for w in words], dtype=torch.long)
    return masked, mask, clean, targets, lengths, words
