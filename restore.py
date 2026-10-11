"""명령줄 복원 도구 (GUI 없이 사용).

python restore.py 입력.png --box x0 y0 x1 y1 [--box ...] -o 결과.png
python restore.py --demo "hello" -o demo.png     # 단어를 그리고 일부 글자를 가린 뒤 복원
"""
import argparse
import random

import numpy as np
from PIL import Image

from inpaint.data import available_fonts, make_sample
from inpaint.infer import inpaint, load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image", nargs="?")
    p.add_argument("--box", nargs=4, type=int, action="append", metavar=("X0", "Y0", "X1", "Y1"))
    p.add_argument("--demo", help="이 단어로 데모 이미지를 만들어 복원")
    p.add_argument("--model", default="word_inpainter.pt")
    p.add_argument("-o", "--out", default="restored.png")
    args = p.parse_args()

    model = load_model(args.model)
    if args.demo:
        masked, mask, clean, _ = make_sample(args.demo.lower(), random.Random(0), available_fonts(), augment=False)
        image = Image.fromarray((masked * 255).astype(np.uint8))
    else:
        if not args.image or not args.box:
            p.error("이미지 경로와 --box 가 필요합니다 (또는 --demo 단어)")
        image = Image.open(args.image).convert("L")
        mask = np.zeros((image.height, image.width), np.float32)
        for x0, y0, x1, y1 in args.box:
            mask[min(y0, y1):max(y0, y1), min(x0, x1):max(x0, x1)] = 1

    result, word = inpaint(model, image, mask)
    panel = Image.new("L", (image.width, image.height * 2 + 4), 128)
    panel.paste(image, (0, 0))
    panel.paste(result, (0, image.height + 4))
    panel.save(args.out)
    print(f"모델이 읽은 단어: {word}\n저장: {args.out} (위: 입력, 아래: 복원)")


if __name__ == "__main__":
    main()
