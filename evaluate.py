"""2023 오토인코더 vs 새 모델을 학습에 쓰지 않은 단어(test split)로 비교.

python evaluate.py                    # 지표 출력 + comparison.png 저장
"""
import argparse

import numpy as np
import torch
from PIL import Image

from inpaint.data import available_fonts, collate, fixed_set, load_words, split_words
from inpaint.infer import load_model
from inpaint.legacy import load_legacy
from inpaint.model import ctc_greedy_decode


def metrics(pred, clean, mask):
    err = torch.abs(pred - clean)
    masked_l1 = (err * mask).sum().item() / mask.sum().item()
    mse = ((pred - clean) ** 2).mean(dim=(1, 2, 3)).clamp(min=1e-10)
    psnr = (-10 * mse.log10()).mean().item()
    # 가려진 영역의 픽셀을 흑/백으로 이진화했을 때 맞힌 비율
    bin_acc = (((pred > 0.5) == (clean > 0.5)).float() * mask).sum().item() / mask.sum().item()
    return dict(masked_L1=masked_l1, PSNR=psnr, masked_pixel_acc=bin_acc)


@torch.no_grad()
def run(args):
    fonts = available_fonts()
    test_words = split_words(load_words())["test"]
    new = load_model(args.model)
    old = load_legacy(args.legacy)

    for name, augment in [("2023 방식 (고정 폰트/위치, 글자 가림)", False),
                          ("증강 (다양한 폰트/위치/노이즈/임의 사각형)", True)]:
        samples = fixed_set(test_words, fonts, args.n, seed=7, augment=augment)
        masked, mask, clean, _, _, words = collate(samples)
        old_out = old(masked)
        old_comp = masked * (1 - mask) + old_out * mask  # 공정하게 마스크 영역만 교체한 버전
        _, new_comp, ctc = new(masked, mask)
        read = ctc_greedy_decode(ctc)
        print(f"\n== {name}, test 단어 {len(samples)}개 ==")
        print(f"{'모델':<34}{'masked L1 ↓':>12}{'PSNR ↑':>9}{'pixel acc ↑':>13}")
        for label, pred in [("2023 Autoencoder (출력 그대로)", old_out),
                            ("2023 Autoencoder (+마스크 합성)", old_comp),
                            ("2026 WordInpainter", new_comp)]:
            m = metrics(pred, clean, mask)
            print(f"{label:<34}{m['masked_L1']:>12.4f}{m['PSNR']:>9.2f}{m['masked_pixel_acc']:>13.3f}")
        acc = np.mean([r == w for r, w in zip(read, words)])
        print(f"새 모델이 가려진 단어 전체를 맞힌 비율 (CTC): {acc:.3f}")

        if not augment:
            rows = []
            for i in range(8):
                rows.append(np.concatenate([masked[i, 0], old_out[i, 0], new_comp[i, 0], clean[i, 0]], 1))
                rows.append(np.full((3, rows[-1].shape[1]), 0.6, np.float32))
            Image.fromarray((np.concatenate(rows, 0) * 255).astype(np.uint8)).save(args.image)
            print(f"비교 이미지 저장: {args.image}  (왼쪽부터: 입력 | 2023 | 2026 | 정답)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="word_inpainter.pt")
    p.add_argument("--legacy", default="autoencoder_model.h5")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--image", default="comparison.png")
    run(p.parse_args())
