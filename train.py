"""새 단어 이미지 복원 모델 학습.

예) python train.py --steps 20000 --batch 64          (GPU 가 있으면 자동 사용)
    python train.py --steps 3000 --out word_inpainter.pt

학습 데이터는 'english Dictionary.csv' 에서 매번 새로 합성하므로 이미지 폴더가 필요 없음.
"""
import argparse
import copy
import math
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from inpaint.data import (WordInpaintDataset, available_fonts, collate, fixed_set, load_words,
                          split_words)
from inpaint.model import WordInpainter, count_params, ctc_greedy_decode


class _Iter(IterableDataset):
    def __init__(self, ds):
        self.ds = ds

    def __iter__(self):
        return iter(self.ds)


def compute_loss(model, batch, device, ctc_weight):
    masked, mask, clean, targets, lengths, _ = batch
    masked, mask, clean = masked.to(device), mask.to(device), clean.to(device)
    logits, composite, ctc = model(masked, mask)
    # 가려진 영역에 가중치를 크게 준 BCE (텍스트는 거의 흑/백이라 BCE 가 선명하게 학습됨)
    weight = 1.0 + 9.0 * mask
    rec = (F.binary_cross_entropy_with_logits(logits, clean, reduction="none") * weight).mean()
    l1 = (torch.abs(composite - clean) * mask).sum() / mask.sum().clamp(min=1)
    t = torch.full((masked.shape[0],), ctc.shape[0], dtype=torch.long)
    ctc_loss = F.ctc_loss(ctc.float().cpu(), targets, t, lengths, zero_infinity=True)
    loss = rec + l1 + ctc_weight * ctc_loss.to(device)
    return loss, dict(rec=rec.item(), l1=l1.item(), ctc=ctc_loss.item())


@torch.no_grad()
def validate(model, samples, device, bs=128):
    model.eval()
    l1_sum, n_pix, correct, mse_sum = 0.0, 0.0, 0, 0.0
    for i in range(0, len(samples), bs):
        masked, mask, clean, _, _, words = collate(samples[i:i + bs])
        masked, mask, clean = masked.to(device), mask.to(device), clean.to(device)
        _, comp, ctc = model(masked, mask)
        l1_sum += (torch.abs(comp - clean) * mask).sum().item()
        n_pix += mask.sum().item()
        mse_sum += ((comp - clean) ** 2).mean(dim=(1, 2, 3)).clamp(min=1e-10).log10().sum().item()
        correct += sum(p == w for p, w in zip(ctc_greedy_decode(ctc.cpu()), words))
    model.train()
    return dict(masked_l1=l1_sum / max(n_pix, 1), psnr=-10 * mse_sum / len(samples),
                word_acc=correct / len(samples))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--ctc-weight", type=float, default=0.3)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--out", default="word_inpainter.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-limit", type=float, default=0, help="분 단위. 0 이면 무제한")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    fonts = available_fonts()
    split = split_words(load_words())
    print(f"device={device} fonts={len(fonts)} words={ {k: len(v) for k, v in split.items()} }")

    loader = DataLoader(_Iter(WordInpaintDataset(split["train"], fonts, True, args.seed)),
                        batch_size=args.batch, num_workers=args.workers, collate_fn=collate,
                        persistent_workers=args.workers > 0)
    val_samples = fixed_set(split["val"], fonts, 1024, seed=123, augment=True)

    model = WordInpainter().to(device)
    ema = copy.deepcopy(model).eval()
    for q in ema.parameters():
        q.requires_grad_(False)
    print(f"params={count_params(model) / 1e6:.2f}M")

    decay, no_decay = [], []
    for n, q in model.named_parameters():
        (no_decay if q.ndim <= 1 or "gamma" in n else decay).append(q)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.98))
    warmup = min(500, args.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup
        else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warmup) / max(1, args.steps - warmup)))))

    best = None
    t0 = time.time()
    it = iter(loader)
    for step in range(1, args.steps + 1):
        batch = next(it)
        loss, parts = compute_loss(model, batch, device, args.ctc_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        d = min(args.ema, (1 + step) / (10 + step))
        with torch.no_grad():
            for e, q in zip(ema.state_dict().values(), model.state_dict().values()):
                e.copy_(e * d + q * (1 - d)) if e.dtype.is_floating_point else e.copy_(q)

        if step % 50 == 0:
            el = time.time() - t0
            print(f"step {step} loss {loss.item():.4f} " + " ".join(f"{k}={v:.4f}" for k, v in parts.items())
                  + f" lr={sched.get_last_lr()[0]:.2e} {el / step:.2f}s/it", flush=True)
        out_of_time = args.time_limit and time.time() - t0 > args.time_limit * 60
        if step % args.val_every == 0 or step == args.steps or out_of_time:
            m = validate(ema, val_samples, device)
            print(f"[val] step {step} {m}", flush=True)
            if best is None or m["masked_l1"] < best["masked_l1"]:
                best = dict(m, step=step)
                torch.save({"config": ema.config, "state_dict": {k: v.half() for k, v in ema.state_dict().items()},
                            "val": best}, args.out)
                print(f"saved {args.out}", flush=True)
        if out_of_time:
            print("time limit reached")
            break
    print("best", best)


if __name__ == "__main__":
    main()
