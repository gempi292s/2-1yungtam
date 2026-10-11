"""마스크 인식 U-Net + Transformer 병목 + 글자 인식(CTC) 보조 헤드.

2023 버전 오토인코더(Conv 5층, 마스크 정보 없음)와 비교
- 입력: [가려진 이미지, 마스크] 2채널 → 어디를 복원해야 하는지 모델이 앎
- ConvNeXt 블록(7x7 depthwise conv + LayerNorm + GELU MLP(x2) + layer scale) 기반 U-Net
- 병목에 Transformer: 단어 전체를 보고 가려진 글자를 추론 (문맥 사용)
- CTC 글자 인식 헤드: 학습 시 "어떤 단어인지 읽어내기"를 함께 배워서 가려진 글자를
  철자 수준으로 맞히도록 유도 (text-prior inpainting). 추론 시 예측 단어도 알려줌
- 출력은 마스크 영역만 교체(합성)해서 보이는 부분은 절대 망가지지 않음
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import ALPHABET, IMG_H, IMG_W

PAD_H, PAD_W = 64, 224  # 8 의 배수로 패딩 (3번 다운샘플)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * 2)
        self.pw2 = nn.Linear(dim * 2, dim)
        self.gamma = nn.Parameter(torch.full((dim,), 1e-2))  # layer scale
        self.drop_path = drop_path

    def forward(self, x):
        y = self.dw(x).permute(0, 2, 3, 1)
        y = self.pw2(F.gelu(self.pw1(self.norm(y))))
        y = (self.gamma * y).permute(0, 3, 1, 2)
        if self.training and self.drop_path > 0:
            keep = torch.rand(x.shape[0], 1, 1, 1, device=x.device) >= self.drop_path
            y = y * keep / (1 - self.drop_path)
        return x + y


def stage(dim, depth, dp=0.0):
    return nn.Sequential(*[ConvNeXtBlock(dim, dp) for _ in range(depth)])


class Bottleneck(nn.Module):
    """2D 사인 위치 임베딩 + Pre-LN Transformer."""

    def __init__(self, dim, depth, heads):
        super().__init__()
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def pos_embed(h, w, dim, device):
        d = dim // 4
        omega = 1.0 / (10000 ** (torch.arange(d, device=device) / d))
        ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device),
                                indexing="ij")
        ey = ys.flatten()[:, None] * omega[None]
        ex = xs.flatten()[:, None] * omega[None]
        return torch.cat([ey.sin(), ey.cos(), ex.sin(), ex.cos()], dim=1)

    def forward(self, x):
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2) + self.pos_embed(h, w, c, x.device)[None]
        t = self.norm(self.tf(t))
        return t.transpose(1, 2).reshape(b, c, h, w)


class WordInpainter(nn.Module):
    """32x112 → 16x56 → 8x28(Transformer) 의 U-Net. 첫 단계에서 2배 줄여서 CPU 에서도 학습 가능하게,
    마지막에 원본 해상도 입력과 합쳐 3x3 conv 로 세밀한 획을 복원."""

    def __init__(self, dims=(32, 64, 160), depths=(1, 2, 2), tf_depth=4, heads=5,
                 drop_path=0.05):
        super().__init__()
        self.config = dict(dims=list(dims), depths=list(depths), tf_depth=tf_depth, heads=heads)
        d0, d1, d2 = dims
        self.stem = nn.Sequential(nn.Conv2d(2, d0, 2, stride=2), LayerNorm2d(d0))
        self.enc0 = stage(d0, depths[0], drop_path)
        self.down1 = nn.Sequential(LayerNorm2d(d0), nn.Conv2d(d0, d1, 2, stride=2))
        self.enc1 = stage(d1, depths[1], drop_path)
        self.down2 = nn.Sequential(LayerNorm2d(d1), nn.Conv2d(d1, d2, 2, stride=2))
        self.enc2 = stage(d2, depths[2], drop_path)
        self.mid = Bottleneck(d2, tf_depth, heads)  # 8 x 28 = 224 토큰, 문맥 추론

        self.up1 = nn.ConvTranspose2d(d2, d1, 2, stride=2)
        self.fuse1 = nn.Conv2d(d1 * 2, d1, 1)
        self.dec1 = stage(d1, 2, drop_path)
        self.up0 = nn.ConvTranspose2d(d1, d0, 2, stride=2)
        self.fuse0 = nn.Conv2d(d0 * 2, d0, 1)
        self.dec0 = stage(d0, 2, drop_path)
        self.up_full = nn.ConvTranspose2d(d0, 16, 2, stride=2)
        self.refine = nn.Sequential(nn.Conv2d(16 + 2, 16, 3, padding=1), nn.GELU(),
                                    nn.Conv2d(16, 1, 3, padding=1))

        # CTC 글자 인식 헤드: 16x56 디코더 특징을 높이 방향으로 모아 56 타임스텝
        self.ctc_head = nn.Sequential(nn.LayerNorm(d1), nn.Linear(d1, len(ALPHABET) + 1))
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, masked, mask):
        """masked, mask: (B,1,50,200) 0~1. 반환: (복원 logits, 합성 이미지, CTC log-probs (T,B,C))"""
        ph, pw = PAD_H - IMG_H, PAD_W - IMG_W
        pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
        x = torch.cat([F.pad(masked, pad, value=1.0), F.pad(mask, pad, value=0.0)], 1)
        x = torch.cat([x[:, :1] * 2 - 1, x[:, 1:]], 1)

        s0 = self.enc0(self.stem(x))
        s1 = self.enc1(self.down1(s0))
        s2 = self.mid(self.enc2(self.down2(s1)))

        d1 = self.dec1(self.fuse1(torch.cat([self.up1(s2), s1], 1)))
        d0 = self.dec0(self.fuse0(torch.cat([self.up0(d1), s0], 1)))
        full = self.refine(torch.cat([F.gelu(self.up_full(d0)), x], 1))
        logits = full[:, :, pad[2]:pad[2] + IMG_H, pad[0]:pad[0] + IMG_W]

        pred = torch.sigmoid(logits)
        composite = masked * (1 - mask) + pred * mask

        seq = d1.mean(2).transpose(1, 2)  # (B, 56, C)
        ctc = self.ctc_head(seq).log_softmax(-1).transpose(0, 1)  # (T, B, C)
        return logits, composite, ctc


def ctc_greedy_decode(ctc):
    """(T,B,C) → 단어 리스트."""
    best = ctc.argmax(-1).transpose(0, 1).tolist()
    out = []
    for seq in best:
        s, prev = [], 0
        for k in seq:
            if k != prev and k != 0:
                s.append(ALPHABET[k - 1])
            prev = k
        out.append("".join(s))
    return out


def count_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    m = WordInpainter()
    print(f"params: {count_params(m) / 1e6:.2f}M")
    x = torch.rand(2, 1, IMG_H, IMG_W)
    k = torch.zeros_like(x)
    lo, comp, ctc = m(x, k)
    print(lo.shape, comp.shape, ctc.shape, math.isfinite(lo.sum().item()))
