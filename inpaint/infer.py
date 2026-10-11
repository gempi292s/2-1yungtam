"""학습된 모델로 임의 크기의 이미지를 복원하는 함수."""
import numpy as np
import torch
from PIL import Image

from .data import IMG_H, IMG_W
from .model import WordInpainter, ctc_greedy_decode


def load_model(path="word_inpainter.pt", device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = WordInpainter(**ckpt["config"])
    model.load_state_dict({k: v.float() for k, v in ckpt["state_dict"].items()})
    return model.to(device).eval()


@torch.no_grad()
def inpaint(model, image, mask):
    """image: PIL 이미지(아무 크기), mask: 같은 크기의 bool/0~1 배열 (1 = 복원할 영역).

    이미지를 50x200 캔버스에 비율 유지로 맞춰 넣고 복원한 뒤, 원래 해상도로 되돌려
    마스크 영역만 교체한다. 반환: (복원된 PIL 이미지 'L', 모델이 읽은 단어)
    """
    gray = image.convert("L")
    w, h = gray.size
    mask = np.asarray(mask, dtype=np.float32)
    assert mask.shape == (h, w), "마스크 크기는 이미지와 같아야 합니다"

    scale = min(IMG_W / w, IMG_H / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    ox, oy = (IMG_W - nw) // 2, (IMG_H - nh) // 2
    bg = int(np.median(np.asarray(gray)[mask < 0.5])) if (mask < 0.5).any() else 255

    canvas = Image.new("L", (IMG_W, IMG_H), bg)
    canvas.paste(gray.resize((nw, nh), Image.LANCZOS), (ox, oy))
    m_small = Image.fromarray((mask * 255).astype(np.uint8)).resize((nw, nh), Image.BILINEAR)
    m_canvas = np.zeros((IMG_H, IMG_W), np.float32)
    m_canvas[oy:oy + nh, ox:ox + nw] = (np.asarray(m_small) > 0).astype(np.float32)

    x = np.asarray(canvas, np.float32) / 255.0
    x = np.where(m_canvas > 0, 0.0, x).astype(np.float32)
    dev = next(model.parameters()).device
    t = lambda a: torch.from_numpy(a)[None, None].to(dev)
    _, comp, ctc = model(t(x), t(m_canvas))
    word = ctc_greedy_decode(ctc.cpu())[0]

    out = (comp[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    out = Image.fromarray(out).crop((ox, oy, ox + nw, oy + nh)).resize((w, h), Image.LANCZOS)
    result = np.where(mask > 0.5, np.asarray(out), np.asarray(gray)).astype(np.uint8)
    return Image.fromarray(result), word
