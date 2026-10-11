"""2023 Keras 오토인코더(autoencoder_model.h5)를 PyTorch 로 옮겨서 같은 조건으로 비교하기 위한 모듈.

TensorFlow 없이 h5py 로 가중치만 읽어옴.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LegacyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(1, 64, 3, padding=1)
        self.c1 = nn.Conv2d(64, 128, 3, padding=1)
        self.c2 = nn.Conv2d(128, 128, 3, padding=1)
        self.c3 = nn.Conv2d(128, 64, 3, padding=1)
        self.c4 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c0(x)), 2, ceil_mode=True)  # Keras padding='same'
        x = F.max_pool2d(F.relu(self.c1(x)), 2, ceil_mode=True)
        x = F.interpolate(F.relu(self.c2(x)), scale_factor=2, mode="nearest")
        x = F.interpolate(F.relu(self.c3(x)), scale_factor=2, mode="nearest")
        x = torch.sigmoid(self.c4(x))
        return x[:, :, 1:-1, :]  # Cropping2D((1, 1), (0, 0))


def load_legacy(path="autoencoder_model.h5"):
    import h5py
    m = LegacyAutoencoder()
    with h5py.File(path, "r") as f:
        g = f["model_weights"]
        for i, conv in enumerate([m.c0, m.c1, m.c2, m.c3, m.c4]):
            name = "conv2d" if i == 0 else f"conv2d_{i}"
            k = torch.from_numpy(g[name][name]["kernel:0"][()])  # (kh, kw, in, out)
            b = torch.from_numpy(g[name][name]["bias:0"][()])
            conv.weight.data.copy_(k.permute(3, 2, 0, 1))
            conv.bias.data.copy_(b)
    return m.eval()
