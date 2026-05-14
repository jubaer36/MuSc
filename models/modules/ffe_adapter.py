import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DCTLayer(nn.Module):
    """
    Orthonormal DCT-II and its inverse over the last dimension.
    Fixed buffer — condition number = 1, gradients preserved exactly.
    """
    def __init__(self, n: int):
        super().__init__()
        k = torch.arange(n, dtype=torch.float64).unsqueeze(1)   # (n, 1)
        j = torch.arange(n, dtype=torch.float64).unsqueeze(0)   # (1, n)
        M = torch.cos(math.pi * k * (2.0 * j + 1.0) / (2.0 * n))
        M[0] /= math.sqrt(n)
        M[1:] /= math.sqrt(n / 2.0)
        self.register_buffer('M', M.float())    # (n, n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.M)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.M.t())


class FFEAdapter(nn.Module):
    """
    Frequency-aware Feature Extraction adapter for one DINOv2 layer.

    Args:
        embed_dim   : DINOv2 embedding dimension (1024 for ViT-L/14)
        window_size : P for P×P non-overlapping spatial windows (default 3)
        lam         : residual mixing weight (0.0 = pass-through, 1.0 = full adapter)
    """
    def __init__(self, embed_dim: int = 1024, window_size: int = 3, lam: float = 0.5):
        super().__init__()
        self.P = window_size
        self.n = window_size * window_size      # 9 spatial positions
        self.lam = lam

        self.dct = DCTLayer(self.n)

        self.linear = nn.Linear(embed_dim, embed_dim, bias=True)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def _to_windows(self, x: torch.Tensor):
        """(B, D, H, W) → (B·nH·nW, P², D) with padding."""
        B, D, H, W = x.shape
        P = self.P
        pad_h = (P - H % P) % P
        pad_w = (P - W % P) % P
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        H2, W2 = x.shape[2], x.shape[3]
        nH, nW = H2 // P, W2 // P
        x = x.reshape(B, D, nH, P, nW, P)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()   # (B, nH, nW, P, P, D)
        x = x.reshape(B * nH * nW, self.n, D)
        return x, (B, D, H, W, nH, nW, pad_h, pad_w)

    def _from_windows(self, x: torch.Tensor, meta) -> torch.Tensor:
        """(B·nH·nW, P², D) → (B, D, H, W)"""
        B, D, H, W, nH, nW, pad_h, pad_w = meta
        P = self.P
        x = x.reshape(B, nH, nW, P, P, D)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()   # (B, D, nH, P, nW, P)
        x = x.reshape(B, D, nH * P, nW * P)
        if pad_h or pad_w:
            x = x[:, :, :H, :W]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, D, H, W) — DINOv2 feature map, after LayerNorm
        Returns frequency-enhanced feature map of identical shape.
        """
        windows, meta = self._to_windows(x)              # (B·nH·nW, P², D)
        # DCT over dim=1 (P² spatial positions); F.linear acts on last dim → transpose
        f_dct = self.dct(windows.transpose(1, 2)).transpose(1, 2)   # (B·nH·nW, P², D)
        f_lin = F.gelu(self.linear(f_dct))                           # Linear over D
        f_ffe = self.dct.inverse(f_lin.transpose(1, 2)).transpose(1, 2)  # IDCT → spatial
        f_ffe = self._from_windows(f_ffe, meta)           # (B, D, H, W)
        return self.lam * f_ffe + (1.0 - self.lam) * x


if __name__ == "__main__":
    adapter = FFEAdapter(embed_dim=1024, window_size=3, lam=0.5)
    x = torch.randn(2, 1024, 37, 37)
    out = adapter(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print("FFEAdapter shape check passed:", out.shape)
