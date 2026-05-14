# FFE Adapter: Implementation Plan

Training one frequency-aware linear adapter per DINOv2 layer on MVTec AD,
then integrating the frozen weights into the existing zero-shot MuSc pipeline.

---

## Overview

```
Phase 1 — Build            Phase 2 — Train              Phase 3 — Integrate
────────────────           ─────────────────────         ──────────────────────────
ffe_adapter.py      →      train_ffe.py                  _LNAMD.py  (modified)
  DCTLayer                   load MVTec AD train          musc.py    (modified)
  FFEAdapter (×4)            freeze DINOv2                musc.yaml  (modified)
                             MFP loss per layer
                             save 4 checkpoints
                                    ↓
                             checkpoints/ffe/
                               ffe_layer_0.pt   (block 5)
                               ffe_layer_1.pt   (block 11)
                               ffe_layer_2.pt   (block 17)
                               ffe_layer_3.pt   (block 23)
```

**Training objective:** Masked Frequency Prediction (MFP) on the MVTec AD **train
split** (normal images only — no anomaly labels required). Each adapter learns the
normal spatial-frequency structure of MVTec objects. At inference, anomalous patches
deviate from this learned distribution, making their MSM nearest-neighbour distances
larger.

**Why train split only:** Keeps training completely label-free. The test split is
reserved for evaluation. The train split has roughly 200–300 normal images per
category × 15 categories ≈ 3,300–4,500 images, sufficient for the 1.05M-parameter
linear layers.

---

## File Map

| File | Status | Purpose |
|------|--------|---------|
| `models/modules/ffe_adapter.py` | **New** | `DCTLayer` + `FFEAdapter` classes |
| `train_ffe.py` | **New** | Full self-contained training script |
| `models/modules/_LNAMD.py` | **Modified** | Accept and apply adapters inside `_embed()` |
| `models/musc.py` | **Modified** | Load adapter weights and pass to LNAMD |
| `configs/musc.yaml` | **Modified** | `ffe_adapter` config block |

---

## Background: Why One Adapter Per Layer

MuSc extracts DINOv2 features from four transformer blocks (0-indexed): **5, 11, 17,
23** out of 24 total in ViT-L/14. Each block contains fundamentally different content:

| Layer index | DINOv2 block | Content | DCT spectrum after windowing |
|-------------|-------------|---------|------------------------------|
| 0 | Block 5 (~21%) | Edges, local textures | High-frequency dominant |
| 1 | Block 11 (~46%) | Object parts, local structure | Broad, mixed |
| 2 | Block 17 (~71%) | Semantic regions | Mid-frequency skewed |
| 3 | Block 23 (~96%) | Near-class-token semantics | DC term dominant |

A shared Linear across all four layers would receive opposing gradient signals
simultaneously — updates that amplify high-frequency bands for block 5 partially
cancel updates that suppress them for block 23. Four independent Linears each
specialise cleanly for their layer's frequency distribution.

Parameter cost: 4 × Linear(1024, 1024) + bias ≈ **4.2M parameters** (1.4% of the
frozen 307M-parameter DINOv2 backbone).

---

## Phase 1 — `models/modules/ffe_adapter.py`

### Architecture per adapter

```
Input: (B, D, H, W)  — LayerNorm-ed DINOv2 feature map for one layer
       │
  Unfold into non-overlapping P×P windows (P=3)
       │  → (B·nH·nW, P², D)     P²=9 spatial positions, nH=nW=⌊37/3⌋=12
       │
  DCT-II over dim=1 (the 9 spatial positions)
       │  → (B·nH·nW, 9, D)     frequency domain
       │
  Linear(D, D) + GELU            ← only trainable parameter
       │  → (B·nH·nW, 9, D)     transformed frequency domain
       │
  IDCT-II over dim=1             inverse DCT, back to spatial
       │  → (B·nH·nW, 9, D)
       │
  Fold windows back into spatial map
       │  → (B, D, H, W)
       │
  Residual mix:  f_hat = λ·f_ffe + (1-λ)·f_input
       │         λ=1.0 during training, configurable at inference
Output: (B, D, H, W)  — same shape as input
```

**Initialisation:** `nn.init.eye_(linear.weight)` makes every adapter an exact
no-op at the start of training. MuSc performance at epoch 0 is therefore identical
to the unchanged baseline — any deviation in performance is attributable solely to
what the Linear learned.

**Window alignment with PatchMaker:** LNAMD's `PatchMaker` also uses a 3×3 window
(`r=3`). The FFE adapter uses the same window size. The adapter refines features at
exactly the spatial scale that LNAMD will aggregate next.

### Complete file

```python
# models/modules/ffe_adapter.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DCTLayer(nn.Module):
    """
    Orthonormal DCT-II and its inverse over the last dimension.
    Stored as a fixed buffer (not a learnable parameter).
    Gradients flow through as a fixed linear map — condition number = 1,
    so gradient norms are preserved exactly (no vanishing / exploding).
    """
    def __init__(self, n: int):
        super().__init__()
        k = torch.arange(n, dtype=torch.float64).unsqueeze(1)   # (n, 1)
        j = torch.arange(n, dtype=torch.float64).unsqueeze(0)   # (1, n)
        M = torch.cos(math.pi * k * (2.0 * j + 1.0) / (2.0 * n))
        M[0] /= math.sqrt(n)
        M[1:] /= math.sqrt(n / 2.0)
        self.register_buffer('M', M.float())    # (n, n), not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., n)
        return F.linear(x, self.M)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        # IDCT = DCT.T  (orthogonal matrix)
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

        # Single trainable Linear — identity init → adapter is a no-op at epoch 0
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
        windows, meta = self._to_windows(x)         # (B·nH·nW, P², D)
        f_dct = self.dct(windows)                    # DCT over 9 spatial positions
        f_lin = F.gelu(self.linear(f_dct))           # Linear + GELU in freq domain
        f_ffe = self.dct.inverse(f_lin)              # IDCT back to spatial
        f_ffe = self._from_windows(f_ffe, meta)      # (B, D, H, W)
        return self.lam * f_ffe + (1.0 - self.lam) * x


if __name__ == "__main__":
    # Quick shape-correctness check
    adapter = FFEAdapter(embed_dim=1024, window_size=3, lam=0.5)
    x = torch.randn(2, 1024, 37, 37)
    out = adapter(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print("FFEAdapter shape check passed:", out.shape)
```

---

## Phase 2 — `train_ffe.py`

### Training objective: MFP (Masked Frequency Prediction)

For each 3×3 spatial window after DCT, zero-mask the 3 highest-frequency bands
(indices 6, 7, 8 of the 9-position output). The Linear is trained to reconstruct
the full DCT output from the masked input.

```
f_target = DCT(window)                        # all 9 frequency bands visible
f_masked = f_target.clone()
f_masked[:, 6:, :] = 0.0                     # zero out bands 6, 7, 8

f_pred   = GELU(Linear(f_masked))            # predict full spectrum from low-freq

loss = weighted_MSE(f_pred, f_target)
       weights[0:6] = 1.0                    # normal weight on visible low-freq bands
       weights[6:9] = 3.0                    # 3× weight on masked high-freq bands
```

The asymmetric weighting is deliberate: high-frequency bands are the ones anomalies
most disrupt. Training the Linear to predict them accurately from low-frequency
context means it internalises what high-frequency patterns are *normal* — which
makes abnormal high-frequency patterns stand out more at inference.

### Gradient flow

```
Weighted MSE Loss
    │  ∂L/∂f_pred  shape: (B·win, 9, D)
    ▼
GELU  →  ∂L/∂Linear_output = ∂L/∂f_pred × GELU'(Linear_output)
    ▼
Linear(D, D)   ← gradient update applied here (only trainable parameter)
    ∂L/∂W = (∂L/∂Linear_out)ᵀ · f_masked
    ▼
f_masked  →  gradient flows to DCT input (via fixed DCT matrix M.T)
    ▼
DINOv2 features  →  FROZEN, param.requires_grad=False, gradient discarded
```

DCT and IDCT are orthogonal matrices (condition number = 1). Gradients pass
through them without scaling — no vanishing, no exploding.

### Complete training script

```python
# train_ffe.py

import os
import math
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import sys
sys.path.insert(0, '.')
import models.backbone._backbones as _backbones
from datasets.mvtec import MVTecDataset, DatasetSplit, _CLASSNAMES
from models.modules.ffe_adapter import FFEAdapter


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mfp_loss(adapter: FFEAdapter, feat_map: torch.Tensor,
             mask_from: int = 6, hi_weight: float = 3.0) -> torch.Tensor:
    """
    Masked Frequency Prediction loss for one adapter on one layer's feature map.

    adapter   : FFEAdapter containing the DCTLayer and trainable Linear
    feat_map  : (B, D, H, W) — LayerNorm-ed DINOv2 features for this layer
    mask_from : band index from which to zero (default 6 → zero bands 6, 7, 8)
    hi_weight : loss weight multiplier for the masked (high-freq) bands
    """
    windows, _ = adapter._to_windows(feat_map)      # (B·nH·nW, P², D)

    with torch.no_grad():
        f_target = adapter.dct(windows)              # full DCT — target

    f_masked = f_target.clone()
    f_masked[:, mask_from:, :] = 0.0                # mask high-freq bands

    f_pred = F.gelu(adapter.linear(f_masked))       # predict from masked input

    n = adapter.n
    weights = torch.ones(n, device=feat_map.device)
    weights[mask_from:] = hi_weight
    loss = ((f_pred - f_target) ** 2 * weights.view(1, n, 1)).mean()
    return loss


def layernorm_feature(feat: torch.Tensor) -> torch.Tensor:
    """Replicate the per-image LayerNorm applied in LNAMD._embed()."""
    return F.layer_norm(feat, [feat.shape[1], feat.shape[2], feat.shape[3]])


def build_loaders(data_path: str, image_size: int,
                  val_fraction: float = 0.15,
                  batch_size: int = 16, seed: int = 42):
    """
    Load all 15 MVTec AD categories, train split (normal images only).
    Returns (train_loader, val_loader).
    """
    all_datasets = [
        MVTecDataset(source=data_path, classname=cls,
                     resize=image_size, imagesize=image_size,
                     split=DatasetSplit.TRAIN)
        for cls in _CLASSNAMES
    ]
    combined = torch.utils.data.ConcatDataset(all_datasets)

    rng = random.Random(seed)
    train_idx, val_idx = [], []
    offset = 0
    for ds in all_datasets:
        n = len(ds)
        idx = list(range(offset, offset + n))
        rng.shuffle(idx)
        n_val = max(1, int(n * val_fraction))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
        offset += n

    train_loader = DataLoader(Subset(combined, train_idx),
                              batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(Subset(combined, val_idx),
                              batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    print(f"Train: {len(train_idx)} images | Val: {len(val_idx)} images")
    return train_loader, val_loader


def extract_features(dino_model, images: torch.Tensor,
                     layer_indices: list, device: torch.device):
    """
    Extract DINOv2 patch features for each requested layer, apply LayerNorm,
    and return as spatial feature maps — matching exactly what LNAMD._embed()
    produces before PatchMaker.

    Returns list of len(layer_indices) tensors, each (B, D, H, W).
    """
    with torch.no_grad():
        patch_tokens = dino_model.get_intermediate_layers(
            images.to(device), n=layer_indices, return_class_token=False
        )   # tuple of (B, n_patches, D) for each layer

    feat_maps = []
    for tokens in patch_tokens:
        B, L, D = tokens.shape
        H = W = int(math.sqrt(L))
        feat = tokens.reshape(B, H, W, D).permute(0, 3, 1, 2)   # (B, D, H, W)
        feat = layernorm_feature(feat)
        feat_maps.append(feat)
    return feat_maps


def train(args):
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # DINOv2 — fully frozen throughout
    print("Loading DINOv2...")
    dino = _backbones.load(args.backbone)
    dino.to(device).eval()
    for p in dino.parameters():
        p.requires_grad_(False)

    # Convert config layer indices to 0-indexed DINOv2 block numbers.
    # musc.py sets features_list = [l+1 for l in feature_layers],
    # then passes n=[l-1 for l in features_list] to get_intermediate_layers.
    # Net result: n = feature_layers (the values in config, 0-indexed).
    layer_indices = args.feature_layers
    n_layers = len(layer_indices)
    print(f"DINOv2 blocks (0-indexed): {layer_indices}")

    # One adapter per layer — lam=1.0 during training (no residual mixing)
    adapters = nn.ModuleList([
        FFEAdapter(embed_dim=args.embed_dim, window_size=args.window_size, lam=1.0)
        for _ in range(n_layers)
    ]).to(device)

    optimizer = torch.optim.Adam(adapters.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    train_loader, val_loader = build_loaders(
        args.data_path, args.image_size,
        val_fraction=0.15, batch_size=args.batch_size, seed=args.seed,
    )

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):

        # ── train ─────────────────────────────────────────────────────────
        adapters.train()
        total_train = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]"):
            images = batch["image"].float()
            feat_maps = extract_features(dino, images, layer_indices, device)

            loss = sum(
                mfp_loss(adapters[i], feat_maps[i],
                         mask_from=args.mask_from, hi_weight=args.hi_weight)
                for i in range(n_layers)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapters.parameters(), max_norm=1.0)
            optimizer.step()
            total_train += loss.item()

        scheduler.step()

        # ── validate ──────────────────────────────────────────────────────
        adapters.eval()
        total_val = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]"):
                images = batch["image"].float()
                feat_maps = extract_features(dino, images, layer_indices, device)
                total_val += sum(
                    mfp_loss(adapters[i], feat_maps[i],
                             mask_from=args.mask_from, hi_weight=args.hi_weight)
                    for i in range(n_layers)
                ).item()

        avg_train = total_train / len(train_loader)
        avg_val   = total_val   / len(val_loader)
        lr_now    = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:03d} | train={avg_train:.6f} | val={avg_val:.6f} | lr={lr_now:.2e}")

        # ── checkpoint ────────────────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            for i, adapter in enumerate(adapters):
                path = os.path.join(args.save_dir, f"ffe_layer_{i}.pt")
                torch.save(adapter.linear.state_dict(), path)
            print(f"  Saved best checkpoints  (val={best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

    print(f"\nDone. Checkpoints in {args.save_dir}/")
    for i in range(n_layers):
        print(f"  ffe_layer_{i}.pt  ← DINOv2 block {layer_indices[i]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",      default="./data/mvtec_anomaly_detection/")
    parser.add_argument("--backbone",       default="dinov2_vitl14")
    parser.add_argument("--feature_layers", nargs="+", type=int, default=[5, 11, 17, 23])
    parser.add_argument("--embed_dim",      type=int,   default=1024)
    parser.add_argument("--image_size",     type=int,   default=518)
    parser.add_argument("--window_size",    type=int,   default=3)
    parser.add_argument("--mask_from",      type=int,   default=6)
    parser.add_argument("--hi_weight",      type=float, default=3.0)
    parser.add_argument("--epochs",         type=int,   default=30)
    parser.add_argument("--batch_size",     type=int,   default=16)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--weight_decay",   type=float, default=1e-4)
    parser.add_argument("--patience",       type=int,   default=5)
    parser.add_argument("--save_dir",       default="./checkpoints/ffe")
    parser.add_argument("--device",         type=int,   default=0)
    parser.add_argument("--seed",           type=int,   default=42)
    args = parser.parse_args()
    train(args)
```

### Run command

```bash
python train_ffe.py \
  --data_path ./data/mvtec_anomaly_detection/ \
  --backbone   dinov2_vitl14 \
  --feature_layers 5 11 17 23 \
  --epochs     30 \
  --batch_size 16 \
  --save_dir   ./checkpoints/ffe
```

### Expected training time (one RTX 3090)

| Mode | Time/epoch | Total (30 epochs) |
|------|-----------|-------------------|
| Live DINOv2 forward pass | ~8 min | ~4 hours |
| Cached DINOv2 features (see below) | ~3 min | ~1.5 hours |

### Optional: offline feature caching

DINOv2 is frozen and its output is deterministic. Extract and save all feature maps
once, then train the Linear on the cache. Reduces per-epoch time by ~60%.

```bash
# Step 1 — extract once (run once, takes ~20 min)
python cache_dino_features.py \
  --data_path ./data/mvtec_anomaly_detection/ \
  --backbone dinov2_vitl14 \
  --feature_layers 5 11 17 23 \
  --save_dir ./cache/mvtec_dino

# Step 2 — train on cache
python train_ffe.py --use_cache --cache_dir ./cache/mvtec_dino ...
```

### Sanity check after training

```python
# Run before the ablation sweep to confirm the Linear actually learned something.
import torch
from models.modules.ffe_adapter import FFEAdapter

for i in range(4):
    adapter = FFEAdapter(embed_dim=1024, window_size=3, lam=1.0)
    adapter.linear.load_state_dict(
        torch.load(f"checkpoints/ffe/ffe_layer_{i}.pt", map_location="cpu")
    )
    W = adapter.linear.weight.data
    deviation = (W - torch.eye(1024)).norm().item()
    print(f"Layer {i} (block {[5,11,17,23][i]}): weight deviation from identity = {deviation:.4f}")

# Healthy range: 0.1 – 2.0
# < 0.01  → adapter barely trained; check LR or loss scale
# > 5.0   → adapter over-trained; reduce epochs or add weight decay
```

---

## Phase 3 — `models/modules/_LNAMD.py` (Modified)

### What changes

The adapter is inserted inside `_embed()` after per-layer LayerNorm and before
`PatchMaker.patchify()`. This is the only correct insertion point: the feature
map is still spatially intact (B, D, H, W), and the change happens before the
`detach().cpu()` call at line 128.

**Lines changed: 2 added, 1 changed (`for` → `for i,`). Zero logic altered.**

### Exact diff

```diff
 class LNAMD(torch.nn.Module):
-    def __init__(self, device, feature_dim=1024, feature_layer=[1,2,3,4], r=3, patchstride=1):
+    def __init__(self, device, feature_dim=1024, feature_layer=[1,2,3,4], r=3, patchstride=1,
+                 ffe_adapters=None):
         super(LNAMD, self).__init__()
         self.device = device
         self.r = r
         self.patch_maker = PatchMaker(r, stride=patchstride)
         self.LNA = Preprocessing(feature_layer, feature_dim)
+        self.ffe_adapters = ffe_adapters   # nn.ModuleList of FFEAdapter, or None

     def _embed(self, features):
         B = features[0].shape[0]

         features_layers = []
-        for feature in features:
+        for i, feature in enumerate(features):
             feature = feature[:, 1:, :]
             feature = feature.reshape(feature.shape[0],
                                       int(math.sqrt(feature.shape[1])),
                                       int(math.sqrt(feature.shape[1])),
                                       feature.shape[2])
             feature = feature.permute(0, 3, 1, 2)
             feature = torch.nn.LayerNorm([feature.shape[1], feature.shape[2],
                                           feature.shape[3]]).to(self.device)(feature)
+            if self.ffe_adapters is not None:
+                feature = self.ffe_adapters[i](feature)
             features_layers.append(feature)

         # remainder of _embed() is unchanged
```

---

## Phase 4 — `models/musc.py` (Modified)

### What changes

1. A new private method `_load_ffe_adapters()` in `MuSc.__init__` chain.
2. `ffe_adapters` passed to every `LNAMD_r` instantiation in `make_category_data()`.

### `_load_ffe_adapters()` — new method

Add this method to the `MuSc` class and call it at the end of `load_backbone()`:

```python
def load_backbone(self):
    if 'dino' in self.model_name:
        self.dino_model = _backbones.load(self.model_name)
        self.dino_model.to(self.device)
        self.preprocess = None
    else:
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, self.image_size, pretrained=self.pretrained
        )
        self.clip_model.to(self.device)
    self.ffe_adapters = self._load_ffe_adapters()   # ← new call


def _load_ffe_adapters(self):
    """Load pretrained FFE adapter weights. Returns frozen nn.ModuleList or None."""
    ffe_cfg = self.cfg.get('ffe_adapter', {})
    if not ffe_cfg.get('enabled', False):
        return None

    from models.modules.ffe_adapter import FFEAdapter

    ckpt_dir  = ffe_cfg['checkpoint_dir']
    lam       = ffe_cfg.get('lam', 0.5)
    win_size  = ffe_cfg.get('window_size', 3)
    embed_dim = ffe_cfg.get('embed_dim', 1024)
    n_layers  = len(self.features_list)

    adapters = torch.nn.ModuleList()
    for i in range(n_layers):
        adapter = FFEAdapter(embed_dim=embed_dim, window_size=win_size, lam=lam)
        ckpt_path = os.path.join(ckpt_dir, f"ffe_layer_{i}.pt")
        state = torch.load(ckpt_path, map_location=self.device)
        adapter.linear.load_state_dict(state)
        adapter.eval()
        for p in adapter.parameters():
            p.requires_grad_(False)
        adapters.append(adapter)

    adapters.to(self.device)
    print(f"FFE adapters loaded ({n_layers} layers, lam={lam}) from {ckpt_dir}")
    return adapters
```

### Change in `make_category_data()`

```diff
-LNAMD_r = LNAMD(device=self.device, r=r, feature_dim=feature_dim, feature_layer=self.features_list)
+LNAMD_r = LNAMD(device=self.device, r=r, feature_dim=feature_dim,
+                feature_layer=self.features_list,
+                ffe_adapters=self.ffe_adapters)
```

---

## Phase 5 — `configs/musc.yaml` (Modified)

Add the following block anywhere after the existing keys:

```yaml
ffe_adapter:
  enabled: false                      # flip to true after training completes
  checkpoint_dir: ./checkpoints/ffe   # directory containing ffe_layer_0..3.pt
  window_size: 3                      # must match --window_size used in training
  lam: 0.5                            # residual mix: 0.0=off, 1.0=full adapter
  embed_dim: 1024                     # must match backbone embed_dim
```

The `enabled: false` guard means the integration changes to `_LNAMD.py` and
`musc.py` are live but dormant. The unchanged zero-shot pipeline runs identically
to the current codebase until this flag is flipped. Zero regression risk.

---

## Phase 6 — Ablation and Verification

Run the following sequence after training. Each row is one full MuSc inference run:

| Run | `enabled` | `lam` | Purpose |
|-----|-----------|-------|---------|
| A | false | — | Baseline — current MuSc, no adapter |
| B | true | 1.0 | Full adapter, no residual passthrough |
| C | true | 0.5 | 50/50 blend — recommended starting point |
| D | true | 0.3 | Conservative blend |
| E | true | 0.1 | Near-baseline blend |

Compare pixel-AUROC, image-AUROC, and SegF1 across runs A–E on both MVTec AD (test
split) and VisA. The adapter is beneficial if any of B–E beats A on VisA (which was
not seen during training). The best `lam` from this sweep becomes the default in
config.

---

## Implementation Order

Each step is independently testable before the next begins.

```
Step 1  Create models/modules/ffe_adapter.py
        → run __main__ block, confirm shape check passes

Step 2  Run train_ffe.py
        → watch train/val loss curve; val loss should decrease and plateau
        → run sanity check script; confirm deviation in range 0.1–2.0

Step 3  Modify models/modules/_LNAMD.py  (5 lines)
        → run MuSc with enabled:false — output must be byte-identical to baseline

Step 4  Modify models/musc.py  (~35 lines)
        → run MuSc with enabled:false again — still identical to baseline

Step 5  Add ffe_adapter block to configs/musc.yaml  (enabled:false)

Step 6  Flip enabled:true, set lam:1.0 — run full MuSc on MVTec AD
        → confirm pipeline completes without errors

Step 7  Run ablation A–E, record metrics, pick best lam
```

---

## Complete Change Summary

| File | Type | Lines changed | Regression risk |
|------|------|--------------|----------------|
| `models/modules/ffe_adapter.py` | New | ~95 lines | None |
| `train_ffe.py` | New | ~175 lines | None (separate script) |
| `models/modules/_LNAMD.py` | Modified | +3, 1 changed | Very low |
| `models/musc.py` | Modified | +35 lines | Very low |
| `configs/musc.yaml` | Modified | +6 lines | None (`enabled:false`) |
