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
from models.backbone.dinov3_backbone import load_dinov3
from datasets.mvtec import MVTecDataset, DatasetSplit, _CLASSNAMES
from models.modules.ffe_adapter import FFEAdapter


def get_backbone_type(name: str) -> str:
    n = name.lower()
    if "dinov3" in n or ("facebook/" in n and "dino" in n):
        return "dinov3"
    if "dinov2" in n:
        return "dinov2"
    return "dino"


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
        # DCT over dim=1 (P² spatial positions)
        f_target = adapter.dct(windows.transpose(1, 2)).transpose(1, 2)  # full DCT

    f_masked = f_target.clone()
    f_masked[:, mask_from:, :] = 0.0                # mask high-freq bands

    f_pred = F.gelu(adapter.linear(f_masked))       # predict from masked input

    n = adapter.n
    weights = torch.ones(n, device=feat_map.device)
    weights[mask_from:] = hi_weight
    loss = ((f_pred - f_target) ** 2 * weights.view(1, n, 1)).mean()
    return loss


def layernorm_feature(feat: torch.Tensor) -> torch.Tensor:
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
    return as spatial feature maps matching what LNAMD._embed() produces.

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

    btype = get_backbone_type(args.backbone)
    print(f"Loading backbone ({btype}): {args.backbone}")
    if btype == "dinov3":
        dino = load_dinov3(args.backbone, device)
    else:
        dino = _backbones.load(args.backbone)
        dino.to(device).eval()
        for p in dino.parameters():
            p.requires_grad_(False)

    # layer_indices are 0-indexed block numbers (same convention as musc.yaml feature_layers)
    layer_indices = args.feature_layers
    n_layers = len(layer_indices)
    print(f"DINOv3 blocks (0-indexed): {layer_indices}")

    # lam=1.0 during training (no residual mixing)
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
        print(f"  ffe_layer_{i}.pt  ← {btype} block {layer_indices[i]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",      default="./data/mvtec_anomaly_detection/")
    parser.add_argument("--backbone",       default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--feature_layers", nargs="+", type=int, default=[5, 11, 17, 23])
    parser.add_argument("--embed_dim",      type=int,   default=1024)
    parser.add_argument("--image_size",     type=int,   default=512)
    parser.add_argument("--window_size",    type=int,   default=3)
    parser.add_argument("--mask_from",      type=int,   default=6)
    parser.add_argument("--hi_weight",      type=float, default=3.0)
    parser.add_argument("--epochs",         type=int,   default=20)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--weight_decay",   type=float, default=1e-4)
    parser.add_argument("--patience",       type=int,   default=5)
    parser.add_argument("--save_dir",       default="./checkpoints/ffe")
    parser.add_argument("--device",         type=int,   default=0)
    parser.add_argument("--seed",           type=int,   default=42)
    args = parser.parse_args()
    train(args)
