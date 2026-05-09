#!/usr/bin/env python3
"""
Generate MVTec AD2 competition submission files.

Global threshold 0.628772 from prior test_public run (mvtec2.log).
Processes test_private + test_private_mixed for all 8 categories.
Saves per category, frees memory immediately before next category.

Output layout:
  submission_folder/
    anomaly_images/{cat}/{split}/{idx:03d}_{suffix}.tiff  (float16)
    anomaly_images_thresholded/{cat}/{split}/{idx:03d}_{suffix}.png  (uint8 0/255)
"""

import argparse
import gc
import glob
import os
import sys

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'backbone'
))

import models.backbone.open_clip as open_clip
from models.modules._LNAMD import LNAMD
from models.modules._MSM import MSM

import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# constants
# ------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_CLASSNAMES = [
    "can", "fabric", "fruit_jelly", "rice",
    "sheet_metal", "vial", "wallplugs", "walnuts",
]

GLOBAL_THRESHOLD = 0.628772


# ------------------------------------------------------------------
# dataset for private splits (no GT, just images)
# ------------------------------------------------------------------
class PrivateSplitDataset(torch.utils.data.Dataset):
    """Loads PNG images directly from test_private / test_private_mixed."""

    def __init__(self, data_path, category, split, image_size=518, clip_transformer=None):
        self.split_dir = os.path.join(data_path, category, split)
        self.image_paths = sorted(glob.glob(os.path.join(self.split_dir, "*.png")))
        if not self.image_paths:
            raise FileNotFoundError(f"No PNG images found in {self.split_dir}")

        if clip_transformer is not None:
            self.transform = clip_transformer
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return {
            "image": self.transform(img),
            "image_path": self.image_paths[idx],
        }


# ------------------------------------------------------------------
# core inference
# ------------------------------------------------------------------
def run_inference(dataset, clip_model, features_list, r_list, device, batch_size, image_size):
    """Run LNAMD + MSM on dataset. Returns (anomaly_maps, image_paths).

    Memory-safe: one pass per r-value — no patch_tokens_list accumulation.
    Peak RAM: Z_layers for one r-value (~3.4 GB float16 for 321 imgs at 518px).

    anomaly_maps: np.ndarray (N, 1, image_size, image_size) float32
    """
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    anomaly_maps_r = []   # one (N, L) float32 tensor per r-value
    image_path_list = None
    feature_dim = None

    for r in r_list:
        print(f"  r={r}: extracting features + LNAMD ...")
        paths_this_r = []
        Z_layers = {}  # str(l) -> list of CPU float16 tensors (B, L, C)
        LNAMD_r = None  # built after first batch (need feature_dim)

        for batch in tqdm(dataloader, desc=f"  r={r}", leave=False):
            image = batch["image"]
            paths_this_r.extend(batch["image_path"])

            with torch.no_grad(), torch.cuda.amp.autocast():
                input_img = image.to(torch.float).to(device)
                _, patch_tokens = clip_model.encode_image(input_img, features_list)
                # patch_tokens: list[len(features_list)] of GPU tensors (B, L+1, C)

                if LNAMD_r is None:
                    feature_dim = patch_tokens[0].shape[-1]
                    LNAMD_r = LNAMD(
                        device=device, r=r,
                        feature_dim=feature_dim,
                        feature_layer=features_list,
                    )

                # LNAMD _embed: GPU in, CPU float16 out  (B, L, n_layers, C)
                features = LNAMD_r._embed(patch_tokens)
                features /= features.norm(dim=-1, keepdim=True)

            # accumulate per layer on CPU — patch_tokens_list NOT kept
            for l in range(len(features_list)):
                key = str(l)
                if key not in Z_layers:
                    Z_layers[key] = []
                Z_layers[key].append(features[:, :, l, :])  # (B, L, C) CPU float16

            del patch_tokens, features
            torch.cuda.empty_cache()

        if image_path_list is None:
            image_path_list = paths_this_r

        del LNAMD_r
        gc.collect()

        # ---- MSM: one layer at a time to cap GPU memory ----
        print(f"  r={r}: MSM ...")
        maps_per_layer = []
        for l_key in sorted(Z_layers.keys()):
            Z = torch.cat(Z_layers[l_key], dim=0).to(device)   # (N, L, C) on GPU
            del Z_layers[l_key]
            torch.cuda.empty_cache()

            print(f"    layer-{l_key} ({Z.shape[0]} imgs) ...")
            maps_msm = MSM(Z=Z, device=device, topmin_min=0, topmin_max=0.3)
            maps_per_layer.append(maps_msm.cpu().float())
            del Z, maps_msm
            torch.cuda.empty_cache()

        del Z_layers
        gc.collect()

        # average over layers → (N, L)
        anomaly_maps_r.append(torch.stack(maps_per_layer, dim=0).mean(0))
        del maps_per_layer
        gc.collect()

    # ---- average over r, interpolate ----
    anomaly_maps = torch.stack(anomaly_maps_r, dim=0).mean(0).to(device)   # (N, L)
    del anomaly_maps_r
    B, L = anomaly_maps.shape
    H = int(np.sqrt(L))
    anomaly_maps = F.interpolate(
        anomaly_maps.view(B, 1, H, H),
        size=image_size,
        mode="bilinear",
        align_corners=True,
    )
    result = anomaly_maps.cpu().float().numpy()   # (N, 1, H, W) float32
    del anomaly_maps
    torch.cuda.empty_cache()
    gc.collect()

    return result, image_path_list


# ------------------------------------------------------------------
# save helpers
# ------------------------------------------------------------------
def save_maps(anomaly_maps, image_paths, category, split, submission_dir, threshold):
    """Save float16 tiff + binary png for one category/split."""
    suffix = "regular" if split == "test_private" else "mixed"

    tiff_dir = submission_dir / "anomaly_images" / category / split
    png_dir  = submission_dir / "anomaly_images_thresholded" / category / split
    tiff_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    for idx, (amap, img_path) in enumerate(zip(anomaly_maps, image_paths)):
        stem = Path(img_path).stem              # e.g. "000_regular"
        amap_2d = amap.squeeze()                # (H, W) float32

        # continuous: float16 tiff
        tifffile.imwrite(
            str(tiff_dir / f"{stem}.tiff"),
            amap_2d.astype(np.float16),
        )

        # thresholded: binary uint8 png, 0 / 255
        binary = (amap_2d >= threshold).astype(np.uint8) * 255
        Image.fromarray(binary, mode="L").save(
            str(png_dir / f"{stem}.png")
        )

    print(f"    saved {len(anomaly_maps)} maps -> {tiff_dir}")


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate MVTec AD2 submission")
    parser.add_argument("--data_path",     default="./data/mvtec_ad_2/")
    parser.add_argument("--submission_dir", default="./submission_folder")
    parser.add_argument("--backbone_name", default="ViT-L-14-336")
    parser.add_argument("--pretrained",    default="openai")
    parser.add_argument("--img_resize",    type=int, default=518)
    parser.add_argument("--feature_layers", type=int, nargs="+", default=[5, 11, 17, 23])
    parser.add_argument("--r_list",        type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--batch_size",    type=int, default=4)
    parser.add_argument("--device",        type=int, default=0)
    parser.add_argument("--threshold",     type=float, default=GLOBAL_THRESHOLD)
    parser.add_argument("--classes",       nargs="+", default=_CLASSNAMES)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    submission_dir = Path(args.submission_dir)
    features_list = [l + 1 for l in args.feature_layers]  # 1-indexed as MuSc expects

    print(f"Threshold      : {args.threshold}")
    print(f"Submission dir : {submission_dir}")
    print(f"Device         : {device}")

    # load backbone once
    print("\nLoading backbone ...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.backbone_name, args.img_resize, pretrained=args.pretrained
    )
    clip_model.to(device)
    clip_model.eval()

    splits = ["test_private", "test_private_mixed"]

    for category in args.classes:
        for split in splits:
            print(f"\n[{category}] {split}")
            dataset = PrivateSplitDataset(
                data_path=args.data_path,
                category=category,
                split=split,
                image_size=args.img_resize,
                clip_transformer=preprocess,
            )
            print(f"  {len(dataset)} images")

            anomaly_maps, image_paths = run_inference(
                dataset=dataset,
                clip_model=clip_model,
                features_list=features_list,
                r_list=args.r_list,
                device=device,
                batch_size=args.batch_size,
                image_size=args.img_resize,
            )

            save_maps(
                anomaly_maps=anomaly_maps,
                image_paths=image_paths,
                category=category,
                split=split,
                submission_dir=submission_dir,
                threshold=args.threshold,
            )

            del anomaly_maps, image_paths, dataset
            gc.collect()
            torch.cuda.empty_cache()

    print("\nAll categories done.")
    print(f"Submission folder: {submission_dir.resolve()}")
    print("\nRun checker:")
    print(f"  cd MVTecAD2_public_code_utils && python check_and_prepare_data_for_upload.py ../{submission_dir}")


if __name__ == "__main__":
    main()
