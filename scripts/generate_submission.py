#!/usr/bin/env python3
"""
Generate MVTec AD2 competition submission files.

Modular backbone: CLIP (ViT-*) | DINOv2 (dinov2_*) | DINO (dino_*) |
                  DINOv3 (facebook/dinov3-* via HuggingFace)

Threshold: pass --threshold VALUE to use a fixed value, or omit (default=None)
to auto-compute from test_public images (Phase 1) and report metrics.

Output layout:
  {backbone}_submission_folder/
    anomaly_images/{cat}/{split}/{idx:03d}_{suffix}.tiff  (float16)
    anomaly_images_thresholded/{cat}/{split}/{idx:03d}_{suffix}.png  (uint8 0/255)
"""

import argparse
import gc
import glob
import os
import re
import sys

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "models", "backbone"))

import models.backbone.open_clip as open_clip
import models.backbone._backbones as _backbones
from models.backbone.dinov3_backbone import load_dinov3
from models.modules._LNAMD import LNAMD
from models.modules._MSM import MSM
from utils.metrics import compute_metrics, find_best_threshold, compute_segf1_at_threshold

import warnings
warnings.filterwarnings("ignore")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_CLASSNAMES = [
    "can", "fabric", "fruit_jelly", "rice",
    "sheet_metal", "vial", "wallplugs", "walnuts",
]


# ------------------------------------------------------------------
# backbone helpers
# ------------------------------------------------------------------
def get_backbone_type(name):
    n = name.lower()
    if "dinov3" in n or ("facebook/" in n and "dino" in n):
        return "dinov3"
    if "dinov2" in n:
        return "dinov2"
    if "dino" in n:
        return "dino"
    return "clip"


def get_short_name(backbone_name):
    name = backbone_name.split("/")[-1]
    if "pretrain" in name or "pretrained" in name:
        parts = re.split(r"[-_]", name)
        keep = []
        for p in parts:
            if p.startswith("pretrain"):
                break
            keep.append(p)
        name = "_".join(keep)
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_backbone(backbone_name, pretrained, img_resize, device):
    btype = get_backbone_type(backbone_name)
    print(f"Backbone type  : {btype}  ({backbone_name})")

    if btype == "clip":
        model, _, preprocess = open_clip.create_model_and_transforms(
            backbone_name, img_resize, pretrained=pretrained
        )
        model.to(device).eval()
        return model, preprocess, btype

    if btype == "dinov3":
        model = load_dinov3(backbone_name, device)
        preprocess = model.get_preprocess(img_resize)
        return model, preprocess, btype

    # dino or dinov2
    model = _backbones.load(backbone_name)
    model.to(device).eval()
    preprocess = transforms.Compose([
        transforms.Resize((img_resize, img_resize)),
        transforms.CenterCrop(img_resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return model, preprocess, btype


# ------------------------------------------------------------------
# unified feature extraction
# ------------------------------------------------------------------
def extract_patch_tokens(model, input_img, backbone_type, features_list):
    if backbone_type == "clip":
        _, raw = model.encode_image(input_img, features_list)
        return [raw[i] for i in range(len(features_list))]

    if backbone_type == "dino":
        raw_all = model.get_intermediate_layers(x=input_img, n=max(features_list))
        return [raw_all[l - 1] for l in features_list]

    # dinov2 / dinov3: returns (B, L, C) WITHOUT CLS
    raw = model.get_intermediate_layers(
        x=input_img,
        n=[l - 1 for l in features_list],
        return_class_token=False,
    )
    fake_cls = [torch.zeros_like(t)[:, :1, :] for t in raw]
    return [torch.cat([fake_cls[i], raw[i]], dim=1) for i in range(len(features_list))]


# ------------------------------------------------------------------
# datasets
# ------------------------------------------------------------------
class PrivateSplitDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, category, split, image_size=518, transform=None):
        self.split_dir = os.path.join(data_path, category, split)
        self.image_paths = sorted(glob.glob(os.path.join(self.split_dir, "*.png")))
        if not self.image_paths:
            raise FileNotFoundError(f"No PNG images in {self.split_dir}")
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return {"image": self.transform(img), "image_path": self.image_paths[idx]}


# ------------------------------------------------------------------
# core inference
# ------------------------------------------------------------------
def run_inference(
    dataset, model, backbone_type, features_list, r_list, device,
    batch_size, image_size, with_masks=False,
):
    """LNAMD + MSM inference. Returns (anomaly_maps, image_paths[, gt_masks]).

    Memory-safe: one forward pass per r-value, no patch_tokens_list accumulation.
    Peak RAM approx Z_layers for one r-value (float16).
    """
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
    )

    anomaly_maps_r = []
    image_path_list = None
    collected_masks = [] if with_masks else None

    for r_idx, r in enumerate(r_list):
        print(f"  r={r}: extract + LNAMD ...")
        paths_this_r = []
        Z_layers = {}
        LNAMD_r = None

        for batch in tqdm(dataloader, desc=f"  r={r}", leave=False):
            image = batch["image"]
            paths_this_r.extend(batch["image_path"])

            if with_masks and r_idx == 0 and "mask" in batch:
                collected_masks.append(batch["mask"])

            with torch.no_grad(), torch.cuda.amp.autocast():
                input_img = image.to(torch.float).to(device)
                patch_tokens = extract_patch_tokens(model, input_img, backbone_type, features_list)

                if LNAMD_r is None:
                    feature_dim = patch_tokens[0].shape[-1]
                    LNAMD_r = LNAMD(
                        device=device, r=r,
                        feature_dim=feature_dim,
                        feature_layer=features_list,
                    )

                features = LNAMD_r._embed(patch_tokens)
                features /= features.norm(dim=-1, keepdim=True)

            for l in range(len(features_list)):
                key = str(l)
                if key not in Z_layers:
                    Z_layers[key] = []
                Z_layers[key].append(features[:, :, l, :])

            del patch_tokens, features
            torch.cuda.empty_cache()

        if image_path_list is None:
            image_path_list = paths_this_r

        del LNAMD_r
        gc.collect()

        print(f"  r={r}: MSM ...")
        maps_per_layer = []
        for l_key in sorted(Z_layers.keys()):
            Z = torch.cat(Z_layers[l_key], dim=0).to(device)
            del Z_layers[l_key]
            torch.cuda.empty_cache()
            print(f"    layer-{l_key} ({Z.shape[0]} imgs) ...")
            maps_msm = MSM(Z=Z, device=device, topmin_min=0, topmin_max=0.3)
            maps_per_layer.append(maps_msm.cpu().float())
            del Z, maps_msm
            torch.cuda.empty_cache()

        del Z_layers
        gc.collect()

        anomaly_maps_r.append(torch.stack(maps_per_layer, dim=0).mean(0))
        del maps_per_layer
        gc.collect()

    anomaly_maps = torch.stack(anomaly_maps_r, dim=0).mean(0).to(device)
    del anomaly_maps_r
    B, L = anomaly_maps.shape
    H = int(np.sqrt(L))
    anomaly_maps = F.interpolate(
        anomaly_maps.view(B, 1, H, H), size=image_size, mode="bilinear", align_corners=True,
    )
    result = anomaly_maps.cpu().float().numpy()
    del anomaly_maps
    torch.cuda.empty_cache()
    gc.collect()

    if with_masks:
        gt = torch.cat(collected_masks, dim=0).numpy() if collected_masks else None
        return result, image_path_list, gt
    return result, image_path_list


# ------------------------------------------------------------------
# Phase 1: compute threshold from test_public
# ------------------------------------------------------------------
def compute_threshold_from_public(args, model, preprocess, backbone_type, device, sam_refiner=None):
    import datasets.mvtec_ad2 as mvtec_ad2

    features_list = [l + 1 for l in args.feature_layers]

    print("\n" + "=" * 60)
    print("Phase 1: computing threshold from test_public")
    print("=" * 60)

    auroc_sp_ls, f1_sp_ls, ap_sp_ls = [], [], []
    auroc_px_ls, f1_px_ls, ap_px_ls, aupro_ls = [], [], [], []
    pr_samples, gt_samples = [], []
    valid_cats = []

    for category in args.classes:
        print(f"\n  [{category}]")
        try:
            dataset = mvtec_ad2.MVTecAD2Dataset(
                source=args.data_path,
                classname=category,
                split=mvtec_ad2.DatasetSplit.TEST,
                resize=args.img_resize,
                imagesize=args.img_resize,
                clip_transformer=preprocess,
            )
        except Exception as e:
            print(f"  Skipping ({e})")
            continue
        if len(dataset) == 0:
            print("  No images, skipping.")
            continue

        if args.gamma_tta:
            from utils.gamma_tta import GammaPreprocess, fuse_gamma_heatmaps

            def _make_ds_pub(gamma):
                pp = GammaPreprocess(preprocess, gamma) if gamma != 1.0 else preprocess
                return mvtec_ad2.MVTecAD2Dataset(
                    source=args.data_path, classname=category,
                    split=mvtec_ad2.DatasetSplit.TEST,
                    resize=args.img_resize, imagesize=args.img_resize,
                    clip_transformer=pp,
                )

            _inf_kwargs_pub = dict(
                model=model, backbone_type=backbone_type,
                features_list=features_list, r_list=args.r_list,
                device=device, batch_size=args.batch_size,
                image_size=args.img_resize,
            )
            maps_o, image_paths, gt_masks = run_inference(
                dataset=_make_ds_pub(1.0), with_masks=True, **_inf_kwargs_pub
            )
            maps_b, _ = run_inference(dataset=_make_ds_pub(0.8), **_inf_kwargs_pub)
            maps_d, _ = run_inference(dataset=_make_ds_pub(1.3), **_inf_kwargs_pub)
            anomaly_maps = fuse_gamma_heatmaps([maps_o, maps_b, maps_d])
            del maps_o, maps_b, maps_d
        else:
            anomaly_maps, image_paths, gt_masks = run_inference(
                dataset=dataset,
                model=model,
                backbone_type=backbone_type,
                features_list=features_list,
                r_list=args.r_list,
                device=device,
                batch_size=args.batch_size,
                image_size=args.img_resize,
                with_masks=True,
            )

        gt_px = gt_masks.astype(np.int32) if gt_masks is not None else None
        pr_sp = anomaly_maps.reshape(len(anomaly_maps), -1).max(-1)
        gt_sp = (
            (gt_px.reshape(len(gt_px), -1).max(-1) > 0).astype(np.int32)
            if gt_px is not None else None
        )

        # Metrics computed on raw float heatmaps (before SAM) for valid AUROC/AUPRO
        image_metric, pixel_metric = compute_metrics(gt_sp, pr_sp, gt_px, anomaly_maps)
        auroc_sp, f1_sp, ap_sp = image_metric
        auroc_px, f1_px, ap_px, aupro = pixel_metric

        auroc_sp_ls.append(auroc_sp); f1_sp_ls.append(f1_sp); ap_sp_ls.append(ap_sp)
        auroc_px_ls.append(auroc_px); f1_px_ls.append(f1_px)
        ap_px_ls.append(ap_px);       aupro_ls.append(aupro)
        valid_cats.append(category)

        # SAM refinement for threshold computation only
        if sam_refiner is not None:
            anomaly_maps = sam_refine_maps(anomaly_maps, image_paths, sam_refiner, args.img_resize)

        pr_px = anomaly_maps
        n_px = pr_px.ravel().shape[0]
        n_samp = min(150_000, n_px)
        rng = np.random.default_rng(42)
        idx = rng.choice(n_px, n_samp, replace=False)
        pr_samples.append(pr_px.ravel()[idx].astype(np.float32))
        if gt_px is not None:
            gt_samples.append(gt_px.ravel()[idx].astype(np.int32))

        del anomaly_maps, gt_masks, pr_px, gt_px
        gc.collect()
        torch.cuda.empty_cache()

    if not pr_samples:
        raise RuntimeError("No test_public data found.")

    combined_pr = np.concatenate(pr_samples)
    combined_gt = np.concatenate(gt_samples) if gt_samples else None
    if combined_gt is None or combined_gt.sum() == 0:
        raise RuntimeError("No anomalous pixels in test_public GT.")

    threshold = find_best_threshold(combined_gt, combined_pr)
    del combined_pr, combined_gt, pr_samples, gt_samples

    print(f"\n  Global threshold: {threshold:.6f}")
    print("\n  Per-category metrics (test_public):")
    for i, cat in enumerate(valid_cats):
        print(
            f"  {cat:12s}  "
            f"auroc_px:{auroc_px_ls[i]*100:.1f}  f1_max:{f1_px_ls[i]*100:.1f}  "
            f"ap:{ap_px_ls[i]*100:.1f}  aupro:{aupro_ls[i]*100:.1f}  "
            f"auroc_sp:{auroc_sp_ls[i]*100:.1f}  f1_sp:{f1_sp_ls[i]*100:.1f}"
        )
    n = len(valid_cats)
    if n > 0:
        print(
            f"  {chr(39)+'mean'+chr(39):12s}  "
            f"auroc_px:{sum(auroc_px_ls)/n*100:.1f}  f1_max:{sum(f1_px_ls)/n*100:.1f}  "
            f"ap:{sum(ap_px_ls)/n*100:.1f}  aupro:{sum(aupro_ls)/n*100:.1f}  "
            f"auroc_sp:{sum(auroc_sp_ls)/n*100:.1f}  f1_sp:{sum(f1_sp_ls)/n*100:.1f}"
        )

    return threshold


# ------------------------------------------------------------------
# save helpers
# ------------------------------------------------------------------
def save_maps(anomaly_maps, image_paths, category, split, submission_dir, threshold):
    """Save anomaly maps as tiff (float16) and thresholded png (binary).

    anomaly_maps: (N, 1, H, W) float32 — raw or SAM-refined heatmaps.
                  Saved as float16 tiff; also thresholded at `threshold` for png.
    """
    tiff_dir = submission_dir / "anomaly_images" / category / split
    png_dir  = submission_dir / "anomaly_images_thresholded" / category / split
    tiff_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)

    for amap, img_path in zip(anomaly_maps, image_paths):
        stem    = Path(img_path).stem
        amap_2d = amap.squeeze()
        tifffile.imwrite(str(tiff_dir / f"{stem}.tiff"), amap_2d.astype(np.float16))
        binary  = (amap_2d >= threshold).astype(np.uint8) * 255
        Image.fromarray(binary, mode="L").save(str(png_dir / f"{stem}.png"))

    print(f"    saved {len(anomaly_maps)} maps -> {tiff_dir}")


def sam_refine_maps(anomaly_maps, image_paths, sam_refiner, image_size):
    """Refine float anomaly maps using SAM mask as a spatial gate.

    Multiplies each heatmap by the SAM binary mask so values outside
    the SAM-predicted anomaly region are zeroed out while preserving
    the original float scores inside. Falls back to the raw heatmap
    when SAM produces an empty mask (flat/normal images).

    Args:
        anomaly_maps: (N, 1, H, W) float32 numpy
        image_paths:  list of str, length N
        sam_refiner:  SAMRefiner instance
        image_size:   int, target resize resolution

    Returns:
        (N, 1, H, W) float32 numpy
    """
    from PIL import Image as PILImage
    refined = []
    for amap, path in tqdm(zip(anomaly_maps, image_paths), total=len(anomaly_maps),
                           desc="SAM refine"):
        img_np = np.array(
            PILImage.open(path).convert("RGB").resize(
                (image_size, image_size), PILImage.BILINEAR
            )
        )
        hmap = amap.squeeze()
        m3 = sam_refiner.refine(img_np, hmap)  # (H, W) uint8 {0,1}
        if m3.sum() == 0:
            # SAM found no region (flat/normal heatmap) — keep raw heatmap
            refined_map = hmap
        else:
            refined_map = hmap * m3.astype(np.float32)
        refined.append(refined_map.astype(np.float32)[np.newaxis])
    return np.stack(refined, axis=0)


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate MVTec AD2 submission")

    parser.add_argument("--data_path",      default="./data/mvtec_ad_2/")
    parser.add_argument("--classes",        nargs="+", default=_CLASSNAMES)
    parser.add_argument(
        "--backbone_name", default="facebook/dinov3-vitl16-pretrain-lvd1689m",
        help=(
            "Model: ViT-L-14-336 | dinov2_vitl14 | dino_vitbase16 | "
            "facebook/dinov3-vitl16-pretrain-lvd1689m"
        ),
    )
    parser.add_argument("--pretrained",     default="openai",
                        help="Pretrained weights tag (CLIP only).")
    parser.add_argument("--img_resize",     type=int, default=518,
                        help="Resize resolution. Use 512 for DINOv3 (patch_size=16).")
    parser.add_argument("--feature_layers", type=int, nargs="+", default=[5, 11, 17, 23])
    parser.add_argument("--r_list",         type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--batch_size",     type=int, default=4)
    parser.add_argument("--device",         type=int, default=0)
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Fixed threshold. Omit to auto-compute from test_public.",
    )
    parser.add_argument("--submission_dir", default=None,
                        help="Default: ./{backbone_short}_sam3_parameter_tuned_submission_folder")
    parser.add_argument("--output_dir",     default=None,
                        help="Default: ./output/mvtec_ad2/{backbone_short}")
    parser.add_argument("--use_sam",         action="store_true", default=True,
                        help="Enable SAM cascaded prompt refinement for segmentation.")
    parser.add_argument("--sam_version",     default="sam3", choices=["sam1", "sam3"],
                        help="SAM backend: 'sam1' (segment_anything) or 'sam3' (Sam3TrackerModel).")
    # SAM1 args
    parser.add_argument("--sam_checkpoint",  default="models/sam_vit_h.pth",
                        help="SAM1 checkpoint path (used when --sam_version sam1).")
    parser.add_argument("--sam_model_type",  default="vit_h",
                        help="SAM1 model type: vit_h | vit_l | vit_b.")
    # SAM3 args
    parser.add_argument("--sam3_model_id",   default="models/sam3",
                        help="SAM3 HuggingFace model ID or local path (used when --sam_version sam3).")
    # Shared args
    parser.add_argument("--sam_k_pos",       type=int, default=2)
    parser.add_argument("--sam_k_neg",       type=int, default=5)
    parser.add_argument("--sam_spacing",     type=int, default=60)
    parser.add_argument("--sam_dilation",    type=int, default=15)
    parser.add_argument("--gamma_tta",        action="store_true", default=False,
                        help="Run TTA with gamma=0.8 and gamma=1.3 variants, fuse 0.6/0.2/0.2.")
    parser.add_argument("--sam_iou_threshold", type=float, default=0.4,
                        help="IoU gate: use M2 instead of M3 when IoU(M2,M3) < this value.")

    args = parser.parse_args()

    device        = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    features_list = [l + 1 for l in args.feature_layers]
    short          = get_short_name(args.backbone_name)
    submission_dir = Path(args.submission_dir or f"./{short}_sam3_parameter_tuned_submission_folder")
    output_dir     = Path(args.output_dir     or f"./output/mvtec_ad2/{short}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Backbone       : {args.backbone_name}")
    print(f"Submission dir : {submission_dir}")
    print(f"Output dir     : {output_dir}")
    print(f"Device         : {device}")
    thr_str = "auto (Phase 1)" if args.threshold is None else str(args.threshold)
    print(f"Threshold      : {thr_str}")

    print("\nLoading backbone ...")
    model, preprocess, backbone_type = load_backbone(
        args.backbone_name, args.pretrained, args.img_resize, device
    )

    sam_refiner = None
    if args.use_sam:
        from models.sam_refiner import create_sam_refiner
        if args.sam_version == "sam1":
            sam_refiner = create_sam_refiner(
                "sam1",
                checkpoint_path=args.sam_checkpoint,
                model_type=args.sam_model_type,
                device=str(device),
                k_pos=args.sam_k_pos,
                k_neg=args.sam_k_neg,
                min_spacing_px=args.sam_spacing,
                dilation_kernel=args.sam_dilation,
                consistency_iou_threshold=args.sam_iou_threshold,
            )
        else:
            sam_refiner = create_sam_refiner(
                "sam3",
                model_id=args.sam3_model_id,
                device=str(device),
                k_pos=args.sam_k_pos,
                k_neg=args.sam_k_neg,
                min_spacing_px=args.sam_spacing,
                dilation_kernel=args.sam_dilation,
                consistency_iou_threshold=args.sam_iou_threshold,
            )
        print(f"SAM{args.sam_version[-1]} refiner loaded.")

    if args.threshold is None:
        threshold = compute_threshold_from_public(
            args, model, preprocess, backbone_type, device, sam_refiner=sam_refiner
        )
    else:
        threshold = args.threshold

    print(f"\nThreshold used : {threshold:.6f}")

    print("\n" + "=" * 60)
    print("Phase 2: generating submission files")
    print("=" * 60)

    for category in args.classes:
        for split in ["test_private", "test_private_mixed"]:
            print(f"\n[{category}] {split}")
            dataset = PrivateSplitDataset(
                data_path=args.data_path,
                category=category,
                split=split,
                image_size=args.img_resize,
                transform=preprocess,
            )
            print(f"  {len(dataset)} images")

            if args.gamma_tta:
                from utils.gamma_tta import GammaPreprocess, fuse_gamma_heatmaps

                def _make_ds_priv(gamma):
                    pp = GammaPreprocess(preprocess, gamma) if gamma != 1.0 else preprocess
                    return PrivateSplitDataset(
                        data_path=args.data_path,
                        category=category,
                        split=split,
                        image_size=args.img_resize,
                        transform=pp,
                    )

                _inf_kwargs_priv = dict(
                    model=model, backbone_type=backbone_type,
                    features_list=features_list, r_list=args.r_list,
                    device=device, batch_size=args.batch_size,
                    image_size=args.img_resize,
                )
                maps_o, image_paths = run_inference(dataset=_make_ds_priv(1.0), **_inf_kwargs_priv)
                maps_b, _ = run_inference(dataset=_make_ds_priv(0.8), **_inf_kwargs_priv)
                maps_d, _ = run_inference(dataset=_make_ds_priv(1.3), **_inf_kwargs_priv)
                anomaly_maps = fuse_gamma_heatmaps([maps_o, maps_b, maps_d])
                del maps_o, maps_b, maps_d
            else:
                anomaly_maps, image_paths = run_inference(
                    dataset=dataset,
                    model=model,
                    backbone_type=backbone_type,
                    features_list=features_list,
                    r_list=args.r_list,
                    device=device,
                    batch_size=args.batch_size,
                    image_size=args.img_resize,
                )

            if sam_refiner is not None:
                anomaly_maps = sam_refine_maps(
                    anomaly_maps, image_paths, sam_refiner, args.img_resize
                )

            save_maps(
                anomaly_maps=anomaly_maps,
                image_paths=image_paths,
                category=category,
                split=split,
                submission_dir=submission_dir,
                threshold=threshold,
            )

            del anomaly_maps, image_paths, dataset
            gc.collect()
            torch.cuda.empty_cache()

    print("\nAll categories done.")
    print(f"Submission folder : {submission_dir.resolve()}")
    print("\nRun checker:")
    print(f"  cd MVTecAD2_public_code_utils && \\")
    print(f"  python check_and_prepare_data_for_upload.py ../{submission_dir}")


if __name__ == "__main__":
    main()
