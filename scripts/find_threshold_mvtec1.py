#!/usr/bin/env python3
"""
Find optimal segF1 threshold on MVTec AD1 test split.

Threshold calibrated on MVTec AD1 (public, independent dataset) avoids any
leakage from the MVTec AD2 competition test_public split. The found threshold
is then passed via --threshold to compute_segf1_dinov3.py and generate_submission.py.

Usage:
    python scripts/find_threshold_mvtec1.py
"""

import argparse
import gc
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from scripts.generate_submission import (
    load_backbone,
    run_inference,
    sam_refine_maps,
)
from utils.metrics import find_best_threshold, compute_segf1_at_threshold

import warnings
warnings.filterwarnings("ignore")

_CLASSNAMES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_name", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--pretrained",    default="openai")
    parser.add_argument("--data_path",     default="./data/mvtec_anomaly_detection/")
    parser.add_argument("--img_resize",    type=int, default=512)
    parser.add_argument("--feature_layers",type=int, nargs="+", default=[5, 11, 17, 23])
    parser.add_argument("--r_list",        type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--batch_size",    type=int, default=4)
    parser.add_argument("--device",        type=int, default=0)
    parser.add_argument("--classes",       nargs="+", default=_CLASSNAMES)
    parser.add_argument("--use_sam",       action="store_true", default=True)
    parser.add_argument("--sam_version",   default="sam3", choices=["sam1", "sam3"])
    parser.add_argument("--sam_checkpoint",default="models/sam_vit_h.pth")
    parser.add_argument("--sam_model_type",default="vit_h")
    parser.add_argument("--sam3_model_id", default="models/sam3")
    parser.add_argument("--sam_k_pos",     type=int, default=2)
    parser.add_argument("--sam_k_neg",     type=int, default=5)
    parser.add_argument("--sam_spacing",   type=int, default=60)
    parser.add_argument("--sam_dilation",  type=int, default=15)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    features_list = [l + 1 for l in args.feature_layers]

    print(f"Backbone : {args.backbone_name}")
    print(f"Device   : {device}")
    print(f"Layers   : {features_list}  r_list={args.r_list}")
    print(f"Data     : {args.data_path}")

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
            )
        print(f"SAM{args.sam_version[-1]} refiner loaded.")

    import datasets.mvtec as mvtec

    pr_samps = []
    gt_samps = []
    cat_maps = {}
    cat_gt   = {}

    print("\n" + "=" * 60)
    print("Inference on MVTec AD1 test split")
    print("=" * 60)

    for category in args.classes:
        print(f"\n[{category}]")
        try:
            dataset = mvtec.MVTecDataset(
                source=args.data_path,
                classname=category,
                split=mvtec.DatasetSplit.TEST,
                resize=args.img_resize,
                imagesize=args.img_resize,
                clip_transformer=preprocess,
            )
        except Exception as e:
            print(f"  Skipping: {e}")
            continue
        if len(dataset) == 0:
            print("  No images.")
            continue

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

        if gt_masks is None or gt_masks.sum() == 0:
            print("  No GT mask pixels — skipping.")
            continue

        if sam_refiner is not None:
            anomaly_maps = sam_refine_maps(anomaly_maps, image_paths, sam_refiner, args.img_resize)

        gt_i32 = gt_masks.astype(np.int32)
        cat_maps[category] = anomaly_maps
        cat_gt[category]   = gt_i32

        pr_flat = anomaly_maps.ravel().astype(np.float32)
        gt_flat = gt_i32.ravel()
        n_samp  = min(150_000, len(pr_flat))
        rng     = np.random.default_rng(42)
        idx     = rng.choice(len(pr_flat), n_samp, replace=False)
        pr_samps.append(pr_flat[idx])
        gt_samps.append(gt_flat[idx])

        del dataset, gt_masks
        gc.collect()
        torch.cuda.empty_cache()

    if not pr_samps:
        print("ERROR: no valid categories found.")
        return

    combined_pr = np.concatenate(pr_samps)
    combined_gt = np.concatenate(gt_samps)
    global_thr  = find_best_threshold(combined_gt, combined_pr)

    print(f"\n{'='*60}")
    print(f"Global threshold (MVTec AD1): {global_thr:.6f}")
    print(f"{'='*60}")
    print(f"\n>>> Pass --threshold {global_thr:.6f} to compute_segf1_dinov3.py and generate_submission.py <<<\n")

    print(f"{'Category':14s}  {'Threshold':>9s}  {'segF1':>8s}  {'Per-cls thr':>11s}  {'segF1@cls':>9s}")
    print("-" * 60)

    segf1_global_ls = []
    segf1_cls_ls    = []

    for category in args.classes:
        if category not in cat_maps:
            continue
        pr_px = cat_maps[category]
        gt_px = cat_gt[category]

        segf1_g = compute_segf1_at_threshold(gt_px, pr_px, global_thr)
        cat_thr = find_best_threshold(gt_px.ravel(), pr_px.ravel())
        segf1_c = compute_segf1_at_threshold(gt_px, pr_px, cat_thr)

        segf1_global_ls.append(segf1_g)
        segf1_cls_ls.append(segf1_c)

        print(
            f"{category:14s}  {global_thr:>9.6f}  {segf1_g*100:>7.2f}%"
            f"  {cat_thr:>11.6f}  {segf1_c*100:>8.2f}%"
        )

    n = len(segf1_global_ls)
    if n > 0:
        print("-" * 60)
        print(
            f"{'mean':14s}  {global_thr:>9.6f}  {sum(segf1_global_ls)/n*100:>7.2f}%"
            f"  {'(per-cls)':>11s}  {sum(segf1_cls_ls)/n*100:>8.2f}%"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
