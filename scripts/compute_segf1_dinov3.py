#!/usr/bin/env python3
"""
Compute segF1 for DINOv3 on MVTecAD2 test_public using a fixed threshold.

Threshold must be passed via --threshold (calibrate on independent dataset
using find_threshold_mvtec1.py or find_threshold_visa.py).
Does NOT regenerate private-split submission files.
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
    get_backbone_type,
    sam_refine_maps,
)
from utils.metrics import compute_segf1_at_threshold

import warnings
warnings.filterwarnings("ignore")

_CLASSNAMES = [
    "can", "fabric", "fruit_jelly", "rice",
    "sheet_metal", "vial", "wallplugs", "walnuts",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_name", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--pretrained",    default="openai")
    parser.add_argument("--data_path",     default="./data/mvtec_ad_2/")
    parser.add_argument("--img_resize",    type=int, default=512)
    parser.add_argument("--feature_layers",type=int, nargs="+", default=[5, 11, 17, 23])
    parser.add_argument("--r_list",        type=int, nargs="+", default=[1 ,3, 5])
    parser.add_argument("--batch_size",    type=int, default=4)
    parser.add_argument("--device",        type=int, default=0)
    parser.add_argument("--classes",       nargs="+", default=_CLASSNAMES)
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
    parser.add_argument("--threshold",       type=float, required=True,
                        help="Fixed segmentation threshold (calibrate via find_threshold_mvtec1.py or find_threshold_visa.py).")
    args = parser.parse_args()

    FIXED_THRESHOLD = args.threshold

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    features_list = [l + 1 for l in args.feature_layers]

    print(f"Backbone : {args.backbone_name}")
    print(f"Device   : {device}")
    print(f"Layers   : {features_list}  r_list={args.r_list}")

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

    import datasets.mvtec_ad2 as mvtec_ad2

    # --- Pass 1: inference per category, store maps + GT ---
    cat_maps  = {}   # category -> (N, 1, H, W) float32 numpy
    cat_gt    = {}   # category -> (N, H, W) int32 numpy

    print("\n" + "=" * 60)
    print("Pass 1: inference on test_public")
    print("=" * 60)

    for category in args.classes:
        print(f"\n[{category}]")
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
        cat_maps[category] = anomaly_maps          # (N, 1, H, W) float32
        cat_gt[category]   = gt_i32               # (N, H, W) int32

        del dataset, gt_masks
        gc.collect()
        torch.cuda.empty_cache()

    if not cat_maps:
        print("ERROR: no valid categories found.")
        return

    print(f"\nFixed threshold: {FIXED_THRESHOLD:.2f}")

    # --- Report table ---
    print("\n" + "=" * 50)
    print(f"{'Category':14s}  {'Threshold':>9s}  {'segF1':>8s}")
    print("=" * 50)

    segf1_ls = []

    for category in args.classes:
        if category not in cat_maps:
            print(f"{'  '+category:14s}  (skipped)")
            continue

        pr_px = cat_maps[category]
        gt_px = cat_gt[category]

        segf1 = compute_segf1_at_threshold(gt_px, pr_px, FIXED_THRESHOLD)
        segf1_ls.append(segf1)

        print(f"{category:14s}  {FIXED_THRESHOLD:>9.2f}  {segf1*100:>7.2f}%")

    n = len(segf1_ls)
    if n > 0:
        print("-" * 50)
        print(f"{'mean':14s}  {FIXED_THRESHOLD:>9.2f}  {sum(segf1_ls)/n*100:>7.2f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
