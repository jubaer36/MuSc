#!/usr/bin/env python3
"""
Compute segF1 for DINOv3 on MVTecAD2 test_public.

Two modes per category:
  1. Global threshold — optimal F1 threshold found across ALL categories combined
  2. Per-class threshold — optimal F1 threshold found per category independently

Does NOT regenerate private-split submission files.
"""

import argparse
import gc
import os
import sys
import datetime

import numpy as np
import torch


class _Tee:
    """Write to both stream and file simultaneously."""
    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh
    def write(self, data):
        self._stream.write(data)
        self._fh.write(data)
    def flush(self):
        self._stream.flush()
        self._fh.flush()
    def fileno(self):
        return self._stream.fileno()
    def isatty(self):
        return self._stream.isatty()

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from scripts.generate_submission import (
    load_backbone,
    run_inference,
    get_backbone_type,
    sam_refine_maps,
)
from utils.metrics import find_best_threshold, compute_segf1_at_threshold

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
    parser.add_argument("--sam_k_pos",       type=int, default=5)
    parser.add_argument("--sam_k_neg",       type=int, default=5)
    parser.add_argument("--sam_spacing",     type=int, default=30)
    parser.add_argument("--sam_dilation",    type=int, default=25)
    parser.add_argument("--gamma_tta",       action="store_true", default=False,
                        help="Run TTA with gamma=0.8 and gamma=1.3 variants, fuse 0.6/0.2/0.2.")
    parser.add_argument("--log_file",        default=None,
                        help="Path to log file. Default: logs/segf1_<timestamp>.log")
    args = parser.parse_args()

    # Set up logging to file + terminal
    if args.log_file is None:
        os.makedirs("logs", exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = f"logs/segf1_{ts}.log"
    log_fh = open(args.log_file, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)
    print(f"Logging to: {args.log_file}")

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
    pr_samps  = []   # subsampled pixels for global threshold
    gt_samps  = []

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

        if args.gamma_tta:
            from utils.gamma_tta import GammaPreprocess, fuse_gamma_heatmaps

            def _make_ds(gamma):
                pp = GammaPreprocess(preprocess, gamma) if gamma != 1.0 else preprocess
                return mvtec_ad2.MVTecAD2Dataset(
                    source=args.data_path, classname=category,
                    split=mvtec_ad2.DatasetSplit.TEST,
                    resize=args.img_resize, imagesize=args.img_resize,
                    clip_transformer=pp,
                )

            _inf_kwargs = dict(
                model=model, backbone_type=backbone_type,
                features_list=features_list, r_list=args.r_list,
                device=device, batch_size=args.batch_size,
                image_size=args.img_resize,
            )
            maps_o, image_paths, gt_masks = run_inference(
                dataset=_make_ds(1.0), with_masks=True, **_inf_kwargs
            )
            maps_b, _ = run_inference(dataset=_make_ds(0.8), **_inf_kwargs)
            maps_d, _ = run_inference(dataset=_make_ds(1.3), **_inf_kwargs)
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

        if gt_masks is None or gt_masks.sum() == 0:
            print("  No GT mask pixels — skipping.")
            continue

        if sam_refiner is not None:
            anomaly_maps = sam_refine_maps(anomaly_maps, image_paths, sam_refiner, args.img_resize)

        gt_i32 = gt_masks.astype(np.int32)
        cat_maps[category] = anomaly_maps          # (N, 1, H, W) float32
        cat_gt[category]   = gt_i32               # (N, H, W) int32

        pr_flat = anomaly_maps.ravel().astype(np.float32)
        gt_flat = gt_i32.ravel()
        n_samp = min(150_000, len(pr_flat))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pr_flat), n_samp, replace=False)
        pr_samps.append(pr_flat[idx])
        gt_samps.append(gt_flat[idx])

        del dataset, gt_masks
        gc.collect()
        torch.cuda.empty_cache()

    if not pr_samps:
        print("ERROR: no valid categories found.")
        return

    # --- Global threshold ---
    combined_pr = np.concatenate(pr_samps)
    combined_gt = np.concatenate(gt_samps)
    global_thr  = find_best_threshold(combined_gt, combined_pr)
    print(f"\nGlobal threshold (combined test_public): {global_thr:.6f}")

    # --- Report table ---
    print("\n" + "=" * 60)
    print(f"{'Category':14s}  {'Global thr':>10s}  {'segF1@global':>12s}  {'Per-cls thr':>11s}  {'segF1@cls':>9s}")
    print("=" * 60)

    segf1_global_ls = []
    segf1_cls_ls    = []
    cls_thr_ls      = []

    for category in args.classes:
        if category not in cat_maps:
            print(f"{'  '+category:14s}  (skipped)")
            continue

        pr_px = cat_maps[category]
        gt_px = cat_gt[category]

        segf1_g = compute_segf1_at_threshold(gt_px, pr_px, global_thr)

        cat_thr = find_best_threshold(gt_px.ravel(), pr_px.ravel())
        segf1_c = compute_segf1_at_threshold(gt_px, pr_px, cat_thr)

        segf1_global_ls.append(segf1_g)
        segf1_cls_ls.append(segf1_c)
        cls_thr_ls.append(cat_thr)

        print(
            f"{category:14s}  {global_thr:>10.6f}  {segf1_g*100:>11.2f}%"
            f"  {cat_thr:>11.6f}  {segf1_c*100:>8.2f}%"
        )

    n = len(segf1_global_ls)
    if n > 0:
        print("-" * 60)
        print(
            f"{'mean':14s}  {global_thr:>10.6f}  {sum(segf1_global_ls)/n*100:>11.2f}%"
            f"  {'(per-cls)':>11s}  {sum(segf1_cls_ls)/n*100:>8.2f}%"
        )

    print("\nDone.")
    log_fh.close()


if __name__ == "__main__":
    main()
