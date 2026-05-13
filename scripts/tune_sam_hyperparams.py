#!/usr/bin/env python3
"""
SAM hyperparameter search for MuSc + cascaded SAM refinement.

Backbone is loaded once. SAM refiner is re-instantiated per trial (cheap).
Results are logged to a CSV and JSON file after every trial.

Default mode: random search (--n_trials N).
Set --mode grid for exhaustive grid search (can be very slow).

Key hyperparameters swept:
  k_pos         -- # positive (foreground) prompt points
  k_neg         -- # negative (background) prompt points in dilation ring
  min_spacing   -- minimum pixel spacing between sampled prompt points
  dilation_kernel -- ellipse kernel size (px) for generating the negative ring
"""

import argparse
import csv
import gc
import itertools
import json
import os
import random
import sys
import time
from datetime import datetime

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
from utils.metrics import find_best_threshold, compute_segf1_at_threshold

import warnings
warnings.filterwarnings("ignore")

_ALL_CLASSNAMES = [
    "can", "fabric", "fruit_jelly", "rice",
    "sheet_metal", "vial", "wallplugs", "walnuts",
]

# Default subset: covers low (can), mid (fabric), high (vial, walnuts) performers
_DEFAULT_CLASSES = ["can", "fabric", "vial", "walnuts"]

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
SEARCH_SPACE = {
    # Positive prompt points. More = better multi-region coverage.
    # Too many risks pulling in non-anomalous pixels.
    "k_pos": [2, 3, 5, 7, 10],

    # Negative prompt points in the background ring.
    # Ring size limits how many can actually be placed.
    "k_neg": [2, 3, 5, 7],

    # Minimum spacing (px) between any two sampled points.
    # Low = cluster around brightest spot (ok for pinpoint defects).
    # High = spread across region (ok for large defects like fabric/walnuts).
    "min_spacing": [10, 20, 30, 45, 60],

    # Dilation kernel (ellipse, px). Determines ring width.
    # Small = ring hugs anomaly edge (precise context, few neg slots).
    # Large = ring extends further out (more neg slots, less precise).
    "dilation_kernel": [10, 15, 25, 35, 50],
}


def build_grid():
    keys = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    combos = list(itertools.product(*values))
    return [dict(zip(keys, c)) for c in combos]


def random_trials(n, seed=42):
    rng = random.Random(seed)
    grid = build_grid()
    if n >= len(grid):
        return grid
    return rng.sample(grid, n)


def run_trial(params, model, preprocess, backbone_type, features_list,
              r_list, device, batch_size, img_resize, data_path,
              sam_version, sam_checkpoint, sam_model_type, sam3_model_id,
              classes=None):
    """Run one HP combo, return dict with per-category and mean segF1."""
    if classes is None:
        classes = _DEFAULT_CLASSES
    import datasets.mvtec_ad2 as mvtec_ad2
    from models.sam_refiner import create_sam_refiner

    # Build SAM refiner with current HPs
    sam_kwargs = dict(
        k_pos=params["k_pos"],
        k_neg=params["k_neg"],
        min_spacing_px=params["min_spacing"],
        dilation_kernel=params["dilation_kernel"],
        device=str(device),
    )
    if sam_version == "sam1":
        sam_refiner = create_sam_refiner(
            "sam1",
            checkpoint_path=sam_checkpoint,
            model_type=sam_model_type,
            **sam_kwargs,
        )
    else:
        sam_refiner = create_sam_refiner(
            "sam3",
            model_id=sam3_model_id,
            **sam_kwargs,
        )

    cat_maps = {}
    cat_gt = {}
    pr_samps = []
    gt_samps = []

    for category in classes:
        try:
            dataset = mvtec_ad2.MVTecAD2Dataset(
                source=data_path,
                classname=category,
                split=mvtec_ad2.DatasetSplit.TEST,
                resize=img_resize,
                imagesize=img_resize,
                clip_transformer=preprocess,
            )
        except Exception:
            continue
        if len(dataset) == 0:
            continue

        anomaly_maps, image_paths, gt_masks = run_inference(
            dataset=dataset,
            model=model,
            backbone_type=backbone_type,
            features_list=features_list,
            r_list=r_list,
            device=device,
            batch_size=batch_size,
            image_size=img_resize,
            with_masks=True,
        )

        if gt_masks is None or gt_masks.sum() == 0:
            continue

        anomaly_maps = sam_refine_maps(anomaly_maps, image_paths, sam_refiner, img_resize)

        gt_i32 = gt_masks.astype(np.int32)
        cat_maps[category] = anomaly_maps
        cat_gt[category] = gt_i32

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

    # Clean up SAM refiner to free VRAM before next trial
    del sam_refiner
    gc.collect()
    torch.cuda.empty_cache()

    if not pr_samps:
        return None

    combined_pr = np.concatenate(pr_samps)
    combined_gt = np.concatenate(gt_samps)
    global_thr = find_best_threshold(combined_gt, combined_pr)

    result = dict(params)
    result["global_thr"] = float(global_thr)

    segf1_global_ls = []
    segf1_cls_ls = []

    for category in classes:
        if category not in cat_maps:
            result[f"{category}_segf1_global"] = None
            result[f"{category}_segf1_cls"] = None
            continue

        pr_px = cat_maps[category]
        gt_px = cat_gt[category]

        segf1_g = compute_segf1_at_threshold(gt_px, pr_px, global_thr)
        cat_thr = find_best_threshold(gt_px.ravel(), pr_px.ravel())
        segf1_c = compute_segf1_at_threshold(gt_px, pr_px, cat_thr)

        segf1_global_ls.append(segf1_g)
        segf1_cls_ls.append(segf1_c)

        result[f"{category}_segf1_global"] = float(segf1_g)
        result[f"{category}_segf1_cls"] = float(segf1_c)

    n = len(segf1_global_ls)
    result["mean_segf1_global"] = float(sum(segf1_global_ls) / n) if n else 0.0
    result["mean_segf1_cls"] = float(sum(segf1_cls_ls) / n) if n else 0.0
    result["n_valid_categories"] = n

    return result


def main():
    parser = argparse.ArgumentParser(
        description="SAM hyperparameter search for MuSc cascaded refinement"
    )
    # Backbone / data
    parser.add_argument("--backbone_name", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--pretrained",    default="openai")
    parser.add_argument("--data_path",     default="./data/mvtec_ad_2/")
    parser.add_argument("--img_resize",    type=int, default=512)
    parser.add_argument("--feature_layers",type=int, nargs="+", default=[5, 11, 17, 23])
    parser.add_argument("--r_list",        type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--batch_size",    type=int, default=4)
    parser.add_argument("--device",        type=int, default=0)
    # SAM backend
    parser.add_argument("--sam_version",      default="sam1", choices=["sam1", "sam3"])
    parser.add_argument("--sam_checkpoint",   default="models/sam_vit_h.pth")
    parser.add_argument("--sam_model_type",   default="vit_h")
    parser.add_argument("--sam3_model_id",    default="models/sam3")
    # Search control
    parser.add_argument("--classes",  nargs="+", default=_DEFAULT_CLASSES,
                        choices=_ALL_CLASSNAMES,
                        help="Categories to evaluate. Default is a representative subset of 4.")
    parser.add_argument("--mode",    default="random", choices=["random", "grid"],
                        help="'random' samples --n_trials combos; 'grid' runs full cartesian product")
    parser.add_argument("--n_trials", type=int, default=12,
                        help="Number of random trials (ignored for --mode grid)")
    parser.add_argument("--seed",    type=int, default=42)
    # Output
    parser.add_argument("--output_dir", default="output/sam_tuning")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = os.path.join(args.output_dir, f"results_{timestamp}.csv")
    json_path = os.path.join(args.output_dir, f"results_{timestamp}.json")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    features_list = [l + 1 for l in args.feature_layers]

    print(f"Device   : {device}")
    print(f"Backbone : {args.backbone_name}")
    print(f"SAM      : {args.sam_version}")
    print(f"Output   : {args.output_dir}")

    # Select trials
    if args.mode == "grid":
        trials = build_grid()
    else:
        trials = random_trials(args.n_trials, seed=args.seed)
    print(f"Trials   : {len(trials)}")

    # Print search space
    print("\nSearch space:")
    for k, v in SEARCH_SPACE.items():
        print(f"  {k}: {v}")
    print()

    # Load backbone ONCE
    print("Loading backbone ...")
    model, preprocess, backbone_type = load_backbone(
        args.backbone_name, args.pretrained, args.img_resize, device
    )
    print("Backbone loaded.\n")

    all_results = []
    csv_writer = None
    csv_file = None

    active_classes = args.classes
    print(f"Classes  : {active_classes}\n")

    for trial_idx, params in enumerate(trials):
        print(f"[Trial {trial_idx+1}/{len(trials)}] {params}")
        t0 = time.time()

        try:
            result = run_trial(
                params=params,
                model=model,
                preprocess=preprocess,
                backbone_type=backbone_type,
                features_list=features_list,
                r_list=args.r_list,
                device=device,
                batch_size=args.batch_size,
                img_resize=args.img_resize,
                data_path=args.data_path,
                sam_version=args.sam_version,
                sam_checkpoint=args.sam_checkpoint,
                sam_model_type=args.sam_model_type,
                sam3_model_id=args.sam3_model_id,
                classes=active_classes,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            result = None

        elapsed = time.time() - t0

        if result is None:
            print(f"  Skipped (no valid categories). [{elapsed:.1f}s]")
            continue

        result["trial_idx"] = trial_idx
        result["elapsed_s"] = round(elapsed, 1)
        result["sam_version"] = args.sam_version

        mean_g = result["mean_segf1_global"] * 100
        mean_c = result["mean_segf1_cls"] * 100
        print(f"  mean segF1 global={mean_g:.2f}%  per-cls={mean_c:.2f}%  [{elapsed:.1f}s]")

        all_results.append(result)

        # Append to CSV (open on first write to get header from first result keys)
        if csv_writer is None:
            csv_file = open(csv_path, "w", newline="")
            fieldnames = list(result.keys())
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()
        csv_writer.writerow(result)
        csv_file.flush()

        # Overwrite JSON each trial (atomic partial result)
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)

    if csv_file:
        csv_file.close()

    if not all_results:
        print("\nNo results collected.")
        return

    # Final ranking by mean segF1 (per-cls threshold, more meaningful for binary SAM output)
    ranked = sorted(all_results, key=lambda r: r["mean_segf1_cls"], reverse=True)

    print("\n" + "=" * 70)
    print("Top 10 configurations (ranked by mean segF1 per-class threshold):")
    print("=" * 70)
    header = f"{'Rank':>4}  {'k_pos':>5}  {'k_neg':>5}  {'spacing':>7}  {'dilation':>8}  {'global%':>8}  {'cls%':>6}"
    print(header)
    print("-" * 70)
    for rank, r in enumerate(ranked[:10], 1):
        print(
            f"{rank:>4}  {r['k_pos']:>5}  {r['k_neg']:>5}  "
            f"{r['min_spacing']:>7}  {r['dilation_kernel']:>8}  "
            f"{r['mean_segf1_global']*100:>7.2f}%  {r['mean_segf1_cls']*100:>5.2f}%"
        )

    best = ranked[0]
    print(f"\nBest config:")
    for k in ["k_pos", "k_neg", "min_spacing", "dilation_kernel"]:
        print(f"  {k}: {best[k]}")
    print(f"  mean segF1 (global thr): {best['mean_segf1_global']*100:.2f}%")
    print(f"  mean segF1 (per-cls):    {best['mean_segf1_cls']*100:.2f}%")

    print(f"\nFull results saved:")
    print(f"  CSV : {csv_path}")
    print(f"  JSON: {json_path}")


if __name__ == "__main__":
    main()
