#!/usr/bin/env python3
"""
SAM hyperparameter grid/random search — full version.

Uses all 8 MVTecAD2 categories. Denser search space than
tune_sam_hyperparams.py; base-case values (k_pos=5, k_neg=5,
min_spacing=30, dilation_kernel=25) are always included.

Modes:
  random (default) -- sample --n_trials combos randomly (seed fixed)
  grid             -- full cartesian product (~192 combos with defaults)

Results appended to CSV + JSON after every trial (safe to interrupt).
Backbone loaded once; SAM refiner re-created per trial and freed.
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

# ---------------------------------------------------------------------------
# Dense search space — base case (5,5,30,25) is present in every dimension
# ---------------------------------------------------------------------------
SEARCH_SPACE = {
    # Positive foreground prompt points.
    # Base=5. Low (2-3) for small/precise defects. High (8-12) for large/diffuse.
    "k_pos": [2, 3, 5, 7, 8, 10, 12],

    # Negative background prompt points in ring.
    # Base=5. Ring geometry naturally caps how many fit.
    "k_neg": [2, 3, 5, 7, 8, 10],

    # Min pixel spacing between any two sampled points.
    # Base=30. Low=clusters (pinpoint), High=spread (region-scale defects).
    "min_spacing": [5, 10, 15, 20, 30, 40],

    # Ellipse dilation kernel size (px) for generating the negative ring.
    # Base=25. Small=tight ring (precise context), Large=wide ring (more neg slots).
    "dilation_kernel": [5, 10, 15, 20, 25, 30, 35],
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
    # Always include the base case
    base = {"k_pos": 5, "k_neg": 5, "min_spacing": 30, "dilation_kernel": 25}
    sample = rng.sample([c for c in grid if c != base], min(n - 1, len(grid) - 1))
    return [base] + sample


def run_trial(params, model, preprocess, backbone_type, features_list,
              r_list, device, batch_size, img_resize, data_path,
              sam_version, sam_checkpoint, sam_model_type, sam3_model_id,
              classes=None):
    """Run one HP combo across all categories; return result dict."""
    if classes is None:
        classes = _ALL_CLASSNAMES
    import datasets.mvtec_ad2 as mvtec_ad2
    from models.sam_refiner import create_sam_refiner

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
        description="SAM full hyperparameter search — all classes, dense space"
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
    parser.add_argument("--sam_version",     default="sam1", choices=["sam1", "sam3"])
    parser.add_argument("--sam_checkpoint",  default="models/sam_vit_h.pth")
    parser.add_argument("--sam_model_type",  default="vit_h")
    parser.add_argument("--sam3_model_id",   default="models/sam3")
    # Class selection
    parser.add_argument("--classes", nargs="+", default=_ALL_CLASSNAMES,
                        choices=_ALL_CLASSNAMES,
                        help="Categories to evaluate. Default: all 8.")
    # Search control
    parser.add_argument("--mode", default="random", choices=["random", "grid"],
                        help=(
                            "'random': sample --n_trials combos (base case always included). "
                            "'grid': full cartesian product "
                            f"({len(build_grid())} combos with default space)."
                        ))
    parser.add_argument("--n_trials", type=int, default=25,
                        help="Random-mode only: number of trials (including base case). "
                             "Default 25 (~5h with all 8 classes at ~12 min/trial).")
    parser.add_argument("--seed", type=int, default=42)
    # Output
    parser.add_argument("--output_dir", default="output/sam_tuning_full")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = os.path.join(args.output_dir, f"results_{timestamp}.csv")
    json_path = os.path.join(args.output_dir, f"results_{timestamp}.json")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    features_list = [l + 1 for l in args.feature_layers]

    # Build trial list
    if args.mode == "grid":
        trials = build_grid()
    else:
        trials = random_trials(args.n_trials, seed=args.seed)

    total_grid = len(build_grid())
    est_min = len(trials) * len(args.classes) * 1.5

    print(f"Device       : {device}")
    print(f"Backbone     : {args.backbone_name}")
    print(f"SAM          : {args.sam_version}")
    print(f"Classes      : {args.classes}  ({len(args.classes)} total)")
    print(f"Mode         : {args.mode}")
    print(f"Trials       : {len(trials)} / {total_grid} total grid combos")
    print(f"Est. runtime : ~{est_min/60:.1f}h  ({est_min:.0f} min @ 1.5 min/class/trial)")
    print(f"Output       : {args.output_dir}")

    print("\nSearch space:")
    for k, v in SEARCH_SPACE.items():
        base_val = {"k_pos": 5, "k_neg": 5, "min_spacing": 30, "dilation_kernel": 25}[k]
        vals_str = ", ".join(
            f"[{x}]" if x == base_val else str(x) for x in v
        )
        print(f"  {k:17s}: [{vals_str}]  (base=[…])")
    print()

    print("Loading backbone ...")
    model, preprocess, backbone_type = load_backbone(
        args.backbone_name, args.pretrained, args.img_resize, device
    )
    print("Backbone loaded.\n")

    all_results = []
    csv_writer = None
    csv_file = None

    for trial_idx, params in enumerate(trials):
        is_base = (params == {"k_pos": 5, "k_neg": 5, "min_spacing": 30, "dilation_kernel": 25})
        tag = " ← BASE" if is_base else ""
        print(f"[Trial {trial_idx+1:>3}/{len(trials)}] {params}{tag}")
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
                classes=args.classes,
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
        result["is_base_case"] = is_base

        mean_g = result["mean_segf1_global"] * 100
        mean_c = result["mean_segf1_cls"] * 100
        print(f"  mean segF1  global={mean_g:.2f}%  per-cls={mean_c:.2f}%  [{elapsed:.1f}s]")

        # Per-category breakdown on same line
        per_cat = "  "
        for cat in args.classes:
            v = result.get(f"{cat}_segf1_cls")
            per_cat += f"{cat}={v*100:.1f}% " if v is not None else f"{cat}=N/A "
        print(per_cat)

        all_results.append(result)

        if csv_writer is None:
            csv_file = open(csv_path, "w", newline="")
            fieldnames = list(result.keys())
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()
        csv_writer.writerow(result)
        csv_file.flush()

        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)

    if csv_file:
        csv_file.close()

    if not all_results:
        print("\nNo results collected.")
        return

    # Rank by mean segF1 per-class threshold
    ranked = sorted(all_results, key=lambda r: r["mean_segf1_cls"], reverse=True)

    print("\n" + "=" * 80)
    print("Top 15 configurations (ranked by mean segF1, per-class threshold):")
    print("=" * 80)
    print(f"{'Rank':>4}  {'k_pos':>5}  {'k_neg':>5}  {'spacing':>7}  {'dilation':>8}  "
          f"{'global%':>8}  {'cls%':>6}  {'note':>6}")
    print("-" * 80)
    for rank, r in enumerate(ranked[:15], 1):
        note = "BASE" if r.get("is_base_case") else ""
        print(
            f"{rank:>4}  {r['k_pos']:>5}  {r['k_neg']:>5}  "
            f"{r['min_spacing']:>7}  {r['dilation_kernel']:>8}  "
            f"{r['mean_segf1_global']*100:>7.2f}%  "
            f"{r['mean_segf1_cls']*100:>5.2f}%  {note:>6}"
        )

    # Show base case rank
    base_rank = next(
        (i + 1 for i, r in enumerate(ranked) if r.get("is_base_case")), None
    )
    if base_rank:
        print(f"\nBase case (5,5,30,25) rank: {base_rank}/{len(ranked)}")

    best = ranked[0]
    print(f"\nBest config:")
    for k in ["k_pos", "k_neg", "min_spacing", "dilation_kernel"]:
        print(f"  {k}: {best[k]}")
    print(f"  mean segF1 (global thr): {best['mean_segf1_global']*100:.2f}%")
    print(f"  mean segF1 (per-cls):    {best['mean_segf1_cls']*100:.2f}%")
    print(f"\nPer-category breakdown (best config):")
    for cat in args.classes:
        v = best.get(f"{cat}_segf1_cls")
        print(f"  {cat:14s}: {v*100:.2f}%" if v is not None else f"  {cat:14s}: N/A")

    print(f"\nFull results saved:")
    print(f"  CSV : {csv_path}")
    print(f"  JSON: {json_path}")


if __name__ == "__main__":
    main()
