# Plan: DINOv3 + CLIP Dual-MuSc — Intersection/Union Point Fusion (Ensemble)

## Strategy: R_pos = R_dino ∩ R_clip, Neg ring = outside R_dino ∪ R_clip

Pos points from pixels BOTH models flag anomalous.  
Neg ring sits outside BOTH models' estimated regions.  
Structurally prevents pos/neg contradiction. Point count stays same (k_pos + k_neg, not doubled).

---

## Architecture

```
DINOv3 (512px) → LNAMD+MSM → H_dino → R_dino
                                              \
                                               R_pos = R_dino ∩ R_clip  → pos points (k_pos, combined score)
                                              /                           \
CLIP ViT-L (336px) → LNAMD+MSM → H_clip → R_clip                        → SAM3 3-pass cascade (unchanged)
                                              \                           /
                                               R_union = R_dino ∪ R_clip → neg ring → neg points (k_neg)
```

Combined score = H_dino_norm + H_clip_norm (both normalized [0,1] per-image independently).

---

## Resolution Handling

DINOv3 ViT-L/16 @ 512px → 32×32 patch grid → H_dino upsampled to 512×512  
CLIP ViT-L/14 @ 336px → 24×24 patch grid → H_clip upsampled to 512×512 (via output_size param)  

Both heatmaps same shape for pixel-wise ∩ / ∪ operations.  
DataLoader for CLIP uses separate 336px dataset; `run_inference(..., output_size=512)` handles upsampling.

---

## Edge Cases (implemented in _build_dual_point_arrays)

| Condition | Handler |
|-----------|---------|
| R_pos strict ∩ < 0.5% image | Try soft ∩ (3px dilation of each R before intersecting) |
| Soft ∩ also < 0.5% | Return (None, None) — SAM skipped, caller uses raw heatmap |
| R_union > 60% of image | Use R_dino only for neg ring (avoids image-border-only ring) |
| Pos coords = 0 after spacing | Return (None, None) |
| Neg ring empty | Pass pos only to SAM (handled in cascade) |

All cases print diagnostic messages — no silent failures.

---

## Files Changed

| File | What changed |
|------|-------------|
| `models/sam_refiner.py` | Added `_build_dual_point_arrays()` to `_SAMRefinerBase`; updated `refine(heatmap2=None)` in both `SAMRefiner` and `SAM3Refiner` |
| `scripts/generate_submission.py` | Added `output_size` param to `run_inference()`; added `sam_refine_maps_ensemble()`; added `--dual_backbone`, `--clip_model_name`, `--clip_pretrained`, `--clip_img_size`, `--clip_features`, `--output_tag` CLI args; auto output naming `ensemble_dinov3_clip` |
| `scripts/compute_segf1_dinov3.py` | Same CLI args as above; loads CLIP secondary model; runs CLIP inference per category; calls `sam_refine_maps_ensemble` when `--dual_backbone` |
| `configs/musc.yaml` | `sam.version: 'sam3'` (was `sam1`); added `dual_backbone` config block |

---

## Output Naming

| Run | Submission dir | Output dir |
|-----|---------------|-----------|
| Single-model (default) | `{backbone_short}_sam3_submission_folder/` | `output/mvtec_ad2/{backbone_short}/` |
| Dual-model (--dual_backbone) | `ensemble_dinov3_clip_sam3_submission_folder/` | `output/mvtec_ad2/ensemble_dinov3_clip/` |
| Custom tag (--output_tag X) | `X_sam3_submission_folder/` | `output/mvtec_ad2/X/` |

---

## Fusion Strategy

After inference, both DINOv3 and CLIP maps are normalized independently to `[0,1]` per image (min-max), then fused before SAM gating.

| `--fusion` | Formula | Behavior |
|------------|---------|----------|
| `max` (default) | `max(H_dino_n, H_clip_n)` | Flags anomaly if EITHER model fires. Higher recall, may raise FP. |
| `geometric_mean` | `sqrt(H_dino_n × H_clip_n)` | Flags anomaly only where BOTH models agree. Lower FP, may miss weak anomalies. |

SAM fallback (empty mask): returns fused map directly (not raw DINOv3 only).

---

## Run Commands

```bash
# Single-model baseline (no ensemble)
conda run -n clip python3 scripts/compute_segf1_dinov3.py \
    --backbone_name facebook/dinov3-vitl16-pretrain-lvd1689m \
    --data_path ./data/mvtec_ad_2/ --img_resize 512 \
    --use_sam --sam_version sam3 --sam3_model_id models/sam3

# Ensemble — max fusion (default, higher recall)
conda run -n clip python3 scripts/compute_segf1_dinov3.py \
    --backbone_name facebook/dinov3-vitl16-pretrain-lvd1689m \
    --data_path ./data/mvtec_ad_2/ --img_resize 512 \
    --use_sam --sam_version sam3 --sam3_model_id models/sam3 \
    --dual_backbone --fusion max \
    --output_tag ensemble_max

# Ensemble — geometric mean fusion (stricter, lower FP)
conda run -n clip python3 scripts/compute_segf1_dinov3.py \
    --backbone_name facebook/dinov3-vitl16-pretrain-lvd1689m \
    --data_path ./data/mvtec_ad_2/ --img_resize 512 \
    --use_sam --sam_version sam3 --sam3_model_id models/sam3 \
    --dual_backbone --fusion geometric_mean \
    --output_tag ensemble_geomean

# Submission generation — max fusion
conda run -n clip python3 scripts/generate_submission.py \
    --backbone_name facebook/dinov3-vitl16-pretrain-lvd1689m \
    --data_path ./data/mvtec_ad_2/ --img_resize 512 \
    --use_sam --sam_version sam3 --sam3_model_id models/sam3 \
    --dual_backbone --fusion max \
    --output_tag ensemble_max

# Submission generation — geometric mean fusion
conda run -n clip python3 scripts/generate_submission.py \
    --backbone_name facebook/dinov3-vitl16-pretrain-lvd1689m \
    --data_path ./data/mvtec_ad_2/ --img_resize 512 \
    --use_sam --sam_version sam3 --sam3_model_id models/sam3 \
    --dual_backbone --fusion geometric_mean \
    --output_tag ensemble_geomean
```

Expected: SAM3 logs show `[dual]` prefix messages per image — R1/R2 coverage, R_pos coverage, point counts, fusion mode.  
Results saved to `./logs/segf1_dinov3_{output_tag}_{timestamp}.log`.  
AUROC-cls unchanged (RsCIN pathway untouched).

---

## What Not Changed

- SAM 3-pass cascade (pass1: points, pass2: points+logit, pass3: points+box+logit) — unchanged
- RsCIN classification pathway — unchanged
- Single-model path (no `--dual_backbone`) — fully backward compatible

## What Changed (bug fixes applied)

| Issue | Fix |
|-------|-----|
| CLIP map discarded from output (only used for SAM prompts) | Now normalized + fused with DINO map before SAM gating |
| SAM fallback returned raw `H_dino` only | Fallback now returns fused map |
| No normalization before fusion (scale mismatch) | `_minmax_norm` applied per image to each map independently |
| CLIP feature layer indices not +1 converted (off by one layer) | `clip_features_list = [l+1 for l in args.clip_features]` in both scripts |
