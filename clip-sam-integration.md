# SAM Integration in MuSc (clip-sam branch)

## Overview

MuSc is a **zero-shot anomaly detection** method that uses a vision backbone (CLIP, DINOv2, or DINOv3) to produce a per-pixel anomaly heatmap via patch-level self-comparison. The `clip-sam` branch adds a **post-hoc SAM (Segment Anything Model) refinement stage** that sharpens those pixel-level heatmaps using instance-level segmentation masks, without touching the image-level classification logic.

The key principle: MuSc's heatmap is good at locating anomalies but can be spatially blurry. SAM's masks are crisp object-shaped regions. Blending them improves pixel-level metrics (AUROC-px, AUPRO) while leaving image-level metrics (AUROC-sp, RsCIN) untouched.

---

## Architecture

```
Input image
    │
    ▼
MuSc backbone (CLIP / DINOv2 / DINOv3)
    │  patch-level cosine similarity self-comparison
    ▼
Anomaly heatmap  (H×W float32)
    │
    ├──────────────────────────────────────────┐
    │  [image-level path]                       │  [pixel-level path]
    │  max-pool → ac_score                      │
    │  RsCIN classification                     │  SAMRefiner.refine()
    │  → auroc_sp, f1_sp, ap_sp               │  → blended heatmap
    │  (SAM does NOT affect this)              │  → auroc_px, f1_px, aupro
    └──────────────────────────────────────────┘
```

---

## New Files

### `models/sam_refiner.py` — `SAMRefiner` class

Core SAM wrapper. Two entry points:

**`refine(image_rgb, heatmap)`** — refines a single image:
1. Normalize heatmap to `[0, 1]`
2. Call `heatmap_to_prompts()` to find anomaly regions and build SAM prompts
3. Run `SamPredictor` per region
4. Blend result: `alpha * SAM_mask + (1 - alpha) * norm_heatmap`

**`refine_batch(image_path_list, anomaly_maps, image_size)`** — loops over a batch, reloads each image from disk (avoids large RAM cache), returns `(B, 1, H, W)` numpy.

### `utils/prompt_utils.py` — three helper functions

**`heatmap_to_prompts(heatmap, percentile, n_neg)`**
- Threshold heatmap at `percentile` (default 95th) → binary mask
- `scipy.ndimage.label` to find connected anomaly regions
- Per region: peak-score pixel as **positive point**, tight bounding box, `n_neg` random low-score pixels as **negative points**
- Returns a list of prompt dicts (one per region)

**`predict_region(predictor, prompt, H, W)`**
- Runs SAM in three modes: point-only, box-only, combined (points + box)
- Takes best mask per mode by SAM's own confidence score
- Merges via confidence-weighted union: `merged = Σ (w_i * mask_i)`, threshold at 0.5

**`merge_region_masks(region_masks)`**
- Pixel-wise max union across all per-region masks → single `(H, W)` float32 mask

---

## Integration into `models/musc.py`

### `_load_sam(cfg)` (new method)
- Called at end of `__init__`
- If `cfg['sam']['enabled']` is `True`, instantiates `SAMRefiner` with config params
- Otherwise sets `self.sam_refiner = None` → zero overhead on disabled path

### `test()` method changes
- `ac_score` (for RsCIN image-level classification) computed from **raw MuSc maps** before SAM — intentional, SAM operates on a different numeric scale
- After all heatmaps accumulated: if `sam_refiner` is not `None`, calls `refine_batch()` to replace `pr_px`
- Pixel metrics then computed on refined maps; image metrics unchanged

---

## Configuration

`configs/musc.yaml` new block:

```yaml
sam:
  enabled: False                  # flip to True to activate
  checkpoint: './models/sam_vit_h.pth'
  model_type: 'vit_h'            # vit_h | vit_l | vit_b
  threshold_percentile: 95       # percentile to detect anomaly regions
  blend_alpha: 0.3               # weight on SAM mask (0 = MuSc only, 1 = SAM only)
  n_neg: 5                       # negative point prompts sampled per region
```

All params also exposed as CLI flags in `examples/musc_main.py`:

```
--sam_enabled True/False
--sam_checkpoint PATH
--sam_model_type vit_h|vit_l|vit_b
--sam_threshold_percentile FLOAT
--sam_blend_alpha FLOAT
--sam_n_neg INT
```

---

## `scripts/generate_submission.py` changes

- `load_sam_refiner(args, device)` helper: instantiates `SAMRefiner` from CLI args
- `compute_threshold_from_public()` accepts optional `sam_refiner` arg; calls `refine_batch()` after inference per category
- Refined maps used for pixel-level threshold computation and segF1 scoring
- Image-level threshold computation path unchanged

---

## What SAM Contributes vs What MuSc Contributes

| Responsibility | Owner |
|---|---|
| Feature extraction | MuSc backbone (CLIP/DINOv2/DINOv3) |
| Anomaly localization (where is the defect?) | MuSc heatmap |
| Image-level anomaly classification | MuSc + RsCIN |
| Pixel-level mask sharpening (crisp boundaries) | SAM |
| Prompt generation from heatmap | `prompt_utils.py` |
| Final pixel score | Blend: `0.3 * SAM + 0.7 * MuSc` (default) |

---

## Design Decisions

- **`blend_alpha = 0.3`** (default): SAM mask is trusted less than MuSc score. Keeps detection sensitivity; adds spatial precision.
- **Three SAM prompt modes** (point-only, box-only, combined): hedges against SAM failing in any single mode; confidence-weighted merge picks best evidence.
- **RsCIN runs on raw maps**: SAM output is `[0, 1]` binary-ish; max-pooling it for image-level scoring would lose ranking nuance. Raw MuSc score preserves original ranking.
- **SAM disabled by default** (`enabled: False`): pipeline runs exactly as original MuSc when SAM is off — no dependency on `segment_anything` unless activated.
- **Images reloaded per sample in `refine_batch`**: avoids holding full image tensors in GPU memory alongside the model.

---

## Prerequisites to Use SAM

```bash
pip install segment-anything
# download checkpoint (vit_h ~2.4 GB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O ./models/sam_vit_h.pth
```

## Example Command

```bash
python examples/musc_main.py \
  --backbone_name facebook/dinov3-vitl16-pretrain-lvd1689m \
  --dataset_name mvtec_ad \
  --data_path ./data/mvtec_anomaly_detection/ \
  --class_name bottle \
  --img_resize 512 \
  --feature_layers 5 11 17 23 \
  --r_list 1 3 5 \
  --sam_enabled True \
  --sam_checkpoint ./models/sam_vit_h.pth \
  --sam_blend_alpha 0.3
```

Without `--sam_enabled True`, pipeline runs identically to original MuSc.

