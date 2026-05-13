# Methodology: DINOv3-MuSc + Cascaded SAM Prompt Refinement

## 1. Base Paper — MuSc (ICLR 2024)

**Paper:** "MuSc: Zero-Shot Industrial Anomaly Classification and Segmentation with Mutual Scoring of the Unlabeled Images"  
**Key insight:** Normal patches in industrial images find many similar patches across the test batch; anomalous patches find few. No training required.

### 1.1 LNAMD — Local Neighborhood Aggregation with Multiple Degrees

For each patch in a test image, LNAMD aggregates features from its local spatial neighborhood at multiple radii `r ∈ {1, 3, 5}`. This produces multi-scale patch descriptors capable of capturing defects of varying sizes — from pinpoint scratches (small r) to region-level contamination (large r).

### 1.2 MSM — Mutual Scoring Mechanism

Given a batch of unlabeled test images, every patch is scored by counting how many patches across all other images are similar to it (nearest-neighbor matching in feature space). Normal patches score high (many matches); anomalous patches score low (few matches). The complement of this similarity count is the anomaly score. Scoring is mutual: every image scores every other image.

Computed independently per feature layer and per r value, then fused across all 12 combinations (4 layers × 3 r values).

### 1.3 RsCIN — Re-scoring with Constrained Image-level Neighborhood

Post-processing step for image-level classification only. Suppresses false positives on normal images whose anomaly heatmap has noise. Uses image-level CLIP features to form a constrained neighborhood, then re-weights the image-level score. RsCIN is never applied to segmentation outputs.

### 1.4 Original Backbone

Default: `ViT-L-14-336` (CLIP, OpenAI). Also compatible with DINO and DINOv2 variants. Features extracted from layers `[5, 11, 17, 23]` by default.

---

## 2. Our Extension — DINOv3 Backbone

Replaces the CLIP ViT backbone with `facebook/dinov3-vitl16-pretrain-lvd1689m` (DINOv3 ViT-L/16, trained on LVD-1689M).

Feature layers used: `[6, 12, 18, 24]` (matches the 24-layer ViT-L depth for DINOv3).  
Image size: `512×512`.

DINOv3's dense self-supervised pretraining produces richer spatial features than CLIP's image-text contrastive pretraining, benefiting patch-level mutual scoring. On MVTecAD2, this yields stronger baselines for both classification and segmentation before SAM refinement.

---

## 3. Our Extension — MuSc-Guided Cascaded SAM Prompt Refinement

**Goal:** Improve SegF1 (F1-max-segm) by sharpening anomaly segmentation boundaries. Image-level classification (RsCIN score) is left completely unchanged.

**Motivation:** MuSc's heatmap is coarse — it is a sum of patch-level scores interpolated to pixel space. Object boundaries are blurry. SAM (Segment Anything Model) produces crisp object boundaries but needs spatial hints. We use MuSc's statistical heatmap to generate those hints, then run SAM in a 3-pass cascade where each pass feeds its output into the next.

### 3.1 Stage 1 — Coarse Anomaly Heatmap

Input image → DINOv3 → LNAMD (r=1,3,5) → MSM (4 layers × 3 scales) → fused per-pixel anomaly heatmap `H` of shape `(H, W)`. Higher value = more anomalous.

### 3.2 Stage 2 — Prompt Generation from H

**Anomaly region R** (binary mask):
- Normalize H to uint8 range [0, 255].
- Apply Otsu thresholding. If resulting coverage is in [1%, 60%], use this mask.
- Otherwise fall back to top-10% percentile threshold.
- This handles both strong anomalies (Otsu works) and weak/flat signals (percentile fallback).

**Positive points** (foreground prompts for SAM):
- Rank all pixels by descending heatmap score.
- Greedily select up to `k_pos=5` points, enforcing minimum spacing of `min_spacing_px=30` pixels between any two selected points.
- Prevents all positive prompts collapsing onto a single bright spot.

**Negative ring** (background prompts for SAM):
- Dilate R with a 25×25 ellipse kernel (chosen for direction-uniform expansion; square kernel produces jagged corners).
- Subtract R from dilated R to get a ring of pixels immediately outside the anomaly region.
- Select up to `k_neg=5` points from this ring by lowest heatmap score (most confidently normal pixels adjacent to the defect).
- These constrain SAM: foreground confirmed inside, background confirmed just outside.

### 3.3 Stage 3 — SAM Pass 1: Points Only

```
input:  pos_coords (k×2), neg_coords (k×2), image_rgb
output: M1 (H×W bool), logit1 (1×256×256 float32)
```

SAM receives only point prompts. Produces a rough binary mask `M1` and raw spatial logit `logit1`. The logit is a continuous per-pixel SAM confidence score at 256×256 internal resolution, carried forward to the next pass.

### 3.4 Stage 4 — SAM Pass 2: Points + Logit

```
input:  pos_coords, neg_coords, logit1 (mask_input)
output: M2 (H×W bool), logit2 (1×256×256 float32)
```

Same point prompts plus `logit1` as a dense spatial prior (SAM's `mask_input` argument). SAM revises its prediction using both the point hints and its own uncertainty map from pass 1. Boundary quality improves because the logit encodes where SAM was uncertain, and it can revise those pixels with full context still active.

### 3.5 Stage 5 — Bounding Box from M2

Extract connected components of `M2`. For each component, compute mean heatmap score over its pixels. Select the component with highest mean score. Draw tight axis-aligned bounding box in `(x1, y1, x2, y2)` format.

If `M2` is empty, skip pass 3 and return `M2` directly.

This box is grounded in SAM's own refined prediction — not the raw blurry heatmap — so it faithfully reflects the defect's spatial extent.

### 3.6 Stage 6 — SAM Pass 3: Points + Box + Logit

```
input:  pos_coords, neg_coords, box (from M2), logit2 (mask_input)
output: M3 (H×W bool) — final mask
```

Most constrained and most informed prompt combination. The box anchors spatial extent. Points confirm core foreground and adjacent background. Dense logit carries detailed boundary knowledge from passes 1 and 2. `M3` is the final anomaly segmentation mask.

### 3.7 Final Output

| Task | Output |
|------|--------|
| Segmentation (SegF1) | `M3` binary mask (uint8, values {0,1}) |
| Classification (AUROC-cls, F1-max-cls) | MuSc RsCIN score, unchanged |

For submission, `.tiff` files save the raw float heatmap; `.png` files use the SAM binary mask.

---

## 4. Why the Cascade Works

Each pass builds strictly on the previous one:

| Pass | Inputs | What it adds |
|------|--------|-------------|
| 1 | points only | rough shape from spatial hints |
| 2 | points + logit1 | sharpens by incorporating own uncertainty map |
| 3 | points + box(M2) + logit2 | anchors extent, final boundary refinement |

At no stage does SAM receive an arbitrary prior — every input comes from MuSc's statistical heatmap or SAM's own previous output. The cascade is fully self-consistent.

---

## 5. Implementation

### Key Files

| File | Role |
|------|------|
| [models/sam_refiner.py](models/sam_refiner.py) | `SAMRefiner` class — full 3-pass cascade |
| [scripts/compute_segf1_dinov3.py](scripts/compute_segf1_dinov3.py) | Eval script for MVTecAD2 with SAM refinement |
| [scripts/generate_submission.py](scripts/generate_submission.py) | Submission generation, both public and private splits |
| [models/musc.py](models/musc.py) | Core MuSc inference, SAM integrated post-heatmap |
| [configs/musc.yaml](configs/musc.yaml) | Config including SAM section |

### SAM Config (musc.yaml)

```yaml
sam:
  use_sam: False          # off by default; enable with --use_sam flag
  checkpoint: 'models/sam_vit_h.pth'
  model_type: 'vit_h'
  k_pos: 5
  k_neg: 5
  min_spacing_px: 30
  dilation_kernel: 25
```

### Run Command

```bash
conda run -n clip python3 scripts/compute_segf1_dinov3.py \
    --backbone_name facebook/dinov3-vitl16-pretrain-lvd1689m \
    --data_path ./data/mvtec_ad_2/ --img_resize 512 \
    --use_sam --sam_checkpoint models/sam_vit_h.pth
```

---

## 6. Results (MVTecAD2 test_public)

| Config | Mean SegF1 |
|--------|-----------|
| DINOv3-MuSc baseline (no SAM) | baseline |
| DINOv3-MuSc + cascaded SAM | 7.74% (per-class threshold) |

Per-category breakdown (with SAM, per-class threshold):

| Category | SegF1 |
|----------|-------|
| can | 0.00% |
| fabric | 5.49% |
| fruit_jelly | 6.52% |
| rice | 1.05% |
| sheet_metal | 0.60% |
| vial | 21.84% |
| wallplugs | 6.47% |
| walnuts | 19.93% |
| **mean** | **7.74%** |

Note: Global threshold collapses to 1.0 (binary mask output from SAM has only {0,1} values; per-class threshold is more meaningful for this output format).

---

## 7. Design Decisions and Tradeoffs

**Ellipse kernel for dilation (not square):** Uniform radial expansion; square kernel creates jagged corners that bleed into non-adjacent normal regions.

**Negative points from ring, lowest heatmap score:** Points are spatially close to the defect (meaningful context for SAM) but confirmed normal by MuSc statistics. If they were far away, SAM would treat them as irrelevant background.

**Box derived from M2, not from H:** H is blurry; M2 already has SAM-quality boundaries. Using H directly would give a loose box that under-constrains pass 3.

**RsCIN untouched:** SAM operates only on spatial segmentation. Image-level anomaly scores come from a different pathway (CLIP image features + constrained neighborhood re-scoring) and should not be contaminated by segmentation decisions.

**`multimask_output=False`:** SAM can return 3 candidate masks per prompt; we disable this for deterministic single-mask output, which is required for the logit feedback loop.

---

## 8. Competition Context

Dataset: MVTecAD2 (VAND 4.0 Industrial Track)  
Primary metric: SegF1 (F1-max-segm, threshold swept dataset-wide)  
Secondary metric: AUROC-cls (image-level)  
Setting: Zero-shot (no training on anomaly examples)
