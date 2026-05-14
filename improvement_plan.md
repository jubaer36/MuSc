# Plan: Improve SegF1 on Mixed Private Set

## Context

Current pipeline: DINOv3-MuSc → raw heatmap → single Otsu/percentile mask → global top-k pos points + ring neg points → 3-pass SAM cascade → M3.

Private-mixed set failures trace to two root causes:
1. **Unstable normalization**: lighting/exposure shifts change score distribution → fixed threshold or raw top-k becomes unreliable
2. **Poor prompt quality**: positives sampled globally (not restricted to R), negatives sit immediately on uncertain boundary

These changes touch one file only: `models/sam_refiner.py` (base class + both refiner classes). Config and CLI scripts need minor param additions.

---

## Step 1 — Robust Heatmap Normalization

**File:** `models/sam_refiner.py` — `_get_anomaly_region()` (line 46) and new helper used before any thresholding.

Replace current min-max normalization in `_get_anomaly_region` with MAD/median → then min-max:

```python
def _robust_normalize(self, heatmap):
    """Median-MAD normalization followed by min-max to [0,1]."""
    eps = 1e-8
    med = float(np.median(heatmap))
    mad = float(np.median(np.abs(heatmap - med))) + eps
    a = (heatmap - med) / mad
    a_min, a_max = a.min(), a.max()
    if a_max - a_min < eps:
        return np.zeros_like(heatmap, dtype=np.float32)
    return ((a - a_min) / (a_max - a_min)).astype(np.float32)
```

Call `_robust_normalize` at the top of `_build_point_arrays` and pass the result downstream. `_get_anomaly_region` continues to receive the normalized map.

---

## Step 2 — Multi-Threshold Regions

**File:** `models/sam_refiner.py` — new method `_get_threshold_regions()` in `_SAMRefinerBase`.

Instead of one binary mask, produce three:

```python
def _get_threshold_regions(self, norm_heatmap):
    """Returns (R_h, R_m, R_l) binary masks at P99, P95, P90."""
    Rh = (norm_heatmap >= np.percentile(norm_heatmap, 99)).astype(np.uint8)
    Rm = (norm_heatmap >= np.percentile(norm_heatmap, 95)).astype(np.uint8)
    Rl = (norm_heatmap >= np.percentile(norm_heatmap, 90)).astype(np.uint8)
    return Rh, Rm, Rl
```

Keep `_get_anomaly_region` as fallback for SAM3 text rescue check (max coverage guard).

---

## Step 3 — Component-Aware Positive Points

**File:** `models/sam_refiner.py` — `_build_point_arrays()` (line 95).

Current code samples positives from the **entire heatmap** without a mask. Fix: use R_m as the sampling mask. This naturally restricts points to connected anomaly components without requiring complex centroid/farthest-point logic.

```python
pos_coords = self._sample_points_with_spacing(
    norm_heatmap, self.k_pos, self.min_spacing_px, mask=R_m.astype(bool)
)
```

If R_m produces zero points (weak anomaly), fall back to R_l as mask.

---

## Step 4 — Safe Negative Ring

**File:** `models/sam_refiner.py` — `_build_point_arrays()` (line 113).

Current ring: `dilate(R) - R` — sits right on uncertain boundary.
New ring: `dilate(R_m) - R_l` — avoids uncertain boundary pixels between R_m and R_l.

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilation_kernel, self.dilation_kernel))
dilated_rm = cv2.dilate(R_m, kernel)
ring = ((dilated_rm.astype(np.int32) - R_l.astype(np.int32)) > 0)
neg_score_map = np.where(ring, -norm_heatmap, -np.inf)
neg_coords = self._sample_points_with_spacing(
    neg_score_map, self.k_neg, self.min_spacing_px, mask=ring
)
```

---

## Step 5 — Box from M2 ∩ R_l (with fallback)

**File:** `models/sam_refiner.py` — inside `refine()` of both `SAMRefiner` (line 210) and `SAM3Refiner` (line 354).

After pass 2, intersect M2 with R_l to prevent overly large bounding boxes:

```python
M2_gated = M2 & R_l.astype(bool)
box = self._get_bbox_from_mask(M2_gated, norm_heatmap)
if box is None:
    box = self._get_bbox_from_mask(M2, norm_heatmap)  # fallback: plain M2
```

Pass `norm_heatmap` (not raw) to `_get_bbox_from_mask` so component scoring is consistent with normalized values.

---

## Step 6 — Cascade Consistency Gate

**File:** `models/sam_refiner.py` — inside `refine()` of both `SAMRefiner` and `SAM3Refiner`, after pass 3.

If M3 diverges too much from M2, it degraded — fall back to M2:

```python
def _iou(self, a, b):
    inter = float((a & b).sum())
    union = float((a | b).sum())
    return inter / union if union > 0 else 0.0

# After pass 3:
if self._iou(M2, M3) < self.consistency_iou_threshold:
    M_final = M2
else:
    M_final = M3
```

Add `consistency_iou_threshold: float = 0.4` as constructor param and config field.

---

## Step 7 — SAM3 Text Rescue (Deferred)

**Status: out of scope for this PR.**

The current `SAM3Refiner` uses `Sam3TrackerModel` from HuggingFace transformers. This model class does not expose a text-prompt interface — it takes `pixel_values`, `input_points`, `input_labels`, `input_boxes`, `input_masks`. Adding text-grounded segmentation would require a different model (`Sam3Model` or a CLIP-guided variant), separate preprocessing, and a different HuggingFace model ID. This risks breaking the working SAM3 cascade without clear gain. Skip for now; implement separately when the text-prompt API is confirmed working.

---

## Files to Modify

| File | Changes |
|------|---------|
| `models/sam_refiner.py` | Add `_robust_normalize`, `_get_threshold_regions`, `_iou`; rewrite `_build_point_arrays`; update `refine()` in both SAMRefiner and SAM3Refiner |
| `configs/musc.yaml` | Add `consistency_iou_threshold: 0.4` under `sam:` |
| `scripts/compute_segf1_dinov3.py` | Pass `consistency_iou_threshold` to `create_sam_refiner` |
| `scripts/generate_submission.py` | Same as above |

---

## Implementation Order (priority high → low)

1. `_robust_normalize` + thread through `_build_point_arrays` (Step 1)
2. `_get_threshold_regions` + component-aware positives (Steps 2–3)
3. Safe negative ring (Step 4)
4. Box from M2 ∩ R_l with fallback (Step 5)
5. IoU consistency gate (Step 6)

Steps 1–5 all live in `models/sam_refiner.py`. Do in one pass.

---

## Verification

```bash
# Quick sanity run on 2 categories
conda run -n clip python3 scripts/compute_segf1_dinov3.py \
    --backbone_name facebook/dinov3-vitl16-pretrain-lvd1689m \
    --data_path ./data/mvtec_ad_2/ --img_resize 512 \
    --use_sam --sam_checkpoint models/sam_vit_h.pth \
    --category vial walnuts

# Full eval
conda run -n clip python3 scripts/compute_segf1_dinov3.py \
    --backbone_name facebook/dinov3-vitl16-pretrain-lvd1689m \
    --data_path ./data/mvtec_ad_2/ --img_resize 512 \
    --use_sam --sam_checkpoint models/sam_vit_h.pth
```

Target: mean SegF1 > 7.74% (current baseline). Watch vial and walnuts first — they have highest baseline so improvements should be most visible there. `can` (0%) and `sheet_metal` (0.6%) may improve more from robust normalization than from SAM cascade changes.
