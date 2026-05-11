# SAM Cascaded Prompt Integration Plan
## MuSc-DINOv3 + SAM → Improved SegF1 on MVTecAD2

---

## 0. Context and Goals

**Baseline:** DINOv3-MuSc produces float heatmaps per image. Global threshold
(from `find_best_threshold` across all categories) binarises them for SegF1.

**Goal:** Replace the raw-heatmap binary mask with a SAM-refined mask M3
produced by a 3-pass cascaded prompt strategy (ClipSAM, arXiv 2510.11028),
adapted for DINOv3-MuSc as the heatmap source.

**Metric:** SegF1 at global dataset-wide threshold — no per-image threshold.
Image-level RsCIN score left completely untouched.

**SAM checkpoint:** `models/sam_vit_h.pth` (ViT-H, already downloaded).
**SAM package:** `segment_anything` available in `clip` conda env.

---

## 1. Files Overview

| Action | File |
|--------|------|
| CREATE | `models/sam_refiner.py` |
| MODIFY | `models/musc.py` |
| MODIFY | `scripts/generate_submission.py` |
| MODIFY | `scripts/compute_segf1_dinov3.py` |
| MODIFY | `configs/musc.yaml` |

---

## 2. New File: `models/sam_refiner.py`

This module owns all SAM interaction. Zero imports from the rest of MuSc.

### 2.1 Class signature

```python
class SAMRefiner:
    def __init__(
        self,
        checkpoint_path: str,          # path to sam_vit_h.pth
        model_type: str = "vit_h",
        device: str = "cuda",
        k_pos: int = 5,                # positive prompt points
        k_neg: int = 5,                # negative prompt points
        min_spacing_px: int = 30,      # minimum distance between sampled points
        dilation_kernel: int = 25,     # ellipse kernel size for negative ring
        region_percentile: float = 0.9, # top fraction → anomaly region R (Otsu fallback)
    )
```

### 2.2 Internal helpers

#### `_sample_points_with_spacing(score_map, k, min_spacing, mask=None)`
- `score_map`: (H, W) float — higher = more preferred
- Iterate pixels in descending score order
- Accept pixel if distance to all already-accepted pixels ≥ `min_spacing`
- Stop when k accepted or candidates exhausted
- Return `np.ndarray` shape (N, 2) in **xy** (col, row) format for SAM

```
flat_idx = np.argsort(score_map.ravel())[::-1]
for idx in flat_idx:
    y, x = divmod(int(idx), W)
    if mask is not None and not mask[y, x]:
        continue
    if all distances to selected ≥ min_spacing:
        selected.append([x, y])   # SAM uses (x, y) = (col, row)
    if len(selected) == k:
        break
return np.array(selected)
```

#### `_get_anomaly_region(heatmap)`
1. Normalise heatmap to uint8 [0, 255].
2. Try Otsu: `thresh, R = cv2.threshold(hmap_u8, 0, 1, THRESH_BINARY + THRESH_OTSU)`
3. Sanity check: if R covers < 1% or > 60% of pixels, fall back to top-10 percentile.
4. Return R as uint8 binary (0/1).

```python
# Otsu
hmap_u8 = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
_, R_otsu = cv2.threshold(hmap_u8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
coverage = R_otsu.mean()
if 0.01 <= coverage <= 0.60:
    return R_otsu
# Fallback: top 10%
thresh = np.percentile(heatmap, 90)
return (heatmap >= thresh).astype(np.uint8)
```

#### `_get_bbox_from_mask(mask, heatmap)`
- `mask`: (H, W) bool — M2 from SAM pass 2
- Find connected components via `skimage.measure.label`
- For each component compute mean heatmap score over its pixels
- Select component with highest mean score
- Return its bbox as `np.array([x1, y1, x2, y2])` (SAM format)
- Return `None` if mask is all-zero

```python
from skimage import measure
labeled = measure.label(mask.astype(np.uint8))
best_bbox = None
best_score = -1.0
for region in measure.regionprops(labeled):
    coords = region.coords          # (N, 2) in yx
    avg = heatmap[coords[:, 0], coords[:, 1]].mean()
    if avg > best_score:
        best_score = avg
        minr, minc, maxr, maxc = region.bbox
        best_bbox = np.array([minc, minr, maxc, maxr], dtype=float)
return best_bbox
```

### 2.3 Core method: `refine(image_rgb, heatmap) → np.ndarray`

```
image_rgb : (H, W, 3) uint8 numpy — resized to image_size
heatmap   : (H, W)   float numpy — MuSc anomaly map
returns   : (H, W)   uint8 binary numpy {0, 1}
```

**Step 1 — Set SAM image**
```python
self.predictor.set_image(image_rgb)
```

**Step 2 — Generate prompts**
```python
R = self._get_anomaly_region(heatmap)

# Positive points: top-k from heatmap with min spacing
pos_coords = self._sample_points_with_spacing(heatmap, self.k_pos, self.min_spacing_px)

# Negative ring
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilation_kernel, self.dilation_kernel))
dilated = cv2.dilate(R, kernel)
ring = (dilated - R).astype(bool)

# Negative points: lowest heatmap score from ring
neg_heatmap = np.where(ring, -heatmap, -np.inf)   # invert: lowest → highest
neg_coords = self._sample_points_with_spacing(neg_heatmap, self.k_neg, self.min_spacing_px, mask=ring)
```

Build SAM point arrays:
```python
if len(neg_coords) == 0:
    point_coords = pos_coords                      # (Np, 2)
    point_labels = np.ones(len(pos_coords), dtype=int)
else:
    point_coords = np.concatenate([pos_coords, neg_coords], axis=0)
    point_labels = np.array([1]*len(pos_coords) + [0]*len(neg_coords), dtype=int)
```

**Step 3 — SAM Pass 1: points only**
```python
masks1, scores1, logits1 = self.predictor.predict(
    point_coords=point_coords,
    point_labels=point_labels,
    multimask_output=False,
)
# masks1: (1, H, W) bool
# logits1: (1, 256, 256) float — raw SAM logit map
M1 = masks1[0]
logit1 = logits1[0:1]   # (1, 256, 256)
```

**Step 4 — SAM Pass 2: points + logit1**
```python
masks2, scores2, logits2 = self.predictor.predict(
    point_coords=point_coords,
    point_labels=point_labels,
    mask_input=logit1,
    multimask_output=False,
)
M2 = masks2[0]
logit2 = logits2[0:1]   # (1, 256, 256)
```

**Step 5 — Extract bbox from M2**
```python
box = self._get_bbox_from_mask(M2, heatmap)
```

**Step 6 — SAM Pass 3: points + bbox + logit2**
```python
if box is not None:
    masks3, _, _ = self.predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        mask_input=logit2,
        multimask_output=False,
    )
    M3 = masks3[0]
else:
    M3 = M2   # fallback if M2 is all-zero
return M3.astype(np.uint8)
```

### 2.4 Edge-case handling

| Situation | Handling |
|-----------|----------|
| No positive points sampled (uniform heatmap) | Return zero mask, skip all SAM passes |
| M2 all-zero (SAM produced no mask in pass 2) | Skip pass 3, return M2 (zeros) |
| box is None (no components in M2) | Use M2 as M3 |
| SAM CUDA OOM | Catch RuntimeError, log warning, return zero mask |

---

## 3. Modify: `models/musc.py`

### 3.1 Config keys added

In `musc.yaml` (section `sam`):
```yaml
sam:
  use_sam: False
  checkpoint: 'models/sam_vit_h.pth'
  model_type: 'vit_h'
  k_pos: 5
  k_neg: 5
  min_spacing_px: 30
  dilation_kernel: 25
```

### 3.2 `__init__` changes

After `self.load_backbone()`:
```python
sam_cfg = cfg.get('sam', {})
if sam_cfg.get('use_sam', False):
    from models.sam_refiner import SAMRefiner
    self.sam_refiner = SAMRefiner(
        checkpoint_path=sam_cfg.get('checkpoint', 'models/sam_vit_h.pth'),
        model_type=sam_cfg.get('model_type', 'vit_h'),
        device=str(self.device),
        k_pos=sam_cfg.get('k_pos', 5),
        k_neg=sam_cfg.get('k_neg', 5),
        min_spacing_px=sam_cfg.get('min_spacing_px', 30),
        dilation_kernel=sam_cfg.get('dilation_kernel', 25),
    )
else:
    self.sam_refiner = None
```

### 3.3 `make_category_data` changes

After the line `pr_px = np.array(anomaly_maps)` (currently line 264):

```python
pr_px = np.array(anomaly_maps)   # (N, 1, H, W) float

if self.sam_refiner is not None:
    from PIL import Image as PILImage
    refined = []
    print('SAM refinement...')
    for i, path in enumerate(image_path_list):
        img_pil = PILImage.open(path).convert("RGB").resize(
            (self.image_size, self.image_size), PILImage.BILINEAR
        )
        img_np = np.array(img_pil)                          # (H, W, 3) uint8
        hmap = pr_px[i].squeeze()                           # (H, W) float
        m3 = self.sam_refiner.refine(img_np, hmap)          # (H, W) uint8 {0,1}
        refined.append(m3.astype(np.float32)[np.newaxis])   # (1, H, W)
    pr_px = np.stack(refined, axis=0)                       # (N, 1, H, W)
```

Return signature unchanged: `return image_metric, pixel_metric, pr_px, gt_px`

The SegF1 global threshold in `main()` is applied to `pr_px` as-is. When `pr_px`
values are binary {0.0, 1.0}, `find_best_threshold` returns ≈ 0.5 and
`compute_segf1_at_threshold` computes F1 of M3 directly. Mechanism unchanged.

**Note:** `image_metric` and `pixel_metric` (AUROC, AUPRO, f1_max, etc.) are
still computed on the raw heatmap before SAM refinement so those metrics are
comparable to baseline. Only `segf1` in the final table uses SAM-refined `pr_px`.

Actually, cleaner separation: compute all metrics on raw heatmap, compute segf1
on SAM-refined mask. Modify return to carry both:

```python
# Return raw heatmap for non-segf1 metrics, SAM mask for segf1
return image_metric, pixel_metric, pr_px_raw, gt_px, pr_px_sam
```

And in `main()`:
```python
image_metric, pixel_metric, pr_px_cat, gt_px_cat, pr_px_sam_cat = \
    self.make_category_data(category=category)
...
# pr_px_all used for global threshold and SegF1
pr_px_all.append(pr_px_sam_cat if pr_px_sam_cat is not None else pr_px_cat)
```

When `use_sam=False`: `pr_px_sam_cat = None` → fallback to raw heatmap.

---

## 4. Modify: `scripts/generate_submission.py`

### 4.1 New CLI args

```python
parser.add_argument("--use_sam",         action="store_true", default=False)
parser.add_argument("--sam_checkpoint",  default="models/sam_vit_h.pth")
parser.add_argument("--sam_model_type",  default="vit_h")
parser.add_argument("--sam_k_pos",       type=int, default=5)
parser.add_argument("--sam_k_neg",       type=int, default=5)
parser.add_argument("--sam_spacing",     type=int, default=30)
parser.add_argument("--sam_dilation",    type=int, default=25)
```

### 4.2 SAM loading

After backbone load:
```python
sam_refiner = None
if args.use_sam:
    from models.sam_refiner import SAMRefiner
    sam_refiner = SAMRefiner(
        checkpoint_path=args.sam_checkpoint,
        model_type=args.sam_model_type,
        device=str(device),
        k_pos=args.sam_k_pos,
        k_neg=args.sam_k_neg,
        min_spacing_px=args.sam_spacing,
        dilation_kernel=args.sam_dilation,
    )
    print("SAM refiner loaded.")
```

### 4.3 New helper function `sam_refine_maps`

```python
def sam_refine_maps(anomaly_maps, image_paths, sam_refiner, image_size):
    """Refine float anomaly maps into binary SAM masks.
    
    anomaly_maps : (N, 1, H, W) float32 numpy
    image_paths  : list of str, length N
    returns      : (N, 1, H, W) float32 numpy with values in {0.0, 1.0}
    """
    from PIL import Image as PILImage
    refined = []
    for amap, path in tqdm(zip(anomaly_maps, image_paths), total=len(image_paths),
                           desc="SAM refine"):
        img_np = np.array(
            PILImage.open(path).convert("RGB").resize(
                (image_size, image_size), PILImage.BILINEAR
            )
        )
        hmap = amap.squeeze()
        m3 = sam_refiner.refine(img_np, hmap)
        refined.append(m3.astype(np.float32)[np.newaxis])
    return np.stack(refined, axis=0)
```

### 4.4 Phase 1 (threshold computation) changes

Inside `compute_threshold_from_public`, after `run_inference` returns `anomaly_maps`:
```python
if sam_refiner is not None:
    anomaly_maps = sam_refine_maps(anomaly_maps, image_paths, sam_refiner, args.img_resize)
pr_px = anomaly_maps
```

Pass `sam_refiner` and `args.img_resize` as new arguments to this function.

Note: `image_paths` must be returned by `run_inference`. Currently it IS returned
as the second element but not used in phase 1. Use it here.

### 4.5 Phase 2 (private split submission) changes

In the Phase 2 loop, after `run_inference` for private splits:
```python
anomaly_maps, image_paths = run_inference(...)

if sam_refiner is not None:
    anomaly_maps = sam_refine_maps(anomaly_maps, image_paths, sam_refiner, args.img_resize)
```

Then `save_maps` is called with the potentially-refined `anomaly_maps`. Since
`save_maps` saves:
- `.tiff` as `float16` → for binary {0,1} maps this stores 0.0 and 1.0 exactly
- `.png` as `uint8 * 255` thresholded → for binary input, threshold 0.5 gives correct result

No changes needed to `save_maps` itself.

---

## 5. Modify: `scripts/compute_segf1_dinov3.py`

### 5.1 New CLI args (same as generate_submission.py)

```python
parser.add_argument("--use_sam",        action="store_true", default=False)
parser.add_argument("--sam_checkpoint", default="models/sam_vit_h.pth")
parser.add_argument("--sam_model_type", default="vit_h")
parser.add_argument("--sam_k_pos",      type=int, default=5)
parser.add_argument("--sam_k_neg",      type=int, default=5)
parser.add_argument("--sam_spacing",    type=int, default=30)
parser.add_argument("--sam_dilation",   type=int, default=25)
```

### 5.2 SAM loading (same pattern as generate_submission.py)

### 5.3 Per-category processing changes

After `run_inference` returns `anomaly_maps, _, gt_masks`:
```python
if sam_refiner is not None:
    # Need image paths - modify run_inference call to return paths
    anomaly_maps, image_paths, gt_masks = run_inference(..., with_masks=True)
    anomaly_maps = sam_refine_maps(anomaly_maps, image_paths, sam_refiner, args.img_resize)
```

Rest of per-category and global-threshold logic unchanged.

---

## 6. Modify: `configs/musc.yaml`

Add SAM section:
```yaml
sam:
  use_sam: False                        # set True to enable
  checkpoint: 'models/sam_vit_h.pth'
  model_type: 'vit_h'
  k_pos: 5
  k_neg: 5
  min_spacing_px: 30
  dilation_kernel: 25
```

---

## 7. Evaluation Protocol

### SegF1 with SAM masks

1. Run full MuSc inference on all categories → float heatmaps
2. Run SAM cascade per image → binary M3 masks
3. Collect all M3 masks dataset-wide as `pr_px_all` (values 0.0 or 1.0)
4. `find_best_threshold(combined_gt, combined_pr_sam)` → global_thr (≈ 0.5)
5. `compute_segf1_at_threshold(gt_px, pr_px_sam, global_thr)` per category
6. Report mean SegF1

The mechanism is unchanged from baseline — same global threshold sweep, same
`compute_segf1_at_threshold` function. Only `pr_px` content changes (SAM binary
instead of raw heatmap float).

### Comparable baseline

Run `compute_segf1_dinov3.py` without `--use_sam` first to establish baseline
SegF1. Then run with `--use_sam --sam_checkpoint models/sam_vit_h.pth` to compare.

---

## 8. Key Hyperparameters and Rationale

| Parameter | Value | Source |
|-----------|-------|--------|
| SAM model | ViT-H | Best SAM quality, checkpoint already present |
| `k_pos` | 5 | Gives spatial coverage without over-specifying |
| `k_neg` | 5 | Matches positive count |
| `min_spacing_px` | 30 | ~6% of 512px image, balances coverage vs clustering. Paper uses 400px but at much higher resolution |
| `dilation_kernel` | 25×25 ellipse | Directly from paper Table 2 ablation (best result) |
| Anomaly region R | Otsu (fallback: top-10%) | Otsu adapts per-image; fallback prevents degenerate Otsu on uniform maps |
| `region_percentile` | 0.9 (top 10%) | Conservative — limits SAM prompt region to clearly high-scoring pixels |

---

## 9. Implementation Order

Execute strictly in this order (each step buildable and testable independently):

### Step 1 — Create `models/sam_refiner.py`
Implement `SAMRefiner` with all helpers. Test standalone:
```bash
conda run -n clip python3 -c "
from models.sam_refiner import SAMRefiner
import numpy as np
r = SAMRefiner('models/sam_vit_h.pth')
img = np.random.randint(0, 255, (512,512,3), dtype=np.uint8)
hmap = np.random.rand(512,512).astype(np.float32)
m3 = r.refine(img, hmap)
print('M3 shape:', m3.shape, 'unique:', np.unique(m3))
"
```

### Step 2 — Add SAM config to `configs/musc.yaml`
Add `sam:` section with `use_sam: False` as default (no behaviour change).

### Step 3 — Modify `models/musc.py`
- Load SAMRefiner in `__init__` based on config
- Run SAM in `make_category_data` when enabled
- Modify return signature to carry SAM masks separately
- Update `main()` to use SAM masks for SegF1

### Step 4 — Modify `scripts/compute_segf1_dinov3.py`
Add SAM args and refinement step.

### Step 5 — Modify `scripts/generate_submission.py`
Add SAM args, `sam_refine_maps` helper, and refinement in both phases.

### Step 6 — Baseline vs SAM evaluation
```bash
# Baseline
conda run -n clip python3 scripts/compute_segf1_dinov3.py \
  --backbone_name facebook/dinov3-vitl16-pretrain-lvd1689m \
  --data_path ./data/mvtec_ad_2/ --img_resize 512

# With SAM
conda run -n clip python3 scripts/compute_segf1_dinov3.py \
  --backbone_name facebook/dinov3-vitl16-pretrain-lvd1689m \
  --data_path ./data/mvtec_ad_2/ --img_resize 512 \
  --use_sam --sam_checkpoint models/sam_vit_h.pth
```

---

## 10. SAM API Reference (segment_anything)

```python
from segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
sam.to(device)
predictor = SamPredictor(sam)

# Set image (must be uint8 RGB numpy)
predictor.set_image(image_rgb)

# Predict (point-only)
masks, scores, logits = predictor.predict(
    point_coords=np.array([[x1,y1],[x2,y2]]),  # (N, 2) xy
    point_labels=np.array([1, 0]),              # 1=fg, 0=bg
    multimask_output=False,
)
# masks: (1, H, W) bool, logits: (1, 256, 256) float32

# Predict (with mask_input and box)
masks, scores, logits = predictor.predict(
    point_coords=point_coords,
    point_labels=point_labels,
    box=np.array([x1, y1, x2, y2]),   # (4,) float, or None
    mask_input=prev_logit,             # (1, 256, 256) float, or None
    multimask_output=False,
)
```

Key SAM constraint: `mask_input` must be shape `(1, 256, 256)` — exactly one
channel at SAM's internal 256×256 resolution. `logits[0:1]` from the previous
call is already the right shape.

---

## 11. Runtime Estimate

SAM ViT-H image encoding ≈ 1–2 s/image on a single GPU.
MVTecAD2 test_public has ~320 images total across 8 categories.
Expected SAM overhead: ~10–20 minutes additional per full run.

Each SAM pass (predict) is <10ms after encoding — the 3-pass cascade adds
negligible time beyond the one-time image encoding.

---

## 12. Potential Issues and Mitigations

| Issue | Mitigation |
|-------|-----------|
| SAM encodes at native resolution but MuSc heatmap is at 512×512 | Resize image to 512×512 before passing to SAM, matching heatmap resolution |
| Otsu on low-contrast normal image gives huge R (>60% coverage) | Fallback to top-10 percentile threshold for R |
| Uniform heatmap → no valid positive points (empty result from spacing constraint) | If `len(pos_coords) == 0`, return zero mask without calling SAM |
| SAM produces full-image mask for normal images (spurious FP) | Acceptable: SAM constrained by negative points from MuSc ring; for truly normal images negative ring points are strong |
| Private split images don't have GT (can't compute SegF1) | As now — private split only produces submission files, no metric |
| `image_paths` not returned by `run_inference` in all call sites | Already returned as second element; verify all callers use it |
