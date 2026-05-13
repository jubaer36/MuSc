# Plan: SNAMD Similarity-Weighted Pooling in LNAMD

## Background

MuSc-V2 replaces uniform neighborhood averaging in LNAMD with exponential-similarity weighting (SNAMD). For a patch at position p with center feature F_c, neighbors F_i get weight:

```
Λ_i = exp(-||F_i - F_c||₂)
```

Normalized weights → weighted sum replaces uniform mean. Benefit: patches at defect boundaries have mixed normal/abnormal neighbors; similarity weighting down-weights dissimilar (normal) neighbors, so the aggregated feature stays closer to the defect's true texture → sharper anomaly signal.

## Current Code Analysis

`models/modules/_LNAMD.py`:

- `MeanMapper.forward`: input `(N, C, r, r)` where N = B×H×W
  - Reshapes to `(N, 1, C*r*r)` → `adaptive_avg_pool1d` → `(N, preprocessing_dim)`
  - With C=1024, preprocessing_dim=1024, r=3: equivalent to `features.mean(dim=[-2,-1])` per channel (uniform average over the 3×3 neighborhood)
- `Preprocessing.__init__`: instantiates one `MeanMapper` per layer

## Files to Modify

**Only `models/modules/_LNAMD.py`** — one new class, one line change. No changes to musc.py, generate_submission.py, or sam_refiner.py.

## Implementation

### Add `SimilarityWeightedMapper` (after MeanMapper, before LNAMD)

```python
class SimilarityWeightedMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(SimilarityWeightedMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        # features: (N, C, r, r)
        N, C, r1, r2 = features.shape
        center = features[:, :, r1 // 2, r2 // 2]      # (N, C)
        flat = features.reshape(N, C, -1)               # (N, C, r*r)
        dist = torch.norm(flat - center.unsqueeze(-1), dim=1)  # (N, r*r)
        weights = torch.exp(-dist)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        agg = (flat * weights.unsqueeze(1)).sum(dim=-1)  # (N, C)
        if C == self.preprocessing_dim:
            return agg
        agg = agg.unsqueeze(1)
        return F.adaptive_avg_pool1d(agg, self.preprocessing_dim).squeeze(1)
```

**Edge cases handled:**
- `r=1`: dist=0 → weights all 1 → weighted sum = center (correct, same as MeanMapper)
- `C=preprocessing_dim=1024`: no channel pooling needed (identity path)
- `C≠preprocessing_dim`: falls back to adaptive_avg_pool1d (future-proof)

### Replace MeanMapper in Preprocessing.__init__ (line 44)

```python
# Before:
module = MeanMapper(output_dim)
# After:
module = SimilarityWeightedMapper(output_dim)
```

## Verification

```bash
cd "/mnt/Work/ML/Code/Anomaly Detection/MuSc"
conda run -n ml python3 scripts/compute_segf1_dinov3.py \
    --backbone_name facebook/dinov3-vitl16-pretrain-lvd1689m \
    --data_path ./data/mvtec_ad_2/ \
    --img_resize 512 \
    --sam_version sam3 \
    --sam3_model_id models/sam3 \
    --sam_k_pos 2 --sam_k_neg 5 \
    --sam_spacing 60 --sam_dilation 15
```

## Baseline (before SNAMD)

| Category    | Global SegF1 | Per-cls SegF1 |
|-------------|-------------|---------------|
| can         | 0.00%       | 0.00%         |
| fabric      | 46.19%      | 57.19%        |
| fruit_jelly | 40.07%      | 42.11%        |
| rice        | 28.92%      | 40.62%        |
| sheet_metal | 13.14%      | 26.15%        |
| vial        | 26.18%      | 33.89%        |
| wallplugs   | 9.81%       | 31.80%        |
| walnuts     | 53.68%      | 60.74%        |
| **mean**    | **27.25%**  | **36.56%**    |

Expected: improvement in texture-rich categories (fabric, walnuts) and boundary-heavy defects. `can` unlikely to change without lighting fix.
