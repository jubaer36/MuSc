import numpy as np
from scipy import ndimage


def heatmap_to_prompts(heatmap: np.ndarray, percentile: float = 95, n_neg: int = 5):
    """Convert a (H, W) float32 anomaly heatmap to SAM prompt dicts.

    Returns list of dicts, one per detected region:
        pos_point : (x, y) int  — peak-score pixel (positive prompt)
        box       : [x0, y0, x1, y1] int  — tight bounding box
        neg_points: list of (x, y) int  — low-score negative prompts
    """
    threshold = np.percentile(heatmap, percentile)
    binary = (heatmap > threshold).astype(np.uint8)
    labeled, n_regions = ndimage.label(binary)

    # low-score background pixels for negative prompts
    low_mask = (heatmap < np.percentile(heatmap, 20))
    low_coords = np.argwhere(low_mask)  # (N, 2) in (row, col)

    prompts = []
    for region_id in range(1, n_regions + 1):
        region_mask = labeled == region_id
        region_scores = heatmap * region_mask
        peak_idx = np.unravel_index(region_scores.argmax(), heatmap.shape)
        peak_y, peak_x = int(peak_idx[0]), int(peak_idx[1])

        coords = np.argwhere(region_mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)

        if len(low_coords) >= n_neg:
            idx = np.random.choice(len(low_coords), n_neg, replace=False)
            neg_pts = [(int(low_coords[i, 1]), int(low_coords[i, 0])) for i in idx]
        else:
            neg_pts = [(int(c[1]), int(c[0])) for c in low_coords]

        prompts.append({
            'pos_point': (peak_x, peak_y),
            'box': [int(x0), int(y0), int(x1), int(y1)],
            'neg_points': neg_pts,
        })

    return prompts


def predict_region(predictor, prompt: dict, H: int, W: int) -> np.ndarray:
    """Run SAM in point-only, box-only, and combined modes; return merged mask.

    Returns (H, W) float32 binary mask.
    """
    pos = prompt['pos_point']
    neg_pts = prompt['neg_points']
    box = np.array(prompt['box'], dtype=np.float32)  # [x0, y0, x1, y1]

    all_pts = np.array([[pos[0], pos[1]]] + [[p[0], p[1]] for p in neg_pts], dtype=np.float32)
    all_labels = np.array([1] + [0] * len(neg_pts), dtype=np.int32)
    pos_pts = np.array([[pos[0], pos[1]]], dtype=np.float32)
    pos_labels = np.array([1], dtype=np.int32)

    best_masks = []
    best_scores = []

    # point-only
    masks, scores, _ = predictor.predict(
        point_coords=pos_pts, point_labels=pos_labels,
        multimask_output=True,
    )
    best_masks.append(masks[scores.argmax()])
    best_scores.append(scores.max())

    # box-only
    masks, scores, _ = predictor.predict(
        box=box[None], multimask_output=True,
    )
    best_masks.append(masks[scores.argmax()])
    best_scores.append(scores.max())

    # combined (points + box)
    masks, scores, _ = predictor.predict(
        point_coords=all_pts, point_labels=all_labels,
        box=box[None], multimask_output=True,
    )
    best_masks.append(masks[scores.argmax()])
    best_scores.append(scores.max())

    # weighted union by SAM confidence
    best_scores = np.array(best_scores, dtype=np.float32)
    weights = best_scores / (best_scores.sum() + 1e-8)
    merged = sum(w * m.astype(np.float32) for w, m in zip(weights, best_masks))
    return (merged > 0.5).astype(np.float32)


def merge_region_masks(region_masks: list) -> np.ndarray:
    """Union of per-region masks. Returns (H, W) float32."""
    if not region_masks:
        return None
    merged = region_masks[0].copy()
    for m in region_masks[1:]:
        merged = np.maximum(merged, m)
    return merged
