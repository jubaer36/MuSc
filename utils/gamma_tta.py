"""Gamma Test-Time Augmentation helpers for MuSc anomaly detection.

Usage:
    from utils.gamma_tta import GammaPreprocess, fuse_gamma_heatmaps

    preprocess_bright = GammaPreprocess(preprocess, gamma=0.8)
    preprocess_dark   = GammaPreprocess(preprocess, gamma=1.3)

    maps_o = run_inference(dataset_with_preprocess, ...)
    maps_b = run_inference(dataset_with_preprocess_bright, ...)
    maps_d = run_inference(dataset_with_preprocess_dark, ...)

    anomaly_maps = fuse_gamma_heatmaps([maps_o, maps_b, maps_d])
"""

import numpy as np
from PIL import Image


class GammaPreprocess:
    """PIL-level gamma correction applied before the base preprocess transform.

    Gamma < 1.0 brightens the image (recovers underexposed defects).
    Gamma > 1.0 darkens the image (recovers overexposed/washed-out defects).

    Applied as: pixel_out = pixel_in^gamma on [0,1]-normalized values.
    The result is re-quantized to uint8 before passing to base_preprocess,
    so DINO's normalization statistics remain in their expected range.
    """

    def __init__(self, base_preprocess, gamma: float):
        self.base = base_preprocess
        self.gamma = gamma

    def __call__(self, img):
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.clip(arr ** self.gamma, 0.0, 1.0)
        img_gamma = Image.fromarray((arr * 255.0).round().astype(np.uint8))
        return self.base(img_gamma)


def robust_normalize(heatmap: np.ndarray) -> np.ndarray:
    """Median/MAD normalization then min-max to [0, 1].

    More robust than plain min-max when one branch has outlier scores
    from gamma-shifted texture statistics.

    Args:
        heatmap: numpy array of any shape (operates element-wise)

    Returns:
        float32 array of same shape, values in [0, 1]
    """
    med = np.median(heatmap)
    mad = np.median(np.abs(heatmap - med)) + 1e-8
    h = (heatmap - med) / mad
    h_min, h_max = h.min(), h.max()
    return ((h - h_min) / (h_max - h_min + 1e-8)).astype(np.float32)


def fuse_gamma_heatmaps(
    maps_list: list,
    weights: tuple = (0.6, 0.2, 0.2),
) -> np.ndarray:
    """Robust-normalize each heatmap then compute weighted sum.

    The original (gamma=1.0) map gets the highest weight (0.6) so the
    baseline MuSc signal dominates; auxiliary gamma variants contribute
    only as weak support.

    Args:
        maps_list: list of (N, 1, H, W) float32 numpy arrays
                   [original, brightened, darkened]
        weights:   per-variant weights, must sum to 1.0

    Returns:
        (N, 1, H, W) float32 numpy array — fused anomaly map
    """
    assert len(maps_list) == len(weights), "maps_list and weights must have same length"
    normed = [robust_normalize(m) for m in maps_list]
    fused = sum(w * m for w, m in zip(weights, normed))
    return fused.astype(np.float32)
