import numpy as np
import cv2
from skimage import measure


class SAMRefiner:
    """3-pass cascaded SAM prompt refinement for MuSc anomaly heatmaps.

    Pass 1: positive + negative points only → M1, logit1
    Pass 2: same points + logit1            → M2, logit2
    Pass 3: same points + bbox(M2) + logit2 → M3 (final mask)
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "vit_h",
        device: str = "cuda",
        k_pos: int = 5,
        k_neg: int = 5,
        min_spacing_px: int = 30,
        dilation_kernel: int = 25,
    ):
        from segment_anything import sam_model_registry, SamPredictor

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device)
        sam.eval()
        self.predictor = SamPredictor(sam)

        self.k_pos = k_pos
        self.k_neg = k_neg
        self.min_spacing_px = min_spacing_px
        self.dilation_kernel = dilation_kernel

    def _sample_points_with_spacing(self, score_map, k, min_spacing, mask=None):
        """Sample k points in descending score order with minimum spacing.

        Args:
            score_map: (H, W) float — higher score = more preferred
            k:         max number of points to return
            min_spacing: minimum Euclidean pixel distance between any two selected points
            mask:      optional (H, W) bool — only consider True pixels

        Returns:
            np.ndarray of shape (N, 2), xy (col, row) format for SAM. N <= k.
        """
        H, W = score_map.shape
        flat_idx = np.argsort(score_map.ravel())[::-1]
        selected_yx = []

        for raw_idx in flat_idx:
            y, x = divmod(int(raw_idx), W)
            if mask is not None and not mask[y, x]:
                continue
            if not np.isfinite(score_map[y, x]):
                break  # scores below -inf threshold
            ok = True
            for sy, sx in selected_yx:
                if (y - sy) ** 2 + (x - sx) ** 2 < min_spacing ** 2:
                    ok = False
                    break
            if ok:
                selected_yx.append((y, x))
                if len(selected_yx) == k:
                    break

        if len(selected_yx) == 0:
            return np.empty((0, 2), dtype=np.float32)

        # SAM uses (x, y) = (col, row)
        return np.array([[x, y] for y, x in selected_yx], dtype=np.float32)

    def _get_anomaly_region(self, heatmap):
        """Threshold heatmap to get binary anomaly region R.

        Uses Otsu when coverage is 1-60 %; falls back to top-10 % percentile.

        Returns:
            (H, W) uint8 binary array with values 0/1.
        """
        h_min, h_max = float(heatmap.min()), float(heatmap.max())
        if h_max - h_min < 1e-8:
            # Flat heatmap — no anomaly signal
            return np.zeros(heatmap.shape, dtype=np.uint8)

        hmap_u8 = ((heatmap - h_min) / (h_max - h_min) * 255.0).astype(np.uint8)

        _, R_otsu = cv2.threshold(hmap_u8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        coverage = float(R_otsu.mean())

        if 0.01 <= coverage <= 0.60:
            return R_otsu.astype(np.uint8)

        # Fallback: top 10 %
        thresh = np.percentile(heatmap, 90)
        return (heatmap >= thresh).astype(np.uint8)

    def _get_bbox_from_mask(self, mask, heatmap):
        """Extract bounding box of the connected component with highest mean heatmap score.

        Args:
            mask:    (H, W) bool numpy
            heatmap: (H, W) float numpy

        Returns:
            np.array([x1, y1, x2, y2]) float32, or None if mask is all-zero.
        """
        if not mask.any():
            return None

        labeled = measure.label(mask.astype(np.uint8))
        best_bbox = None
        best_score = -1.0

        for region in measure.regionprops(labeled):
            coords = region.coords  # (N, 2) in (row, col) = yx
            avg_score = float(heatmap[coords[:, 0], coords[:, 1]].mean())
            if avg_score > best_score:
                best_score = avg_score
                minr, minc, maxr, maxc = region.bbox
                # SAM box format: (x1, y1, x2, y2) = (col_min, row_min, col_max, row_max)
                best_bbox = np.array([minc, minr, maxc, maxr], dtype=np.float32)

        return best_bbox

    def refine(self, image_rgb, heatmap):
        """Run 3-pass SAM cascade to produce a refined binary anomaly mask.

        Args:
            image_rgb: (H, W, 3) uint8 numpy array (resized to model image_size)
            heatmap:   (H, W) float32 numpy array (MuSc anomaly map)

        Returns:
            (H, W) uint8 binary numpy array with values {0, 1}.
        """
        H, W = heatmap.shape
        zero_mask = np.zeros((H, W), dtype=np.uint8)

        # ── Anomaly region R ──────────────────────────────────────────────
        R = self._get_anomaly_region(heatmap)
        if R.sum() == 0:
            return zero_mask

        # ── Positive points: top-k from full heatmap ─────────────────────
        pos_coords = self._sample_points_with_spacing(
            heatmap, self.k_pos, self.min_spacing_px
        )
        if len(pos_coords) == 0:
            return zero_mask

        # ── Negative ring: dilate(R) − R ─────────────────────────────────
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.dilation_kernel, self.dilation_kernel)
        )
        dilated = cv2.dilate(R, kernel)
        ring = (dilated.astype(np.int32) - R.astype(np.int32)) > 0  # bool

        # Negative points: lowest heatmap score in the ring
        # Invert heatmap so lowest scores rank first in _sample_points_with_spacing
        neg_score_map = np.where(ring, -heatmap, -np.inf)
        neg_coords = self._sample_points_with_spacing(
            neg_score_map, self.k_neg, self.min_spacing_px, mask=ring
        )

        # ── Build combined point arrays ───────────────────────────────────
        if len(neg_coords) > 0:
            point_coords = np.concatenate([pos_coords, neg_coords], axis=0)
            point_labels = np.array(
                [1] * len(pos_coords) + [0] * len(neg_coords), dtype=int
            )
        else:
            point_coords = pos_coords
            point_labels = np.ones(len(pos_coords), dtype=int)

        # ── 3-pass SAM cascade ────────────────────────────────────────────
        try:
            self.predictor.set_image(image_rgb)

            # Pass 1: points only → M1, logit1
            masks1, _, logits1 = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )
            M1 = masks1[0]          # (H, W) bool
            logit1 = logits1[0:1]   # (1, 256, 256) float32

            # Pass 2: points + logit1 → M2, logit2
            masks2, _, logits2 = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=logit1,
                multimask_output=False,
            )
            M2 = masks2[0]          # (H, W) bool
            logit2 = logits2[0:1]   # (1, 256, 256) float32

            # Bounding box from highest-scoring M2 component
            box = self._get_bbox_from_mask(M2, heatmap)

            # Pass 3: points + bbox(M2) + logit2 → M3
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
                M3 = M2

            return M3.astype(np.uint8)

        except RuntimeError as exc:
            print(f"[SAMRefiner] Warning: {exc}")
            return zero_mask
