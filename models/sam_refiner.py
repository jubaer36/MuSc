import numpy as np
import cv2
from skimage import measure


class _SAMRefinerBase:
    """Shared prompt-generation helpers for all SAM backend variants."""

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
                break
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

        return np.array([[x, y] for y, x in selected_yx], dtype=np.float32)

    def _get_anomaly_region(self, heatmap):
        """Threshold heatmap → binary anomaly region R.

        Uses Otsu when coverage is 1-60 %; falls back to top-10 % percentile.

        Returns:
            (H, W) uint8 binary array with values 0/1.
        """
        h_min, h_max = float(heatmap.min()), float(heatmap.max())
        if h_max - h_min < 1e-8:
            return np.zeros(heatmap.shape, dtype=np.uint8)

        hmap_u8 = ((heatmap - h_min) / (h_max - h_min) * 255.0).astype(np.uint8)
        _, R_otsu = cv2.threshold(hmap_u8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        coverage = float(R_otsu.mean())

        if 0.01 <= coverage <= 0.60:
            return R_otsu.astype(np.uint8)

        thresh = np.percentile(heatmap, 90)
        return (heatmap >= thresh).astype(np.uint8)

    def _get_bbox_from_mask(self, mask, heatmap):
        """Bounding box of the connected component with highest mean heatmap score.

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
            coords = region.coords
            avg_score = float(heatmap[coords[:, 0], coords[:, 1]].mean())
            if avg_score > best_score:
                best_score = avg_score
                minr, minc, maxr, maxc = region.bbox
                best_bbox = np.array([minc, minr, maxc, maxr], dtype=np.float32)

        return best_bbox

    def _build_point_arrays(self, heatmap):
        """Build combined pos+neg point arrays from heatmap.

        Returns:
            point_coords: (N, 2) float32 numpy, xy pixel coords
            point_labels: (N,) int numpy, 1=foreground 0=background
            or (None, None) if no positive points found.
        """
        R = self._get_anomaly_region(heatmap)
        if R.sum() == 0:
            return None, None

        pos_coords = self._sample_points_with_spacing(
            heatmap, self.k_pos, self.min_spacing_px
        )
        if len(pos_coords) == 0:
            return None, None

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.dilation_kernel, self.dilation_kernel)
        )
        dilated = cv2.dilate(R, kernel)
        ring = (dilated.astype(np.int32) - R.astype(np.int32)) > 0

        neg_score_map = np.where(ring, -heatmap, -np.inf)
        neg_coords = self._sample_points_with_spacing(
            neg_score_map, self.k_neg, self.min_spacing_px, mask=ring
        )

        if len(neg_coords) > 0:
            point_coords = np.concatenate([pos_coords, neg_coords], axis=0)
            point_labels = np.array(
                [1] * len(pos_coords) + [0] * len(neg_coords), dtype=int
            )
        else:
            point_coords = pos_coords
            point_labels = np.ones(len(pos_coords), dtype=int)

        return point_coords, point_labels


# ---------------------------------------------------------------------------
# SAM1 backend (segment_anything library)
# ---------------------------------------------------------------------------

class SAMRefiner(_SAMRefinerBase):
    """3-pass cascaded SAM (v1) prompt refinement for MuSc anomaly heatmaps.

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

    def refine(self, image_rgb, heatmap):
        """Run 3-pass SAM cascade to produce a refined binary anomaly mask.

        Args:
            image_rgb: (H, W, 3) uint8 numpy array
            heatmap:   (H, W) float32 numpy array (MuSc anomaly map)

        Returns:
            (H, W) uint8 binary numpy array with values {0, 1}.
        """
        H, W = heatmap.shape
        zero_mask = np.zeros((H, W), dtype=np.uint8)

        point_coords, point_labels = self._build_point_arrays(heatmap)
        if point_coords is None:
            return zero_mask

        try:
            self.predictor.set_image(image_rgb)

            # Pass 1: points only → M1, logit1
            masks1, _, logits1 = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )
            M1 = masks1[0]
            logit1 = logits1[0:1]   # (1, 256, 256)

            # Pass 2: points + logit1 → M2, logit2
            masks2, _, logits2 = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=logit1,
                multimask_output=False,
            )
            M2 = masks2[0]
            logit2 = logits2[0:1]

            # Pass 3: points + bbox(M2) + logit2 → M3
            box = self._get_bbox_from_mask(M2, heatmap)
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


# ---------------------------------------------------------------------------
# SAM3 backend (transformers Sam3TrackerModel — drop-in SAM2 replacement)
# ---------------------------------------------------------------------------

class SAM3Refiner(_SAMRefinerBase):
    """3-pass cascaded SAM3 Tracker prompt refinement for MuSc anomaly heatmaps.

    Uses Sam3TrackerModel (HuggingFace transformers) instead of segment_anything.
    Same cascade logic; image embeddings are cached after the first forward pass.

    Pass 1: positive + negative points only → M1, logit1
    Pass 2: same points + logit1            → M2, logit2
    Pass 3: same points + bbox(M2) + logit2 → M3 (final mask)

    Args:
        model_id: HuggingFace model ID or local path, e.g. "facebook/sam3.1"
    """

    def __init__(
        self,
        model_id: str = "models/sam3",
        device: str = "cuda",
        k_pos: int = 5,
        k_neg: int = 5,
        min_spacing_px: int = 30,
        dilation_kernel: int = 25,
    ):
        import torch
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor
        from PIL import Image as _PIL
        self._PIL = _PIL
        self._torch = torch

        self.model = Sam3TrackerModel.from_pretrained(model_id)
        self.model.to(device).eval()
        self.processor = Sam3TrackerProcessor.from_pretrained(model_id)
        self.device = device

        self.k_pos = k_pos
        self.k_neg = k_neg
        self.min_spacing_px = min_spacing_px
        self.dilation_kernel = dilation_kernel

    def refine(self, image_rgb, heatmap):
        """Run 3-pass SAM3 cascade to produce a refined binary anomaly mask.

        Args:
            image_rgb: (H, W, 3) uint8 numpy array
            heatmap:   (H, W) float32 numpy array (MuSc anomaly map)

        Returns:
            (H, W) uint8 binary numpy array with values {0, 1}.
        """
        torch = self._torch
        H, W = heatmap.shape
        zero_mask = np.zeros((H, W), dtype=np.uint8)

        point_coords, point_labels = self._build_point_arrays(heatmap)
        if point_coords is None:
            return zero_mask

        # SAM3Tracker processor expects nested lists:
        #   input_points: [batch[obj[pt[x, y]]]]  →  shape (1, 1, N, 2) after processing
        #   input_labels: [batch[obj[label]]]      →  shape (1, 1, N)
        pts_nested = [[[float(x), float(y)] for x, y in point_coords]]
        lbl_nested = [[int(l) for l in point_labels]]

        pil_img = self._PIL.fromarray(image_rgb)

        try:
            # Encode image once; reuse embeddings for all 3 passes
            img_enc = self.processor(images=pil_img, return_tensors="pt").to(self.device)
            original_sizes = img_enc["original_sizes"]
            with torch.no_grad():
                img_emb = self.model.get_image_embeddings(
                    pixel_values=img_enc.pixel_values
                )

            def _run_pass(input_masks=None, box_np=None):
                # boxes: processor expects [image, box, coords] = 3 levels
                # [[x1,y1,x2,y2]] = 1 box per image; wrap in batch list → [[[x1,y1,x2,y2]]]
                box_nested = (
                    [[[float(b) for b in box_np.tolist()]]] if box_np is not None else None
                )
                enc = self.processor(
                    images=pil_img,
                    input_points=[pts_nested],
                    input_labels=[lbl_nested],
                    input_boxes=box_nested,
                    return_tensors="pt",
                ).to(self.device)
                # Drop pixel_values (use cached embeddings) and original_sizes
                # (not a model input — kept separately for post_process_masks).
                enc.pop("pixel_values", None)
                enc.pop("original_sizes", None)

                with torch.no_grad():
                    out = self.model(
                        image_embeddings=img_emb,
                        input_masks=input_masks,
                        multimask_output=False,
                        **enc,
                    )

                # pred_masks shape: (batch, num_objects, num_masks, H_low, W_low) = 5D
                # post_process_masks iterates over batch dim, returns list of upsampled tensors
                # Each element: (num_objects, num_masks, H_orig, W_orig)
                masks_list = self.processor.post_process_masks(
                    out.pred_masks, original_sizes
                )
                M = masks_list[0][0][0].cpu().numpy()  # (H, W) bool

                # Squeeze object dim before reuse as input_masks (conv2d needs 4D)
                # (1, 1, 1, H, W) → (1, 1, H, W)
                logit_out = out.pred_masks[:, 0] if out.pred_masks.dim() == 5 else out.pred_masks
                return M, logit_out

            # Pass 1: points only
            M1, logit1 = _run_pass()

            # Pass 2: points + logit feedback
            M2, logit2 = _run_pass(input_masks=logit1)

            # Pass 3: points + bbox(M2) + logit feedback
            box = self._get_bbox_from_mask(M2, heatmap)
            if box is not None:
                M3, _ = _run_pass(input_masks=logit2, box_np=box)
            else:
                M3 = M2

            return M3.astype(np.uint8)

        except Exception as exc:
            print(f"[SAM3Refiner] Warning: {exc}")
            return zero_mask


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_sam_refiner(version, **kwargs):
    """Instantiate a SAM refiner by version string.

    Args:
        version: "sam1" or "sam3"
        **kwargs: forwarded to the chosen class constructor.
            For sam1: checkpoint_path, model_type, device, k_pos, k_neg,
                      min_spacing_px, dilation_kernel
            For sam3: model_id, device, k_pos, k_neg,
                      min_spacing_px, dilation_kernel

    Returns:
        SAMRefiner or SAM3Refiner instance.
    """
    if version == "sam1":
        return SAMRefiner(**kwargs)
    if version == "sam3":
        return SAM3Refiner(**kwargs)
    raise ValueError(f"Unknown SAM version '{version}'. Choose 'sam1' or 'sam3'.")
