import numpy as np
from PIL import Image
from torchvision import transforms

from utils.prompt_utils import heatmap_to_prompts, predict_region, merge_region_masks

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_UNNORM = transforms.Compose([
    transforms.Normalize(
        mean=[-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1.0 / s for s in IMAGENET_STD],
    )
])


class SAMRefiner:
    """Refines MuSc anomaly heatmaps using SAM mask predictions.

    SAM runs on the same resized images MuSc uses (image_size x image_size).
    Prompts are derived from the heatmap: peak pixel per region (positive),
    bounding box per region, and low-score background samples (negative).

    The final score is: alpha * SAM_mask + (1 - alpha) * norm_heatmap
    so it stays in [0, 1] and is differentiable with the original MuSc ranking.
    """

    def __init__(
        self,
        checkpoint: str,
        model_type: str = 'vit_h',
        device: str = 'cuda',
        threshold_percentile: float = 95.0,
        blend_alpha: float = 0.3,
        n_neg: int = 5,
    ):
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device)
        self.predictor = SamPredictor(sam)
        self.threshold_percentile = threshold_percentile
        self.blend_alpha = blend_alpha
        self.n_neg = n_neg

    def _tensor_to_rgb(self, img_tensor):
        """Convert ImageNet-normalized (3, H, W) float tensor to (H, W, 3) uint8."""
        img = _UNNORM(img_tensor.cpu().float()).clamp(0, 1)
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return img

    def refine(self, image_rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        """Refine single heatmap.

        Args:
            image_rgb: (H, W, 3) uint8 numpy array
            heatmap:   (H, W) float32 numpy array

        Returns:
            (H, W) float32 blended score in [0, 1]
        """
        H, W = heatmap.shape
        h_min, h_max = heatmap.min(), heatmap.max()
        norm_h = (heatmap - h_min) / (h_max - h_min + 1e-8)

        prompts = heatmap_to_prompts(heatmap, self.threshold_percentile, self.n_neg)
        if not prompts:
            return norm_h

        self.predictor.set_image(image_rgb)
        region_masks = [predict_region(self.predictor, p, H, W) for p in prompts]
        sam_mask = merge_region_masks(region_masks)
        if sam_mask is None:
            return norm_h

        return self.blend_alpha * sam_mask + (1.0 - self.blend_alpha) * norm_h

    def refine_batch(
        self,
        image_path_list: list,
        anomaly_maps: np.ndarray,
        image_size: int,
    ) -> np.ndarray:
        """Refine a full batch of heatmaps.

        Args:
            image_path_list: list of str image file paths (len B)
            anomaly_maps:    (B, 1, H, W) or (B, H, W) float32 numpy
            image_size:      target H=W for loading images

        Returns:
            (B, 1, H, W) float32 numpy refined maps
        """
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        B = anomaly_maps.shape[0]
        maps_4d = anomaly_maps.reshape(B, image_size, image_size)
        refined = np.empty_like(maps_4d)

        for i, path in enumerate(image_path_list):
            img_pil = Image.open(path).convert('RGB')
            img_t = transform(img_pil)
            img_rgb = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            refined[i] = self.refine(img_rgb, maps_4d[i])

        return refined.reshape(B, 1, image_size, image_size)
