"""
DINOv3 HuggingFace backbone wrapper.

Exposes get_intermediate_layers() matching the DINOv2 torch.hub interface so
the rest of the MuSc pipeline (LNAMD / MSM) works without modification.

hidden_states layout from HuggingFace transformers ViT:
  hidden_states[0]  = initial patch+position embeddings (before any block)
  hidden_states[i]  = output of block (i-1)  for i >= 1
So to get 0-indexed block b: hidden_states[b + 1]
"""

import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class DINOv3Backbone:
    """Wraps a HuggingFace AutoModel (ViT) for use as a MuSc backbone.

    Args:
        model_id: HuggingFace model ID, e.g. 'facebook/dinov3-vitl16-pretrain-lvd1689m'
        device: torch.device to place the model on
    """

    def __init__(self, model_id: str, device: torch.device):
        from transformers import AutoModel
        self.model_id = model_id
        self.device = device
        self._model = AutoModel.from_pretrained(model_id)
        self._model.to(device)
        self._model.eval()

    @property
    def patch_size(self) -> int:
        return self._model.config.patch_size

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n,
        return_class_token: bool = False,
    ):
        """Extract intermediate transformer block outputs.

        Handles models with register tokens (CLS + N_reg + N_patch layout).
        n_prefix = total non-patch prefix tokens = seq_len - n_patches.

        Args:
            x: preprocessed image tensor (B, 3, H, W) on self.device
            n: list of 0-indexed block numbers, e.g. [5, 11, 17, 23]
            return_class_token: if False (default) → (B, N_patch, C)
                                 if True           → (B, 1 + N_patch, C)

        Returns:
            tuple of tensors, one per entry in n.
        """
        outputs = self._model(pixel_values=x, output_hidden_states=True)
        # hidden_states[i+1] = output of block i
        selected = [outputs.hidden_states[i + 1] for i in n]

        # n_patches from spatial grid (exact, handles non-square inputs)
        H, W = x.shape[-2], x.shape[-1]
        n_patches = (H // self.patch_size) * (W // self.patch_size)
        # n_prefix = CLS + any register tokens
        n_prefix = selected[0].shape[1] - n_patches

        if not return_class_token:
            # strip CLS + registers; keep only spatial patch tokens → (B, N_patch, C)
            selected = [s[:, n_prefix:, :] for s in selected]
        else:
            # keep CLS at position 0, drop registers, keep patches
            # result: (B, 1 + N_patch, C)
            if n_prefix > 1:
                cls  = [s[:, :1, :]       for s in selected]
                ptok = [s[:, n_prefix:, :] for s in selected]
                selected = [torch.cat([cls[i], ptok[i]], dim=1) for i in range(len(selected))]
            # if n_prefix == 1: only CLS, no registers — nothing to drop

        return tuple(selected)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Returns global image representation (pooler or CLS token)."""
        outputs = self._model(pixel_values=x)
        if outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]

    def get_preprocess(self, image_size: int) -> transforms.Compose:
        """Standard ImageNet transform matching DINOv3 training pipeline."""
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def load_dinov3(model_id: str, device: torch.device) -> DINOv3Backbone:
    """Load DINOv3 model from HuggingFace and return wrapped backbone."""
    return DINOv3Backbone(model_id, device)
