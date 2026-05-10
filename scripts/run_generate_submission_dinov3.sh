#!/bin/bash
# Generate MVTec AD2 submission using DINOv3 backbone.
# Threshold is auto-computed from test_public (Phase 1).
# Submission saved to: ./dinov3_vitl16_submission_folder/
#
# SAM refinement (optional): add --sam_enabled and point --sam_checkpoint
# to the downloaded SAM ViT-H weights (~2.4 GB).
#   pip install git+https://github.com/facebookresearch/segment-anything.git

python scripts/generate_submission.py \
    --backbone_name facebook/dinov3-vitl16-pretrain-lvd1689m \
    --img_resize 512 \
    --feature_layers 5 11 17 23 \
    --r_list 1 3 5 \
    --batch_size 4 \
    --device 0
    # --threshold 0.XXXXXX         # uncomment to skip Phase 1
    # --sam_enabled \              # uncomment to enable SAM boundary refinement
    # --sam_checkpoint ./models/sam_vit_h.pth \
    # --sam_model_type vit_h \
    # --sam_blend_alpha 0.3 \
    # --sam_threshold_percentile 95
