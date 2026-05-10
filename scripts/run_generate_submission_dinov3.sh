#!/bin/bash
# Generate MVTec AD2 submission using DINOv3 backbone.
# Threshold is auto-computed from test_public (Phase 1).
# Submission saved to: ./dinov3_vitl16_submission_folder/

python scripts/generate_submission.py \
    --backbone_name facebook/dinov3-vitl16-pretrain-lvd1689m \
    --img_resize 512 \
    --feature_layers 5 11 17 23 \
    --r_list 1 3 5 \
    --batch_size 4 \
    --device 0
    # --threshold 0.XXXXXX   # uncomment to use a fixed threshold
