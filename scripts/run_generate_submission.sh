#!/bin/bash
# Generate MVTec AD2 submission using CLIP ViT-L-14-336.
# Uses fixed threshold from mvtec2.log.
# Submission saved to: ./vit_l_14_336_submission_folder/

python scripts/generate_submission.py \
    --backbone_name ViT-L-14-336 \
    --pretrained openai \
    --img_resize 518 \
    --feature_layers 5 11 17 23 \
    --r_list 1 3 5 \
    --batch_size 4 \
    --device 0 \
    --threshold 0.628772
