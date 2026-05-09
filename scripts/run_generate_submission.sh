#!/bin/bash
# Generate MVTec AD2 submission using threshold from mvtec2.log (0.628772)

python scripts/generate_submission.py \
    --data_path ./data/mvtec_ad_2/ \
    --submission_dir ./submission_folder \
    --backbone_name ViT-L-14-336 \
    --pretrained openai \
    --img_resize 518 \
    --feature_layers 5 11 17 23 \
    --r_list 1 3 5 \
    --batch_size 4 \
    --device 0 \
    --threshold 0.628772
