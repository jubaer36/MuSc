Step 0 — DINOv3 fix in musc.py:
  - load_backbone(): dinov3/facebook/ checked first (before dino catchall), sets
   self.backbone_type
  - Feature extraction: if self.backbone_type == 'dinov3' → uses fake_cls trick
  (same as generate_submission.py)
  - Old dino/dinov2 branches preserved intact

  Step 1 — utils/prompt_utils.py (new):
  - heatmap_to_prompts: scipy.ndimage.label on percentile-thresholded heatmap →
  peak pixel + bbox + neg samples per region
  - predict_region: runs SAM in 3 modes (point-only, box-only, combined), merges
   by SAM confidence weights
  - merge_region_masks: union of all region masks

  Step 2 — models/sam_refiner.py (new):
  - SAMRefiner.__init__: loads SamPredictor from segment_anything
  - refine(image_rgb, heatmap): normalizes heatmap, runs prompts, blends
  alpha*SAM + (1-alpha)*norm_heatmap
  - refine_batch(image_path_list, anomaly_maps, image_size): reloads images from
   paths (avoids RAM caching), returns (B, 1, H, W)
   
  Step 3 — configs/musc.yaml: Added sam: block with all params, defaults
  enabled: False

  Step 4 — models/musc.py wired:
  - _load_sam(): lazy-loads SAMRefiner only if sam.enabled
  - ac_score computed from raw MuSc maps (RsCIN unchanged)
  - pr_px = SAM-refined if enabled, else raw maps

  Step 5 — examples/musc_main.py: All --sam_* CLI args added

  Pilot command (bottle, DINOv3):
  python examples/musc_main.py \
    --backbone_name facebook/dinov3-vitl16-pretrain-lvd1689m \
    --dataset_name mvtec_ad \
    --data_path ./data/mvtec_anomaly_detection/ \
    --class_name bottle \
    --img_resize 512 \
    --feature_layers 5 11 17 23 \
    --r_list 1 3 5 \
    --sam_enabled True \
    --sam_checkpoint ./models/sam_vit_h.pth \
    --sam_blend_alpha 0.3
   
  Need segment-anything installed and sam_vit_h_4b8939.pth at
  ./models/sam_vit_h.pth before running with SAM enabled. Without --sam_enabled 
  True, pipeline runs exactly as before.



python scripts/generate_submission.py 
      --backbone_name facebook/dinov3-vitl16-pretrain-lvd1689m 
      --img_resize 512 
      --feature_layers 5 11 17 23 
      --r_list 1 3 5 
      --device 0 
      --sam_enabled 
      --sam_checkpoint ./models/sam_vit_h.pth 
      --sam_blend_alpha 0.3