Created:
  - models/sam_refiner.py — SAMRefiner class with 3-pass cascade: _get_anomaly_region
  (Otsu + fallback), _sample_points_with_spacing (top-k with min spacing),
  _get_bbox_from_mask (highest-scoring connected component), refine() (full cascade)

  Modified:
  - configs/musc.yaml — added sam: section (use_sam: False by default)
  - models/musc.py — SAM loaded in __init__, refinement runs per-image after heatmap
  computation, pr_px_sam_all used for SegF1 global threshold
  - scripts/generate_submission.py — added sam_refine_maps(), binary_override in
  save_maps, SAM args, SAM loaded once, applied in both Phase 1 (threshold compute) and
  Phase 2 (private splits); tiff still saves raw float heatmap, png uses SAM binary mask
  - scripts/compute_segf1_dinov3.py — SAM args added, image_paths captured from
  run_inference, SAM refinement applied before threshold computation

  To run with SAM:
  conda run -n clip python3 scripts/compute_segf1_dinov3.py \
    --backbone_name facebook/dinov3-vitl16-pretrain-lvd1689m \
    --data_path ./data/mvtec_ad_2/ --img_resize 512 \
    --use_sam --sam_checkpoint models/sam_vit_h.pth