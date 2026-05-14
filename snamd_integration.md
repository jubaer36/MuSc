# SNAMD (MuSc-V2) Replacement Plan

## Context

Current LNAMD uses uniform average pooling (MeanMapper) over r×r neighborhoods before mutual scoring. This dilutes small anomaly signals — at r=5, one anomalous patch surrounded by 24 normal patches gets 1/25 of its contribution. MuSc-V2 fixes this with SWPooling (similarity-weighted pooling) and collapses the per-r MSM loop into a single pass on concatenated 3C features.

Goals:
1. Replace MeanMapper with SWPooling in neighborhood aggregation
2. Remove the outer `for r in r_list` loop — extract all 3 scales per batch, concat → 3C → one MSM pass
3. Backbone runs once per batch (not 3×)

**compute_segf1_dinov3.py imports run_inference from generate_submission.py (line 41-46) — gets changes automatically, no edits needed.**

---

## Critical Files

- `models/modules/_LNAMD.py` — add SWPooling, Preprocessing update, new SNAMD class
- `scripts/generate_submission.py` — restructure run_inference (remove outer r-loop)
- `models/musc.py` — restructure make_category_data (remove per-r LNAMD loop)

---

## Step 1: `models/modules/_LNAMD.py`

### Add SWPooling class

```python
class SWPooling(torch.nn.Module):
    def __init__(self, r):
        super(SWPooling, self).__init__()
        self.r = r
        self.n_neighbors = r * r

    def forward(self, features):
        # features: (B*P, C, r, r)
        BP, C, _, _ = features.shape
        center = features[:, :, self.r // 2, self.r // 2]      # (BP, C)
        neigh  = features.reshape(BP, C, -1).permute(0, 2, 1)  # (BP, r*r, C)
        diff   = neigh - center.unsqueeze(1)                    # (BP, r*r, C)
        lam    = torch.exp(-(diff ** 2).sum(dim=-1))            # (BP, r*r)
        F_agg  = (lam.unsqueeze(-1) * neigh).sum(dim=1) / self.n_neighbors  # (BP, C)
        return F_agg
```

Normalization: `mean(Λ ⊙ F(N))` = divide by |N| (not by sum of weights). Mean is regular, weights are scaling factors.

r=1 degenerates correctly: diff=0, lam=1, F_agg = center.

### Modify Preprocessing

```python
class Preprocessing(torch.nn.Module):
    def __init__(self, input_layers, output_dim, use_sw_pooling=False, r=3):
        super(Preprocessing, self).__init__()
        self.output_dim = output_dim
        self.preprocessing_modules = torch.nn.ModuleList()
        for _ in input_layers:
            if use_sw_pooling:
                module = SWPooling(r=r)
            else:
                module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)
    # forward: unchanged
```

### Add SNAMD class (LNAMD untouched)

New class `SNAMD` in same file. Stores one PatchMaker + Preprocessing per r value at init time.

```python
class SNAMD(torch.nn.Module):
    def __init__(self, device, feature_dim=1024, feature_layer=[1,2,3,4],
                 r_list=[1,3,5], patchstride=1):
        super(SNAMD, self).__init__()
        self.device = device
        self.r_list = r_list
        self.feature_dim = feature_dim
        self.feature_layer = feature_layer
        # One PatchMaker and Preprocessing per r — stateless, no learned params
        self.patch_makers = {r: PatchMaker(r, stride=patchstride) for r in r_list}
        self.processors = torch.nn.ModuleDict({
            str(r): Preprocessing(feature_layer, feature_dim, use_sw_pooling=True, r=r)
            for r in r_list
        })

    def _embed_single_r(self, features, r):
        """Like LNAMD._embed but for a specific r using SWPooling.
        Returns (B, P, L, C) CPU tensor, NOT normalized.
        """
        B = features[0].shape[0]
        features_layers = []
        for feature in features:
            feature = feature[:, 1:, :]  # remove CLS
            feature = feature.reshape(B, int(math.sqrt(feature.shape[1])),
                                      int(math.sqrt(feature.shape[1])), feature.shape[2])
            feature = feature.permute(0, 3, 1, 2)
            feature = torch.nn.LayerNorm(
                [feature.shape[1], feature.shape[2], feature.shape[3]]
            ).to(self.device)(feature)
            features_layers.append(feature)

        pm = self.patch_makers[r]
        if r != 1:
            features_layers = [pm.patchify(x, return_spatial_info=True) for x in features_layers]
            patch_shapes = [x[1] for x in features_layers]
            features_layers = [x[0] for x in features_layers]
        else:
            patch_shapes = [f.shape[-2:] for f in features_layers]
            features_layers = [
                f.reshape(f.shape[0], f.shape[1], -1, 1, 1).permute(0, 2, 1, 3, 4)
                for f in features_layers
            ]

        # spatial alignment (identical to LNAMD._embed lines 98-121)
        ref_num_patches = patch_shapes[0]
        for i in range(1, len(features_layers)):
            patch_dims = patch_shapes[i]
            if patch_dims[0] == ref_num_patches[0] and patch_dims[1] == ref_num_patches[1]:
                continue
            _features = features_layers[i]
            _features = _features.reshape(_features.shape[0], patch_dims[0], patch_dims[1],
                                          *_features.shape[2:])
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(_features.unsqueeze(1),
                                      size=(ref_num_patches[0], ref_num_patches[1]),
                                      mode="bilinear", align_corners=False).squeeze(1)
            _features = _features.reshape(*perm_base_shape[:-2],
                                          ref_num_patches[0], ref_num_patches[1])
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features_layers[i] = _features

        features_layers = [x.reshape(-1, *x.shape[-3:]) for x in features_layers]
        # each: (B*P, C, r, r)

        proc = self.processors[str(r)]
        agg = proc(features_layers)         # (B*P, L, C)
        agg = agg.reshape(B, -1, *agg.shape[-2:])  # (B, P, L, C)
        return agg.detach().cpu()

    def _embed(self, features):
        """Multi-scale SNAMD embed. Returns (B, P, L, 3C) normalized CPU tensor.
        Drop-in replacement for LNAMD._embed — callers do NOT call .norm() after this.
        """
        agg_per_r = [self._embed_single_r(features, r) for r in self.r_list]
        combined = torch.cat(agg_per_r, dim=-1)  # (B, P, L, 3C)
        combined = combined / combined.norm(dim=-1, keepdim=True)
        return combined
```

**Shape flow:**
```
patch_tokens: list of L × (B, P+1, C)
_embed_single_r for each r:
  → (B, P, L, C)  [not normalized]
cat on dim=-1:
  → (B, P, L, 3C)
normalize:
  → (B, P, L, 3C)  unit vectors in 3C space
```

---

## Step 2: `scripts/generate_submission.py` — restructure `run_inference`

**Change:** Remove outer `for r_idx, r in enumerate(r_list)` loop. Single dataloader pass. SNAMD handles r_list internally.

**New signature:** Add `use_snamd=True` kwarg (default on).

```python
def run_inference(dataset, model, backbone_type, features_list, r_list, device,
                  batch_size, image_size, with_masks=False, use_snamd=True):
    from models.modules._LNAMD import SNAMD, LNAMD

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    Z_layers = {}
    image_path_list = []
    collected_masks = [] if with_masks else None
    embed_model = None  # SNAMD or LNAMD, init on first batch

    for batch in tqdm(dataloader):
        image = batch["image"]
        image_path_list.extend(batch["image_path"])
        if with_masks and "mask" in batch:
            collected_masks.append(batch["mask"])

        with torch.no_grad(), torch.cuda.amp.autocast():
            input_img = image.to(torch.float).to(device)
            patch_tokens = extract_patch_tokens(model, input_img, backbone_type, features_list)
            # ↑ ONE backbone forward pass per batch

            if embed_model is None:
                feature_dim = patch_tokens[0].shape[-1]
                if use_snamd:
                    embed_model = SNAMD(device=device, feature_dim=feature_dim,
                                        feature_layer=features_list, r_list=r_list)
                else:
                    # legacy path preserved
                    embed_model = LNAMD(device=device, r=r_list[0], feature_dim=feature_dim,
                                        feature_layer=features_list)

            features = embed_model._embed(patch_tokens)
            # SNAMD: (B, P, L, 3C) — already normalized
            # LNAMD: (B, P, L, C) — NOT normalized (legacy path adds norm below)
            if not use_snamd:
                features = features / features.norm(dim=-1, keepdim=True)

        for l in range(len(features_list)):
            key = str(l)
            if key not in Z_layers:
                Z_layers[key] = []
            Z_layers[key].append(features[:, :, l, :])  # (B, P, 3C or C)

        del patch_tokens, features
        torch.cuda.empty_cache()

    del embed_model
    gc.collect()

    # Single MSM per layer
    maps_per_layer = []
    for l_key in sorted(Z_layers.keys()):
        Z = torch.cat(Z_layers[l_key], dim=0).to(device)  # (N, P, 3C)
        del Z_layers[l_key]
        torch.cuda.empty_cache()
        maps_msm = MSM(Z=Z, device=device, topmin_min=0, topmin_max=0.3)
        maps_per_layer.append(maps_msm.cpu().float())
        del Z, maps_msm
        torch.cuda.empty_cache()

    del Z_layers
    gc.collect()

    anomaly_maps = torch.stack(maps_per_layer, dim=0).mean(0).to(device)  # (N, P)
    del maps_per_layer
    B_n, L_p = anomaly_maps.shape
    H = int(np.sqrt(L_p))
    anomaly_maps = F.interpolate(anomaly_maps.view(B_n, 1, H, H),
                                  size=image_size, mode="bilinear", align_corners=True)
    result = anomaly_maps.cpu().float().numpy()
    del anomaly_maps
    torch.cuda.empty_cache()
    gc.collect()

    if with_masks:
        gt = torch.cat(collected_masks, dim=0).numpy() if collected_masks else None
        return result, image_path_list, gt
    return result, image_path_list
```

**Also update `compute_threshold_from_public` and private-split loop in `generate_submission.py`** — both also call LNAMD+MSM in the same per-r pattern. Apply same restructuring using `use_snamd=True` and SNAMD._embed.

---

## Step 3: `models/musc.py` — restructure `make_category_data`

Replace lines 207-243 (the per-r LNAMD loop):

```python
# After accumulating patch_tokens_list...

feature_dim = patch_tokens_list[0][0].shape[-1]
from models.modules._LNAMD import SNAMD
snamd = SNAMD(device=self.device, feature_dim=feature_dim,
              feature_layer=self.features_list, r_list=self.r_list)

Z_layers = {}
start_time = time.time()
for im in range(len(patch_tokens_list)):
    patch_tokens = [p.to(self.device) for p in patch_tokens_list[im]]
    with torch.no_grad(), torch.cuda.amp.autocast():
        features = snamd._embed(patch_tokens)  # (B, P, L, 3C) normalized
    for l in range(len(self.features_list)):
        key = str(l)
        if key not in Z_layers:
            Z_layers[key] = []
        Z_layers[key].append(features[:, :, l, :])  # (B, P, 3C)
end_time = time.time()
print('SNAMD embed: {}ms per image'.format((end_time-start_time)*1000/subset_num))

del snamd
gc.collect()

# Single MSM pass (was: per-r MSM × 3)
anomaly_maps_l = torch.tensor([]).double()
start_time = time.time()
for l in sorted(Z_layers.keys()):
    Z = torch.cat(Z_layers[l], dim=0).to(self.device)  # (N, P, 3C)
    print('layer-{} mutual scoring...'.format(l))
    anomaly_maps_msm = MSM(Z=Z, device=self.device, topmin_min=0, topmin_max=0.3)
    anomaly_maps_l = torch.cat(
        (anomaly_maps_l, anomaly_maps_msm.unsqueeze(0).cpu()), dim=0
    )
    del Z, anomaly_maps_msm
    torch.cuda.empty_cache()
end_time = time.time()
print('MSM: {}ms per image'.format((end_time-start_time)*1000/subset_num))

anomaly_maps_iter = torch.mean(anomaly_maps_l, 0).to(self.device)  # (N, P)
del anomaly_maps_l, Z_layers
torch.cuda.empty_cache()
```

---

## Memory Budget

| Scenario | Z shape | GPU RAM |
|---|---|---|
| N=200, P=1369, C=1024 | (200, 1369, 3072) | ~3.4 GB |
| N=300, P=1369, C=1024 | (300, 1369, 3072) | ~5.0 GB |
| N=400, P=1369, C=1024 | (400, 1369, 3072) | ~6.7 GB |

Chunked MSM keeps cdist intermediates at `(P, chunk*P)` — ~384 MB at 3C, chunk=32. Z is the dominant cost.

---

## Normalization Order (Critical)

**Wrong (per-r normalization):**
```
F_r1 = embed(r=1); F_r1 /= norm  # C-space unit vectors
F_r3 = embed(r=3); F_r3 /= norm
concat → 3C: each block separately unit-normed, scales differ
```

**Correct (concat then normalize):**
```
F_r1 = embed(r=1)  # raw
F_r3 = embed(r=3)  # raw
cat → 3C; then 3C_vec /= norm  # 3C unit vector
```

Encapsulated in `SNAMD._embed` — callers never call `.norm()` on returned features.

---

## Verification

### Unit tests

```python
# 1. r=1 sanity: SWPooling = identity
feat = torch.randn(100, 512, 1, 1)
sw = SWPooling(r=1)
out = sw(feat)
assert torch.allclose(out, feat.squeeze(-1).squeeze(-1), atol=1e-5)

# 2. Shape check for SNAMD._embed
B, C, L = 4, 1024, 4
patch_tokens = [torch.rand(B, 1370, C).cuda() for _ in range(L)]
snamd = SNAMD(device='cuda:0', feature_dim=C, feature_layer=list(range(L)), r_list=[1,3,5])
out = snamd._embed(patch_tokens)
assert out.shape == (B, 1369, L, 3*C)
assert torch.allclose(out.norm(dim=-1), torch.ones(B, 1369, L).cuda(), atol=1e-5)

# 3. Backward compat: LNAMD unchanged
lnamd = LNAMD(device='cuda:0', r=3, feature_dim=C, feature_layer=list(range(L)))
out_old = lnamd._embed(patch_tokens)  # still works
assert out_old.shape == (B, 1369, L, C)
```

### End-to-end

Run `compute_segf1_dinov3.py` on one category with `--use_snamd` (add flag) vs without. Confirm:
- No OOM on target GPU
- Anomaly maps shape `(N, 1, img_size, img_size)` unchanged
- SegF1 score plausibly different (algorithm change, not a bug)
- Log shows single MSM pass per layer (not 3×)

---

## Implementation Order

1. `_LNAMD.py` — SWPooling + Preprocessing update + SNAMD class
2. `generate_submission.py` — restructure `run_inference` + `compute_threshold_from_public` + private loop
3. `musc.py` — restructure `make_category_data`
4. Run unit tests
5. Run integration test on one category
