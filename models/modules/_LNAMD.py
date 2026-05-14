"""
    PatchMaker, Preprocessing and MeanMapper are copied from https://github.com/amazon-science/patchcore-inspection.
    SNAMD (Similarity Neighborhood Aggregation with Multi-Degrees) from MuSc-V2.
"""

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import math

class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features


class SWPooling(torch.nn.Module):
    """Similarity-Weighted Pooling from MuSc-V2 SNAMD.

    Weights each neighbor by exp(-||F_neighbor - F_center||^2) before averaging,
    preserving small anomaly signals that uniform pooling dilutes.
    r=1 degenerates to identity (only neighbor is the center itself).
    """

    def __init__(self, r):
        super(SWPooling, self).__init__()
        self.r = r
        self.n_neighbors = r * r

    def forward(self, features):
        # features: (B*P, C, r, r)
        BP, C, _, _ = features.shape
        center = features[:, :, self.r // 2, self.r // 2]       # (BP, C)
        neigh  = features.reshape(BP, C, -1).permute(0, 2, 1)   # (BP, r*r, C)
        diff   = neigh - center.unsqueeze(1)                     # (BP, r*r, C)
        lam    = torch.exp(-(diff ** 2).sum(dim=-1))             # (BP, r*r)
        # mean(Λ ⊙ F(N)): weighted sum divided by |N| (not by sum of weights)
        F_agg  = (lam.unsqueeze(-1) * neigh).sum(dim=1) / self.n_neighbors  # (BP, C)
        return F_agg


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

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class LNAMD(torch.nn.Module):
    def __init__(self, device, feature_dim=1024, feature_layer=[1,2,3,4], r=3, patchstride=1):
        super(LNAMD, self).__init__()
        self.device = device
        self.r = r
        self.patch_maker = PatchMaker(r, stride=patchstride)
        self.LNA = Preprocessing(feature_layer, feature_dim)

    def _embed(self, features):
        B = features[0].shape[0]

        features_layers = []
        for feature in features:
            # reshape and layer normalization
            feature = feature[:, 1:, :] # remove the cls token
            feature = feature.reshape(feature.shape[0],
                                      int(math.sqrt(feature.shape[1])),
                                      int(math.sqrt(feature.shape[1])),
                                      feature.shape[2])
            feature = feature.permute(0, 3, 1, 2)
            feature = torch.nn.LayerNorm([feature.shape[1], feature.shape[2],
                                          feature.shape[3]]).to(self.device)(feature)
            features_layers.append(feature)

        if self.r != 1:
            # divide into patches
            features_layers = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features_layers]
            patch_shapes = [x[1] for x in features_layers]
            features_layers = [x[0] for x in features_layers]
        else:
            patch_shapes = [f.shape[-2:] for f in features_layers]
            features_layers = [f.reshape(f.shape[0], f.shape[1], -1, 1, 1).permute(0, 2, 1, 3, 4) for f in features_layers]

        ref_num_patches = patch_shapes[0]
        for i in range(1, len(features_layers)):
            patch_dims = patch_shapes[i]
            if patch_dims[0] == ref_num_patches[0] and patch_dims[1] == ref_num_patches[1]:
                continue
            _features = features_layers[i]
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features_layers[i] = _features
        features_layers = [x.reshape(-1, *x.shape[-3:]) for x in features_layers]

        # aggregation
        features_layers = self.LNA(features_layers)
        features_layers = features_layers.reshape(B, -1, *features_layers.shape[-2:])   # (B, L, layer, C)

        return features_layers.detach().cpu()


def _spatial_align(features_layers, patch_shapes):
    """Align all layers to the spatial resolution of the first layer via bilinear interpolation."""
    ref_num_patches = patch_shapes[0]
    for i in range(1, len(features_layers)):
        patch_dims = patch_shapes[i]
        if patch_dims[0] == ref_num_patches[0] and patch_dims[1] == ref_num_patches[1]:
            continue
        _features = features_layers[i]
        _features = _features.reshape(
            _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
        )
        _features = _features.permute(0, -3, -2, -1, 1, 2)
        perm_base_shape = _features.shape
        _features = _features.reshape(-1, *_features.shape[-2:])
        _features = F.interpolate(
            _features.unsqueeze(1),
            size=(ref_num_patches[0], ref_num_patches[1]),
            mode="bilinear",
            align_corners=False,
        )
        _features = _features.squeeze(1)
        _features = _features.reshape(
            *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
        )
        _features = _features.permute(0, -2, -1, 1, 2, 3)
        _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
        features_layers[i] = _features
    return features_layers


class SNAMD(torch.nn.Module):
    """Similarity Neighborhood Aggregation with Multi-Degrees (MuSc-V2).

    Replaces LNAMD for zero-shot anomaly detection. Key differences:
    - SWPooling instead of uniform MeanMapper (preserves small anomaly signals)
    - All r-scales processed in one _embed call, features concatenated to 3C
    - Single MSM pass on (N, P, 3C) instead of 3 separate (N, P, C) passes

    _embed() returns (B, P, L, 3C) L2-normalized CPU tensor.
    Callers do NOT call .norm() on the output.
    """

    def __init__(self, device, feature_dim=1024, feature_layer=[1, 2, 3, 4],
                 r_list=[1, 3, 5], patchstride=1):
        super(SNAMD, self).__init__()
        self.device = device
        self.r_list = r_list
        self.feature_dim = feature_dim
        self.feature_layer = feature_layer
        self.patch_makers = {r: PatchMaker(r, stride=patchstride) for r in r_list}
        # ModuleDict keys must be strings
        self.processors = torch.nn.ModuleDict({
            str(r): Preprocessing(feature_layer, feature_dim, use_sw_pooling=True, r=r)
            for r in r_list
        })

    def _embed_single_r(self, features, r):
        """Extract and SWPool neighborhood features for radius r.
        Returns (B, P, L, C) CPU tensor, NOT normalized.
        """
        B = features[0].shape[0]
        features_layers = []
        for feature in features:
            feature = feature[:, 1:, :]  # remove CLS token
            feature = feature.reshape(
                B,
                int(math.sqrt(feature.shape[1])),
                int(math.sqrt(feature.shape[1])),
                feature.shape[2],
            )
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

        features_layers = _spatial_align(features_layers, patch_shapes)
        features_layers = [x.reshape(-1, *x.shape[-3:]) for x in features_layers]
        # each: (B*P, C, r, r)

        proc = self.processors[str(r)]
        agg = proc(features_layers)                         # (B*P, L, C)
        agg = agg.reshape(B, -1, *agg.shape[-2:])           # (B, P, L, C)
        return agg.detach().cpu()

    def _embed(self, features):
        """Multi-scale SNAMD embed.

        Runs _embed_single_r for each r in r_list, concatenates on the feature
        dimension, and L2-normalizes the concatenated 3C vector.

        Returns: (B, P, L, len(r_list)*C) CPU tensor, unit-normed on last dim.
        Callers must NOT apply additional .norm() normalization.
        """
        agg_per_r = [self._embed_single_r(features, r) for r in self.r_list]
        combined = torch.cat(agg_per_r, dim=-1)             # (B, P, L, R*C)
        combined = combined / combined.norm(dim=-1, keepdim=True)
        return combined


if __name__ == "__main__":
    import time
    device = 'cuda:0'

    # LNAMD backward compat test
    LNAMD_r = LNAMD(device=device, r=3, feature_dim=1024, feature_layer=[1,2,3,4])
    B = 32
    patch_tokens = [torch.rand((B, 1370, 1024)) for _ in range(4)]
    patch_tokens = [f.to('cuda:0') for f in patch_tokens]
    s = time.time()
    features = LNAMD_r._embed(patch_tokens)
    e = time.time()
    print('LNAMD: {:.2f}ms/img, shape={}'.format((e-s)*1000/B, features.shape))

    # SNAMD test
    snamd = SNAMD(device=device, feature_dim=1024, feature_layer=[1,2,3,4], r_list=[1,3,5])
    s = time.time()
    out = snamd._embed(patch_tokens)
    e = time.time()
    print('SNAMD: {:.2f}ms/img, shape={}'.format((e-s)*1000/B, out.shape))
    norms = out.norm(dim=-1)
    print('SNAMD norms close to 1:', torch.allclose(norms, torch.ones_like(norms), atol=1e-4))
