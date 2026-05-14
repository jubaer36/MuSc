import torch
from tqdm import tqdm

"""
We provide two implementations of the MSM module.
The above commented out function provides faster speeds, but because more tensors are loaded onto the GPU at once, the memory consumption is higher.
By default, our program uses the following function, which is slower but consumes less GPU memory.
"""

def compute_scores_fast(Z, i, device, topmin_min=0, topmin_max=0.3):
    # speed fast but space large
    # compute anomaly scores
    image_num, patch_num, c = Z.shape
    patch2image = torch.tensor([]).to(device)
    Z_ref = torch.cat((Z[:i], Z[i+1:]), dim=0)
    patch2image = torch.cdist(Z[i:i+1], Z_ref.reshape(-1, c)).reshape(patch_num, image_num-1, patch_num)
    patch2image = torch.min(patch2image, -1)[0]
    # interval average
    k_max = topmin_max
    k_min = topmin_min
    if k_max < 1:
        k_max = int(patch2image.shape[1]*k_max)
    if k_min < 1:
        k_min = int(patch2image.shape[1]*k_min)
    if k_max < k_min:
        k_max, k_min = k_min, k_max
    vals, _ = torch.topk(patch2image.float(), k_max, largest=False, sorted=True)
    vals, _ = torch.topk(vals.float(), k_max-k_min, largest=True, sorted=True)
    patch2image = vals.clone()
    return torch.mean(patch2image, dim=1)

def compute_scores_slow(Z, i, device, topmin_min=0, topmin_max=0.3):
    # space small but speed slow
    # compute anomaly scores
    patch2image = torch.tensor([]).to(device)
    for j in range(Z.shape[0]):
        if j != i:
            patch2image = torch.cat((patch2image, torch.min(torch.cdist(Z[i], Z[j]), 1)[0].unsqueeze(1)), dim=1)
    # interval average
    k_max = topmin_max
    k_min = topmin_min
    if k_max < 1:
        k_max = int(patch2image.shape[1]*k_max)
    if k_min < 1:
        k_min = int(patch2image.shape[1]*k_min)
    if k_max < k_min:
        k_max, k_min = k_min, k_max
    vals, _ = torch.topk(patch2image.float(), k_max, largest=False, sorted=True)
    vals, _ = torch.topk(vals.float(), k_max-k_min, largest=True, sorted=True)
    patch2image = vals.clone()
    return torch.mean(patch2image, dim=1)

def compute_scores_chunked(Z, i, device, topmin_min=0, topmin_max=0.3, ref_chunk=32):
    """Same result as compute_scores_fast but processes ref images in chunks.
    Limits peak cdist intermediate to (patch_num, ref_chunk*patch_num) instead of
    (patch_num, (N-1)*patch_num), reducing peak VRAM from ~2.4 GB to ~0.25 GB for N=321.
    Z should be on GPU (float16 recommended to keep VRAM manageable with 3C features)."""
    image_num, patch_num, c = Z.shape
    Z_i = Z[i]  # (patch_num, c)
    ref_indices = list(range(0, i)) + list(range(i + 1, image_num))

    patch2image_parts = []
    for chunk_start in range(0, len(ref_indices), ref_chunk):
        chunk_idx = ref_indices[chunk_start:chunk_start + ref_chunk]
        Z_chunk = Z[chunk_idx]  # (chunk, patch_num, c)
        dists = torch.cdist(Z_i, Z_chunk.reshape(-1, c))  # (patch_num, chunk*patch_num)
        min_dists = torch.min(
            dists.reshape(patch_num, len(chunk_idx), patch_num), dim=2
        )[0]  # (patch_num, chunk)
        patch2image_parts.append(min_dists)
        del dists, Z_chunk, min_dists

    patch2image = torch.cat(patch2image_parts, dim=1)  # (patch_num, image_num-1)
    del patch2image_parts

    k_max = topmin_max
    k_min = topmin_min
    if k_max < 1:
        k_max = int(patch2image.shape[1] * k_max)
    if k_min < 1:
        k_min = int(patch2image.shape[1] * k_min)
    if k_max < k_min:
        k_max, k_min = k_min, k_max
    vals, _ = torch.topk(patch2image.float(), k_max, largest=False, sorted=True)
    vals, _ = torch.topk(vals.float(), k_max - k_min, largest=True, sorted=True)
    return torch.mean(vals, dim=1)


def MSM(Z, device, topmin_min=0, topmin_max=0.3, ref_chunk=32):
    anomaly_scores_matrix = torch.tensor([]).double().to(device)
    for i in tqdm(range(Z.shape[0])):
        anomaly_scores_i = compute_scores_chunked(Z, i, device, topmin_min, topmin_max, ref_chunk=ref_chunk).unsqueeze(0)
        anomaly_scores_matrix = torch.cat((anomaly_scores_matrix, anomaly_scores_i.double()), dim=0)
    return anomaly_scores_matrix

if __name__ == "__main__":
    device = 'cuda:0'
    import time
    s_time = time.time()
    Z = torch.rand(200, 1369, 1024).to(device)
    MSM(Z, device)
    e_time = time.time()
    print((e_time-s_time)*1000)
    