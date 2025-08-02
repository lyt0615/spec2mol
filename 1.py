import torch
import numpy as np
mask_ratio=0.15
silent_start=1800
silent_end=2800
wavenumbers=torch.linspace(4000,400,1024)
spec = torch.randn(2,1,1024)
B, *_ = spec.shape
len_keep = int((1 - mask_ratio) * spec[0].numel())
noise = torch.rand(B, spec[0].numel(), device=spec.device)
ids_shuffle = torch.argsort(noise, dim=1)
ids_restore = torch.argsort(ids_shuffle, dim=1)

mask = torch.ones_like(spec.view(B, -1))
mask[:, :len_keep] = 0
mask = torch.gather(mask, 1, ids_restore).view_as(spec).bool()

masked_spec = spec.masked_fill(mask, 0.0)
    
    # mask_ratio=0.15
# silent_start=1800
# silent_end=2800
# wavenumbers=torch.linspace(4000,400,1024)
# spec = torch.randn(2,1,1024)
# B, *_ = spec.shape
# flat_len = spec[0].numel()
# # 1. Build boolean mask of *allowed* positions
# if wavenumbers is not None:
#     # wavenumbers assumed monotonically increasing
#     valid = (wavenumbers < silent_start) | (wavenumbers > silent_end)   # [L]
# else:
#     # fallback: assume full spectrum is valid
#     valid = torch.ones(flat_len, dtype=torch.bool, device=spec.device)
# # 2. Sample indices only from allowed positions
# valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)          # [K]
# K = valid_idx.numel()
# num_mask = int(mask_ratio * K)
# perm = torch.randperm(K, device=spec.device)[:num_mask]
# mask_indices = valid_idx[perm]                                 # [num_mask]
# # 3. Build full mask tensor (True = masked)
# mask = torch.zeros(flat_len, dtype=torch.bool, device=spec.device)
# mask[mask_indices] = True
# # mask = mask.view_as(spec)
# masked_spec = spec.masked_fill(mask, 0.0)
# ids_restore = torch.arange(flat_len, device=spec.device).repeat(B, 1)
# print 