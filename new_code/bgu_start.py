import cv2
import torch
import numpy as np

from polex_refactored import (
    to_torch_bgra,
    extract_potential_alignments,
    score_alignment,
    warp_single_image,
    tensor_to_rgba,
    pad_to_same_size,
)

# 1) Load BGRA images (with alpha) as numpy arrays
tgt_np  = cv2.imread("target.png",     cv2.IMREAD_UNCHANGED)
src_np  = cv2.imread("source.png",     cv2.IMREAD_UNCHANGED)
#ext_tgt = cv2.imread("target_ext.png", cv2.IMREAD_UNCHANGED)
#ext_src = cv2.imread("source_ext.png", cv2.IMREAD_UNCHANGED)

# 2) Convert to [1,4,H,W] CUDA tensors
dev   = torch.device("cuda")
tgt_t = to_torch_bgra(tgt_np,  pad_by=200, device=dev)
src_t = to_torch_bgra(src_np,  pad_by=200, device=dev)
#ext_t = to_torch_bgra(ext_tgt, pad_by=200, device=dev)
#ext_s = to_torch_bgra(ext_src, pad_by=200, device=dev)

# 3) Extract all geometric candidates
cands = extract_potential_alignments(
    tgt_t, src_t,
    gap=2.0,
    min_edge_length=20.0,
    min_length_ratio=0.8,
    epsilon_ratio=0.005,
    angle_threshold_deg=10.0,
    smoothing_kernel_size=3,
    pad_by=200
)

# 4) Score each on the extrapolated bands
for c in cands:
    c["score"] = 1
    #c["score"] = score_alignment(
    #   tgt_t, src_t, ext_t, ext_s,
    #    c, pure_dissimilarity=True
    #)

# 5) Keep top-5, warp & visualize
best = sorted(cands, key=lambda x: x["score"])#[:5]
for i, b in enumerate(best, 1):
    ang = torch.tensor([b["rotation"]], device=dev)
    tx  = torch.tensor([b["translation_x"]], device=dev)
    ty  = torch.tensor([b["translation_y"]], device=dev)
    warped = warp_single_image(src_t, ang, tx, ty)

    tgt_rgba    = tensor_to_rgba(tgt_t)
    warped_rgba = tensor_to_rgba(warped)
    pad_t, pad_w = pad_to_same_size(tgt_rgba, warped_rgba)
    #overlay = cv2.addWeighted(pad_t, 1.0, pad_w, 1.0, 0)

    overlay = pad_t + pad_w
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    #overlay = np.array(overlay)

    # display with matplotlib:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4,4))
    #plt.imshow(overlay)
    plt.title(f"Candidate {i}, score={b['score']:.4f}")
    plt.axis('off')
    #plt.show()
    plt.imsave(f"2Candidate {i}, score={b['score']:.4f}.png", overlay)