from typing import List, Dict, Optional
import numpy as np
from sympy import refine
import torch
from torch import Tensor
import cv2
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.ops as ops
import kornia


# def to_torch_bgra(npy, pad_by=200, device='cuda'):
#     padded_npy = np.pad(npy, ((pad_by, pad_by), (pad_by, pad_by), (0, 0)), mode='constant', constant_values=0)
#     t = torch.from_numpy(padded_npy.transpose(2,0,1)).float().unsqueeze(0).to(device)
#     return t/255.0

def to_torch_bgra(npy, pad_by=200, device='cuda'):
    """
    Accepts:
      - H×W  gray-scale array
      - H×W×3  BGR array
      - H×W×4  BGRA array
    Pads height and width by `pad_by`, adds missing channels as needed,
    and returns a [1,4,H+2*pad,W+2*pad] float tensor in [0..1] on the given device.
    """
    import numpy as np, torch

    arr = np.asarray(npy)
    # Step 1: ensure a H×W×4 BGRA numpy array
    if arr.ndim == 2:
        # gray → replicate to BGR, alpha=0
        h, w = arr.shape
        bgra = np.zeros((h, w, 4), dtype=arr.dtype)
        bgra[..., 0] = arr
        bgra[..., 1] = arr
        bgra[..., 2] = arr
        # alpha channel left at 0
    elif arr.ndim == 3:
        h, w, c = arr.shape
        if c == 4:
            bgra = arr
        elif c == 3:
            # BGR → add zero alpha
            alpha = np.zeros((h, w, 1), dtype=arr.dtype)
            bgra = np.concatenate([arr, alpha], axis=2)
        else:
            raise ValueError(f"to_torch_bgra: unsupported channel count {c}")
    else:
        raise ValueError(f"to_torch_bgra: unsupported array ndim {arr.ndim}")

    # Step 2: pad H and W, but never pad channels
    pad = pad_by
    pad_width = ((pad, pad), (pad, pad), (0, 0))  # only height/width
    padded = np.pad(bgra, pad_width, mode='constant', constant_values=0)

    # Step 3: HWC → CHW, add batch, normalize, move to device
    # pick CUDA if available
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    t = torch.from_numpy(padded.transpose(2, 0, 1)).float().unsqueeze(0).to(dev)
    return t / 255.0

def tensor_to_rgba(im_tensor):
    im_np = im_tensor[0].cpu().numpy()  # shape [4,H,W]
    im_np = np.clip(im_np * 255, 0, 255).astype(np.uint8)
    rgba = cv2.cvtColor(im_np.transpose(1,2,0), cv2.COLOR_BGRA2RGBA)
    return rgba


def pad_to_same_size(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    new_h = max(h1, h2)
    new_w = max(w1, w2)
    pad_img1 = np.zeros((new_h, new_w, img1.shape[2]), dtype=img1.dtype)
    pad_img2 = np.zeros((new_h, new_w, img2.shape[2]), dtype=img2.dtype)
    pad_img1[:h1, :w1] = img1
    pad_img2[:h2, :w2] = img2
    return pad_img1, pad_img2


def merge_collinear_segments(points: np.ndarray, angle_threshold_deg: float = 5.0):
    if len(points) < 3:
        return points

    new_pts = points.tolist()
    i = 0
    # Merge pass
    while i < len(new_pts) and len(new_pts) >= 3:
        i0 = i
        i1 = (i + 1) % len(new_pts)
        i2 = (i + 2) % len(new_pts)
        p0 = new_pts[i0]
        p1 = new_pts[i1]
        p2 = new_pts[i2]
        
        v1 = (p1[0] - p0[0], p1[1] - p0[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        
        dotp = v1[0]*v2[0] + v1[1]*v2[1]
        norm1 = math.sqrt(v1[0]**2 + v1[1]**2)
        norm2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if norm1 < 1e-8 or norm2 < 1e-8:
            angle = 0.0
        else:
            cosA = dotp / (norm1 * norm2)
            cosA = max(-1.0, min(1.0, cosA))
            angle = math.degrees(math.acos(cosA))
        
        if angle < angle_threshold_deg:
            # p1 is basically on a straight line between p0 and p2
            new_pts.pop(i1)
        else:
            i += 1
        
        # Safety check to avoid infinite loops
        if i > len(new_pts)*2:
            break

    return np.array(new_pts, dtype=points.dtype)


def extract_polygon_from_alpha(image_tensor: torch.Tensor, 
                               epsilon_ratio: float = 0.01,
                               min_contour_area: float = 100.0,
                               angle_threshold_deg: float = 10.0,
                               morph_kernel_size: int = 15,
                               smoothing_kernel_size: int = 3):
    """
    image_tensor shape: [N, 4, H, W], 
    where N=1, channels=[B, G, R, A], 
    and we want the alpha channel at index 3.
    """

    # 1) Extract alpha channel as a PyTorch tensor on GPU
    alpha_tensor = image_tensor[0, 3]  # shape: [H, W]

    # 2) Binarize the alpha (threshold > 0 => 1, else 0)
    mask = (alpha_tensor > 0).float()  # shape: [H, W]

    # 3) Morphological close in PyTorch on GPU
    kernel = torch.ones(
        (1, 1, morph_kernel_size, morph_kernel_size), 
        dtype=torch.float, 
        device=mask.device
    )
    mask = mask.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]

    # Dilation
    dilated = F.conv2d(mask, kernel, padding=morph_kernel_size // 2)
    dilated = (dilated > 0).float()

    # Erosion
    closed = F.conv2d(dilated, kernel, padding=morph_kernel_size // 2)
    closed = (closed == kernel.numel()).float()

    # 4) Convert back to a NumPy mask for OpenCV
    closed_mask = closed[0, 0].detach().cpu().numpy().astype(np.uint8) * 255

    # 4.5) Smoothing step:
    # Apply Gaussian blur to reduce the pixelated noise.
    smoothed_mask = cv2.GaussianBlur(closed_mask, (smoothing_kernel_size, smoothing_kernel_size), 0)
    # Re-threshold to get a clean binary mask.
    _, smoothed_mask = cv2.threshold(smoothed_mask, 127, 255, cv2.THRESH_BINARY)

    # 5) Find contours of the smoothed mask
    contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contour is found, return None
    if len(contours) == 0:
        return None

    # 6) Take the largest contour (or use union if desired)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    max_idx = np.argmax(areas)
    cnt = contours[max_idx]

    if cv2.contourArea(cnt) < min_contour_area:
        return None

    # 7) Polygonal approximation on the unified contour
    arc_len = cv2.arcLength(cnt, True)
    epsilon = epsilon_ratio * arc_len
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    polygon = approx.reshape(-1, 2)

    # 8) Optionally merge nearly collinear segments
    polygon_merged = merge_collinear_segments(polygon, angle_threshold_deg=angle_threshold_deg)

    return polygon_merged


def angle_between_edges(e1, e2):
    """
    Returns the absolute angle in degrees between two edges e1, e2.
    Each edge eX has 'angle' in radians => we can just compare them.
    """
    diff = abs(e1['angle'] - e2['angle'])
    # Because orientation can wrap around pi, let's ensure the difference is <= pi
    if diff > math.pi:
        diff = 2*math.pi - diff
    return math.degrees(diff)


def get_augmented_edges_from_polygon(
    polygon: np.ndarray,
    min_edge_length: float = 20.0,
    angle_threshold_deg: float = 10.0,
    small_edge_ratio: float = 1.5
):
    """
    1) Build the base edges from p[i]->p[i+1], skipping any with length < min_edge_length.
    2) Single-pass scanning for consecutive triplets (e1, e2, e3) in the *closed* polygon:
         If e2.length < small_edge_threshold
         AND angle_between(e1, e3) < angle_threshold_deg,
         => add a new 'span' edge from e1['p1'] -> e3['p2'].
    3) We do NOT remove any of the original edges (including the small one).
       We only ADD the new merged/spanned edge.
    4) Returns a final list of edges (the original ones plus any newly added).
    """
    small_edge_threshold = min_edge_length * small_edge_ratio
    N = len(polygon)
    if N < 3:
        return []

    # -------------------------------------------------------------------------
    # Step A: Build the base edges
    # -------------------------------------------------------------------------
    base_edges = []
    for i in range(N):
        p1 = polygon[i]
        p2 = polygon[(i+1) % N]
        vec = p2 - p1
        length = np.linalg.norm(vec)
        # if length < min_edge_length:
        #     continue
        angle_rad = math.atan2(vec[1], vec[0])
        midpoint = (p1 + p2) / 2.0
        base_edges.append({
            'p1': p1,
            'p2': p2,
            'midpoint': midpoint,
            'angle': angle_rad,
            'length': length
        })

    # We'll store the final edges in an "augmented" list
    augmented_edges = list(base_edges)  # copy them initially

    M = len(base_edges)
    if M < 3:
        # Not enough edges to form a triplet => no merges
        return augmented_edges

    # -------------------------------------------------------------------------
    # Step B: Single-pass scanning of triplets (e1, e2, e3)
    # -------------------------------------------------------------------------
    for i in range(M):
        e1 = base_edges[i]
        e2 = base_edges[(i+1) % M]
        e3 = base_edges[(i+2) % M]

        # 1) Check if e2 is "small"
        if e2['length'] >= small_edge_threshold:
            continue

        # 2) Check near-collinearity only between e1 & e3
        ab = angle_between_edges(e1, e3)
        if ab < angle_threshold_deg:
            # Then we add a new "span" edge from e1['p1'] -> e3['p2']
            p1 = e1['p1']
            p2 = e3['p2']
            vec = p2 - p1
            length = np.linalg.norm(vec)
            angle_rad = math.atan2(vec[1], vec[0])
            midpoint = (p1 + p2)/2.0
            new_edge = {
                'p1': p1,
                'p2': p2,
                'midpoint': midpoint,
                'angle': angle_rad,
                'length': length
            }
            augmented_edges.append(new_edge)
        
        # remove the small edges
        augmented_edges = [e for e in augmented_edges if e['length'] >= small_edge_threshold]

    return augmented_edges


def find_image_centroid_on_gpu(image_t: torch.Tensor):
    """
    [1,4,H,W] BGRA float in [0..1]. 
    Returns (cx, cy) as 2 GPU scalars (0D Tensors).
    """
    # alpha => [1,H,W]
    alpha = image_t[:, 3, :, :]
    alpha_flat = alpha.view(-1)              # [H*W]
    H, W = alpha.shape[-2], alpha.shape[-1]

    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=image_t.device, dtype=image_t.dtype),
        torch.arange(W, device=image_t.device, dtype=image_t.dtype),
        indexing='ij'
    )
    x_coords = x_coords.reshape(-1)
    y_coords = y_coords.reshape(-1)

    m00 = alpha_flat.sum()
    if m00 < 1e-5:
        # fallback => center
        cx_t = torch.tensor([W/2], device=image_t.device, dtype=image_t.dtype)
        cy_t = torch.tensor([H/2], device=image_t.device, dtype=image_t.dtype)
        return cx_t, cy_t

    m10 = (x_coords * alpha_flat).sum()
    m01 = (y_coords * alpha_flat).sum()
    cx_t = m10 / m00  # 0D Tensor
    cy_t = m01 / m00
    return cx_t, cy_t  # shape ()


def build_single_affine_matrix(
    image_t: torch.Tensor,
    angle_deg: torch.Tensor,  # 0D or 1D scalar on GPU
    tx_t: torch.Tensor,       # 0D scalar on GPU
    ty_t: torch.Tensor        # 0D scalar on GPU
):
    """
    Builds a single [1,2,3] matrix to rotate around alpha-based centroid 
    by angle_deg (ccw) and then translate by (tx, ty). 
    All on GPU, no .item().
    """
    device = image_t.device
    # angle_deg => 0D or 1D tensor
    angle_rad = angle_deg * (math.pi / 180.0)  
    cosA = torch.cos(angle_rad)
    sinA = torch.sin(angle_rad)

    B, C, H, W = image_t.shape
    cx, cy = find_image_centroid_on_gpu(image_t)  # both 0D Tensors

    # M = [
    #   [ cosA, -sinA,  cx - cx*cosA + cy*sinA + tx ],
    #   [ sinA,  cosA,  cy - cx*sinA - cy*cosA + ty ]
    # ]
    M = torch.zeros((1,2,3), device=device, dtype=image_t.dtype)
    M[:,0,0] = cosA
    M[:,0,1] = -sinA
    M[:,1,0] = sinA
    M[:,1,1] = cosA

    rot_offset_x = cx - cx*cosA + cy*sinA
    rot_offset_y = cy - cx*sinA - cy*cosA

    M[:,0,2] = rot_offset_x + tx_t
    M[:,1,2] = rot_offset_y + ty_t

    return M  # shape [1,2,3]

def warp_single_image(
    image_t: torch.Tensor,
    angle_deg: torch.Tensor,
    tx_t: torch.Tensor,
    ty_t: torch.Tensor
):
    """
    Warps a single [1,4,H,W] image with the single transform 
    built above. 
    Returns => [1,4,H,W].
    """
    B, C, H, W = image_t.shape
    M = build_single_affine_matrix(image_t, angle_deg, tx_t, ty_t)
    out = kornia.geometry.transform.warp_affine(
        image_t, M, dsize=(H, W),
        mode='bilinear',
        align_corners=True
    )
    return out


def extract_extrapolated_band_on_gpu(piece_t: torch.Tensor, piece_extrap_t: torch.Tensor):
    """
    GPU: zero out piece_extrap where piece_t has alpha>0.
    """
    alpha_mask = (piece_t[:, 3:4, :, :] > 0)
    out = piece_extrap_t.clone()
    out[alpha_mask.expand_as(out)] = 0.0
    return out


def arrange_canvases_on_gpu(target_band: torch.Tensor, src_band: torch.Tensor):
    B, C, Hd, Wd = target_band.shape
    _, _, Hs, Ws = src_band.shape
    device = target_band.device
    dtype = target_band.dtype

    new_w = max(Wd, Ws)
    new_h = max(Hd, Hs)

    out1 = torch.zeros((1,4,new_h,new_w), device=device, dtype=dtype)
    out2 = torch.zeros((1,4,new_h,new_w), device=device, dtype=dtype)

    out1[:, :, :Hd, :Wd] = target_band
    out2[:, :, :Hs, :Ws] = src_band

    alpha1 = (out1[:,3:4,:,:]>0)
    alpha2 = (out2[:,3:4,:,:]>0)
    overlap = alpha1 & alpha2
    overlap4 = overlap.expand(-1,4,-1,-1)
    out1 = out1 * overlap4
    out2 = out2 * overlap4
    return out1, out2


def lab_dissimilarity_score_random_patch_sizes_stride_on_gpu(
    imageA: torch.Tensor,  # [1,4,H,W] BGRA in [0,1]
    imageB: torch.Tensor,  # [1,4,H,W]
    min_patch_size: int = 8,
    max_patch_size: int = 32,
    stride: int = 16,
    aggregator: str = "pnorm",   # "mean", "max", or "pnorm"
    p_value: float = 4.0         # used if aggregator=="pnorm"
) -> torch.Tensor:
    """
    Computes the LAB dissimilarity between imageA and imageB using patches of random sizes
    sampled on a grid defined by the given stride.
    
    Steps:
      1) Compute the per-pixel LAB deltaE (Euclidean distance in LAB) between imageA and imageB.
      2) Create a valid overlap mask from the alpha channels.
      3) Define a grid of top-left patch positions (ensuring that a patch of size max_patch_size fits).
      4) For each grid point, sample a random patch size between min_patch_size and max_patch_size.
      5) Build ROIs from these grid positions and patch sizes.
      6) Use ROI align (with output_size=(1,1)) to compute the average deltaE and valid mask per patch.
      7) Compute the average dissimilarity for each patch (sum_deltaE divided by sum_valid).
      8) Aggregate the patch scores using the specified method.
      9) Normalize the final score by dividing by ~441.7.
      
    Returns:
      A single scalar tensor representing the normalized dissimilarity. If no overlap is found,
      returns 1.0.
    """
    device = imageA.device
    eps = 1e-8

    # 1) Create valid overlap mask from the alpha channel.
    alphaA = (imageA[:, 3:4, :, :] > 0)
    alphaB = (imageB[:, 3:4, :, :] > 0)
    overlap = alphaA & alphaB
    if not overlap.any():
        return torch.tensor([1.0], device=device, dtype=imageA.dtype)
    
    # 2) Convert BGRA -> LAB.
    def bgr_to_lab(bgra: torch.Tensor):
        bgr = bgra[:, 0:3, :, :]
        rgb = torch.flip(bgr, dims=[1])  # convert BGR -> RGB
        return kornia.color.rgb_to_lab(rgb)
    labA = bgr_to_lab(imageA)
    labB = bgr_to_lab(imageB)
    diff = labA - labB
    deltaE = torch.sqrt((diff * diff).sum(dim=1, keepdim=True))  # shape [1,1,H,W]
    
    # 3) Create valid_mask as float and zero out non-overlap in deltaE.
    valid_mask = overlap.float()
    deltaE = deltaE * valid_mask

    # 4) Determine image dimensions.
    _, _, H, W = deltaE.shape

    # 5) Create a grid for patch top–left corners.
    # We only choose positions where a patch of size max_patch_size will fully fit.
    xs = torch.arange(0, W - max_patch_size + 1, step=stride, device=device)
    ys = torch.arange(0, H - max_patch_size + 1, step=stride, device=device)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')
    grid_x = grid_x.reshape(-1)  # shape [N]
    grid_y = grid_y.reshape(-1)  # shape [N]
    N = grid_x.shape[0]

    # 6) For each grid position, sample a random patch size (ensuring the patch fits).
    patch_sizes = torch.randint(low=min_patch_size, high=max_patch_size + 1, size=(N,), device=device)

    # 7) Build ROIs tensor in the format [batch_idx, x1, y1, x2, y2].
    rois = torch.zeros((N, 5), device=device, dtype=torch.float32)
    rois[:, 0] = 0  # batch index is 0 for all patches
    rois[:, 1] = grid_x.float()
    rois[:, 2] = grid_y.float()
    rois[:, 3] = (grid_x + patch_sizes).float()
    rois[:, 4] = (grid_y + patch_sizes).float()

    # 8) Use ROI align to compute the average deltaE and valid_mask per patch.
    roi_deltaE = ops.roi_align(deltaE, rois, output_size=(1, 1), spatial_scale=1.0, sampling_ratio=-1, aligned=True)
    roi_valid = ops.roi_align(valid_mask, rois, output_size=(1, 1), spatial_scale=1.0, sampling_ratio=-1, aligned=True)
    roi_deltaE = roi_deltaE.view(-1)
    roi_valid = roi_valid.view(-1)

    # 9) Recover the total deltaE and total valid count per patch from the ROI averages.
    patch_areas = patch_sizes.float() * patch_sizes.float()  # area for each patch
    sum_deltaE = roi_deltaE * patch_areas
    sum_valid = roi_valid * patch_areas
    patch_avgs = sum_deltaE / (sum_valid + eps)

    # 10) Aggregate the patch scores.
    if aggregator == "max":
        score = patch_avgs.max()
    elif aggregator == "mean":
        score = patch_avgs.mean()
    elif aggregator == "pnorm":
        score = (patch_avgs ** p_value).mean().clamp_min(eps) ** (1.0 / p_value)
    else:
        raise ValueError(f"Unknown aggregator={aggregator}")

    # 11) Normalize the score by ~441.7 (maximum possible deltaE).
    max_dist = torch.tensor([441.7], device=device, dtype=imageA.dtype)
    norm_score = score / max_dist.clamp_min(eps)

    return norm_score


def detect_color_exception_amp(
    regionA: torch.Tensor,  # [1,4,H,W], BGRA in [0,1]
    regionB: torch.Tensor,  # [1,4,H,W], BGRA in [0,1]
    unusual_color_threshold: float = 15.0,   # threshold in LAB distance
    min_cluster_size: int = 40,            # minimum # of pixels to call it "prominent"
    mismatch_color_threshold: float = 20.0, # if regionB's color is more than this from regionA's unusual color => mismatch
    mismatch_amp: float = 10.0               # multiplier if mismatch occurs
) -> float:
    """
    1) Find a "prominent unusual color region" in regionA's shared area:
       - Convert regionA to LAB, compute the mean color of the shared area.
       - Find the pixel whose color is farthest from that mean => call it 'unusual_color'.
       - Gather all shared pixels whose color is within 'unusual_color_threshold' of 'unusual_color'.
       - If that cluster is >= min_cluster_size => "color exception" found.
    2) Compare that same region in regionB:
       - Compute average LAB color in regionB for those same pixel locations.
       - If the distance to 'unusual_color' > mismatch_color_threshold => mismatch => return mismatch_amp.
    3) Otherwise => return 1.0
    """
    device = regionA.device
    dtype = regionA.dtype

    # 1) Identify shared alpha in regionA (and regionB). We'll focus on regionA first.
    alphaA = (regionA[:,3:4,:,:] > 0)
    alphaB = (regionB[:,3:4,:,:] > 0)
    shared = alphaA & alphaB  # shape [1,1,H,W]
    if not shared.any():
        # No shared region => no mismatch
        return 1.0
    
    # Convert BGRA->LAB
    # We'll define a small helper for GPU-based BGR->LAB
    def bgr_to_lab(bgra: torch.Tensor):
        bgr = bgra[:, 0:3, :, :]
        rgb = torch.flip(bgr, dims=[1])  # BGR->RGB
        return kornia.color.rgb_to_lab(rgb)

    labA = bgr_to_lab(regionA)  # shape [1,3,H,W]
    labB = bgr_to_lab(regionB)  # shape [1,3,H,W]

    # We'll flatten the shared region for easy analysis
    # shape => [num_pixels, 3]
    shared_mask = shared.view(-1)  # [H*W], bool
    coords = torch.where(shared_mask)[0]  # indices in flattened space
    if coords.numel() < min_cluster_size:
        # Not enough shared pixels to matter
        return 1.0

    # Extract the LAB values for regionA's shared area
    labA_3HW = labA.view(3, -1).permute(1,0)  # => [H*W, 3]
    shared_labA = labA_3HW[shared_mask]       # => [num_shared, 3]

    # 2) Find the mean color of regionA's shared area
    mean_color = shared_labA.mean(dim=0, keepdim=True)  # shape [1,3]

    # 3) Find the pixel in shared_labA that is farthest from mean_color
    # => we compute distance from mean_color for each pixel
    diff = shared_labA - mean_color
    dist = torch.sqrt((diff * diff).sum(dim=1))  # shape [num_shared]
    # find the index of the max distance
    max_idx = torch.argmax(dist)
    if dist[max_idx] < 1.0:
        # The entire region is basically uniform => no "unusual" color
        return 1.0

    unusual_color = shared_labA[max_idx]  # shape [3]

    # 4) Gather all pixels whose distance from unusual_color < unusual_color_threshold
    diff2 = shared_labA - unusual_color
    dist2 = torch.sqrt((diff2 * diff2).sum(dim=1))  # shape [num_shared]
    cluster_mask = (dist2 < unusual_color_threshold)  # shape [num_shared]
    cluster_indices = torch.where(cluster_mask)[0]    # shape [M], M <= num_shared

    # If the cluster is too small => no color exception
    if cluster_indices.numel() < min_cluster_size:
        return 1.0

    # We do have a "prominent color exception" => let's see if regionB matches it

    # 5) Identify the same pixel locations in regionB
    # The shape [num_shared] in shared_labA corresponds exactly to 'coords' in the flattened image
    # cluster_indices are indices in the sub-array => we can map them back to 'coords'
    cluster_coords = coords[cluster_indices]  # these are the absolute flattened indices in [H*W]

    # Extract regionB's LAB for those same cluster pixels
    labB_3HW = labB.view(3, -1).permute(1,0)  # => [H*W, 3]
    cluster_labB = labB_3HW[cluster_coords]   # => [M, 3]

    # 6) Compare the average color of regionB's cluster to the "unusual_color"
    mean_clusterB = cluster_labB.mean(dim=0, keepdim=True)  # shape [1,3]
    color_diff = mean_clusterB - unusual_color.unsqueeze(0) # => shape [1,3]
    color_dist = torch.sqrt((color_diff * color_diff).sum(dim=1))  # shape [1]
    if color_dist.item() > mismatch_color_threshold:
        # Mismatch => return amplifier
        return mismatch_amp
    
    # If we get here => no mismatch => amplifier=1.0
    return 1.0


def compute_candidate_transform_translation(
    target_img: torch.Tensor,
    src_img: torch.Tensor,
    target_edge: dict,
    src_edge: dict,
    gap: float = 10.0
):
    """
    Computes a candidate transform (rotation [deg], tx, ty) so that:
      1) The source edge is reversed relative to the target edge:
         angle = angle_target + pi - angle_src,
      2) The midpoint of the source edge is placed side-by-side with the midpoint
         of the target edge, offset by 'gap' outward from the target edge.
    
    Steps:
      - angle_target = atan2(d_edge), angle_src = atan2(s_edge).
      - rotation = (angle_target + pi) - angle_src (so they 'face' each other).
      - normal_d => outward normal from the target edge => used to offset by 'gap'.
      - Ms => midpoint of source edge, Md => midpoint of target edge.
      - Rotate Ms about the alpha‐centroid of src by rotation, then offset it so that
        Ms_rot + T = Md_offset, where Md_offset = Md + gap * normal_d.
    """
    # 1) Edge midpoints
    Md = (target_edge['p1'] + target_edge['p2']) / 2.0
    Ms = (src_edge['p1'] + src_edge['p2']) / 2.0

    # 2) Edge orientations
    vd = target_edge['p2'] - target_edge['p1']  # target edge vector
    vs = src_edge['p2'] - src_edge['p1']    # source edge vector
    angle_d = math.atan2(vd[1], vd[0])
    angle_s = math.atan2(vs[1], vs[0])

    # "Face each other" => rotation = (angle_d + pi) - angle_s
    theta_candidate = (angle_d + math.pi) - angle_s
    rotation_deg = math.degrees(theta_candidate)

    # 3) Outward normal for the target edge => rotate 'vd' by +90°, pick sign so it points outward
    #    (for "gap" offset).
    #    We'll do something similar to your old "n_candidate1" logic:
    n_candidate1 = np.array([-vd[1], vd[0]], dtype=np.float64)
    n_candidate2 = -n_candidate1
    # We'll pick whichever points outward from the alpha-based centroid of target.
    dcx, dcy = find_image_centroid_on_gpu(target_img)
    alpha_target_centroid = np.array([dcx.item(), dcy.item()])
    # If dot((Md - alpha_target_centroid), n_candidate1) >= 0 => use n_candidate1 else n_candidate2
    if np.dot(Md - alpha_target_centroid, n_candidate1) >= 0:
        normal_d = n_candidate1
    else:
        normal_d = n_candidate2
    norm_len = np.linalg.norm(normal_d)
    if norm_len < 1e-8:
        normal_d = np.array([0.0, 0.0], dtype=np.float64)
    else:
        normal_d /= norm_len

    # 4) Offset the target midpoint by 'gap' => Md_offset
    Md_offset = Md + gap * normal_d

    # 5) Rotate Ms about the alpha‐centroid of src => Ms_rot
    scx, scy = find_image_centroid_on_gpu(src_img)
    R_src = np.array([scx.item(), scy.item()])
    Ms_rel = Ms - R_src
    cosT = math.cos(theta_candidate)
    sinT = math.sin(theta_candidate)
    Ms_rot_rel = np.array([cosT * Ms_rel[0] - sinT * Ms_rel[1],
                           sinT * Ms_rel[0] + cosT * Ms_rel[1]])
    Ms_rot = R_src + Ms_rot_rel

    # 6) Translation => T = Md_offset - Ms_rot
    T = Md_offset - Ms_rot
    tx = T[0]
    ty = T[1]

    return (rotation_deg, tx, ty)


def compute_pictorial_dissimilarity_on_gpu(
    target_img: torch.Tensor,
    src_img: torch.Tensor,
    target_extrap: torch.Tensor,
    src_extrap: torch.Tensor,
    angle_t: torch.Tensor,
    tx_t: torch.Tensor,
    ty_t: torch.Tensor,
    max_generations: int = 100,
    dynamic_penalty: bool = False,
    pure_dissimilarity: bool = False,
    lambda_joint: float = 50.0,
    lambda_overlap: float = 500.0,
    min_ext_overlap: int = 300,
    penalty_scale: float = 1.0
):
    """
    Warps using [circleAngle, rotationAngle, push], then measures patchwise LAB + coverage + overlap.
    """
    device = target_img.device
    eps = 1e-8

    target_band = extract_extrapolated_band_on_gpu(target_img, target_extrap)
    warped_src = warp_single_image(src_img, angle_t, tx_t, ty_t)
    warped_src_ex = warp_single_image(src_extrap, angle_t, tx_t, ty_t)
    src_band = extract_extrapolated_band_on_gpu(warped_src, warped_src_ex)

    region_target, region_src = arrange_canvases_on_gpu(target_band, src_band)
    lab_loss = lab_dissimilarity_score_random_patch_sizes_stride_on_gpu(region_target, region_src, min_patch_size=8, max_patch_size=32, stride=6, aggregator="pnorm", p_value=2.0)
    
    prominent_color_mismatch_amplifier = detect_color_exception_amp(region_target, region_src)
    lab_loss = lab_loss * prominent_color_mismatch_amplifier
    
    # if pure_dissimilarity:
    #     return lab_loss
    

    alpha_target_ex = (target_band[:, 3:4, :, :] > 0).float()
    alpha_src_ex  = (src_band[:, 3:4, :, :] > 0).float()
    coverage_count = (region_target[:, 3:4, :, :] > 0).sum().float()
    


    total_target_ex = alpha_target_ex.sum().float()
    total_src_ex  = alpha_src_ex.sum().float()
    max_extrap_possible = torch.minimum(total_target_ex, total_src_ex)
    min_ext_overlap = max_extrap_possible*0.05
    
    coverage_ratio = coverage_count/torch.clamp(max_extrap_possible, min=1.0)

    # measure real overlap of main content
    warped_src = warp_single_image(src_img, angle_t, tx_t, ty_t)
    B, C, Hd, Wd = target_img.shape
    _, _, Hs, Ws = warped_src.shape
    new_w = max(Wd, Ws)
    new_h = max(Hd, Hs)

    pad_target = torch.zeros((1,C,new_h,new_w), device=device, dtype=target_img.dtype)
    pad_src  = torch.zeros((1,C,new_h,new_w), device=device, dtype=src_img.dtype)
    pad_target[:, :, :Hd, :Wd] = target_img
    pad_src[:, :, :Hs, :Ws]  = warped_src

    alphaD = (pad_target[:, 3:4, :, :] > 0).float()
    alphaS = (pad_src[:, 3:4, :, :] > 0).float()
    actual_overlap = (alphaD*alphaS).sum().float()
    
    if pure_dissimilarity:
        return lab_loss + actual_overlap
    
    max_actual_possible = torch.minimum(alphaD.sum(), alphaS.sum())
    actual_overlap_ratio = actual_overlap/torch.clamp(max_actual_possible, min=eps)

    coverage_factor = 1.0/(coverage_ratio + eps)
    overlap_penalty = actual_overlap_ratio*lambda_overlap

    if pure_dissimilarity:
        return lab_loss*coverage_factor
    
    if dynamic_penalty:
        global CURRENT_GEN
        overlap_penalty = overlap_penalty*(CURRENT_GEN/max_generations)

    # coverage penalty if coverage is too small
    min_ext_overlap_count = max_extrap_possible*0.2
    extra_penalty = torch.zeros(1, device=device, dtype=target_img.dtype)
    if coverage_count < min_ext_overlap_count:
        deficit = min_ext_overlap_count - coverage_count
        extra_penalty = deficit*penalty_scale



    loss = lambda_joint*(lab_loss*coverage_factor) + overlap_penalty + extra_penalty
    return loss


def extract_potential_alignments(
    target_img: Tensor,
    src_img: Tensor,
    min_edge_length: float = 10.0,
    min_length_ratio: float = 0.7,
    angle_threshold_deg: float = 10.0,
    epsilon_ratio: float = 0.005,
    smoothing_kernel_size: int = 3,
    gap: float = 10.0,
    pad_by: int = 200,
    refine: bool = False,                # ← new, default False
    device: Optional[torch.device] = None
) -> List[Dict[str, float]]:
    """
    Compute all candidate alignment transforms between two fragments based on their polygonal edges.

    Args:
        target_img: [1,4,H,W] BGRA tensor for the target fragment.
        src_img: [1,4,H,W] BGRA tensor for the source fragment.
        min_edge_length: minimum length to consider an edge.
        min_length_ratio: allowed ratio between edge lengths to match.
        angle_threshold_deg: collinearity threshold for edge augmentation.
        epsilon_ratio: tolerance for polygon simplification.
        smoothing_kernel_size: kernel for contour smoothing.
        gap: offset distance along edge normal to separate fragments slightly.
        pad_by: padding applied before tensor conversion if needed.
        device: torch device.

    Returns:
        List of dicts, each with keys 'rotation', 'translation_x', 'translation_y',
        'target_edge', and 'src_edge'.
    """
    # Ensure tensors on correct device
    if device is None:
        device = target_img.device

    # Extract polygons
    tgt_poly = extract_polygon_from_alpha(
        target_img,
        epsilon_ratio=epsilon_ratio,
        smoothing_kernel_size=smoothing_kernel_size,
        angle_threshold_deg=angle_threshold_deg
    )
    src_poly = extract_polygon_from_alpha(
        src_img,
        epsilon_ratio=epsilon_ratio,
        smoothing_kernel_size=smoothing_kernel_size,
        angle_threshold_deg=angle_threshold_deg
    )
    if tgt_poly is None or src_poly is None:
        return []

    # Build multiscale edges
    tgt_edges = get_augmented_edges_from_polygon(
        tgt_poly,
        min_edge_length=min_edge_length,
        angle_threshold_deg=angle_threshold_deg
    )
    src_edges = get_augmented_edges_from_polygon(
        src_poly,
        min_edge_length=min_edge_length,
        angle_threshold_deg=angle_threshold_deg
    )
    candidates: List[Dict[str, float]] = []
    # For each pair of edges, compute transform if length ratio ok
    for d_edge in tgt_edges:
        for s_edge in src_edges:
            Ld, Ls = d_edge['length'], s_edge['length']
            if min(Ld, Ls) / max(Ld, Ls) < min_length_ratio:
                continue
            # Compute geometric transform
            rot_deg, tx, ty = compute_candidate_transform_translation(
                target_img, src_img, d_edge, s_edge, gap=gap
            )
            candidates.append({
                'rotation': rot_deg,
                'translation_x': tx,
                'translation_y': ty,
                'target_edge': d_edge,
                'src_edge': s_edge
            })
    if refine:
        # for now we just swallow it (or log a warning):
        print("extract_potential_alignments: refine=True ignored (not implemented).")

    return candidates

def score_alignment(
    target_img: Tensor,
    src_img: Tensor,
    target_extrap: Tensor,
    src_extrap: Tensor,
    transform: Dict[str, float],
    **scoring_kwargs
) -> float:
    """
    Score a single alignment transform for pictorial compatibility.

    Args:
        target_img: [1,4,H,W] BGRA tensor for the target fragment.
        src_img: [1,4,H,W] BGRA tensor for the source fragment.
        target_extrap: extrapolated band tensor for target.
        src_extrap: extrapolated band tensor for source.
        transform: dict with keys 'rotation', 'translation_x', 'translation_y'.
        scoring_kwargs: passed to compute_pictorial_dissimilarity_on_gpu, e.g.
            pure_dissimilarity, dynamic_penalty, lambda_joint, etc.

    Returns:
        Scalar compatibility score (lower is better).
    """
    angle = torch.tensor([transform['rotation']], device=target_img.device)
    tx = torch.tensor([transform['translation_x']], device=target_img.device)
    ty = torch.tensor([transform['translation_y']], device=target_img.device)
    score_t: Tensor = compute_pictorial_dissimilarity_on_gpu(
        target_img, src_img, target_extrap, src_extrap,
        angle, tx, ty,
        **scoring_kwargs
    )
    return float(score_t.item())


def convert_transform_to_legacy_dict(angle_deg: float,
                                                 tx: float,
                                                 ty: float) -> dict:
    """
    Convert a transform from the new framework (rotate around alpha-based centroid
    by 'angle_deg', then translate by (tx, ty)) to a dictionary for the legacy
    apply_transformation() function, which does:
       1) paste the piece at (max(0,int(translation_x)), max(0,int(translation_y)))
       2) rotate the entire canvas about its bounding-box center by 'rotation'.

    Because these two frameworks pivot around different centers, the final result
    is generally not exactly the same. However, this function preserves the typical
    sign convention (i.e. legacy code often uses negative for CCW rotation) and
    reuses (tx, ty) as translation. If you want an exact match, you'll need a more
    elaborate geometry offset.
    """
    # Often, in the old code, a positive "angle_deg" in the new framework
    # corresponds to 'rotation' = -angle_deg in the old code.
    # The translation_x, translation_y remain the same, but
    # the final arrangement won't be identical unless the pivot is the same.
    legacy_dict = {
        'rotation':     -float(angle_deg),
        'translation_x': float(tx),
        'translation_y': float(ty)
    }
    return legacy_dict
