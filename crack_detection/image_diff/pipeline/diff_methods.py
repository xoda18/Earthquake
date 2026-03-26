"""Diff algorithms: gradient, local_norm, raw, ssim, edge_xor, lab, combined."""

import cv2
import numpy as np


# ── Existing methods (moved from detect.py) ──────────────────────────────

def diff_gradient(gray_before, gray_after, blur_ksize, patch_size=10):
    """Sobel gradient magnitude difference averaged over patches."""
    def mag(g):
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx ** 2 + gy ** 2)

    m_before = mag(gray_before)
    m_after = mag(gray_after)
    diff = np.abs(m_before - m_after)

    # Average over patch_size x patch_size blocks
    kernel = np.ones((patch_size, patch_size), dtype=np.float32) / (patch_size * patch_size)
    diff = cv2.filter2D(diff, -1, kernel)

    dmax = diff.max()
    if dmax > 0:
        diff /= dmax
    diff = cv2.GaussianBlur(diff, (blur_ksize, blur_ksize), 0)
    return diff


def diff_local_norm(gray_before, gray_after, tile_size, blur_ksize):
    """Tile-based local normalisation difference."""
    def normalise(g):
        h, w = g.shape
        out = np.zeros_like(g, dtype=np.float32)
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                tile = g[y:y + tile_size, x:x + tile_size]
                mean = tile.mean()
                std = tile.std() + 1.0
                out[y:y + tile_size, x:x + tile_size] = (tile - mean) / std
        return out

    n_before = normalise(gray_before)
    n_after = normalise(gray_after)
    diff = np.abs(n_before - n_after)
    dmax = diff.max()
    if dmax > 0:
        diff /= dmax
    diff = cv2.GaussianBlur(diff, (blur_ksize, blur_ksize), 0)
    return diff


def diff_raw(gray_before, gray_after, blur_ksize, patch_size=30):
    """Raw pixel difference averaged over patches."""
    diff = np.abs(gray_before - gray_after) / 255.0

    # Average over patch_size x patch_size blocks
    kernel = np.ones((patch_size, patch_size), dtype=np.float32) / (patch_size * patch_size)
    diff = cv2.filter2D(diff, -1, kernel)

    diff = cv2.GaussianBlur(diff, (blur_ksize, blur_ksize), 0)
    return diff


def diff_ssim(gray_before, gray_after, blur_ksize, win_size=11):
    """SSIM-based difference — compares local patches."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    sigma = win_size / 6.0
    k = win_size

    mu_b = cv2.GaussianBlur(gray_before, (k, k), sigma)
    mu_a = cv2.GaussianBlur(gray_after, (k, k), sigma)

    mu_b_sq = mu_b ** 2
    mu_a_sq = mu_a ** 2
    mu_ba = mu_b * mu_a

    sigma_b_sq = cv2.GaussianBlur(gray_before ** 2, (k, k), sigma) - mu_b_sq
    sigma_a_sq = cv2.GaussianBlur(gray_after ** 2, (k, k), sigma) - mu_a_sq
    sigma_ba = cv2.GaussianBlur(gray_before * gray_after, (k, k), sigma) - mu_ba

    sigma_b_sq = np.maximum(sigma_b_sq, 0)
    sigma_a_sq = np.maximum(sigma_a_sq, 0)

    ssim_map = ((2 * mu_ba + C1) * (2 * sigma_ba + C2)) / \
               ((mu_b_sq + mu_a_sq + C1) * (sigma_b_sq + sigma_a_sq + C2))

    diff = np.clip(1.0 - ssim_map, 0, 1).astype(np.float32)
    diff = cv2.GaussianBlur(diff, (blur_ksize, blur_ksize), 0)
    return diff


# ── New methods ───────────────────────────────────────────────────────────

def diff_edge_xor(gray_before, gray_after, dilate_radius=5,
                  canny_low=50, canny_high=150):
    """Edge XOR: detect objects that appeared or disappeared.

    Canny edges of both images are dilated (tolerance for sub-pixel alignment
    error), then XOR'd.  Unchanged objects have edges in both → cancel out.
    Only truly new/removed objects produce orphan edges.

    Returns a float32 diff map in [0, 1].
    """
    gb = gray_before.astype(np.uint8)
    ga = gray_after.astype(np.uint8)

    edges_before = cv2.Canny(gb, canny_low, canny_high)
    edges_after = cv2.Canny(ga, canny_low, canny_high)

    # Dilate edges — tolerance for alignment imperfection
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_radius * 2 + 1, dilate_radius * 2 + 1)
    )
    dilated_before = cv2.dilate(edges_before, kernel)
    dilated_after = cv2.dilate(edges_after, kernel)

    # Appeared: edges in after that are NOT near any edge in before
    appeared = cv2.bitwise_and(edges_after, cv2.bitwise_not(dilated_before))
    # Disappeared: edges in before that are NOT near any edge in after
    disappeared = cv2.bitwise_and(edges_before, cv2.bitwise_not(dilated_after))

    change = cv2.bitwise_or(appeared, disappeared)

    # Morphological close to merge nearby orphan edges into blobs
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    change = cv2.morphologyEx(change, cv2.MORPH_CLOSE, close_kernel)

    # Dilate to fill object regions around orphan edges
    fill_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    change = cv2.dilate(change, fill_kernel)

    # Convert to float and blur for smooth diff map
    diff = (change / 255.0).astype(np.float32)
    blur_k = 15
    if blur_k % 2 == 0:
        blur_k += 1
    diff = cv2.GaussianBlur(diff, (blur_k, blur_k), 0)

    return diff


def diff_lab(before_bgr, after_bgr, blur_ksize):
    """LAB chrominance difference — robust to illumination changes.

    Computes diff on A and B channels only (chrominance), ignoring L (lightness).
    Objects have distinct colors; shadows and lighting shifts mostly affect L.

    Returns a float32 diff map in [0, 1].
    """
    before_lab = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    after_lab = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Chrominance channels (A and B)
    dA = before_lab[:, :, 1] - after_lab[:, :, 1]
    dB = before_lab[:, :, 2] - after_lab[:, :, 2]

    diff = np.sqrt(dA ** 2 + dB ** 2)

    # Normalize
    dmax = diff.max()
    if dmax > 0:
        diff /= dmax

    diff = cv2.GaussianBlur(diff, (blur_ksize, blur_ksize), 0)
    return diff


def diff_combined(gray_before, gray_after, before_bgr, after_bgr, args):
    """Run edge_xor + lab and combine with OR or AND.

    Returns a float32 diff map in [0, 1].
    """
    d_xor = diff_edge_xor(
        gray_before, gray_after,
        dilate_radius=args.edge_xor_dilate,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
    )
    d_lab = diff_lab(before_bgr, after_bgr, args.blur)

    # Threshold both into binary masks
    thresh = args.sensitivity
    mask_xor = (d_xor > thresh).astype(np.float32)
    mask_lab = (d_lab > thresh).astype(np.float32)

    if args.combine_mode == "and":
        combined_mask = mask_xor * mask_lab  # intersection
    else:
        combined_mask = np.clip(mask_xor + mask_lab, 0, 1)  # union

    # Smooth the combined mask back into a diff-like map
    blur_k = args.blur
    combined = cv2.GaussianBlur(combined_mask, (blur_k, blur_k), 0)
    return combined


# ── Masks ─────────────────────────────────────────────────────────────────

def build_edge_suppression_mask(gray_before, radius):
    """Mask that suppresses diff near strong edges in the before image.

    Edges shift the most under imperfect alignment. By zeroing diff near
    existing edges, we keep only changes in flat/smooth regions.
    """
    if radius <= 0:
        return None

    edges = cv2.Canny(gray_before.astype(np.uint8), 50, 150)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    dilated = cv2.dilate(edges, kernel)

    suppress = 1.0 - (dilated / 255.0).astype(np.float32)
    return suppress


def build_tag_mask(tags_before, crop_offset, shape, padding=50):
    """Mask that excludes AprilTag regions from diff computation."""
    x0_crop, y0_crop = crop_offset[0], crop_offset[1]
    h, w = shape[:2]
    mask = np.ones((h, w), dtype=np.float32)

    for tid, corners in tags_before.items():
        pts = corners.astype(np.int32).copy()
        pts[:, 0] -= x0_crop
        pts[:, 1] -= y0_crop

        x_min = max(0, pts[:, 0].min() - padding)
        x_max = min(w, pts[:, 0].max() + padding)
        y_min = max(0, pts[:, 1].min() - padding)
        y_max = min(h, pts[:, 1].max() + padding)

        mask[y_min:y_max, x_min:x_max] = 0.0
        print(f"  Masked tag {tid}: [{x_min}:{x_max}, {y_min}:{y_max}] ({padding}px padding)")

    return mask


# ── Orchestrator ──────────────────────────────────────────────────────────

def compute_diffs(crop_before, crop_after, args, tags_before=None,
                  crop_offset=None, overlap_mask=None):
    """Compute all requested diff maps. Returns dict of name → diff_map."""
    gb = cv2.cvtColor(crop_before, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ga = cv2.cvtColor(crop_after, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Edge suppression mask (only for legacy pixel-level methods)
    use_edge_suppress = args.method in ("gradient", "local_norm", "raw", "ssim", "all")
    edge_mask = build_edge_suppression_mask(gb, args.edge_suppress) if use_edge_suppress else None

    # Tag mask
    tag_mask = None
    if tags_before is not None and crop_offset is not None:
        tag_mask = build_tag_mask(tags_before, crop_offset, crop_before.shape)

    # Overlap validity mask (from canonical frame alignment)
    validity_mask = None
    if overlap_mask is not None:
        validity_mask = (overlap_mask > 0).astype(np.float32)

    results = {}

    # Legacy methods
    if args.method in ("gradient", "all"):
        results["gradient"] = diff_gradient(gb, ga, args.blur)
    if args.method in ("local_norm", "all"):
        results["local_norm"] = diff_local_norm(gb, ga, args.tile_size, args.blur)
    if args.method in ("raw", "all"):
        results["raw"] = diff_raw(gb, ga, args.blur)
    if args.method in ("ssim", "all"):
        results["ssim"] = diff_ssim(gb, ga, args.blur)

    # New methods
    if args.method in ("edge_xor", "all"):
        results["edge_xor"] = diff_edge_xor(
            gb, ga,
            dilate_radius=args.edge_xor_dilate,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
        )
    if args.method in ("lab", "all"):
        results["lab"] = diff_lab(crop_before, crop_after, args.blur)
    if args.method == "combined":
        results["combined"] = diff_combined(gb, ga, crop_before, crop_after, args)

    # Apply edge suppression (legacy methods only)
    if edge_mask is not None:
        for name in results:
            if name in ("gradient", "local_norm", "raw", "ssim"):
                results[name] = results[name] * edge_mask
        print(f"  Edge suppression: radius={args.edge_suppress}px applied")

    # Apply tag mask to all diff maps
    if tag_mask is not None:
        for name in results:
            results[name] = results[name] * tag_mask

    # Apply overlap validity mask to all diff maps
    if validity_mask is not None:
        for name in results:
            results[name] = results[name] * validity_mask

    # Noise floor removal + normalization
    for name in results:
        d = results[name]
        valid_px = np.ones_like(d, dtype=bool)
        if tag_mask is not None:
            valid_px &= tag_mask > 0
        if edge_mask is not None and name in ("gradient", "local_norm", "raw", "ssim"):
            valid_px &= edge_mask > 0.5
        if validity_mask is not None:
            valid_px &= validity_mask > 0.5
        valid_vals = d[valid_px]

        if len(valid_vals) > 0:
            noise_floor = np.percentile(valid_vals, 95)
            signal_cap = np.percentile(valid_vals, 99.9)
            d = np.clip(d - noise_floor, 0, None)
            sig_range = signal_cap - noise_floor
            if sig_range > 1e-6:
                d = np.clip(d / sig_range, 0, 1).astype(np.float32)
            else:
                d = np.zeros_like(d)
            results[name] = d
            print(f"  {name}: noise_floor(p95)={noise_floor:.4f}, signal_cap(p99.9)={signal_cap:.4f}")

    return results
