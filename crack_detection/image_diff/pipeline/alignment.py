"""Homography computation, ECC refinement, and overlap cropping."""

import sys

import cv2
import numpy as np


def compute_homography(tags_before, tags_after, tag_ids, method="ransac", thresh=5.0):
    """Compute homography mapping after→before using matched tag corners."""
    pts_before = []
    pts_after = []
    for tid in sorted(tag_ids):
        pts_before.append(tags_before[tid])
        pts_after.append(tags_after[tid])

    pts_before = np.vstack(pts_before).astype(np.float64)
    pts_after = np.vstack(pts_after).astype(np.float64)

    methods = {
        "ransac": cv2.RANSAC,
        "lmeds": cv2.LMEDS,
        "rho": cv2.RHO,
    }
    cv_method = methods[method]

    H, mask = cv2.findHomography(pts_after, pts_before, cv_method, thresh)

    # Reprojection error
    pts_after_h = np.hstack([pts_after, np.ones((len(pts_after), 1))])
    projected = (H @ pts_after_h.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    error = np.mean(np.linalg.norm(projected - pts_before, axis=1))
    print(f"  Homography reprojection error: {error:.3f} px")

    if error > 5.0:
        print("  [WARNING] Reprojection error > 5px — alignment may be poor")

    return H, error


def refine_ecc(before_img, after_img, H, max_iter=200, epsilon=1e-6):
    """Refine homography with ECC for sub-pixel alignment accuracy."""
    gray_b = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
    gray_a = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)

    # Downscale for speed — ECC on full-res is slow
    scale = 0.5
    small_b = cv2.resize(gray_b, None, fx=scale, fy=scale)
    small_a = cv2.resize(gray_a, None, fx=scale, fy=scale)

    # Scale the homography for the smaller images
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
    S_inv = np.array([[1/scale, 0, 0], [0, 1/scale, 0], [0, 0, 1]], dtype=np.float64)
    H_small = S @ H @ S_inv

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, epsilon)
    try:
        _, H_refined_small = cv2.findTransformECC(
            small_b, small_a, H_small.astype(np.float32),
            cv2.MOTION_HOMOGRAPHY, criteria,
        )
        H_refined = S_inv @ H_refined_small.astype(np.float64) @ S
        print(f"  ECC refinement converged")
        return H_refined
    except cv2.error as e:
        print(f"  [WARNING] ECC refinement failed ({e}), using original homography")
        return H


def compute_valid_overlap(mask_before, mask_after):
    """AND of two valid-pixel masks → overlap where both images have data."""
    return cv2.bitwise_and(mask_before, mask_after)


def crop_to_overlap(before, after, overlap_mask):
    """Crop both images and the overlap mask to the bounding box of valid pixels.

    Returns
    -------
    crop_before, crop_after : ndarray (H, W, 3)
    crop_mask : ndarray (H, W) uint8 — valid overlap within the crop
    crop_offset : (x0, y0, x1, y1)
    """
    valid = overlap_mask > 0
    ys, xs = np.where(valid)
    if len(ys) == 0:
        print("  [ERROR] No valid overlap pixels")
        sys.exit(1)

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    h, w = before.shape[:2]
    crop_area = (x1 - x0) * (y1 - y0)
    total_area = h * w
    ratio = crop_area / total_area if total_area > 0 else 0
    if ratio < 0.10:
        print(f"  [WARNING] Valid overlap is only {ratio:.1%} of image — likely bad warp")

    crop_before = before[y0:y1, x0:x1]
    crop_after = after[y0:y1, x0:x1]
    crop_mask = overlap_mask[y0:y1, x0:x1]

    print(f"  Crop dimensions: {x1 - x0} x {y1 - y0} (from {w} x {h})")
    return crop_before, crop_after, crop_mask, (x0, y0, x1, y1)
