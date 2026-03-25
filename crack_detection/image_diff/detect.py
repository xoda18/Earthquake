#!/usr/bin/env python3
"""
Building Change Detection — AprilTag-aligned image differencing.

Compares before/after photos of a surface using AprilTag-based homography
alignment and illumination-robust change detection (gradient, local norm,
raw diff). Outputs a heatmap overlay highlighting structural changes.
"""

import argparse
import glob
import os
import re
import sys
import shutil
from datetime import datetime

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pupil_apriltags import Detector

# ── Configurable defaults ────────────────────────────────────────────────
SENSITIVITY = 0.30
BLUR_KSIZE = 9
TILE_SIZE = 32
MIN_AREA = 5000       # minimum changed region area in px² (filters noise/small shifts)
EDGE_SUPPRESS_RADIUS = 7  # px radius to suppress diff near edges (0 = disabled)
TAG_FAMILY = "tag25h9"
TAG_IDS = [0, 1]
INPUT_EXTENSIONS = [".tif", ".tiff", ".jpg", ".jpeg", ".JPG", ".JPEG"]
INPUT_DIR = "input"
OUTPUT_DIR = "output"
BACKUP_DIR = "output_backup"


# ── CLI ──────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Detect structural changes between aligned before/after images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir", default=INPUT_DIR,
        help="Directory containing before-*/after-* image pairs",
    )
    p.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help="Directory to write results into (overwritten each run)",
    )
    p.add_argument(
        "--backup-dir", default=BACKUP_DIR,
        help="Backup directory — results are copied here and never overwritten",
    )
    p.add_argument(
        "--no-backup", action="store_true",
        help="Skip copying results to backup directory",
    )
    p.add_argument(
        "--sensitivity", type=float, default=SENSITIVITY,
        help="Change threshold 0-1 (lower = more sensitive)",
    )
    p.add_argument(
        "--min-area", type=int, default=MIN_AREA,
        help="Minimum changed region area in px² — smaller regions are filtered out",
    )
    p.add_argument(
        "--blur", type=int, default=BLUR_KSIZE,
        help="Gaussian blur kernel size (odd number)",
    )
    p.add_argument(
        "--tile-size", type=int, default=TILE_SIZE,
        help="Tile size in px for local-normalisation method",
    )
    p.add_argument(
        "--tag-family", default=TAG_FAMILY,
        help="AprilTag family string",
    )
    p.add_argument(
        "--tag-ids", type=int, nargs="+", default=TAG_IDS,
        help="Expected AprilTag IDs in scene",
    )
    p.add_argument(
        "--method", choices=["gradient", "local_norm", "raw", "ssim", "all"], default="gradient",
        help="Diff method for the primary output heatmap",
    )
    p.add_argument(
        "--no-ecc", action="store_true",
        help="Skip ECC sub-pixel alignment refinement after homography",
    )
    p.add_argument(
        "--ecc-iterations", type=int, default=200,
        help="Max iterations for ECC refinement",
    )
    p.add_argument(
        "--ecc-epsilon", type=float, default=1e-6,
        help="ECC convergence threshold",
    )
    p.add_argument(
        "--alpha", type=float, default=0.4,
        help="Heatmap blend alpha (0=transparent, 1=opaque)",
    )
    p.add_argument(
        "--no-debug", action="store_true",
        help="Skip generating the 3-panel debug image",
    )
    p.add_argument(
        "--no-annotate", action="store_true",
        help="Skip drawing legend, bounding boxes, and tag markers",
    )
    p.add_argument(
        "--pair", type=str, default=None,
        help="Process only this date suffix, e.g. 25-03-26 (default: all pairs)",
    )
    p.add_argument(
        "--homography-method", choices=["ransac", "lmeds", "rho"],
        default="ransac", help="OpenCV homography estimation method",
    )
    p.add_argument(
        "--ransac-thresh", type=float, default=5.0,
        help="RANSAC reprojection threshold in pixels",
    )
    p.add_argument(
        "--edge-suppress", type=int, default=EDGE_SUPPRESS_RADIUS,
        help="Suppress diff within N px of edges in the before image (0 = off). "
             "Reduces false detections from alignment shifts along wires, edges, etc.",
    )
    return p.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────────
def find_pairs(input_dir, extensions, date_filter=None):
    """Scan input_dir for before-*/after-* pairs grouped by date suffix."""
    before_files = {}
    after_files = {}
    pattern = re.compile(r"^(before|after)-(.+?)(\.[^.]+)$", re.IGNORECASE)

    for fname in os.listdir(input_dir):
        m = pattern.match(fname)
        if not m:
            continue
        kind, date_suffix, ext = m.group(1).lower(), m.group(2), m.group(3)
        if ext.lower() not in [e.lower() for e in extensions]:
            continue
        if date_filter and date_suffix != date_filter:
            continue
        full = os.path.join(input_dir, fname)
        if kind == "before":
            before_files[date_suffix] = full
        else:
            after_files[date_suffix] = full

    pairs = []
    all_dates = sorted(set(before_files.keys()) | set(after_files.keys()))
    for d in all_dates:
        b = before_files.get(d)
        a = after_files.get(d)
        if not b:
            print(f"[ERROR] Missing before image for date suffix '{d}'")
            sys.exit(1)
        if not a:
            print(f"[ERROR] Missing after image for date suffix '{d}'")
            sys.exit(1)
        pairs.append((d, b, a))

    if not pairs:
        print(f"[ERROR] No matching before/after pairs found in {input_dir}")
        sys.exit(1)

    return pairs


def detect_tags(image, detector, tag_ids, image_label):
    """Detect AprilTags and return dict {tag_id: corners(4x2)}."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray)

    found = {}
    for r in results:
        if r.tag_id in tag_ids:
            found[r.tag_id] = r.corners  # 4x2 numpy array

    for tid in tag_ids:
        if tid not in found:
            print(
                f"[ERROR] Tag ID {tid} not detected in {image_label}. "
                "Check lighting, occlusion, or print quality."
            )
            sys.exit(1)

    return found


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
        # Scale back to full resolution
        H_refined = S_inv @ H_refined_small.astype(np.float64) @ S
        print(f"  ECC refinement converged")
        return H_refined
    except cv2.error as e:
        print(f"  [WARNING] ECC refinement failed ({e}), using original homography")
        return H


def warp_and_crop(before_img, after_img, H):
    """Warp after image into before's coordinate space and crop to valid overlap."""
    h, w = before_img.shape[:2]
    warped_after = cv2.warpPerspective(after_img, H, (w, h))

    # Valid pixel mask
    mask = np.ones((after_img.shape[0], after_img.shape[1]), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(mask, H, (w, h))

    valid = warped_mask > 0
    ys, xs = np.where(valid)
    if len(ys) == 0:
        print("  [ERROR] No valid pixels after warping")
        sys.exit(1)

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    crop_area = (x1 - x0) * (y1 - y0)
    total_area = h * w
    ratio = crop_area / total_area
    if ratio < 0.10:
        print(f"  [WARNING] Valid crop region is only {ratio:.1%} of before image — likely bad warp")

    crop_before = before_img[y0:y1, x0:x1]
    crop_after = warped_after[y0:y1, x0:x1]

    print(f"  Crop dimensions: {x1 - x0} x {y1 - y0} (from {w} x {h})")
    return crop_before, crop_after, (x0, y0, x1, y1)


# ── Diff methods ─────────────────────────────────────────────────────────
def diff_gradient(gray_before, gray_after, blur_ksize):
    """Method A: Sobel gradient magnitude difference (light-robust)."""
    def mag(g):
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx ** 2 + gy ** 2)

    m_before = mag(gray_before)
    m_after = mag(gray_after)
    diff = np.abs(m_before - m_after)
    dmax = diff.max()
    if dmax > 0:
        diff /= dmax
    diff = cv2.GaussianBlur(diff, (blur_ksize, blur_ksize), 0)
    return diff


def diff_local_norm(gray_before, gray_after, tile_size, blur_ksize):
    """Method B: Tile-based local normalisation difference."""
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


def diff_raw(gray_before, gray_after, blur_ksize):
    """Method C: Raw pixel difference (baseline)."""
    diff = np.abs(gray_before - gray_after) / 255.0
    diff = cv2.GaussianBlur(diff, (blur_ksize, blur_ksize), 0)
    return diff


def diff_ssim(gray_before, gray_after, blur_ksize, win_size=11):
    """Method D: SSIM-based difference — compares local patches, robust to small misalignment."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    k = win_size

    mu_b = cv2.GaussianBlur(gray_before, (k, k), 1.5)
    mu_a = cv2.GaussianBlur(gray_after, (k, k), 1.5)

    mu_b_sq = mu_b ** 2
    mu_a_sq = mu_a ** 2
    mu_ba = mu_b * mu_a

    sigma_b_sq = cv2.GaussianBlur(gray_before ** 2, (k, k), 1.5) - mu_b_sq
    sigma_a_sq = cv2.GaussianBlur(gray_after ** 2, (k, k), 1.5) - mu_a_sq
    sigma_ba = cv2.GaussianBlur(gray_before * gray_after, (k, k), 1.5) - mu_ba

    ssim_map = ((2 * mu_ba + C1) * (2 * sigma_ba + C2)) / \
               ((mu_b_sq + mu_a_sq + C1) * (sigma_b_sq + sigma_a_sq + C2))

    # Convert similarity (1=identical) to difference (0=identical, 1=max change)
    diff = np.clip(1.0 - ssim_map, 0, 1).astype(np.float32)
    diff = cv2.GaussianBlur(diff, (blur_ksize, blur_ksize), 0)
    return diff


def build_edge_suppression_mask(gray_before, radius):
    """Build a mask that suppresses diff values near strong edges in the before image.

    Edges shift the most under imperfect alignment (parallax, sub-pixel error).
    By zeroing out diff near existing edges, we keep only changes in flat/smooth
    regions — which are real structural changes, not alignment artifacts.
    """
    if radius <= 0:
        return None

    edges = cv2.Canny(gray_before.astype(np.uint8), 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    dilated = cv2.dilate(edges, kernel)

    # 1.0 = keep, 0.0 = suppress (near edge)
    suppress = 1.0 - (dilated / 255.0).astype(np.float32)
    return suppress


def compute_diffs(crop_before, crop_after, args):
    """Compute all requested diff maps. Returns dict of name→diff_map."""
    gb = cv2.cvtColor(crop_before, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ga = cv2.cvtColor(crop_after, cv2.COLOR_BGR2GRAY).astype(np.float32)

    edge_mask = build_edge_suppression_mask(gb, args.edge_suppress)

    results = {}
    if args.method in ("gradient", "all"):
        results["gradient"] = diff_gradient(gb, ga, args.blur)
    if args.method in ("local_norm", "all"):
        results["local_norm"] = diff_local_norm(gb, ga, args.tile_size, args.blur)
    if args.method in ("raw", "all"):
        results["raw"] = diff_raw(gb, ga, args.blur)
    if args.method in ("ssim", "all"):
        results["ssim"] = diff_ssim(gb, ga, args.blur)

    # Apply edge suppression to all diff maps
    if edge_mask is not None:
        suppressed = sum((v * edge_mask < v).sum() for v in results.values())
        print(f"  Edge suppression: radius={args.edge_suppress}px, suppressed {suppressed} diff values near edges")
        for name in results:
            results[name] = results[name] * edge_mask

    return results


# ── Heatmap + annotation ────────────────────────────────────────────────
def build_heatmap_overlay(crop_after, diff_map, sensitivity, alpha, min_area):
    """Build yellow marker overlay highlighting whole changed objects."""
    output = crop_after.copy().astype(np.float64)
    raw_mask = (diff_map > sensitivity).astype(np.uint8)

    # Opening: erode then dilate — removes thin line artifacts (wires, edges)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    opened = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_open)

    # Closing: merge nearby pixels into blobs
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

    # Filter out small regions (noise, minor shifts) — keep only large blobs
    filtered = np.zeros_like(closed)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant = [c for c in contours if cv2.contourArea(c) >= min_area]
    if significant:
        cv2.drawContours(filtered, significant, -1, 1, thickness=cv2.FILLED)

    # Dilate kept regions slightly to cover full object extent
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    mask = cv2.dilate(filtered, kernel_dilate, iterations=1)

    # Solid yellow highlight (BGR: 0, 255, 255)
    yellow = np.full_like(output, (0, 255, 255), dtype=np.float64)

    mask_bool = mask > 0
    mask_3ch = np.stack([mask_bool] * 3, axis=-1)
    output = np.where(
        mask_3ch,
        (1.0 - alpha) * output + alpha * yellow,
        output,
    )

    n_regions = len(significant)
    print(f"  Regions found: {len(contours)} raw → {n_regions} after min_area={min_area}px² filter")
    return np.clip(output, 0, 255).astype(np.uint8), mask_bool


def annotate(output, change_mask, sensitivity, method_name, tags_before, crop_offset, annotate_enabled):
    """Draw legend, bounding boxes around all changed regions, and tag markers."""
    if not annotate_enabled:
        return output

    h, w = output.shape[:2]
    x0_crop, y0_crop = crop_offset[0], crop_offset[1]

    # Legend — white text on dark background for readability
    text = f"Change map: {method_name} | sensitivity={sensitivity:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(output, (5, h - th - 20), (tw + 15, h - 5), (0, 0, 0), -1)
    cv2.putText(output, text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Bounding boxes around ALL changed regions > 200px²
    binary = change_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            cv2.rectangle(output, (bx, by), (bx + bw, by + bh), (0, 0, 255), 3)

    # Tag locations (mapped to crop coordinates)
    for tid, corners in tags_before.items():
        center = corners.mean(axis=0).astype(int)
        cx = center[0] - x0_crop
        cy = center[1] - y0_crop
        if 0 <= cx < w and 0 <= cy < h:
            sz = 20
            cv2.rectangle(output, (cx - sz, cy - sz), (cx + sz, cy + sz), (0, 255, 0), 3)
            cv2.putText(output, f"T{tid}", (cx - sz, cy - sz - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    return output


def severity_label(pct):
    if pct < 1.0:
        return "None"
    elif pct < 8.0:
        return "Low"
    elif pct < 25.0:
        return "Moderate"
    else:
        return "High"


def save_debug_panel(crop_before, crop_after, change_mask, diff_map, path):
    """Save a 3-panel debug image: before | after | change mask (yellow on dark)."""
    # Third panel: dim version of after image with yellow overlay on changes
    dim_after = (crop_after.astype(np.float64) * 0.3).astype(np.uint8)
    yellow = np.full_like(dim_after, (0, 230, 230), dtype=np.uint8)
    mask_3ch = np.stack([change_mask] * 3, axis=-1)
    change_panel = np.where(mask_3ch, yellow, dim_after)

    # Resize all to same height
    h = min(crop_before.shape[0], crop_after.shape[0], change_panel.shape[0])
    panels = []
    labels = ["BEFORE", "AFTER", "CHANGES"]
    for img, label in zip([crop_before, crop_after, change_panel], labels):
        ratio = h / img.shape[0]
        resized = cv2.resize(img, (int(img.shape[1] * ratio), h))
        # Add label at top
        cv2.putText(resized, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
        panels.append(resized)
    debug = np.hstack(panels)
    cv2.imwrite(path, debug)


# ── Main pipeline ────────────────────────────────────────────────────────
def process_pair(date_suffix, before_path, after_path, args):
    """Process a single before/after pair."""
    print(f"\n{'='*60}")
    print(f"Processing pair: {date_suffix}")
    print(f"  Before: {before_path}")
    print(f"  After:  {after_path}")

    # 1. Load
    before_img = cv2.imread(before_path, cv2.IMREAD_COLOR)
    after_img = cv2.imread(after_path, cv2.IMREAD_COLOR)
    if before_img is None:
        print(f"[ERROR] Cannot read {before_path}")
        sys.exit(1)
    if after_img is None:
        print(f"[ERROR] Cannot read {after_path}")
        sys.exit(1)
    print(f"  Before shape: {before_img.shape}")
    print(f"  After shape:  {after_img.shape}")

    # 2. Detect AprilTags
    detector = Detector(families=args.tag_family)
    tags_before = detect_tags(before_img, detector, args.tag_ids, "before")
    tags_after = detect_tags(after_img, detector, args.tag_ids, "after")
    print(f"  Tags detected — before: {sorted(tags_before.keys())}, after: {sorted(tags_after.keys())}")
    del detector  # free C library early to avoid segfault at shutdown

    # 3. Homography
    H, reproj_err = compute_homography(
        tags_before, tags_after, args.tag_ids,
        method=args.homography_method, thresh=args.ransac_thresh,
    )

    # 3b. ECC sub-pixel refinement
    if not args.no_ecc:
        H = refine_ecc(before_img, after_img, H,
                        max_iter=args.ecc_iterations, epsilon=args.ecc_epsilon)

    # 4. Warp + 5. Crop
    crop_before, crop_after, crop_offset = warp_and_crop(before_img, after_img, H)

    # 6. Compute diffs
    diffs = compute_diffs(crop_before, crop_after, args)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S")

    def save_result(method_name, diff_map, suffix=""):
        """Generate heatmap, annotate, and save for one method."""
        overlay, mask = build_heatmap_overlay(crop_after, diff_map, args.sensitivity, args.alpha, args.min_area)
        overlay = annotate(
            overlay, mask, args.sensitivity, method_name,
            tags_before, crop_offset, not args.no_annotate,
        )

        tag = f"-{suffix}" if suffix else ""
        out_path = os.path.join(args.output_dir, f"diff-{date_suffix}_{timestamp}{tag}.png")
        cv2.imwrite(out_path, overlay)
        print(f"  Saved {out_path}")

        if not args.no_debug:
            debug_path = os.path.join(args.output_dir, f"diff_debug-{date_suffix}_{timestamp}{tag}.png")
            save_debug_panel(crop_before, crop_after, mask, diff_map, debug_path)
            print(f"  Saved {debug_path}")

        # Backup
        if not args.no_backup:
            os.makedirs(args.backup_dir, exist_ok=True)
            for src in [out_path] + ([debug_path] if not args.no_debug else []):
                dst = os.path.join(args.backup_dir, os.path.basename(src))
                if os.path.exists(dst):
                    base, ext = os.path.splitext(dst)
                    i = 1
                    while os.path.exists(f"{base}_{i}{ext}"):
                        i += 1
                    dst = f"{base}_{i}{ext}"
                shutil.copy2(src, dst)
                print(f"  Backup {dst}")

        return diff_map, mask

    # 7–9. Generate outputs — one per method when "all", otherwise just the selected one
    if args.method == "all":
        primary_diff = None
        for method_name, diff_map in diffs.items():
            print(f"\n  --- {method_name} ---")
            d, _ = save_result(method_name, diff_map, suffix=method_name)
            if primary_diff is None:
                primary_diff = d  # first method for summary stats
    else:
        primary_diff = diffs[args.method]
        save_result(args.method, primary_diff)

    # Summary
    changed_pct = (primary_diff > args.sensitivity).mean() * 100
    max_score = primary_diff.max()
    sev = severity_label(changed_pct)
    print(f"  Changed area: {changed_pct:.1f}% | Max score: {max_score:.3f} | Severity: {sev}")

    return changed_pct, max_score, sev


def main():
    args = parse_args()
    print("Building Change Detection")
    print(f"  Input:       {args.input_dir}")
    print(f"  Output:      {args.output_dir}")
    print(f"  Sensitivity: {args.sensitivity}")
    print(f"  Method:      {args.method}")
    print(f"  Tag family:  {args.tag_family}, IDs: {args.tag_ids}")

    # Clear output dir so it only contains results from this run
    if os.path.exists(args.output_dir):
        for f in os.listdir(args.output_dir):
            fp = os.path.join(args.output_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)

    pairs = find_pairs(args.input_dir, INPUT_EXTENSIONS, args.pair)
    print(f"\nFound {len(pairs)} pair(s) to process")

    results = []
    for date_suffix, before_path, after_path in pairs:
        r = process_pair(date_suffix, before_path, after_path, args)
        results.append((date_suffix, *r))

    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'Date':<15} {'Changed%':>10} {'Max Score':>10} {'Severity':>10}")
    print("-" * 50)
    for date_suffix, pct, mx, sev in results:
        print(f"{date_suffix:<15} {pct:>9.1f}% {mx:>10.3f} {sev:>10}")
    print()


if __name__ == "__main__":
    main()
    # Force immediate process exit — pupil_apriltags C library segfaults
    # during Python's normal shutdown/garbage collection sequence.
    import signal
    os.kill(os.getpid(), signal.SIGKILL)
