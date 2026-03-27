#!/usr/bin/env python3
"""
Building Change Detection — AprilTag-aligned image differencing.

Compares before/after photos of a surface using AprilTag-based homography
alignment and illumination-robust change detection. Supports multiple diff
methods: gradient, local_norm, raw, ssim, edge_xor, lab, combined.
"""

import argparse
import os
import sys
from datetime import datetime

import requests

import cv2
import numpy as np
from pupil_apriltags import Detector

from pipeline.preprocessing import (
    detect_tags,
    compute_canonical_frame,
    normalize_to_canonical,
    histogram_match_l_channel,
    wall_scan_crop,
)
from pipeline.alignment import (
    compute_homography,
    refine_ecc,
    compute_valid_overlap,
    crop_to_overlap,
)
from pipeline.diff_methods import compute_diffs
from pipeline.postprocessing import build_heatmap_overlay, annotate, severity_label
from pipeline.io import find_pairs, save_result, save_debug_panel, backup_file

# ── Defaults ─────────────────────────────────────────────────────────────
SENSITIVITY = 0.30
BLUR_KSIZE = 9
TILE_SIZE = 32
MIN_AREA = 5000
EDGE_SUPPRESS_RADIUS = 7
TAG_FAMILY = "tag25h9"
TAG_IDS = [0, 1]
INPUT_DIR = "input"
OUTPUT_DIR = "output"
BACKUP_DIR = "output_backup"


# ── CLI ──────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Detect structural changes between aligned before/after images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", default=INPUT_DIR)
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--backup-dir", default=BACKUP_DIR)
    p.add_argument("--no-backup", action="store_true")
    p.add_argument("--sensitivity", type=float, default=SENSITIVITY)
    p.add_argument("--min-area", type=int, default=MIN_AREA)
    p.add_argument("--blur", type=int, default=BLUR_KSIZE)
    p.add_argument("--tile-size", type=int, default=TILE_SIZE)
    p.add_argument("--tag-family", default=TAG_FAMILY)
    p.add_argument("--tag-ids", type=int, nargs="+", default=TAG_IDS)
    p.add_argument(
        "--any-tags", action="store_true",
        help="Auto-detect all AprilTags instead of requiring specific IDs",
    )
    p.add_argument(
        "--method",
        choices=["gradient", "local_norm", "raw", "ssim",
                 "edge_xor", "lab", "combined", "all"],
        default="gradient",
    )
    p.add_argument("--no-ecc", action="store_true")
    p.add_argument("--ecc-iterations", type=int, default=200)
    p.add_argument("--ecc-epsilon", type=float, default=1e-6)
    p.add_argument("--alpha", type=float, default=0.4)
    p.add_argument("--no-debug", action="store_true")
    p.add_argument("--no-annotate", action="store_true")
    p.add_argument("--pair", type=str, default=None)
    p.add_argument("--homography-method", choices=["ransac", "lmeds", "rho"], default="ransac")
    p.add_argument("--ransac-thresh", type=float, default=5.0)
    p.add_argument("--edge-suppress", type=int, default=EDGE_SUPPRESS_RADIUS)

    # New flags
    p.add_argument(
        "--no-canonical", action="store_true",
        help="Skip canonical frame normalisation (legacy warp-after-only mode)",
    )
    p.add_argument(
        "--combine-mode", choices=["or", "and"], default="or",
        help="How to combine edge_xor + lab in combined mode",
    )
    p.add_argument("--edge-xor-dilate", type=int, default=5,
                   help="Edge XOR dilation radius (px)")
    p.add_argument("--canny-low", type=int, default=50)
    p.add_argument("--canny-high", type=int, default=150)
    p.add_argument(
        "--fill-mode", choices=["none", "convex", "flood"], default="none",
        help="How to expand detected change blobs: "
             "none = as-is, convex = convex hull of nearby blobs, "
             "flood = edge-bounded flood-fill into after image",
    )
    p.add_argument(
        "--min-solidity", type=float, default=0.0,
        help="Reject contours with solidity below this (0-1). "
             "Thin lines ~0.1-0.2, compact objects ~0.4+. 0 = no filter.",
    )

    # ── Wall-scan crop mode ───────────────────────────────────────────
    p.add_argument(
        "--wall-scan", action="store_true",
        help="Wall-scan crop mode: crop a single image to the AprilTag-defined "
             "block region and save to --output-resize-dir",
    )
    p.add_argument(
        "--image", type=str, default=None,
        help="Path to a single image (used with --wall-scan)",
    )
    p.add_argument(
        "--first-block", action="store_true",
        help="First block in wall sequence (no overlap trimming)",
    )
    p.add_argument(
        "--trim-edges", nargs="+",
        choices=["left", "right", "top", "bottom"],
        default=["left", "right"],
        help="Edges to trim for overlap removal on non-first blocks",
    )
    p.add_argument(
        "--output-resize-dir", default="output_resize",
        help="Output directory for wall-scan cropped images",
    )

    return p.parse_args()


# ── Wall-scan crop ────────────────────────────────────────────────────────
def process_wall_scan(args):
    """Detect two AprilTags, classify their arrangement, crop, and save."""
    if not args.image:
        print("[ERROR] --wall-scan requires --image <path>")
        sys.exit(1)

    print(f"\nWall-Scan Crop Mode")
    print(f"  Image:       {args.image}")
    print(f"  First block: {args.first_block}")
    if not args.first_block:
        print(f"  Trim edges:  {args.trim_edges}")

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        print(f"[ERROR] Cannot read {args.image}")
        sys.exit(1)
    print(f"  Image shape: {image.shape}")

    # Detect any AprilTags (auto-discovery)
    detector = Detector(families=args.tag_family, quad_decimate=1.0)
    tags = detect_tags(image, detector, None, "wall-scan image")
    del detector

    if len(tags) < 2:
        print(f"[ERROR] Need at least 2 AprilTags, found {len(tags)}")
        sys.exit(1)

    if len(tags) > 2:
        # Keep the pair whose centres are furthest apart (the diagonal pair)
        ids = list(tags.keys())
        best_pair = (ids[0], ids[1])
        max_dist = 0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                d = np.linalg.norm(
                    tags[ids[i]].mean(axis=0) - tags[ids[j]].mean(axis=0)
                )
                if d > max_dist:
                    max_dist = d
                    best_pair = (ids[i], ids[j])
        print(f"  Found {len(tags)} tags, using diagonal pair: {best_pair}")
        tags = {tid: tags[tid] for tid in best_pair}

    print(f"  Tags detected: {sorted(tags.keys())}")

    cropped, crop_rect, arrangement = wall_scan_crop(
        image, tags,
        first_block=args.first_block,
        trim_edges=args.trim_edges,
    )

    # Save cropped image
    os.makedirs(args.output_resize_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(args.image))[0]
    output_path = os.path.join(args.output_resize_dir, f"{basename}_cropped.png")
    cv2.imwrite(output_path, cropped)
    print(f"  Saved: {output_path}")

    return output_path


# ── Pipeline ─────────────────────────────────────────────────────────────
def process_pair(date_suffix, before_path, after_path, args):
    """Process a single before/after pair."""
    print(f"\n{'='*60}")
    print(f"Processing pair: {date_suffix}")
    print(f"  Before: {before_path}")
    print(f"  After:  {after_path}")

    # 1. Load images
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
    detector = Detector(families=args.tag_family, quad_decimate=1.0)
    requested_ids = None if args.any_tags else args.tag_ids
    tags_before = detect_tags(before_img, detector, requested_ids, "before")
    tags_after = detect_tags(after_img, detector, requested_ids, "after")

    if args.any_tags:
        common_ids = sorted(set(tags_before.keys()) & set(tags_after.keys()))
        if len(common_ids) < 2:
            print(f"[ERROR] Need at least 2 common tags, found {len(common_ids)}: {common_ids}")
            sys.exit(1)
        # Keep only the common tags
        tags_before = {tid: tags_before[tid] for tid in common_ids}
        tags_after = {tid: tags_after[tid] for tid in common_ids}
        args.tag_ids = common_ids

    print(f"  Tags detected — before: {sorted(tags_before.keys())}, after: {sorted(tags_after.keys())}")
    del detector

    # 3. Alignment
    if args.no_canonical:
        # Legacy mode: warp only the after image → before coordinate space
        H, reproj_err = compute_homography(
            tags_before, tags_after, args.tag_ids,
            method=args.homography_method, thresh=args.ransac_thresh,
        )
        if not args.no_ecc:
            H = refine_ecc(before_img, after_img, H,
                           max_iter=args.ecc_iterations, epsilon=args.ecc_epsilon)

        h, w = before_img.shape[:2]
        warped_after = cv2.warpPerspective(after_img, H, (w, h))

        # Valid mask for after only
        ones = np.ones((after_img.shape[0], after_img.shape[1]), dtype=np.uint8) * 255
        mask_after = cv2.warpPerspective(ones, H, (w, h))
        mask_before = np.ones((h, w), dtype=np.uint8) * 255

        overlap = compute_valid_overlap(mask_before, mask_after)
        crop_before, crop_after, crop_mask, crop_offset = crop_to_overlap(
            before_img, warped_after, overlap,
        )
        # Tags stay in before-image coordinate space
        canonical_tags = tags_before
    else:
        # Canonical frame: warp BOTH images to a neutral coordinate system
        print("  Computing canonical frame...")
        canonical_corners = compute_canonical_frame(
            tags_before, tags_after, args.tag_ids,
        )

        canon_before, mask_before, H_before, shifted_corners_b = normalize_to_canonical(
            before_img, tags_before, canonical_corners, args.tag_ids,
        )
        canon_after, mask_after, H_after, shifted_corners_a = normalize_to_canonical(
            after_img, tags_after, canonical_corners, args.tag_ids,
        )

        # Ensure both canonical images are the same size (take the intersection)
        h = min(canon_before.shape[0], canon_after.shape[0])
        w = min(canon_before.shape[1], canon_after.shape[1])
        canon_before = canon_before[:h, :w]
        canon_after = canon_after[:h, :w]
        mask_before = mask_before[:h, :w]
        mask_after = mask_after[:h, :w]

        print(f"  Canonical frame size: {w} x {h}")

        # Optional ECC refinement on the canonical images
        if not args.no_ecc:
            # Compute a residual homography between the two canonical images
            H_residual = np.eye(3, dtype=np.float64)
            H_residual = refine_ecc(
                canon_before, canon_after, H_residual,
                max_iter=args.ecc_iterations, epsilon=args.ecc_epsilon,
            )
            # Apply residual correction to after
            canon_after = cv2.warpPerspective(canon_after, H_residual, (w, h))
            ones = np.ones((h, w), dtype=np.uint8) * 255
            mask_after_refined = cv2.warpPerspective(mask_after, H_residual, (w, h))
            mask_after = cv2.bitwise_and(mask_after, mask_after_refined)

        overlap = compute_valid_overlap(mask_before, mask_after)
        crop_before, crop_after, crop_mask, crop_offset = crop_to_overlap(
            canon_before, canon_after, overlap,
        )
        # Use the shifted canonical tag positions
        canonical_tags = shifted_corners_b

    # 4. Histogram match L channel for illumination normalization
    crop_after = histogram_match_l_channel(crop_before, crop_after)

    # 5. Compute diffs
    diffs = compute_diffs(
        crop_before, crop_after, args,
        tags_before=canonical_tags, crop_offset=crop_offset,
        overlap_mask=crop_mask,
    )

    # 6. Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S")

    if args.method == "all":
        primary_diff = None
        method_overlays = {}
        for method_name, diff_map in diffs.items():
            print(f"\n  --- {method_name} ---")
            d, mask = save_result(
                method_name, diff_map, crop_before, crop_after, args,
                canonical_tags, crop_offset, date_suffix, timestamp,
            )
            if primary_diff is None:
                primary_diff = d
            overlay_img, _ = build_heatmap_overlay(
                crop_after, diff_map, args.sensitivity, args.alpha, args.min_area,
                min_solidity=args.min_solidity, fill_mode=args.fill_mode,
            )
            method_overlays[method_name] = overlay_img

        # Side-by-side comparison panel
        if len(method_overlays) > 1:
            h_panel = crop_after.shape[0]
            panels = []
            for mname, overlay_img in method_overlays.items():
                panel = overlay_img.copy()
                label = mname.upper()
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
                cv2.rectangle(panel, (0, 0), (tw + 20, th + 30), (0, 0, 0), -1)
                cv2.putText(panel, label, (10, th + 15), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (255, 255, 255), 3, cv2.LINE_AA)
                ratio = h_panel / panel.shape[0]
                panel = cv2.resize(panel, (int(panel.shape[1] * ratio), h_panel))
                panels.append(panel)
            comparison = np.hstack(panels)
            comp_path = os.path.join(
                args.output_dir, f"diff-{date_suffix}_{timestamp}-comparison.png"
            )
            cv2.imwrite(comp_path, comparison)
            print(f"\n  Saved comparison panel: {comp_path}")
    else:
        primary_diff = diffs[args.method]
        save_result(
            args.method, primary_diff, crop_before, crop_after, args,
            canonical_tags, crop_offset, date_suffix, timestamp,
        )

    # Summary
    changed_pct = (primary_diff > args.sensitivity).mean() * 100
    max_score = primary_diff.max()
    sev = severity_label(changed_pct)
    print(f"  Changed area: {changed_pct:.1f}% | Max score: {max_score:.3f} | Severity: {sev}")

    return changed_pct, max_score, sev


def main():
    args = parse_args()

    if args.wall_scan:
        process_wall_scan(args)
        return

    print("Building Change Detection")
    print(f"  Input:       {args.input_dir}")
    print(f"  Output:      {args.output_dir}")
    print(f"  Sensitivity: {args.sensitivity}")
    print(f"  Method:      {args.method}")
    print(f"  Canonical:   {'off' if args.no_canonical else 'on'}")
    tag_info = "auto (any)" if args.any_tags else str(args.tag_ids)
    print(f"  Tag family:  {args.tag_family}, IDs: {tag_info}")

    # Clear output dir
    if os.path.exists(args.output_dir):
        for f in os.listdir(args.output_dir):
            fp = os.path.join(args.output_dir, f)
            if os.path.isfile(fp):
                os.remove(fp)

    pairs = find_pairs(args.input_dir, date_filter=args.pair)
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

    notify_orchestrator(results)


def notify_orchestrator(results):
    """Notify orchestrator that image diff is complete."""
    url = os.environ.get("ORCHESTRATOR_URL", "")
    if not url:
        return
    severities = [sev for _, _, _, sev in results]
    worst = max(severities, key=lambda s: ["low", "moderate", "high", "critical"].index(s)
                if s in ["low", "moderate", "high", "critical"] else 0) if severities else "none"
    try:
        requests.post(f"{url}/step/done", json={
            "step": "image_diff",
            "status": "success",
            "detail": f"pairs={len(results)} worst_severity={worst}",
        }, timeout=5)
        print(f"[orchestrator] Notified: image_diff done")
    except Exception as e:
        print(f"[orchestrator] Failed to notify: {e}")


if __name__ == "__main__":
    main()
    # Force immediate exit — pupil_apriltags C library segfaults during shutdown
    import signal
    os.kill(os.getpid(), signal.SIGKILL)
