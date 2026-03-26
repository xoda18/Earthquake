"""File I/O: pair discovery, result saving, backup, debug panels."""

import os
import re
import shutil
import sys
from datetime import datetime

import cv2
import numpy as np

# Supported input file extensions
INPUT_EXTENSIONS = [".tif", ".tiff", ".jpg", ".jpeg", ".JPG", ".JPEG", ".webp", ".png"]


def find_pairs(input_dir, extensions=None, date_filter=None):
    """Scan input_dir for before-*/after-* pairs grouped by date suffix."""
    if extensions is None:
        extensions = INPUT_EXTENSIONS

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


def save_debug_panel(crop_before, crop_after, change_mask, diff_map, path):
    """Save a 3-panel debug image: before | after | change mask (yellow on dark)."""
    dim_after = (crop_after.astype(np.float64) * 0.3).astype(np.uint8)
    yellow = np.full_like(dim_after, (0, 230, 230), dtype=np.uint8)
    mask_3ch = np.stack([change_mask] * 3, axis=-1)
    change_panel = np.where(mask_3ch, yellow, dim_after)

    h = min(crop_before.shape[0], crop_after.shape[0], change_panel.shape[0])
    panels = []
    labels = ["BEFORE", "AFTER", "CHANGES"]
    for img, label in zip([crop_before, crop_after, change_panel], labels):
        ratio = h / img.shape[0]
        resized = cv2.resize(img, (int(img.shape[1] * ratio), h))
        cv2.putText(resized, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 255, 255), 3, cv2.LINE_AA)
        panels.append(resized)
    debug = np.hstack(panels)
    cv2.imwrite(path, debug)


def backup_file(src, backup_dir):
    """Copy src to backup_dir, appending a counter if it already exists."""
    os.makedirs(backup_dir, exist_ok=True)
    dst = os.path.join(backup_dir, os.path.basename(src))
    if os.path.exists(dst):
        base, ext = os.path.splitext(dst)
        i = 1
        while os.path.exists(f"{base}_{i}{ext}"):
            i += 1
        dst = f"{base}_{i}{ext}"
    shutil.copy2(src, dst)
    print(f"  Backup {dst}")


def save_cropped_changes(crop_after, change_mask, output_dir, date_suffix,
                         timestamp, method_suffix, padding_ratio=0.5):
    """Extract individual change regions as separate cropped images.

    Each detected change blob is saved as its own image, padded by
    `padding_ratio` (0.5 = 50%) of the blob's bounding-box size on every side.
    """
    resized_dir = os.path.join(os.path.dirname(output_dir.rstrip("/")),
                               "output_resized")
    os.makedirs(resized_dir, exist_ok=True)

    binary = change_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    h, w = crop_after.shape[:2]
    count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) < 200:
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)
        pad_w = int(bw * padding_ratio)
        pad_h = int(bh * padding_ratio)

        x0 = max(0, bx - pad_w)
        y0 = max(0, by - pad_h)
        x1 = min(w, bx + bw + pad_w)
        y1 = min(h, by + bh + pad_h)

        crop = crop_after[y0:y1, x0:x1]
        count += 1
        out_path = os.path.join(
            resized_dir,
            f"change-{date_suffix}_{timestamp}{method_suffix}_{count}.png",
        )
        cv2.imwrite(out_path, crop)
        print(f"  Saved crop {out_path}")

    if count:
        print(f"  Extracted {count} change region(s) to {resized_dir}")


def save_result(method_name, diff_map, crop_before, crop_after, args,
                tags_before, crop_offset, date_suffix, timestamp):
    """Generate heatmap, annotate, save, and optionally backup.

    Returns (diff_map, change_mask).
    """
    from .postprocessing import build_heatmap_overlay, annotate

    overlay, mask = build_heatmap_overlay(
        crop_after, diff_map, args.sensitivity, args.alpha, args.min_area,
        min_solidity=args.min_solidity, fill_mode=args.fill_mode,
    )
    overlay = annotate(
        overlay, mask, args.sensitivity, method_name,
        tags_before, crop_offset, not args.no_annotate,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    suffix = f"-{method_name}" if args.method == "all" else ""
    out_path = os.path.join(
        args.output_dir, f"diff-{date_suffix}_{timestamp}{suffix}.png"
    )
    cv2.imwrite(out_path, overlay)
    print(f"  Saved {out_path}")

    debug_path = None
    if not args.no_debug:
        debug_path = os.path.join(
            args.output_dir, f"diff_debug-{date_suffix}_{timestamp}{suffix}.png"
        )
        save_debug_panel(crop_before, crop_after, mask, diff_map, debug_path)
        print(f"  Saved {debug_path}")

    if not args.no_backup:
        for src in [out_path] + ([debug_path] if debug_path else []):
            backup_file(src, args.backup_dir)

    # Extract individual change crops to output_resized/
    save_cropped_changes(crop_after, mask, args.output_dir, date_suffix,
                         timestamp, suffix)

    return diff_map, mask
