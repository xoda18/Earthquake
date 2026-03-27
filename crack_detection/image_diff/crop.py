"""
crop.py — Crop drone images to the region defined by AprilTags.

No diff analysis, no heatmaps — just detect the two diagonal AprilTags
and crop the image to that bounding box.  Cropped images are saved to
the output directory for downstream VLM analysis.

Usage:
    python crop.py
    python crop.py --input-dir /app/input --output-dir /app/output
    python crop.py --tag-family tag25h9 --any-tags
"""

import argparse
import glob
import os
import sys

import cv2
import requests
from pupil_apriltags import Detector

from pipeline.preprocessing import detect_tags, wall_scan_crop

ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "")
TAG_FAMILY = "tag25h9"
TAG_IDS = [0, 1]
INPUT_DIR = "input"
OUTPUT_DIR = "output"


def parse_args():
    p = argparse.ArgumentParser(description="Crop images to AprilTag region")
    p.add_argument("--input-dir", default=INPUT_DIR)
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--tag-family", default=TAG_FAMILY)
    p.add_argument("--tag-ids", type=int, nargs="+", default=TAG_IDS)
    p.add_argument("--any-tags", action="store_true",
                   help="Auto-detect all AprilTags instead of requiring specific IDs")
    return p.parse_args()


def find_images(input_dir):
    """Find all image files in the input directory."""
    extensions = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.tif", "*.tiff", "*.webp"]
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(input_dir, ext)))
    return sorted(images)


def crop_image(image_path, detector, tag_ids, any_tags):
    """Detect AprilTags and crop the image to the tag-defined region.

    Returns the cropped image or None if tags not found.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"  [ERROR] Cannot read {image_path}")
        return None

    requested_ids = None if any_tags else tag_ids
    try:
        tags = detect_tags(image, detector, requested_ids, os.path.basename(image_path))
    except SystemExit:
        print(f"  [WARN] Tags not found in {os.path.basename(image_path)}, skipping")
        return None

    if any_tags:
        if len(tags) < 2:
            print(f"  [WARN] Need at least 2 tags, found {len(tags)} in {os.path.basename(image_path)}")
            return None

    cropped, crop_rect, arrangement = wall_scan_crop(image, tags, first_block=True)
    print(f"  Cropped {os.path.basename(image_path)}: {cropped.shape[1]}x{cropped.shape[0]} ({arrangement})")
    return cropped


def main():
    args = parse_args()

    print("AprilTag Image Cropper")
    print(f"  Input:      {args.input_dir}")
    print(f"  Output:     {args.output_dir}")
    tag_info = "auto (any)" if args.any_tags else str(args.tag_ids)
    print(f"  Tag family: {args.tag_family}, IDs: {tag_info}")

    images = find_images(args.input_dir)
    if not images:
        print(f"[ERROR] No images found in {args.input_dir}")
        notify_orchestrator(0, "failed")
        return

    print(f"\nFound {len(images)} image(s)")

    os.makedirs(args.output_dir, exist_ok=True)
    detector = Detector(families=args.tag_family, quad_decimate=1.0)

    cropped_count = 0
    for image_path in images:
        cropped = crop_image(image_path, detector, args.tag_ids, args.any_tags)
        if cropped is None:
            continue

        basename = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(args.output_dir, f"{basename}_cropped.jpg")
        cv2.imwrite(out_path, cropped)
        print(f"  Saved: {out_path}")
        cropped_count += 1

    print(f"\nCropped {cropped_count}/{len(images)} image(s)")
    notify_orchestrator(cropped_count, "success" if cropped_count > 0 else "failed")


def notify_orchestrator(count, status):
    if not ORCHESTRATOR_URL:
        return
    try:
        requests.post(f"{ORCHESTRATOR_URL}/step/done", json={
            "step": "image_crop",
            "status": status,
            "detail": f"cropped={count}",
            "run_id": os.environ.get("RUN_ID"),
        }, timeout=5)
        print(f"[orchestrator] Notified: image_crop {status}")
    except Exception as e:
        print(f"[orchestrator] Failed to notify: {e}")


if __name__ == "__main__":
    main()
