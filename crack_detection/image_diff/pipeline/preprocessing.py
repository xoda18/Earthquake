"""AprilTag detection and canonical-frame normalization."""

import sys

import cv2
import numpy as np
from pupil_apriltags import Detector


def detect_tags(image, detector, tag_ids, image_label):
    """Detect AprilTags and return dict {tag_id: corners(4x2)}.

    If *tag_ids* is ``None``, every detected tag is kept (auto-discovery mode).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = detector.detect(gray)

    if tag_ids is None:
        # Auto mode: accept every tag the detector finds
        found = {r.tag_id: r.corners for r in results}
        if not found:
            print(f"[ERROR] No AprilTags detected in {image_label}.")
            sys.exit(1)
        return found

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


def _tag_center(corners):
    """Mean of the four corners → tag center (x, y)."""
    return corners.mean(axis=0)


def compute_canonical_frame(tags_before, tags_after, tag_ids):
    """Compute a canonical coordinate frame based on AprilTag positions.

    Uses the before-image tag corners as the canonical reference, preserving
    the natural orientation and scale.  Both images will be warped to match
    these positions so that interpolation artifacts are symmetric.

    Returns
    -------
    canonical_corners : dict  {tag_id: 4x2 ndarray}
        Target corner positions for each tag in the canonical frame
        (same as the before-image tag corners).
    output_size : (width, height)
        Canvas sized to fit the before image.
    """
    # Use the before-image tag corners directly — this preserves the natural
    # orientation and means the before image gets a near-identity warp.
    canonical_corners = {}
    for tid in sorted(tag_ids):
        canonical_corners[tid] = tags_before[tid].copy()

    return canonical_corners


def normalize_to_canonical(image, tags, canonical_corners, tag_ids):
    """Warp an image so its AprilTags land on the canonical positions.

    Returns
    -------
    warped : ndarray  (H, W, 3) BGR
    valid_mask : ndarray  (H, W) uint8, 255 where pixels are valid
    H_shifted : 3x3 homography used (includes canvas shift)
    shifted_canonical : dict  {tag_id: 4x2} canonical corners in shifted coords
    """
    sorted_ids = sorted(tag_ids)

    src_pts = []
    dst_pts = []
    for tid in sorted_ids:
        src_pts.append(tags[tid])
        dst_pts.append(canonical_corners[tid])

    src_pts = np.vstack(src_pts).astype(np.float64)
    dst_pts = np.vstack(dst_pts).astype(np.float64)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Compute bounding box of the warped image to size the canvas
    ih, iw = image.shape[:2]
    img_corners = np.array([
        [0, 0], [iw, 0], [iw, ih], [0, ih]
    ], dtype=np.float64)
    img_corners_h = np.hstack([img_corners, np.ones((4, 1))])
    warped_corners = (H @ img_corners_h.T).T
    warped_corners = warped_corners[:, :2] / warped_corners[:, 2:3]

    # Also include canonical tag positions so they're guaranteed to be in-frame
    all_canon = np.vstack(list(canonical_corners.values()))
    all_pts = np.vstack([warped_corners, all_canon])

    x_min = int(np.floor(all_pts[:, 0].min()))
    x_max = int(np.ceil(all_pts[:, 0].max()))
    y_min = int(np.floor(all_pts[:, 1].min()))
    y_max = int(np.ceil(all_pts[:, 1].max()))

    # Shift so everything fits in positive coordinates
    T_shift = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1],
    ], dtype=np.float64)

    final_w = x_max - x_min
    final_h = y_max - y_min
    H_shifted = T_shift @ H

    warped = cv2.warpPerspective(image, H_shifted, (final_w, final_h))

    # Valid pixel mask
    ones = np.ones((ih, iw), dtype=np.uint8) * 255
    valid_mask = cv2.warpPerspective(ones, H_shifted, (final_w, final_h))

    # Shift canonical corners to match the canvas offset
    shifted_canonical = {}
    for tid, corners in canonical_corners.items():
        shifted_canonical[tid] = corners - np.array([x_min, y_min])

    return warped, valid_mask, H_shifted, shifted_canonical


def histogram_match_l_channel(before_bgr, after_bgr):
    """Match the L-channel histogram of `after` to `before` (LAB space).

    Returns the corrected after image in BGR.
    """
    before_lab = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2LAB)
    after_lab = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2LAB)

    # Compute CDFs
    def cdf(channel):
        hist, _ = np.histogram(channel.flatten(), bins=256, range=(0, 256))
        c = hist.cumsum()
        # Normalize to 0-255
        c = (c * 255.0 / c[-1]).astype(np.uint8)
        return c

    cdf_before = cdf(before_lab[:, :, 0])
    cdf_after = cdf(after_lab[:, :, 0])

    # Build mapping: for each after L value, find the before L value with
    # the closest CDF value
    mapping = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        diff = np.abs(cdf_before.astype(int) - cdf_after[i].astype(int))
        mapping[i] = np.argmin(diff)

    # Apply mapping to after L channel
    after_lab[:, :, 0] = mapping[after_lab[:, :, 0]]

    return cv2.cvtColor(after_lab, cv2.COLOR_LAB2BGR)


# ── Wall-scan crop helpers ────────────────────────────────────────────────

def get_tag_size(corners):
    """Pixel width and height of an AprilTag from its 4 corners."""
    xs = corners[:, 0]
    ys = corners[:, 1]
    return float(xs.max() - xs.min()), float(ys.max() - ys.min())


def classify_tag_positions(tags):
    """Classify the opposite-corner arrangement of exactly 2 AprilTags.

    Determines whether the tags form a top-left / bottom-right (``tl_br``)
    or bottom-left / top-right (``bl_tr``) diagonal based on their centres.

    Returns
    -------
    arrangement : str  'tl_br' or 'bl_tr'
    left_id : int      Tag ID on the left side
    right_id : int     Tag ID on the right side
    """
    ids = sorted(tags.keys())
    if len(ids) != 2:
        print(f"[ERROR] Wall-scan mode requires exactly 2 AprilTags, found {len(ids)}")
        sys.exit(1)

    center_0 = tags[ids[0]].mean(axis=0)
    center_1 = tags[ids[1]].mean(axis=0)

    if center_0[0] <= center_1[0]:
        left_id, right_id = ids[0], ids[1]
        left_center, right_center = center_0, center_1
    else:
        left_id, right_id = ids[1], ids[0]
        left_center, right_center = center_1, center_0

    # Smaller y = higher in image (top)
    if left_center[1] <= right_center[1]:
        return "tl_br", left_id, right_id
    else:
        return "bl_tr", left_id, right_id


def wall_scan_crop(image, tags, first_block=True, trim_edges=None):
    """Crop an image to the region defined by two diagonal AprilTags.

    First block  → full bounding-box of both tags.
    Later blocks → same box, minus the overlap strips on *trim_edges*
                   whose width equals the tag size on each edge.

    Parameters
    ----------
    image : ndarray (H, W, 3) BGR
    tags : dict  {tag_id: corners(4×2)}  — exactly 2 tags
    first_block : bool
    trim_edges : list[str] | None
        Edges to trim, e.g. ['left', 'right'].  Default (None) → ['left', 'right'].

    Returns
    -------
    cropped : ndarray
    crop_rect : (x0, y0, x1, y1)
    arrangement : str
    """
    if trim_edges is None:
        trim_edges = ["left", "right"]

    arrangement, left_id, right_id = classify_tag_positions(tags)

    corners_left = tags[left_id]
    corners_right = tags[right_id]
    all_corners = np.vstack([corners_left, corners_right])

    x0 = int(np.floor(all_corners[:, 0].min()))
    y0 = int(np.floor(all_corners[:, 1].min()))
    x1 = int(np.ceil(all_corners[:, 0].max()))
    y1 = int(np.ceil(all_corners[:, 1].max()))

    # Clamp to image bounds
    ih, iw = image.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(iw, x1), min(ih, y1)

    trimmed = []
    if not first_block:
        for edge in trim_edges:
            if edge == "left":
                tag_w, _ = get_tag_size(corners_left)
                x0 += int(np.ceil(tag_w))
                trimmed.append(f"left ({int(np.ceil(tag_w))}px)")
            elif edge == "right":
                tag_w, _ = get_tag_size(corners_right)
                x1 -= int(np.ceil(tag_w))
                trimmed.append(f"right ({int(np.ceil(tag_w))}px)")
            elif edge == "top":
                # Pick the tag with smaller y (higher in image)
                if corners_left.mean(axis=0)[1] < corners_right.mean(axis=0)[1]:
                    _, tag_h = get_tag_size(corners_left)
                else:
                    _, tag_h = get_tag_size(corners_right)
                y0 += int(np.ceil(tag_h))
                trimmed.append(f"top ({int(np.ceil(tag_h))}px)")
            elif edge == "bottom":
                if corners_left.mean(axis=0)[1] > corners_right.mean(axis=0)[1]:
                    _, tag_h = get_tag_size(corners_left)
                else:
                    _, tag_h = get_tag_size(corners_right)
                y1 -= int(np.ceil(tag_h))
                trimmed.append(f"bottom ({int(np.ceil(tag_h))}px)")

    print(f"  Arrangement: {arrangement} (left=tag {left_id}, right=tag {right_id})")
    print(f"  Crop region: ({x0}, {y0}) -> ({x1}, {y1})  [{x1 - x0} x {y1 - y0}]")
    if trimmed:
        print(f"  Trimmed overlap: {', '.join(trimmed)}")

    cropped = image[y0:y1, x0:x1].copy()
    return cropped, (x0, y0, x1, y1), arrangement
