"""Heatmap overlay, annotation, and severity classification."""

import cv2
import numpy as np


def normalize_diff(diff, percentile=99.0):
    """Normalize diff map to 0-1 range using percentile clipping."""
    cap = np.percentile(diff, percentile)
    if cap < 1e-6:
        return np.zeros_like(diff)
    normed = np.clip(diff / cap, 0, 1).astype(np.float32)
    return normed


def _contour_solidity(cnt):
    """Area / convex-hull area.  Thin lines → low, compact blobs → high."""
    area = cv2.contourArea(cnt)
    if area < 1:
        return 0.0
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area < 1:
        return 0.0
    return area / hull_area


def _group_nearby_contours(contours, gap=60):
    """Group contours whose bounding boxes are within `gap` px of each other.

    Returns a list of groups, where each group is a list of contour indices.
    Uses simple union-find on bounding-box proximity.
    """
    n = len(contours)
    if n == 0:
        return []

    bboxes = [cv2.boundingRect(c) for c in contours]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    def boxes_close(a, b, gap):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        # Expand box a by gap on each side and check overlap with b
        return not (bx > ax + aw + gap or bx + bw < ax - gap or
                    by > ay + ah + gap or by + bh < ay - gap)

    for i in range(n):
        for j in range(i + 1, n):
            if boxes_close(bboxes[i], bboxes[j], gap):
                union(i, j)

    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    return list(groups.values())


def _fill_convex(contours, shape):
    """Group nearby contours and fill each group's convex hull."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if not contours:
        return mask

    groups = _group_nearby_contours(contours, gap=60)
    for group in groups:
        # Merge all points from contours in this group
        pts = np.vstack([contours[i] for i in group])
        hull = cv2.convexHull(pts)
        cv2.drawContours(mask, [hull], -1, 1, thickness=cv2.FILLED)

    return mask


def _fill_flood(contours, after_gray, shape, max_expand=5.0):
    """Use change blobs as seeds, flood-fill bounded by edges in after image.

    Finds the full object boundary around each change region by expanding
    into homogeneous areas of the after image.  Expansion is capped at
    max_expand * seed_area to prevent leaking across large uniform surfaces.
    """
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if not contours:
        return mask

    # Strong edge map as flood-fill boundary
    edges = cv2.Canny(after_gray, 50, 150)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edges = cv2.dilate(edges, edge_kernel)

    h, w = shape[:2]

    for cnt in contours:
        seed_area = cv2.contourArea(cnt)
        if seed_area < 1:
            continue
        max_fill_area = seed_area * max_expand

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))

        # Restrict flood fill to a bounding box around the seed (prevents global leak)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        expand_px = int(max(bw, bh) * 0.5)
        rx0 = max(0, bx - expand_px)
        ry0 = max(0, by - expand_px)
        rx1 = min(w, bx + bw + expand_px)
        ry1 = min(h, by + bh + expand_px)

        # Extract ROI
        roi_gray = after_gray[ry0:ry1, rx0:rx1]
        roi_edges = edges[ry0:ry1, rx0:rx1]
        rh, rw = roi_gray.shape

        # Build flood-fill mask for ROI
        ff_mask = np.zeros((rh + 2, rw + 2), dtype=np.uint8)
        ff_mask[1:-1, 1:-1] = (roi_edges > 0).astype(np.uint8)

        # Seed point in ROI coordinates
        seed_x = cx - rx0
        seed_y = cy - ry0
        seed_x = max(0, min(rw - 1, seed_x))
        seed_y = max(0, min(rh - 1, seed_y))

        if ff_mask[seed_y + 1, seed_x + 1] != 0:
            # Find a nearby non-boundary point
            found = False
            for dy in range(-10, 11):
                for dx in range(-10, 11):
                    ny, nx = seed_y + dy, seed_x + dx
                    if 0 <= ny < rh and 0 <= nx < rw and ff_mask[ny + 1, nx + 1] == 0:
                        seed_x, seed_y = nx, ny
                        found = True
                        break
                if found:
                    break
            if not found:
                # Just use the original contour
                cv2.drawContours(mask, [cnt], -1, 1, thickness=cv2.FILLED)
                continue

        ff_mask_copy = ff_mask.copy()
        cv2.floodFill(roi_gray, ff_mask_copy, (seed_x, seed_y), 255,
                       loDiff=10, upDiff=10,
                       flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8))

        filled_roi = (ff_mask_copy[1:-1, 1:-1] > 0).astype(np.uint8)
        fill_area = filled_roi.sum()

        if fill_area > max_fill_area:
            # Fill leaked — fall back to convex hull of the seed contour
            hull = cv2.convexHull(cnt)
            cv2.drawContours(mask, [hull], -1, 1, thickness=cv2.FILLED)
        else:
            mask[ry0:ry1, rx0:rx1] = np.maximum(
                mask[ry0:ry1, rx0:rx1], filled_roi
            )

    return mask


def build_heatmap_overlay(crop_after, diff_map, sensitivity, alpha, min_area,
                          min_solidity=0.0, fill_mode="none"):
    """Build yellow marker overlay highlighting changed objects.

    Parameters
    ----------
    min_solidity : float
        Reject contours with solidity below this (0-1). 0 = no filter.
        Thin lines (laptop border) have ~0.1-0.2, objects have 0.4+.
    fill_mode : str  "none" | "convex" | "flood"
        none   = current behavior (just dilate kept blobs)
        convex = group nearby blobs and fill their convex hull
        flood  = edge-bounded flood-fill from change seeds into after image
    """
    output = crop_after.copy().astype(np.float64)
    raw_mask = (diff_map > sensitivity).astype(np.uint8)

    # Opening: remove thin line artifacts
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    opened = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel_open)

    # Closing: merge nearby pixels into blobs
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

    # Filter by area and solidity
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant = []
    rejected_area = 0
    rejected_solidity = 0
    for c in contours:
        if cv2.contourArea(c) < min_area:
            rejected_area += 1
            continue
        if min_solidity > 0 and _contour_solidity(c) < min_solidity:
            rejected_solidity += 1
            continue
        significant.append(c)

    n_total = len(contours)
    n_kept = len(significant)
    print(f"  Regions: {n_total} raw -> {n_total - rejected_area} area filter "
          f"-> {n_kept} solidity filter (min_solidity={min_solidity:.2f})")

    # Fill mode
    if fill_mode == "convex" and significant:
        mask = _fill_convex(significant, crop_after.shape)
    elif fill_mode == "flood" and significant:
        after_gray = cv2.cvtColor(crop_after, cv2.COLOR_BGR2GRAY)
        mask = _fill_flood(significant, after_gray, crop_after.shape)
    else:
        # Default: draw kept contours and dilate
        filtered = np.zeros_like(closed)
        if significant:
            cv2.drawContours(filtered, significant, -1, 1, thickness=cv2.FILLED)
        mask = filtered

    # Dilate to cover full object extent
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)

    # Solid yellow highlight (BGR: 0, 255, 255)
    yellow = np.full_like(output, (0, 255, 255), dtype=np.float64)

    mask_bool = mask > 0
    mask_3ch = np.stack([mask_bool] * 3, axis=-1)
    output = np.where(
        mask_3ch,
        (1.0 - alpha) * output + alpha * yellow,
        output,
    )

    return np.clip(output, 0, 255).astype(np.uint8), mask_bool


def annotate(output, change_mask, sensitivity, method_name, tags_before,
             crop_offset, annotate_enabled):
    """Draw legend, bounding boxes around changed regions, and tag markers."""
    if not annotate_enabled:
        return output

    h, w = output.shape[:2]
    x0_crop, y0_crop = crop_offset[0], crop_offset[1]

    # Legend
    text = f"Change map: {method_name} | sensitivity={sensitivity:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(output, (5, h - th - 20), (tw + 15, h - 5), (0, 0, 0), -1)
    cv2.putText(output, text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)

    # Bounding boxes around changed regions > 200px2
    binary = change_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            cv2.rectangle(output, (bx, by), (bx + bw, by + bh), (0, 0, 255), 3)

    # Tag locations
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
    """Classify change percentage into severity."""
    if pct < 1.0:
        return "None"
    elif pct < 8.0:
        return "Low"
    elif pct < 25.0:
        return "Moderate"
    else:
        return "High"
