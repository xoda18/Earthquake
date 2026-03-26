# Image Diff — Building Change Detection

Compares before/after photos of a building surface using AprilTag-based alignment and illumination-robust change detection. Outputs a heatmap overlay highlighting structural changes.

## How it works

1. Detects AprilTags in both images for alignment
2. Warps **both** images to a canonical coordinate frame (tag-distance-based) so interpolation artifacts are symmetric and cancel out
3. Refines alignment to sub-pixel accuracy using ECC (Enhanced Correlation Coefficient)
4. Matches illumination (LAB histogram matching)
5. Computes difference using one of seven methods
6. Filters by area and shape (solidity), optionally expands to full object boundaries
7. Overlays yellow highlights on detected changes

## Project structure

```
crack_detection/image_diff/
├── detect.py              # CLI entry point + orchestration
├── pipeline/
│   ├── preprocessing.py   # AprilTag detection, canonical frame, histogram matching
│   ├── alignment.py       # Homography, ECC refinement, overlap cropping
│   ├── diff_methods.py    # All 7 diff algorithms + masks
│   ├── postprocessing.py  # Morphology, heatmap, annotation, fill modes
│   └── io.py              # File I/O, backup, debug panels
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Diff methods

### `gradient` (default)
Sobel edge gradient magnitude difference. Robust to global lighting changes. Best for outdoor scenes.

### `edge_xor`
Canny edges of both images are dilated (tolerance for sub-pixel error), then XOR'd. Unchanged objects cancel out — only truly new/removed objects produce orphan edges. **Best for detecting moved/added/removed objects with zero false positives on static edges.**

### `lab`
Chrominance-only difference in LAB color space (A+B channels). Ignores lightness (L), so shadows, exposure changes, and lighting shifts are filtered out. **Best when lighting varies between shots.**

### `combined`
Runs both `edge_xor` and `lab`, combines results with `--combine-mode or` (union, default) or `--combine-mode and` (intersection, stricter).

### `ssim`
SSIM (Structural Similarity Index) — compares local 11x11 patches. Tolerant to 1-2px alignment error. Best for noisy conditions.

### `local_norm`
Tile-based local brightness normalisation. Handles uneven lighting (patchy shade).

### `raw`
Direct pixel intensity difference. Simple, fast, sensitive. Best for controlled indoor environments.

### `all`
Runs all methods and saves a side-by-side comparison panel.

## Quick start

```bash
cd crack_detection/image_diff

# Build the Docker image
docker compose build

# Place images in input/ as before-<date>.jpg and after-<date>.jpg
# e.g. before-25-03-26.jpg, after-25-03-26.jpg
```

## Running — recommended scenarios

### Best overall detection (start here)

```bash
docker compose run --rm crack-detector \
  --method combined \
  --fill-mode convex \
  --min-solidity 0.3
```

This runs edge XOR + LAB color diff combined, fills partial detections with convex hulls (so moved objects are fully highlighted), and rejects thin-line false positives (laptop borders, wires).

### If objects are only partially highlighted

The convex hull might not be enough for complex shapes. Try flood-fill which follows actual object boundaries:

```bash
docker compose run --rm crack-detector \
  --method combined \
  --fill-mode flood \
  --min-solidity 0.3
```

### If too many false positives remain

Increase sensitivity threshold and solidity filter:

```bash
docker compose run --rm crack-detector \
  --method edge_xor \
  --fill-mode convex \
  --min-solidity 0.4 \
  --sensitivity 0.4 \
  --min-area 8000
```

`edge_xor` alone is the cleanest method — it ignores static edges entirely. Bump `--min-solidity` to reject more thin shapes.

### If real changes are being missed

Lower thresholds and use the most sensitive method:

```bash
docker compose run --rm crack-detector \
  --method raw \
  --fill-mode convex \
  --sensitivity 0.15 \
  --min-area 2000 \
  --min-solidity 0.2
```

### Compare all methods side by side

```bash
docker compose run --rm crack-detector --method all --fill-mode convex --min-solidity 0.3
```

Outputs a comparison panel image with all 7 methods. Check which one works best for your scene.

### Legacy mode (old behavior, no canonical frame)

```bash
docker compose run --rm crack-detector --method gradient --no-canonical --fill-mode none
```

### Process a specific image pair only

```bash
docker compose run --rm crack-detector --pair 25-03-26 --method combined --fill-mode convex --min-solidity 0.3
```

## All parameters

### Core
| Parameter | Default | Description |
|---|---|---|
| `--method` | `gradient` | `gradient`, `edge_xor`, `lab`, `combined`, `ssim`, `local_norm`, `raw`, `all` |
| `--sensitivity` | `0.30` | Change threshold 0-1. Lower = more sensitive |
| `--min-area` | `5000` | Minimum region size in px2. Filters noise |
| `--pair` | all | Process only one date, e.g. `25-03-26` |

### Shape filtering & fill
| Parameter | Default | Description |
|---|---|---|
| `--fill-mode` | `none` | `none` = raw blobs, `convex` = convex hull of nearby blobs, `flood` = edge-bounded flood-fill |
| `--min-solidity` | `0.0` | Reject contours with solidity below this (0-1). Thin lines ~0.1-0.2, objects ~0.4+. 0 = off |

### Alignment
| Parameter | Default | Description |
|---|---|---|
| `--no-canonical` | off | Skip canonical frame (legacy warp-after-only mode) |
| `--no-ecc` | off | Skip ECC sub-pixel refinement |
| `--ecc-iterations` | `200` | Max ECC iterations |
| `--ecc-epsilon` | `1e-6` | ECC convergence threshold |
| `--homography-method` | `ransac` | `ransac`, `lmeds`, or `rho` |
| `--ransac-thresh` | `5.0` | RANSAC reprojection threshold (px) |

### Combined method
| Parameter | Default | Description |
|---|---|---|
| `--combine-mode` | `or` | `or` = union (either method), `and` = intersection (both agree) |
| `--edge-xor-dilate` | `5` | Edge XOR dilation radius in px |
| `--canny-low` | `50` | Canny low threshold |
| `--canny-high` | `150` | Canny high threshold |

### Output
| Parameter | Default | Description |
|---|---|---|
| `--alpha` | `0.4` | Heatmap opacity (0=transparent, 1=solid) |
| `--no-debug` | off | Skip 3-panel debug image |
| `--no-annotate` | off | Skip legend, bounding boxes, tag markers |
| `--no-backup` | off | Skip backup copy to output_backup/ |

### AprilTags
| Parameter | Default | Description |
|---|---|---|
| `--tag-family` | `tag25h9` | AprilTag family string |
| `--tag-ids` | `0 1` | Expected tag IDs in scene |
| `--edge-suppress` | `7` | Suppress diff within N px of edges (0=off) |

## Tuning guide

| Problem | Fix |
|---|---|
| Thin lines highlighted (wires, borders) | Add `--min-solidity 0.3` (or higher) |
| Object only partially highlighted | Add `--fill-mode convex` or `--fill-mode flood` |
| Too many false positives overall | Switch to `--method edge_xor`, increase `--sensitivity` |
| Lighting changes cause false diffs | Use `--method lab` or `--method combined` |
| Missing real changes | Lower `--sensitivity 0.15`, lower `--min-area 2000` |
| Images from very different angles | Make sure both AprilTags are visible, use `--no-canonical` if alignment fails |

## Severity scale

| Changed area | Severity |
|---|---|
| < 1% | None |
| 1-8% | Low |
| 8-25% | Moderate |
| > 25% | High |

## Input format

Place image pairs in `input/` as `before-<date>.<ext>` and `after-<date>.<ext>`.

Supported extensions: `.jpg`, `.jpeg`, `.tif`, `.tiff`

Results go to `output/`, backups to `output_backup/`.
