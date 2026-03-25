# Image Diff — Building Change Detection

Compares before/after photos of a building surface using AprilTag-based alignment and illumination-robust change detection. Outputs a heatmap overlay highlighting structural changes.

## How it works

1. Detects AprilTags in both images for alignment
2. Computes homography to warp the "after" image into the "before" coordinate space
3. Refines alignment to sub-pixel accuracy using ECC (Enhanced Correlation Coefficient)
4. Computes difference using one of four methods
5. Thresholds + filters the diff into change regions
6. Overlays yellow highlights on detected changes

## Alignment pipeline

The alignment works in two stages:

**Stage 1 — AprilTag homography:** Finds known AprilTag markers in both images and computes a perspective transform (homography) that maps the "after" image onto the "before" image. This gets alignment to within a few pixels, but not perfect — small errors cause false change detections across the whole image.

**Stage 2 — ECC refinement (on by default):** Takes the rough homography from stage 1 and iteratively adjusts it to maximize pixel-level correlation between the two images. This achieves sub-pixel alignment accuracy, eliminating the scattered false detections caused by tiny shifts. Disable with `--no-ecc` if tags are highly accurate or ECC fails to converge.

## Diff methods

Four methods are available for computing the difference between aligned images. Each has different strengths:

### `gradient` (default)
Computes Sobel edge gradients on both images and compares their magnitudes. Since edges are defined by relative pixel differences (not absolute brightness), this method is robust to global lighting changes. Best for: outdoor scenes where light varies between shots but structure is what matters.

### `ssim` (recommended for noisy conditions)
**SSIM (Structural Similarity Index)** compares images using local 11×11 patches rather than individual pixels. For each patch it considers three things: luminance (brightness), contrast (variance), and structure (correlation pattern). The key advantage: if a pixel shifts by 1–2px due to imperfect alignment, the surrounding patch still looks nearly identical, so SSIM won't flag it as a change. Only actual structural differences (a crack appearing, an object moved) produce high SSIM diff scores. Best for: real-world conditions where perfect pixel alignment is hard to achieve.

### `local_norm`
Divides the image into tiles and normalizes each tile's brightness independently before comparing. This handles uneven lighting across the surface (e.g. one corner in shadow, the other in sun). Best for: surfaces with patchy lighting or partial shade.

### `raw`
Direct pixel intensity difference. Simple and fast, but sensitive to any lighting change between shots. Best for: controlled indoor environments with fixed lighting.

### `all`
Runs all four methods and uses `gradient` as the primary output. Useful for comparing which method works best for your specific scene — check the debug panel.

## Running

```bash
cd crack_detection/image_diff

# Default run (gradient + ECC refinement)
docker compose run --rm crack-detector

# SSIM method (best for noisy/imperfect alignment)
docker compose run --rm crack-detector --method ssim

# Compare all methods side by side
docker compose run --rm crack-detector --method all

# Custom parameters
docker compose run --rm crack-detector --sensitivity 0.2 --method ssim --min-area 3000 --no-backup
```

Place image pairs in `input/` as `before-<date>.jpg` and `after-<date>.jpg` (e.g. `before-25-03-26.jpg`).

Results go to `output/`, backups to `output_backup/`.

## Key parameters

| Parameter | Default | Description |
|---|---|---|
| `--sensitivity` | `0.30` | Change threshold 0–1. Lower = more sensitive |
| `--min-area` | `5000` | Minimum region size in px². Filters out noise |
| `--method` | `gradient` | `gradient`, `local_norm`, `raw`, `ssim`, or `all` |
| `--no-ecc` | off | Skip ECC sub-pixel alignment refinement |
| `--ecc-iterations` | `200` | Max iterations for ECC convergence |
| `--alpha` | `0.4` | Heatmap opacity (0 = transparent, 1 = solid) |
| `--blur` | `5` | Gaussian blur kernel size |
| `--pair` | all | Process only one date, e.g. `25-03-26` |
| `--no-debug` | off | Skip 3-panel debug image |
| `--no-backup` | off | Skip backup copy |
| `--tag-family` | `tag25h9` | AprilTag family |
| `--tag-ids` | `0 1` | Expected tag IDs in scene |
| `--homography-method` | `ransac` | `ransac`, `lmeds`, or `rho` |
| `--ransac-thresh` | `5.0` | RANSAC reprojection threshold (px) |

## Tuning detection

**Too many false positives (noise, shadows, minor shifts):**
- Switch to `--method ssim` — naturally ignores small pixel shifts
- Increase `--sensitivity` (e.g. 0.4–0.5)
- Increase `--min-area` (e.g. 10000) to filter small blobs
- Make sure ECC is enabled (don't use `--no-ecc`)

**Missing real changes:**
- Decrease `--sensitivity` (e.g. 0.1–0.2)
- Decrease `--min-area` (e.g. 1000) to keep smaller regions
- Try `--method all` to compare which method captures the change best

## Severity scale

Based on percentage of changed pixels:

| Changed area | Severity |
|---|---|
| < 1% | None |
| 1–8% | Low |
| 8–25% | Moderate |
| > 25% | High |
