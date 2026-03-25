# Building Change Detection — Claude Code Spec

## Project goal
Compare two `.tif` OR `.JPG` photos of the same wall/building taken from a drone at
**different heights** (and possibly slightly different angles). Detect structural
changes (new cracks, missing material, collapse) that are robust to lighting
differences between the two shots.

Two **AprilTags** (family `tag25h9`, IDs 0 and 1, 15*15 sm) are physically mounted at
known positions in the scene. They are visible in both photos and are used to
compute a full **homography** that corrects scale, rotation, and perspective
before any diffing happens.

---

## Folder layout

```
crack_detection/image_diff/
├── input/
│   ├── before-DD-MM-YY.tif      ← reference photo (the "ground truth" state)
│   └── after--DD-MM-YY.tif       ← comparison photo (taken later / from different height)
├── output/
│   └── diff-DD-MM-YY_<time_generated>.png        ← after image with change heatmap overlaid (written by script)
├── detect.py           ← main script

```

The script reads **only from `input/`** and writes **only to `output/`**.
Never modify the input files.

---

## Requirements (`requirements.txt`)

```
opencv-python
numpy
Pillow
pupil-apriltags
matplotlib
```

Build inside docker container with mounted volume folder.
Write docker script installing all neeede things 

---

## What `detect.py` must do — step by step

### 1. Load images
- Read ALL (iterate over) `input/before-*.tif/.JPG` and `input/after-*.tif/.JPG` using `cv2.imread(..., cv2.IMREAD_COLOR)`.
- If either file is missing, print a clear error and exit with code 1.
- Print both image shapes after loading.

### 2. Detect AprilTags in both images
- Use `pupil_apriltags.Detector`.
- Convert each image to grayscale before detection.
- Detect tags in both `before` and `after`.
- **Required**: both tag IDs 0 and 1 must be found in **both** images.
- If any tag is missing from either image, print which tag is missing and in
  which image, then exit with code 1.

### 3. Compute homography
- Match tags by ID across the two images.
- Each tag provides 4 corner points → 2 tags = 8 point correspondences.
- Compute `H, mask = cv2.findHomography(pts_after, pts_before, cv2.RANSAC, 5.0)`.
  - `pts_after`: 8×2 array of tag corners in the after image.
  - `pts_before`: 8×2 array of the **same** tag corners in the before image.
- `H` maps the after image coordinate space → before image coordinate space.
- Print the reprojection error (mean distance between mapped pts_after and
  pts_before) so the user can sanity-check alignment quality.

### 4. Warp the after image
- `warped_after = cv2.warpPerspective(after_img, H, (before_img.shape[1], before_img.shape[0]))`
- Also warp a white mask (`np.ones`) the same way to know which pixels are valid
  after the warp (some borders will be black/invalid).

### 5. Crop to valid intersection
- The valid region is where the warped mask > 0 AND within before image bounds.
- Compute bounding box: `x0, y0, x1, y1` = tightest rectangle of valid pixels.
- Crop both `before_img` and `warped_after` to `[y0:y1, x0:x1]`.
- Print the crop dimensions.

### 6. Compute illumination-robust change map
Run **all three** methods and use `gradient` as the primary output.
Keep the others for debugging.

**Method A — Gradient (primary, light-robust)**
```
For each image:
  1. Convert to grayscale float32
  2. Apply Sobel in X and Y: gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                              gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
  3. Magnitude: mag = sqrt(gx² + gy²)
Diff = |mag_before − mag_after|
Normalise to [0, 1]
```

**Method B — Local normalisation (handles shadow/patch lighting)**
```
For each image:
  Divide image into 32×32 pixel tiles.
  Within each tile: subtract tile mean, divide by tile std+1.
Diff = |normed_before − normed_after|
Normalise to [0, 1]
```

**Method C — Raw pixel diff (baseline, breaks with lighting)**
```
Diff = |gray_before − gray_after| / 255
```

Apply Gaussian blur `cv2.GaussianBlur(diff, (5,5), 0)` to all three diffs
to reduce sensor noise.

### 7. Threshold + build heatmap overlay
- Threshold = `SENSITIVITY` (default 0.20, configurable at top of script as a
  constant).
- For each pixel where `diff_gradient > SENSITIVITY`, colour it using the **jet
  colormap** scaled from threshold to 1.0.
- Alpha-blend the heatmap over the **cropped after image**:
  `output = 0.6 * cropped_after + 0.4 * heatmap` in changed regions.
  Unchanged pixels stay as-is.

### 8. Annotate output image
Draw on the output image (using `cv2.putText` / `cv2.rectangle`):
- A small legend in the bottom-left: `"Change map: gradient method | sensitivity=0.20"`
- Bounding box around the largest contiguous changed region (if any region > 200px²).
- Tag locations from the `before` image: draw small green squares where tags were
  found, labelled `T0` and `T1`.

### 9. Save output
- `cv2.imwrite('output/diff.png', output_bgr)`
- Also save `output/diff_debug.png` — a 3-panel image (before crop | after crop |
  heatmap only) for debugging alignment. Use `np.hstack` or `matplotlib`.
- Print: `"Saved output/diff.png"` and a summary line:
  `"Changed area: X.X% | Max score: 0.XXX | Severity: Low/Moderate/High/None"`

Severity thresholds: `< 1%` → None, `1–8%` → Low, `8–25%` → Moderate, `> 25%` → High.

---

## Configurable constants (top of `detect.py`)

```python
SENSITIVITY   = 0.20   # change threshold, 0–1
BLUR_KSIZE    = 5      # Gaussian blur kernel (odd number)
TILE_SIZE     = 32     # local normalisation tile size in px
TAG_FAMILY    = 'tag36h11'
TAG_IDS       = [0, 1] # expected tag IDs in scene
INPUT_BEFORE  = 'input/before.tif'
INPUT_AFTER   = 'input/after.tif'
OUTPUT_DIFF   = 'output/diff.png'
OUTPUT_DEBUG  = 'output/diff_debug.png'
```

---

## Error handling requirements

| Situation | Behaviour |
|---|---|
| Input file missing | Print filename, exit code 1 |
| Tag not detected | Print which tag, which image, suggest checking lighting/occlusion, exit code 1 |
| Homography reprojection error > 5px | Print warning but continue — do not exit |
| Valid crop region < 10% of before image | Print warning, likely bad warp, continue |
| output/ folder doesn't exist | Create it automatically |

---

## Physical setup reminder (not code — for the operator)

```
┌─────────────────────────┐
│  [TAG ID=0]             │  ← top-left of inspection zone
│                         │
│      WALL / SURFACE     │
│                         │
│             [TAG ID=1]  │  ← bottom-right of inspection zone
└─────────────────────────┘
```

- Tag family: **tag36h11**
- Print size: **15×15 cm** with at least 1 tag-width white border
- Both tags must be **fully in frame** in both photos
- Tags must be **co-planar** with the surface being inspected
- Get tag PNG files from:
  https://github.com/AprilRobotics/apriltag-imgs/tree/master/tag36h11
  → `tag36_11_00000.png` (ID 0) and `tag36_11_00001.png` (ID 1)

---

## How to run

```bash
pip install -r requirements.txt
# Place before.tif and after.tif in input/
python detect.py
# Output written to output/diff.png and output/diff_debug.png
```

---

## What NOT to do

- Do not use any deep learning model — pure classical CV only.
- Do not hardcode image dimensions.
- Do not modify files in `input/`.
- Do not use `skimage` — only `opencv-python`, `numpy`, `Pillow`, `matplotlib`, `pupil-apriltags`.
- Do not assume both images are the same size — the homography handles this.