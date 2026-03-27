# VLM Damage Recognition

Vision Language Model-based structural damage recognition for drone imagery. Uses **LLaVA** to analyze building damage, identify damage types, and assess severity levels.

## Features

- **Multi-format image support**: JPG, PNG, TIFF, GeoTIFF (with automatic geolocation extraction)
- **Vision-Language Analysis**: Uses LLaVA to understand complex damage patterns
- **Semantic Damage Classification**: Identifies damage types (cracks, collapses, displacement, etc.)
- **Severity Assessment**: Classifies damage as low, moderate, high, or critical
- **Supabase Integration**: Writes reports directly to the drone_reports database table
- **GPU/CPU Auto-detection**: Runs on both CUDA and CPU with optional int8 quantization
- **Batch Processing**: Process entire directories of images in one run
- **Coordinate Extraction**: Automatically extracts geolocation from GeoTIFF or EXIF metadata

## Installation

### 1. Install Dependencies

```bash
pip install -r VLM_damage_recognition/requirements.txt
```

For GPU acceleration (CUDA):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CPU only (more portable):
```bash
pip install torch torchvision torchaudio
```

### 2. Verify Installation

```bash
python -c "from VLM_damage_recognition import DamageAnalyzer; print('OK')"
```

## Usage

### Single Image Analysis

Analyze a single image with specified coordinates:

```bash
python VLM_damage_recognition/main.py \
    --image /path/to/image.jpg \
    --lat 34.765 \
    --lon 32.420 \
    --building "Hotel" \
    --supabase-write
```

### Batch Processing

Process all images in a directory:

```bash
# Write to Supabase
python VLM_damage_recognition/main.py \
    --input-dir crack_detection/image_diff/input/ \
    --supabase-write

# Output to JSON only (no Supabase write)
python VLM_damage_recognition/main.py \
    --input-dir input/ \
    --output-json damage_reports.jsonl

# Quantized CPU inference
python VLM_damage_recognition/main.py \
    --input-dir input/ \
    --quantize \
    --supabase-write
```

### Advanced Options

```bash
python VLM_damage_recognition/main.py \
    --input-dir input/ \
    --drone-id "JASS-DRONE-02" \
    --model "llava-hf/llava-1.5-13b-hf" \
    --quantize \
    --output-json reports.jsonl \
    --supabase-write
```

## Python API

### Basic Usage

```python
from VLM_damage_recognition import DamageAnalyzer, ImageProcessor, SupabaseReporter
from PIL import Image

# Initialize analyzer
analyzer = DamageAnalyzer(quantize=False)

# Load image
image, coords = ImageProcessor.load_image("image.jpg")
lat, lon = coords or (34.765, 32.420)

# Analyze
report = analyzer.analyze_image(image, lat, lon, building_name="Hotel")

# Write to Supabase
reporter = SupabaseReporter(use_supabase=True)
reporter.write_report(report)

print(report)
```

### Batch Analysis

```python
from VLM_damage_recognition import DamageAnalyzer, ImageProcessor, SupabaseReporter

# Load all images from directory
images = ImageProcessor.batch_load_images("input/")

# Analyze all
analyzer = DamageAnalyzer()
reports = analyzer.analyze_batch(images)

# Write to Supabase
reporter = SupabaseReporter(use_supabase=True)
stats = reporter.write_batch(reports)
print(f"Success: {stats['success']}, Failed: {stats['failed']}")

# Save JSON backup
reporter.save_json_reports(reports, "damage_reports.jsonl")
```

## Output Format

Damage reports match the existing `drone_reports` schema:

```json
{
  "event_id": "uuid",
  "epoch": 1711000000.0,
  "lat": 34.765,
  "lon": 32.420,
  "severity": "high",
  "damage_type": "crack, displacement",
  "building": "Hotel",
  "description": "Multiple horizontal cracks in load-bearing walls...",
  "drone_id": "JASS-DRONE-01",
  "confidence": 0.87,
  "_vlm_analysis": {
    "area_percent": 35,
    "affected_elements": ["wall", "window_frame"],
    "raw_response": "..."
  }
}
```

### Damage Types

Recognized damage types:
- **crack** - Fractures in structure
- **spalling** - Surface material breakage
- **collapse** - Structural failure
- **displacement** - Movement/misalignment
- **deformation** - Shape change
- **corrosion** - Material degradation
- **efflorescence** - Surface deposits
- **water_damage** - Moisture/flooding effects
- **settlement** - Foundation subsidence
- **tilt** - Structural tilt
- **buckling** - Compression failure
- **crushing** - Material crushing
- **shear_failure** - Shear stress failure
- **rebar_exposure** - Exposed reinforcement

### Severity Levels

- **low**: Minor cosmetic damage, cracks < 1mm
- **moderate**: Visible damage, cracks 1-5mm, affects appearance
- **high**: Significant damage, cracks > 5mm, may affect structural integrity
- **critical**: Severe damage, obvious collapse risk, immediate danger

## Performance

### Hardware Requirements

**GPU (Recommended)**
- NVIDIA GPU with 6GB+ VRAM (RTX 3060 or better)
- Inference time: ~2-5 seconds per image

**CPU Only**
- Any modern CPU (Intel/AMD)
- Use `--quantize` flag for int8 quantization
- Inference time: ~30-60 seconds per image (quantized)

### Model Sizes

- **llava-1.5-7b-hf** (default): ~15GB VRAM, good balance
- **llava-1.5-13b-hf**: ~20GB VRAM, more capable
- **Custom quantized**: ~2-3GB VRAM with int8 quantization

## Coordinate Extraction

### GeoTIFF Images
Automatically extracts lat/lon from GeoTIFF spatial reference:
```python
image, (lat, lon) = ImageProcessor.load_image("orthophoto.tif")
# Returns center coordinates from raster bounds
```

### EXIF Metadata (JPG/PNG)
Extracts GPS coordinates from image EXIF:
```python
image, (lat, lon) = ImageProcessor.load_image("drone_photo.jpg")
# Returns (latitude, longitude) from EXIF GPS tags
```

### Fallback
If no geolocation found, generates random Paphos coordinates:
```python
image, coords = ImageProcessor.load_image("image.jpg")
# coords will be None, then randomly generated in main.py
```

## Integration with Earthquake System

### With Drone Workflow

Replace or supplement `drone/mock_damage.py`:
```bash
# Generate real damage reports instead of mock data
python VLM_damage_recognition/main.py \
    --input-dir drone_images/ \
    --supabase-write
```

### With Crack Detection Pipeline

Use alongside `crack_detection/image_diff/detect.py`:
```bash
# Image diff finds changes, VLM classifies damage types
python crack_detection/image_diff/detect.py \
    --input-dir images/ \
    --output-dir output/

# Then analyze the detected changes with VLM
python VLM_damage_recognition/main.py \
    --input-dir output/ \
    --supabase-write
```

### With EarthAgent

Tools can call VLM analyzer:
```python
# In tools.py
def analyze_drone_image(image_path: str) -> dict:
    from VLM_damage_recognition import DamageAnalyzer, ImageProcessor

    analyzer = DamageAnalyzer()
    image, coords = ImageProcessor.load_image(image_path)
    lat, lon = coords or (34.765, 32.420)

    report = analyzer.analyze_image(image, lat, lon)
    return report
```

## Troubleshooting

### Out of Memory

Use quantization on CPU or smaller model:
```bash
python VLM_damage_recognition/main.py \
    --input-dir input/ \
    --quantize \
    --model "llava-hf/llava-1.5-7b-hf"
```

### Slow Inference

- **GPU not detected**: Verify CUDA installation
- **CPU only**: Expected ~30-60s per image
- **Smaller model**: Use 7B variant instead of 13B

### Image Loading Errors

Ensure image format is supported and file is readable:
```bash
file image.jpg  # Check file type
identify image.jpg  # ImageMagick tool
```

For GeoTIFF issues:
```bash
python -c "import rasterio; print(rasterio.__version__)"
```

## Documentation

### Damage Report Schema

The system uses a **standardized JSON schema** for all damage reports:

- **[DAMAGE_REPORT_SCHEMA_GUIDE.md](DAMAGE_REPORT_SCHEMA_GUIDE.md)** - Basic schema with required/optional fields
- **[DETAILED_CRACK_SCHEMA_GUIDE.md](DETAILED_CRACK_SCHEMA_GUIDE.md)** - **NEW: Comprehensive per-crack measurements, location tracking, and temporal analysis**
- **[CRACK_TRACKING_GUIDE.md](CRACK_TRACKING_GUIDE.md)** - Comparing multiple measurements over time

### Key Features of New Schema

✅ **Per-Crack Measurements** - Each crack has detailed dimensions (length, width, depth)
✅ **Location Coordinates** - Region-based and normalized pixel coordinates
✅ **Individual Severity & Status** - Each crack can have different severity (low/moderate/high/critical) and status (stable/growing/recovering/unknown)
✅ **Summary Statistics** - Automatically calculated from crack array
✅ **Expandable Format** - Add custom fields as needed
✅ **Database Ready** - JSONB compatible for Supabase and other databases

### Example Report Output

```json
{
  "file": "photo.jpg",
  "lat": 34.765,
  "lon": 32.42,
  "severity": "critical",
  "status": "growing",
  "confidence": 0.92,
  "cracks": [
    {
      "id": 1,
      "location": "top-left",
      "measurements": {
        "length_mm": 450,
        "width_mm": 5.0,
        "depth_estimate": "deep",
        "area_mm2": 2250,
        "pattern": "straight"
      },
      "severity": "critical",
      "status": "growing",
      "confidence": 0.92,
      "description": "Vertical crack with progressive widening",
      "normalized_coords": {"x": 0.15, "y": 0.25}
    }
  ],
  "_summary_statistics": {
    "total_cracks": 1,
    "total_crack_area_mm2": 2250,
    "largest_crack_length_mm": 450,
    "crack_density": "moderate"
  }
}
```

See `test_results/analysis_metadata.json` for complete examples.

## Advanced Configuration

Edit `config.yaml`:
```yaml
model:
  quantize: true
  model_id: "llava-hf/llava-1.5-7b-hf"

output:
  supabase_write: true
  verbose: true

analysis:
  confidence_threshold: 0.3
```

## Future Enhancements

- [ ] Real-time RTSP stream processing
- [ ] Fine-tuned damage classifier
- [ ] Multi-image time-series analysis
- [ ] Building-specific damage patterns
- [ ] Automated severity escalation alerts
- [ ] Damage report PDF generation
- [ ] Integration with DJI drone APIs

## References

- [LLaVA Paper](https://llava-vl.github.io/)
- [HuggingFace Models](https://huggingface.co/llava-hf)
- [Structural Damage Types](https://en.wikipedia.org/wiki/Building_assessment)
- [GeoTIFF Specification](https://www.ogc.org/standards/geotiff)

## License

MIT License - See parent Earthquake project LICENSE
