# VLM Damage Recognition - Setup Guide

Step-by-step guide to install and run the VLM damage recognition system.

## Prerequisites

- Python 3.8+ (tested with Python 3.11)
- pip package manager
- 6GB+ available disk space (for model download)
- GPU recommended but not required (CPU works with quantization)

## Installation Steps

### Step 1: Verify Python Installation

```bash
python --version
# Should output: Python 3.x.x

pip --version
# Should output: pip x.x.x
```

### Step 2: Install Dependencies

```bash
cd /home/morozov-mikhail/Earthquake

pip install -r VLM_damage_recognition/requirements.txt
```

For GPU acceleration (CUDA 11.8):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Test Installation

```bash
python VLM_damage_recognition/test_installation.py
```

Expected output:
```
✓ PyTorch.................................. OK
✓ HuggingFace Transformers.................. OK
✓ Pillow................................... OK
✓ piexif................................... OK
✓ NumPy.................................... OK
✓ CUDA device detected: NVIDIA ...
✓ All tests passed! Ready to use.
```

## Quick Start

### Option 1: Single Image Analysis

```bash
cd /home/morozov-mikhail/Earthquake

python VLM_damage_recognition/main.py \
    --image path/to/image.jpg \
    --lat 34.765 \
    --lon 32.420 \
    --building "Hotel"
```

### Option 2: Batch Processing (Recommended)

```bash
# Process all images in a directory
python VLM_damage_recognition/main.py \
    --input-dir crack_detection/image_diff/input/ \
    --supabase-write

# Output to JSON instead of Supabase
python VLM_damage_recognition/main.py \
    --input-dir crack_detection/image_diff/input/ \
    --output-json damage_reports.jsonl
```

### Option 3: CPU-Only Mode

If you don't have GPU or want lower memory usage:

```bash
python VLM_damage_recognition/main.py \
    --input-dir input/ \
    --quantize \
    --supabase-write
```

## First Run

### First Run - Model Download

The first time you run the analyzer, it will download the LLaVA model (~15GB):

```bash
python VLM_damage_recognition/main.py \
    --input-dir crack_detection/image_diff/input/ \
    --supabase-write
```

You'll see:
```
[LLaVA] Loading model: llava-hf/llava-1.5-7b-hf
# ... downloading model files ...
[LLaVA] Model loaded successfully
```

This may take 5-10 minutes on first run. Subsequent runs will be faster.

## Testing with Sample Images

### Option A: Use Existing Images

If you have images in `crack_detection/image_diff/input/`:
```bash
python VLM_damage_recognition/main.py \
    --input-dir crack_detection/image_diff/input/ \
    --output-json test_reports.jsonl
```

### Option B: Create Test Directory

```bash
mkdir -p test_images
# Copy some JPG or PNG images there
cp /path/to/damaged_building.jpg test_images/

python VLM_damage_recognition/main.py \
    --input-dir test_images/ \
    --output-json test_reports.jsonl
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Solution**: Install dependencies
```bash
pip install -r VLM_damage_recognition/requirements.txt
```

### Issue: "CUDA out of memory"

**Solutions**:
1. Use quantization on CPU:
```bash
python VLM_damage_recognition/main.py --input-dir input/ --quantize
```

2. Use smaller model:
```bash
python VLM_damage_recognition/main.py \
    --input-dir input/ \
    --model llava-hf/llava-1.5-7b-hf
```

3. Close other GPU applications

### Issue: "No images found"

**Solution**: Check image format and location
```bash
# Verify images exist
ls -lh crack_detection/image_diff/input/

# Check file type
file crack_detection/image_diff/input/image.jpg
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.geotiff`

### Issue: Slow Inference (CPU)

**Expected behavior**: CPU inference is slow (30-60s per image)

**Options**:
1. Use GPU if available
2. Use smaller model (7B instead of 13B)
3. Enable quantization (int8)

### Issue: "Supabase connection failed"

**Solution**: Verify credentials in `supabase_client.py`
```bash
# Check if supabase_client.py exists
ls -l ../supabase_client.py

# Run without Supabase write
python VLM_damage_recognition/main.py \
    --input-dir input/ \
    --output-json damage_reports.jsonl
```

## Usage Examples

### Example 1: Process drone images with coordinates

```bash
python VLM_damage_recognition/main.py \
    --image drone_photos/photo_001.jpg \
    --lat 34.765 \
    --lon 32.420 \
    --building "Residential Block A" \
    --drone-id "DJI-01" \
    --supabase-write
```

### Example 2: Batch with confidence filtering

```python
from VLM_damage_recognition import DamageAnalyzer, ImageProcessor, SupabaseReporter
from VLM_damage_recognition.utils import filter_reports_by_confidence

# Load and analyze
images = ImageProcessor.batch_load_images("input/")
analyzer = DamageAnalyzer()
reports = analyzer.analyze_batch(images)

# Filter high-confidence reports
high_conf = filter_reports_by_confidence(reports, min_confidence=0.7)

# Write to Supabase
reporter = SupabaseReporter(use_supabase=True)
reporter.write_batch(high_conf)
```

### Example 3: Generate reports in multiple formats

```python
from VLM_damage_recognition import ImageProcessor, DamageAnalyzer, SupabaseReporter
from VLM_damage_recognition.utils import reports_to_csv, reports_to_geojson

# Process images
images = ImageProcessor.batch_load_images("input/")
analyzer = DamageAnalyzer()
reports = analyzer.analyze_batch(images)

# Save in multiple formats
reporter = SupabaseReporter()
reporter.save_json_reports(reports, "damage_reports.jsonl")
reports_to_csv(reports, "damage_reports.csv")
reports_to_geojson(reports, "damage_reports.geojson")
reporter.write_batch(reports)  # Also to Supabase
```

## Next Steps

1. **Test with real images**: Process actual drone imagery
2. **Integrate with workflow**: Add to `tools.py` for swarm integration
3. **Fine-tune thresholds**: Adjust severity and confidence thresholds in `config.yaml`
4. **Monitor quality**: Review reports and feedback for model improvements
5. **Automate pipeline**: Set up batch processing with cron or triggers

## Advanced Configuration

Edit `VLM_damage_recognition/config.yaml` to customize:

```yaml
model:
  model_id: "llava-hf/llava-1.5-7b-hf"
  quantize: false
  max_new_tokens: 512

analysis:
  confidence_threshold: 0.3

output:
  supabase_write: true
  verbose: true
```

## Performance Benchmarks

| Hardware | Model | Quantized | Time/Image | VRAM |
|----------|-------|-----------|-----------|------|
| RTX 3090 | 7B    | No        | 2-3s      | 6GB  |
| RTX 3060 | 7B    | No        | 3-5s      | 6GB  |
| CPU i7   | 7B    | Yes (int8)| 30-60s    | 3GB  |
| CPU i7   | 7B    | No        | 120-300s  | 8GB  |

## Support & Documentation

- **README.md**: Comprehensive documentation
- **integration_example.py**: Code examples for integration
- **test_installation.py**: Installation verification
- **config.yaml**: Configuration reference

## Next Steps

1. Run installation test: `python VLM_damage_recognition/test_installation.py`
2. Process sample images: `python VLM_damage_recognition/main.py --input-dir test_images/`
3. Check output: `cat damage_reports.jsonl`
4. Integrate with system: See `integration_example.py`

Good luck! 🚀
