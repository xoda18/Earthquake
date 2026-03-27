# VLM Damage Recognition - Module Manifest

Complete documentation of all files in the VLM damage recognition module.

## Module Overview

**Purpose**: Vision Language Model-based structural damage recognition for drone imagery
**Tech Stack**: LLaVA (HuggingFace), PyTorch, Transformers
**Integration**: Standalone module with Supabase connectivity

## File Structure

```
VLM_damage_recognition/
├── __init__.py                 # Package initialization
├── __main__.py                 # Package entry point
├── main.py                     # CLI interface (primary entry point)
├── requirements.txt            # Python dependencies
├── config.yaml                 # Configuration settings
├── README.md                   # Main documentation
├── SETUP_GUIDE.md             # Installation & quick start guide
├── MANIFEST.md                # This file
├── test_installation.py        # Installation verification script
├── integration_example.py      # Integration examples with Earthquake system
│
├── Core Modules:
├── inference.py               # LLaVA model wrapper & inference
├── image_processor.py         # Multi-format image loading & geo-extraction
├── damage_analyzer.py         # Main analysis pipeline
├── prompt_templates.py        # Structured prompts for damage analysis
├── supabase_reporter.py       # Supabase database integration
└── utils.py                   # Utility functions (filtering, export, etc.)
```

## Core Modules Reference

### 1. **inference.py** - LLaVA Model Inference
**Purpose**: Load and run LLaVA model inference

**Key Classes**:
- `LLaVAInference`: Main model wrapper
  - `__init__(model_id, quantize)`: Initialize with model and options
  - `analyze_image(image, prompt, max_new_tokens)`: Run inference on image
  - `_detect_device()`: Auto-detect GPU/CPU/MPS

**Usage**:
```python
from VLM_damage_recognition.inference import LLaVAInference

llava = LLaVAInference(quantize=False)
response = llava.analyze_image(image, prompt)
```

**Dependencies**:
- `torch`, `transformers`, `PIL`

---

### 2. **image_processor.py** - Multi-Format Image Loading
**Purpose**: Load and process images with geolocation extraction

**Key Classes**:
- `ImageProcessor`: Static methods for image handling
  - `load_image(file_path)`: Load any format, extract geo
  - `batch_load_images(directory)`: Load all images from directory
  - `generate_paphos_coordinates()`: Random fallback coords
  - `_extract_exif_geolocation()`: Parse EXIF GPS
  - `_load_geotiff()`: GeoTIFF with rasterio

**Supported Formats**:
- JPEG/JPG (with EXIF GPS)
- PNG (with EXIF GPS)
- TIFF/TIF
- GeoTIFF (with geolocation)

**Usage**:
```python
from VLM_damage_recognition.image_processor import ImageProcessor

image, (lat, lon) = ImageProcessor.load_image("image.jpg")
images = ImageProcessor.batch_load_images("input_dir/")
```

**Dependencies**:
- `PIL`, `piexif`, `rasterio` (optional)

---

### 3. **damage_analyzer.py** - Analysis Pipeline
**Purpose**: Main pipeline orchestrating analysis

**Key Classes**:
- `DamageAnalyzer`: Coordinates image analysis
  - `__init__(quantize, model_id)`: Initialize with LLaVA
  - `analyze_image(image, lat, lon, building_name, drone_id)`: Single image
  - `analyze_batch(image_list, drone_id)`: Multiple images

**Output Format**:
```python
{
  "event_id": "uuid",
  "epoch": 1711000000.0,
  "lat": 34.765,
  "lon": 32.420,
  "severity": "high",
  "damage_type": "crack, displacement",
  "building": "Hotel",
  "description": "...",
  "drone_id": "JASS-DRONE-01",
  "confidence": 0.87,
  "_vlm_analysis": {...}
}
```

**Usage**:
```python
from VLM_damage_recognition import DamageAnalyzer

analyzer = DamageAnalyzer()
report = analyzer.analyze_image(image, lat, lon)
```

---

### 4. **prompt_templates.py** - Damage Analysis Prompts
**Purpose**: Structured prompts for consistent LLaVA outputs

**Key Classes**:
- `DamagePrompts`: Static prompt templates
  - `structural_damage_analysis()`: Comprehensive analysis
  - `damage_type_detection()`: Damage classification
  - `severity_assessment()`: Severity-focused
  - `building_type_detection()`: Structure type
  - `parse_json_response()`: Extract JSON from LLaVA output
  - `normalize_severity()`: Standardize severity strings
  - `validate_damage_response()`: Check required fields

**Usage**:
```python
from VLM_damage_recognition.prompt_templates import DamagePrompts

prompt = DamagePrompts.structural_damage_analysis()
response = llava.analyze_image(image, prompt)
parsed = DamagePrompts.parse_json_response(response)
```

---

### 5. **supabase_reporter.py** - Database Integration
**Purpose**: Write damage reports to Supabase

**Key Classes**:
- `SupabaseReporter`: Handles database operations
  - `__init__(use_supabase)`: Initialize connection
  - `write_report(report)`: Single report to database
  - `write_batch(reports)`: Multiple reports
  - `save_json_reports(reports, path)`: Export to JSONL

**Usage**:
```python
from VLM_damage_recognition import SupabaseReporter

reporter = SupabaseReporter(use_supabase=True)
reporter.write_report(report)
reporter.save_json_reports(reports, "output.jsonl")
```

**Dependencies**:
- `supabase_client.py` (from parent directory)

---

### 6. **utils.py** - Utility Functions
**Purpose**: Helper functions for report processing

**Key Functions**:
- `reports_to_csv(reports, path)`: Export to CSV
- `reports_to_geojson(reports, path)`: Export to GeoJSON
- `load_jsonl_reports(path)`: Load from JSONL
- `filter_reports_by_severity(reports, min_severity)`: Filter
- `filter_reports_by_confidence(reports, min_confidence)`: Filter
- `summary_statistics(reports)`: Generate stats
- `print_summary(reports)`: Pretty-print summary

**Usage**:
```python
from VLM_damage_recognition.utils import filter_reports_by_severity, print_summary

high_severity = filter_reports_by_severity(reports, "high")
print_summary(reports)
```

---

## CLI & Entry Points

### **main.py** - Command Line Interface
**Purpose**: Primary user interface

**Key Commands**:
```bash
# Single image
python main.py --image path.jpg --lat 34.765 --lon 32.420

# Batch directory
python main.py --input-dir input/ --supabase-write

# CPU quantized
python main.py --input-dir input/ --quantize --supabase-write

# Output to JSON
python main.py --input-dir input/ --output-json reports.jsonl
```

**Options**:
- `--image`: Single image path
- `--input-dir`: Directory for batch processing
- `--lat, --lon`: Coordinates (required with --image)
- `--building`: Building name/type
- `--drone-id`: Drone identifier (default: JASS-DRONE-01)
- `--supabase-write`: Write to Supabase
- `--output-json`: Save to JSONL file
- `--quantize`: Use int8 quantization
- `--model`: HuggingFace model ID

---

### **__main__.py** - Package Entry Point
**Purpose**: Allow running as module

**Usage**:
```bash
python -m VLM_damage_recognition --help
```

---

## Test & Verification

### **test_installation.py** - Installation Verification
**Purpose**: Verify all dependencies and components

**Checks**:
- ✓ Required imports (torch, transformers, PIL, etc.)
- ✓ Optional imports (rasterio)
- ✓ VLM_damage_recognition modules
- ✓ Device detection (GPU/CPU)
- ✓ Supabase connectivity

**Usage**:
```bash
python test_installation.py
```

---

### **integration_example.py** - Integration Examples
**Purpose**: Show how to use VLM module with Earthquake system

**Examples**:
1. Single image analysis
2. Batch processing
3. Integration with tools.py
4. Report filtering
5. Custom prompts

**Usage**:
```bash
python integration_example.py
```

---

## Configuration & Documentation

### **config.yaml** - Configuration File
**Purpose**: Model and analysis settings

**Sections**:
- `model`: Model ID, quantization, token limits
- `analysis`: Confidence thresholds, sampling parameters
- `severity`: Damage severity thresholds
- `output`: Database and logging options
- `drone`: Drone metadata defaults
- `input`: Supported formats
- `paphos_zones`: Default coordinate zones

---

### **README.md** - Main Documentation
**Purpose**: Comprehensive user guide

**Sections**:
- Features overview
- Installation instructions
- Usage examples
- Python API reference
- Output format specification
- Performance benchmarks
- Troubleshooting guide
- Integration examples

---

### **SETUP_GUIDE.md** - Quick Start Guide
**Purpose**: Step-by-step installation and first run

**Sections**:
- Prerequisites
- Installation steps
- Quick start examples
- First run instructions
- Troubleshooting
- Performance benchmarks
- Usage examples

---

### **requirements.txt** - Python Dependencies
**Purpose**: Pip dependency specification

**Key Packages**:
- torch>=2.0.0: PyTorch core
- transformers>=4.35.0: HuggingFace models
- Pillow>=10.0.0: Image processing
- piexif>=1.1.3: EXIF metadata
- rasterio>=1.3.0: GeoTIFF support
- requests>=2.31.0: HTTP client
- pydantic>=2.0.0: Data validation

---

## Integration Points with Earthquake System

### 1. **Supabase Connection** (via `supabase_client.py`)
- Writes to `drone_reports` table
- Maintains schema compatibility with mock_damage.py

### 2. **Image Input** (via `crack_detection/image_diff/`)
- Can process output from image_diff pipeline
- Supports same image formats

### 3. **Agent Integration** (via `tools.py`)
- Can be called as a tool function
- Returns structured damage reports
- Integrates with EarthAgent workflow

---

## Data Flow

```
User Input (CLI or Python API)
    ↓
[image_processor.py] Load & extract geolocation
    ↓
[damage_analyzer.py] Prepare analysis
    ↓
[inference.py] Run LLaVA model
    ↓
[prompt_templates.py] Parse structured output
    ↓
[damage_analyzer.py] Format report
    ↓
[supabase_reporter.py] Write to database
    ↓
[utils.py] Export/filter results
```

---

## Deployment Checklist

- [x] Core modules implemented (inference, image_processor, damage_analyzer)
- [x] Prompt templates for damage analysis
- [x] Supabase integration
- [x] CLI interface (main.py)
- [x] Batch processing support
- [x] GPU/CPU auto-detection
- [x] Multi-format image support (JPG, PNG, TIFF, GeoTIFF)
- [x] Geolocation extraction
- [x] Confidence scoring
- [x] Severity classification
- [x] Utility functions (export, filter, stats)
- [x] Installation test script
- [x] Integration examples
- [x] Comprehensive documentation
- [x] Configuration system

---

## Version Information

- **Module Version**: 0.1.0
- **LLaVA Model**: llava-hf/llava-1.5-7b-hf (default)
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **Transformers**: 4.35+

---

## Support & Troubleshooting

See **SETUP_GUIDE.md** for:
- Installation troubleshooting
- Performance optimization
- GPU/CPU configuration
- Common error solutions

See **README.md** for:
- API reference
- Advanced usage
- Integration patterns
- Performance benchmarks

See **integration_example.py** for:
- Code examples
- Earthquake system integration
- Report processing workflows
- Custom analysis

---

## Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Test installation: `python test_installation.py`
3. Run quick example: `python main.py --help`
4. Process images: `python main.py --input-dir input/ --supabase-write`
5. Integrate with tools.py: See `integration_example.py`
6. Monitor results: Query Supabase drone_reports table

---

Generated: 2026-03-27
