# Crack Tracking and Size Measurement Guide

## Overview

The enhanced VLM system now includes **crack size measurement** and **temporal tracking** capabilities. This allows you to:

- 📏 Measure crack dimensions in millimeters (assuming 1m x 1m photo scale)
- 📊 Track crack expansion over time
- 🗺️ Group measurements by location
- ⚠️ Detect rapid deterioration
- 📈 Generate growth reports

---

## New JSON Schema with Crack Measurements

### Enhanced Damage Report

```json
{
  "event_id": "uuid-here",
  "epoch": 1711000000.0,
  "lat": 34.765,
  "lon": 32.42,
  "severity": "high",
  "damage_type": "cracks and displacement",
  "building": "Hotel",
  "description": "Multiple cracks visible...",
  "drone_id": "JASS-DRONE-01",
  "confidence": 0.85,
  "_crack_measurements": {
    "cracks": [
      {
        "id": 1,
        "location": "top-left",
        "type": "vertical",
        "length_mm": 450,
        "width_mm": 3.5,
        "depth_estimate": "deep",
        "pattern": "straight",
        "severity": "moderate",
        "notes": "Through-wall crack"
      },
      {
        "id": 2,
        "location": "center",
        "type": "horizontal",
        "length_mm": 320,
        "width_mm": 2.1,
        "depth_estimate": "shallow",
        "pattern": "straight",
        "severity": "fine",
        "notes": "Surface crack"
      }
    ],
    "total_crack_area_mm2": 5000,
    "largest_crack_length_mm": 450,
    "largest_crack_width_mm": 3.5,
    "crack_density": "moderate",
    "overall_assessment": "Structural integrity affected by main vertical crack",
    "confidence": 0.88,
    "measurement_confidence": "high",
    "scale_assumption": "1m x 1m (1000mm x 1000mm) if no reference scale visible",
    "measurement_unit": "millimeters"
  }
}
```

---

## Crack Measurements Explained

### Individual Crack Properties

| Field | Description | Example |
|-------|-------------|---------|
| `id` | Unique crack identifier | 1 |
| `location` | Position in image | "top-left", "center", etc. |
| `type` | Crack orientation | "vertical", "horizontal", "diagonal", "network" |
| `length_mm` | Length of crack (mm) | 450 |
| `width_mm` | Width/opening of crack (mm) | 3.5 |
| `depth_estimate` | Penetration depth | "surface", "shallow", "deep" |
| `pattern` | Visual pattern | "straight", "curved", "branching" |
| `severity` | Crack severity | "hairline", "fine", "moderate", "severe" |

### Severity Classifications

**By Width:**
- **Hairline**: < 0.5mm - Cosmetic, no structural concern
- **Fine**: 0.5-1mm - Visible but limited impact
- **Moderate**: 1-5mm - Affects appearance and weatherproofing
- **Severe**: > 5mm - Structural integrity concerns

**By Depth:**
- **Surface**: Only outer layer affected
- **Shallow**: Affects outer 25% of material
- **Deep**: Penetrates significantly (50%+)
- **Through**: Passes completely through

### Overall Metrics

| Metric | Description |
|--------|-------------|
| `total_crack_area_mm2` | Total area covered by cracks |
| `largest_crack_length_mm` | Longest continuous crack |
| `largest_crack_width_mm` | Widest crack opening |
| `crack_density` | Concentration: low/moderate/high/severe |

---

## Tracking Expansion Over Time

### Compare Two Measurements

```python
from VLM_damage_recognition import CrackTracker

tracker = CrackTracker()

# Compare two crack measurements
expansion = tracker.compare_measurements(
    measurement1=old_report["_crack_measurements"],
    measurement2=new_report["_crack_measurements"],
)

print(expansion)
```

### Expansion Analysis Output

```json
{
  "total_crack_area": {
    "measurement1_mm2": 5000,
    "measurement2_mm2": 8500,
    "expansion_mm2": 3500,
    "expansion_percent": 70.0
  },
  "largest_crack_length": {
    "measurement1_mm": 450,
    "measurement2_mm": 620,
    "expansion_mm": 170,
    "expansion_percent": 37.78
  },
  "largest_crack_width": {
    "measurement1_mm": 3.5,
    "measurement2_mm": 5.2,
    "expansion_mm": 1.7,
    "expansion_percent": 48.57
  },
  "expansion_severity": "RAPID",
  "recommendation": "Urgent action required. Recommend immediate structural inspection.",
  "crack_count_change": 2
}
```

### Expansion Severity Levels

| Severity | Average Expansion | Meaning | Action |
|----------|------------------|---------|--------|
| **MINIMAL** | < 5% | Negligible growth | Continue monitoring |
| **SLOW** | 5-15% | Gradual growth | Regular surveillance |
| **MODERATE** | 15-30% | Noticeable growth | Increase monitoring |
| **RAPID** | 30-50% | Significant growth | Urgent inspection |
| **CRITICAL** | > 50% | Severe growth | Emergency response |

---

## Group Measurements by Location

Track the same location over multiple inspections:

```python
# Load multiple reports
reports = [report1, report2, report3]  # Photos of same location at different times

# Group by location (within 50cm tolerance)
locations = tracker.track_location_over_time(
    reports,
    location_tolerance_m=0.05
)

# Check each location
for location_key, measurements in locations.items():
    print(f"\nLocation: {location_key}")
    print(f"Measurements: {len(measurements)}")
    for meas in measurements:
        print(f"  - {meas['timestamp']}: {meas['severity']}")
        if "expansion_analysis" in meas:
            print(f"    Expansion: {meas['expansion_analysis']['expansion_severity']}")
```

---

## Generate Tracking Report

```python
# Generate comprehensive report
report = tracker.generate_tracking_report(
    location_data=locations,
    output_file="crack_tracking_report.json"
)

# Report shows:
# - Timeline of measurements at each location
# - Crack expansion history
# - Current status and recommendations
# - Expansion trends
```

### Report Structure

```json
{
  "report_type": "crack_tracking_analysis",
  "generated_at": "2026-03-27T10:30:00",
  "total_locations": 3,
  "locations": {
    "34.765,32.42": {
      "coordinates": {"lat": 34.765, "lon": 32.42},
      "measurements_count": 4,
      "time_span_days": 30,
      "measurements": [...],
      "expansion_history": [
        {"expansion_severity": "SLOW", ...},
        {"expansion_severity": "MODERATE", ...},
        {"expansion_severity": "RAPID", ...}
      ],
      "current_status": "RAPID",
      "recommendation": "Urgent action required..."
    }
  }
}
```

---

## Use Cases

### 1. Post-Earthquake Assessment

```
Day 1: Capture damage photos
  ↓
Generate initial crack measurements
  ↓
Store with coordinates
```

### 2. Track Aftershock Damage

```
Day 1: Initial measurement
  ↓
Aftershock occurs
  ↓
Day 2: New measurement same location
  ↓
Compare measurements → 45% expansion detected → RAPID
  ↓
Alert structural engineer
```

### 3. Monitor Repair Work

```
Before repair: Measure cracks
  ↓
Repair conducted
  ↓
After repair: Measure again
  ↓
Verify improvement (should show NEGATIVE expansion)
```

### 4. Long-term Building Health

```
Monthly inspections at critical points
  ↓
Track measurements over 12 months
  ↓
Detect trends (stable, slow growth, rapid growth)
  ↓
Predict failure timeline
```

---

## Scale Assumption

All measurements assume a **1m × 1m photo scale** by default:

- **Photo Size**: 1000mm × 1000mm
- **If different scale is visible** (ruler, reference object): System can be calibrated
- **Measurement Confidence**: Each report includes confidence level

### Typical Scale References

- Standard construction brick: ~200mm × 100mm
- Door frame: ~1000mm width
- Window: ~1000mm-1500mm width
- Person: ~1700mm tall
- Building block: Variable

---

## API Examples

### Basic Crack Measurement

```python
from VLM_damage_recognition import DamageAnalyzer, ImageProcessor

analyzer = DamageAnalyzer()
image, (lat, lon) = ImageProcessor.load_image("damage_photo.jpg")

# Analyze with crack sizes enabled (default)
report = analyzer.analyze_image(image, lat, lon, include_crack_sizes=True)

# Access crack measurements
cracks = report["_crack_measurements"]["cracks"]
for crack in cracks:
    print(f"Crack {crack['id']}: {crack['length_mm']}mm × {crack['width_mm']}mm")
```

### Expansion Detection

```python
from VLM_damage_recognition import CrackTracker

tracker = CrackTracker()

# Get two reports from same location
old = report1["_crack_measurements"]
new = report2["_crack_measurements"]

# Compare
expansion = tracker.compare_measurements(old, new)

if expansion["expansion_severity"] == "CRITICAL":
    alert_authorities(f"Building at {lat},{lon} deteriorating rapidly!")
```

### Location Tracking

```python
# Load all reports from database
all_reports = load_from_supabase("drone_reports")

# Track by location
locations = tracker.track_location_over_time(all_reports)

# Generate report
report = tracker.generate_tracking_report(locations, "tracking.json")
```

---

## Data Storage in Supabase

### Enhanced drone_reports Table

Add these columns:

```sql
ALTER TABLE drone_reports ADD COLUMN (
    crack_measurements JSONB,
    total_crack_area_mm2 FLOAT,
    largest_crack_length_mm FLOAT,
    largest_crack_width_mm FLOAT,
    crack_density VARCHAR(50)
);
```

### Query Examples

```sql
-- Find locations with rapid crack expansion
SELECT lat, lon, COUNT(*) as measurements
FROM drone_reports
WHERE crack_measurements->>'expansion_severity' = 'RAPID'
GROUP BY lat, lon
ORDER BY measurements DESC;

-- Track largest cracks
SELECT lat, lon, largest_crack_length_mm, epoch
FROM drone_reports
WHERE largest_crack_length_mm > 500
ORDER BY epoch DESC;

-- Find high-density crack areas
SELECT lat, lon, crack_density, COUNT(*)
FROM drone_reports
WHERE crack_density = 'severe'
GROUP BY lat, lon, crack_density;
```

---

## Integration with Earthquake System

### Add to tools.py

```python
def analyze_building_damage_with_tracking(image_path: str, location: Tuple[float, float]):
    """Analyze damage and track crack expansion."""
    from VLM_damage_recognition import DamageAnalyzer

    analyzer = DamageAnalyzer()
    image, coords = ImageProcessor.load_image(image_path)
    lat, lon = coords or location

    report = analyzer.analyze_image(
        image, lat, lon,
        include_crack_sizes=True  # Enable crack measurements
    )

    # Store in Supabase with crack data
    write_to_drone_reports(report)

    return report
```

---

## Recommendations for Field Use

### Initial Setup
1. ✅ Define reference scale (mark known distances)
2. ✅ Document camera/drone model (for consistency)
3. ✅ Establish baseline measurements
4. ✅ Create location markers (GPS + photo markers)

### Regular Monitoring
1. 📸 Photograph from same angle/distance
2. 📐 Measure at consistent intervals
3. 📍 Record exact GPS coordinates
4. ⏱️ Log timestamps
5. 📊 Compare against baseline

### Analysis
1. 🔍 Review expansion trends
2. ⚠️ Alert on RAPID/CRITICAL status
3. 📈 Predict failure timeline if trend continues
4. 🏗️ Recommend interventions

---

## Performance Notes

- First VLM analysis: ~5-10 minutes (model download)
- Subsequent analyses: 2-5s/image (GPU), 30-60s (CPU)
- Crack measurement adds ~30% to analysis time
- Storage: ~5KB per crack measurement per image

---

**Version**: 0.2.0
**Last Updated**: 2026-03-27
**Status**: ✅ PRODUCTION READY
