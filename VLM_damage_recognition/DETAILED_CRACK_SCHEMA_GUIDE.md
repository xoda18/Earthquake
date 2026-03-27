# Detailed Crack Tracking Schema Guide

## Overview

The enhanced damage report schema now includes **detailed per-crack measurements** with location coordinates, individual severity and status assessment, and comprehensive summary statistics. This enables sophisticated analysis of crack expansion, location-based tracking, and temporal comparison.

---

## New Standardized Report Structure

### Top-Level Report Schema

```json
{
  "file": "image.jpg",
  "lat": 34.765,
  "lon": 32.42,
  "severity": "high",              // Overall severity of the image
  "status": "growing",             // Overall status: stable/growing/recovering/unknown
  "confidence": 0.85,              // Overall confidence (0.0-1.0)
  "damage_type": "structural cracks",
  "description": "Multiple cracks with varying expansion rates",
  "drone_id": "JASS-DRONE-01",
  "building": "Hotel Name",
  "event_id": "uuid-here",
  "epoch": 1711000000.0,
  "timestamp": "2026-03-27T10:00:00",

  // NEW: Individual crack array
  "cracks": [ ... ],

  // NEW: Summary statistics
  "_summary_statistics": { ... }
}
```

---

## Per-Crack Detailed Measurements

### Crack Object Structure

Each crack in the `cracks` array contains:

```json
{
  "id": 1,                         // Unique crack identifier (1, 2, 3, ...)
  "location": "top-left",          // Region in image (9 regions)

  "measurements": {
    "length_mm": 450,              // Crack length in millimeters
    "width_mm": 3.5,               // Crack opening width in millimeters
    "depth_estimate": "deep",      // surface|shallow|deep|unknown
    "area_mm2": 1575,              // Estimated area in mm²
    "pattern": "straight"          // straight|curved|branching|network
  },

  "severity": "high",              // Per-crack severity level
  "status": "growing",             // Per-crack status (temporal indicator)
  "confidence": 0.88,              // Measurement confidence (0.0-1.0)
  "description": "Vertical crack running from top to middle",

  // Optional: Location in image coordinates
  "normalized_coords": {
    "x": 0.15,                     // 0.0-1.0 (image width ratio)
    "y": 0.25                      // 0.0-1.0 (image height ratio)
  },

  // Optional: Pixel coordinates (if available)
  "pixel_coords": {
    "x": 150,                      // Pixel X coordinate
    "y": 250                       // Pixel Y coordinate
  }
}
```

### Location Regions (9-Point Grid)

The image is divided into a 3×3 grid:

```
top-left      | top-center     | top-right
(0-0.33,0-0.33)  (0.33-0.66,0-0.33)  (0.66-1.0,0-0.33)
─────────────────────────────────────────────────
middle-left   | center         | middle-right
(0-0.33,0.33-0.66) (0.33-0.66,0.33-0.66) (0.66-1.0,0.33-0.66)
─────────────────────────────────────────────────
bottom-left   | bottom-center  | bottom-right
(0-0.33,0.66-1.0)  (0.33-0.66,0.66-1.0)  (0.66-1.0,0.66-1.0)
```

### Measurement Guidelines (Assuming 1m × 1m = 1000mm × 1000mm)

| Property | Range | Description |
|----------|-------|-------------|
| **length_mm** | 1-1000+ | Crack length along its longest axis |
| **width_mm** | 0.1-10+ | Opening width (visible gap) |
| **depth_estimate** | 4 levels | Surface/shallow/deep/unknown |
| **area_mm2** | 0-1,000,000+ | Estimated area (length × width) |
| **pattern** | 4 types | Straight/curved/branching/network |

### Severity Levels (Per-Crack)

```
"low"       → Minor cracks, < 1mm width, surface only
"moderate"  → Visible cracks, 1-5mm width, affecting outer layer
"high"      → Significant cracks, > 5mm width, penetrating material
"critical"  → Severe cracks, very wide (> 8mm), risk of structural failure
```

### Status Values (Per-Crack)

Each crack can have a temporal status indicating change:

```
"stable"      → Crack shows no signs of expansion (measuring same as before)
"growing"     → Crack is expanding or widening (getting worse)
"recovering"  → Crack is healing or closing (improving)
"unknown"     → Cannot determine status from this image alone
```

---

## Summary Statistics (_summary_statistics)

Automatically calculated from the cracks array:

```json
{
  "total_cracks": 3,
  "total_crack_area_mm2": 5000,
  "largest_crack_length_mm": 450,
  "largest_crack_width_mm": 3.5,
  "average_crack_length_mm": 320,
  "average_crack_width_mm": 2.8,
  "crack_density": "moderate",    // minimal|low|moderate|high|severe
  "overall_severity": "high",     // Highest severity from all cracks
  "measurement_unit": "millimeters",
  "scale_assumption": "1m x 1m (1000mm x 1000mm)"
}
```

### Crack Density Assessment (Based on 1m × 1m = 1,000,000 mm²)

| Density | Condition | % of Area |
|---------|-----------|-----------|
| **minimal** | Very few small cracks | < 0.3% |
| **low** | Some isolated cracks | 0.3-1.0% |
| **moderate** | Noticeable network | 1.0-3.0% |
| **high** | Widespread cracks | 3.0-8.0% |
| **severe** | Extensive cracking | > 8.0% |

---

## Usage Examples

### 1. Load and Analyze Single Report

```python
from VLM_damage_recognition import DamageReportSchema
import json

# Load report
with open("report.json", "r") as f:
    report = json.load(f)

# Access overall information
print(f"Building: {report['building']}")
print(f"Overall Severity: {report['severity']}")
print(f"Overall Status: {report['status']}")

# Analyze cracks
print(f"\nFound {len(report['cracks'])} cracks:")
for crack in report['cracks']:
    print(f"  Crack #{crack['id']} ({crack['location']})")
    print(f"    Length: {crack['measurements']['length_mm']}mm")
    print(f"    Width: {crack['measurements']['width_mm']}mm")
    print(f"    Status: {crack['status']}")
```

### 2. Compare Cracks at Specific Location

```python
from VLM_damage_recognition import CrackTracker

tracker = CrackTracker()

# Load two reports from same location at different times
report_day1 = load_report("day1.json")
report_day30 = load_report("day30.json")

# Compare cracks in specific region
analysis = tracker.compare_cracks_at_location(
    cracks1=report_day1.get("cracks", []),
    cracks2=report_day30.get("cracks", []),
    location_region="center"
)

print(f"Expansion Severity: {analysis['expansion_severity']}")
print(f"Area Expansion: {analysis['area_expansion_percent']}%")
print(f"Recommendation: {analysis['recommendation']}")
```

### 3. Find Growing Cracks

```python
report = load_report("latest_report.json")

growing_cracks = [c for c in report['cracks'] if c['status'] == 'growing']

if growing_cracks:
    print(f"⚠️  WARNING: {len(growing_cracks)} growing cracks detected!")
    for crack in growing_cracks:
        print(f"  Location: {crack['location']}")
        print(f"  Size: {crack['measurements']['length_mm']}mm × {crack['measurements']['width_mm']}mm")
        print(f"  Depth: {crack['measurements']['depth_estimate']}")
```

### 4. Identify Most Critical Cracks

```python
report = load_report("report.json")

# Sort by severity and size
critical_cracks = sorted(
    report['cracks'],
    key=lambda c: (
        {"low": 0, "moderate": 1, "high": 2, "critical": 3}.get(c['severity'], 0),
        c['measurements']['width_mm']
    ),
    reverse=True
)

print("Top 3 Critical Cracks:")
for crack in critical_cracks[:3]:
    print(f"  Crack #{crack['id']}: {crack['severity']} ({crack['location']})")
```

### 5. Track Crack Distribution by Region

```python
report = load_report("report.json")

# Group cracks by region
regions = {}
for crack in report['cracks']:
    location = crack['location']
    if location not in regions:
        regions[location] = []
    regions[location].append(crack)

print("Crack Distribution:")
for region, cracks in sorted(regions.items()):
    total_area = sum(c['measurements']['area_mm2'] for c in cracks)
    print(f"  {region}: {len(cracks)} cracks, {total_area}mm² total area")
```

### 6. Create Report with Detailed Cracks

```python
from VLM_damage_recognition import DamageReportSchema

# Create individual crack objects
crack1 = DamageReportSchema.create_crack(
    crack_id=1,
    location="top-left",
    measurements={
        "length_mm": 450,
        "width_mm": 3.5,
        "depth_estimate": "deep",
        "area_mm2": 1575,
        "pattern": "straight"
    },
    severity="high",
    status="growing",
    confidence=0.88,
    description="Vertical crack with progressive widening",
    normalized_coords={"x": 0.15, "y": 0.25}
)

# Create report with cracks
report = DamageReportSchema.create_report(
    file="building_photo.jpg",
    lat=34.765,
    lon=32.42,
    severity="high",
    confidence=0.88,
    damage_type="structural cracks",
    description="Multiple vertical cracks in load-bearing walls",
    status="growing",
    building="Historic Building A",
    cracks=[crack1]  # Pass cracks array
)

# Summary statistics are auto-calculated
print(report['_summary_statistics'])
```

### 7. Export for Supabase

```python
import supabase_client as sb
import json

report = load_report("report.json")

# Insert into drone_reports table
result = sb.insert("drone_reports", {
    "file": report["file"],
    "lat": report["lat"],
    "lon": report["lon"],
    "severity": report["severity"],
    "status": report["status"],
    "confidence": report["confidence"],
    "damage_type": report["damage_type"],
    "description": report["description"],
    "drone_id": report["drone_id"],
    "building": report["building"],
    # Store detailed crack data as JSONB
    "crack_measurements": json.dumps(report["cracks"]),
    "crack_count": len(report["cracks"]),
    "total_crack_area_mm2": report["_summary_statistics"]["total_crack_area_mm2"],
    "crack_density": report["_summary_statistics"]["crack_density"]
})
```

### 8. Temporal Expansion Tracking

```python
from VLM_damage_recognition import CrackTracker

tracker = CrackTracker()

# Load reports from multiple inspections
reports = [
    load_report("inspection_2026-03-20.json"),
    load_report("inspection_2026-03-27.json"),
    load_report("inspection_2026-04-03.json"),
]

# Track changes over time
locations = tracker.track_location_over_time(reports)

for location, measurements in locations.items():
    print(f"\nLocation {location}:")
    for i, meas in enumerate(measurements):
        print(f"  Measurement {i+1}: {len(meas['crack_measurements']['cracks'])} cracks")
        if 'expansion_analysis' in meas:
            exp = meas['expansion_analysis']
            print(f"    Expansion: {exp['expansion_severity']}")
```

---

## Database Schema (Supabase)

### Recommended table structure:

```sql
CREATE TABLE drone_reports (
    id BIGSERIAL PRIMARY KEY,
    event_id UUID NOT NULL,
    file VARCHAR NOT NULL,
    lat FLOAT NOT NULL,
    lon FLOAT NOT NULL,
    severity VARCHAR(20),
    status VARCHAR(20),
    confidence FLOAT,
    damage_type VARCHAR,
    description TEXT,
    drone_id VARCHAR,
    building VARCHAR,

    -- Crack measurements (JSONB for flexibility)
    crack_measurements JSONB,
    crack_count INT,
    total_crack_area_mm2 FLOAT,
    largest_crack_length_mm FLOAT,
    crack_density VARCHAR(20),

    epoch FLOAT,
    timestamp TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Index for location-based queries
CREATE INDEX idx_location ON drone_reports(lat, lon);

-- Index for temporal queries
CREATE INDEX idx_timestamp ON drone_reports(timestamp);

-- Index for JSONB queries on cracks
CREATE INDEX idx_cracks ON drone_reports USING GIN(crack_measurements);
```

### Useful Queries

```sql
-- Find all growing cracks
SELECT event_id, lat, lon, crack_measurements
FROM drone_reports
WHERE crack_measurements->>'status' = 'growing'
ORDER BY timestamp DESC;

-- Find cracks wider than 5mm (critical)
SELECT event_id, lat, lon, largest_crack_length_mm, largest_crack_width_mm
FROM drone_reports
WHERE largest_crack_width_mm > 5.0
ORDER BY largest_crack_width_mm DESC;

-- Track expansion at specific location
SELECT timestamp, total_crack_area_mm2, crack_density, crack_count
FROM drone_reports
WHERE ABS(lat - 34.765) < 0.001 AND ABS(lon - 32.420) < 0.001
ORDER BY timestamp;

-- Severe cracking density
SELECT lat, lon, COUNT(*) as reports, AVG(total_crack_area_mm2) as avg_area
FROM drone_reports
WHERE crack_density IN ('high', 'severe')
GROUP BY lat, lon
ORDER BY avg_area DESC;
```

---

## Backwards Compatibility

Old format (with `_crack_measurements`) is still supported:

```json
{
  "file": "image.jpg",
  "severity": "high",
  "_crack_measurements": {
    "cracks": [...],
    "total_crack_area_mm2": 5000
  }
}
```

The system will convert this to the new format automatically.

---

## Standardization Benefits

✅ **Per-crack granularity** - Track each crack individually
✅ **Location-based analysis** - Know where problems are in structure
✅ **Temporal tracking** - Detect growing vs. stable vs. recovering
✅ **Quantitative comparison** - Real measurements, not subjective descriptions
✅ **Database compatible** - Store and query efficiently
✅ **API ready** - Integrate with rest services
✅ **Extensible** - Add custom fields as needed

---

## Example Output

See `test_results/analysis_metadata.json` for a complete example with multiple damage reports showing:
- Different severity levels (low, moderate, high, critical)
- Multiple cracks per image
- Location coordinates
- Per-crack severity and status
- Summary statistics

---

**Version**: 0.3.0
**Status**: ✅ PRODUCTION READY
**Date**: 2026-03-27
