#!/usr/bin/env python3
"""
Integration Example

Shows how to integrate VLM damage recognition with the Earthquake system.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for supabase_client import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from damage_analyzer import DamageAnalyzer
from image_processor import ImageProcessor
from supabase_reporter import SupabaseReporter
from utils import print_summary, filter_reports_by_severity


def example_1_single_image():
    """Example 1: Analyze a single image and write to Supabase."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Image Analysis")
    print("="*60)

    # Initialize
    analyzer = DamageAnalyzer()
    reporter = SupabaseReporter(use_supabase=True)

    # Load image
    image_path = "crack_detection/image_diff/input/some_image.jpg"
    if not Path(image_path).exists():
        print(f"[SKIP] Test image not found: {image_path}")
        return

    image, coords = ImageProcessor.load_image(image_path)
    lat, lon = coords or (34.765, 32.420)

    # Analyze
    report = analyzer.analyze_image(
        image,
        lat=lat,
        lon=lon,
        building_name="Test Building",
        drone_id="JASS-DRONE-01"
    )

    # Write to Supabase
    ok = reporter.write_report(report)

    print(f"\nReport:")
    print(f"  Event ID: {report['event_id']}")
    print(f"  Location: ({report['lat']}, {report['lon']})")
    print(f"  Severity: {report['severity']}")
    print(f"  Damage: {report['damage_type']}")
    print(f"  Confidence: {report['confidence']:.2f}")
    print(f"  Supabase: {'Written' if ok else 'Failed'}")


def example_2_batch_processing():
    """Example 2: Batch process multiple images."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Processing")
    print("="*60)

    # Load all images
    images = ImageProcessor.batch_load_images("crack_detection/image_diff/input/")
    if not images:
        print("[SKIP] No images found in crack_detection/image_diff/input/")
        return

    # Analyze all
    analyzer = DamageAnalyzer()
    reports = analyzer.analyze_batch(images)

    # Write to Supabase
    reporter = SupabaseReporter(use_supabase=True)
    stats = reporter.write_batch(reports)

    print(f"\nResults:")
    print(f"  Total: {len(reports)}")
    print(f"  Success: {stats['success']}")
    print(f"  Failed: {stats['failed']}")

    # Summary
    print_summary(reports)


def example_3_integration_with_tools():
    """Example 3: Use as a tool in the Earthquake swarm system."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Integration with tools.py")
    print("="*60)

    print("""
To integrate with tools.py, add this function:

```python
# In tools.py
def analyze_drone_image(image_path: str) -> dict:
    '''Analyze drone image for structural damage using VLM.'''
    from VLM_damage_recognition import DamageAnalyzer, ImageProcessor

    analyzer = DamageAnalyzer()
    image, coords = ImageProcessor.load_image(image_path)

    if coords is None:
        # Use default Paphos location or query from metadata
        lat, lon = 34.765, 32.420
    else:
        lat, lon = coords

    report = analyzer.analyze_image(image, lat, lon)
    return report
```

Then agents can call:
```python
# In EarthAgent
damage_report = analyze_drone_image("path/to/drone/frame.jpg")
report_observation(f"Damage found: {damage_report['severity']}")
```
    """)


def example_4_filtering_reports():
    """Example 4: Filter and summarize reports."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Filtering Reports")
    print("="*60)

    # Load images
    images = ImageProcessor.batch_load_images("crack_detection/image_diff/input/")
    if not images:
        print("[SKIP] No images found")
        return

    # Analyze
    analyzer = DamageAnalyzer()
    reports = analyzer.analyze_batch(images)

    # Filter by severity
    high_severity = filter_reports_by_severity(reports, min_severity="high")

    print(f"\nAll Reports: {len(reports)}")
    print(f"High Severity Only: {len(high_severity)}")

    if high_severity:
        print("\nHigh Severity Reports:")
        for report in high_severity:
            print(f"  - {report['building']}: {report['damage_type']}")


def example_5_custom_prompts():
    """Example 5: Use custom prompts for specialized analysis."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Custom Prompts")
    print("="*60)

    from prompt_templates import DamagePrompts
    from inference import LLaVAInference
    from PIL import Image

    image_path = "crack_detection/image_diff/input/some_image.jpg"
    if not Path(image_path).exists():
        print(f"[SKIP] Test image not found: {image_path}")
        return

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Initialize LLaVA
    llava = LLaVAInference()

    # Example: Severity-focused analysis
    severity_prompt = DamagePrompts.severity_assessment()
    response = llava.analyze_image(image, severity_prompt)

    print(f"\nSeverity Assessment:")
    print(response[:300])


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("VLM DAMAGE RECOGNITION - INTEGRATION EXAMPLES")
    print("="*60)

    try:
        example_1_single_image()
        example_2_batch_processing()
        example_3_integration_with_tools()
        example_4_filtering_reports()
        example_5_custom_prompts()

        print("\n" + "="*60)
        print("Examples completed!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
