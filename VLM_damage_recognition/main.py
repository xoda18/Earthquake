#!/usr/bin/env python3
"""
VLM Damage Recognition - Main CLI

Analyze structural damage in drone imagery using LLaVA Vision Language Model.

Usage:
    # Single image
    python VLM_damage_recognition/main.py --image /path/to/image.jpg --lat 34.765 --lon 32.420

    # Batch directory
    python VLM_damage_recognition/main.py --input-dir crack_detection/image_diff/input/ --supabase-write

    # CPU-only with quantization
    python VLM_damage_recognition/main.py --input-dir input/ --quantize --supabase-write

    # Output to JSON only
    python VLM_damage_recognition/main.py --input-dir input/ --output-json damage_reports.jsonl
"""

import argparse
import sys
from pathlib import Path

from image_processor import ImageProcessor
from damage_analyzer import DamageAnalyzer
from supabase_reporter import SupabaseReporter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VLM Damage Recognition - Analyze structural damage in drone imagery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        type=str,
        help="Single image file path"
    )
    input_group.add_argument(
        "--input-dir",
        type=str,
        help="Directory containing images to batch process"
    )

    # Coordinates (for single image)
    parser.add_argument("--lat", type=float, help="Latitude (required with --image)")
    parser.add_argument("--lon", type=float, help="Longitude (required with --image)")

    # Metadata
    parser.add_argument(
        "--building",
        type=str,
        default=None,
        help="Building name/type"
    )
    parser.add_argument(
        "--drone-id",
        type=str,
        default="JASS-DRONE-01",
        help="Drone identifier"
    )

    # Output options
    parser.add_argument(
        "--supabase-write",
        action="store_true",
        help="Write reports to Supabase"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Save reports to JSONL file"
    )

    # Model options
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Use int8 quantization (recommended for CPU)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="HuggingFace model ID"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("\n" + "="*60)
    print("VLM DAMAGE RECOGNITION")
    print("="*60)

    # Initialize analyzer
    print("\n[Init] Loading LLaVA model...")
    try:
        analyzer = DamageAnalyzer(quantize=args.quantize, model_id=args.model)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    # Initialize reporter
    reporter = SupabaseReporter(use_supabase=args.supabase_write)

    reports = []

    # Single image mode
    if args.image:
        if args.lat is None or args.lon is None:
            print("[ERROR] --lat and --lon required with --image")
            sys.exit(1)

        print(f"\n[Process] Single image: {args.image}")
        try:
            image, coords = ImageProcessor.load_image(args.image)
            if coords:
                lat, lon = coords
            else:
                lat, lon = args.lat, args.lon
                print(f"  Using provided coordinates: ({lat}, {lon})")

            report = analyzer.analyze_image(
                image,
                lat=lat,
                lon=lon,
                building_name=args.building,
                drone_id=args.drone_id
            )
            reports.append(report)

            print(f"  Damage: {report['damage_type']}")
            print(f"  Severity: {report['severity']}")
            print(f"  Confidence: {report['confidence']:.2f}")

            # Write to Supabase
            if args.supabase_write:
                reporter.write_report(report)

        except Exception as e:
            print(f"[ERROR] Analysis failed: {e}")
            sys.exit(1)

    # Batch mode
    elif args.input_dir:
        print(f"\n[Process] Batch: {args.input_dir}")
        images = ImageProcessor.batch_load_images(args.input_dir)

        if not images:
            print("[WARN] No images found")
            sys.exit(0)

        print(f"[OK] Found {len(images)} images\n")

        try:
            reports = analyzer.analyze_batch(images, drone_id=args.drone_id)
        except Exception as e:
            print(f"[ERROR] Batch analysis failed: {e}")
            sys.exit(1)

        # Write to Supabase
        if args.supabase_write:
            stats = reporter.write_batch(reports)
            print(f"\n[Summary] Success: {stats['success']}, Failed: {stats['failed']}")

    # Save JSON output
    if args.output_json and reports:
        reporter.save_json_reports(reports, args.output_json)

    # Final summary
    print("\n" + "="*60)
    print(f"Processed: {len(reports)} images")
    if reports:
        print(f"Severity breakdown:")
        severity_counts = {}
        for report in reports:
            sev = report['severity']
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        for sev, count in sorted(severity_counts.items()):
            print(f"  {sev:>10}: {count}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
