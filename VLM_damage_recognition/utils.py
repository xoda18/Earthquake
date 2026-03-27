"""
Utility functions for VLM damage recognition.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any


def reports_to_csv(reports: List[Dict[str, Any]], output_path: str) -> bool:
    """
    Convert damage reports to CSV format.

    Args:
        reports: List of damage report dicts
        output_path: Path to output CSV file

    Returns:
        True if successful
    """
    if not reports:
        return False

    try:
        with open(output_path, "w", newline="") as f:
            # Use main fields only (exclude _vlm_analysis)
            fieldnames = [
                "event_id", "epoch", "lat", "lon", "severity",
                "damage_type", "building", "description",
                "drone_id", "confidence"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for report in reports:
                row = {field: report.get(field, "") for field in fieldnames}
                writer.writerow(row)

        print(f"[OK] Saved {len(reports)} reports to CSV: {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save CSV: {e}")
        return False


def reports_to_geojson(reports: List[Dict[str, Any]], output_path: str) -> bool:
    """
    Convert damage reports to GeoJSON format.

    Args:
        reports: List of damage report dicts
        output_path: Path to output GeoJSON file

    Returns:
        True if successful
    """
    try:
        features = []
        for report in reports:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [report["lon"], report["lat"]]
                },
                "properties": {
                    "event_id": report["event_id"],
                    "epoch": report["epoch"],
                    "severity": report["severity"],
                    "damage_type": report["damage_type"],
                    "building": report["building"],
                    "description": report["description"],
                    "confidence": report["confidence"],
                    "drone_id": report["drone_id"],
                }
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        with open(output_path, "w") as f:
            json.dump(geojson, f, indent=2)

        print(f"[OK] Saved {len(reports)} reports to GeoJSON: {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save GeoJSON: {e}")
        return False


def load_jsonl_reports(file_path: str) -> List[Dict[str, Any]]:
    """
    Load damage reports from JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of report dicts
    """
    reports = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    reports.append(json.loads(line))
        print(f"[OK] Loaded {len(reports)} reports from {file_path}")
        return reports
    except Exception as e:
        print(f"[ERROR] Failed to load JSONL: {e}")
        return []


def filter_reports_by_severity(
    reports: List[Dict[str, Any]],
    min_severity: str = "moderate"
) -> List[Dict[str, Any]]:
    """
    Filter reports by minimum severity level.

    Args:
        reports: List of damage reports
        min_severity: Minimum severity (low, moderate, high, critical)

    Returns:
        Filtered list
    """
    severity_rank = {"low": 1, "moderate": 2, "high": 3, "critical": 4}
    min_rank = severity_rank.get(min_severity.lower(), 2)

    filtered = [
        r for r in reports
        if severity_rank.get(r.get("severity", "low"), 1) >= min_rank
    ]
    return filtered


def filter_reports_by_confidence(
    reports: List[Dict[str, Any]],
    min_confidence: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Filter reports by minimum confidence.

    Args:
        reports: List of damage reports
        min_confidence: Minimum confidence (0.0-1.0)

    Returns:
        Filtered list
    """
    return [
        r for r in reports
        if r.get("confidence", 0.0) >= min_confidence
    ]


def summary_statistics(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary statistics from reports.

    Args:
        reports: List of damage reports

    Returns:
        Stats dict
    """
    if not reports:
        return {}

    severity_counts = {}
    confidence_values = []
    damage_type_counts = {}

    for report in reports:
        # Severity
        sev = report.get("severity", "unknown")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

        # Confidence
        conf = report.get("confidence", 0.0)
        confidence_values.append(conf)

        # Damage types
        damages = report.get("damage_type", "").split(",")
        for damage in damages:
            d = damage.strip()
            if d:
                damage_type_counts[d] = damage_type_counts.get(d, 0) + 1

    avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0

    return {
        "total_reports": len(reports),
        "severity_breakdown": severity_counts,
        "damage_type_breakdown": damage_type_counts,
        "avg_confidence": round(avg_confidence, 3),
        "min_confidence": min(confidence_values) if confidence_values else 0,
        "max_confidence": max(confidence_values) if confidence_values else 0,
    }


def print_summary(reports: List[Dict[str, Any]]) -> None:
    """Print formatted summary of reports."""
    stats = summary_statistics(reports)

    if not stats:
        print("No reports to summarize")
        return

    print("\n" + "="*60)
    print("DAMAGE ASSESSMENT SUMMARY")
    print("="*60)
    print(f"Total Reports: {stats['total_reports']}")
    print(f"Average Confidence: {stats['avg_confidence']:.2f}")
    print(f"Confidence Range: {stats['min_confidence']:.2f} - {stats['max_confidence']:.2f}")

    print("\nSeverity Breakdown:")
    for sev in ["critical", "high", "moderate", "low"]:
        count = stats["severity_breakdown"].get(sev, 0)
        if count > 0:
            pct = (count / stats["total_reports"]) * 100
            print(f"  {sev:>10}: {count:>3} ({pct:>5.1f}%)")

    print("\nDamage Types Found:")
    sorted_damages = sorted(
        stats["damage_type_breakdown"].items(),
        key=lambda x: x[1],
        reverse=True
    )
    for damage, count in sorted_damages[:10]:
        print(f"  {damage:>20}: {count:>3}")

    print("="*60 + "\n")
