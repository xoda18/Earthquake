"""
Standardized Damage Report Schema

Provides a consistent JSON template for all damage assessments.
Use this schema for files, databases, and Supabase.
"""

import json
import uuid
import time
from typing import Dict, Any, Optional, List
from datetime import datetime


class DamageReportSchema:
    """Standardized damage report template."""

    @staticmethod
    def create_report(
        file: str,
        lat: float,
        lon: float,
        severity: str,
        confidence: float,
        damage_type: str,
        description: str,
        status: str = "unknown",
        drone_id: str = "JASS-DRONE-01",
        building: Optional[str] = None,
        cracks: Optional[List[Dict[str, Any]]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a standardized damage report with detailed crack information.

        Args:
            file: Source image filename
            lat: Latitude
            lon: Longitude
            severity: low|moderate|high|critical (overall)
            confidence: 0.0-1.0 confidence score (overall)
            damage_type: Type of damage detected
            description: Detailed description
            status: growing|stable|recovering|unknown (overall)
            drone_id: Drone identifier
            building: Building name or type
            cracks: List of detailed crack measurements
            additional_data: Extra fields to include

        Returns:
            Standardized report dict
        """
        report = {
            "file": file,
            "lat": lat,
            "lon": lon,
            "severity": severity.lower(),
            "status": status.lower(),
            "confidence": float(confidence),
            "damage_type": damage_type,
            "description": description,
            "drone_id": drone_id,
            "building": building or "Unknown",
            "event_id": str(uuid.uuid4()),
            "epoch": time.time(),
            "timestamp": datetime.now().isoformat(),
        }

        # Add cracks if provided
        if cracks:
            report["cracks"] = cracks
            # Calculate summary statistics from cracks
            report["_summary_statistics"] = DamageReportSchema._calculate_crack_statistics(cracks)

        # Add optional fields
        if additional_data:
            report.update(additional_data)

        return report

    @staticmethod
    def create_crack(
        crack_id: int,
        location: str,
        measurements: Dict[str, Any],
        severity: str = "moderate",
        status: str = "stable",
        confidence: float = 0.85,
        description: str = "",
        normalized_coords: Optional[Dict[str, float]] = None,
        pixel_coords: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """
        Create a standardized crack measurement object.

        Args:
            crack_id: Unique crack identifier (1, 2, 3, ...)
            location: Region in image (top-left, top-center, top-right, middle-left, center, middle-right, bottom-left, bottom-center, bottom-right)
            measurements: Dict with length_mm, width_mm, depth_estimate, area_mm2, pattern
            severity: low|moderate|high|critical
            status: growing|stable|recovering|unknown
            confidence: 0.0-1.0
            description: Detailed crack description
            normalized_coords: {x, y} normalized to 0-1 (image ratio coords)
            pixel_coords: {x, y} pixel coordinates in original image

        Returns:
            Standardized crack dict
        """
        crack = {
            "id": crack_id,
            "location": location,
            "measurements": {
                "length_mm": measurements.get("length_mm", 0),
                "width_mm": measurements.get("width_mm", 0),
                "depth_estimate": measurements.get("depth_estimate", "unknown"),
                "area_mm2": measurements.get("area_mm2", 0),
                "pattern": measurements.get("pattern", "unknown"),
            },
            "severity": severity.lower(),
            "status": status.lower(),
            "confidence": float(confidence),
            "description": description,
        }

        # Add optional coordinate fields
        if normalized_coords:
            crack["normalized_coords"] = normalized_coords
        if pixel_coords:
            crack["pixel_coords"] = pixel_coords

        return crack

    @staticmethod
    def _calculate_crack_statistics(cracks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics from crack array."""
        if not cracks:
            return {}

        total_area = sum(c.get("measurements", {}).get("area_mm2", 0) for c in cracks)
        lengths = [c.get("measurements", {}).get("length_mm", 0) for c in cracks]
        widths = [c.get("measurements", {}).get("width_mm", 0) for c in cracks]

        severity_rank = {"low": 1, "moderate": 2, "high": 3, "critical": 4}
        max_severity_val = max([severity_rank.get(c.get("severity", "low"), 1) for c in cracks])
        severity_map = {1: "low", 2: "moderate", 3: "high", 4: "critical"}

        return {
            "total_cracks": len(cracks),
            "total_crack_area_mm2": total_area,
            "largest_crack_length_mm": max(lengths) if lengths else 0,
            "largest_crack_width_mm": max(widths) if widths else 0,
            "average_crack_length_mm": sum(lengths) / len(lengths) if lengths else 0,
            "average_crack_width_mm": sum(widths) / len(widths) if widths else 0,
            "crack_density": DamageReportSchema._assess_crack_density(total_area),
            "overall_severity": severity_map.get(max_severity_val, "unknown"),
            "measurement_unit": "millimeters",
            "scale_assumption": "1m x 1m (1000mm x 1000mm)",
        }

    @staticmethod
    def _assess_crack_density(total_area_mm2: float) -> str:
        """Assess crack density based on total area (1m x 1m = 1,000,000 mm²)."""
        density_percent = (total_area_mm2 / 1000000) * 100
        if density_percent < 0.3:
            return "minimal"
        elif density_percent < 1.0:
            return "low"
        elif density_percent < 3.0:
            return "moderate"
        elif density_percent < 8.0:
            return "high"
        else:
            return "severe"

    @staticmethod
    def create_batch_reports(
        reports_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Create multiple standardized reports.

        Args:
            reports_data: List of report data dicts

        Returns:
            List of standardized reports
        """
        return [
            DamageReportSchema.create_report(**data)
            for data in reports_data
        ]

    @staticmethod
    def to_json_string(report: Dict[str, Any], indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(report, indent=indent, default=str)

    @staticmethod
    def to_jsonl_string(report: Dict[str, Any]) -> str:
        """Convert report to JSONL format (one line)."""
        return json.dumps(report, default=str)

    @staticmethod
    def save_to_file(report: Dict[str, Any], file_path: str) -> bool:
        """
        Save report to JSON file.

        Args:
            report: Report dict
            file_path: Output file path

        Returns:
            True if successful
        """
        try:
            with open(file_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save report: {e}")
            return False

    @staticmethod
    def append_to_jsonl(report: Dict[str, Any], file_path: str) -> bool:
        """
        Append report to JSONL file (one report per line).

        Args:
            report: Report dict
            file_path: Output file path

        Returns:
            True if successful
        """
        try:
            with open(file_path, "a") as f:
                f.write(DamageReportSchema.to_jsonl_string(report) + "\n")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to append report: {e}")
            return False

    @staticmethod
    def load_from_file(file_path: str) -> Optional[Dict[str, Any]]:
        """Load single report from JSON file."""
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load report: {e}")
            return None

    @staticmethod
    def load_from_jsonl(file_path: str) -> List[Dict[str, Any]]:
        """Load multiple reports from JSONL file."""
        reports = []
        try:
            with open(file_path, "r") as f:
                for line in f:
                    if line.strip():
                        reports.append(json.loads(line))
        except Exception as e:
            print(f"[ERROR] Failed to load JSONL: {e}")
        return reports

    @staticmethod
    def validate_report(report: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate report has all required fields.

        Args:
            report: Report dict

        Returns:
            (is_valid, list_of_errors)
        """
        required_fields = [
            "file", "lat", "lon", "severity", "status",
            "confidence", "damage_type", "description"
        ]

        errors = []

        for field in required_fields:
            if field not in report:
                errors.append(f"Missing required field: {field}")

        # Validate field types
        if "lat" in report and not isinstance(report["lat"], (int, float)):
            errors.append("lat must be numeric")
        if "lon" in report and not isinstance(report["lon"], (int, float)):
            errors.append("lon must be numeric")
        if "confidence" in report:
            conf = report["confidence"]
            if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
                errors.append("confidence must be 0.0-1.0")

        # Validate severity values
        valid_severities = ["low", "moderate", "high", "critical"]
        if "severity" in report and report["severity"].lower() not in valid_severities:
            errors.append(f"severity must be one of: {valid_severities}")

        # Validate status values
        valid_statuses = ["growing", "stable", "recovering", "unknown"]
        if "status" in report and report["status"].lower() not in valid_statuses:
            errors.append(f"status must be one of: {valid_statuses}")

        return len(errors) == 0, errors

    @staticmethod
    def get_template() -> Dict[str, Any]:
        """Get empty template for reference."""
        return {
            "file": "image_name.jpg",
            "lat": 34.765,
            "lon": 32.42,
            "severity": "moderate",
            "status": "stable",
            "confidence": 0.85,
            "damage_type": "cracks and wall damage",
            "description": "Multiple cracks visible in wall structure...",
            "drone_id": "JASS-DRONE-01",
            "building": "Hotel",
            "event_id": "uuid-here",
            "epoch": 1711000000.0,
            "timestamp": "2026-03-27T10:00:00",
            "cracks": [
                {
                    "id": 1,
                    "location": "top-left",
                    "measurements": {
                        "length_mm": 450,
                        "width_mm": 3.5,
                        "depth_estimate": "deep",
                        "area_mm2": 1575,
                        "pattern": "straight"
                    },
                    "severity": "high",
                    "status": "growing",
                    "confidence": 0.88,
                    "description": "Vertical crack running from top to middle",
                    "normalized_coords": {"x": 0.15, "y": 0.25},
                    "pixel_coords": {"x": 150, "y": 250}
                }
            ],
            "_summary_statistics": {
                "total_cracks": 1,
                "total_crack_area_mm2": 1575,
                "largest_crack_length_mm": 450,
                "largest_crack_width_mm": 3.5,
                "average_crack_length_mm": 450,
                "average_crack_width_mm": 3.5,
                "crack_density": "low",
                "overall_severity": "high",
                "measurement_unit": "millimeters",
                "scale_assumption": "1m x 1m (1000mm x 1000mm)"
            }
        }

    @staticmethod
    def print_template():
        """Print template for reference."""
        template = DamageReportSchema.get_template()
        print("\nSTANDARDIZED DAMAGE REPORT TEMPLATE")
        print("=" * 80)
        print(json.dumps(template, indent=2))
        print("=" * 80)
        print("\nRequired Fields (Report Level):")
        print("  - file: Source image filename")
        print("  - lat: Latitude coordinate")
        print("  - lon: Longitude coordinate")
        print("  - severity: low | moderate | high | critical (overall)")
        print("  - status: growing | stable | recovering | unknown (overall)")
        print("  - confidence: 0.0 - 1.0 (overall confidence score)")
        print("  - damage_type: Type of damage (e.g., 'cracks', 'collapse')")
        print("  - description: Detailed description of damage")
        print("\nCracks Array (Per-Crack Details):")
        print("  - id: Unique crack identifier (1, 2, 3, ...)")
        print("  - location: Region in image (top-left, center, bottom-right, etc.)")
        print("  - measurements: {length_mm, width_mm, depth_estimate, area_mm2, pattern}")
        print("  - severity: Per-crack severity (low | moderate | high | critical)")
        print("  - status: Per-crack status (growing | stable | recovering | unknown)")
        print("  - confidence: Per-crack measurement confidence")
        print("  - description: Per-crack description")
        print("  - normalized_coords: {x, y} from 0.0-1.0 (optional)")
        print("  - pixel_coords: {x, y} pixel coordinates (optional)")
        print("\nOptional Fields:")
        print("  - drone_id: Identifier of drone (default: JASS-DRONE-01)")
        print("  - building: Building name or type")
        print("  - event_id: Unique event identifier (auto-generated)")
        print("  - epoch: Unix timestamp (auto-generated)")
        print("  - timestamp: ISO timestamp (auto-generated)")
        print("  - _summary_statistics: Auto-calculated from cracks array")
        print()


class StatusTracker:
    """Track damage status (growing, stable, recovering)."""

    @staticmethod
    def assess_status(
        old_severity: Optional[str],
        new_severity: str,
        old_confidence: Optional[float] = None,
        new_confidence: Optional[float] = None,
    ) -> str:
        """
        Assess damage status by comparing old and new measurements.

        Args:
            old_severity: Previous severity level
            new_severity: Current severity level
            old_confidence: Previous confidence
            new_confidence: Current confidence

        Returns:
            Status: growing|stable|recovering|unknown
        """
        severity_rank = {"low": 1, "moderate": 2, "high": 3, "critical": 4}

        if old_severity is None:
            return "unknown"

        old_rank = severity_rank.get(old_severity.lower(), 0)
        new_rank = severity_rank.get(new_severity.lower(), 0)

        if new_rank > old_rank:
            return "growing"
        elif new_rank < old_rank:
            return "recovering"
        elif new_rank == old_rank:
            return "stable"
        else:
            return "unknown"

    @staticmethod
    def get_status_description(status: str) -> str:
        """Get human-readable description of status."""
        descriptions = {
            "growing": "Damage is getting worse",
            "stable": "Damage is not changing",
            "recovering": "Damage is improving",
            "unknown": "Status cannot be determined",
        }
        return descriptions.get(status.lower(), "Unknown status")


def create_damage_report(
    file: str,
    lat: float,
    lon: float,
    severity: str,
    confidence: float,
    damage_type: str,
    description: str,
    status: str = "unknown",
    **kwargs
) -> Dict[str, Any]:
    """Convenience function to create a report."""
    return DamageReportSchema.create_report(
        file=file,
        lat=lat,
        lon=lon,
        severity=severity,
        confidence=confidence,
        damage_type=damage_type,
        description=description,
        status=status,
        **kwargs
    )
