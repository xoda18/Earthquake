"""
Crack Tracking and Expansion Analysis

Tracks crack growth over time by comparing measurements from photos
of the same location taken at different times.
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


class CrackTracker:
    """Track and analyze crack expansion over time."""

    @staticmethod
    def compare_measurements(
        measurement1: Dict[str, Any],
        measurement2: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compare two crack measurements to detect growth.

        Args:
            measurement1: Earlier measurement
            measurement2: Later measurement

        Returns:
            Expansion analysis dict
        """
        if not measurement1 or not measurement2:
            return {"error": "Invalid measurements"}

        # Extract crack data
        cracks1 = measurement1.get("cracks", [])
        cracks2 = measurement2.get("cracks", [])

        # Overall metrics
        length1 = measurement1.get("total_crack_area_mm2", 0)
        length2 = measurement2.get("total_crack_area_mm2", 0)

        largest1 = measurement1.get("largest_crack_length_mm", 0)
        largest2 = measurement2.get("largest_crack_length_mm", 0)

        width1 = measurement1.get("largest_crack_width_mm", 0)
        width2 = measurement2.get("largest_crack_width_mm", 0)

        # Calculate expansion
        area_expansion_mm2 = length2 - length1
        area_expansion_pct = (area_expansion_mm2 / length1 * 100) if length1 > 0 else 0

        length_expansion_mm = largest2 - largest1
        length_expansion_pct = (length_expansion_mm / largest1 * 100) if largest1 > 0 else 0

        width_expansion_mm = width2 - width1
        width_expansion_pct = (width_expansion_mm / width1 * 100) if width1 > 0 else 0

        # Severity assessment
        expansion_severity = CrackTracker._assess_expansion_severity(
            area_expansion_pct,
            length_expansion_pct,
            width_expansion_pct,
        )

        return {
            "total_crack_area": {
                "measurement1_mm2": length1,
                "measurement2_mm2": length2,
                "expansion_mm2": area_expansion_mm2,
                "expansion_percent": round(area_expansion_pct, 2),
            },
            "largest_crack_length": {
                "measurement1_mm": largest1,
                "measurement2_mm": largest2,
                "expansion_mm": length_expansion_mm,
                "expansion_percent": round(length_expansion_pct, 2),
            },
            "largest_crack_width": {
                "measurement1_mm": width1,
                "measurement2_mm": width2,
                "expansion_mm": width_expansion_mm,
                "expansion_percent": round(width_expansion_pct, 2),
            },
            "expansion_severity": expansion_severity,
            "recommendation": CrackTracker._get_recommendation(expansion_severity),
            "crack_count_change": len(cracks2) - len(cracks1),
        }

    @staticmethod
    def compare_cracks_at_location(
        cracks1: List[Dict[str, Any]],
        cracks2: List[Dict[str, Any]],
        location_region: str = None,
    ) -> Dict[str, Any]:
        """
        Compare cracks at a specific location between two measurements.

        Args:
            cracks1: Crack array from first measurement
            cracks2: Crack array from second measurement
            location_region: Optional region filter (e.g., "top-left")

        Returns:
            Per-location expansion analysis
        """
        # Filter by location if specified
        if location_region:
            cracks1 = [c for c in cracks1 if c.get("location") == location_region]
            cracks2 = [c for c in cracks2 if c.get("location") == location_region]

        if not cracks1 or not cracks2:
            return {"error": "No cracks found in specified location"}

        # Compare crack metrics
        total_area1 = sum(c.get("measurements", {}).get("area_mm2", 0) for c in cracks1)
        total_area2 = sum(c.get("measurements", {}).get("area_mm2", 0) for c in cracks2)

        lengths1 = [c.get("measurements", {}).get("length_mm", 0) for c in cracks1]
        lengths2 = [c.get("measurements", {}).get("length_mm", 0) for c in cracks2]

        widths1 = [c.get("measurements", {}).get("width_mm", 0) for c in cracks1]
        widths2 = [c.get("measurements", {}).get("width_mm", 0) for c in cracks2]

        # Calculate expansions
        area_expansion_pct = ((total_area2 - total_area1) / total_area1 * 100) if total_area1 > 0 else 0
        length_expansion_pct = ((max(lengths2, default=0) - max(lengths1, default=0)) / max(lengths1, default=0.1) * 100) if lengths1 else 0
        width_expansion_pct = ((max(widths2, default=0) - max(widths1, default=0)) / max(widths1, default=0.1) * 100) if widths1 else 0

        expansion_severity = CrackTracker._assess_expansion_severity(
            area_expansion_pct,
            length_expansion_pct,
            width_expansion_pct,
        )

        # Analyze status changes
        statuses1 = [c.get("status", "unknown") for c in cracks1]
        statuses2 = [c.get("status", "unknown") for c in cracks2]

        return {
            "location": location_region,
            "crack_count_change": len(cracks2) - len(cracks1),
            "area_expansion_percent": round(area_expansion_pct, 2),
            "length_expansion_percent": round(length_expansion_pct, 2),
            "width_expansion_percent": round(width_expansion_pct, 2),
            "expansion_severity": expansion_severity,
            "status_summary": {
                "measurement1": statuses1,
                "measurement2": statuses2,
            },
            "recommendation": CrackTracker._get_recommendation(expansion_severity),
        }

    @staticmethod
    def _assess_expansion_severity(
        area_pct: float,
        length_pct: float,
        width_pct: float,
    ) -> str:
        """Assess severity of crack expansion."""
        avg_expansion = (abs(area_pct) + abs(length_pct) + abs(width_pct)) / 3

        if avg_expansion < 5:
            return "MINIMAL"
        elif avg_expansion < 15:
            return "SLOW"
        elif avg_expansion < 30:
            return "MODERATE"
        elif avg_expansion < 50:
            return "RAPID"
        else:
            return "CRITICAL"

    @staticmethod
    def _get_recommendation(severity: str) -> str:
        """Get recommendation based on expansion severity."""
        recommendations = {
            "MINIMAL": "Continue monitoring. No immediate action required.",
            "SLOW": "Monitor regularly. Maintain surveillance schedule.",
            "MODERATE": "Increase monitoring frequency. Consider structural assessment.",
            "RAPID": "Urgent action required. Recommend immediate structural inspection.",
            "CRITICAL": "Emergency response needed. Building may be at risk of collapse.",
        }
        return recommendations.get(severity, "Unknown severity")

    @staticmethod
    def track_location_over_time(
        reports: List[Dict[str, Any]],
        location_tolerance_m: float = 0.05,  # 50cm tolerance
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group damage reports by location to track changes over time.

        Args:
            reports: List of damage reports with crack measurements
            location_tolerance_m: Tolerance for grouping locations (meters)

        Returns:
            Dict mapping location to list of measurements over time
        """
        locations = {}

        for report in reports:
            lat = report.get("lat")
            lon = report.get("lon")

            if lat is None or lon is None:
                continue

            # Find existing location cluster
            found = False
            for loc_key, measurements in locations.items():
                loc_lat, loc_lon = map(float, loc_key.split(","))
                distance = CrackTracker._haversine_distance(lat, lon, loc_lat, loc_lon)

                if distance < location_tolerance_m:
                    measurements.append({
                        "epoch": report.get("epoch"),
                        "timestamp": datetime.fromtimestamp(report.get("epoch", 0)).isoformat(),
                        "crack_measurements": report.get("_crack_measurements"),
                        "severity": report.get("severity"),
                        "event_id": report.get("event_id"),
                    })
                    found = True
                    break

            # Create new location cluster if not found
            if not found:
                loc_key = f"{lat},{lon}"
                locations[loc_key] = [{
                    "epoch": report.get("epoch"),
                    "timestamp": datetime.fromtimestamp(report.get("epoch", 0)).isoformat(),
                    "crack_measurements": report.get("_crack_measurements"),
                    "severity": report.get("severity"),
                    "event_id": report.get("event_id"),
                }]

        # Sort measurements by time and analyze expansion
        for loc_key, measurements in locations.items():
            measurements.sort(key=lambda x: x["epoch"])

            # Add expansion analysis between consecutive measurements
            for i in range(len(measurements) - 1):
                meas1 = measurements[i]["crack_measurements"]
                meas2 = measurements[i + 1]["crack_measurements"]

                if meas1 and meas2:
                    expansion = CrackTracker.compare_measurements(meas1, meas2)
                    measurements[i + 1]["expansion_analysis"] = expansion

        return locations

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance between two coordinates in meters.

        Args:
            lat1, lon1: First coordinate
            lat2, lon2: Second coordinate

        Returns:
            Distance in meters
        """
        from math import radians, sin, cos, sqrt, atan2

        R = 6371000  # Earth radius in meters

        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        delta_lat = radians(lat2 - lat1)
        delta_lon = radians(lon2 - lon1)

        a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c

        return distance

    @staticmethod
    def generate_tracking_report(
        location_data: Dict[str, List[Dict[str, Any]]],
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive crack tracking report.

        Args:
            location_data: Grouped measurements by location
            output_file: Optional file to save report

        Returns:
            Comprehensive tracking report
        """
        report = {
            "report_type": "crack_tracking_analysis",
            "generated_at": datetime.now().isoformat(),
            "total_locations": len(location_data),
            "locations": {},
        }

        for loc_key, measurements in location_data.items():
            lat, lon = map(float, loc_key.split(","))
            location_report = {
                "coordinates": {"lat": lat, "lon": lon},
                "measurements_count": len(measurements),
                "time_span_days": (
                    (measurements[-1]["epoch"] - measurements[0]["epoch"]) / 86400
                    if len(measurements) > 1
                    else 0
                ),
                "measurements": measurements,
                "expansion_history": [
                    m.get("expansion_analysis")
                    for m in measurements[1:]
                    if m.get("expansion_analysis")
                ],
            }

            # Overall assessment
            if location_report["expansion_history"]:
                latest_expansion = location_report["expansion_history"][-1]
                location_report["current_status"] = latest_expansion.get("expansion_severity", "UNKNOWN")
                location_report["recommendation"] = latest_expansion.get("recommendation", "")

            report["locations"][loc_key] = location_report

        # Save if output file specified
        if output_file:
            with open(output_file, "w") as f:
                json.dump(report, f, indent=2, default=str)

        return report


def create_tracker() -> CrackTracker:
    """Factory function to create tracker."""
    return CrackTracker()
