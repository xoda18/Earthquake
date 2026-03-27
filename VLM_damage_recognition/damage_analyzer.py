"""
Damage Analyzer Pipeline

Main pipeline: image → LLaVA → structured damage report.
"""

import uuid
import time
from typing import Optional, Dict, Any
from PIL import Image

from .inference import LLaVAInference
from .image_processor import ImageProcessor
from .prompt_templates import DamagePrompts
from .damage_report_schema import DamageReportSchema


class DamageAnalyzer:
    """Analyze images for structural damage using LLaVA."""

    def __init__(self, quantize: bool = False, model_id: str = "llava-hf/llava-1.5-7b-hf"):
        """
        Initialize analyzer with LLaVA model.

        Args:
            quantize: Use int8 quantization for CPU
            model_id: HuggingFace model identifier
        """
        self.llava = LLaVAInference(model_id=model_id, quantize=quantize)
        self.damage_types = [
            "crack", "spalling", "collapse", "displacement", "deformation",
            "corrosion", "efflorescence", "water_damage", "settlement", "tilt",
            "buckling", "crushing", "shear_failure", "rebar_exposure"
        ]

    def analyze_image(
        self,
        image: Image.Image,
        lat: float,
        lon: float,
        building_name: Optional[str] = None,
        drone_id: str = "JASS-DRONE-01",
        include_crack_sizes: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze single image for structural damage.

        Args:
            image: PIL Image
            lat: Latitude
            lon: Longitude
            building_name: Optional building name/type
            drone_id: Drone identifier
            include_crack_sizes: Include detailed crack size analysis

        Returns:
            Structured damage report dict
        """
        # Step 1: Pre-screening — skip only if VERY confident there's no damage
        screen_response = self.llava.analyze_image(image, DamagePrompts.has_damage_screening())
        screen = DamagePrompts.parse_json_response(screen_response)
        has_damage = screen.get("has_damage", True)  # default True = proceed to full analysis
        screen_confidence = float(screen.get("confidence", 0.5))

        # Only skip if model is very confident there is NO damage (has_damage=false + confidence >= 0.75)
        if not has_damage and screen_confidence >= 0.75:
            return DamageReportSchema.create_report(
                file="",
                lat=lat,
                lon=lon,
                severity="none",
                confidence=screen_confidence,
                damage_type="none",
                description=screen.get("reason", "No structural damage detected."),
                status="stable",
                drone_id=drone_id,
                building=building_name or "Structure",
            )

        # Step 2: Full damage analysis (only runs if damage confirmed)
        prompt = DamagePrompts.multi_aspect_analysis(include_building_type=True)
        raw_response = self.llava.analyze_image(image, prompt)

        # Parse response
        parsed = DamagePrompts.parse_json_response(raw_response)

        # Extract fields with defaults
        damage_types = parsed.get("damage_types", [])
        if not isinstance(damage_types, list):
            damage_types = []

        severity = parsed.get("severity", "moderate")
        severity = DamagePrompts.normalize_severity(severity)

        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

        area_percent = float(parsed.get("damage_area_percent", 0))
        area_percent = max(0, min(100, area_percent))

        description = parsed.get("description", raw_response[:500])

        # Infer building type if not provided
        if not building_name:
            building_type = parsed.get("building_type", "Unknown")
            building_name = building_type if building_type != "Unknown" else "Structure"

        # Step 3: Detailed crack analysis (only if damage confirmed and cracks mentioned)
        cracks = []
        overall_status = "unknown"
        if include_crack_sizes and confidence >= 0.6:
            crack_data = self._analyze_cracks_detailed(image)
            if crack_data and "cracks" in crack_data:
                cracks = crack_data.get("cracks", [])
                overall_status = crack_data.get("overall_status", "unknown")

        # Determine overall status from cracks
        if cracks:
            statuses = [c.get("status", "unknown") for c in cracks]
            if "growing" in statuses:
                overall_status = "growing"
            elif "recovering" in statuses:
                overall_status = "recovering"
            elif all(s == "stable" for s in statuses):
                overall_status = "stable"

        # Create standardized damage report using schema
        report = DamageReportSchema.create_report(
            file="",  # Will be set by caller
            lat=lat,
            lon=lon,
            severity=severity,
            confidence=confidence,
            damage_type=", ".join(damage_types) if damage_types else "Unknown damage",
            description=description,
            status=overall_status,
            drone_id=drone_id,
            building=building_name,
            cracks=cracks if cracks else None,
            additional_data={
                "_vlm_analysis": {
                    "area_percent": area_percent,
                    "affected_elements": parsed.get("affected_elements", []),
                    "raw_response": raw_response,
                }
            }
        )

        return report

    def _analyze_cracks_detailed(self, image: Image.Image) -> Optional[Dict[str, Any]]:
        """
        Analyze all cracks in detail with location and status tracking.

        Args:
            image: PIL Image

        Returns:
            Detailed crack data with individual crack objects or None
        """
        try:
            prompt = DamagePrompts.detailed_crack_tracking()
            raw_response = self.llava.analyze_image(image, prompt)
            crack_data = DamagePrompts.parse_json_response(raw_response)

            # Convert crack array to standardized format
            standardized_cracks = []
            for crack_info in [c for c in crack_data.get("cracks", []) if float(c.get("confidence", 0)) >= 0.6]:
                crack = DamageReportSchema.create_crack(
                    crack_id=crack_info.get("id", len(standardized_cracks) + 1),
                    location=crack_info.get("location_region", "unknown"),
                    measurements=crack_info.get("measurements", {}),
                    severity=crack_info.get("severity", "moderate"),
                    status=crack_info.get("status", "unknown"),
                    confidence=float(crack_info.get("confidence", 0.85)),
                    description=crack_info.get("description", ""),
                    normalized_coords=crack_info.get("normalized_coords"),
                )
                standardized_cracks.append(crack)

            return {
                "cracks": standardized_cracks,
                "overall_status": crack_data.get("overall_status", "unknown"),
                "summary_statistics": crack_data.get("summary_statistics", {}),
                "confidence": float(crack_data.get("confidence", 0.85)),
                "scale_assumption": crack_data.get("scale_assumption", "1m x 1m (1000mm x 1000mm)"),
            }
        except Exception as e:
            print(f"[WARN] Detailed crack analysis failed: {e}")
            return None

    def _analyze_crack_sizes(self, image: Image.Image) -> Optional[Dict[str, Any]]:
        """
        Analyze crack sizes in image (assumes 1m x 1m scale).
        Legacy method - kept for backwards compatibility.

        Args:
            image: PIL Image

        Returns:
            Crack measurements dict or None
        """
        try:
            prompt = DamagePrompts.crack_size_analysis()
            raw_response = self.llava.analyze_image(image, prompt)
            crack_data = DamagePrompts.parse_json_response(raw_response)

            # Add scale information
            crack_data["scale_assumption"] = "1m x 1m (1000mm x 1000mm) if no reference scale visible"
            crack_data["measurement_unit"] = "millimeters"

            return crack_data
        except Exception as e:
            print(f"[WARN] Crack size analysis failed: {e}")
            return None

    def analyze_batch(
        self,
        image_list: list,
        drone_id: str = "JASS-DRONE-01"
    ) -> list:
        """
        Analyze batch of images.

        Args:
            image_list: List of (file_path, PIL Image, (lat, lon)) tuples
            drone_id: Drone identifier

        Returns:
            List of damage reports
        """
        reports = []

        for idx, (file_path, image, (lat, lon)) in enumerate(image_list, 1):
            print(f"\n[{idx}/{len(image_list)}] Analyzing: {file_path}")
            try:
                report = self.analyze_image(image, lat, lon, drone_id=drone_id)
                reports.append(report)
                print(f"  ✓ Severity: {report['severity']} | Confidence: {report['confidence']:.2f}")
                print(f"  ✓ Damage: {report['damage_type']}")
            except Exception as e:
                print(f"  ✗ Error: {e}")

        return reports


def create_analyzer(quantize: bool = False) -> DamageAnalyzer:
    """Factory function to create analyzer."""
    return DamageAnalyzer(quantize=quantize)
