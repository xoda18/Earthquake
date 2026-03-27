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
        drone_id: str = "JASS-DRONE-01"
    ) -> Dict[str, Any]:
        """
        Analyze single image for structural damage.

        Args:
            image: PIL Image
            lat: Latitude
            lon: Longitude
            building_name: Optional building name/type
            drone_id: Drone identifier

        Returns:
            Structured damage report dict
        """
        # Generate LLaVA analysis
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

        # Create damage report matching existing schema
        report = {
            "event_id": str(uuid.uuid4()),
            "epoch": time.time(),
            "lat": lat,
            "lon": lon,
            "severity": severity,
            "damage_type": ", ".join(damage_types) if damage_types else "Unknown damage",
            "building": building_name,
            "description": description,
            "drone_id": drone_id,
            "confidence": confidence,
            # Additional VLM-specific fields
            "_vlm_analysis": {
                "area_percent": area_percent,
                "affected_elements": parsed.get("affected_elements", []),
                "raw_response": raw_response,
            }
        }

        return report

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
