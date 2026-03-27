"""
Prompt Templates for Structural Damage Analysis

Provides structured prompts for LLaVA to analyze building damage.
"""

import json


class DamagePrompts:
    """Collection of prompts for damage analysis."""

    @staticmethod
    def structural_damage_analysis() -> str:
        """Prompt for comprehensive structural damage analysis."""
        return """Analyze this structural damage image carefully. Identify and classify all visible damage.

Respond ONLY in valid JSON format (no other text):
{
  "damage_types": ["list", "of", "damage", "types"],
  "severity": "low|moderate|high|critical",
  "confidence": 0.0,
  "area_percent": 0,
  "affected_elements": ["list", "of", "structural", "elements"],
  "description": "Brief description of damage observed"
}

Damage types can include: crack, spalling, collapse, displacement, deformation, corrosion, efflorescence, water_damage, settlement, tilt, buckling, crushing, shear_failure, etc.

Affected structural elements can include: wall, roof, foundation, column, beam, floor, window_frame, door_frame, facade, parapet, balcony, etc.

Severity guidelines:
- low: Minor cosmetic damage, cracks < 1mm
- moderate: Visible damage, cracks 1-5mm, affecting appearance
- high: Significant damage, cracks > 5mm, may affect structural integrity
- critical: Severe damage, obvious collapse risk, immediate danger

Confidence: 0.0-1.0 (how confident are you in this assessment)
Area percent: 0-100 (estimated percentage of visible area affected by damage)"""

    @staticmethod
    def crack_size_analysis() -> str:
        """Prompt for detailed crack size measurement."""
        return """Analyze cracks in this structural damage image and estimate their sizes.

IMPORTANT: Assume the photo shows a 1m x 1m area if no scale reference is visible.

Respond ONLY in valid JSON format (no other text):
{
  "cracks": [
    {
      "id": 1,
      "location": "top-left|top-center|top-right|mid-left|center|mid-right|bottom-left|bottom-center|bottom-right",
      "type": "vertical|horizontal|diagonal|network",
      "length_mm": 150,
      "width_mm": 2.5,
      "depth_estimate": "surface|shallow|deep|unknown",
      "pattern": "straight|curved|branching|web-like",
      "severity": "hairline|fine|moderate|severe",
      "notes": "Additional observations"
    }
  ],
  "total_crack_area_mm2": 5000,
  "largest_crack_length_mm": 450,
  "largest_crack_width_mm": 3.5,
  "crack_density": "low|moderate|high|severe",
  "overall_assessment": "Structural description",
  "confidence": 0.85,
  "measurement_confidence": "high|medium|low"
}

Measurement Guidelines (assuming 1m x 1m = 1000mm x 1000mm):
- Hairline cracks: < 0.5mm width
- Fine cracks: 0.5-1mm width
- Moderate cracks: 1-5mm width
- Severe cracks: > 5mm width

Depth estimates:
- surface: Only on surface, cosmetic
- shallow: Affects outer layer only
- deep: Penetrates significantly into material
- unknown: Cannot determine from image

Crack patterns:
- straight: Linear cracks
- curved: Curved or wavy cracks
- branching: Cracks that branch out
- web-like: Multiple interconnected cracks (like spider web)
- network: Dense network of cracks"""

    @staticmethod
    def detailed_crack_tracking() -> str:
        """Prompt for detailed crack tracking with location and status."""
        return """Analyze all cracks in this structural damage image in detail.

IMPORTANT: Assume the photo shows a 1m x 1m area if no scale reference is visible.
For each crack, provide location coordinates (0.0-1.0 normalized to image size).

Respond ONLY in valid JSON format (no other text):
{
  "cracks": [
    {
      "id": 1,
      "location_region": "top-left|top-center|top-right|middle-left|center|middle-right|bottom-left|bottom-center|bottom-right",
      "normalized_coords": {"x": 0.15, "y": 0.25},
      "measurements": {
        "length_mm": 450,
        "width_mm": 3.5,
        "depth_estimate": "surface|shallow|deep|unknown",
        "area_mm2": 1575,
        "pattern": "straight|curved|branching|network"
      },
      "severity": "low|moderate|high|critical",
      "status": "stable|growing|recovering|unknown",
      "confidence": 0.88,
      "description": "Detailed description of this specific crack"
    }
  ],
  "summary_statistics": {
    "total_cracks": 1,
    "total_crack_area_mm2": 1575,
    "largest_crack_length_mm": 450,
    "crack_density": "minimal|low|moderate|high|severe",
    "overall_severity": "low|moderate|high|critical"
  },
  "overall_status": "stable|growing|recovering|unknown",
  "confidence": 0.85,
  "scale_assumption": "1m x 1m (1000mm x 1000mm)"
}

Location regions guide:
- Divide image into 3x3 grid
- top-left (0.0-0.33, 0.0-0.33)
- top-center (0.33-0.66, 0.0-0.33)
- top-right (0.66-1.0, 0.0-0.33)
- middle-left (0.0-0.33, 0.33-0.66)
- center (0.33-0.66, 0.33-0.66)
- middle-right (0.66-1.0, 0.33-0.66)
- bottom-left (0.0-0.33, 0.66-1.0)
- bottom-center (0.33-0.66, 0.66-1.0)
- bottom-right (0.66-1.0, 0.66-1.0)

Status assessment:
- stable: Crack shows no signs of expansion or change
- growing: Crack appears to be expanding or widening (signs of active damage)
- recovering: Crack appears to be healing or closing
- unknown: Cannot determine status from image alone"""

    @staticmethod
    def damage_type_detection() -> str:
        """Prompt focused on identifying damage types."""
        return """Identify all types of structural damage visible in this image.

List each damage type found, with confidence level (0.0-1.0).

Respond in JSON format:
{
  "damage_types": [
    {"type": "crack", "confidence": 0.95, "location": "top left wall"},
    {"type": "spalling", "confidence": 0.87, "location": "foundation"}
  ],
  "overall_severity": "high",
  "summary": "Brief summary of findings"
}"""

    @staticmethod
    def severity_assessment() -> str:
        """Prompt focused on severity rating."""
        return """Assess the severity of structural damage shown in this image.

Consider:
1. Size and extent of damage
2. Type of damage (cracks, collapses, etc.)
3. Structural significance
4. Safety implications

Respond in JSON:
{
  "severity": "low|moderate|high|critical",
  "severity_score": 0.0,
  "reasoning": "Explanation for severity rating",
  "confidence": 0.0
}"""

    @staticmethod
    def building_type_detection() -> str:
        """Prompt to identify building type."""
        return """What type of building structure is shown in this image?

Possible types: residential, commercial, industrial, institutional, educational, religious, medical, bridge, road, utility, mixed_use

Respond in JSON:
{
  "building_type": "category",
  "confidence": 0.0,
  "description": "Brief description of building type indicators"
}"""

    @staticmethod
    def multi_aspect_analysis(include_building_type: bool = False) -> str:
        """Combined prompt for multiple aspects."""
        prompt = """Analyze this structural damage image comprehensively.

Respond ONLY in valid JSON format:
{
  "damage_types": ["list", "of", "identified", "damage", "types"],
  "severity": "low|moderate|high|critical",
  "severity_confidence": 0.0,
  "damage_area_percent": 0,
  "affected_elements": ["structural", "elements", "involved"],
  "description": "Detailed description of all observed damage"
}"""

        if include_building_type:
            prompt += """
Additionally, identify the building type:
{
  ...
  "building_type": "residential|commercial|industrial|institutional|other",
  "building_confidence": 0.0
}"""

        return prompt

    @staticmethod
    def parse_json_response(response: str) -> dict:
        """
        Parse JSON response from LLaVA.

        Args:
            response: Raw text response from model

        Returns:
            Parsed JSON dict or empty dict on failure
        """
        try:
            # Try to extract JSON from response
            response = response.strip()

            # Find JSON object
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            pass

        return {}

    @staticmethod
    def validate_damage_response(parsed_json: dict) -> bool:
        """Check if parsed response has required fields."""
        required_fields = ["damage_types", "severity", "confidence"]
        return all(field in parsed_json for field in required_fields)

    @staticmethod
    def normalize_severity(severity_str: str) -> str:
        """Normalize severity string to valid category."""
        severity_map = {
            "low": "low",
            "minor": "low",
            "moderate": "moderate",
            "medium": "moderate",
            "high": "high",
            "severe": "high",
            "critical": "critical",
            "urgent": "critical",
        }

        normalized = severity_map.get(severity_str.lower(), "moderate")
        return normalized
