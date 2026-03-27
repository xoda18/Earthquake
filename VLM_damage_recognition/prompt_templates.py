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
