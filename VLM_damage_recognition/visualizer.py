"""
Damage Visualization

Draws bounding boxes and annotations for detected damage on images.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict, Any
import random


class DamageVisualizer:
    """Visualize detected damage on images."""

    # Color scheme for different damage types
    DAMAGE_COLORS = {
        "crack": (0, 0, 255),  # Red
        "spalling": (0, 165, 255),  # Orange
        "collapse": (0, 0, 139),  # Dark red
        "displacement": (255, 0, 0),  # Blue
        "deformation": (0, 255, 255),  # Yellow
        "corrosion": (165, 42, 42),  # Brown
        "efflorescence": (200, 200, 200),  # Gray
        "water_damage": (0, 128, 255),  # Orange-blue
        "settlement": (128, 0, 128),  # Purple
        "tilt": (0, 255, 0),  # Green
        "buckling": (255, 255, 0),  # Cyan
        "crushing": (255, 0, 255),  # Magenta
        "shear_failure": (0, 255, 128),  # Spring green
        "rebar_exposure": (210, 180, 140),  # Tan
    }

    SEVERITY_COLORS = {
        "low": (0, 255, 0),  # Green
        "moderate": (0, 165, 255),  # Orange
        "high": (0, 0, 255),  # Red
        "critical": (0, 0, 139),  # Dark red
    }

    @staticmethod
    def draw_damage_grid(
        image: Image.Image,
        damage_areas: List[Tuple[float, float, float, float]],
        damage_types: List[str],
        severity: str = "moderate",
        confidence: float = 0.8,
        area_percent: float = 0.0,
    ) -> Image.Image:
        """
        Draw damage detection grid on image.

        Args:
            image: PIL Image
            damage_areas: List of (x_min, y_min, x_max, y_max) normalized coordinates
            damage_types: List of damage type labels
            severity: Overall severity level
            confidence: Detection confidence
            area_percent: Percentage of image affected

        Returns:
            Annotated PIL Image
        """
        img = image.copy()
        width, height = img.size

        draw = ImageDraw.Draw(img, "RGBA")

        # Draw severity overlay at top
        severity_color = DamageVisualizer.SEVERITY_COLORS.get(severity, (0, 0, 255))
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay, "RGBA")

        # Semi-transparent top bar
        overlay_draw.rectangle([(0, 0), (width, 80)], fill=(*severity_color, 80))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Draw damage rectangles
        for idx, (x_min, y_min, x_max, y_max) in enumerate(damage_areas):
            # Convert normalized coordinates to pixel coordinates
            x1 = int(x_min * width)
            y1 = int(y_min * height)
            x2 = int(x_max * width)
            y2 = int(y_max * height)

            # Get color for this damage type
            damage_type = damage_types[idx] if idx < len(damage_types) else "damage"
            color = DamageVisualizer.DAMAGE_COLORS.get(damage_type, (0, 0, 255))

            # Draw rectangle
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)

            # Draw label
            label = f"{damage_type} {(damage_areas[idx][2] - damage_areas[idx][0])*100:.0f}%"
            draw.text((x1 + 5, y1 - 20), label, fill=color)

        # Draw info text at top
        info_text = f"Severity: {severity.upper()} | Confidence: {confidence:.0%} | Area: {area_percent:.1f}%"
        draw.text((10, 10), info_text, fill=(255, 255, 255))

        return img

    @staticmethod
    def draw_damage_regions(
        image: Image.Image,
        damage_mask: np.ndarray = None,
        severity: str = "moderate",
        confidence: float = 0.8,
        damage_types: List[str] = None,
    ) -> Image.Image:
        """
        Draw damage regions with heatmap overlay.

        Args:
            image: PIL Image
            damage_mask: Binary or float mask showing damage regions
            severity: Severity level
            confidence: Confidence score
            damage_types: List of damage types

        Returns:
            Annotated PIL Image
        """
        if damage_mask is None:
            return image

        img = np.array(image)
        height, width = img.shape[:2]

        # Resize mask to match image if needed
        if damage_mask.shape != (height, width):
            damage_mask = cv2.resize(damage_mask, (width, height))

        # Normalize mask to 0-255
        if damage_mask.max() > 1:
            mask_norm = damage_mask / damage_mask.max()
        else:
            mask_norm = damage_mask

        # Create heatmap
        heatmap = cv2.applyColorMap((mask_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Blend with original image
        severity_alpha = {"low": 0.2, "moderate": 0.4, "high": 0.6, "critical": 0.8}
        alpha = severity_alpha.get(severity, 0.4)
        result = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Severity: {severity.upper()} | Conf: {confidence:.0%}"
        cv2.putText(result, text, (10, 30), font, 1, (255, 255, 255), 2)

        if damage_types:
            types_str = ", ".join(damage_types[:3])
            cv2.putText(result, f"Types: {types_str}", (10, 70), font, 0.7, (255, 255, 255), 2)

        return Image.fromarray(result)

    @staticmethod
    def draw_contours_on_damage(
        image: Image.Image,
        damage_contours: List[np.ndarray],
        severity: str = "moderate",
        confidence: float = 0.8,
    ) -> Image.Image:
        """
        Draw contours of detected damage regions.

        Args:
            image: PIL Image
            damage_contours: List of contour arrays from OpenCV
            severity: Severity level
            confidence: Confidence score

        Returns:
            Annotated PIL Image
        """
        img = np.array(image)
        height, width = img.shape[:2]

        # Draw contours
        severity_color = DamageVisualizer.SEVERITY_COLORS.get(severity, (0, 0, 255))
        for contour in damage_contours:
            cv2.drawContours(img, [contour], 0, severity_color, 3)

        # Add info
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"{severity.upper()} | {confidence:.0%} confidence | {len(damage_contours)} regions"
        cv2.putText(img, text, (10, 30), font, 1, (255, 255, 255), 2)

        return Image.fromarray(img)

    @staticmethod
    def draw_damage_from_text(
        image: Image.Image,
        damage_description: str,
        severity: str = "moderate",
        confidence: float = 0.8,
    ) -> Image.Image:
        """
        Draw damage regions based on text description.
        Uses rule-based damage location estimation.

        Args:
            image: PIL Image
            damage_description: Text description of damage
            severity: Severity level
            confidence: Confidence score

        Returns:
            Annotated PIL Image
        """
        img = image.copy()
        width, height = img.size
        draw = ImageDraw.Draw(img)

        # Parse description for damage locations
        desc_lower = damage_description.lower()

        # Simple heuristic-based damage localization
        damage_regions = []

        if any(word in desc_lower for word in ["top", "upper", "above", "roof"]):
            damage_regions.append(("Top", 0, 0, 1, 0.3))
        if any(word in desc_lower for word in ["bottom", "lower", "below", "foundation"]):
            damage_regions.append(("Bottom", 0, 0.7, 1, 1))
        if any(word in desc_lower for word in ["left", "left side"]):
            damage_regions.append(("Left", 0, 0, 0.3, 1))
        if any(word in desc_lower for word in ["right", "right side"]):
            damage_regions.append(("Right", 0.7, 0, 1, 1))
        if any(word in desc_lower for word in ["center", "middle", "central"]):
            damage_regions.append(("Center", 0.25, 0.25, 0.75, 0.75))
        if any(word in desc_lower for word in ["corner", "corners"]):
            damage_regions.append(("Corners", 0, 0, 0.2, 0.2))

        # If no specific regions found, mark distributed damage
        if not damage_regions:
            # Create grid of potential damage
            grid_size = 3
            cell_width = 1.0 / grid_size
            cell_height = 1.0 / grid_size
            for i in range(grid_size):
                for j in range(grid_size):
                    if random.random() > 0.5:  # 50% chance
                        x_min = i * cell_width
                        y_min = j * cell_height
                        x_max = (i + 1) * cell_width
                        y_max = (j + 1) * cell_height
                        damage_regions.append((f"Area {i},{j}", x_min, y_min, x_max, y_max))

        # Draw damage regions
        severity_color = DamageVisualizer.SEVERITY_COLORS.get(severity, (0, 0, 255))
        for label, x_min, y_min, x_max, y_max in damage_regions:
            x1 = int(x_min * width)
            y1 = int(y_min * height)
            x2 = int(x_max * width)
            y2 = int(y_max * height)

            # Draw rectangle
            draw.rectangle([(x1, y1), (x2, y2)], outline=severity_color, width=3)
            draw.text((x1 + 5, y1 - 20), label, fill=severity_color)

        # Add header
        draw.rectangle([(0, 0), (width, 60)], fill=(*severity_color, 100))
        draw.text((10, 10), f"{severity.upper()} - {confidence:.0%} confidence", fill=(255, 255, 255))

        return img

    @staticmethod
    def create_summary_panel(
        reports: List[Dict[str, Any]],
        images: List[Image.Image],
        title: str = "Damage Assessment Summary",
    ) -> Image.Image:
        """
        Create a summary panel showing all results.

        Args:
            reports: List of damage reports
            images: List of original images
            title: Panel title

        Returns:
            Summary panel image
        """
        # Create panel
        panel_width = 1200
        panel_height = 100 + len(reports) * 80

        panel = Image.new("RGB", (panel_width, panel_height), color=(240, 240, 240))
        draw = ImageDraw.Draw(panel)

        # Title
        draw.text((10, 10), title, fill=(0, 0, 0))

        # Summary rows
        y_offset = 50
        for idx, (report, image) in enumerate(zip(reports, images)):
            severity = report["severity"]
            damage_type = report["damage_type"]
            confidence = report["confidence"]

            color = DamageVisualizer.SEVERITY_COLORS.get(severity, (0, 0, 255))

            text = f"{idx+1}. {severity.upper():>10} | {damage_type[:50]:50} | {confidence:.0%}"
            draw.text((10, y_offset), text, fill=color)
            y_offset += 25

        return panel
