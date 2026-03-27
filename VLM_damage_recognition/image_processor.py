"""
Multi-format Image Processor

Handles loading of JPG, PNG, TIFF, and GeoTIFF images with geolocation extraction.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import piexif
import random

try:
    import rasterio
    from rasterio.crs import CRS

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class ImageProcessor:
    """Load and process images with geolocation extraction."""

    SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".geotiff", ".tfw"]

    @staticmethod
    def is_supported(file_path: str) -> bool:
        """Check if file is a supported image format."""
        ext = Path(file_path).suffix.lower()
        return ext in ImageProcessor.SUPPORTED_FORMATS

    @staticmethod
    def load_image(file_path: str) -> Tuple[Image.Image, Optional[Tuple[float, float]]]:
        """
        Load image and extract geolocation if available.

        Args:
            file_path: Path to image file

        Returns:
            Tuple of (PIL Image, (lat, lon) or None)
        """
        file_path = str(file_path)
        ext = Path(file_path).suffix.lower()

        if ext in [".tif", ".tiff", ".geotiff"]:
            return ImageProcessor._load_geotiff(file_path)
        else:
            # JPG/PNG
            image = Image.open(file_path).convert("RGB")
            lat_lon = ImageProcessor._extract_exif_geolocation(file_path)
            return image, lat_lon

    @staticmethod
    def _load_geotiff(file_path: str) -> Tuple[Image.Image, Optional[Tuple[float, float]]]:
        """Load GeoTIFF and extract geolocation."""
        if not HAS_RASTERIO:
            # Fallback: load as regular TIFF
            image = Image.open(file_path).convert("RGB")
            return image, None

        try:
            with rasterio.open(file_path) as src:
                # Read first band and convert to RGB
                if src.count >= 3:
                    rgb = src.read([1, 2, 3])
                else:
                    rgb = src.read(1)

                # Convert to PIL Image
                if len(rgb.shape) == 3:
                    rgb = (rgb * 255 / rgb.max()).astype("uint8").transpose(1, 2, 0)
                else:
                    rgb = (rgb * 255 / rgb.max()).astype("uint8")

                image = Image.fromarray(rgb)

                # Extract center coordinates
                bounds = src.bounds
                lat = (bounds.top + bounds.bottom) / 2
                lon = (bounds.left + bounds.right) / 2

                return image, (lat, lon)
        except Exception as e:
            print(f"[WARN] Error reading GeoTIFF {file_path}: {e}")
            image = Image.open(file_path).convert("RGB")
            return image, None

    @staticmethod
    def _extract_exif_geolocation(file_path: str) -> Optional[Tuple[float, float]]:
        """Extract geolocation from EXIF metadata."""
        try:
            exif_dict = piexif.load(file_path)
            gps_ifd = exif_dict.get("GPS", {})

            if not gps_ifd:
                return None

            # Extract GPS coordinates
            lat = ImageProcessor._dms_to_decimal(gps_ifd.get(piexif.GPSIFD.GPSLatitude, ()))
            lon = ImageProcessor._dms_to_decimal(gps_ifd.get(piexif.GPSIFD.GPSLongitude, ()))

            if lat is not None and lon is not None:
                # Check hemisphere indicators
                lat_ref = gps_ifd.get(piexif.GPSIFD.GPSLatitudeRef, b"N").decode()
                lon_ref = gps_ifd.get(piexif.GPSIFD.GPSLongitudeRef, b"E").decode()

                if lat_ref == "S":
                    lat = -lat
                if lon_ref == "W":
                    lon = -lon

                return (lat, lon)
        except Exception as e:
            pass  # No EXIF geolocation

        return None

    @staticmethod
    def _dms_to_decimal(dms_tuple: tuple) -> Optional[float]:
        """Convert degrees/minutes/seconds to decimal degrees."""
        try:
            if not dms_tuple or len(dms_tuple) < 3:
                return None

            degrees = dms_tuple[0][0] / dms_tuple[0][1]
            minutes = dms_tuple[1][0] / dms_tuple[1][1]
            seconds = dms_tuple[2][0] / dms_tuple[2][1]

            return degrees + minutes / 60.0 + seconds / 3600.0
        except (TypeError, ZeroDivisionError):
            return None

    @staticmethod
    def generate_paphos_coordinates() -> Tuple[float, float]:
        """Generate random coordinates in Paphos, Cyprus region."""
        # Paphos LAND-ONLY zones (from mock_damage.py)
        zones = [
            {"lat": (34.755, 34.775), "lon": (32.415, 32.435)},
            {"lat": (34.750, 34.760), "lon": (32.418, 32.430)},
            {"lat": (34.770, 34.785), "lon": (32.420, 32.445)},
            {"lat": (34.760, 34.778), "lon": (32.435, 32.460)},
        ]
        zone = random.choice(zones)
        lat = round(random.uniform(*zone["lat"]), 6)
        lon = round(random.uniform(*zone["lon"]), 6)
        return lat, lon

    @staticmethod
    def batch_load_images(directory: str) -> list:
        """
        Load all supported images from directory.

        Returns:
            List of (file_path, PIL Image, (lat, lon)) tuples
        """
        results = []
        dir_path = Path(directory)

        if not dir_path.exists():
            print(f"[ERROR] Directory not found: {directory}")
            return results

        for file_path in sorted(dir_path.glob("**/*")):
            if not file_path.is_file():
                continue

            if ImageProcessor.is_supported(str(file_path)):
                try:
                    image, lat_lon = ImageProcessor.load_image(str(file_path))
                    if lat_lon is None:
                        lat_lon = ImageProcessor.generate_paphos_coordinates()
                    results.append((str(file_path), image, lat_lon))
                    print(f"[OK] Loaded: {file_path.name} at {lat_lon}")
                except Exception as e:
                    print(f"[ERROR] Failed to load {file_path}: {e}")

        return results
