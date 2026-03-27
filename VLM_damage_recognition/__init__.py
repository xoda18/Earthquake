"""
VLM Damage Recognition Module

A Vision Language Model-based system for analyzing structural damage from drone imagery.
Uses LLaVA to identify damage types and severity levels.
"""

__version__ = "0.1.0"

from .damage_analyzer import DamageAnalyzer
from .supabase_reporter import SupabaseReporter
from .image_processor import ImageProcessor

__all__ = ["DamageAnalyzer", "SupabaseReporter", "ImageProcessor"]
