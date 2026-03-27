"""
VLM Damage Recognition Module

A Vision Language Model-based system for analyzing structural damage from drone imagery.
Uses LLaVA to identify damage types, severity levels, and crack measurements.
Includes crack tracking for monitoring expansion over time.
"""

__version__ = "0.2.0"

from .damage_analyzer import DamageAnalyzer
from .supabase_reporter import SupabaseReporter
from .image_processor import ImageProcessor
from .crack_tracking import CrackTracker
from .damage_report_schema import DamageReportSchema, StatusTracker, create_damage_report

__all__ = [
    "DamageAnalyzer",
    "SupabaseReporter",
    "ImageProcessor",
    "CrackTracker",
    "DamageReportSchema",
    "StatusTracker",
    "create_damage_report",
]
