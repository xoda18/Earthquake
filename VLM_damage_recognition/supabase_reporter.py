"""
Supabase Reporter

Write damage reports to Supabase database.
"""

import sys
import os
from typing import List, Dict, Any
import json

# Add parent directory to path for supabase_client import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import supabase_client as sb
    HAS_SUPABASE = True
except ImportError:
    HAS_SUPABASE = False


class SupabaseReporter:
    """Write damage reports to Supabase."""

    def __init__(self, use_supabase: bool = True):
        """
        Initialize reporter.

        Args:
            use_supabase: Whether to write to Supabase (False = JSON only)
        """
        self.use_supabase = use_supabase and HAS_SUPABASE
        if self.use_supabase:
            print("[Supabase] Connected")
        else:
            print("[Supabase] Disabled (JSON output only)")

    def write_report(self, report: Dict[str, Any]) -> bool:
        """
        Write single damage report to Supabase.

        Args:
            report: Damage report dict

        Returns:
            True if successful, False otherwise
        """
        if not self.use_supabase:
            return False

        # Prepare data for Supabase (remove VLM-specific fields)
        supabase_data = {
            "event_id": report["event_id"],
            "epoch": report["epoch"],
            "lat": report["lat"],
            "lon": report["lon"],
            "severity": report["severity"],
            "damage_type": report["damage_type"],
            "building": report["building"],
            "description": report["description"],
            "drone_id": report["drone_id"],
            "confidence": report["confidence"],
        }

        try:
            ok = sb.insert("drone_reports", supabase_data)
            if ok:
                print(f"  [OK] Supabase: {report['severity']} | {report['damage_type'][:50]}")
            else:
                print(f"  [FAIL] Supabase write failed")
            return ok
        except Exception as e:
            print(f"  [ERROR] Supabase error: {e}")
            return False

    def write_batch(self, reports: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Write batch of reports.

        Args:
            reports: List of damage reports

        Returns:
            Stats dict with success/failure counts
        """
        stats = {"success": 0, "failed": 0}

        for report in reports:
            if self.write_report(report):
                stats["success"] += 1
            else:
                stats["failed"] += 1

        return stats

    def save_json_reports(self, reports: List[Dict[str, Any]], output_path: str) -> bool:
        """
        Save reports as JSONL file.

        Args:
            reports: List of damage reports
            output_path: Path to output file

        Returns:
            True if successful
        """
        try:
            with open(output_path, "w") as f:
                for report in reports:
                    f.write(json.dumps(report) + "\n")
            print(f"[OK] Saved {len(reports)} reports to {output_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save JSON reports: {e}")
            return False


def create_reporter(use_supabase: bool = True) -> SupabaseReporter:
    """Factory function to create reporter."""
    return SupabaseReporter(use_supabase=use_supabase)
