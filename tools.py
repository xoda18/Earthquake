"""
Extra tools for EarthAgent — reads sensor and drone data from Supabase.
"""

import json
import os
import sys
from strands import tool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import supabase_client as sb


@tool
def read_sensor_log(last_n: int = 10) -> str:
    """Read the last N earthquake detections from the sensor.
    Each entry has: epoch, probability, pga_g, ax, ay, az, magnitude_g.

    Args:
        last_n: Number of recent entries to return.

    Returns:
        JSON array of recent earthquake detections, or '(no detections)'.
    """
    rows = sb.select("sensor_events", order_by="created_at", limit=last_n)
    if not rows:
        return "(no detections)"
    return json.dumps(rows, indent=2)


@tool
def read_drone_log(last_n: int = 10) -> str:
    """Read the last N damage reports from the drone.
    Each entry has: lat, lon, severity, damage_type, building, confidence.

    Args:
        last_n: Number of recent entries to return.

    Returns:
        JSON array of recent drone damage reports, or '(no reports)'.
    """
    rows = sb.select("drone_reports", order_by="created_at", limit=last_n)
    if not rows:
        return "(no reports)"
    return json.dumps(rows, indent=2)


@tool
def get_new_earthquakes(since_epoch: float = 0) -> str:
    """Get earthquake detections newer than a given timestamp.
    Use this to poll for new events since your last check.

    Args:
        since_epoch: Unix timestamp — only return events after this time.

    Returns:
        JSON array of new earthquake events, or '[]' if none.
    """
    rows = sb.select("sensor_events", order_by="created_at", limit=50, since_epoch=since_epoch)
    return json.dumps(rows, indent=2) if rows else "[]"


@tool
def get_new_damage_reports(since_epoch: float = 0) -> str:
    """Get drone damage reports newer than a given timestamp.
    Use this to poll for new crack/damage findings since your last check.

    Args:
        since_epoch: Unix timestamp — only return reports after this time.

    Returns:
        JSON array of new damage reports, or '[]' if none.
    """
    rows = sb.select("drone_reports", order_by="created_at", limit=50, since_epoch=since_epoch)
    return json.dumps(rows, indent=2) if rows else "[]"


def get_extra_tools():
    return [read_sensor_log, read_drone_log, get_new_earthquakes, get_new_damage_reports]
