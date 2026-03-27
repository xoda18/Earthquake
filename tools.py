"""
Extra tools for EarthAgent — reads sensor and drone data from Supabase.
Tracks last-seen timestamps to avoid re-reporting old events.
"""

import json
import os
import sys
import time
from strands import tool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import supabase_client as sb

# Track last seen epochs to prevent duplicate reports
_state = {
    "last_sensor_epoch": time.time(),   # start from NOW — ignore all old data
    "last_drone_epoch": time.time(),
}


@tool
def read_sensor_log(last_n: int = 10) -> str:
    """Read the last N earthquake detections from the sensor (for answering questions).
    Does NOT update the tracking epoch — use get_new_earthquakes for that.

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
    """Read the last N damage reports from the drone (for answering questions).
    Does NOT update the tracking epoch — use get_new_damage_reports for that.

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
def get_new_earthquakes() -> str:
    """Get earthquake detections that happened AFTER the last check.
    Automatically tracks the last-seen epoch. Returns only truly new events.
    If result is '[]' — there are no new earthquakes. Do NOT report anything.

    Returns:
        JSON array of new earthquake events since last check, or '[]' if none.
    """
    rows = sb.select("sensor_events", order_by="created_at", limit=20,
                      since_epoch=_state["last_sensor_epoch"])
    if rows:
        # Update tracking to newest epoch
        max_epoch = max(r.get("epoch", 0) for r in rows)
        if max_epoch > _state["last_sensor_epoch"]:
            _state["last_sensor_epoch"] = max_epoch
        return json.dumps(rows, indent=2)
    return "[]"


@tool
def get_new_damage_reports() -> str:
    """Get drone damage reports that happened AFTER the last check.
    Automatically tracks the last-seen epoch. Returns only truly new reports.
    If result is '[]' — there are no new damage reports. Do NOT report anything.

    Returns:
        JSON array of new damage reports since last check, or '[]' if none.
    """
    rows = sb.select("drone_reports", order_by="created_at", limit=20,
                      since_epoch=_state["last_drone_epoch"])
    if rows:
        max_epoch = max(r.get("epoch", 0) for r in rows)
        if max_epoch > _state["last_drone_epoch"]:
            _state["last_drone_epoch"] = max_epoch
        return json.dumps(rows, indent=2)
    return "[]"


def get_extra_tools():
    return [read_sensor_log, read_drone_log, get_new_earthquakes, get_new_damage_reports]
