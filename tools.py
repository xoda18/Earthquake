"""
Extra tools for EarthAgent — reads sensor and drone logs.
Paths are relative to this file (works both standalone and as swarm submodule).
"""

import json
import os
from strands import tool

_REPO = os.path.dirname(os.path.abspath(__file__))
SENSOR_LOG = os.path.join(_REPO, "detection", "sensor_log.jsonl")
DRONE_LOG = os.path.join(_REPO, "drone", "drone_log.jsonl")


@tool
def read_sensor_log(last_n: int = 10) -> str:
    """Read the last N earthquake detections from the LSTM sensor log.
    Each entry has: timestamp, probability, pga_g, ax, ay, az, magnitude_g.

    Args:
        last_n: Number of recent entries to return.

    Returns:
        JSON lines of recent earthquake detections, or '(no detections)'.
    """
    if not os.path.exists(SENSOR_LOG):
        return "(no detections — sensor log not found)"
    try:
        with open(SENSOR_LOG, "r") as f:
            lines = f.readlines()
        recent = lines[-last_n:] if len(lines) >= last_n else lines
        if not recent:
            return "(no detections)"
        entries = [json.loads(line.strip()) for line in recent if line.strip()]
        return json.dumps(entries, indent=2)
    except Exception as e:
        return f"(error reading sensor log: {e})"


@tool
def read_drone_log(last_n: int = 10) -> str:
    """Read the last N damage reports from the drone log.
    Each entry has: timestamp, lat, lon, severity, damage_type, building, confidence.

    Args:
        last_n: Number of recent entries to return.

    Returns:
        JSON lines of recent drone damage reports, or '(no reports)'.
    """
    if not os.path.exists(DRONE_LOG):
        return "(no reports — drone log not found)"
    try:
        with open(DRONE_LOG, "r") as f:
            lines = f.readlines()
        recent = lines[-last_n:] if len(lines) >= last_n else lines
        if not recent:
            return "(no reports)"
        entries = [json.loads(line.strip()) for line in recent if line.strip()]
        return json.dumps(entries, indent=2)
    except Exception as e:
        return f"(error reading drone log: {e})"


@tool
def get_new_earthquakes(since_epoch: float = 0) -> str:
    """Get earthquake detections newer than a given timestamp.
    Use this to poll for new events since your last check.

    Args:
        since_epoch: Unix timestamp — only return events after this time.

    Returns:
        JSON array of new earthquake events, or '[]' if none.
    """
    if not os.path.exists(SENSOR_LOG):
        return "[]"
    try:
        with open(SENSOR_LOG, "r") as f:
            lines = f.readlines()
        new_events = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("epoch", 0) > since_epoch:
                new_events.append(entry)
        return json.dumps(new_events, indent=2)
    except Exception as e:
        return f"(error: {e})"


@tool
def get_new_damage_reports(since_epoch: float = 0) -> str:
    """Get drone damage reports newer than a given timestamp.
    Use this to poll for new crack/damage findings since your last check.

    Args:
        since_epoch: Unix timestamp — only return reports after this time.

    Returns:
        JSON array of new damage reports, or '[]' if none.
    """
    if not os.path.exists(DRONE_LOG):
        return "[]"
    try:
        with open(DRONE_LOG, "r") as f:
            lines = f.readlines()
        new_reports = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("epoch", 0) > since_epoch:
                new_reports.append(entry)
        return json.dumps(new_reports, indent=2)
    except Exception as e:
        return f"(error: {e})"


def get_extra_tools():
    return [read_sensor_log, read_drone_log, get_new_earthquakes, get_new_damage_reports]
