"""
dispatch.py
Send drone dispatch command to blackboard (earthquake detected → drone fly out).

Usage:
    # Dispatch drone to scan Paphos
    python drone/dispatch.py --lat 34.765 --lon 32.420 --pga 0.15

    # Dispatch with default Paphos center
    python drone/dispatch.py
"""

import argparse
import json
import time
import requests

BLACKBOARD_URL = "https://blackboard.jass.school/blackboard"
AGENT_NAME = "earthquake"


def dispatch_drone(lat=34.765, lon=32.420, pga=0.1):
    """Post drone dispatch command to blackboard."""
    content = {
        "command": "dispatch",
        "target_lat": lat,
        "target_lon": lon,
        "pga": round(pga, 4),
        "scan_radius_m": 500,
        "timestamp": time.time(),
    }
    resp = requests.post(
        BLACKBOARD_URL,
        json={
            "agent": AGENT_NAME,
            "type": "drone_dispatch",
            "content": json.dumps(content),
            "confidence": 1.0,
        },
        timeout=5,
    )
    if resp.status_code == 200:
        print(f"Drone dispatched to ({lat}, {lon}) | PGA={pga}g")
    else:
        print(f"Error: HTTP {resp.status_code}")


def main():
    parser = argparse.ArgumentParser(description="Dispatch drone after earthquake")
    parser.add_argument("--lat", type=float, default=34.765, help="Target latitude")
    parser.add_argument("--lon", type=float, default=32.420, help="Target longitude")
    parser.add_argument("--pga", type=float, default=0.1, help="Peak ground acceleration")
    args = parser.parse_args()
    dispatch_drone(args.lat, args.lon, args.pga)


if __name__ == "__main__":
    main()
