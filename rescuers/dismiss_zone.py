"""
dismiss_zone.py
Rescuer tool — dismiss damage zones from the map.

Posts dismiss_zone / dismiss_all_zones to blackboard.
The Leaflet map reads these and hides dismissed markers.

Usage:
    # Dismiss one zone by eventId
    python rescuers/dismiss_zone.py --id <eventId>

    # Dismiss ALL zones
    python rescuers/dismiss_zone.py --all
"""

import argparse
import json
import time
import requests

BLACKBOARD_URL = "https://blackboard.jass.school/blackboard"
AGENT_NAME = "rescue_control"


def dismiss_one(event_id):
    """Dismiss a single damage zone by eventId."""
    resp = requests.post(
        BLACKBOARD_URL,
        json={
            "agent": AGENT_NAME,
            "type": "dismiss_zone",
            "content": json.dumps({"eventId": event_id, "timestamp": time.time()}),
            "confidence": 1.0,
        },
        timeout=5,
    )
    if resp.status_code == 200:
        print(f"Dismissed zone: {event_id}")
    else:
        print(f"Error: HTTP {resp.status_code}")


def dismiss_all():
    """Dismiss all damage zones."""
    resp = requests.post(
        BLACKBOARD_URL,
        json={
            "agent": AGENT_NAME,
            "type": "dismiss_all_zones",
            "content": json.dumps({"timestamp": time.time()}),
            "confidence": 1.0,
        },
        timeout=5,
    )
    if resp.status_code == 200:
        print("Dismissed ALL zones")
    else:
        print(f"Error: HTTP {resp.status_code}")


def main():
    parser = argparse.ArgumentParser(description="Rescuer — dismiss damage zones")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--id", type=str, help="Dismiss zone by eventId")
    group.add_argument("--all", action="store_true", help="Dismiss all zones")
    args = parser.parse_args()

    if args.all:
        dismiss_all()
    else:
        dismiss_one(args.id)


if __name__ == "__main__":
    main()
