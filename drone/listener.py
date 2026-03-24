"""
listener.py
Drone listens for dispatch commands on blackboard (GET polling).

When a drone_dispatch command is found, starts scanning and posting damage reports.

Usage:
    python drone/listener.py
    python drone/listener.py --poll 3    # poll every 3 seconds
"""

import argparse
import json
import time
import random
import uuid
from datetime import datetime, timezone

import requests

BLACKBOARD_URL = "https://blackboard.jass.school/blackboard"
AGENT_NAME = "drone_agent"

# Import land zones from mock_damage
PAPHOS_ZONES = [
    {"lat": (34.755, 34.775), "lon": (32.415, 32.435)},
    {"lat": (34.750, 34.760), "lon": (32.418, 32.430)},
    {"lat": (34.770, 34.785), "lon": (32.420, 32.445)},
    {"lat": (34.760, 34.778), "lon": (32.435, 32.460)},
]

SEVERITY_LEVELS = ["low", "moderate", "high", "critical"]
DAMAGE_TYPES = [
    "Wall crack detected", "Roof collapse detected", "Foundation crack detected",
    "Structural beam damage", "Road surface crack", "Facade partial collapse",
]
BUILDINGS = [
    "Residential building", "Commercial building", "School", "Hospital wing",
    "Hotel", "Warehouse", "Office building", "Apartment block",
]


def post_drone_status(status, lat=0, lon=0):
    """Post drone status update."""
    requests.post(
        BLACKBOARD_URL,
        json={
            "agent": AGENT_NAME,
            "type": "drone_status",
            "content": json.dumps({
                "status": status,
                "lat": lat, "lon": lon,
                "timestamp": time.time(),
            }),
            "confidence": 1.0,
        },
        timeout=5,
    )
    print(f"  [{status}] ({lat:.4f}, {lon:.4f})")


def post_damage(lat, lon):
    """Post one damage report."""
    report = {
        "eventId": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "severity": random.choice(SEVERITY_LEVELS),
        "damage_type": random.choice(DAMAGE_TYPES),
        "building": random.choice(BUILDINGS),
        "description": f"{random.choice(DAMAGE_TYPES)} at {random.choice(BUILDINGS)}",
        "drone_id": "JASS-DRONE-01",
        "confidence": round(random.uniform(0.7, 0.99), 2),
    }
    requests.post(
        BLACKBOARD_URL,
        json={
            "agent": AGENT_NAME,
            "type": "damage_zone",
            "content": json.dumps(report),
            "confidence": report["confidence"],
        },
        timeout=5,
    )
    print(f"  [DAMAGE] {report['severity']:>8} | {report['description']}")


def scan_area(target_lat, target_lon, n_findings=None):
    """Simulate drone scanning area — fly, find damage, report."""
    if n_findings is None:
        n_findings = random.randint(3, 8)

    print(f"\nDrone scanning area around ({target_lat:.4f}, {target_lon:.4f})...")
    post_drone_status("takeoff", target_lat, target_lon)
    time.sleep(2)

    post_drone_status("en_route", target_lat, target_lon)
    time.sleep(3)

    post_drone_status("scanning", target_lat, target_lon)

    for i in range(n_findings):
        time.sleep(random.uniform(2, 5))
        zone = random.choice(PAPHOS_ZONES)
        lat = random.uniform(*zone["lat"])
        lon = random.uniform(*zone["lon"])
        post_damage(lat, lon)

    time.sleep(2)
    post_drone_status("returning", target_lat, target_lon)
    time.sleep(2)
    post_drone_status("landed", target_lat, target_lon)
    print(f"\nScan complete. Found {n_findings} damage zones.")


def poll_for_dispatch(interval=5):
    """Poll blackboard for drone_dispatch commands."""
    print(f"Drone listener started — polling every {interval}s")
    print(f"Waiting for drone_dispatch command...\n")

    seen_dispatches = set()

    while True:
        try:
            resp = requests.get(BLACKBOARD_URL, timeout=5)
            if resp.status_code != 200:
                time.sleep(interval)
                continue

            data = resp.json()
            entries = data.get("entries", [])

            for entry in entries:
                if entry.get("type") != "drone_dispatch":
                    continue

                try:
                    content = json.loads(entry["content"]) if isinstance(entry["content"], str) else entry["content"]
                except (json.JSONDecodeError, TypeError):
                    continue

                ts = content.get("timestamp", 0)
                dispatch_id = f"{ts}"

                if dispatch_id in seen_dispatches:
                    continue

                seen_dispatches.add(dispatch_id)
                lat = content.get("target_lat", 34.765)
                lon = content.get("target_lon", 32.420)
                print(f"DISPATCH RECEIVED! Target: ({lat}, {lon})")
                scan_area(lat, lon)

        except Exception as e:
            print(f"Poll error: {e}")

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Drone listener — waits for dispatch commands")
    parser.add_argument("--poll", type=int, default=5, help="Poll interval in seconds")
    args = parser.parse_args()

    try:
        poll_for_dispatch(args.poll)
    except KeyboardInterrupt:
        print("\nDrone listener stopped.")


if __name__ == "__main__":
    main()
