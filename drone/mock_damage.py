"""
mock_damage.py
Simulates drone damage assessment — posts random crack locations in Paphos to blackboard.

Usage:
    python drone/mock_damage.py                  # post once
    python drone/mock_damage.py --loop 10        # post every 10 seconds
"""

import argparse
import json
import random
import time
import uuid
from datetime import datetime, timezone

import requests

BLACKBOARD_URL = "https://blackboard.jass.school/blackboard"
AGENT_NAME = "drone_agent"

# Paphos LAND-ONLY zones (avoids sea west of ~32.41)
PAPHOS_ZONES = [
    # City center (inland)
    {"lat": (34.755, 34.775), "lon": (32.415, 32.435)},
    # Kato Paphos (east of harbor)
    {"lat": (34.750, 34.760), "lon": (32.418, 32.430)},
    # Upper Paphos (fully inland)
    {"lat": (34.770, 34.785), "lon": (32.420, 32.445)},
    # East residential
    {"lat": (34.760, 34.778), "lon": (32.435, 32.460)},
]

SEVERITY_LEVELS = ["low", "moderate", "high", "critical"]
DAMAGE_TYPES = [
    "Wall crack detected",
    "Roof collapse detected",
    "Foundation crack detected",
    "Structural beam damage",
    "Window frame displacement",
    "Facade partial collapse",
    "Column crack detected",
    "Road surface crack",
]

BUILDINGS = [
    "Residential building",
    "Commercial building",
    "School",
    "Hospital wing",
    "Church",
    "Hotel",
    "Warehouse",
    "Office building",
    "Apartment block",
    "Government building",
]


def generate_damage_report():
    """Generate a random damage report in Paphos (land only)."""
    zone = random.choice(PAPHOS_ZONES)
    lat = random.uniform(*zone["lat"])
    lon = random.uniform(*zone["lon"])
    severity = random.choice(SEVERITY_LEVELS)
    damage = random.choice(DAMAGE_TYPES)
    building = random.choice(BUILDINGS)

    return {
        "eventId": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "severity": severity,
        "damage_type": damage,
        "building": building,
        "description": f"{damage} at {building}",
        "drone_id": "JASS-DRONE-01",
        "confidence": round(random.uniform(0.7, 0.99), 2),
    }


def post_to_blackboard(report):
    """Post damage report to blackboard."""
    try:
        resp = requests.post(
            BLACKBOARD_URL,
            json={
                "agent": AGENT_NAME,
                "type": "damage_zone",
                "content": json.dumps(report),
                "confidence": report["confidence"],
            },
            timeout=5,
        )
        if resp.status_code == 200:
            print(f"Posted: {report['severity']:>8} | {report['lat']:.4f}, {report['lon']:.4f} | {report['description']}")
        else:
            print(f"Error: HTTP {resp.status_code}")
    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Mock drone damage assessment for Paphos")
    parser.add_argument("--loop", type=int, default=0,
                        help="Post every N seconds (0 = post once)")
    parser.add_argument("--count", type=int, default=1,
                        help="Number of damage reports per post")
    args = parser.parse_args()

    print(f"Drone damage simulator — Paphos")
    print(f"Blackboard: {BLACKBOARD_URL}")
    print(f"Agent: {AGENT_NAME}")
    print("-" * 60)

    if args.loop > 0:
        print(f"Posting {args.count} report(s) every {args.loop}s (Ctrl+C to stop)\n")
        try:
            while True:
                for _ in range(args.count):
                    report = generate_damage_report()
                    post_to_blackboard(report)
                time.sleep(args.loop)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        for _ in range(args.count):
            report = generate_damage_report()
            post_to_blackboard(report)


if __name__ == "__main__":
    main()
