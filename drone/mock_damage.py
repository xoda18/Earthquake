"""
mock_damage.py
Simulates drone damage assessment — writes crack locations to drone_log.jsonl.

Usage:
    python drone/mock_damage.py                  # write once
    python drone/mock_damage.py --loop 10        # write every 10 seconds
    python drone/mock_damage.py --count 5        # 5 damage reports
"""

import argparse
import json
import os
import random
import time
import uuid
from datetime import datetime, timezone

DRONE_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drone_log.jsonl")

# Paphos LAND-ONLY zones
PAPHOS_ZONES = [
    {"lat": (34.755, 34.775), "lon": (32.415, 32.435)},
    {"lat": (34.750, 34.760), "lon": (32.418, 32.430)},
    {"lat": (34.770, 34.785), "lon": (32.420, 32.445)},
    {"lat": (34.760, 34.778), "lon": (32.435, 32.460)},
]

SEVERITY_LEVELS = ["low", "moderate", "high", "critical"]
DAMAGE_TYPES = [
    "Wall crack detected", "Roof collapse detected", "Foundation crack detected",
    "Structural beam damage", "Window frame displacement", "Facade partial collapse",
    "Column crack detected", "Road surface crack",
]
BUILDINGS = [
    "Residential building", "Commercial building", "School", "Hospital wing",
    "Church", "Hotel", "Warehouse", "Office building", "Apartment block",
]


def generate_and_log():
    """Generate a random damage report and write to drone_log.jsonl."""
    zone = random.choice(PAPHOS_ZONES)
    lat = round(random.uniform(*zone["lat"]), 6)
    lon = round(random.uniform(*zone["lon"]), 6)
    severity = random.choice(SEVERITY_LEVELS)
    damage = random.choice(DAMAGE_TYPES)
    building = random.choice(BUILDINGS)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "epoch": time.time(),
        "type": "damage",
        "eventId": str(uuid.uuid4()),
        "lat": lat,
        "lon": lon,
        "severity": severity,
        "damage_type": damage,
        "building": building,
        "description": f"{damage} at {building}",
        "drone_id": "JASS-DRONE-01",
        "confidence": round(random.uniform(0.7, 0.99), 2),
    }

    with open(DRONE_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"Logged: {severity:>8} | {lat:.4f}, {lon:.4f} | {entry['description']}")
    return entry


def main():
    parser = argparse.ArgumentParser(description="Mock drone damage → drone_log.jsonl")
    parser.add_argument("--loop", type=int, default=0, help="Write every N seconds (0 = once)")
    parser.add_argument("--count", type=int, default=1, help="Reports per cycle")
    args = parser.parse_args()

    print(f"Drone damage simulator → {DRONE_LOG}")
    print("-" * 60)

    if args.loop > 0:
        print(f"Writing {args.count} report(s) every {args.loop}s (Ctrl+C to stop)\n")
        try:
            while True:
                for _ in range(args.count):
                    generate_and_log()
                time.sleep(args.loop)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        for _ in range(args.count):
            generate_and_log()


if __name__ == "__main__":
    main()
