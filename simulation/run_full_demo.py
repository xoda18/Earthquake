"""
run_full_demo.py
Full earthquake response simulation:

1. Earthquake detected (LSTM probability posted to blackboard)
2. Drone dispatched to scan area
3. Drone flies, finds damage, posts damage_zone reports
4. Rescuers receive alerts, dispatch to critical zones
5. Rescuers dismiss resolved zones

Usage:
    python simulation/run_full_demo.py
    python simulation/run_full_demo.py --fast    # shorter delays
"""

import argparse
import json
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone

import requests

BLACKBOARD_URL = "https://blackboard.jass.school/blackboard"

# Paphos land zones
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


def post(agent, type_, content, confidence=1.0):
    """Post to blackboard."""
    requests.post(
        BLACKBOARD_URL,
        json={
            "agent": agent,
            "type": type_,
            "content": json.dumps(content) if isinstance(content, dict) else content,
            "confidence": confidence,
        },
        timeout=5,
    )


def wait(seconds, fast=False):
    t = max(0.5, seconds * (0.3 if fast else 1.0))
    time.sleep(t)


def run_simulation(fast=False):
    print("=" * 60)
    print("  JASS EARTHQUAKE RESPONSE SIMULATION")
    print("=" * 60)

    # ── Phase 1: Earthquake Detection ────────────────────────
    print("\n[PHASE 1] Earthquake Detection")
    print("-" * 40)

    pga = round(random.uniform(0.1, 0.6), 3)
    prob = round(random.uniform(0.85, 1.0), 4)
    eq_lat = round(random.uniform(34.755, 34.775), 6)
    eq_lon = round(random.uniform(32.415, 32.435), 6)

    print(f"  Sensor detected seismic activity...")
    wait(2, fast)

    post("earthquake", "earthquake_sensor", {
        "timestamp": time.time(),
        "ax": round(random.uniform(-0.5, 0.5), 3),
        "ay": round(random.uniform(-0.5, 0.5), 3),
        "az": round(random.uniform(-1.5, -0.5), 3),
        "magnitude_g": round(random.uniform(1.0, 2.5), 4),
        "amplitude_g": pga,
        "probability": prob,
        "label": "earthquake",
        "profile": "lstm",
    }, confidence=prob)

    print(f"  EARTHQUAKE DETECTED!")
    print(f"  LSTM Probability: {prob:.1%}")
    print(f"  PGA: {pga}g")
    print(f"  Location: ({eq_lat}, {eq_lon})")
    wait(2, fast)

    # ── Phase 2: Drone Dispatch ──────────────────────────────
    print("\n[PHASE 2] Drone Dispatch")
    print("-" * 40)

    print(f"  Dispatching drone to ({eq_lat}, {eq_lon})...")
    post("earthquake", "drone_dispatch", {
        "command": "dispatch",
        "target_lat": eq_lat,
        "target_lon": eq_lon,
        "pga": pga,
        "scan_radius_m": 500,
        "timestamp": time.time(),
    })
    wait(1, fast)

    # Drone status updates
    for status in ["takeoff", "en_route", "scanning"]:
        post("drone_agent", "drone_status", {
            "status": status,
            "lat": eq_lat, "lon": eq_lon,
            "timestamp": time.time(),
        })
        print(f"  Drone: {status}")
        wait(2, fast)

    # ── Phase 3: Damage Assessment ───────────────────────────
    print("\n[PHASE 3] Damage Assessment")
    print("-" * 40)

    n_findings = random.randint(4, 8)
    damage_reports = []

    for i in range(n_findings):
        zone = random.choice(PAPHOS_ZONES)
        lat = round(random.uniform(*zone["lat"]), 6)
        lon = round(random.uniform(*zone["lon"]), 6)
        severity = random.choices(
            SEVERITY_LEVELS,
            weights=[3, 4, 2, 1],  # mostly moderate/low
        )[0]
        damage = random.choice(DAMAGE_TYPES)
        building = random.choice(BUILDINGS)
        event_id = str(uuid.uuid4())

        report = {
            "eventId": event_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "lat": lat, "lon": lon,
            "severity": severity,
            "damage_type": damage,
            "building": building,
            "description": f"{damage} at {building}",
            "drone_id": "JASS-DRONE-01",
            "confidence": round(random.uniform(0.7, 0.99), 2),
        }
        damage_reports.append(report)

        post("drone_agent", "damage_zone", report, confidence=report["confidence"])
        print(f"  Found: {severity:>8} | {damage} at {building}")
        print(f"         ({lat:.4f}, {lon:.4f}) | ID: {event_id[:8]}...")
        wait(random.uniform(1, 3), fast)

    # Drone returns
    for status in ["returning", "landed"]:
        post("drone_agent", "drone_status", {
            "status": status,
            "lat": eq_lat, "lon": eq_lon,
            "timestamp": time.time(),
        })
        print(f"  Drone: {status}")
        wait(1, fast)

    # ── Phase 4: Rescue Dispatch ─────────────────────────────
    print("\n[PHASE 4] Rescue Response")
    print("-" * 40)

    critical_zones = [r for r in damage_reports if r["severity"] in ("critical", "high")]
    if not critical_zones:
        critical_zones = damage_reports[:2]

    print(f"  {len(critical_zones)} priority zones for rescue teams:")
    for r in critical_zones:
        print(f"    -> {r['severity']:>8} | {r['description']} ({r['lat']:.4f}, {r['lon']:.4f})")

    wait(3, fast)

    # ── Phase 5: Rescuers Resolve Zones ──────────────────────
    print("\n[PHASE 5] Zone Resolution")
    print("-" * 40)

    resolved = random.sample(damage_reports, min(3, len(damage_reports)))
    for r in resolved:
        wait(random.uniform(2, 4), fast)
        post("rescue_control", "dismiss_zone", {
            "eventId": r["eventId"],
            "timestamp": time.time(),
        })
        print(f"  Resolved: {r['description']}")
        print(f"            Zone {r['eventId'][:8]}... dismissed from map")

    # ── Summary ──────────────────────────────────────────────
    remaining = len(damage_reports) - len(resolved)
    print("\n" + "=" * 60)
    print("  SIMULATION COMPLETE")
    print("=" * 60)
    print(f"  Earthquake PGA:     {pga}g")
    print(f"  Damage zones found: {n_findings}")
    print(f"  Zones resolved:     {len(resolved)}")
    print(f"  Zones remaining:    {remaining}")
    print(f"\n  Open http://localhost:8080 to see the map!")


def main():
    parser = argparse.ArgumentParser(description="Full earthquake response simulation")
    parser.add_argument("--fast", action="store_true", help="Speed up delays")
    args = parser.parse_args()
    run_simulation(fast=args.fast)


if __name__ == "__main__":
    main()
