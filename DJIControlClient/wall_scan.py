"""
wall_scan.py — Capture wall images from drone and send to VLM for crack analysis.

Modes:
  python wall_scan.py              — fly drone, capture images
  python wall_scan.py --simulate   — send existing scans/ images to VLM (no drone)
  python wall_scan.py --dir path/  — send images from a specific directory to VLM

Environment:
  VLM_URL           — VLM server endpoint (default: http://localhost:5060)
  ORCHESTRATOR_URL  — orchestrator endpoint (optional)
"""

import glob
import os
import sys
import time
from datetime import datetime

import requests
from DJIControlClient import DJIControlClient

VLM_URL = os.environ.get("VLM_URL", "http://localhost:5060")
ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "")

# ---- Drone configuration ----
DRONE_IP = "192.168.1.130"
DRONE_PORT = 8080

RISE_HEIGHT = 0.3
MOVE_DISTANCE = 1.0
NUM_PHOTOS = 3
SETTLE_TIME = 2
HEADING_TOLERANCE = 5  # degrees — correct if drift exceeds this

DEFAULT_SCANS_DIR = os.path.join(os.path.dirname(__file__), "scans")


# ---------------------------------------------------------------------------
# VLM + orchestrator helpers
# ---------------------------------------------------------------------------
def correct_heading(client, initial_heading):
    """If the drone has drifted more than HEADING_TOLERANCE degrees, rotate back."""
    current_heading = client.getHeading()  # returns a number
    delta = (current_heading - initial_heading + 180) % 360 - 180  # normalise to [-180, 180]
    if abs(delta) <= HEADING_TOLERANCE:
        return
    print(f"  [heading] Drift detected: {delta:+.1f}° (current={current_heading:.1f}, initial={initial_heading:.1f})")
    if delta > 0:
        client.rotateCounterClockwise(delta)
    else:
        client.rotateClockwise(-delta)
    time.sleep(SETTLE_TIME)
    print(f"  [heading] Corrected → {client.getHeading():.1f}°")


def send_to_vlm(image_path, run_id=""):
    """POST an image to the VLM /analyze endpoint for crack analysis."""
    filename = os.path.basename(image_path)
    try:
        with open(image_path, "rb") as f:
            resp = requests.post(
                f"{VLM_URL}/analyze",
                files={"image": (filename, f, "image/jpeg")},
                params={"run_id": run_id},
                timeout=600,
            )
        if resp.status_code == 200:
            result = resp.json()
            severity = result.get("analysis", {}).get("severity", "unknown")
            print(f"  [VLM] {filename}: severity={severity}, url={result.get('image_url')}")
            return result
        else:
            print(f"  [VLM] ERROR {resp.status_code}: {resp.text[:200]}")
            return None
    except requests.exceptions.ConnectionError:
        print(f"  [VLM] Cannot connect to {VLM_URL} — is the VLM server running?")
        return None
    except Exception as e:
        print(f"  [VLM] Failed to send {filename}: {e}")
        return None


def notify_orchestrator(run_id, analyzed, cracks):
    """Notify orchestrator that VLM analysis is complete."""
    if not ORCHESTRATOR_URL:
        return
    try:
        requests.post(f"{ORCHESTRATOR_URL}/step/done", json={
            "step": "vlm_analysis",
            "status": "success",
            "detail": f"analyzed={analyzed} cracks={cracks}",
            "run_id": run_id,
        }, timeout=5)
        print("[orchestrator] Notified: vlm_analysis done")
    except Exception as e:
        print(f"[orchestrator] Failed to notify: {e}")


# ---------------------------------------------------------------------------
# Live drone scan
# ---------------------------------------------------------------------------
def scan_wall():
    """Full wall-scanning flight."""

    run_id = f"run_{int(time.time())}"
    print(f"Run ID: {run_id}")

    # Output folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("scans", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Images will be saved to: {output_dir}")

    # Connect to drone
    print(f"Connecting to drone at {DRONE_IP}:{DRONE_PORT}...")
    client = DJIControlClient(DRONE_IP, DRONE_PORT)
    client.setLandingProtectionState(False)
    print("Drone connected")

    vlm_results = []

    try:
        # Take off
        print("\n[1] Taking off...")
        client.takeOff()
        time.sleep(5)
        client.pitchGimbal(0)  # look straight

        # Rise to scanning height
        client.moveUp(RISE_HEIGHT)
        time.sleep(SETTLE_TIME)

        # Record initial heading so we can correct drift later
        initial_heading = client.getHeading()  # returns a number
        print(f"  Initial heading: {initial_heading:.1f}°")

        # Capture images, moving right between each
        for i in range(NUM_PHOTOS):
            print(f"\n--- Photo {i + 1}/{NUM_PHOTOS} ---")

            print(f"  Moving right {MOVE_DISTANCE}m...")
            client.moveRight(MOVE_DISTANCE)
            time.sleep(SETTLE_TIME)

            # Correct heading drift before capturing
            correct_heading(client, initial_heading)

            photo_path = os.path.join(output_dir, f"wall_{i + 1}.jpg")
            client.takeImage(photo_path)
            print(f"  Saved: {photo_path}")

            result = send_to_vlm(photo_path, run_id)
            if result:
                vlm_results.append(result)

        # Return to start position
        total_distance = NUM_PHOTOS * MOVE_DISTANCE
        print(f"\n[3] Returning: moving left {total_distance}m...")
        client.moveLeft(total_distance)
        time.sleep(SETTLE_TIME)

        # Land
        print("[4] Landing...")
        client.land()
        print("Landed successfully")

    except Exception as e:
        print(f"\nERROR: {e}")

    analyzed = len(vlm_results)
    cracks = sum(1 for r in vlm_results
                 if r.get("analysis", {}).get("severity", "none") not in ("none", "unknown"))
    print(f"\nDone! {analyzed} images analyzed, {cracks} with damage detected.")
    print(f"Local copies in: {output_dir}")
    notify_orchestrator(run_id, analyzed, cracks)


# ---------------------------------------------------------------------------
# Simulation mode — replay existing images through VLM
# ---------------------------------------------------------------------------
def find_latest_scan_dir(base_dir):
    """Find the most recent timestamped subdirectory in scans/."""
    subdirs = sorted(d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d))
    return subdirs[-1] if subdirs else None


def find_images(directory):
    """Find all jpg/png images in a directory."""
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(images)


def simulate(image_dir):
    """Send existing images from disk through the VLM pipeline."""
    run_id = f"run_{int(time.time())}"

    images = find_images(image_dir)
    if not images:
        print(f"No images found in {image_dir}")
        return

    print(f"Run ID:    {run_id}")
    print(f"VLM:       {VLM_URL}/analyze")
    print(f"Images:    {len(images)} from {image_dir}")
    print()

    vlm_results = []
    for i, path in enumerate(images):
        print(f"--- [{i + 1}/{len(images)}] {os.path.basename(path)} ---")
        result = send_to_vlm(path, run_id)
        if result:
            vlm_results.append(result)

    analyzed = len(vlm_results)
    cracks = sum(1 for r in vlm_results
                 if r.get("analysis", {}).get("severity", "none") not in ("none", "unknown"))
    print(f"\nDone! {analyzed} images analyzed, {cracks} with damage detected.")
    notify_orchestrator(run_id, analyzed, cracks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)

    elif "--simulate" in sys.argv:
        image_dir = find_latest_scan_dir(DEFAULT_SCANS_DIR)
        if image_dir is None:
            print(f"No scan directories found in {DEFAULT_SCANS_DIR}")
            sys.exit(1)
        simulate(image_dir)

    elif "--dir" in sys.argv:
        idx = sys.argv.index("--dir")
        if idx + 1 >= len(sys.argv):
            print("Usage: python wall_scan.py --dir <path>")
            sys.exit(1)
        simulate(sys.argv[idx + 1])

    else:
        scan_wall()
