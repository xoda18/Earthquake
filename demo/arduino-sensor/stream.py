"""
stream.py — Stream MPU6500 data with optional real-time detection.

Usage:
    python demo/arduino-sensor/stream.py
    python demo/arduino-sensor/stream.py --save data.csv
    python demo/arduino-sensor/stream.py --save data.csv --detect
    python demo/arduino-sensor/stream.py --port /dev/ttyACM0 --rate 10 --detect
"""

import argparse
import csv
import sys
import os
import time
import glob
import threading
import serial
import numpy as np

# Add demo/ to path so detect_earthquake is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from detect_earthquake import load_config, preprocess, sta_lta, detect_spikes, find_event_window


def auto_detect_port():
    candidates = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    if candidates:
        return candidates[0]
    raise RuntimeError("No serial port found. Is the Arduino plugged in?")


def run_detection(buf, lock, result):
    """Background thread: run STA/LTA detection on rolling buffer."""
    while result["running"]:
        time.sleep(0.5)  # analyze twice per second
        with lock:
            if len(buf) < 200:
                continue
            window = np.array(buf[-600:])  # last 6 seconds

        mags = np.sqrt(window[:, 0]**2 + window[:, 1]**2 + window[:, 2]**2)
        try:
            filtered = preprocess(mags, 100)
            ratio = sta_lta(filtered, 100)
            spikes = detect_spikes(ratio, filtered, 100)
            event = find_event_window(spikes, ratio, 100)
        except Exception:
            continue

        if event and event != result["last_event"]:
            result["last_event"] = event
            result["count"] += 1
            result["alert"] = f"EVENT #{result['count']}"
        elif not event and result["alert"]:
            result["alert"] = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None)
    parser.add_argument("--save", default=None, metavar="FILE")
    parser.add_argument("--rate", type=int, default=10, help="Print rows per second (default 10)")
    parser.add_argument("--detect", action="store_true", help="Run real-time earthquake detection")
    parser.add_argument("--mode", default="table_knock", choices=["earthquake", "table_knock"])
    args = parser.parse_args()

    port = args.port or auto_detect_port()

    print(f"Connecting to {port}...")
    ser = serial.Serial(port, 115200, timeout=2)
    time.sleep(1.5)
    ser.reset_input_buffer()
    for _ in range(5):
        ser.readline()

    csv_file = None
    writer = None
    if args.save:
        csv_file = open(args.save, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["timestamp_s", "ax", "ay", "az", "gx", "gy", "gz"])

    # Detection setup
    detect_result = {"running": True, "last_event": None, "count": 0, "alert": None}
    buf = []
    buf_lock = threading.Lock()

    if args.detect:
        load_config(args.mode)
        t = threading.Thread(target=run_detection, args=(buf, buf_lock, detect_result), daemon=True)
        t.start()

    print_interval = 1.0 / args.rate
    last_print = 0
    sample_count = 0
    start = time.time()
    mode_label = f" | Detection: {args.mode}" if args.detect else ""

    print(f"Streaming 100 Hz | Print {args.rate}/s{mode_label} | Ctrl+C to stop\n")
    if args.detect:
        print(f"{'Time':>8}  {'ax':>8}  {'ay':>8}  {'az':>8}  {'|a|':>8}  {'Status'}")
        print("-" * 60)
    else:
        print(f"{'Time':>8}  {'ax':>8}  {'ay':>8}  {'az':>8}  {'|a|':>8}  {'gx':>7}  {'gy':>7}  {'gz':>7}")
        print("-" * 76)

    try:
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line or line.startswith("timestamp") or line.startswith("#"):
                continue

            parts = line.split(",")
            if len(parts) not in (4, 7):
                continue

            try:
                ts = float(parts[0]) / 1000.0
                ax, ay, az = float(parts[1]), float(parts[2]), float(parts[3])
                gx = float(parts[4]) if len(parts) == 7 else 0.0
                gy = float(parts[5]) if len(parts) == 7 else 0.0
                gz = float(parts[6]) if len(parts) == 7 else 0.0
            except ValueError:
                continue

            mag = (ax**2 + ay**2 + az**2) ** 0.5
            sample_count += 1

            if writer:
                writer.writerow([f"{ts:.3f}", ax, ay, az, gx, gy, gz])

            if args.detect:
                with buf_lock:
                    buf.append((ax, ay, az))

            now = time.time()
            if now - last_print >= print_interval:
                last_print = now
                if args.detect:
                    status = detect_result["alert"] or "OK"
                    marker = " <-- !!!" if detect_result["alert"] else ""
                    print(f"{ts:8.2f}  {ax:8.4f}  {ay:8.4f}  {az:8.4f}  {mag:8.4f}  {status}{marker}")
                else:
                    print(f"{ts:8.2f}  {ax:8.4f}  {ay:8.4f}  {az:8.4f}  {mag:8.4f}  {gx:7.3f}  {gy:7.3f}  {gz:7.3f}")

    except KeyboardInterrupt:
        detect_result["running"] = False
        elapsed = time.time() - start
        print(f"\nStopped | {elapsed:.1f}s | {sample_count} samples | {sample_count/elapsed:.1f} Hz")
        if args.detect:
            print(f"Total events detected: {detect_result['count']}")
    finally:
        ser.close()
        if csv_file:
            csv_file.close()
            print(f"Saved → {args.save}")


if __name__ == "__main__":
    main()
