"""
infer.py — Reads Arduino serial, classifies with LSTM model, posts to ORB blackboard.

Serial reading runs in a separate thread to prevent buffer overflow.

Usage:
    docker build -f detection/streaming/Dockerfile -t earthquake-detector .
    docker run --rm --device /dev/ttyACM0 earthquake-detector
    docker run --rm --device /dev/ttyACM0 earthquake-detector --profile table
"""

import argparse
import json
import sys
import time
import glob
import pickle
import numpy as np
import serial
import os
import requests
import threading
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow import keras

BLACKBOARD_URL = "https://blackboard.jass.school/blackboard"
AGENT_NAME = "EarthQuakeSensorProvider"

PROFILES = {
    "default": {"gain": 1.0, "threshold": 0.7, "rate": 2.0, "description": "Balanced"},
    "table":   {"gain": 3.0, "threshold": 0.8, "rate": 2.0, "description": "Table demo — knocks detected"},
    "sensitive":{"gain": 5.0, "threshold": 0.6, "rate": 2.0, "description": "High sensitivity"},
    "earthquake":{"gain": 1.0, "threshold": 0.9, "rate": 1.0, "description": "Real seismic only"},
}


def auto_detect_port():
    candidates = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    if candidates:
        return candidates[0]
    raise RuntimeError("No serial port")


def load_model():
    model = keras.models.load_model('/app/model.h5', compile=False)
    with open('/app/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


def classify_window(model, scaler, window_100x3):
    X = window_100x3.T.reshape(1, 3, 100)
    X_flat = X.reshape(-1, 100)
    X_flat = scaler.transform(X_flat)
    X = X_flat.reshape(1, 3, 100)
    return float(model.predict(X, verbose=0)[0][0])


def post_to_blackboard(data: dict):
    try:
        requests.post(BLACKBOARD_URL, json={
            "agent": AGENT_NAME,
            "type": "earthquake_sensor",
            "content": json.dumps(data),
            "confidence": data.get("probability", 0),
        }, timeout=3)
    except Exception:
        pass


def serial_reader_thread(ser, sample_queue, running):
    """Continuously reads serial and puts parsed samples into a queue.
    Never blocks on anything except serial.readline()."""
    while running[0]:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line or line.startswith("timestamp") or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            ax, ay, az = float(parts[1]), float(parts[2]), float(parts[3])
            sample_queue.append((ax, ay, az))
        except Exception:
            continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None)
    parser.add_argument("--profile", default="default", choices=list(PROFILES.keys()))
    parser.add_argument("--rate", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--gain", type=float, default=None)
    parser.add_argument("--mode", default="probability", choices=["probability", "magnitude"])
    parser.add_argument("--no-blackboard", action="store_true")
    parser.add_argument("--stdin", action="store_true", help="Read CSV from stdin instead of serial port")
    args = parser.parse_args()

    p = PROFILES[args.profile]
    gain = args.gain if args.gain is not None else p["gain"]
    threshold = args.threshold if args.threshold is not None else p["threshold"]
    rate = args.rate if args.rate is not None else p["rate"]
    use_blackboard = not args.no_blackboard

    print("Loading LSTM model...", flush=True)
    model, scaler = load_model()
    print("Model loaded.", flush=True)

    # Input source: serial port or stdin
    ser = None
    if args.stdin:
        print("Reading from stdin...", flush=True)
    else:
        port = args.port or auto_detect_port()
        print(f"Connecting to {port}...", flush=True)
        ser = serial.Serial(port, 115200, timeout=1)
        time.sleep(1)
        ser.reset_input_buffer()
        ser.read(ser.in_waiting or 1)
        print(f"Connected.", flush=True)

    print(f"\n  Profile:    {args.profile} — {p['description']}", flush=True)
    print(f"  Gain:       {gain}x | Threshold: {threshold} | Rate: {rate}/s", flush=True)
    print(f"  Input:      {'stdin' if args.stdin else port}", flush=True)
    print(f"  Blackboard: {'ON' if use_blackboard else 'OFF'}", flush=True)
    print(flush=True)

    # Start reader thread
    sample_queue = deque(maxlen=5000)
    running = [True]
    if args.stdin:
        def stdin_reader(q, r):
            for line in sys.stdin:
                if not r[0]:
                    break
                line = line.strip()
                if not line or line.startswith("timestamp") or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) < 4:
                    continue
                try:
                    q.append((float(parts[1]), float(parts[2]), float(parts[3])))
                except Exception:
                    continue
        reader = threading.Thread(target=stdin_reader, args=(sample_queue, running), daemon=True)
    else:
        reader = threading.Thread(target=serial_reader_thread, args=(ser, sample_queue, running), daemon=True)
    reader.start()

    # Calibrate gravity
    print("Calibrating...", flush=True)
    while len(sample_queue) < 20:
        time.sleep(0.05)
    cal = list(sample_queue)[:20]
    gravity = np.mean(cal, axis=0)
    sample_queue.clear()
    print(f"Done. Gravity: [{gravity[0]:.4f}, {gravity[1]:.4f}, {gravity[2]:.4f}]\n", flush=True)

    WINDOW = 100
    buf = deque(maxlen=WINDOW)
    sample_count = 0
    last_classify = time.time()
    last_blackboard = 0
    classify_interval = 1.0 / rate

    if args.mode == "probability":
        print(f"{'Time':>8}  {'ax':>7}  {'ay':>7}  {'az':>7}  {'|a|':>7}  {'amp':>7}  {'Prob':>6}  Result", flush=True)
        print("=" * 72, flush=True)
    else:
        print("Listening... (prints only when earthquake detected)", flush=True)
        print("=" * 72, flush=True)

    try:
        while True:
            # Drain all available samples from reader thread
            drained = 0
            while sample_queue:
                ax, ay, az = sample_queue.popleft()
                sample_count += 1
                drained += 1
                dx = (ax - gravity[0]) * gain
                dy = (ay - gravity[1]) * gain
                dz = (az - gravity[2]) * gain
                buf.append((dx, dy, dz))

            if drained == 0:
                time.sleep(0.01)  # nothing to read, wait briefly
                continue

            # Classify at the requested rate
            now = time.time()
            if len(buf) >= WINDOW and (now - last_classify) >= classify_interval:
                last_classify = now

                window = np.array(list(buf), dtype=np.float32)
                prob = classify_window(model, scaler, window)
                ts = sample_count / 100.0
                mag = (ax**2 + ay**2 + az**2) ** 0.5
                amp = ((ax - gravity[0])**2 + (ay - gravity[1])**2 + (az - gravity[2])**2) ** 0.5
                is_eq = prob > threshold
                label = "EARTHQUAKE" if is_eq else "quiet"

                if args.mode == "probability":
                    marker = " <<<" if is_eq else ""
                    print(f"{ts:8.1f}  {ax:7.4f}  {ay:7.4f}  {az:7.4f}  {mag:7.4f}  {amp:7.4f}  {prob:6.1%}  {label}{marker}", flush=True)
                else:
                    if is_eq:
                        print(f"[{ts:7.1f}s] EARTHQUAKE  prob={prob:.1%}  amp={amp:.4f}g", flush=True)

                # Post to blackboard 1/sec
                if use_blackboard and (now - last_blackboard) >= 1.0:
                    last_blackboard = now
                    threading.Thread(target=post_to_blackboard, args=({
                        "timestamp": time.time(),
                        "ax": round(ax, 4), "ay": round(ay, 4), "az": round(az, 4),
                        "magnitude_g": round(mag, 4), "amplitude_g": round(amp, 4),
                        "probability": round(prob, 3), "label": label,
                        "profile": args.profile,
                    },), daemon=True).start()

    except KeyboardInterrupt:
        pass
    finally:
        running[0] = False
        ser.close()
        print(f"\nProcessed {sample_count} samples.", flush=True)


if __name__ == "__main__":
    main()
