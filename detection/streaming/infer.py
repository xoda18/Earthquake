"""
infer.py — Reads Arduino serial, classifies with LSTM model.

Profiles control sensitivity for different use-cases.

Usage:
    docker build -f detection/streaming/Dockerfile -t earthquake-detector .
    docker run --rm --device /dev/ttyACM0 earthquake-detector
    docker run --rm --device /dev/ttyACM0 earthquake-detector --profile table
    docker run --rm --device /dev/ttyACM0 earthquake-detector --profile sensitive
    docker run --rm --device /dev/ttyACM0 earthquake-detector --threshold 0.8 --gain 5
"""

import argparse
import sys
import time
import glob
import pickle
import numpy as np
import serial
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow import keras

# ── Profiles ──────────────────────────────────────────────────────────────────
# gain: multiplier on gravity-removed signal before feeding to model.
#   Higher gain = sensor noise looks bigger = more sensitive = more false positives.
#   Lower gain  = only strong shakes trigger.
# threshold: LSTM probability above which → EARTHQUAKE.
# rate: classifications per second.

PROFILES = {
    "default": {
        "gain": 1.0,
        "threshold": 0.7,
        "rate": 2.0,
        "description": "Balanced — works for moderate shakes",
    },
    "table": {
        "gain": 3.0,
        "threshold": 0.8,
        "rate": 2.0,
        "description": "Table demo — amplifies signal so table knocks are detected",
    },
    "sensitive": {
        "gain": 5.0,
        "threshold": 0.6,
        "rate": 2.0,
        "description": "High sensitivity — detects light taps and vibrations",
    },
    "earthquake": {
        "gain": 1.0,
        "threshold": 0.9,
        "rate": 1.0,
        "description": "Real seismic — only strong, sustained shaking triggers",
    },
}


def auto_detect_port():
    candidates = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    if candidates:
        return candidates[0]
    raise RuntimeError("No serial port. Pass --device /dev/ttyACM0 to docker run")


def load_model():
    model = keras.models.load_model('/app/model.h5', compile=False)
    with open('/app/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


def classify_window(model, scaler, window_100x3):
    """Input: (100, 3) array. Returns probability 0.0-1.0."""
    X = window_100x3.T.reshape(1, 3, 100)
    X_flat = X.reshape(-1, 100)
    X_flat = scaler.transform(X_flat)
    X = X_flat.reshape(1, 3, 100)
    return float(model.predict(X, verbose=0)[0][0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None)
    parser.add_argument("--profile", default="default", choices=list(PROFILES.keys()),
                        help="Detection profile (default, table, sensitive, earthquake)")
    parser.add_argument("--rate", type=float, default=None, help="Override: classifications/sec")
    parser.add_argument("--threshold", type=float, default=None, help="Override: probability threshold")
    parser.add_argument("--gain", type=float, default=None, help="Override: signal amplification")
    parser.add_argument("--mode", default="probability", choices=["probability", "magnitude"])
    args = parser.parse_args()

    # Apply profile, then overrides
    p = PROFILES[args.profile]
    gain = args.gain if args.gain is not None else p["gain"]
    threshold = args.threshold if args.threshold is not None else p["threshold"]
    rate = args.rate if args.rate is not None else p["rate"]

    print("Loading LSTM model...", flush=True)
    model, scaler = load_model()
    print("Model loaded.", flush=True)

    port = args.port or auto_detect_port()
    print(f"Connecting to {port}...", flush=True)
    ser = serial.Serial(port, 115200, timeout=2)
    time.sleep(2)
    ser.reset_input_buffer()
    for _ in range(5):
        ser.readline()

    print(f"Connected.\n", flush=True)
    print(f"  Profile:   {args.profile} — {p['description']}", flush=True)
    print(f"  Gain:      {gain}x", flush=True)
    print(f"  Threshold: {threshold}", flush=True)
    print(f"  Rate:      {rate}/s", flush=True)
    print(f"  Mode:      {args.mode}", flush=True)
    print(flush=True)

    # Calibrate gravity
    print("Calibrating (hold sensor still)...", flush=True)
    cal = []
    while len(cal) < 50:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line or line.startswith("timestamp") or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 4:
            continue
        try:
            cal.append([float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            continue
    gravity = np.mean(cal, axis=0)
    print(f"Done. Gravity: [{gravity[0]:.4f}, {gravity[1]:.4f}, {gravity[2]:.4f}]\n", flush=True)

    WINDOW = 100
    buf = []
    sample_count = 0
    last_classify = time.time()
    classify_interval = 1.0 / rate

    if args.mode == "probability":
        print(f"{'Time':>8}  {'ax':>7}  {'ay':>7}  {'az':>7}  {'|a|':>7}  {'amp':>7}  {'Prob':>6}  Result", flush=True)
        print("=" * 72, flush=True)
    else:
        print("Listening... (prints only when earthquake detected)", flush=True)
        print("=" * 72, flush=True)

    try:
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line or line.startswith("timestamp") or line.startswith("#"):
                continue

            parts = line.split(",")
            if len(parts) < 4:
                continue

            try:
                ax, ay, az = float(parts[1]), float(parts[2]), float(parts[3])
            except (ValueError, IndexError):
                continue

            sample_count += 1

            # Remove gravity and apply gain
            dx = (ax - gravity[0]) * gain
            dy = (ay - gravity[1]) * gain
            dz = (az - gravity[2]) * gain
            buf.append((dx, dy, dz))

            if len(buf) > WINDOW:
                buf = buf[-WINDOW:]

            now = time.time()
            if len(buf) >= WINDOW and (now - last_classify) >= classify_interval:
                last_classify = now

                window = np.array(buf[-WINDOW:], dtype=np.float32)  # (100, 3)
                prob = classify_window(model, scaler, window)
                ts = sample_count / 100.0
                mag = (ax**2 + ay**2 + az**2) ** 0.5
                amp = ((ax - gravity[0])**2 + (ay - gravity[1])**2 + (az - gravity[2])**2) ** 0.5
                is_eq = prob > threshold

                if args.mode == "probability":
                    label = "EARTHQUAKE" if is_eq else "quiet"
                    marker = " <<<" if is_eq else ""
                    print(f"{ts:8.1f}  {ax:7.4f}  {ay:7.4f}  {az:7.4f}  {mag:7.4f}  {amp:7.4f}  {prob:6.1%}  {label}{marker}", flush=True)
                else:
                    if is_eq:
                        print(f"[{ts:7.1f}s] EARTHQUAKE  prob={prob:.1%}  amp={amp:.4f}g", flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        print(f"\nProcessed {sample_count} samples.", flush=True)


if __name__ == "__main__":
    main()
