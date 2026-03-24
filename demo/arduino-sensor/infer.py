"""
infer.py — Reads Arduino serial, classifies with LSTM model.

Usage:
    docker run --rm --device /dev/ttyACM0 earthquake-detector
    docker run --rm --device /dev/ttyACM0 earthquake-detector --rate 5 --threshold 0.7
    docker run --rm --device /dev/ttyACM0 earthquake-detector --mode magnitude
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
    X = window_100x3.T.reshape(1, 3, 100)   # (1, 3, 100)
    X_flat = X.reshape(-1, 100)              # (3, 100)
    X_flat = scaler.transform(X_flat)
    X = X_flat.reshape(1, 3, 100)
    return float(model.predict(X, verbose=0)[0][0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None)
    parser.add_argument("--rate", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--mode", default="probability", choices=["probability", "magnitude"])
    args = parser.parse_args()

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
    print(f"Connected.", flush=True)
    print(f"  Rate: {args.rate}/s | Threshold: {args.threshold} | Mode: {args.mode}\n", flush=True)

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
    buf = []  # list of (ax, ay, az) tuples — raw g-force with gravity removed
    sample_count = 0
    last_classify = time.time()
    classify_interval = 1.0 / args.rate

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
            buf.append((ax - gravity[0], ay - gravity[1], az - gravity[2]))

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
                is_eq = prob > args.threshold

                if args.mode == "probability":
                    label = "EARTHQUAKE" if is_eq else "quiet"
                    marker = " <<<" if is_eq else ""
                    print(f"{ts:8.1f}  {ax:7.4f}  {ay:7.4f}  {az:7.4f}  {mag:7.4f}  {amp:7.4f}  {prob:6.1%}  {label}{marker}", flush=True)
                else:
                    if is_eq:
                        print(f"[{ts:7.1f}s] EARTHQUAKE  prob={prob:.1%}  amp={amp:.4f}g  ax={ax:.4f} ay={ay:.4f} az={az:.4f}", flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        print(f"\nProcessed {sample_count} samples.", flush=True)


if __name__ == "__main__":
    main()
