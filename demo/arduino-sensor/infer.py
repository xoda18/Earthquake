"""
infer.py — Reads Arduino serial, classifies with LSTM model.

Converts sensor g-force data to raw count scale matching training data,
then runs through the LSTM for earthquake/quiet classification.

Usage:
    docker run --rm --device /dev/ttyACM0 earthquake-detector
    docker run --rm --device /dev/ttyACM0 earthquake-detector --rate 5 --threshold 0.8
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


def classify_window(model, scaler, window_3x100):
    X = window_3x100.reshape(1, 3, 100)
    X_flat = X.reshape(-1, 100)
    X_flat = scaler.transform(X_flat)
    X = X_flat.reshape(1, 3, 100)
    return float(model.predict(X, verbose=0)[0][0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None)
    parser.add_argument("--rate", type=float, default=5.0, help="Classifications per second")
    parser.add_argument("--scale", type=float, default=1000.0, help="G-to-raw scale (lower=less sensitive)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Probability threshold (0.0-1.0)")
    parser.add_argument("--mode", default="probability", choices=["probability", "magnitude"],
                        help="probability: always show prob. magnitude: only print when earthquake detected")
    args = parser.parse_args()

    g_to_raw = args.scale

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
    print(f"  Rate:      {args.rate}/s", flush=True)
    print(f"  Scale:     {args.scale}", flush=True)
    print(f"  Threshold: {args.threshold}", flush=True)
    print(f"  Mode:      {args.mode}", flush=True)
    print(flush=True)

    # Phase 1: Calibrate gravity from first 50 samples
    print("Calibrating gravity (hold sensor still)...", flush=True)
    cal_ax, cal_ay, cal_az = [], [], []
    cal_count = 0
    while cal_count < 50:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line or line.startswith("timestamp") or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 4:
            continue
        try:
            cal_ax.append(float(parts[1]))
            cal_ay.append(float(parts[2]))
            cal_az.append(float(parts[3]))
            cal_count += 1
        except ValueError:
            continue

    gravity = np.array([np.mean(cal_ax), np.mean(cal_ay), np.mean(cal_az)])
    print(f"Calibration done. Gravity: [{gravity[0]:.4f}, {gravity[1]:.4f}, {gravity[2]:.4f}]", flush=True)
    print(flush=True)

    # Phase 2: Stream and classify
    WINDOW = 100
    buf_ax, buf_ay, buf_az = [], [], []
    sample_count = 0
    last_classify = time.time()
    classify_interval = 1.0 / args.rate

    if args.mode == "probability":
        print(f"{'Time':>8}  {'|a|':>7}  {'amp':>7}  {'Prob':>6}  Result", flush=True)
        print("=" * 52, flush=True)
    else:
        print("Listening... (only prints when earthquake detected)", flush=True)
        print("=" * 52, flush=True)

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

            # Remove gravity, scale to raw counts, clip
            raw_ax = np.clip((ax - gravity[0]) * g_to_raw, -596, 596)
            raw_ay = np.clip((ay - gravity[1]) * g_to_raw, -596, 596)
            raw_az = np.clip((az - gravity[2]) * g_to_raw, -596, 596)

            buf_ax.append(raw_ax)
            buf_ay.append(raw_ay)
            buf_az.append(raw_az)

            if len(buf_ax) > WINDOW:
                buf_ax = buf_ax[-WINDOW:]
                buf_ay = buf_ay[-WINDOW:]
                buf_az = buf_az[-WINDOW:]

            now = time.time()
            if len(buf_ax) >= WINDOW and (now - last_classify) >= classify_interval:
                last_classify = now
                window = np.array([buf_ax[-WINDOW:], buf_ay[-WINDOW:], buf_az[-WINDOW:]], dtype=np.float32)
                prob = classify_window(model, scaler, window)
                ts = sample_count / 100.0

                # Compute amplitude (deviation from gravity)
                mag = (ax**2 + ay**2 + az**2) ** 0.5
                amp = ((ax - gravity[0])**2 + (ay - gravity[1])**2 + (az - gravity[2])**2) ** 0.5

                is_earthquake = prob > args.threshold

                if args.mode == "probability":
                    label = "EARTHQUAKE" if is_earthquake else "quiet"
                    marker = " <<<" if is_earthquake else ""
                    print(f"{ts:8.1f}  {mag:7.4f}  {amp:7.4f}  {prob:6.1%}  {label}{marker}", flush=True)
                else:
                    # magnitude mode: only print when earthquake
                    if is_earthquake:
                        print(f"[{ts:7.1f}s] EARTHQUAKE  prob={prob:.1%}  amp={amp:.4f}g", flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        print(f"\nProcessed {sample_count} samples.", flush=True)


if __name__ == "__main__":
    main()
