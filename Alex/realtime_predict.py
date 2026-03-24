"""
realtime_predict.py
Real-time earthquake prediction using the LSTM model.

Compatible with the MPU6500 demo sensor output (100 Hz, ax/ay/az in g units).
Serial format expected from Arduino:  timestamp_ms,ax,ay,az,gx,gy,gz

Usage:
    # Live sensor (Arduino connected via USB):
    python Alex/realtime_predict.py --port /dev/ttyACM0

    # Offline test on saved CSV from demo:
    python Alex/realtime_predict.py --csv demo/accelerometer_data.csv
"""

import argparse
import os
import sys
import time
import pickle
from collections import deque

import numpy as np

# ─── Constants (must match demo sensor) ──────────────────────────────────────
WINDOW_SIZE  = 100   # samples per inference window (1.0 s at 100 Hz)
STRIDE       = 50    # run inference every 50 new samples (0.5 s step)
SAMPLE_RATE  = 100   # Hz — matches Arduino sketch.ino delay(10)
THRESHOLD    = 0.5   # probability threshold for earthquake alert
# ─────────────────────────────────────────────────────────────────────────────


def _model_dir():
    """Return the directory where this script lives (Alex/)."""
    return os.path.dirname(os.path.abspath(__file__))


def load_model_and_scaler():
    """Load trained LSTM model and StandardScaler from Alex/."""
    from tensorflow import keras

    base = _model_dir()
    model  = keras.models.load_model(os.path.join(base, "lstm_earthquake_model.h5"))
    with open(os.path.join(base, "lstm_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    print("Model and scaler loaded.")
    return model, scaler


def predict_window(window_xyz, model, scaler):
    """
    Run LSTM inference on one window.

    Args:
        window_xyz: np.ndarray shape (100, 3) — columns ax, ay, az (in g)
    Returns:
        (prob, label) where prob is float 0-1 and label is "EARTHQUAKE" or "noise"
    """
    X = window_xyz.T                            # (3, 100) — axes first, time second
    X_flat = X.reshape(-1, WINDOW_SIZE)         # (3, 100) flat for scaler
    X_norm = scaler.transform(X_flat)           # normalize using training statistics
    X_input = X_norm.reshape(1, 3, WINDOW_SIZE) # (1, 3, 100) — batch of 1
    prob = float(model.predict(X_input, verbose=0)[0][0])
    label = "EARTHQUAKE" if prob > THRESHOLD else "noise"
    return prob, label


# ─── Serial streaming ─────────────────────────────────────────────────────────

def stream_serial(port, model, scaler):
    """
    Read live data from Arduino/MPU6500 over serial and run inference.

    Expected serial line format (from demo/arduino-sensor/sketch.ino):
        timestamp_ms,ax,ay,az,gx,gy,gz
        e.g.: 1234,0.006,-0.006,0.991,0.030,-0.012,0.001
    """
    import serial

    print(f"Connecting to {port} at 115200 baud...")
    ser = serial.Serial(port, 115200, timeout=2)
    time.sleep(1.5)
    ser.reset_input_buffer()
    for _ in range(5):          # discard header + stale lines
        ser.readline()

    buf = deque(maxlen=WINDOW_SIZE)
    samples_since_last_inference = 0

    print(f"Streaming | {WINDOW_SIZE}-sample window | stride {STRIDE} | Ctrl+C to stop\n")
    print(f"{'Time':>8}  {'ax':>8}  {'ay':>8}  {'az':>8}  {'Prob':>7}  Status")
    print("-" * 58)

    try:
        while True:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw or raw.startswith("timestamp") or raw.startswith("#"):
                continue

            parts = raw.split(",")
            if len(parts) not in (4, 7):
                continue

            try:
                ts_s = float(parts[0]) / 1000.0      # ms → s
                ax, ay, az = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                continue

            buf.append((ax, ay, az))
            samples_since_last_inference += 1

            # Run inference once buffer is full and enough new samples arrived
            if len(buf) == WINDOW_SIZE and samples_since_last_inference >= STRIDE:
                samples_since_last_inference = 0
                window = np.array(buf)              # (100, 3)
                prob, label = predict_window(window, model, scaler)
                marker = "  <-- !!!" if label == "EARTHQUAKE" else ""
                print(f"{ts_s:8.2f}  {ax:8.4f}  {ay:8.4f}  {az:8.4f}  {prob:6.1%}  {label}{marker}")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()


# ─── Offline CSV testing ──────────────────────────────────────────────────────

def stream_csv(csv_path, model, scaler):
    """
    Run inference on a CSV file saved by the demo system.

    Supports both column naming conventions used in demo/:
        accelerometer_data.csv  →  timestamp, x, y, z
        stream.py --save        →  timestamp_s, ax, ay, az, gx, gy, gz
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()

    # Resolve axis column names (x/y/z or ax/ay/az)
    axes = [c for c in ("x", "y", "z") if c in df.columns]
    if not axes:
        axes = [c for c in ("ax", "ay", "az") if c in df.columns]
    if len(axes) != 3:
        raise ValueError(
            f"CSV must contain columns x,y,z or ax,ay,az. Found: {list(df.columns)}"
        )

    data = df[axes].values.astype(float)   # (N, 3)
    n_windows = (len(data) - WINDOW_SIZE) // STRIDE

    print(f"CSV: {csv_path}  |  {len(data)} samples  |  {n_windows} inference windows\n")
    print(f"{'Window':>8}  {'Sample':>8}  {'Prob':>7}  Status")
    print("-" * 38)

    earthquake_count = 0
    for w_idx, start in enumerate(range(0, len(data) - WINDOW_SIZE, STRIDE)):
        window = data[start : start + WINDOW_SIZE]   # (100, 3)
        prob, label = predict_window(window, model, scaler)
        if label == "EARTHQUAKE":
            earthquake_count += 1
        marker = "  <-- !!!" if label == "EARTHQUAKE" else ""
        print(f"{w_idx:8d}  {start:8d}  {prob:6.1%}  {label}{marker}")

    print(f"\nDone. Earthquake windows: {earthquake_count}/{n_windows}")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Real-time LSTM earthquake detection, compatible with demo/ sensor format."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--port", metavar="PORT",
                     help="Serial port for live sensor, e.g. /dev/ttyACM0")
    src.add_argument("--csv",  metavar="FILE",
                     help="Saved CSV from demo sensor (offline test)")
    args = parser.parse_args()

    model, scaler = load_model_and_scaler()

    if args.port:
        stream_serial(args.port, model, scaler)
    else:
        stream_csv(args.csv, model, scaler)


if __name__ == "__main__":
    main()
