"""
train_lstm_real.py
Train the LSTM earthquake classifier on the real Mendeley Central Italy MEMS dataset.

Dataset: models/dataset/waveforms.hdf5
  - 328 recordings from ADXL355 sensors (250 Hz, ~100 s each)
  - Unit: counts (raw ADC), sensitivity = 3.9e-6 g/count
  - Each recording contains a real earthquake

Noise extraction:
  Pre-P-wave sections of each recording are used as noise samples
  (standard seismology technique — before P-wave arrival = quiet ground).

Amplitude rescaling:
  ADXL355 noise floor ≈ 0.0005g, MPU6500 noise floor ≈ 0.004g.
  Each recording is rescaled so its pre-P noise std matches MPU6500,
  preserving waveform shape while ensuring scaler compatibility at inference.

Supplemental noise:
  If real pre-P sections are too short, synthetic noise from generator.py fills the gap.

Output:
  models/lstm_earthquake_model.h5  (overwrites)
  models/lstm_scaler.pkl           (overwrites)

Usage:
    python models/train_lstm_real.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

# Add earthquake_simulator to path for supplemental noise generation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "earthquake_simulator"))
from generator import IndependentEarthquakeGenerator

# ── Parameters ────────────────────────────────────────────────────────────────
WINDOW_SIZE      = 100    # samples per window (must match realtime_predict.py)
STRIDE           = 50     # sliding window step
DATASET_HZ       = 250    # Hz — Mendeley MEMS dataset sample rate
SENSOR_SENS      = 3.9e-6 # g/count — ADXL355 sensitivity from metadata
MPU6500_NOISE_G  = 0.004  # g — target noise std after rescaling (MPU6500 floor)
P_BUFFER_SAMPLES = 500    # samples to drop before P-wave (2 s safety margin at 250 Hz)
MIN_NOISE_SAMPLES = WINDOW_SIZE * 4   # minimum pre-P samples needed to be useful
SUPPLEMENTAL_NOISE = 400  # synthetic noise recordings if real noise count is insufficient
EPOCHS           = 30
BATCH_SIZE       = 32
# ─────────────────────────────────────────────────────────────────────────────

DATASET_HDF5 = os.path.join(os.path.dirname(__file__), "dataset", "waveforms.hdf5")
DATASET_META = os.path.join(os.path.dirname(__file__), "dataset", "metadata.csv")
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))


def make_windows(data_3xN, label):
    """Slice (3, N) waveform into overlapping (3, WINDOW_SIZE) windows."""
    windows = []
    n = data_3xN.shape[1]
    for start in range(0, n - WINDOW_SIZE, STRIDE):
        windows.append(data_3xN[:, start:start + WINDOW_SIZE])
    if not windows:
        return np.empty((0, 3, WINDOW_SIZE)), np.empty(0, dtype=int)
    X = np.array(windows)          # (n_windows, 3, WINDOW_SIZE)
    y = np.full(len(X), label, dtype=int)
    return X, y


def load_real_data():
    """
    Load the Mendeley dataset, convert to g-units, rescale to MPU6500 amplitude,
    and extract noise (pre-P-wave) + earthquake (post-P-wave) windows.
    """
    print("Loading HDF5 dataset...")
    with h5py.File(DATASET_HDF5, "r") as f:
        raw = f["data/bucket0"][:]   # (328, 3, 25002), dtype float64

    print(f"  Loaded {raw.shape[0]} recordings, {raw.shape[2]} samples each at {DATASET_HZ} Hz")

    meta = pd.read_csv(DATASET_META)
    assert len(meta) == raw.shape[0], "Metadata row count doesn't match HDF5"

    X_eq, y_eq, X_noise, y_noise = [], [], [], []
    skipped = 0

    for i, row in meta.iterrows():
        # Skip recordings without a confirmed P-wave pick
        if pd.isna(row["trace_p_pick_time"]):
            skipped += 1
            continue

        # Convert counts → g
        data_g = raw[i] * SENSOR_SENS   # (3, 25002)

        # Compute P-wave sample index — strip timezone if present
        def to_naive(ts_str):
            ts = pd.Timestamp(ts_str)
            return ts.tz_convert(None) if ts.tzinfo else ts

        t_start  = to_naive(row["trace_start_time"])
        t_p      = to_naive(row["trace_p_pick_time"])
        p_sample = int((t_p - t_start).total_seconds() * DATASET_HZ)
        p_sample = max(0, min(p_sample, data_g.shape[1] - 1))

        # Rescale: normalize noise floor to MPU6500 level
        noise_end = max(0, p_sample - P_BUFFER_SAMPLES)
        if noise_end >= MIN_NOISE_SAMPLES:
            noise_seg  = data_g[:, :noise_end]
            noise_std  = noise_seg.std() + 1e-12
            scale      = MPU6500_NOISE_G / noise_std
        else:
            # Not enough pre-P: use whole-recording std as reference
            scale     = MPU6500_NOISE_G / (data_g.std() + 1e-12)
            noise_end = 0  # no noise windows from this recording

        data_scaled = data_g * scale   # (3, N) — waveform shapes preserved

        # Noise windows (pre-P-wave, minus safety buffer)
        if noise_end >= MIN_NOISE_SAMPLES:
            X_w, y_w = make_windows(data_scaled[:, :noise_end], label=0)
            if len(X_w):
                X_noise.append(X_w)
                y_noise.append(y_w)

        # Earthquake windows (post-P-wave)
        eq_start = p_sample
        if data_scaled.shape[1] - eq_start >= WINDOW_SIZE:
            X_w, y_w = make_windows(data_scaled[:, eq_start:], label=1)
            if len(X_w):
                X_eq.append(X_w)
                y_eq.append(y_w)

    print(f"  Used {len(meta) - skipped} recordings with confirmed P-picks ({skipped} skipped — no pick)")

    X_eq    = np.vstack(X_eq)    if X_eq    else np.empty((0, 3, WINDOW_SIZE))
    X_noise = np.vstack(X_noise) if X_noise else np.empty((0, 3, WINDOW_SIZE))
    y_eq    = np.ones(len(X_eq),    dtype=int)
    y_noise = np.zeros(len(X_noise), dtype=int)

    print(f"  Real earthquake windows : {len(X_eq)}")
    print(f"  Real noise windows      : {len(X_noise)}")
    return X_eq, y_eq, X_noise, y_noise


def generate_supplemental_noise(n_recordings, sample_rate=100):
    """Generate synthetic noise at MPU6500 noise level to supplement real noise."""
    gen = IndependentEarthquakeGenerator(
        sampling_rate=sample_rate,
        duration_sec=60,
    )
    X_list = []
    for i in range(n_recordings):
        noise_level = np.random.uniform(0.002, 0.006)   # g — MPU6500 range
        data = gen.generate_noise_data(noise_level=noise_level)   # (N, 3)
        X_w, _ = make_windows(data.T, label=0)
        if len(X_w):
            X_list.append(X_w)
    return np.vstack(X_list) if X_list else np.empty((0, 3, WINDOW_SIZE))


def build_model():
    model = keras.Sequential([
        keras.layers.LSTM(64, activation="relu",
                          input_shape=(3, WINDOW_SIZE), return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    print("=" * 60)
    print("LSTM Training on Real Mendeley MEMS Dataset")
    print("=" * 60)

    # 1. Load real data
    X_eq, y_eq, X_noise, y_noise = load_real_data()

    # 2. Supplement noise if needed
    n_eq    = len(X_eq)
    n_noise = len(X_noise)

    if n_noise < n_eq:
        deficit = n_eq - n_noise
        print(f"\nNoise deficit: {deficit} windows — generating synthetic noise...")
        # Estimate how many recordings we need
        n_recordings = max(SUPPLEMENTAL_NOISE, deficit // 100 + 50)
        X_syn = generate_supplemental_noise(n_recordings)
        # Take only what we need
        np.random.shuffle(X_syn)
        X_syn = X_syn[:deficit]
        y_syn = np.zeros(len(X_syn), dtype=int)
        X_noise = np.vstack([X_noise, X_syn]) if len(X_noise) else X_syn
        y_noise = np.hstack([y_noise, y_syn]) if len(y_noise) else y_syn
        print(f"  After supplement: {len(X_noise)} noise windows")

    # 3. Combine and shuffle
    X = np.vstack([X_eq, X_noise])
    y = np.hstack([y_eq, y_noise])

    shuffle_idx = np.random.permutation(len(X))
    X, y = X[shuffle_idx], y[shuffle_idx]

    print(f"\nFinal dataset:")
    print(f"  Earthquake (1): {(y == 1).sum()}")
    print(f"  Noise      (0): {(y == 0).sum()}")
    print(f"  Total         : {len(X)}")

    # 4. Normalize
    print("\nFitting StandardScaler...")
    X_flat = X.reshape(-1, WINDOW_SIZE)   # (N*3, 100)
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat)
    X = X_flat.reshape(-1, 3, WINDOW_SIZE)

    print(f"  Scaler mean range: {scaler.mean_.min():.6f} – {scaler.mean_.max():.6f}")
    print(f"  Scaler std  range: {scaler.scale_.min():.6f} – {scaler.scale_.max():.6f}")

    # 5. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

    # 6. Train
    print("\nTraining LSTM...")
    model = build_model()
    model.summary()

    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1,
    )

    # 7. Evaluate
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    acc  = accuracy_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Confusion matrix:\n  TN={cm[0,0]}  FP={cm[0,1]}\n  FN={cm[1,0]}  TP={cm[1,1]}")

    # 8. Save
    model_path  = os.path.join(OUTPUT_DIR, "lstm_earthquake_model.h5")
    scaler_path = os.path.join(OUTPUT_DIR, "lstm_scaler.pkl")

    model.save(model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nSaved model  → {model_path}")
    print(f"Saved scaler → {scaler_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
