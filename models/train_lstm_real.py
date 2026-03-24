"""
train_lstm_real.py
Train the LSTM earthquake classifier on two real datasets + synthetic knocks.

Datasets:
  1. LEN-DB (629K EQ + 616K noise, 20 Hz, broadband seismographs)
     → resample to 100 Hz, rescale to MPU6500 amplitude
  2. Central Italy MEMS (328 recordings, 250 Hz, ADXL355)
     → extract pre-P noise + post-P earthquake, rescale to MPU6500

Synthetic:
  3. Table knocks (generator.py) — negative examples to reduce false positives

Output:
  models/lstm_earthquake_model.h5
  models/lstm_scaler.pkl

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
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "earthquake_simulator"))
from generator import IndependentEarthquakeGenerator

# ── Parameters ────────────────────────────────────────────────────────────────
WINDOW_SIZE      = 100    # samples per window (must match realtime_predict.py)
STRIDE           = 50     # sliding window step
MPU6500_NOISE_G  = 0.004  # g — target noise std after rescaling
N_KNOCKS         = 600    # synthetic table knock recordings
EPOCHS           = 30
BATCH_SIZE       = 32

# LEN-DB sampling
LENDB_EQ_SAMPLE  = 5000   # how many earthquakes to sample from LEN-DB
LENDB_AN_SAMPLE  = 5000   # how many noise recordings to sample
LENDB_HZ         = 20     # LEN-DB native sample rate
TARGET_HZ        = 100    # our inference sample rate

# Central Italy
ITALY_HZ         = 250
SENSOR_SENS      = 3.9e-6  # g/count
P_BUFFER_SAMPLES = 500
MIN_NOISE_SAMPLES = WINDOW_SIZE * 4
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
LENDB_PATH = os.path.join(BASE_DIR, "datasets_for_trainings", "Dataset_LEN-DB", "LEN-DB.hdf5")
ITALY_HDF5 = os.path.join(BASE_DIR, "datasets_for_trainings", "Dataset_Central_Italy_2", "waveforms.hdf5")
ITALY_META = os.path.join(BASE_DIR, "datasets_for_trainings", "Dataset_Central_Italy_2", "metadata.csv")
OUTPUT_DIR = BASE_DIR


def make_windows(data_3xN, label):
    """Slice (3, N) waveform into overlapping (3, WINDOW_SIZE) windows."""
    windows = []
    n = data_3xN.shape[1]
    for start in range(0, n - WINDOW_SIZE, STRIDE):
        windows.append(data_3xN[:, start:start + WINDOW_SIZE])
    if not windows:
        return np.empty((0, 3, WINDOW_SIZE)), np.empty(0, dtype=int)
    X = np.array(windows)
    y = np.full(len(X), label, dtype=int)
    return X, y


def rescale_to_mpu6500(data_3xN):
    """Rescale recording so its noise std ≈ MPU6500_NOISE_G."""
    std = data_3xN.std() + 1e-15
    return data_3xN * (MPU6500_NOISE_G / std)


def load_lendb():
    """Load sampled data from LEN-DB, resample 20→100 Hz, rescale."""
    print(f"\n--- LEN-DB ---")
    print(f"Loading {LENDB_EQ_SAMPLE} EQ + {LENDB_AN_SAMPLE} noise from LEN-DB...")

    f = h5py.File(LENDB_PATH, "r")

    X_eq_list, X_noise_list = [], []

    # Sample random earthquake keys
    eq_keys = list(f["EQ"].keys())
    np.random.seed(42)
    eq_idx = np.random.choice(len(eq_keys), size=min(LENDB_EQ_SAMPLE, len(eq_keys)), replace=False)

    for count, i in enumerate(eq_idx):
        if count % 500 == 0:
            print(f"  EQ: {count}/{len(eq_idx)}")
        data = f["EQ"][eq_keys[i]][:]  # (3, 540)

        # Resample 20 Hz → 100 Hz: 540 samples → 2700 samples
        n_target = int(data.shape[1] * TARGET_HZ / LENDB_HZ)
        data_100hz = np.array([resample(data[ch], n_target) for ch in range(3)])  # (3, 2700)

        # Rescale to MPU6500 amplitude
        data_scaled = rescale_to_mpu6500(data_100hz)

        X_w, y_w = make_windows(data_scaled, label=1)
        if len(X_w):
            X_eq_list.append(X_w)

    # Sample random noise keys
    an_keys = list(f["AN"].keys())
    an_idx = np.random.choice(len(an_keys), size=min(LENDB_AN_SAMPLE, len(an_keys)), replace=False)

    for count, i in enumerate(an_idx):
        if count % 500 == 0:
            print(f"  Noise: {count}/{len(an_idx)}")
        data = f["AN"][an_keys[i]][:]  # (3, 540)

        n_target = int(data.shape[1] * TARGET_HZ / LENDB_HZ)
        data_100hz = np.array([resample(data[ch], n_target) for ch in range(3)])

        data_scaled = rescale_to_mpu6500(data_100hz)

        X_w, y_w = make_windows(data_scaled, label=0)
        if len(X_w):
            X_noise_list.append(X_w)

    f.close()

    X_eq = np.vstack(X_eq_list) if X_eq_list else np.empty((0, 3, WINDOW_SIZE))
    X_noise = np.vstack(X_noise_list) if X_noise_list else np.empty((0, 3, WINDOW_SIZE))

    print(f"  LEN-DB earthquake windows: {len(X_eq)}")
    print(f"  LEN-DB noise windows:      {len(X_noise)}")
    return X_eq, X_noise


def load_central_italy():
    """Load Central Italy MEMS dataset (128 recordings with P-picks)."""
    print(f"\n--- Central Italy MEMS ---")

    if not os.path.exists(ITALY_HDF5):
        print("  Not found, skipping.")
        return np.empty((0, 3, WINDOW_SIZE)), np.empty((0, 3, WINDOW_SIZE))

    with h5py.File(ITALY_HDF5, "r") as f:
        raw = f["data/bucket0"][:]

    meta = pd.read_csv(ITALY_META)
    X_eq_list, X_noise_list = [], []
    skipped = 0

    for i, row in meta.iterrows():
        if pd.isna(row["trace_p_pick_time"]):
            skipped += 1
            continue

        data_g = raw[i] * SENSOR_SENS

        def to_naive(ts_str):
            ts = pd.Timestamp(ts_str)
            return ts.tz_convert(None) if ts.tzinfo else ts

        t_start = to_naive(row["trace_start_time"])
        t_p = to_naive(row["trace_p_pick_time"])
        p_sample = int((t_p - t_start).total_seconds() * ITALY_HZ)
        p_sample = max(0, min(p_sample, data_g.shape[1] - 1))

        noise_end = max(0, p_sample - P_BUFFER_SAMPLES)
        if noise_end >= MIN_NOISE_SAMPLES:
            noise_std = data_g[:, :noise_end].std() + 1e-12
            scale = MPU6500_NOISE_G / noise_std
        else:
            scale = MPU6500_NOISE_G / (data_g.std() + 1e-12)
            noise_end = 0

        data_scaled = data_g * scale

        if noise_end >= MIN_NOISE_SAMPLES:
            X_w, _ = make_windows(data_scaled[:, :noise_end], label=0)
            if len(X_w):
                X_noise_list.append(X_w)

        eq_start = p_sample
        if data_scaled.shape[1] - eq_start >= WINDOW_SIZE:
            X_w, _ = make_windows(data_scaled[:, eq_start:], label=1)
            if len(X_w):
                X_eq_list.append(X_w)

    print(f"  Used {len(meta) - skipped} recordings ({skipped} skipped)")

    X_eq = np.vstack(X_eq_list) if X_eq_list else np.empty((0, 3, WINDOW_SIZE))
    X_noise = np.vstack(X_noise_list) if X_noise_list else np.empty((0, 3, WINDOW_SIZE))

    print(f"  Italy earthquake windows: {len(X_eq)}")
    print(f"  Italy noise windows:      {len(X_noise)}")
    return X_eq, X_noise


def generate_knock_windows(n_recordings, sample_rate=100):
    """Generate table knocks as negative examples."""
    gen = IndependentEarthquakeGenerator(sampling_rate=sample_rate, duration_sec=60)
    X_list = []
    for i in range(n_recordings):
        data = gen.generate_knock_data(
            n_knocks=np.random.randint(1, 8),
            knock_amplitude=np.random.uniform(0.03, 1.0),
            noise_level=np.random.uniform(0.002, 0.006),
        )
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
    print("LSTM Training — LEN-DB + Central Italy + Knocks")
    print("=" * 60)

    # 1. Load LEN-DB
    lendb_eq, lendb_noise = load_lendb()

    # 2. Load Central Italy
    italy_eq, italy_noise = load_central_italy()

    # 3. Generate knocks
    print(f"\n--- Synthetic knocks ---")
    print(f"Generating {N_KNOCKS} table knock recordings...")
    knock_windows = generate_knock_windows(N_KNOCKS)
    print(f"  Knock windows: {len(knock_windows)}")

    # 4. Combine all
    all_eq = [x for x in [lendb_eq, italy_eq] if len(x)]
    all_noise = [x for x in [lendb_noise, italy_noise, knock_windows] if len(x)]

    X_eq = np.vstack(all_eq) if all_eq else np.empty((0, 3, WINDOW_SIZE))
    X_noise = np.vstack(all_noise) if all_noise else np.empty((0, 3, WINDOW_SIZE))

    # Balance classes (downsample majority)
    n_eq, n_noise = len(X_eq), len(X_noise)
    n_target = min(n_eq, n_noise)
    if n_eq > n_target:
        idx = np.random.choice(n_eq, n_target, replace=False)
        X_eq = X_eq[idx]
    if n_noise > n_target:
        idx = np.random.choice(n_noise, n_target, replace=False)
        X_noise = X_noise[idx]

    X = np.vstack([X_eq, X_noise])
    y = np.hstack([np.ones(len(X_eq), dtype=int), np.zeros(len(X_noise), dtype=int)])

    shuffle_idx = np.random.permutation(len(X))
    X, y = X[shuffle_idx], y[shuffle_idx]

    print(f"\n--- Final dataset ---")
    print(f"  Earthquake (1): {(y == 1).sum()}")
    print(f"  Noise      (0): {(y == 0).sum()}")
    print(f"  Total         : {len(X)}")

    # 5. Normalize
    print("\nFitting StandardScaler...")
    X_flat = X.reshape(-1, WINDOW_SIZE)
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat)
    X = X_flat.reshape(-1, 3, WINDOW_SIZE)

    print(f"  Scaler mean range: {scaler.mean_.min():.6f} – {scaler.mean_.max():.6f}")
    print(f"  Scaler std  range: {scaler.scale_.min():.6f} – {scaler.scale_.max():.6f}")

    # 6. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

    # 7. Train
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

    # 8. Evaluate
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

    # 9. Save
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
