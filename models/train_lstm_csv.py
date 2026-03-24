"""
train_lstm_csv.py
Train the LSTM earthquake classifier on synthetic g-unit data.

Uses the generator (earthquake_simulator/generator.py) to produce
waveforms in g-units (±2.0 g), matching MPU6500 sensor output directly.

Usage:
    python models/train_lstm_csv.py
"""

import sys
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

# Add earthquake_simulator to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "earthquake_simulator"))
from generator import IndependentEarthquakeGenerator

# ── Parameters ────────────────────────────────────────────────────────────────
WINDOW_SIZE   = 100    # samples per window (1.0 s at 100 Hz)
STRIDE        = 50     # sliding window step
SAMPLE_RATE   = 100    # Hz — matches demo/arduino-sensor/sketch.ino
N_EARTHQUAKES = 600    # number of earthquake recordings to generate
N_NOISE       = 600    # number of noise recordings to generate
DURATION_SEC  = 60     # seconds per recording
EPOCHS        = 30
BATCH_SIZE    = 32
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def make_windows(data_3xN, label):
    """
    Slice (3, N) waveform into overlapping windows of shape (3, WINDOW_SIZE).

    Returns:
        X: np.ndarray (n_windows, 3, WINDOW_SIZE)
        y: np.ndarray (n_windows,)
    """
    windows = []
    n = data_3xN.shape[1]
    for start in range(0, n - WINDOW_SIZE, STRIDE):
        windows.append(data_3xN[:, start:start + WINDOW_SIZE])
    X = np.array(windows)          # (n_windows, 3, WINDOW_SIZE)
    y = np.full(len(X), label)
    return X, y


def generate_dataset():
    """Generate synthetic earthquake + noise dataset in g-units."""
    gen = IndependentEarthquakeGenerator(
        sampling_rate=SAMPLE_RATE,
        duration_sec=DURATION_SEC
    )

    X_list, y_list = [], []

    print(f"Generating {N_EARTHQUAKES} earthquake recordings...")
    magnitudes = np.random.uniform(2.5, 6.5, N_EARTHQUAKES)
    p_starts   = np.random.uniform(5, 20, N_EARTHQUAKES)
    durations  = np.random.uniform(8, 20, N_EARTHQUAKES)

    for i, (mag, p_start, dur) in enumerate(zip(magnitudes, p_starts, durations)):
        if i % 100 == 0:
            print(f"  {i}/{N_EARTHQUAKES}")
        data = gen.generate_earthquake_data(
            magnitude=mag,
            p_wave_start_sec=p_start,
            earthquake_duration_sec=dur,
        )                           # (N_samples, 3)
        X_w, y_w = make_windows(data.T, label=1)
        X_list.append(X_w)
        y_list.append(y_w)

    print(f"Generating {N_NOISE} noise recordings...")
    for i in range(N_NOISE):
        if i % 100 == 0:
            print(f"  {i}/{N_NOISE}")
        # Vary noise level slightly to improve robustness
        noise_level = np.random.uniform(0.002, 0.008)
        data = gen.generate_noise_data(noise_level=noise_level)
        X_w, y_w = make_windows(data.T, label=0)
        X_list.append(X_w)
        y_list.append(y_w)

    X = np.vstack(X_list)    # (total_windows, 3, 100)
    y = np.hstack(y_list)

    print(f"\nTotal windows: {len(X)}")
    print(f"  Earthquake (1): {(y == 1).sum()}")
    print(f"  Noise      (0): {(y == 0).sum()}")
    return X, y


def build_model():
    model = keras.Sequential([
        keras.layers.LSTM(64, activation='relu', input_shape=(3, WINDOW_SIZE), return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main():
    print("=" * 60)
    print("LSTM Training on Synthetic g-Unit Data")
    print("=" * 60)

    # 1. Generate data
    X, y = generate_dataset()

    # 2. Normalize (fit on g-unit data)
    print("\nFitting StandardScaler on g-unit data...")
    X_flat = X.reshape(-1, WINDOW_SIZE)          # (N*3, 100)
    scaler = StandardScaler()
    X_flat = scaler.fit_transform(X_flat)
    X = X_flat.reshape(-1, 3, WINDOW_SIZE)

    print(f"Scaler mean range: {scaler.mean_.min():.4f} – {scaler.mean_.max():.4f}")
    print(f"Scaler std  range: {scaler.scale_.min():.4f} – {scaler.scale_.max():.4f}")

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

    # 4. Train
    print("\nTraining LSTM...")
    model = build_model()
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1,
    )

    # 5. Evaluate
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

    # 6. Save (overwrite existing model + scaler)
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
