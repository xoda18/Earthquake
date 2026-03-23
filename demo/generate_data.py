"""
generate_data.py
Generates synthetic accelerometer CSV data with an embedded earthquake event.
Output: accelerometer_data.csv  (columns: timestamp, x, y, z)
"""

import numpy as np
import pandas as pd

SAMPLE_RATE = 100          # Hz
DURATION_S  = 120          # total recording length in seconds
EQ_START_S  = 45.0         # earthquake onset
EQ_DURATION_S = 30.0       # earthquake duration
NOISE_STD   = 0.02         # background noise (g)
EQ_AMP      = 0.6          # peak earthquake amplitude (g)
SEED        = 42

rng = np.random.default_rng(SEED)
n_samples = DURATION_S * SAMPLE_RATE
t = np.linspace(0, DURATION_S, n_samples, endpoint=False)

# Background noise on all three axes
x = rng.normal(0, NOISE_STD, n_samples)
y = rng.normal(0, NOISE_STD, n_samples)
z = rng.normal(0, NOISE_STD, n_samples) + 1.0   # +1 g for gravity on Z

# Earthquake signal: bandpass-like burst (2–15 Hz) with smooth envelope
eq_mask = (t >= EQ_START_S) & (t < EQ_START_S + EQ_DURATION_S)
t_eq = t[eq_mask] - EQ_START_S

# Hann envelope so onset/offset are smooth
envelope = np.hanning(eq_mask.sum())
quake_x = envelope * EQ_AMP * (
    0.6 * np.sin(2 * np.pi * 4.0  * t_eq) +
    0.3 * np.sin(2 * np.pi * 9.0  * t_eq) +
    0.1 * np.sin(2 * np.pi * 14.0 * t_eq)
) * rng.normal(1, 0.15, eq_mask.sum())

quake_y = envelope * EQ_AMP * 0.7 * (
    np.sin(2 * np.pi * 5.5 * t_eq) +
    0.4 * np.sin(2 * np.pi * 11.0 * t_eq)
) * rng.normal(1, 0.15, eq_mask.sum())

x[eq_mask] += quake_x
y[eq_mask] += quake_y

# Build timestamps starting at a fixed wall-clock time
start_time = pd.Timestamp("2026-03-23 14:32:00.000")
timestamps = pd.date_range(start=start_time, periods=n_samples, freq=f"{1_000_000_000 // SAMPLE_RATE}ns")

df = pd.DataFrame({"timestamp": timestamps, "x": x, "y": y, "z": z})
df.to_csv("accelerometer_data.csv", index=False)
print(f"Saved {len(df):,} samples → accelerometer_data.csv")
print(f"Earthquake embedded: {start_time + pd.Timedelta(seconds=EQ_START_S)}  "
      f"→  {start_time + pd.Timedelta(seconds=EQ_START_S + EQ_DURATION_S)}")
