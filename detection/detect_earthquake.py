"""
detect_earthquake.py
Full earthquake detection pipeline (Stages 1-7).

Usage:
    python detect_earthquake.py                        # uses accelerometer_data.csv
    python detect_earthquake.py my_data.csv            # custom file
"""

import sys
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
import os as _os
import matplotlib
if not _os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")      # headless default — overridden by MPLBACKEND env var
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Configuration Profiles ───────────────────────────────────────────────────
# Earthquake: long-duration seismic events (30+ seconds)
# Table Knock: short impulse events on table (0.5-2 seconds)
CONFIG_PROFILES = {
    "earthquake": {
        "BP_LOW_HZ": 1.0,
        "BP_HIGH_HZ": 20.0,
        "STA_WINDOW_S": 0.5,
        "LTA_WINDOW_S": 10.0,
        "STA_LTA_THRESH": 3.0,
        "AMP_SIGMA_THRESH": 4.0,
        "MERGE_GAP_S": 3.0,
        "QUIET_GUARD_S": 8.0,
    },
    "table_knock": {
        "BP_LOW_HZ": 2.0,
        "BP_HIGH_HZ": 25.0,
        "STA_WINDOW_S": 0.2,
        "LTA_WINDOW_S": 5.0,
        "STA_LTA_THRESH": 2.5,
        "AMP_SIGMA_THRESH": 4.0,
        "MERGE_GAP_S": 1.0,
        "QUIET_GUARD_S": 2.0,
    },
}

# Default configuration (can be overridden by load_config)
SAMPLE_RATE      = 100     # Hz  (auto-detected if timestamps are present)
BP_LOW_HZ        = 1.0     # bandpass low  cut (Hz)
BP_HIGH_HZ       = 20.0    # bandpass high cut (Hz)
STA_WINDOW_S     = 0.5     # short-term average window  (seconds)
LTA_WINDOW_S     = 10.0    # long-term  average window  (seconds)
STA_LTA_THRESH   = 3.0     # ratio threshold to declare a spike
AMP_SIGMA_THRESH = 4.0     # amplitude threshold (× std above baseline)
MERGE_GAP_S      = 3.0     # merge spikes closer than this (seconds)
QUIET_GUARD_S    = 8.0     # calm period required before closing the event
OUTPUT_IMAGE     = "earthquake_report.png"
# ─────────────────────────────────────────────────────────────────────────────


def load_config(mode: str = "earthquake") -> None:
    """
    Load configuration profile by name.

    Args:
        mode: "earthquake" or "table_knock"
    """
    global BP_LOW_HZ, BP_HIGH_HZ, STA_WINDOW_S, LTA_WINDOW_S
    global STA_LTA_THRESH, AMP_SIGMA_THRESH, MERGE_GAP_S, QUIET_GUARD_S

    if mode not in CONFIG_PROFILES:
        raise ValueError(f"Unknown mode: {mode}. Choose from {list(CONFIG_PROFILES.keys())}")

    cfg = CONFIG_PROFILES[mode]
    BP_LOW_HZ = cfg["BP_LOW_HZ"]
    BP_HIGH_HZ = cfg["BP_HIGH_HZ"]
    STA_WINDOW_S = cfg["STA_WINDOW_S"]
    LTA_WINDOW_S = cfg["LTA_WINDOW_S"]
    STA_LTA_THRESH = cfg["STA_LTA_THRESH"]
    AMP_SIGMA_THRESH = cfg["AMP_SIGMA_THRESH"]
    MERGE_GAP_S = cfg["MERGE_GAP_S"]
    QUIET_GUARD_S = cfg["QUIET_GUARD_S"]
    print(f"Loaded '{mode}' configuration profile")


# ── Stage 1 — Data ingestion ─────────────────────────────────────────────────
def load_data(path: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (time_seconds, magnitude, sample_rate)."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    # Detect timestamp column
    time_col = next((c for c in df.columns if "time" in c), None)
    if time_col:
        col = df[time_col]
        # Float seconds (e.g. 0.062, 0.076) — use directly
        if pd.api.types.is_numeric_dtype(col):
            t_s = (col - col.iloc[0]).values.astype(float)
        else:
            col = pd.to_datetime(col)
            t_s = (col - col.iloc[0]).dt.total_seconds().values
        dt = np.median(np.diff(t_s))
        fs = round(1.0 / dt) if dt > 0 else SAMPLE_RATE
    else:
        fs = SAMPLE_RATE
        t_s = np.arange(len(df)) / fs

    # Compute vector magnitude — support both x/y/z and ax/ay/az column names
    axes = [c for c in ("x", "y", "z") if c in df.columns]
    if not axes:
        axes = [c for c in ("ax", "ay", "az") if c in df.columns]
    if not axes:
        raise ValueError("CSV must contain columns x,y,z or ax,ay,az.")
    mag = np.sqrt(sum(df[a].values ** 2 for a in axes))

    print(f"Loaded {len(mag):,} samples  |  fs = {fs} Hz  |  duration = {t_s[-1]:.1f} s")
    return t_s, mag, float(fs), df.get(time_col)


# ── Stage 2 — Pre-processing ─────────────────────────────────────────────────
def preprocess(mag: np.ndarray, fs: float) -> np.ndarray:
    """Remove DC offset and apply bandpass filter."""
    sig = mag - np.mean(mag)                    # remove gravity / DC offset

    nyq = fs / 2.0
    low  = BP_LOW_HZ  / nyq
    high = min(BP_HIGH_HZ / nyq, 0.99)
    sos  = butter(4, [low, high], btype="band", output="sos")
    filtered = sosfiltfilt(sos, sig)

    return filtered


# ── Stage 3 — Feature extraction (STA/LTA) ───────────────────────────────────
def sta_lta(sig: np.ndarray, fs: float) -> np.ndarray:
    """Compute the STA/LTA ratio for every sample."""
    sta_n = max(1, int(STA_WINDOW_S * fs))
    lta_n = max(1, int(LTA_WINDOW_S * fs))

    energy = sig ** 2
    # Cumulative sum trick for O(n) rolling mean
    cs = np.cumsum(energy)
    cs = np.concatenate(([0.0], cs))

    sta = np.zeros(len(sig))
    lta = np.zeros(len(sig))

    for i in range(len(sig)):
        sta_start = max(0, i - sta_n + 1)
        lta_start = max(0, i - lta_n + 1)
        sta[i] = (cs[i + 1] - cs[sta_start]) / (i - sta_start + 1)
        lta[i] = (cs[i + 1] - cs[lta_start]) / (i - lta_start + 1)

    ratio = np.where(lta > 1e-12, sta / lta, 0.0)
    return ratio


# ── Stage 4 — Spike detection ────────────────────────────────────────────────
def detect_spikes(ratio: np.ndarray, sig: np.ndarray,
                  fs: float) -> list[tuple[int, int]]:
    """Return list of (start_idx, end_idx) spike segments."""
    baseline_std = np.std(sig)
    amp_thresh   = AMP_SIGMA_THRESH * baseline_std

    above = (ratio > STA_LTA_THRESH) | (np.abs(sig) > amp_thresh)

    # Find contiguous blocks
    spikes: list[tuple[int, int]] = []
    in_spike = False
    for i, flag in enumerate(above):
        if flag and not in_spike:
            start = i
            in_spike = True
        elif not flag and in_spike:
            spikes.append((start, i))
            in_spike = False
    if in_spike:
        spikes.append((start, len(above) - 1))

    # Merge nearby spikes
    gap_n = int(MERGE_GAP_S * fs)
    merged: list[tuple[int, int]] = []
    for s, e in spikes:
        if merged and s - merged[-1][1] <= gap_n:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    return merged


# ── Stage 5 — Event windowing ────────────────────────────────────────────────
def find_event_window(spikes: list[tuple[int, int]],
                      ratio: np.ndarray,
                      fs: float) -> tuple[int, int] | None:
    """Merge all spike clusters into a single earthquake window."""
    if not spikes:
        return None

    event_start = spikes[0][0]
    event_end   = spikes[-1][1]

    # Apply quiet-guard: extend end until quiet_guard_s of calm
    quiet_n = int(QUIET_GUARD_S * fs)
    i = event_end
    calm_count = 0
    while i < len(ratio) and calm_count < quiet_n:
        if ratio[i] < STA_LTA_THRESH:
            calm_count += 1
        else:
            calm_count = 0
            event_end = i
        i += 1

    return event_start, min(event_end, len(ratio) - 1)


# ── Stage 6 — Visualization ──────────────────────────────────────────────────
def plot_results(t_s: np.ndarray, mag: np.ndarray, filtered: np.ndarray,
                 ratio: np.ndarray, spikes: list[tuple[int, int]],
                 window: tuple[int, int] | None,
                 timestamps: pd.Series | None) -> None:

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Earthquake Detection Report", fontsize=14, fontweight="bold")

    def idx_to_label(idx):
        if timestamps is not None:
            return str(timestamps.iloc[idx])
        return f"{t_s[idx]:.2f} s"

    # — Panel 1: raw magnitude
    ax1 = axes[0]
    ax1.plot(t_s, mag, color="#4a90d9", linewidth=0.6, label="Raw magnitude |a|")
    ax1.set_ylabel("Magnitude (g)")
    ax1.legend(loc="upper right", fontsize=8)

    # — Panel 2: bandpass-filtered signal
    ax2 = axes[1]
    ax2.plot(t_s, filtered, color="#2ecc71", linewidth=0.6, label="Filtered signal")
    ax2.set_ylabel("Filtered (g)")
    ax2.legend(loc="upper right", fontsize=8)

    # — Panel 3: STA/LTA ratio
    ax3 = axes[2]
    ax3.plot(t_s, ratio, color="#e67e22", linewidth=0.8, label="STA/LTA ratio")
    ax3.axhline(STA_LTA_THRESH, color="red", linestyle="--",
                linewidth=1.0, label=f"Threshold ({STA_LTA_THRESH})")
    ax3.set_ylabel("STA/LTA")
    ax3.set_xlabel("Time (s)")
    ax3.legend(loc="upper right", fontsize=8)

    # Overlay spike markers and earthquake window on all panels
    for ax in axes:
        # Shaded earthquake window
        if window:
            ws, we = window
            ax.axvspan(t_s[ws], t_s[we], color="red", alpha=0.12, zorder=0)

        # Red vertical lines at each spike start
        for s, _ in spikes:
            ax.axvline(t_s[s], color="red", linewidth=1.0, alpha=0.6)

    # Annotate start / end on top panel
    if window:
        ws, we = window
        ax1.annotate(f"Start\n{idx_to_label(ws)}",
                     xy=(t_s[ws], ax1.get_ylim()[1]),
                     fontsize=7, color="red", ha="center",
                     xytext=(t_s[ws], ax1.get_ylim()[1]),
                     textcoords="data")
        ax1.annotate(f"End\n{idx_to_label(we)}",
                     xy=(t_s[we], ax1.get_ylim()[1]),
                     fontsize=7, color="darkred", ha="center",
                     xytext=(t_s[we], ax1.get_ylim()[1]),
                     textcoords="data")

    patch = mpatches.Patch(color="red", alpha=0.3, label="Earthquake window")
    fig.legend(handles=[patch], loc="lower center", ncol=1, fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(OUTPUT_IMAGE, dpi=150)
    print(f"Diagram saved → {OUTPUT_IMAGE}")


# ── Stage 7 — Output report ──────────────────────────────────────────────────
def print_report(window: tuple[int, int] | None,
                 t_s: np.ndarray,
                 timestamps: pd.Series | None) -> None:
    sep = "─" * 55
    print(sep)
    if window is None:
        print("✗  No earthquake detected in this recording.")
    else:
        ws, we = window
        duration = t_s[we] - t_s[ws]
        if timestamps is not None:
            start_label = str(timestamps.iloc[ws])
            end_label   = str(timestamps.iloc[we])
        else:
            start_label = f"{t_s[ws]:.3f} s"
            end_label   = f"{t_s[we]:.3f} s"
        print("✓  Earthquake detected!")
        print(f"   Start    : {start_label}")
        print(f"   End      : {end_label}")
        print(f"   Duration : {duration:.1f} s")
    print(sep)


# ── Main ─────────────────────────────────────────────────────────────────────
def main(mode: str = "earthquake"):
    path = sys.argv[1] if len(sys.argv) > 1 else "accelerometer_data.csv"

    print("\n=== Earthquake Detection Pipeline ===\n")

    load_config(mode)
    print("[1/7] Loading data...")
    t_s, mag, fs, timestamps = load_data(path)

    print("[2/7] Pre-processing (DC removal + bandpass filter)...")
    filtered = preprocess(mag, fs)

    print("[3/7] Computing STA/LTA ratio...")
    ratio = sta_lta(filtered, fs)

    print("[4/7] Detecting spikes...")
    spikes = detect_spikes(ratio, filtered, fs)
    print(f"      Found {len(spikes)} spike cluster(s).")

    print("[5/7] Determining event window...")
    window = find_event_window(spikes, ratio, fs)

    print("[6/7] Generating diagram...")
    plot_results(t_s, mag, filtered, ratio, spikes, window, timestamps)

    print("[7/7] Report:")
    print_report(window, t_s, timestamps)


if __name__ == "__main__":
    main()
