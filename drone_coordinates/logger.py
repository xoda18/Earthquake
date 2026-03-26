#!/usr/bin/env python3
"""
logger.py — Drone Coordinates: log accelerometer data at 10 Hz with live graph.

Reads Arduino serial (MPU6050/6500), downsamples to 10 samples/sec,
writes timestamped CSV logs, and shows a real-time matplotlib plot.

Usage:
    python drone_coordinates/logger.py                        # auto-detect port
    python drone_coordinates/logger.py --port /dev/ttyACM0    # specific port
    python drone_coordinates/logger.py --duration 300         # 5 minutes
    python drone_coordinates/logger.py --no-graph             # log only, no GUI
"""

import argparse
import csv
import glob
import os
import sys
import time
import threading
import math
from collections import deque
from datetime import datetime, timezone

import numpy as np

os.environ.setdefault("MPLBACKEND", "TkAgg")

# ── Style (matches existing earthquake visualizer) ───────────────────────────
COLOR_X = "#ff6b6b"
COLOR_Y = "#51cf66"
COLOR_Z = "#339af0"
COLOR_MAG = "#ffd43b"
BG_FIG = "#0d1117"
BG_AX = "#161b22"
GRID_COLOR = "#30363d"
TEXT_COLOR = "#c9d1d9"
TICK_COLOR = "#8b949e"

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
SAMPLE_RATE = 10  # target Hz
SAMPLE_INTERVAL = 1.0 / SAMPLE_RATE
WINDOW_SECONDS = 30
WINDOW_SIZE = WINDOW_SECONDS * SAMPLE_RATE  # 300 samples in rolling window


def auto_detect_port():
    candidates = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    if candidates:
        return candidates[0]
    raise RuntimeError("No serial port found. Is the Arduino plugged in?")


def make_log_path():
    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(LOG_DIR, f"{stamp}.csv")


# ── Serial reader thread ────────────────────────────────────────────────────

def reader_thread(port, baudrate, data_deque, lock, log_path, running, duration):
    """Read Arduino serial, downsample to 10 Hz, log to CSV and push to deque."""
    import serial

    try:
        ser = serial.Serial(port, baudrate, timeout=1)
    except Exception as e:
        print(f"Serial error: {e}")
        running[0] = False
        return

    time.sleep(2.0)  # wait for Arduino reset
    ser.reset_input_buffer()
    for _ in range(10):
        ser.readline()  # discard stale lines

    print(f"Connected to {port} at {baudrate} baud")

    csv_file = open(log_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["timestamp_iso", "timestamp_epoch", "ax", "ay", "az", "magnitude"])

    last_log_time = 0.0
    sample_count = 0
    start = time.time()

    try:
        while running[0] and (time.time() - start) < duration:
            raw = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw or raw.startswith("timestamp") or raw.startswith("#"):
                continue

            parts = raw.split(",")
            if len(parts) < 4:
                continue

            try:
                ax = float(parts[1])
                ay = float(parts[2])
                az = float(parts[3])
            except (ValueError, IndexError):
                continue

            # Downsample: only keep one sample per ~100ms
            now = time.time()
            if now - last_log_time < SAMPLE_INTERVAL:
                continue
            last_log_time = now

            mag = math.sqrt(ax * ax + ay * ay + az * az)
            ts_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            ts_epoch = round(now, 3)

            # Write CSV
            writer.writerow([ts_iso, ts_epoch, f"{ax:.4f}", f"{ay:.4f}", f"{az:.4f}", f"{mag:.4f}"])
            csv_file.flush()

            # Push to shared deque for graph
            with lock:
                data_deque.append((now, ax, ay, az, mag))

            sample_count += 1
            if sample_count % SAMPLE_RATE == 0:
                elapsed = time.time() - start
                print(f"  [{elapsed:6.1f}s] {sample_count} samples | ax={ax:+.3f} ay={ay:+.3f} az={az:+.3f} |a|={mag:.3f}")

    except Exception as e:
        print(f"Reader error: {e}")
    finally:
        running[0] = False
        ser.close()
        csv_file.close()
        elapsed = time.time() - start
        rate = sample_count / max(elapsed, 0.001)
        print(f"\nLogged {sample_count} samples in {elapsed:.1f}s ({rate:.1f} Hz)")
        print(f"CSV saved: {log_path}")


# ── Live graph ───────────────────────────────────────────────────────────────

def run_graph(data_deque, lock, running):
    """Real-time matplotlib visualization — runs on main thread."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    plt.style.use("dark_background")

    fig, (ax_xyz, ax_mag) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.patch.set_facecolor(BG_FIG)
    fig.suptitle("Drone Coordinates — Accelerometer", fontsize=14,
                 fontweight="bold", color=TEXT_COLOR)

    for ax in (ax_xyz, ax_mag):
        ax.set_facecolor(BG_AX)
        ax.tick_params(colors=TICK_COLOR, labelsize=9)
        ax.grid(True, alpha=0.15, color=GRID_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)

    line_x, = ax_xyz.plot([], [], COLOR_X, lw=0.9, label="X", alpha=0.9)
    line_y, = ax_xyz.plot([], [], COLOR_Y, lw=0.9, label="Y", alpha=0.9)
    line_z, = ax_xyz.plot([], [], COLOR_Z, lw=0.9, label="Z", alpha=0.9)
    ax_xyz.set_ylabel("Acceleration (g)", color=TICK_COLOR)
    ax_xyz.set_title("X / Y / Z", color=TEXT_COLOR, fontsize=11)
    ax_xyz.legend(loc="upper right", fontsize=9, facecolor=BG_AX, edgecolor=GRID_COLOR)

    line_mag, = ax_mag.plot([], [], COLOR_MAG, lw=1.2)
    ax_mag.set_ylabel("Magnitude (g)", color=TICK_COLOR)
    ax_mag.set_xlabel("Time (s)", color=TICK_COLOR)
    ax_mag.set_title("|a| = sqrt(ax² + ay² + az²)", color=TEXT_COLOR, fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    def update(frame):
        with lock:
            if len(data_deque) < 2:
                return
            data = list(data_deque)

        arr = np.array(data)
        t = arr[:, 0]
        t_rel = t - t[0]  # seconds since start
        x, y, z, mag = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]

        line_x.set_data(t_rel, x)
        line_y.set_data(t_rel, y)
        line_z.set_data(t_rel, z)
        ax_xyz.set_xlim(t_rel[0], max(t_rel[-1], 1.0))
        lo = min(x.min(), y.min(), z.min()) - 0.1
        hi = max(x.max(), y.max(), z.max()) + 0.1
        ax_xyz.set_ylim(lo, hi)

        line_mag.set_data(t_rel, mag)
        ax_mag.set_xlim(t_rel[0], max(t_rel[-1], 1.0))
        ax_mag.set_ylim(max(0, mag.min() - 0.1), mag.max() + 0.2)

    ani = FuncAnimation(fig, update, interval=200, cache_frame_data=False)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        running[0] = False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Drone Coordinates: accelerometer logger with live graph")
    parser.add_argument("--port", default=None, help="Serial port (auto-detect if omitted)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default 115200)")
    parser.add_argument("--duration", type=int, default=600, help="Max duration in seconds (default 600)")
    parser.add_argument("--no-graph", action="store_true", help="Disable live graph (log only)")
    args = parser.parse_args()

    port = args.port or auto_detect_port()
    log_path = make_log_path()

    print(f"Drone Coordinates Logger")
    print(f"  Port:     {port}")
    print(f"  Rate:     {SAMPLE_RATE} Hz")
    print(f"  Duration: {args.duration}s")
    print(f"  Graph:    {'OFF' if args.no_graph else 'ON'}")
    print(f"  Log:      {log_path}")
    print()

    data_deque = deque(maxlen=WINDOW_SIZE)
    lock = threading.Lock()
    running = [True]

    t = threading.Thread(
        target=reader_thread,
        args=(port, args.baud, data_deque, lock, log_path, running, args.duration),
        daemon=True,
    )
    t.start()

    if args.no_graph:
        try:
            t.join()
        except KeyboardInterrupt:
            print("\nStopping...")
            running[0] = False
            t.join(timeout=3)
    else:
        run_graph(data_deque, lock, running)
        running[0] = False
        t.join(timeout=3)

    print("Done.")


if __name__ == "__main__":
    main()
