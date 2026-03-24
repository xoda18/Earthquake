#!/usr/bin/env python3
"""
realtime_visualizer.py
Advanced real-time visualization of accelerometer data with earthquake detection highlighting.

Shows live 3D acceleration vector, waveforms, magnitude, and STA/LTA ratio.
Earthquake periods are highlighted in red.

Usage:
    python realtime_visualizer.py --port /dev/ttyACM0 --duration 120 --mode optimized
    python realtime_visualizer.py --duration 60  # auto-detect port
"""

import argparse
import sys
import time
import threading
from collections import deque
from typing import Tuple, Optional
import numpy as np

import os
os.environ["MPLBACKEND"] = "TkAgg"  # must be set before any matplotlib import
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

from hardware.mpu6050_interface import MPU6050Reader
from hardware.sensor_buffer import StreamBuffer
from detect_earthquake import (
    load_config, preprocess, sta_lta, detect_spikes,
    find_event_window, CONFIG_PROFILES
)

# Add "optimized" profile if it doesn't exist
if "optimized" not in CONFIG_PROFILES:
    CONFIG_PROFILES["optimized"] = {
        "BP_LOW_HZ": 1.5,
        "BP_HIGH_HZ": 22.0,
        "STA_WINDOW_S": 0.3,
        "LTA_WINDOW_S": 7.0,
        "STA_LTA_THRESH": 5.0,
        "AMP_SIGMA_THRESH": 5.0,
        "MERGE_GAP_S": 2.0,
        "QUIET_GUARD_S": 4.0,
    }


class RealtimeVisualizer:
    """Real-time visualization of 3-axis accelerometer with earthquake detection."""

    def __init__(
        self,
        port: Optional[str] = None,
        duration: int = 60,
        mode: str = "optimized",
        save_csv: Optional[str] = None,
        print_data: bool = False,
        from_csv: Optional[str] = None,
        from_blackboard: Optional[str] = None,
    ):
        """
        Initialize visualizer.

        Args:
            port: Serial port (auto-detect if None).
            duration: Monitoring duration in seconds.
            mode: Detection mode ("earthquake", "table_knock", or "optimized").
            save_csv: Path to save CSV recording (None to skip).
        """
        self.reader = MPU6050Reader(port=port)
        self.buffer = StreamBuffer(capacity=1000, sample_rate=100)
        self.duration = duration
        self.mode = mode
        self.save_csv = save_csv
        self.print_data = print_data
        self.from_csv = from_csv
        self.from_blackboard = from_blackboard
        self.running = False
        self.last_print_time = 0
        self.samples_collected = []
        self.earthquake_regions = []  # List of (start, end) indices
        self.detection_count = 0
        self.last_update_time = 0

        load_config(mode)
        self.config = CONFIG_PROFILES[mode]

        # Data storage for analysis
        self.all_timestamps = deque(maxlen=1000)
        self.all_x = deque(maxlen=1000)
        self.all_y = deque(maxlen=1000)
        self.all_z = deque(maxlen=1000)
        self.all_mag = deque(maxlen=1000)
        self.all_ratio = deque(maxlen=1000)

        # Matplotlib figure setup
        self.fig = None
        self.axes = {}

    def reader_thread_func(self):
        """Background thread: read from hardware or tail CSV file."""
        csv_out_file = None
        csv_writer = None

        if self.from_blackboard:
            self._read_from_blackboard()
            return
        if self.from_csv:
            self._read_from_csv()
            return

        try:
            self.reader.connect()
            start_time = time.time()

            if self.save_csv:
                import csv
                csv_out_file = open(self.save_csv, "w")
                csv_writer = csv.writer(csv_out_file)
                csv_writer.writerow(["timestamp_s", "ax", "ay", "az"])

            if self.print_data:
                print(f"{'Time':>8}  {'ax':>7}  {'ay':>7}  {'az':>7}  {'|a|':>7}")
                print("-" * 46)

            while self.running and (time.time() - start_time) < self.duration:
                try:
                    sample = self.reader.read_sample()
                    self.buffer.append(sample)
                    self.samples_collected.append(sample)

                    if csv_writer:
                        csv_writer.writerow(sample)

                    if self.print_data:
                        now = time.time()
                        if now - self.last_print_time >= 0.2:
                            self.last_print_time = now
                            ts, ax, ay, az = sample
                            mag = (ax**2 + ay**2 + az**2) ** 0.5
                            print(f"{ts:8.2f}  {ax:7.4f}  {ay:7.4f}  {az:7.4f}  {mag:7.4f}")

                except Exception as e:
                    print(f"Reader warning: {e}", file=sys.stderr)

        except Exception as e:
            print(f"Reader error: {e}", file=sys.stderr)
        finally:
            self.reader.disconnect()
            if csv_out_file:
                csv_out_file.close()
                print(f"Saved to {self.save_csv}")

    def _read_from_csv(self):
        """Tail a growing CSV file, appending samples to buffer."""
        print(f"Reading from CSV: {self.from_csv}")
        start_time = time.time()
        file_pos = 0

        while self.running and (time.time() - start_time) < self.duration:
            try:
                with open(self.from_csv, "r") as f:
                    f.seek(file_pos)
                    new_lines = f.readlines()
                    file_pos = f.tell()

                for line in new_lines:
                    line = line.strip()
                    if not line or line.startswith("timestamp") or line.startswith("#"):
                        continue
                    parts = line.split(",")
                    if len(parts) < 4:
                        continue
                    try:
                        ts = float(parts[0])
                        ax, ay, az = float(parts[1]), float(parts[2]), float(parts[3])
                        self.buffer.append((ts, ax, ay, az))
                        self.samples_collected.append((ts, ax, ay, az))
                    except ValueError:
                        continue

                time.sleep(0.05)

            except FileNotFoundError:
                time.sleep(0.2)
            except Exception as e:
                time.sleep(0.1)

    def _read_from_blackboard(self):
        """Poll blackboard API for sensor data posted by Docker."""
        import requests
        import json

        url = self.from_blackboard
        print(f"Polling blackboard: {url}")
        start_time = time.time()
        sample_idx = 0
        last_ts = 0

        while self.running and (time.time() - start_time) < self.duration:
            try:
                resp = requests.get(url, timeout=2)
                if resp.status_code != 200:
                    time.sleep(0.3)
                    continue

                data = resp.json()
                # Find earthquake_sensor entries
                entries = data if isinstance(data, list) else data.get("entries", [data])

                for entry in entries:
                    if entry.get("type") != "earthquake_sensor":
                        continue
                    try:
                        content = json.loads(entry.get("content", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        continue

                    ts = content.get("timestamp", 0)
                    if ts <= last_ts:
                        continue  # already seen
                    last_ts = ts

                    ax = content.get("ax", 0)
                    ay = content.get("ay", 0)
                    az = content.get("az", 0)

                    sample = (float(sample_idx) / 100.0, ax, ay, az)
                    self.buffer.append(sample)
                    self.samples_collected.append(sample)
                    sample_idx += 1

                    if self.print_data:
                        prob = content.get("probability", 0)
                        label = content.get("label", "?")
                        mag = content.get("magnitude_g", 0)
                        print(f"  BB: ax={ax:.4f} ay={ay:.4f} az={az:.4f} |a|={mag:.4f} prob={prob:.1%} {label}")

                time.sleep(0.3)  # poll 3 times/sec

            except Exception as e:
                time.sleep(0.5)

    def analyze_buffer(self) -> Tuple[Optional[Tuple[int, int]], dict]:
        """
        Analyze current buffer and detect earthquakes.

        Returns:
            (event_window, metrics) where event_window = (start_idx, end_idx) or None.
        """
        if len(self.buffer) < 100:
            return None, {}

        # Extract data from buffer — get_numpy_data returns 4 separate arrays
        timestamps, x, y, z = self.buffer.get_numpy_data()
        if len(timestamps) < 100:
            return None, {}

        # Compute magnitude
        mag = np.sqrt(x**2 + y**2 + z**2)

        # Always return raw data even if detection fails
        metrics = {
            "timestamps": timestamps,
            "x": x,
            "y": y,
            "z": z,
            "mag": mag,
            "filtered": np.zeros_like(mag),
            "ratio": np.zeros_like(mag),
            "spikes": [],
        }

        # Try detection pipeline — if it fails, plots still update with raw data
        fs = self.buffer.sample_rate
        try:
            filtered = preprocess(mag, fs)
            ratio = sta_lta(filtered, fs)
            spikes = detect_spikes(ratio, filtered, fs)
            event_window = find_event_window(spikes, ratio, fs)
            metrics["filtered"] = filtered
            metrics["ratio"] = ratio
            metrics["spikes"] = spikes
            return event_window, metrics
        except Exception:
            return None, metrics

    def setup_figure(self):
        """Create matplotlib figure with 5 subplots."""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle(
            f"Real-Time Seismic Monitoring ({self.mode} mode)",
            fontsize=16, fontweight="bold"
        )

        # 1. 3D acceleration vector
        self.axes["3d"] = self.fig.add_subplot(2, 3, 1, projection="3d")
        self.axes["3d"].set_xlabel("X (g)")
        self.axes["3d"].set_ylabel("Y (g)")
        self.axes["3d"].set_zlabel("Z (g)")
        self.axes["3d"].set_title("3D Acceleration Vector")
        self.axes["3d"].set_xlim(-2, 2)
        self.axes["3d"].set_ylim(-2, 2)
        self.axes["3d"].set_zlim(-2, 2)

        # 2. X, Y, Z waveforms
        self.axes["xyz"] = self.fig.add_subplot(2, 3, 2)
        self.axes["xyz"].set_xlabel("Time (s)")
        self.axes["xyz"].set_ylabel("Acceleration (g)")
        self.axes["xyz"].set_title("X, Y, Z Waveforms")
        self.axes["xyz"].grid(True, alpha=0.3)
        self.axes["xyz"].legend(["X", "Y", "Z"], loc="upper right")

        # 3. Magnitude
        self.axes["mag"] = self.fig.add_subplot(2, 3, 3)
        self.axes["mag"].set_xlabel("Time (s)")
        self.axes["mag"].set_ylabel("Magnitude (g)")
        self.axes["mag"].set_title("Resultant Acceleration Magnitude")
        self.axes["mag"].grid(True, alpha=0.3)

        # 4. STA/LTA ratio
        self.axes["ratio"] = self.fig.add_subplot(2, 3, 4)
        self.axes["ratio"].set_xlabel("Time (s)")
        self.axes["ratio"].set_ylabel("STA/LTA Ratio")
        self.axes["ratio"].set_title("STA/LTA Ratio (Detection Signal)")
        self.axes["ratio"].grid(True, alpha=0.3)

        # 5. Peak acceleration & detection status
        self.axes["stats"] = self.fig.add_subplot(2, 3, 5)
        self.axes["stats"].axis("off")
        self.axes["stats"].set_title("Detection Status")

        # 6. Spectrogram (simplified: magnitude histogram)
        self.axes["hist"] = self.fig.add_subplot(2, 3, 6)
        self.axes["hist"].set_xlabel("Acceleration (g)")
        self.axes["hist"].set_ylabel("Frequency")
        self.axes["hist"].set_title("Acceleration Distribution")
        self.axes["hist"].grid(True, alpha=0.3)

        plt.tight_layout()

    def update_frame(self, frame):
        """Update all plots with latest data."""
        event_window, metrics = self.analyze_buffer()

        if not metrics:
            return

        x = metrics["x"]
        y = metrics["y"]
        z = metrics["z"]
        mag = metrics["mag"]
        ratio = metrics["ratio"]
        timestamps = metrics["timestamps"]
        spikes = metrics["spikes"]

        # Convert timestamps to relative seconds (already in seconds from read_sample)
        if len(timestamps) > 0:
            t_s = timestamps - timestamps[0]
        else:
            return

        # ===== Plot 1: 3D Vector =====
        self.axes["3d"].clear()
        if len(x) > 0:
            # Plot trajectory
            self.axes["3d"].plot(x, y, z, "b-", alpha=0.3, linewidth=0.5)
            # Plot current point
            self.axes["3d"].scatter([x[-1]], [y[-1]], [z[-1]], color="red", s=100, label="Current")
        self.axes["3d"].set_xlabel("X (g)")
        self.axes["3d"].set_ylabel("Y (g)")
        self.axes["3d"].set_zlabel("Z (g)")
        self.axes["3d"].set_xlim(-2, 2)
        self.axes["3d"].set_ylim(-2, 2)
        self.axes["3d"].set_zlim(-2, 2)
        self.axes["3d"].set_title("3D Acceleration Vector")
        self.axes["3d"].legend()

        # ===== Plot 2: X, Y, Z Waveforms =====
        self.axes["xyz"].clear()
        self.axes["xyz"].plot(t_s, x, "r-", label="X", linewidth=1, alpha=0.8)
        self.axes["xyz"].plot(t_s, y, "g-", label="Y", linewidth=1, alpha=0.8)
        self.axes["xyz"].plot(t_s, z, "b-", label="Z", linewidth=1, alpha=0.8)

        # Highlight earthquake regions
        if event_window:
            start_idx, end_idx = event_window
            if start_idx < len(t_s) and end_idx < len(t_s):
                t_start = t_s[start_idx]
                t_end = t_s[end_idx]
                self.axes["xyz"].axvspan(t_start, t_end, alpha=0.2, color="red", label="Earthquake")

        self.axes["xyz"].set_xlabel("Time (s)")
        self.axes["xyz"].set_ylabel("Acceleration (g)")
        self.axes["xyz"].set_title("X, Y, Z Waveforms")
        self.axes["xyz"].grid(True, alpha=0.3)
        self.axes["xyz"].legend(loc="upper right")

        # ===== Plot 3: Magnitude =====
        self.axes["mag"].clear()
        self.axes["mag"].plot(t_s, mag, "k-", linewidth=1.5, label="Magnitude")
        self.axes["mag"].axhline(
            y=np.mean(mag[: len(mag) // 4]) if len(mag) > 4 else 1,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label="Baseline"
        )

        # Highlight earthquake
        if event_window:
            start_idx, end_idx = event_window
            if start_idx < len(t_s) and end_idx < len(t_s):
                t_start = t_s[start_idx]
                t_end = t_s[end_idx]
                self.axes["mag"].axvspan(t_start, t_end, alpha=0.2, color="red")
                peak_mag = np.max(mag[start_idx:end_idx])
                self.axes["mag"].scatter(
                    [(t_start + t_end) / 2], [peak_mag], color="red", s=100, zorder=5
                )

        self.axes["mag"].set_xlabel("Time (s)")
        self.axes["mag"].set_ylabel("Magnitude (g)")
        self.axes["mag"].set_title("Resultant Acceleration Magnitude")
        self.axes["mag"].grid(True, alpha=0.3)
        self.axes["mag"].legend()

        # ===== Plot 4: STA/LTA Ratio =====
        self.axes["ratio"].clear()
        self.axes["ratio"].plot(t_s, ratio, "purple", linewidth=1.5, label="STA/LTA")
        threshold = self.config["STA_LTA_THRESH"]
        self.axes["ratio"].axhline(
            y=threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold ({threshold})"
        )
        self.axes["ratio"].fill_between(t_s, 0, ratio, alpha=0.2, color="purple")

        # Highlight spikes (spikes is list of (start, end) tuples)
        if len(spikes) > 0:
            for s_start, s_end in spikes:
                if s_start < len(t_s):
                    self.axes["ratio"].axvline(t_s[s_start], color="red", alpha=0.5, linewidth=1)

        # Highlight earthquake window
        if event_window:
            start_idx, end_idx = event_window
            if start_idx < len(t_s) and end_idx < len(t_s):
                t_start = t_s[start_idx]
                t_end = t_s[end_idx]
                self.axes["ratio"].axvspan(t_start, t_end, alpha=0.2, color="red")

        self.axes["ratio"].set_xlabel("Time (s)")
        self.axes["ratio"].set_ylabel("STA/LTA Ratio")
        self.axes["ratio"].set_title("STA/LTA Ratio (Detection Signal)")
        self.axes["ratio"].grid(True, alpha=0.3)
        self.axes["ratio"].legend(loc="upper right")

        # ===== Plot 5: Status Text =====
        self.axes["stats"].clear()
        self.axes["stats"].axis("off")

        peak_mag = np.max(mag) if len(mag) > 0 else 0
        current_mag = mag[-1] if len(mag) > 0 else 0
        current_ratio = ratio[-1] if len(ratio) > 0 else 0

        status_text = f"""
DETECTION STATUS
{'=' * 40}

Mode: {self.mode.upper()}
Duration: {t_s[-1]:.1f}s / {self.duration}s

Current Acceleration:
  X: {x[-1]:+.3f} g
  Y: {y[-1]:+.3f} g
  Z: {z[-1]:+.3f} g

Magnitude:
  Current: {current_mag:.3f} g
  Peak: {peak_mag:.3f} g
  Baseline: {np.mean(mag[:min(100, len(mag))]):.3f} g

STA/LTA:
  Current: {current_ratio:.3f}
  Threshold: {threshold:.3f}
  Status: {'🔴 EARTHQUAKE' if current_ratio > threshold else '🟢 QUIET'}

Detections: {self.detection_count}
"""

        if event_window:
            status_text += f"\n✓ EARTHQUAKE DETECTED\n  Window: {event_window[0]}-{event_window[1]}"
            self.detection_count += 1

        self.axes["stats"].text(
            0.05, 0.95, status_text, transform=self.axes["stats"].transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        )

        # ===== Plot 6: Magnitude Distribution =====
        self.axes["hist"].clear()
        self.axes["hist"].hist(mag, bins=30, color="steelblue", alpha=0.7, edgecolor="black")
        self.axes["hist"].axvline(
            x=np.mean(mag), color="green", linestyle="--", linewidth=2, label="Mean"
        )
        self.axes["hist"].set_xlabel("Acceleration (g)")
        self.axes["hist"].set_ylabel("Frequency")
        self.axes["hist"].set_title("Acceleration Distribution")
        self.axes["hist"].grid(True, alpha=0.3)
        self.axes["hist"].legend()

    def run(self):
        """Start real-time monitoring and visualization."""
        print(f"Starting real-time visualizer (mode={self.mode})...")
        print(f"Duration: {self.duration}s")
        print(f"Port: {self.reader.port}")
        if self.save_csv:
            print(f"Recording to: {self.save_csv}")

        self.running = True

        # Start reader thread
        reader_thread = threading.Thread(target=self.reader_thread_func, daemon=True)
        reader_thread.start()

        # Setup matplotlib
        self.setup_figure()

        # Create animation
        ani = FuncAnimation(
            self.fig,
            self.update_frame,
            interval=200,  # Update every 200ms
            blit=False,
            cache_frame_data=False,
        )

        try:
            plt.show()
        except KeyboardInterrupt:
            print("\n✓ Stopped by user")
        finally:
            self.running = False
            reader_thread.join(timeout=5)

        print(f"\n=== Analysis Summary ===")
        print(f"Total samples collected: {len(self.samples_collected)}")
        print(f"Earthquakes detected: {self.detection_count}")
        if self.samples_collected:
            mags = [np.sqrt(s[1]**2 + s[2]**2 + s[3]**2) for s in self.samples_collected]
            print(f"Peak acceleration: {max(mags):.3f}g")
            print(f"Average acceleration: {np.mean(mags):.3f}g")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time earthquake detection visualization"
    )
    parser.add_argument(
        "--port", type=str, default=None, help="Serial port (auto-detect if not specified)"
    )
    parser.add_argument(
        "--duration", type=int, default=60, help="Monitoring duration in seconds"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["earthquake", "table_knock", "optimized"],
        default="optimized",
        help="Detection mode",
    )
    parser.add_argument(
        "--save-csv", type=str, default=None, help="Save recording to CSV file"
    )
    parser.add_argument(
        "--print", dest="print_data", action="store_true",
        help="Also print data to terminal (5 Hz)"
    )
    parser.add_argument(
        "--from-csv", type=str, default=None,
        help="Read from a growing CSV file instead of serial port"
    )
    parser.add_argument(
        "--from-blackboard", type=str, default=None,
        help="Read from blackboard API (e.g. https://blackboard.jass.school/blackboard)"
    )

    args = parser.parse_args()

    visualizer = RealtimeVisualizer(
        port=args.port,
        duration=args.duration,
        mode=args.mode,
        save_csv=args.save_csv,
        print_data=args.print_data,
        from_csv=args.from_csv,
        from_blackboard=args.from_blackboard,
    )
    visualizer.run()


if __name__ == "__main__":
    main()
