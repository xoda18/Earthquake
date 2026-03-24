"""
realtime_monitor.py
Real-time streaming from MPU6050 with live detection and plotting.

Usage:
    python realtime_monitor.py --port /dev/ttyUSB0 --duration 60 --save-csv knock_rec.csv
    python realtime_monitor.py --duration 120  # auto-detect port, no saving
"""

import argparse
import sys
import time
import csv
import threading
from collections import deque
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Interactive plotting
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from hardware.mpu6050_interface import MPU6050Reader
from hardware.sensor_buffer import StreamBuffer
from detect_earthquake import load_config, preprocess, sta_lta
from detect_earthquake import detect_spikes, find_event_window, STA_LTA_THRESH


class RealtimeMonitor:
    """Live monitoring of accelerometer with event detection."""

    def __init__(self, port=None, duration=60, save_csv=None, mode="table_knock"):
        """
        Initialize monitor.

        Args:
            port: Serial port (auto-detect if None).
            duration: Monitoring duration in seconds.
            save_csv: Path to save recording CSV (None to skip).
            mode: Detection mode ("earthquake" or "table_knock").
        """
        self.reader = MPU6050Reader(port=port)
        self.buffer = StreamBuffer(capacity=600, sample_rate=100)
        self.duration = duration
        self.save_csv = save_csv
        self.mode = mode
        self.running = False
        self.samples_collected = []
        self.csv_file = None
        self.csv_writer = None
        self.last_detection_time = None
        self.detection_count = 0

        load_config(mode)

    def reader_thread_func(self):
        """Background thread: read from hardware and append to buffer."""
        try:
            self.reader.connect()
            start_time = time.time()

            while self.running and (time.time() - start_time) < self.duration:
                try:
                    sample = self.reader.read_sample()
                    self.buffer.append(sample)
                    self.samples_collected.append(sample)

                    # Write to CSV if enabled
                    if self.csv_writer:
                        self.csv_writer.writerow(sample)

                except Exception as e:
                    print(f"Warning: {e}")

        except Exception as e:
            print(f"Reader error: {e}")
        finally:
            self.reader.disconnect()

    def run(self):
        """Start monitoring."""
        self.running = True

        # Open CSV file if specified
        if self.save_csv:
            self.csv_file = open(self.save_csv, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["timestamp", "x", "y", "z"])

        # Start reader thread
        reader_thread = threading.Thread(target=self.reader_thread_func, daemon=True)
        reader_thread.start()

        # Setup plotting
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        fig.suptitle(f"Real-Time {self.mode.upper()} Monitoring", fontsize=12, fontweight="bold")

        lines = [ax.plot([], [], lw=1)[0] for ax in axes]
        axes[0].set_ylabel("Magnitude (g)")
        axes[1].set_ylabel("Filtered (g)")
        axes[2].set_ylabel("STA/LTA")
        axes[2].set_xlabel("Time (s)")

        # Set axis limits
        for ax in axes:
            ax.set_xlim(0, 20)

        axes[0].set_ylim(-2, 2)
        axes[1].set_ylim(-1, 1)
        axes[2].set_ylim(0, 5)

        # Add threshold line
        axes[2].axhline(STA_LTA_THRESH, color="red", linestyle="--", lw=1, label=f"Threshold")
        axes[2].legend(loc="upper right", fontsize=8)

        # Status text
        status_text = fig.text(0.02, 0.02, "", fontsize=9, family="monospace")

        def animate(frame):
            """Update plot."""
            if not self.buffer.is_full():
                return lines + [status_text]

            # Get window data
            window = self.buffer.get_window(600)  # Last 6 seconds
            if window is None:
                return lines + [status_text]

            t_data = window[:, 0] - window[0, 0]
            mag = np.sqrt(window[:, 1] ** 2 + window[:, 2] ** 2 + window[:, 3] ** 2)

            # Pre-process and compute features
            filtered = preprocess(mag, 100)
            ratio = sta_lta(filtered, 100)

            # Detect
            spikes = detect_spikes(ratio, filtered, 100)
            event_window = find_event_window(spikes, ratio, 100)

            if event_window and self.last_detection_time != event_window[0]:
                self.last_detection_time = event_window[0]
                self.detection_count += 1
                print(f"\n🚨 EVENT DETECTED at {t_data[event_window[0]]:.1f}s")

            # Update lines
            lines[0].set_data(t_data, mag)
            lines[1].set_data(t_data, filtered)
            lines[2].set_data(t_data, ratio)

            # Highlight event window
            for ax in axes:
                ax.clear()
                ax.axhline(0, color="k", lw=0.5, alpha=0.3)
                if event_window:
                    ax.axvspan(
                        t_data[event_window[0]], t_data[event_window[1]],
                        color="red", alpha=0.15
                    )
                if ax == axes[2]:
                    ax.axhline(STA_LTA_THRESH, color="red", linestyle="--", lw=1)

            axes[0].plot(t_data, mag, color="#4a90d9", lw=0.8)
            axes[0].set_ylabel("Magnitude (g)")
            axes[0].set_xlim(0, 6)
            axes[0].set_ylim(0, max(mag) * 1.2 if len(mag) > 0 else 1)

            axes[1].plot(t_data, filtered, color="#2ecc71", lw=0.8)
            axes[1].set_ylabel("Filtered (g)")
            axes[1].set_xlim(0, 6)
            axes[1].set_ylim(-1, 1)

            axes[2].plot(t_data, ratio, color="#e67e22", lw=0.8)
            axes[2].set_ylabel("STA/LTA")
            axes[2].set_xlabel("Time (s)")
            axes[2].set_xlim(0, 6)
            axes[2].set_ylim(0, max(ratio) * 1.2 if len(ratio) > 0 else 5)

            # Update status
            elapsed = time.time() - (time.time() - self.duration)
            fs = self.reader.get_sample_rate()
            fill = self.buffer.get_fill_ratio()
            status = (
                f"Mode: {self.mode}  |  Samples: {len(self.samples_collected)}  |  "
                f"Fs: {fs:.0f} Hz  |  Buffer: {fill*100:.0f}%  |  "
                f"Detections: {self.detection_count}"
            )
            status_text.set_text(status)

            return lines + [status_text]

        # Animate
        anim = FuncAnimation(
            fig, animate, interval=200, blit=True, repeat=True
        )

        plt.tight_layout()
        plt.show()

        # Stop monitoring
        self.running = False
        reader_thread.join(timeout=2)

        # Close CSV
        if self.csv_file:
            self.csv_file.close()
            print(f"\nSaved recording → {self.save_csv}")

        print(f"\nMonitoring stopped. Total detections: {self.detection_count}")


def main():
    parser = argparse.ArgumentParser(description="Real-time accelerometer monitoring.")
    parser.add_argument("--port", default=None, help="Serial port (auto-detect if not specified)")
    parser.add_argument("--duration", type=int, default=60, help="Monitoring duration in seconds")
    parser.add_argument(
        "--save-csv", default=None, help="Optional: save recording to CSV file"
    )
    parser.add_argument(
        "--mode", default="table_knock", choices=["earthquake", "table_knock"],
        help="Detection mode (default: table_knock)"
    )

    args = parser.parse_args()

    try:
        monitor = RealtimeMonitor(
            port=args.port,
            duration=args.duration,
            save_csv=args.save_csv,
            mode=args.mode,
        )
        monitor.run()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
