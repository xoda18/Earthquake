#!/usr/bin/env python3
"""
realtime_visualizer.py
Real-time LSTM earthquake detection with visualization.

3 threads: reader (serial) + inference (LSTM) + main (plots)
Auto-reconnects Arduino if it hangs.

Usage:
    python detection/realtime_visualizer.py --port /dev/tty.usbmodem11301
"""

import argparse
import sys
import os
import time
import threading
import json
from collections import deque
from datetime import datetime, timezone
import numpy as np

os.environ["MPLBACKEND"] = "TkAgg"
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from realtime_predict import load_model_and_scaler, predict_window, WINDOW_SIZE, STRIDE, THRESHOLD

# ── Style ─────────────────────────────────────────────────────────────────────
plt.style.use("dark_background")
COLOR_X = "#ff6b6b"
COLOR_Y = "#51cf66"
COLOR_Z = "#339af0"
COLOR_MAG = "#ffd43b"
COLOR_PROB_OK = "#51cf66"
COLOR_PROB_EQ = "#ff6b6b"
COLOR_THRESH = "#ff922b"
BG_QUIET = "#1a2f1a"
BG_EARTHQUAKE = "#3f1a1a"

# Supabase client (shared)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import supabase_client as sb

# Live data file for web dashboard
LIVE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sensor_live.json")


class LSTMVisualizer:

    def __init__(self, port, duration=600):
        self.port = port
        self.duration = duration
        self.running = False

        print("Loading LSTM model...")
        self.model, self.scaler = load_model_and_scaler()

        self.lock = threading.Lock()

        # Plot data
        N = 800
        self.times = deque(maxlen=N)
        self.data_x = deque(maxlen=N)
        self.data_y = deque(maxlen=N)
        self.data_z = deque(maxlen=N)
        self.data_mag = deque(maxlen=N)

        # LSTM
        self.lstm_buffer = deque(maxlen=WINDOW_SIZE)
        self.samples_since_inference = 0
        self.current_prob = 0.0
        self.current_label = "noise"
        self.prob_history = deque(maxlen=60)
        self.prob_times = deque(maxlen=60)
        self.detection_count = 0
        self.peak_accel = 0.0
        self.last_was_earthquake = False
        self.total_samples = 0
        self.session_start = time.time()

        self.inference_ready = threading.Event()
        self.inference_window = None
        self.last_sample_time = time.time()
        self.last_live_write = 0

    def write_live_data(self):
        """Write latest sensor buffer to local file. Supabase writes throttled."""
        now = time.time()
        # Local file every 0.3s
        if now - self.last_live_write < 0.3:
            return
        self.last_live_write = now
        try:
            with self.lock:
                data = {
                    "x": [round(v, 4) for v in list(self.data_x)[-100:]],
                    "y": [round(v, 4) for v in list(self.data_y)[-100:]],
                    "z": [round(v, 4) for v in list(self.data_z)[-100:]],
                    "mag": [round(v, 4) for v in list(self.data_mag)[-100:]],
                    "prob": round(self.current_prob, 4),
                    "label": self.current_label,
                    "pga": round(self.peak_accel, 4),
                    "detections": self.detection_count,
                    "samples": self.total_samples,
                    "elapsed": round(now - self.session_start, 1),
                    "prob_history": [round(v, 4) for v in list(self.prob_history)],
                }
            data_json = json.dumps(data)
            # Local file (instant)
            tmp = LIVE_FILE + ".tmp"
            with open(tmp, "w") as f:
                f.write(data_json)
            os.replace(tmp, LIVE_FILE)
            # Supabase only every 2 sec, reuse single thread
            if not hasattr(self, '_sb_busy'):
                self._sb_busy = False
                self._sb_last = 0
            if not self._sb_busy and (now - self._sb_last) > 2.0:
                self._sb_busy = True
                self._sb_last = now
                def _send(d):
                    try: sb.insert("sensor_stream", {"data": d})
                    except: pass
                    self._sb_busy = False
                threading.Thread(target=_send, args=(data_json,), daemon=True).start()
        except Exception:
            pass

    def _open_serial(self):
        """Open serial port, return connection or None."""
        import serial
        try:
            ser = serial.Serial(self.port, 115200, timeout=1)
            time.sleep(2.0)  # wait for Arduino reset
            ser.reset_input_buffer()
            for _ in range(10):
                ser.readline()
            print(f"Connected to {self.port}")
            self.last_sample_time = time.time()
            return ser
        except Exception as e:
            print(f"Serial error: {e}")
            return None

    def reader_thread(self):
        """Read serial with auto-reconnect on Arduino hang."""
        ser = self._open_serial()
        if not ser:
            self.running = False
            return

        start = time.time()

        try:
            while self.running and (time.time() - start) < self.duration:
                # Check for Arduino hang (no data for 3 seconds)
                if time.time() - self.last_sample_time > 3.0:
                    print("\nArduino hang detected — reconnecting...")
                    try:
                        ser.close()
                    except Exception:
                        pass
                    time.sleep(0.5)
                    ser = self._open_serial()
                    if not ser:
                        time.sleep(2)
                        ser = self._open_serial()
                        if not ser:
                            print("Reconnect failed.")
                            break
                    continue

                raw = ser.readline().decode("utf-8", errors="ignore").strip()
                if not raw:
                    # readline returned empty = timeout, check if Arduino hung
                    if time.time() - self.last_sample_time > 3.0:
                        print("\nArduino timeout — reconnecting...")
                        try: ser.close()
                        except Exception: pass
                        time.sleep(0.5)
                        ser = self._open_serial()
                        if not ser:
                            time.sleep(2)
                            ser = self._open_serial()
                            if not ser:
                                print("Reconnect failed.")
                                break
                    continue
                if raw.startswith("timestamp") or raw.startswith("#"):
                    continue

                parts = raw.split(",")
                if len(parts) not in (4, 7):
                    continue

                try:
                    ts_s = float(parts[0]) / 1000.0
                    ax, ay, az = float(parts[1]), float(parts[2]), float(parts[3])
                except ValueError:
                    continue

                self.last_sample_time = time.time()
                self.total_samples += 1
                mag = np.sqrt(ax**2 + ay**2 + az**2)

                with self.lock:
                    self.times.append(ts_s)
                    self.data_x.append(ax)
                    self.data_y.append(ay)
                    self.data_z.append(az)
                    self.data_mag.append(mag)
                    if mag > self.peak_accel:
                        self.peak_accel = mag

                    self.lstm_buffer.append((ax, ay, az))
                    self.samples_since_inference += 1

                    if (len(self.lstm_buffer) == WINDOW_SIZE
                            and self.samples_since_inference >= STRIDE
                            and not self.inference_ready.is_set()):
                        self.samples_since_inference = 0
                        self.inference_window = np.array(self.lstm_buffer)
                        self.inference_ready.set()

                self.write_live_data()

        except Exception as e:
            print(f"Reader error: {e}")
        finally:
            try:
                ser.close()
            except Exception:
                pass

    def log_earthquake(self, prob, pga, ax, ay, az, mag):
        """Write earthquake event to Supabase (read by swarm agent)."""
        entry = {
            "epoch": time.time(),
            "probability": round(prob, 4),
            "pga_g": round(pga, 4),
            "ax": round(ax, 3),
            "ay": round(ay, 3),
            "az": round(az, 3),
            "magnitude_g": round(mag, 4),
        }
        ok = sb.insert("sensor_events", entry)
        if ok:
            print(f"  >> Supabase: earthquake logged (P={prob:.1%}, PGA={pga:.3f}g)")
        else:
            print(f"  >> Supabase: failed to log")

    def inference_thread(self):
        while self.running:
            if not self.inference_ready.wait(timeout=1.0):
                continue
            self.inference_ready.clear()

            with self.lock:
                window = self.inference_window
                ts = self.times[-1] if self.times else 0
                pga = self.peak_accel
                last_ax = self.data_x[-1] if self.data_x else 0
                last_ay = self.data_y[-1] if self.data_y else 0
                last_az = self.data_z[-1] if self.data_z else 0
                last_mag = self.data_mag[-1] if self.data_mag else 0

            if window is None:
                continue

            prob, label = predict_window(window, self.model, self.scaler)

            with self.lock:
                self.current_prob = prob
                self.current_label = label
                self.prob_history.append(prob)
                self.prob_times.append(ts)
                if label == "EARTHQUAKE" and not self.last_was_earthquake:
                    self.detection_count += 1
                    self.log_earthquake(prob, pga, last_ax, last_ay, last_az, last_mag)
                self.last_was_earthquake = (label == "EARTHQUAKE")

    def setup_figure(self):
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.patch.set_facecolor("#0d1117")

        gs = self.fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25,
                                    left=0.06, right=0.98, top=0.92, bottom=0.06)

        # Top left: XYZ (tall)
        self.ax_xyz = self.fig.add_subplot(gs[0:2, 0])
        # Top right: Magnitude
        self.ax_mag = self.fig.add_subplot(gs[0, 1])
        # Middle right: LSTM probability
        self.ax_prob = self.fig.add_subplot(gs[1, 1])
        # Bottom: Status bar
        self.ax_status = self.fig.add_subplot(gs[2, :])

        for ax in [self.ax_xyz, self.ax_mag, self.ax_prob]:
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#8b949e", labelsize=9)
            ax.grid(True, alpha=0.15, color="#30363d")
            for spine in ax.spines.values():
                spine.set_color("#30363d")

        self.ax_status.set_facecolor(BG_QUIET)
        self.ax_status.axis("off")

        # Pre-create line objects for speed (no clear/replot)
        self.line_x, = self.ax_xyz.plot([], [], COLOR_X, lw=0.9, label="X", alpha=0.9)
        self.line_y, = self.ax_xyz.plot([], [], COLOR_Y, lw=0.9, label="Y", alpha=0.9)
        self.line_z, = self.ax_xyz.plot([], [], COLOR_Z, lw=0.9, label="Z", alpha=0.9)
        self.ax_xyz.set_title("Acceleration  X / Y / Z", color="#c9d1d9", fontsize=12, fontweight="bold")
        self.ax_xyz.set_ylabel("g", color="#8b949e")
        self.ax_xyz.legend(loc="upper right", fontsize=9, facecolor="#161b22", edgecolor="#30363d")

        self.line_mag, = self.ax_mag.plot([], [], COLOR_MAG, lw=1.2)
        self.ax_mag.set_title("Magnitude |a|", color="#c9d1d9", fontsize=12, fontweight="bold")
        self.ax_mag.set_ylabel("g", color="#8b949e")

        self.ax_prob.set_title("LSTM Probability", color="#c9d1d9", fontsize=12, fontweight="bold")
        self.ax_prob.set_ylabel("P(earthquake)", color="#8b949e")
        self.ax_prob.set_ylim(-0.05, 1.05)
        self.ax_prob.axhline(y=THRESHOLD, color=COLOR_THRESH, linestyle="--", lw=1.5, alpha=0.8)

        self.fig.suptitle("LSTM Earthquake Detection", fontsize=16,
                          fontweight="bold", color="#f0f6fc")

    def update_frame(self, frame):
        with self.lock:
            if len(self.times) < 10:
                return
            t = np.array(self.times)
            x = np.array(self.data_x)
            y = np.array(self.data_y)
            z = np.array(self.data_z)
            mag = np.array(self.data_mag)
            prob = self.current_prob
            label = self.current_label
            peak = self.peak_accel
            det = self.detection_count
            n_samples = self.total_samples
            elapsed_s = time.time() - self.session_start
            ph = list(self.prob_history)
            pt = list(self.prob_times)

        t0 = t[0]
        tr = t - t0

        # Update XYZ lines (fast — no clear)
        self.line_x.set_data(tr, x)
        self.line_y.set_data(tr, y)
        self.line_z.set_data(tr, z)
        self.ax_xyz.set_xlim(tr[0], tr[-1])
        lo = min(x.min(), y.min(), z.min()) - 0.1
        hi = max(x.max(), y.max(), z.max()) + 0.1
        self.ax_xyz.set_ylim(lo, hi)

        # Update magnitude line
        self.line_mag.set_data(tr, mag)
        self.ax_mag.set_xlim(tr[0], tr[-1])
        self.ax_mag.set_ylim(max(0, mag.min() - 0.1), mag.max() + 0.1)

        # Update probability bars (must redraw — bar count changes)
        self.ax_prob.clear()
        self.ax_prob.set_facecolor("#161b22")
        self.ax_prob.grid(True, alpha=0.15, color="#30363d")
        self.ax_prob.tick_params(colors="#8b949e", labelsize=9)
        for spine in self.ax_prob.spines.values():
            spine.set_color("#30363d")

        if ph:
            pt_arr = np.array(pt) - t0
            colors = [COLOR_PROB_EQ if p > THRESHOLD else COLOR_PROB_OK for p in ph]
            self.ax_prob.bar(pt_arr, ph, width=0.35, color=colors, alpha=0.85,
                             edgecolor="none")
            self.ax_prob.set_xlim(tr[0], tr[-1])

        self.ax_prob.axhline(y=THRESHOLD, color=COLOR_THRESH, linestyle="--", lw=1.5, alpha=0.8)
        self.ax_prob.set_ylim(-0.05, 1.05)
        self.ax_prob.set_title("LSTM Probability", color="#c9d1d9", fontsize=12, fontweight="bold")
        self.ax_prob.set_ylabel("P(earthquake)", color="#8b949e")

        # Status bar
        self.ax_status.clear()
        self.ax_status.axis("off")

        if label == "EARTHQUAKE":
            self.ax_status.set_facecolor(BG_EARTHQUAKE)
            icon = "EARTHQUAKE"
            clr = COLOR_PROB_EQ
        else:
            self.ax_status.set_facecolor(BG_QUIET)
            icon = "QUIET"
            clr = COLOR_PROB_OK

        pga_str = f"{peak:.3f}" if peak < 10 else f"{peak:.1f}"

        self.ax_status.text(
            0.5, 0.5,
            f"  {icon}   |   P = {prob:.1%}   |   PGA = {pga_str} g   |   "
            f"Detections: {det}   |   Samples: {n_samples}   |   {elapsed_s:.0f}s  ",
            transform=self.ax_status.transAxes, fontsize=15,
            va="center", ha="center", fontfamily="monospace",
            fontweight="bold", color=clr,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d1117", edgecolor=clr, lw=2),
        )

    def run(self):
        print(f"Starting on {self.port} | duration {self.duration}s | Ctrl+C to stop")
        self.running = True

        t1 = threading.Thread(target=self.reader_thread, daemon=True)
        t2 = threading.Thread(target=self.inference_thread, daemon=True)
        t1.start()
        t2.start()

        self.setup_figure()

        ani = FuncAnimation(
            self.fig, self.update_frame,
            interval=250, blit=False, cache_frame_data=False,
        )

        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            self.running = False
            t1.join(timeout=3)
            t2.join(timeout=3)

        print(f"Total detections: {self.detection_count}")


def main():
    parser = argparse.ArgumentParser(description="Real-time LSTM earthquake visualizer")
    parser.add_argument("--port", type=str, required=True)
    parser.add_argument("--duration", type=int, default=600)
    args = parser.parse_args()
    LSTMVisualizer(port=args.port, duration=args.duration).run()


if __name__ == "__main__":
    main()
