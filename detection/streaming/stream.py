"""
stream.py — All-in-one: read Arduino, detect earthquakes, post to blackboard, visualize.

One program, one serial port. Everything runs in parallel threads.

Usage:
    python detection/streaming/stream.py                              # stream only
    python detection/streaming/stream.py --detect                     # + STA/LTA detection
    python detection/streaming/stream.py --detect --blackboard        # + post to blackboard
    python detection/streaming/stream.py --detect --blackboard --viz  # + live graphs
    python detection/streaming/stream.py --detect --blackboard --save data.csv  # + CSV
"""

import argparse
import csv
import json
import sys
import os
import time
import glob
import threading
import serial
import requests
import numpy as np
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("MPLBACKEND", "TkAgg")

from detect_earthquake import load_config, preprocess, sta_lta, detect_spikes, find_event_window, CONFIG_PROFILES

if "optimized" not in CONFIG_PROFILES:
    CONFIG_PROFILES["optimized"] = {
        "BP_LOW_HZ": 1.5, "BP_HIGH_HZ": 22.0, "STA_WINDOW_S": 0.3,
        "LTA_WINDOW_S": 7.0, "STA_LTA_THRESH": 5.0, "AMP_SIGMA_THRESH": 5.0,
        "MERGE_GAP_S": 2.0, "QUIET_GUARD_S": 4.0,
    }

BLACKBOARD_URL = "https://blackboard.jass.school/blackboard"
AGENT_NAME = "EarthQuakeSensorProvider"


def auto_detect_port():
    candidates = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    if candidates:
        return candidates[0]
    raise RuntimeError("No serial port found. Is the Arduino plugged in?")


# ── Serial reader thread ─────────────────────────────────────────────────────

def serial_reader(ser, queue, running):
    """Reads serial non-stop, puts (ax, ay, az, gx, gy, gz, ts_ms) into queue."""
    while running[0]:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line or line.startswith("timestamp") or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            ts = float(parts[0])
            ax, ay, az = float(parts[1]), float(parts[2]), float(parts[3])
            gx = float(parts[4]) if len(parts) >= 7 else 0.0
            gy = float(parts[5]) if len(parts) >= 7 else 0.0
            gz = float(parts[6]) if len(parts) >= 7 else 0.0
            queue.append((ts, ax, ay, az, gx, gy, gz))
        except Exception:
            continue


# ── Detection thread ─────────────────────────────────────────────────────────

def detection_thread(det_buf, det_lock, det_state):
    """Runs STA/LTA detection every 0.5s on rolling buffer."""
    while det_state["running"]:
        time.sleep(0.5)
        with det_lock:
            if len(det_buf) < 300:
                continue
            window = np.array(list(det_buf)[-600:])

        mags = np.sqrt(window[:, 0]**2 + window[:, 1]**2 + window[:, 2]**2)
        try:
            filtered = preprocess(mags, 100)
            ratio = sta_lta(filtered, 100)
            spikes = detect_spikes(ratio, filtered, 100)
            event = find_event_window(spikes, ratio, 100)
        except Exception:
            continue

        if event and event != det_state.get("last_event"):
            det_state["last_event"] = event
            det_state["count"] += 1
            det_state["alert"] = f"EVENT #{det_state['count']}"
        elif not event:
            det_state["alert"] = None


# ── Blackboard poster ────────────────────────────────────────────────────────

def post_blackboard(ax, ay, az, mag, amp, label, det_count):
    """Post one reading to blackboard (runs in daemon thread)."""
    try:
        requests.post(BLACKBOARD_URL, json={
            "agent": AGENT_NAME,
            "type": "earthquake_sensor",
            "content": json.dumps({
                "timestamp": time.time(),
                "ax": round(ax, 4), "ay": round(ay, 4), "az": round(az, 4),
                "magnitude_g": round(mag, 4), "amplitude_g": round(amp, 4),
                "label": label, "detections": det_count,
            }),
            "confidence": 0.0 if label == "quiet" else 0.9,
        }, timeout=3)
    except Exception:
        pass


# ── Visualization ────────────────────────────────────────────────────────────

def start_visualizer(viz_buf, viz_lock, running, mode, duration):
    """Runs matplotlib visualizer in a thread, reading from shared buffer."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    load_config(mode)
    config = CONFIG_PROFILES[mode]

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Real-Time Seismic Monitor ({mode})", fontsize=14, fontweight="bold")

    # 2x2 grid: 3D + 3 plots
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_xyz = fig.add_subplot(2, 2, 2)
    ax_mag = fig.add_subplot(2, 2, 3)
    ax_ratio = fig.add_subplot(2, 2, 4)
    axes = [ax_xyz, ax_mag, ax_ratio]

    def update(frame):
        with viz_lock:
            if len(viz_buf) < 100:
                return
            data = np.array(list(viz_buf))

        t_s = data[:, 0] / 1000.0
        t_s = t_s - t_s[0]
        x, y, z = data[:, 1], data[:, 2], data[:, 3]
        mag = np.sqrt(x**2 + y**2 + z**2)

        try:
            filtered = preprocess(mag, 100)
            ratio = sta_lta(filtered, 100)
            spikes = detect_spikes(ratio, filtered, 100)
            event = find_event_window(spikes, ratio, 100)
        except Exception:
            filtered = np.zeros_like(mag)
            ratio = np.zeros_like(mag)
            event = None

        # ===== 3D acceleration vector =====
        ax3d.clear()
        # Trajectory (last 200 points)
        trail = min(200, len(x))
        ax3d.plot(x[-trail:], y[-trail:], z[-trail:], "b-", alpha=0.3, lw=0.5)
        # Current position
        ax3d.scatter([x[-1]], [y[-1]], [z[-1]], color="red", s=80, zorder=5)
        # Origin marker
        ax3d.scatter([0], [0], [0], color="gray", s=20, alpha=0.5)
        ax3d.set_xlabel("X (g)")
        ax3d.set_ylabel("Y (g)")
        ax3d.set_zlabel("Z (g)")
        ax3d.set_xlim(-1.5, 1.5)
        ax3d.set_ylim(-1.5, 1.5)
        ax3d.set_zlim(-1.5, 0.5)
        ax3d.set_title("3D Acceleration Vector")

        for a in axes:
            a.clear()

        # ===== XYZ waveforms =====
        ax_xyz.plot(t_s, x, "r-", lw=0.8, label="X")
        ax_xyz.plot(t_s, y, "g-", lw=0.8, label="Y")
        ax_xyz.plot(t_s, z, "b-", lw=0.8, label="Z")
        if event:
            s, e = event
            if s < len(t_s) and e < len(t_s):
                ax_xyz.axvspan(t_s[s], t_s[e], color="red", alpha=0.15)
        ax_xyz.set_ylabel("Accel (g)")
        ax_xyz.legend(loc="upper right", fontsize=8)
        ax_xyz.grid(True, alpha=0.3)

        # ===== Magnitude =====
        ax_mag.plot(t_s, mag, "k-", lw=1)
        if event:
            s, e = event
            if s < len(t_s) and e < len(t_s):
                ax_mag.axvspan(t_s[s], t_s[e], color="red", alpha=0.15)
        ax_mag.set_ylabel("Magnitude (g)")
        ax_mag.set_xlabel("Time (s)")
        ax_mag.grid(True, alpha=0.3)

        # ===== STA/LTA =====
        ax_ratio.plot(t_s, ratio, "purple", lw=1)
        ax_ratio.axhline(config["STA_LTA_THRESH"], color="red", ls="--", lw=1.5)
        ax_ratio.fill_between(t_s, 0, ratio, alpha=0.15, color="purple")
        if event:
            s, e = event
            if s < len(t_s) and e < len(t_s):
                ax_ratio.axvspan(t_s[s], t_s[e], color="red", alpha=0.15)
        ax_ratio.set_ylabel("STA/LTA")
        ax_ratio.set_xlabel("Time (s)")
        ax_ratio.grid(True, alpha=0.3)
        axes[2].grid(True, alpha=0.3)

    ani = FuncAnimation(fig, update, interval=300, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    running[0] = False  # signal other threads to stop when window closed


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sensor streaming + detection + blackboard + visualization")
    parser.add_argument("--port", default=None)
    parser.add_argument("--save", default=None, metavar="FILE")
    parser.add_argument("--rate", type=int, default=5, help="Print rate per second")
    parser.add_argument("--detect", action="store_true", help="Enable STA/LTA detection")
    parser.add_argument("--blackboard", action="store_true", help="Post data to ORB blackboard")
    parser.add_argument("--viz", action="store_true", help="Show live graphs (disable with no flag)")
    parser.add_argument("--mode", default="table_knock", choices=["earthquake", "table_knock", "optimized"])
    parser.add_argument("--duration", type=int, default=600, help="Max duration in seconds")
    args = parser.parse_args()

    port = args.port or auto_detect_port()

    # Connect
    print(f"Connecting to {port}...")
    ser = serial.Serial(port, 115200, timeout=1)
    time.sleep(1)
    ser.reset_input_buffer()
    ser.read(ser.in_waiting or 1)

    print(f"Connected.\n")
    print(f"  Detect:     {'ON' if args.detect else 'OFF'}")
    print(f"  Blackboard: {'ON → ' + BLACKBOARD_URL if args.blackboard else 'OFF'}")
    print(f"  Visualize:  {'ON' if args.viz else 'OFF'}")
    print(f"  Save:       {args.save or 'OFF'}")
    print(f"  Rate:       {args.rate}/s")
    print()

    # Shared state
    sample_queue = deque(maxlen=5000)
    running = [True]

    # Buffers for detection and visualization
    det_buf = deque(maxlen=600)
    det_lock = threading.Lock()
    viz_buf = deque(maxlen=1000)
    viz_lock = threading.Lock()

    det_state = {"running": True, "last_event": None, "count": 0, "alert": None}

    # Start serial reader thread
    threading.Thread(target=serial_reader, args=(ser, sample_queue, running), daemon=True).start()

    # Calibrate gravity
    print("Calibrating...")
    while len(sample_queue) < 20:
        time.sleep(0.05)
    cal = [(s[1], s[2], s[3]) for s in list(sample_queue)[:20]]
    gravity = np.mean(cal, axis=0)
    sample_queue.clear()
    print(f"Gravity: [{gravity[0]:.4f}, {gravity[1]:.4f}, {gravity[2]:.4f}]\n")

    # Start detection thread
    if args.detect:
        load_config(args.mode)
        threading.Thread(target=detection_thread, args=(det_buf, det_lock, det_state), daemon=True).start()

    # Start visualizer (runs in main-ish thread via matplotlib)
    if args.viz:
        # Visualization needs to run on the main thread for TkAgg.
        # So we move the data processing loop to a thread instead.
        viz_thread_started = True
    else:
        viz_thread_started = False

    # CSV
    csv_file = None
    csv_writer = None
    if args.save:
        csv_file = open(args.save, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp_s", "ax", "ay", "az", "gx", "gy", "gz"])

    print_interval = 1.0 / args.rate
    last_print = 0
    last_bb = 0
    sample_count = 0
    start_time = time.time()

    if args.detect:
        print(f"{'Time':>8}  {'ax':>8}  {'ay':>8}  {'az':>8}  {'|a|':>8}  Status")
    else:
        print(f"{'Time':>8}  {'ax':>8}  {'ay':>8}  {'az':>8}  {'|a|':>8}")
    print("=" * 58)

    def data_loop():
        nonlocal sample_count, last_print, last_bb
        ax = ay = az = mag = amp = 0.0
        try:
            while running[0] and (time.time() - start_time) < args.duration:
                # Drain samples
                drained = 0
                while sample_queue:
                    ts_ms, ax, ay, az, gx, gy, gz = sample_queue.popleft()
                    sample_count += 1
                    drained += 1

                    if csv_writer:
                        csv_writer.writerow([f"{ts_ms/1000:.3f}", ax, ay, az, gx, gy, gz])

                    if args.detect:
                        with det_lock:
                            det_buf.append((ax, ay, az))

                    if args.viz:
                        with viz_lock:
                            viz_buf.append((ts_ms, ax, ay, az))

                if drained == 0:
                    time.sleep(0.01)
                    continue

                mag = (ax**2 + ay**2 + az**2) ** 0.5
                amp = ((ax - gravity[0])**2 + (ay - gravity[1])**2 + (az - gravity[2])**2) ** 0.5

                now = time.time()
                if now - last_print >= print_interval:
                    last_print = now
                    ts = sample_count / 100.0
                    if args.detect:
                        alert = det_state["alert"]
                        if alert:
                            print(f"{ts:8.1f}  {ax:8.4f}  {ay:8.4f}  {az:8.4f}  {mag:8.4f}  <<< {alert} >>>")
                        else:
                            print(f"{ts:8.1f}  {ax:8.4f}  {ay:8.4f}  {az:8.4f}  {mag:8.4f}  OK")
                    else:
                        print(f"{ts:8.1f}  {ax:8.4f}  {ay:8.4f}  {az:8.4f}  {mag:8.4f}")

                # Blackboard 1/sec
                if args.blackboard and (now - last_bb) >= 1.0:
                    last_bb = now
                    label = "EARTHQUAKE" if det_state["alert"] else "quiet"
                    threading.Thread(
                        target=post_blackboard,
                        args=(ax, ay, az, mag, amp, label, det_state["count"]),
                        daemon=True
                    ).start()

        except KeyboardInterrupt:
            pass
        finally:
            running[0] = False
            det_state["running"] = False

    if args.viz:
        # Data loop runs in background, matplotlib on main thread
        threading.Thread(target=data_loop, daemon=True).start()
        start_visualizer(viz_buf, viz_lock, running, args.mode, args.duration)
    else:
        # No viz — data loop runs on main thread
        try:
            data_loop()
        except KeyboardInterrupt:
            pass

    # Cleanup
    running[0] = False
    det_state["running"] = False
    ser.close()
    if csv_file:
        csv_file.close()
        print(f"Saved → {args.save}")

    elapsed = time.time() - start_time
    print(f"\nDone. {sample_count} samples in {elapsed:.1f}s ({sample_count/max(elapsed,1):.0f} Hz)")
    if args.detect:
        print(f"Events: {det_state['count']}")


if __name__ == "__main__":
    main()
