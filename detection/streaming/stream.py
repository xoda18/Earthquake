"""
stream.py — All-in-one: read Arduino, classify with LSTM, post to blackboard, visualize.

LSTM runs as a subprocess via Python 3.11 (TensorFlow).
Everything else runs in Python 3.14 with parallel threads.

Usage:
    python3 detection/streaming/stream.py --detect --blackboard --viz
    python3 detection/streaming/stream.py --detect --viz --threshold 0.7
    python3 detection/streaming/stream.py --detect --blackboard --save data.csv
"""

import argparse
import csv
import json
import sys
import os
import time
import glob
import threading
import subprocess
import serial
import requests
import numpy as np
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("MPLBACKEND", "TkAgg")

BLACKBOARD_URL = "https://blackboard.jass.school/blackboard"
AGENT_NAME = "EarthQuakeSensorProvider"
PYTHON311 = "/home/vector/python3.11/bin/python3.11"
WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "lstm_worker.py")


def auto_detect_port():
    candidates = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    if candidates:
        return candidates[0]
    raise RuntimeError("No serial port found. Is the Arduino plugged in?")


# ── Serial reader thread ─────────────────────────────────────────────────────

def serial_reader(ser, queue, running):
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


# ── LSTM detection thread ────────────────────────────────────────────────────

def lstm_detection_thread(det_buf, det_lock, det_state, threshold):
    """Spawns lstm_worker.py as subprocess, sends windows, reads predictions."""
    try:
        proc = subprocess.Popen(
            [PYTHON311, WORKER_SCRIPT],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1
        )
    except FileNotFoundError:
        print(f"ERROR: {PYTHON311} not found. Cannot run LSTM.", file=sys.stderr)
        return

    # Wait for ready
    ready = proc.stdout.readline()
    if "ready" not in ready:
        print(f"LSTM worker failed to start: {ready}", file=sys.stderr)
        return
    print("LSTM model loaded.", flush=True)

    cooldown_until = 0

    while det_state["running"]:
        time.sleep(0.5)
        with det_lock:
            if len(det_buf) < 100:
                continue
            window = list(det_buf)[-100:]  # last 100 samples, each (ax,ay,az)

        # Send window to LSTM worker
        try:
            msg = json.dumps({"window": window}) + "\n"
            proc.stdin.write(msg)
            proc.stdin.flush()
            response = proc.stdout.readline()
            result = json.loads(response)
        except Exception:
            continue

        prob = result.get("probability", 0)
        label = result.get("label", "quiet")
        det_state["probability"] = prob
        det_state["label"] = label

        now = time.time()
        if label == "EARTHQUAKE" and prob > threshold:
            if now > cooldown_until:
                det_state["count"] += 1
                cooldown_until = now + det_state.get("cooldown", 5)
            det_state["alert"] = f"EARTHQUAKE #{det_state['count']} ({prob:.0%})"
        else:
            det_state["alert"] = None

    proc.terminate()


# ── Blackboard poster ────────────────────────────────────────────────────────

def post_blackboard(ax, ay, az, mag, amp, label, prob, det_count):
    try:
        requests.post(BLACKBOARD_URL, json={
            "agent": AGENT_NAME,
            "type": "earthquake_sensor",
            "content": json.dumps({
                "timestamp": time.time(),
                "ax": round(ax, 4), "ay": round(ay, 4), "az": round(az, 4),
                "magnitude_g": round(mag, 4), "amplitude_g": round(amp, 4),
                "probability": round(prob, 3), "label": label,
                "detections": det_count,
            }),
            "confidence": round(prob, 3),
        }, timeout=3)
    except Exception:
        pass


# ── Visualization ────────────────────────────────────────────────────────────

def start_visualizer(viz_buf, viz_lock, det_state, running, duration):
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Real-Time Seismic Monitor (LSTM)", fontsize=14, fontweight="bold")

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_xyz = fig.add_subplot(2, 2, 2)
    ax_mag = fig.add_subplot(2, 2, 3)
    ax_status = fig.add_subplot(2, 2, 4)

    def update(frame):
        with viz_lock:
            if len(viz_buf) < 50:
                return
            data = np.array(list(viz_buf))

        t_s = data[:, 0] / 1000.0
        t_s = t_s - t_s[0]
        x, y, z = data[:, 1], data[:, 2], data[:, 3]
        mag = np.sqrt(x**2 + y**2 + z**2)

        # 3D
        ax3d.clear()
        trail = min(200, len(x))
        ax3d.plot(x[-trail:], y[-trail:], z[-trail:], "b-", alpha=0.3, lw=0.5)
        ax3d.scatter([x[-1]], [y[-1]], [z[-1]], color="red", s=80, zorder=5)
        ax3d.set_xlabel("X (g)")
        ax3d.set_ylabel("Y (g)")
        ax3d.set_zlabel("Z (g)")
        ax3d.set_xlim(-1.5, 1.5)
        ax3d.set_ylim(-1.5, 1.5)
        ax3d.set_zlim(-1.5, 0.5)
        ax3d.set_title("3D Acceleration")

        # XYZ
        ax_xyz.clear()
        ax_xyz.plot(t_s, x, "r-", lw=0.8, label="X")
        ax_xyz.plot(t_s, y, "g-", lw=0.8, label="Y")
        ax_xyz.plot(t_s, z, "b-", lw=0.8, label="Z")
        ax_xyz.set_ylabel("Accel (g)")
        ax_xyz.legend(loc="upper right", fontsize=8)
        ax_xyz.grid(True, alpha=0.3)

        # Magnitude
        ax_mag.clear()
        ax_mag.plot(t_s, mag, "k-", lw=1)
        ax_mag.set_ylabel("Magnitude (g)")
        ax_mag.set_xlabel("Time (s)")
        ax_mag.grid(True, alpha=0.3)

        # Status panel
        ax_status.clear()
        ax_status.axis("off")
        prob = det_state.get("probability", 0)
        label = det_state.get("label", "?")
        alert = det_state.get("alert")
        count = det_state.get("count", 0)

        color = "#ff4444" if alert else "#44aa44"
        status_str = alert or "quiet"

        text = (
            f"LSTM Classification\n"
            f"{'='*30}\n\n"
            f"Status:      {status_str}\n"
            f"Probability: {prob:.1%}\n"
            f"Detections:  {count}\n\n"
            f"Current:\n"
            f"  X: {x[-1]:+.4f} g\n"
            f"  Y: {y[-1]:+.4f} g\n"
            f"  Z: {z[-1]:+.4f} g\n"
            f"  |a|: {mag[-1]:.4f} g"
        )
        ax_status.text(0.05, 0.95, text, transform=ax_status.transAxes,
                       fontsize=11, va="top", fontfamily="monospace",
                       bbox=dict(boxstyle="round", facecolor=color, alpha=0.2))

    ani = FuncAnimation(fig, update, interval=300, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    running[0] = False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sensor streaming + LSTM detection + blackboard + visualization")
    parser.add_argument("--port", default=None)
    parser.add_argument("--save", default=None, metavar="FILE")
    parser.add_argument("--rate", type=int, default=5, help="Print rate per second")
    parser.add_argument("--detect", action="store_true", help="Enable LSTM earthquake detection")
    parser.add_argument("--blackboard", action="store_true", help="Post data to ORB blackboard")
    parser.add_argument("--viz", action="store_true", help="Show live graphs")
    parser.add_argument("--threshold", type=float, default=0.7, help="LSTM probability threshold (default 0.7)")
    parser.add_argument("--cooldown", type=float, default=5.0, help="Seconds between detections (default 5)")
    parser.add_argument("--duration", type=int, default=600, help="Max duration in seconds")
    args = parser.parse_args()

    port = args.port or auto_detect_port()

    print(f"Connecting to {port}...")
    ser = serial.Serial(port, 115200, timeout=1)
    time.sleep(1)
    ser.reset_input_buffer()
    ser.read(ser.in_waiting or 1)

    print(f"Connected.\n")
    print(f"  Detect:     {'LSTM' if args.detect else 'OFF'}")
    print(f"  Threshold:  {args.threshold}")
    print(f"  Cooldown:   {args.cooldown}s")
    print(f"  Blackboard: {'ON' if args.blackboard else 'OFF'}")
    print(f"  Visualize:  {'ON' if args.viz else 'OFF'}")
    print(f"  Save:       {args.save or 'OFF'}")
    print()

    # Shared state
    sample_queue = deque(maxlen=5000)
    running = [True]

    det_buf = deque(maxlen=200)
    det_lock = threading.Lock()
    viz_buf = deque(maxlen=1000)
    viz_lock = threading.Lock()

    det_state = {
        "running": True, "count": 0, "alert": None,
        "probability": 0, "label": "quiet", "cooldown": args.cooldown
    }

    # Start serial reader
    threading.Thread(target=serial_reader, args=(ser, sample_queue, running), daemon=True).start()

    # Calibrate gravity
    print("Calibrating...")
    while len(sample_queue) < 20:
        time.sleep(0.05)
    cal = [(s[1], s[2], s[3]) for s in list(sample_queue)[:20]]
    gravity = np.mean(cal, axis=0)
    sample_queue.clear()
    print(f"Gravity: [{gravity[0]:.4f}, {gravity[1]:.4f}, {gravity[2]:.4f}]\n")

    # Start LSTM detection
    if args.detect:
        threading.Thread(
            target=lstm_detection_thread,
            args=(det_buf, det_lock, det_state, args.threshold),
            daemon=True
        ).start()

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

    header = f"{'Time':>8}  {'ax':>8}  {'ay':>8}  {'az':>8}  {'|a|':>8}"
    if args.detect:
        header += f"  {'Prob':>6}  Status"
    print(header)
    print("=" * (len(header) + 5))

    def data_loop():
        nonlocal sample_count, last_print, last_bb
        ax = ay = az = mag = amp = 0.0
        try:
            while running[0] and (time.time() - start_time) < args.duration:
                drained = 0
                while sample_queue:
                    ts_ms, ax, ay, az, gx, gy, gz = sample_queue.popleft()
                    sample_count += 1
                    drained += 1

                    if csv_writer:
                        csv_writer.writerow([f"{ts_ms/1000:.3f}", ax, ay, az, gx, gy, gz])

                    if args.detect:
                        with det_lock:
                            det_buf.append((ax - gravity[0], ay - gravity[1], az - gravity[2]))

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
                    prob = det_state.get("probability", 0)
                    alert = det_state.get("alert")

                    if args.detect:
                        if alert:
                            print(f"{ts:8.1f}  {ax:8.4f}  {ay:8.4f}  {az:8.4f}  {mag:8.4f}  {prob:6.1%}  <<< {alert} >>>")
                        else:
                            print(f"{ts:8.1f}  {ax:8.4f}  {ay:8.4f}  {az:8.4f}  {mag:8.4f}  {prob:6.1%}  quiet")
                    else:
                        print(f"{ts:8.1f}  {ax:8.4f}  {ay:8.4f}  {az:8.4f}  {mag:8.4f}")

                # Blackboard 1/sec
                if args.blackboard and (now - last_bb) >= 1.0:
                    last_bb = now
                    label = det_state.get("label", "quiet")
                    prob = det_state.get("probability", 0)
                    threading.Thread(
                        target=post_blackboard,
                        args=(ax, ay, az, mag, amp, label, prob, det_state["count"]),
                        daemon=True
                    ).start()

        except KeyboardInterrupt:
            pass
        finally:
            running[0] = False
            det_state["running"] = False

    if args.viz:
        threading.Thread(target=data_loop, daemon=True).start()
        start_visualizer(viz_buf, viz_lock, det_state, running, args.duration)
    else:
        try:
            data_loop()
        except KeyboardInterrupt:
            pass

    running[0] = False
    det_state["running"] = False
    ser.close()
    if csv_file:
        csv_file.close()
        print(f"Saved -> {args.save}")

    elapsed = time.time() - start_time
    print(f"\nDone. {sample_count} samples in {elapsed:.1f}s ({sample_count/max(elapsed,1):.0f} Hz)")
    if args.detect:
        print(f"Earthquakes: {det_state['count']}")


if __name__ == "__main__":
    main()
