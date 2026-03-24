"""
stream.py — Stream MPU6500 sensor data with real-time earthquake detection + LLM analysis.

Reads accelerometer data from Arduino, runs STA/LTA detection on a rolling buffer,
and when an event is detected, calls the Hugging Face LLM to generate a geological
hypothesis. Results are printed to terminal and optionally saved to CSV.

Usage:
    python detection/streaming/stream.py                          # stream only
    python detection/streaming/stream.py --save data.csv          # stream + save
    python detection/streaming/stream.py --detect                 # stream + detection
    python detection/streaming/stream.py --detect --llm           # stream + detection + LLM
    python detection/streaming/stream.py --detect --llm --save data.csv  # everything
"""

import argparse
import csv
import sys
import os
import time
import glob
import threading
import serial
import numpy as np

# Add detection/ to path so detect_earthquake is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from detect_earthquake import load_config, preprocess, sta_lta, detect_spikes, find_event_window

# LLM imports (optional)
LLM_AVAILABLE = False
try:
    from huggingface_hub import InferenceClient
    from dotenv import load_dotenv
    load_dotenv()
    LLM_AVAILABLE = bool(os.getenv("HF_TOKEN"))
except ImportError:
    pass


def auto_detect_port():
    candidates = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    if candidates:
        return candidates[0]
    raise RuntimeError("No serial port found. Is the Arduino plugged in?")


def call_llm(event_stats: dict) -> str:
    """Call Hugging Face LLM to analyze detected event."""
    token = os.getenv("HF_TOKEN")
    if not token:
        return "[LLM skipped: HF_TOKEN not set]"

    try:
        client = InferenceClient(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            token=token
        )

        prompt = f"""Analyze this seismic sensor reading from an MPU6500 accelerometer:

- Peak acceleration: {event_stats['peak_g']:.3f} g
- Duration of event: {event_stats['duration_s']:.1f} seconds
- Mean magnitude during event: {event_stats['mean_mag']:.4f} g
- Baseline magnitude (quiet): {event_stats['baseline_mag']:.4f} g
- STA/LTA peak ratio: {event_stats['peak_ratio']:.1f}
- Signal-to-noise ratio: {event_stats['snr']:.1f}

In 2-3 sentences: Is this likely an earthquake, a table knock, or sensor noise?
Estimate the approximate Richter magnitude if seismic. Be concise."""

        response = client.chat_completion(
            messages=[
                {"role": "system", "content": "You are a seismologist analyzing real-time accelerometer data from a field sensor."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM error: {e}]"


def compute_event_stats(window: np.ndarray, event_start: int, event_end: int) -> dict:
    """Compute statistics for a detected event window."""
    mags = np.sqrt(window[:, 0]**2 + window[:, 1]**2 + window[:, 2]**2)

    # Baseline: first 100 samples (before event)
    baseline = mags[:min(100, event_start)] if event_start > 10 else mags[:10]
    event_data = mags[event_start:event_end+1]

    return {
        "peak_g": float(np.max(np.abs(window[event_start:event_end+1]))),
        "duration_s": (event_end - event_start) / 100.0,
        "mean_mag": float(np.mean(event_data)),
        "baseline_mag": float(np.mean(baseline)),
        "peak_ratio": float(np.max(event_data) / (np.mean(baseline) + 1e-9)),
        "snr": float(np.std(event_data) / (np.std(baseline) + 1e-9)),
    }


def run_detection(buf, lock, result, use_llm):
    """Background thread: STA/LTA detection + optional LLM analysis."""
    while result["running"]:
        time.sleep(0.5)
        with lock:
            if len(buf) < 300:
                continue
            window = np.array(buf[-600:])

        mags = np.sqrt(window[:, 0]**2 + window[:, 1]**2 + window[:, 2]**2)
        try:
            filtered = preprocess(mags, 100)
            ratio = sta_lta(filtered, 100)
            spikes = detect_spikes(ratio, filtered, 100)
            event = find_event_window(spikes, ratio, 100)
        except Exception:
            continue

        if event and event != result.get("last_event"):
            result["last_event"] = event
            result["count"] += 1
            stats = compute_event_stats(window, event[0], event[1])
            result["alert"] = f"EVENT #{result['count']}"
            result["stats"] = stats

            # Call LLM in background (don't block detection)
            if use_llm:
                result["llm_status"] = "analyzing..."
                try:
                    hypothesis = call_llm(stats)
                    result["llm_result"] = hypothesis
                    result["llm_status"] = "done"
                except Exception as e:
                    result["llm_result"] = f"[error: {e}]"
                    result["llm_status"] = "error"
        elif not event:
            result["alert"] = None


def main():
    parser = argparse.ArgumentParser(description="Arduino sensor streaming + earthquake detection")
    parser.add_argument("--port", default=None, help="Serial port (auto-detect if omitted)")
    parser.add_argument("--save", default=None, metavar="FILE", help="Save CSV to file")
    parser.add_argument("--rate", type=int, default=10, help="Print rate per second (default 10)")
    parser.add_argument("--detect", action="store_true", help="Enable real-time STA/LTA detection")
    parser.add_argument("--llm", action="store_true", help="Enable LLM analysis on detected events")
    parser.add_argument("--mode", default="table_knock", choices=["earthquake", "table_knock"])
    parser.add_argument("--pipe", action="store_true", help="Output raw CSV to stdout (for piping to Docker)")
    args = parser.parse_args()

    port = args.port or auto_detect_port()

    if args.llm and not LLM_AVAILABLE:
        print("WARNING: --llm requested but HF_TOKEN not set or huggingface_hub not installed.", file=sys.stderr)

    # Connect
    print(f"Connecting to {port}...", file=sys.stderr)
    ser = serial.Serial(port, 115200, timeout=2)
    time.sleep(1.5)
    ser.reset_input_buffer()
    for _ in range(5):
        ser.readline()
    print(f"Connected.", file=sys.stderr)

    # Pipe mode: just output raw CSV to stdout and exit early
    if args.pipe:
        print("timestamp_ms,ax,ay,az,gx,gy,gz", flush=True)
        try:
            while True:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line or line.startswith("timestamp") or line.startswith("#"):
                    continue
                print(line, flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            ser.close()
        return

    print()

    # CSV output
    csv_file = None
    writer = None
    if args.save:
        csv_file = open(args.save, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["timestamp_s", "ax", "ay", "az", "gx", "gy", "gz"])

    # Detection state
    detect_result = {
        "running": True, "last_event": None, "count": 0,
        "alert": None, "stats": None,
        "llm_status": None, "llm_result": None
    }
    buf = []
    buf_lock = threading.Lock()

    if args.detect:
        load_config(args.mode)
        t = threading.Thread(
            target=run_detection,
            args=(buf, buf_lock, detect_result, args.llm),
            daemon=True
        )
        t.start()

    print_interval = 1.0 / args.rate
    last_print = 0
    last_llm_shown = 0
    sample_count = 0
    start = time.time()

    flags = []
    if args.detect:
        flags.append(f"Detection: {args.mode}")
    if args.llm:
        flags.append("LLM: on")
    flag_str = f" | {' | '.join(flags)}" if flags else ""

    print(f"Streaming 100 Hz | Print {args.rate}/s{flag_str} | Ctrl+C to stop\n")
    print(f"{'Time':>8}  {'ax':>8}  {'ay':>8}  {'az':>8}  {'|a|':>8}  {'Status'}")
    print("=" * 65)

    try:
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line or line.startswith("timestamp") or line.startswith("#"):
                continue

            parts = line.split(",")
            if len(parts) not in (4, 7):
                continue

            try:
                ts = float(parts[0]) / 1000.0
                ax, ay, az = float(parts[1]), float(parts[2]), float(parts[3])
                gx = float(parts[4]) if len(parts) == 7 else 0.0
                gy = float(parts[5]) if len(parts) == 7 else 0.0
                gz = float(parts[6]) if len(parts) == 7 else 0.0
            except ValueError:
                continue

            mag = (ax**2 + ay**2 + az**2) ** 0.5
            sample_count += 1

            if writer:
                writer.writerow([f"{ts:.3f}", ax, ay, az, gx, gy, gz])

            if args.detect:
                with buf_lock:
                    buf.append((ax, ay, az))

            now = time.time()
            if now - last_print >= print_interval:
                last_print = now

                if args.detect:
                    alert = detect_result["alert"]
                    if alert:
                        print(f"{ts:8.2f}  {ax:8.4f}  {ay:8.4f}  {az:8.4f}  {mag:8.4f}  <<< {alert} >>>")
                    else:
                        print(f"{ts:8.2f}  {ax:8.4f}  {ay:8.4f}  {az:8.4f}  {mag:8.4f}  OK")

                    # Show LLM result once when available
                    if (detect_result["llm_result"]
                            and detect_result["count"] > last_llm_shown):
                        last_llm_shown = detect_result["count"]
                        stats = detect_result["stats"]
                        print()
                        print("=" * 65)
                        print(f"  LLM ANALYSIS (Event #{detect_result['count']})")
                        print("-" * 65)
                        if stats:
                            print(f"  Peak: {stats['peak_g']:.3f}g | "
                                  f"Duration: {stats['duration_s']:.1f}s | "
                                  f"SNR: {stats['snr']:.1f}")
                        print()
                        print(f"  {detect_result['llm_result']}")
                        print("=" * 65)
                        print()
                        detect_result["llm_result"] = None
                else:
                    print(f"{ts:8.2f}  {ax:8.4f}  {ay:8.4f}  {az:8.4f}  {mag:8.4f}")

    except KeyboardInterrupt:
        detect_result["running"] = False
        elapsed = time.time() - start
        print(f"\nStopped | {elapsed:.1f}s | {sample_count} samples | {sample_count/elapsed:.1f} Hz")
        if args.detect:
            print(f"Events detected: {detect_result['count']}")
    finally:
        ser.close()
        if csv_file:
            csv_file.close()
            print(f"Saved → {args.save}")


if __name__ == "__main__":
    main()
