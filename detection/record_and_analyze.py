"""
record_and_analyze.py
Record N seconds from MPU6050, save to CSV, then run detection.

Usage:
    python record_and_analyze.py --port /dev/ttyUSB0 --duration 30 --output knock_data.csv
    python record_and_analyze.py --duration 20  # auto-detect port
"""

import argparse
import sys
import time
import pandas as pd
import numpy as np

# Import hardware modules
from hardware.mpu6050_interface import MPU6050Reader
from detect_earthquake import load_config, load_data, preprocess, sta_lta
from detect_earthquake import detect_spikes, find_event_window, plot_results, print_report


def record_samples(port=None, duration=30):
    """
    Record accelerometer data for N seconds.

    Args:
        port: Serial port (auto-detect if None).
        duration: Recording duration in seconds.

    Returns:
        List of (timestamp_s, x_g, y_g, z_g) samples.
    """
    reader = MPU6050Reader(port=port)
    try:
        reader.connect()
        print(f"\nRecording for {duration} seconds... Press Ctrl+C to stop early.\n")

        samples = []
        start_time = time.time()

        while time.time() - start_time < duration:
            try:
                sample = reader.read_sample()
                samples.append(sample)

                # Progress indicator
                elapsed = time.time() - start_time
                bar_len = 40
                filled = int(bar_len * elapsed / duration)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(
                    f"\r[{bar}] {elapsed:.1f}s/{duration}s "
                    f"({len(samples)} samples @ {reader.get_sample_rate():.0f} Hz)",
                    end="",
                )
            except KeyboardInterrupt:
                print("\n\nRecording stopped by user.")
                break

        print(f"\n\nRecorded {len(samples)} samples in {elapsed:.1f} seconds\n")
        return samples

    finally:
        reader.disconnect()


def save_csv(samples, output_path):
    """
    Save samples to CSV file.

    Args:
        samples: List of (timestamp_s, x_g, y_g, z_g) tuples.
        output_path: Output file path.
    """
    df = pd.DataFrame(samples, columns=["timestamp", "x", "y", "z"])
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} samples → {output_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Record from MPU6050 and analyze for events."
    )
    parser.add_argument("--port", default=None, help="Serial port (auto-detect if not specified)")
    parser.add_argument("--duration", type=int, default=30, help="Recording duration in seconds")
    parser.add_argument(
        "--output", default="knock_recording.csv", help="Output CSV file path"
    )
    parser.add_argument(
        "--mode", default="table_knock", choices=["earthquake", "table_knock"],
        help="Detection mode (default: table_knock)"
    )

    args = parser.parse_args()

    try:
        # Record samples
        samples = record_samples(port=args.port, duration=args.duration)

        if not samples:
            print("No samples recorded. Exiting.")
            return

        # Save to CSV
        save_csv(samples, args.output)

        # Run detection
        print(f"=== Running {args.mode} detection ===\n")
        load_config(args.mode)

        t_s, mag, fs, timestamps = load_data(args.output)
        filtered = preprocess(mag, fs)
        ratio = sta_lta(filtered, fs)
        spikes = detect_spikes(ratio, filtered, fs)
        print(f"Found {len(spikes)} spike cluster(s).")

        window = find_event_window(spikes, ratio, fs)

        print("\n[6/7] Generating diagram...")
        plot_results(t_s, mag, filtered, ratio, spikes, window, timestamps)

        print("[7/7] Report:")
        print_report(window, t_s, timestamps)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
