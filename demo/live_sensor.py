"""
live_sensor.py
Continuously stream MPU6500 data, detect events, and save to CSV.
Runs until Ctrl+C.

Usage:
    python demo/live_sensor.py --port /dev/ttyACM0
    python demo/live_sensor.py --port /dev/ttyACM0 --save sensor_data.csv
"""

import argparse
import csv
import sys
import time
import numpy as np
from hardware.mpu6050_interface import MPU6050Reader
from detect_earthquake import load_config, preprocess, sta_lta
from detect_earthquake import detect_spikes, find_event_window, STA_LTA_THRESH


def main():
    parser = argparse.ArgumentParser(description="Live sensor streaming + detection")
    parser.add_argument("--port", default=None, help="Serial port (auto-detect if omitted)")
    parser.add_argument("--save", default=None, help="Save CSV to this file")
    parser.add_argument("--mode", default="table_knock", choices=["earthquake", "table_knock"])
    parser.add_argument("--rate", type=int, default=10, help="Print rate per second (default 10)")
    args = parser.parse_args()

    load_config(args.mode)

    reader = MPU6050Reader(port=args.port)
    reader.connect()

    csv_file = None
    csv_writer = None
    if args.save:
        csv_file = open(args.save, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp_s", "ax", "ay", "az"])

    # Ring buffer for detection (last 6 seconds at 100 Hz)
    buf_size = 600
    buf = []
    detection_count = 0
    last_print = 0
    print_interval = 1.0 / args.rate

    print(f"\nStreaming at ~100 Hz | Printing {args.rate}/s | Ctrl+C to stop\n")
    print(f"{'Time':>8s}  {'ax':>8s}  {'ay':>8s}  {'az':>8s}  {'|a|':>8s}  {'Status'}")
    print("-" * 60)

    try:
        while True:
            ts, ax, ay, az = reader.read_sample()
            mag = (ax**2 + ay**2 + az**2) ** 0.5
            buf.append((ts, ax, ay, az))

            if csv_writer:
                csv_writer.writerow([f"{ts:.3f}", f"{ax:.4f}", f"{ay:.4f}", f"{az:.4f}"])

            # Keep buffer at fixed size
            if len(buf) > buf_size:
                buf = buf[-buf_size:]

            # Print at requested rate
            now = time.time()
            if now - last_print >= print_interval:
                last_print = now

                # Run detection if buffer is full
                status = "collecting..."
                if len(buf) >= buf_size:
                    window = np.array(buf)
                    mags = np.sqrt(window[:, 1]**2 + window[:, 2]**2 + window[:, 3]**2)
                    filtered = preprocess(mags, 100)
                    ratio = sta_lta(filtered, 100)
                    spikes = detect_spikes(ratio, filtered, 100)
                    event = find_event_window(spikes, ratio, 100)

                    if event:
                        detection_count += 1
                        status = f"EVENT DETECTED! (#{detection_count})"
                    else:
                        status = "OK"

                print(f"{ts:8.2f}  {ax:8.4f}  {ay:8.4f}  {az:8.4f}  {mag:8.4f}  {status}")

    except KeyboardInterrupt:
        print(f"\n\nStopped. Total detections: {detection_count}")
        print(f"Total samples: {reader.sample_count}")
        print(f"Sample rate: {reader.get_sample_rate():.1f} Hz")
    finally:
        reader.disconnect()
        if csv_file:
            csv_file.close()
            print(f"Saved → {args.save}")


if __name__ == "__main__":
    main()
