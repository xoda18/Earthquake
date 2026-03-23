"""
mpu6050_interface.py
Serial communication with MPU6050 accelerometer via Arduino/Raspberry Pi.

Expected format (CSV over serial):
    timestamp_ms,x_g,y_g,z_g
    1234567,0.01,0.02,9.81
    1234568,0.01,0.01,9.82
"""

import serial
import glob
import time
from typing import Tuple, List, Optional


class MPU6050Reader:
    """Interface to read 3-axis acceleration from MPU6050 via serial."""

    def __init__(self, port: Optional[str] = None, baudrate: int = 115200):
        """
        Initialize MPU6050 reader.

        Args:
            port: Serial port (e.g., '/dev/ttyUSB0'). If None, auto-detect.
            baudrate: Serial baud rate (default 115200).
        """
        self.port = port or self._auto_detect_port()
        self.baudrate = baudrate
        self.ser = None
        self.start_time = None
        self.sample_count = 0

    @staticmethod
    def _auto_detect_port() -> str:
        """
        Auto-detect serial port by scanning common locations.

        Returns:
            Port string (e.g., '/dev/ttyUSB0') or raises error if none found.
        """
        candidates = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
        if candidates:
            return candidates[0]
        raise RuntimeError(
            "No serial port detected. Please specify port manually or "
            "connect Arduino/Raspberry Pi via USB."
        )

    def connect(self) -> None:
        """Establish serial connection and flush buffers."""
        if self.ser is not None:
            print(f"Already connected to {self.port}")
            return

        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=2)
            time.sleep(0.5)  # Wait for Arduino to reset
            self.ser.flush()
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()

            # Discard first few lines (header, stale data)
            for _ in range(5):
                self.ser.readline()

            self.start_time = time.time()
            self.sample_count = 0
            print(f"Connected to {self.port} at {self.baudrate} baud")
        except serial.SerialException as e:
            raise RuntimeError(f"Failed to connect to {self.port}: {e}")

    def disconnect(self) -> None:
        """Close serial connection."""
        if self.ser:
            self.ser.close()
            self.ser = None
            print(f"Disconnected from {self.port}")

    def is_connected(self) -> bool:
        """Check if connected."""
        return self.ser is not None and self.ser.is_open

    def read_sample(self) -> Tuple[float, float, float, float]:
        """
        Read one sample from accelerometer.

        Returns:
            (timestamp_s, x_g, y_g, z_g) tuple
                timestamp_s: seconds since connection
                x_g, y_g, z_g: acceleration in Gs

        Raises:
            RuntimeError: If not connected or parse error.
        """
        if not self.is_connected():
            raise RuntimeError("Not connected. Call connect() first.")

        line = self.ser.readline().decode("utf-8", errors="ignore").strip()
        if not line or line.startswith("timestamp"):  # Skip header
            return self.read_sample()

        try:
            parts = line.split(",")
            if len(parts) != 4:
                raise ValueError(f"Expected 4 fields, got {len(parts)}")

            ts_ms = float(parts[0])
            x_g = float(parts[1])
            y_g = float(parts[2])
            z_g = float(parts[3])

            # Validate ranges (accelerometers typically ±20g max)
            if not (-20 <= x_g <= 20 and -20 <= y_g <= 20 and -20 <= z_g <= 20):
                raise ValueError(f"Acceleration out of range: {x_g}, {y_g}, {z_g}")

            ts_s = ts_ms / 1000.0
            self.sample_count += 1
            return ts_s, x_g, y_g, z_g

        except (ValueError, IndexError) as e:
            # Skip malformed lines
            print(f"Warning: Skipped malformed line: {line} ({e})")
            return self.read_sample()

    def read_samples(self, n: int) -> List[Tuple[float, float, float, float]]:
        """
        Read n samples from accelerometer.

        Args:
            n: Number of samples to read.

        Returns:
            List of (timestamp_s, x_g, y_g, z_g) tuples.
        """
        samples = []
        for _ in range(n):
            try:
                samples.append(self.read_sample())
            except RuntimeError:
                break
        return samples

    def get_sample_rate(self) -> float:
        """
        Estimate sample rate based on recent samples.

        Returns:
            Estimated sample rate in Hz (requires at least 2 samples).
        """
        if self.sample_count < 2:
            return 0.0
        elapsed = time.time() - self.start_time
        return self.sample_count / elapsed
