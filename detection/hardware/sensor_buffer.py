"""
sensor_buffer.py
Circular buffer for real-time streaming accelerometer data.

Stores the last N samples in a fixed-size rolling buffer.
Thread-safe for background data collection.
"""

import numpy as np
from collections import deque
from threading import Lock
from typing import Tuple, Optional


class StreamBuffer:
    """Fixed-size circular buffer for streaming accelerometer data."""

    def __init__(self, capacity: int = 600, sample_rate: float = 100.0):
        """
        Initialize buffer.

        Args:
            capacity: Max number of samples to store (default 6 seconds at 100 Hz).
            sample_rate: Expected sample rate in Hz (for timestamp generation).
        """
        self.capacity = capacity
        self.sample_rate = sample_rate
        self.lock = Lock()
        self.buffer: deque = deque(maxlen=capacity)
        self.sample_count = 0

    def append(self, sample: Tuple[float, float, float, float]) -> None:
        """
        Add a sample to the buffer.

        Args:
            sample: (timestamp_s, x_g, y_g, z_g) tuple.
        """
        with self.lock:
            self.buffer.append(sample)
            self.sample_count += 1

    def get_window(self, n_samples: int) -> Optional[np.ndarray]:
        """
        Get the last n samples as numpy array.

        Args:
            n_samples: Number of recent samples to extract.

        Returns:
            Array of shape (n, 4) with [ts, x, y, z] columns, or None if buffer too small.
        """
        with self.lock:
            if len(self.buffer) < n_samples:
                return None
            # Get last n samples in order
            window = list(self.buffer)[-n_samples:]
            return np.array(window, dtype=np.float64)

    def get_numpy_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all data in buffer as separate arrays.

        Returns:
            (timestamps, x, y, z) tuple of numpy arrays.
        """
        with self.lock:
            if not self.buffer:
                return (
                    np.array([]),
                    np.array([]),
                    np.array([]),
                    np.array([]),
                )
            data = np.array(list(self.buffer), dtype=np.float64)
            return data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    def is_full(self) -> bool:
        """Check if buffer has reached capacity."""
        with self.lock:
            return len(self.buffer) == self.capacity

    def get_fill_ratio(self) -> float:
        """Get buffer fill ratio (0.0 to 1.0)."""
        with self.lock:
            return len(self.buffer) / self.capacity if self.capacity > 0 else 0.0

    def clear(self) -> None:
        """Clear all data from buffer."""
        with self.lock:
            self.buffer.clear()
            self.sample_count = 0

    def __len__(self) -> int:
        """Get current number of samples in buffer."""
        with self.lock:
            return len(self.buffer)
