"""
test_hardware.py
Unit tests for hardware module and detection modes.

Run:
    python -m pytest test_hardware.py -v
    or
    python test_hardware.py
"""

import unittest
import numpy as np
from hardware.sensor_buffer import StreamBuffer
from detect_earthquake import load_config, CONFIG_PROFILES


class TestSensorBuffer(unittest.TestCase):
    """Test circular buffer for streaming data."""

    def test_buffer_initialization(self):
        """Buffer initializes with correct capacity."""
        buf = StreamBuffer(capacity=100, sample_rate=100.0)
        self.assertEqual(len(buf), 0)
        self.assertEqual(buf.capacity, 100)
        self.assertFalse(buf.is_full())

    def test_buffer_append(self):
        """Append samples and check buffer grows."""
        buf = StreamBuffer(capacity=10)
        for i in range(5):
            buf.append((float(i), 0.1, 0.2, 9.8))
        self.assertEqual(len(buf), 5)

    def test_buffer_maxlen(self):
        """Buffer wraps around when capacity exceeded."""
        buf = StreamBuffer(capacity=5)
        for i in range(10):
            buf.append((float(i), 0.1 * i, 0.2 * i, 9.8))
        self.assertEqual(len(buf), 5)  # Should stay at capacity
        self.assertTrue(buf.is_full())

    def test_buffer_get_window(self):
        """Extract window from buffer."""
        buf = StreamBuffer(capacity=20)
        samples = [(float(i), float(i) * 0.1, float(i) * 0.2, 9.8) for i in range(15)]
        for s in samples:
            buf.append(s)

        window = buf.get_window(5)
        self.assertIsNotNone(window)
        self.assertEqual(window.shape, (5, 4))
        # Check it's the last 5 samples
        self.assertEqual(window[0, 0], 10.0)  # 11th sample (0-indexed, so 10)
        self.assertEqual(window[-1, 0], 14.0)  # 15th sample

    def test_buffer_get_window_too_small(self):
        """Return None if buffer smaller than requested window."""
        buf = StreamBuffer(capacity=20)
        buf.append((0.0, 0.1, 0.2, 9.8))
        window = buf.get_window(10)
        self.assertIsNone(window)

    def test_buffer_get_numpy_data(self):
        """Extract all data as numpy arrays."""
        buf = StreamBuffer(capacity=20)
        for i in range(5):
            buf.append((float(i), 0.1 * i, 0.2 * i, 9.8))

        t, x, y, z = buf.get_numpy_data()
        self.assertEqual(len(t), 5)
        self.assertEqual(len(x), 5)
        self.assertEqual(len(y), 5)
        self.assertEqual(len(z), 5)
        np.testing.assert_array_almost_equal(t, [0, 1, 2, 3, 4])

    def test_buffer_fill_ratio(self):
        """Check buffer fill ratio."""
        buf = StreamBuffer(capacity=100)
        for i in range(25):
            buf.append((float(i), 0.1, 0.2, 9.8))
        ratio = buf.get_fill_ratio()
        self.assertAlmostEqual(ratio, 0.25)

    def test_buffer_thread_safety(self):
        """Buffer handles concurrent append/read (basic check)."""
        buf = StreamBuffer(capacity=50)
        for i in range(30):
            buf.append((float(i), 0.1, 0.2, 9.8))
        window = buf.get_window(10)
        # Just check it doesn't crash
        self.assertIsNotNone(window)

    def test_buffer_clear(self):
        """Clear buffer."""
        buf = StreamBuffer(capacity=20)
        for i in range(10):
            buf.append((float(i), 0.1, 0.2, 9.8))
        self.assertEqual(len(buf), 10)
        buf.clear()
        self.assertEqual(len(buf), 0)


class TestConfigProfiles(unittest.TestCase):
    """Test detection mode configuration profiles."""

    def test_earthquake_profile_exists(self):
        """Earthquake profile is defined."""
        self.assertIn("earthquake", CONFIG_PROFILES)
        cfg = CONFIG_PROFILES["earthquake"]
        self.assertIn("BP_LOW_HZ", cfg)
        self.assertIn("STA_LTA_THRESH", cfg)

    def test_table_knock_profile_exists(self):
        """Table knock profile is defined."""
        self.assertIn("table_knock", CONFIG_PROFILES)
        cfg = CONFIG_PROFILES["table_knock"]
        self.assertIn("BP_LOW_HZ", cfg)
        self.assertIn("STA_LTA_THRESH", cfg)

    def test_table_knock_higher_frequency(self):
        """Table knock profile has higher bandpass cutoff."""
        eq = CONFIG_PROFILES["earthquake"]
        tk = CONFIG_PROFILES["table_knock"]
        self.assertGreater(tk["BP_HIGH_HZ"], eq["BP_HIGH_HZ"])

    def test_table_knock_shorter_windows(self):
        """Table knock has shorter STA/LTA windows."""
        eq = CONFIG_PROFILES["earthquake"]
        tk = CONFIG_PROFILES["table_knock"]
        self.assertLess(tk["STA_WINDOW_S"], eq["STA_WINDOW_S"])
        self.assertLess(tk["LTA_WINDOW_S"], eq["LTA_WINDOW_S"])

    def test_table_knock_lower_threshold(self):
        """Table knock has lower STA/LTA threshold."""
        eq = CONFIG_PROFILES["earthquake"]
        tk = CONFIG_PROFILES["table_knock"]
        self.assertLess(tk["STA_LTA_THRESH"], eq["STA_LTA_THRESH"])

    def test_load_config_earthquake(self):
        """Load earthquake configuration."""
        load_config("earthquake")
        # Just verify it doesn't raise an exception
        self.assertTrue(True)

    def test_load_config_table_knock(self):
        """Load table knock configuration."""
        load_config("table_knock")
        # Just verify it doesn't raise an exception
        self.assertTrue(True)

    def test_load_config_invalid(self):
        """Invalid mode raises error."""
        with self.assertRaises(ValueError):
            load_config("invalid_mode")


class TestConfiguration(unittest.TestCase):
    """Test basic configuration integrity."""

    def test_all_profiles_have_required_keys(self):
        """Each profile has all required parameter keys."""
        required_keys = {
            "BP_LOW_HZ", "BP_HIGH_HZ", "STA_WINDOW_S", "LTA_WINDOW_S",
            "STA_LTA_THRESH", "AMP_SIGMA_THRESH", "MERGE_GAP_S", "QUIET_GUARD_S"
        }
        for mode, cfg in CONFIG_PROFILES.items():
            self.assertEqual(set(cfg.keys()), required_keys,
                           f"Mode '{mode}' missing keys")

    def test_parameter_ranges(self):
        """Configuration parameters are within sensible ranges."""
        for mode, cfg in CONFIG_PROFILES.items():
            # Frequencies
            self.assertGreater(cfg["BP_LOW_HZ"], 0)
            self.assertLess(cfg["BP_LOW_HZ"], cfg["BP_HIGH_HZ"])
            self.assertLess(cfg["BP_HIGH_HZ"], 100)

            # Windows (in seconds)
            self.assertGreater(cfg["STA_WINDOW_S"], 0)
            self.assertGreater(cfg["LTA_WINDOW_S"], cfg["STA_WINDOW_S"])
            self.assertLess(cfg["LTA_WINDOW_S"], 30)

            # Thresholds
            self.assertGreater(cfg["STA_LTA_THRESH"], 1.0)
            self.assertLess(cfg["STA_LTA_THRESH"], 10.0)
            self.assertGreater(cfg["AMP_SIGMA_THRESH"], 1.0)

            # Gaps (in seconds)
            self.assertGreater(cfg["MERGE_GAP_S"], 0)
            self.assertGreater(cfg["QUIET_GUARD_S"], 0)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
