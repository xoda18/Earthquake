# Earthquake Detection System

Detects seismic events and table knocks using real-time accelerometer analysis. Supports both synthetic data generation for testing and real MPU6050 hardware integration.

## Quick Start (Synthetic Data)

```bash
cd demo

# Generate synthetic earthquake data
python3 generate_data.py

# Run detection (earthquake mode)
python3 detect_earthquake.py

# Run detection (table knock mode - more sensitive)
python3 detect_earthquake.py --mode table_knock
```

Output:
- `accelerometer_data.csv` — synthetic sensor readings
- `earthquake_report.png` — visualization with 3 panels (raw, filtered, STA/LTA)
- Console report with event start/end times

---

## Real Hardware Integration (MPU6050)

### Hardware Setup

**Components:**
- Arduino (Uno, Nano) or Raspberry Pi with USB
- MPU6050 3-axis accelerometer (I2C module)
- Jumper wires

**Wiring (Arduino):**
```
MPU6050 Pin | Arduino Pin | Function
VCC         | 5V          | Power
GND         | GND         | Ground
SDA         | A4          | I2C Data
SCL         | A5          | I2C Clock
```

### Step 1: Upload Arduino Firmware

1. Open Arduino IDE
2. Install library: Sketch → Include Library → Manage Libraries
   - Search "MPU6050"
   - Install "MPU6050 by Jeff Rowberg"
3. Copy `demo/hardware/arduino_mpu6050.ino` into Arduino IDE
4. Select your board type and port
5. Upload

**Verify connection:**
- Open Serial Monitor (9600 baud)
- Should see CSV header: `timestamp_ms,x_g,y_g,z_g`
- Knock the table → values should change

### Step 2: Install Python Dependencies

```bash
pip3 install pyserial --break-system-packages
# (other dependencies: numpy, pandas, scipy, matplotlib already installed)
```

---

## Usage Modes

### Mode 1: Record & Analyze (Recommended for first demo)

Record N seconds, save to CSV, then run detection:

```bash
cd demo

# Auto-detect serial port, record 30 seconds
python3 record_and_analyze.py --duration 30

# Specify port explicitly
python3 record_and_analyze.py --port /dev/ttyUSB0 --duration 30 --output knock_recording.csv

# Use earthquake detection mode (default: table_knock)
python3 record_and_analyze.py --duration 30 --mode earthquake
```

**What happens:**
1. Connects to MPU6050
2. Streams data to terminal with progress bar
3. Saves to CSV file
4. Runs detection algorithm
5. Generates `earthquake_report.png`
6. Prints detection results

### Mode 2: Real-Time Streaming (Live visualization)

Monitor accelerometer in real-time with interactive plotting:

```bash
python3 realtime_monitor.py --duration 60
```

**Features:**
- Live 3-panel plot (updates every ~200 ms)
- Displays current sample rate, buffer fill, detection count
- Red shaded region shows detected events
- Optional CSV recording: `--save-csv recording.csv`

**Controls:**
- Close plot window to stop monitoring
- Press Ctrl+C to force exit

---

## Detection Modes

### Earthquake Mode (Default)

Optimized for long-duration seismic signals (30+ seconds):
- Bandpass filter: 1–20 Hz
- STA window: 0.5 s
- LTA window: 10 s
- Threshold: 3.0

Use when:
- Analyzing real seismic data
- Looking for sustained ground motion

### Table Knock Mode

Optimized for short impulse events (0.5–2 seconds):
- Bandpass filter: 2–25 Hz (higher frequency)
- STA window: 0.2 s (shorter)
- LTA window: 5 s (shorter)
- Threshold: 2.5 (lower/more sensitive)

Use when:
- Detecting table knocks
- Short, sharp vibration events

**Switch modes:**
```bash
python3 record_and_analyze.py --mode table_knock
python3 realtime_monitor.py --mode earthquake
python3 detect_earthquake.py  # reads from CSV, uses earthquake mode
```

---

## Analysis Pipeline (7 Stages)

1. **Data Ingestion** — Load CSV or stream from sensor
2. **Pre-processing** — Remove DC offset, apply bandpass filter
3. **Feature Extraction** — Compute STA/LTA ratio (seismic standard)
4. **Spike Detection** — Find energy anomalies via dual threshold
5. **Event Windowing** — Merge spikes, apply quiet-guard logic
6. **Visualization** — Generate 3-panel diagnostic plot
7. **Report** — Print detection summary (start, end, duration)

---

## Output Format

### CSV Data
```
timestamp,x,y,z
0.0,0.01,0.02,9.81
0.01,0.01,0.01,9.82
0.02,0.02,0.00,9.80
...
```

### Report (Console)
```
═══════════════════════════════════════════════════
✓  Earthquake detected!
   Start    : 2026-03-23 14:32:45.000
   End      : 2026-03-23 14:33:15.000
   Duration : 30.0 s
═══════════════════════════════════════════════════
```

### Diagram (`earthquake_report.png`)

Three stacked panels:
1. **Raw Magnitude** — acceleration vector |a| = √(x² + y² + z²)
2. **Filtered Signal** — bandpass-filtered acceleration
3. **STA/LTA Ratio** — seismic detection metric with threshold line

Red shaded region = detected event window
Red vertical lines = spike cluster starts

---

## Troubleshooting

### "No serial port detected"
- Check Arduino/Raspberry Pi is connected via USB
- List available ports: `ls /dev/tty*`
- Specify port explicitly: `--port /dev/ttyUSB0`

### "ModuleNotFoundError: No module named 'serial'"
```bash
pip3 install pyserial --break-system-packages
```

### Hardware not responding
1. Check I2C wiring (SDA/SCL pins correct?)
2. Verify baud rate (115200 default)
3. Upload firmware to Arduino again
4. Check MPU6050 LED (should blink on activity)

### Detection too sensitive / not sensitive enough
- **Too many false positives?** Use `--mode earthquake` (higher threshold)
- **Missing events?** Use `--mode table_knock` (lower threshold)
- For custom tuning, edit `CONFIG_PROFILES` in `detect_earthquake.py`

### Matplotlib error on remote server
Use headless mode:
```bash
# In record_and_analyze.py, no plot shown (image saved)
# Real-time monitor not available (use record_and_analyze instead)
```

---

## Testing

Run unit tests:
```bash
python3 test_hardware.py
```

Tests validate:
- Sensor buffer (circular queue)
- Configuration profiles
- Parameter ranges
- Mode switching

---

## Files

```
demo/
├── detect_earthquake.py              Main detection pipeline (modified with config profiles)
├── generate_data.py                 Synthetic data generator
├── earthquake_notebook.ipynb        Interactive Jupyter notebook
├── record_and_analyze.py            Record N sec + analyze
├── realtime_monitor.py              Live streaming + plotting
├── test_hardware.py                 Unit tests
│
└── hardware/
    ├── __init__.py
    ├── mpu6050_interface.py         Serial communication (MPU6050)
    ├── sensor_buffer.py             Circular buffer (streaming)
    └── arduino_mpu6050.ino          Arduino firmware (reference)
```

---

## Development Notes

### STA/LTA Algorithm

Standard in seismology for event detection:

- **STA (Short-Term Average)**: 0.5 s window, captures sudden energy
- **LTA (Long-Term Average)**: 10 s window, estimates background noise
- **Ratio**: STA / LTA; spike when >threshold (typically 3.0)

Why it works:
- Normalizes for varying noise floors
- Robust to amplitude variations
- Efficient O(n) with cumulative sum

### Parameter Tuning

For custom events, modify `CONFIG_PROFILES["custom"]` in `detect_earthquake.py`:

```python
CONFIG_PROFILES["custom"] = {
    "BP_LOW_HZ": 0.5,           # Lower for slower events
    "BP_HIGH_HZ": 50.0,         # Higher for faster events
    "STA_WINDOW_S": 1.0,        # Longer = smoother but slower response
    "LTA_WINDOW_S": 20.0,
    "STA_LTA_THRESH": 3.5,      # Higher = less sensitive
    "AMP_SIGMA_THRESH": 5.0,    # Higher = less sensitive
    "MERGE_GAP_S": 2.0,
    "QUIET_GUARD_S": 10.0,      # Longer = more conservative window
}
```

Then use: `--mode custom`

---

## References

- MPU6050 Datasheet: https://invensense.tdk.com/products/motion-tracking/6-axis/
- STA/LTA in seismology: https://en.wikipedia.org/wiki/STA/LTA

---

## License

Educational project. Use freely for learning and experimentation.
