# Quick Start Guide

## In 2 Minutes: Test with Synthetic Data

```bash
cd demo

# Generate fake sensor data with an earthquake
python3 generate_data.py

# Run detection
python3 detect_earthquake.py

# Output: earthquake_report.png + console report
```

## With Real Hardware (MPU6050 + Arduino)

### Hardware Setup (5 min)
1. Wire MPU6050 to Arduino (VCC → 5V, GND → GND, SDA → A4, SCL → A5)
2. Upload `demo/hardware/arduino_mpu6050.ino` to Arduino
3. Check serial monitor (9600 baud) — should show CSV data

### Recording & Analysis (30 sec)
```bash
# Record 30 seconds (knock table a few times)
python3 record_and_analyze.py --duration 30

# Output: knock_recording.csv + earthquake_report.png
```

### Live Monitoring (real-time)
```bash
python3 realtime_monitor.py --duration 60
```

---

## Two Detection Modes

| Aspect | Earthquake | Table Knock |
|--------|-----------|-------------|
| Use for | Seismic events | Impulse events (knocks) |
| Duration | 30+ seconds | 0.5–2 seconds |
| Frequency | 1–20 Hz | 2–25 Hz |
| Sensitivity | Lower (threshold 3.0) | Higher (threshold 2.5) |
| Command | `--mode earthquake` | `--mode table_knock` (default) |

---

## Common Workflows

### 1. Quick Test (No Hardware)
```bash
python3 generate_data.py
python3 detect_earthquake.py --mode earthquake
```

### 2. Detect Table Knock (Hardware)
```bash
python3 record_and_analyze.py --duration 20 --mode table_knock
# Knock table during recording
```

### 3. Live Monitoring (Hardware)
```bash
python3 realtime_monitor.py --duration 120 --save-csv recording.csv
# Knocks appear in real-time plot with detection alerts
```

### 4. Analyze Existing CSV
```bash
python3 detect_earthquake.py my_recording.csv
```

---

## Troubleshooting

**"No serial port detected"**
- Check Arduino is connected via USB
- Explicitly specify: `--port /dev/ttyUSB0`

**"Missing module: serial"**
```bash
pip3 install pyserial --break-system-packages
```

**Not detecting events**
- Use `--mode table_knock` (more sensitive)
- Check signal amplitude (knock harder!)

**Too many false alarms**
- Use `--mode earthquake` (less sensitive)
- Or reduce STA/LTA threshold in `detect_earthquake.py`

---

## Key Files

| File | Purpose |
|------|---------|
| `detect_earthquake.py` | Core detection algorithm (7 stages) |
| `record_and_analyze.py` | Record from hardware + detect |
| `realtime_monitor.py` | Live streaming + plotting |
| `generate_data.py` | Synthetic test data |
| `hardware/mpu6050_interface.py` | Serial communication |
| `hardware/arduino_mpu6050.ino` | Arduino firmware |

---

## Output: What You Get

**earthquake_report.png:**
- 3-panel plot showing raw, filtered, and STA/LTA signals
- Red shaded region = detected event
- Red lines = spike cluster starts

**Console report:**
```
✓  Earthquake detected!
   Start    : 2026-03-23 14:32:45.000
   End      : 2026-03-23 14:33:15.000
   Duration : 30.0 s
```

**CSV recording** (optional):
- Timestamped x, y, z acceleration values
- Can be re-analyzed anytime

---

## Learn More

See **README.md** for:
- Detailed setup instructions
- Parameter tuning guide
- STA/LTA algorithm explanation
- Hardware pin diagrams

Run tests:
```bash
python3 test_hardware.py
```
