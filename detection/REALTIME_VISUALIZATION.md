# Real-Time Earthquake Visualization

Advanced real-time 3D visualization system for seismic sensor data with live earthquake detection highlighting.

## Overview

`realtime_visualizer.py` provides interactive monitoring of accelerometer data with 6 synchronized plots:

1. **3D Acceleration Vector** - Shows the live acceleration vector in 3D space
2. **X, Y, Z Waveforms** - Three-axis acceleration traces over time
3. **Magnitude Plot** - Resultant acceleration magnitude (√(x²+y²+z²))
4. **STA/LTA Ratio** - Detection signal with threshold line and spike highlights
5. **Detection Status** - Real-time metrics and earthquake alerts
6. **Acceleration Distribution** - Histogram of acceleration values

All earthquake-detected regions are **highlighted in red** across all plots for easy visual identification.

---

## Quick Start

### Basic Usage (Auto-detect serial port)

```bash
cd /home/morozov-mikhail/Earthquake/detection

# Monitor for 60 seconds
python3 realtime_visualizer.py --duration 60

# Monitor for 2 minutes in earthquake mode
python3 realtime_visualizer.py --duration 120 --mode earthquake

# Monitor and save data to CSV
python3 realtime_visualizer.py --duration 60 --save-csv my_recording.csv
```

### Specify Serial Port Explicitly

```bash
# List available ports
ls /dev/tty*

# Use specific port
python3 realtime_visualizer.py --port /dev/ttyACM0 --duration 60
```

---

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--port PORT` | auto-detect | Serial port (e.g., `/dev/ttyACM0`) |
| `--duration SEC` | 60 | Monitoring duration in seconds |
| `--mode MODE` | optimized | Detection mode: `earthquake`, `table_knock`, or `optimized` |
| `--save-csv FILE` | None | Save sensor data to CSV file (e.g., `data.csv`) |

---

## Detection Modes

### Earthquake Mode
- **Frequency range**: 1–20 Hz (natural seismic)
- **STA window**: 0.5s | **LTA window**: 10s
- **Threshold**: 3.0
- **Best for**: Long-duration ground motion (30+ seconds), real earthquakes

```bash
python3 realtime_visualizer.py --mode earthquake --duration 120
```

### Table Knock Mode
- **Frequency range**: 2–25 Hz (higher frequency)
- **STA window**: 0.2s | **LTA window**: 5s
- **Threshold**: 8.0 (more sensitive)
- **Best for**: Short, sharp impacts (< 2 seconds), lab testing

```bash
python3 realtime_visualizer.py --mode table_knock --duration 60
```

### Optimized Mode (Default)
- **Hybrid parameters** tuned for Mediterranean deployment
- **Adaptive thresholding** based on local noise
- **Best for**: Production/continuous monitoring

```bash
python3 realtime_visualizer.py --mode optimized --duration 60
```

---

## Plot Descriptions

### 1. 3D Acceleration Vector
- **Shows**: Real-time position of acceleration vector in 3D space
- **Color code**:
  - Blue line: trajectory history
  - Red dot: current acceleration vector
- **How to read**: Watch for large deviations from origin (0,0,0) indicating motion

### 2. X, Y, Z Waveforms
- **Shows**: Individual acceleration on each axis vs. time
- **Red shaded region**: Detected earthquake period
- **How to read**: Look for synchronized increase in all three axes during earthquakes

### 3. Magnitude Plot
- **Shows**: Resultant acceleration magnitude over time
- **Red shaded region**: Detected earthquake
- **Dashed line**: Baseline (average of first 100 samples)
- **Red dot**: Peak acceleration during earthquake
- **How to read**: Magnitude > baseline indicates motion; peaks show intensity

### 4. STA/LTA Ratio
- **Shows**: Short-term / long-term energy ratio (detection signal)
- **Red dashed line**: Detection threshold (crosses above = earthquake detected)
- **Red dots**: Detected spikes
- **Red shaded region**: Earthquake event window
- **How to read**: Higher ratio = more likely earthquake; sudden spikes indicate event onset

### 5. Detection Status
- **Real-time metrics**:
  - Current X, Y, Z acceleration
  - Current/peak magnitude
  - Baseline acceleration
  - Current STA/LTA ratio vs. threshold
- **Status indicator**:
  - 🟢 QUIET (below threshold)
  - 🔴 EARTHQUAKE (above threshold)
- **Detection count**: Number of earthquakes found

### 6. Acceleration Distribution
- **Shows**: Histogram of all acceleration values
- **Green line**: Mean acceleration
- **How to read**: Narrow distribution = calm; broad distribution = active seismic period

---

## Real-Time Features

### Live Updates
- All plots update every **200 ms**
- 1000-sample rolling window (10 seconds @ 100 Hz)
- No lag between sensor and visualization

### Earthquake Highlighting
- **Automatic detection** using STA/LTA algorithm
- **Red highlighting** appears across all plots
- **Synchronized** across X/Y/Z waveforms, magnitude, and ratio plots

### Interactive Controls
- **Zoom**: Click + drag to zoom
- **Pan**: Right-click + drag to move
- **Reset**: Double-click to reset view
- **Close**: Click window close button to stop

---

## Data Recording

### Save Sensor Data During Monitoring

```bash
python3 realtime_visualizer.py --duration 60 --save-csv earthquake_data.csv
```

**Output format:**
```
timestamp_ms,x_g,y_g,z_g
1234567,0.01,0.02,9.81
1234568,0.01,0.01,9.82
...
```

### Use Saved Data for Further Analysis

```bash
# Analyze with offline detection
python3 detect_earthquake.py earthquake_data.csv

# Load in Jupyter notebook
import pandas as pd
data = pd.read_csv("earthquake_data.csv")
```

---

## Troubleshooting

### No Serial Connection
```
RuntimeError: No serial port detected...
```

**Solution:**
1. Check Arduino is plugged in: `ls /dev/tty*`
2. Specify port explicitly: `python3 realtime_visualizer.py --port /dev/ttyACM0`
3. Run: `./setup_arduino.sh` to upload firmware

### Plot Updates Too Slow
- Reduce `--duration` to lower update frequency
- Close other CPU-intensive applications
- Use `--mode optimized` (may be lighter than `earthquake`)

### Earthquake Not Detected
- Try `--mode table_knock` (more sensitive)
- Verify sensor is connected and moving data
- Check threshold value in mode config

### No Data Points on Plot
- Wait 1-2 seconds for buffer to fill (needs 100 samples at 100 Hz)
- Check serial connection: `python3 -c "from hardware.mpu6050_interface import MPU6050Reader; r = MPU6050Reader(); r.connect(); print(r.read_sample())"`

---

## Example Scenarios

### Scenario 1: Test with Table Knock
```bash
# Start visualizer
python3 realtime_visualizer.py --mode table_knock --duration 30

# In another terminal, tap table/sensor to generate events
# Watch plots for 2-second high-frequency bursts
```

**Expected**: Quick spikes in all plots, red highlighting appears.

### Scenario 2: Record Ambient Seismic Activity
```bash
python3 realtime_visualizer.py --mode optimized --duration 300 --save-csv ambient_noise.csv
```

**Expected**: Low, steady baseline; occasional small spikes from natural vibrations.

### Scenario 3: Continuous Monitoring
```bash
# Monitor indefinitely (Ctrl+C to stop)
python3 realtime_visualizer.py --duration 3600 --save-csv daily_log.csv
```

**Expected**: 1 hour of continuous monitoring, saved to CSV.

---

## Integration with Detection Pipeline

The visualizer integrates seamlessly with the broader earthquake detection system:

```
Hardware (MPU6050 @ 100Hz)
    ↓
realtime_visualizer.py (live plots)
    ↓ (uses same detection as...)
    ↓
detect_earthquake.py (batch analysis)
    ↓
ORB Blackboard (publish results)
```

**All three use the same**:
- STA/LTA algorithm
- Configuration profiles
- Detection thresholds

---

## Performance

| Metric | Value |
|--------|-------|
| Plot update interval | 200 ms |
| Buffer window | 1000 samples (10 sec @ 100 Hz) |
| CPU usage | ~15–20% (matplotlib on TkAgg) |
| Memory usage | ~80–100 MB |
| Latency (sensor → plot) | < 500 ms |

---

## Files

- **`realtime_visualizer.py`** - Main visualization script
- **`detect_earthquake.py`** - Core STA/LTA detection algorithm
- **`hardware/mpu6050_interface.py`** - Serial communication
- **`hardware/sensor_buffer.py`** - Streaming buffer

---

## Next Steps

After visualization and recording:

1. **Analyze offline**: `python3 detect_earthquake.py <csv_file>`
2. **Test ML model**: `python3 realtime_predict.py --csv <csv_file>`
3. **Generate hypothesis**: Python script with `hypothesis_generator/earthquake_analyzer.py`
4. **Publish to ORB**: Use ORB integration to share findings

---

*Real-Time Visualization Module · Detection System v2.0 · 2026-03*
