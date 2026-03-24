# Arduino Sensor — Real-Time Earthquake Detection

Real-time accelerometer streaming from MPU6500 sensor via Arduino, with LSTM neural network classification running in Docker.

## Architecture

```
MPU6500 sensor (I2C, 100Hz)
    ↓
Arduino Uno (USB Serial, 115200 baud, CSV)
    ↓
Docker container (Python 3.11 + TensorFlow)
    ├── 1. Calibrate gravity (first 50 samples, ~0.5s)
    ├── 2. Remove gravity from readings
    ├── 3. Scale g-force → raw counts (×--scale)
    ├── 4. Clip to ±596 (training data range)
    ├── 5. Fill 100-sample sliding window
    ├── 6. Normalize with StandardScaler
    ├── 7. Feed into LSTM model → probability 0.0–1.0
    └── 8. If prob > --threshold → EARTHQUAKE
             ↓
        Terminal output + (future) ORB blackboard
```

## Hardware

| Sensor Pin | Arduino Pin | Notes |
|------------|-------------|-------|
| VCC        | 3.3V        | NOT 5V |
| GND        | GND         | |
| SDA        | A4          | I2C data |
| SCL        | A5          | I2C clock |
| AD0        | GND (on-board) | I2C address = 0x68 |

**Chip:** MPU6500 (6-axis: accel + gyro, WHO_AM_I = 0x70)

## Quick Start

```bash
# 1. Upload firmware to Arduino (each time you reconnect USB)
./demo/arduino-sensor/setup.sh

# 2. Run LSTM classifier in Docker
docker run --rm --device /dev/ttyACM0 earthquake-detector
```

## Docker Parameters

```bash
docker run --rm --device /dev/ttyACM0 earthquake-detector [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--rate N` | 5 | Classifications per second |
| `--threshold N` | 0.7 | Probability above which → EARTHQUAKE (0.0–1.0) |
| `--scale N` | 1000 | G-force to raw count multiplier (lower = less sensitive) |
| `--mode` | probability | `probability`: print every reading. `magnitude`: only print earthquakes |
| `--port` | auto | Serial port (auto-detected) |

### Examples

```bash
# Default: 5/sec, show all readings
docker run --rm --device /dev/ttyACM0 earthquake-detector

# Less sensitive (recommended starting point)
docker run --rm --device /dev/ttyACM0 earthquake-detector --scale 500 --threshold 0.8

# Silent mode: only print when earthquake detected
docker run --rm --device /dev/ttyACM0 earthquake-detector --mode magnitude --threshold 0.85

# Very sensitive (detects light taps)
docker run --rm --device /dev/ttyACM0 earthquake-detector --scale 2000 --threshold 0.6
```

## Output Columns

**Probability mode** (default):
```
    Time      |a|      amp    Prob  Result
================================================
     3.2   1.0247   0.0031   47.3%  quiet
    22.9   1.0310   0.1520   98.2%  EARTHQUAKE <<<
```

| Column | Meaning |
|--------|---------|
| `Time` | Seconds since start |
| `\|a\|` | Total acceleration magnitude (always ~1.02g when still) |
| `amp` | Deviation from gravity — how much the sensor is actually moving (0 = still) |
| `Prob` | LSTM model output (0–100%). ~47% = quiet baseline, >threshold = earthquake |
| `Result` | `EARTHQUAKE` if prob > threshold, else `quiet` |

**Magnitude mode** (`--mode magnitude`):
```
[  22.9s] EARTHQUAKE  prob=98.2%  amp=0.1520g
```
Only prints when an earthquake is detected. Silent otherwise.

## How the LSTM Model Works

The model (`lstm_earthquake_model.h5`) is a pre-trained LSTM neural network from the `Alex/` directory.

### Model Architecture
```
Input: (3, 100) — 3 axes × 100 timesteps
    ↓
LSTM(64, relu) → Dropout(0.2)
    ↓
LSTM(32, relu) → Dropout(0.2)
    ↓
Dense(16, relu) → Dense(1, sigmoid)
    ↓
Output: probability 0.0–1.0 (earthquake vs quiet)
```

### Data Pipeline Inside Docker

1. **Calibration** — First 50 samples measure gravity while sensor is still. Prints "Calibration done" with gravity vector.

2. **Gravity removal** — Each reading has gravity subtracted: `ax_clean = ax - gravity_x`. This centers the signal around zero so only vibrations remain.

3. **Scale conversion** — Our sensor outputs g-force (±2g). The model was trained on raw accelerometer counts (±596). The `--scale` flag controls this conversion: `raw = clean_g × scale`. Default 1000 means 0.001g movement → 1 count.

4. **Clipping to ±596** — Values outside ±596 are clamped. This matches the training data range. There are ~1192 possible values (-596 to +596).

5. **100-sample window** — The LSTM needs exactly 100 timesteps (1 second at 100Hz). This is fixed by the model architecture — cannot be changed without retraining.

6. **StandardScaler** — Normalizes using mean/std from training data (`lstm_scaler.pkl`).

7. **LSTM inference** — The model outputs a sigmoid probability. ~47% is the quiet baseline (not exactly 0% because the model was trained on different sensor hardware). Above `--threshold` → earthquake.

### Why Quiet ≈ 47% Not 0%

The model was trained on seismological waveform data (HDF5 from a research dataset), not on MPU6500 accelerometer data. The statistical properties differ slightly, so the model's "zero point" sits at ~47% rather than ~0%. This is normal when using a model on data it wasn't specifically trained on. The threshold parameter compensates for this.

## Streaming Without Docker

For raw data streaming without the LSTM model:

```bash
source ../venv/bin/activate

# Stream + save CSV
python3 demo/arduino-sensor/stream.py --save sensor_data.csv

# Stream + STA/LTA detection
python3 demo/arduino-sensor/stream.py --detect --save sensor_data.csv

# Pipe raw CSV (for custom processing)
python3 demo/arduino-sensor/stream.py --pipe > raw_data.csv
```

### Offline Analysis

After collecting data, run the full 7-stage STA/LTA pipeline:
```bash
python3 demo/detect_earthquake.py sensor_data.csv
# Generates earthquake_report.png
```

## Integration with Other Services (ORB Blackboard)

The JASS system uses an **ORB blackboard** for inter-service communication. When an earthquake is detected, the event can be posted to the blackboard for other agents to consume.

### Current ORB Configuration

From `ORB/service.py`:
- **ORB URL:** `https://crooked-jessenia-nongenerating.ngrok-free.dev`
- **Service name:** `jass_earthquake_analysis`
- **Agent name:** `earthquake_agent`

### How Events Flow to Other Services

```
Docker (LSTM detects earthquake)
    ↓ POST /blackboard
ORB Blackboard
    ↓ read by
├── Pipeline 2: Satellite risk mapping (Misha, Iishak)
├── Pipeline 3: Drone damage assessment (Thomas)
└── Central LLM (Satya) — aggregates all pipeline outputs
```

### Blackboard Payload Format

When an earthquake is detected, the following JSON is posted:
```json
{
    "agent": "jass_earthquake_analysis",
    "type": "seismic_detection",
    "content": "Earthquake detected: prob=92.3%, amplitude=0.152g",
    "confidence": 0.923,
    "timestamp": 1774342044.5
}
```

Other agents poll or subscribe to the blackboard to receive these events.

### Integration Status

- **Working now:** Docker reads Arduino → LSTM classifies → prints to terminal
- **Next step:** Add ORB blackboard POST inside `infer.py` when earthquake is detected
- **Requires:** ORB server running (Satya's component)

To enable ORB posting, set the environment variable:
```bash
docker run --rm --device /dev/ttyACM0 \
    -e ORB_URL="https://your-orb-url.ngrok.dev" \
    earthquake-detector
```

## Files

| File | Purpose |
|------|---------|
| `sketch.ino` | Arduino firmware — raw I2C reads from MPU6500, CSV output at 100Hz |
| `stream.py` | Host-side streaming — reads Arduino, optional STA/LTA detection |
| `infer.py` | Docker entrypoint — reads Arduino, runs LSTM, prints classifications |
| `Dockerfile` | Python 3.11 + TensorFlow + pyserial |
| `setup.sh` | One-command Arduino upload + verification |
| `lstm_earthquake_model.h5` | Pre-trained LSTM weights (from Alex/) |
| `lstm_scaler.pkl` | StandardScaler fitted on training data |
| `TROUBLESHOOTING.md` | Hardware debugging guide |

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for:
- Checking Arduino connection
- Fixing serial port permissions
- Verifying sensor wiring
- Common errors and solutions

## Rebuilding Docker Image

After editing `infer.py`:
```bash
cd demo/arduino-sensor
docker build -t earthquake-detector .
```
