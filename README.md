# Earthquake Detection System

Real-time earthquake detection using MPU6500 accelerometer + Arduino + LSTM neural network in Docker.

## Setup

```bash
git clone https://github.com/JASS-2026-Cyprus/Earthquake.git
cd Earthquake
./setup.sh
```

## USING:

```bash

 cd ~/Desktop/NUP/JASS/Earthquake && source ../venv/bin/activate &&
  python3 detection/streaming/stream.py --detect --blackboard --viz
  --threshold 0.7 --cooldown 5

```

Needs: Docker, Arduino Uno + MPU6500 sensor connected via USB.

## Run

### Option 1: Docker (LSTM classification + blackboard)

```bash
docker run --rm --device /dev/ttyACM0 earthquake-detector --profile table
```

### Option 2: All-in-one Python (streaming + detection + blackboard + visualization)

```bash
cd ~/Desktop/NUP/JASS/Earthquake && source ../venv/bin/activate

# Everything: data + detection + blackboard + live graphs
python3 detection/streaming/stream.py --detect --blackboard --viz

# Without graphs
python3 detection/streaming/stream.py --detect --blackboard

# Stream only
python3 detection/streaming/stream.py

# Stream + graphs (no blackboard)
python3 detection/streaming/stream.py --viz

# Save to CSV
python3 detection/streaming/stream.py --detect --blackboard --viz --save data.csv
```

### Flags

| Flag | Description |
|------|-------------|
| `--detect` | Enable STA/LTA earthquake detection |
| `--blackboard` | Post data to ORB blackboard |
| `--viz` | Show live graphs (3D vector + waveforms + STA/LTA) |
| `--save FILE` | Save data to CSV |
| `--mode` | Detection mode: `earthquake`, `table_knock`, `optimized` |
| `--rate N` | Print rate per second (default 5) |
| `--port` | Serial port (auto-detected) |

## Docker Profiles

| Profile | Use case | Gain | Threshold |
|---------|----------|------|-----------|
| `default` | Balanced | 1x | 0.7 |
| `table` | Table demo — knocks detected | 3x | 0.8 |
| `sensitive` | Light taps and vibrations | 5x | 0.6 |
| `earthquake` | Real seismic only | 1x | 0.9 |

## Hardware Wiring

```
MPU6500 VCC → Arduino 3.3V
MPU6500 GND → Arduino GND
MPU6500 SDA → Arduino A4
MPU6500 SCL → Arduino A5
```

## How It Works

```
MPU6500 → Arduino (100Hz serial) → Python/Docker → detection + blackboard + visualization
```

1. Sensor reads acceleration (ax, ay, az) at 100 Hz
2. Gravity is calibrated and removed from signal
3. STA/LTA detection finds energy spikes (or LSTM classifies in Docker)
4. Results posted to ORB blackboard for other agents
5. Live 3D + waveform visualization (optional)

## Docs

- [Streaming & Docker details](detection/streaming/README.md)
- [Troubleshooting](detection/streaming/TROUBLESHOOTING.md)
- [System prompt for ORB](earthquake_agent_system_prompt.md)

## If Arduino Disconnects

```bash
./detection/streaming/setup.sh
```
