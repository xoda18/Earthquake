# JASS Earthquake Detection — Swarm Agent

Seismic detection + drone damage assessment for the JASS 2026 Smart Eco City (Paphos, Cyprus).

Integrates with the [JASS Swarm](https://github.com/JASS-2026-Cyprus/swarm) multi-agent system via Redis.

## Architecture

```
MPU6500 Sensor → LSTM Detection → detection/sensor_log.jsonl
                                        ↓
                               EarthAgent (swarm)
                               reads logs via tools.py
                                        ↓
                               report_observation → Redis shared log
                                        ↓
                               MaintenanceAgent dispatches drone
                                        ↓
                               Drone scans → drone/drone_log.jsonl
                                        ↓
                               EarthAgent reads drone log
                                        ↓
                               report_observation → damage locations to all agents
```

## Quick Start

### 1. Run sensor (writes to `detection/sensor_log.jsonl`)

```bash
# With visualization
python3 detection/realtime_visualizer.py --port /dev/tty.usbmodemXXXX

# CLI only
python3 detection/realtime_predict.py --port /dev/tty.usbmodemXXXX

# Find your port
ls /dev/tty.usb*
```

### 2. Run drone simulator (writes to `drone/drone_log.jsonl`)

```bash
python3 drone/mock_damage.py --count 5
python3 drone/mock_damage.py --loop 10 --count 2
```

### 3. Run swarm (in the swarm repo)

```bash
cd /path/to/swarm
docker compose up --build
# or
python main.py
```

EarthAgent reads `sensor_log.jsonl` and `drone_log.jsonl` via its tools.

## Swarm Integration

### Agent config: `swarm/agents/earth/agent.toml`

EarthAgent monitors seismic activity and coordinates drone response.

### Tools: `swarm/agents/earth/tools.py`

| Tool | Description |
|------|-------------|
| `read_sensor_log(last_n)` | Read last N earthquake detections |
| `read_drone_log(last_n)` | Read last N drone damage reports |
| `get_new_earthquakes(since_epoch)` | Poll for new earthquakes since timestamp |
| `get_new_damage_reports(since_epoch)` | Poll for new damage since timestamp |

### Log formats

**`detection/sensor_log.jsonl`** (one JSON per line):
```json
{"timestamp":"2026-03-25T10:00:00Z","epoch":1774000000,"type":"earthquake","probability":0.95,"pga_g":0.15,"ax":0.08,"ay":-0.01,"az":-1.02,"magnitude_g":1.03}
```

**`drone/drone_log.jsonl`** (one JSON per line):
```json
{"timestamp":"2026-03-25T10:01:00Z","epoch":1774000060,"type":"damage","eventId":"uuid","lat":34.765,"lon":32.420,"severity":"high","damage_type":"Wall crack detected","building":"Hotel","confidence":0.92}
```

## Model Training

```bash
# Train on real datasets (LEN-DB + Central Italy)
python3 models/train_lstm_real.py

# Train on synthetic data
python3 models/train_lstm_csv.py

# Test model
python3 models/earthquake_simulator/test_model.py
```

## Hardware Wiring

```
MPU6500 VCC → Arduino 3.3V
MPU6500 GND → Arduino GND
MPU6500 SDA → Arduino A4
MPU6500 SCL → Arduino A5
```

## Project Structure

```
Earthquake/
├── detection/                    # Sensor + LSTM inference
│   ├── realtime_predict.py       # CLI inference
│   ├── realtime_visualizer.py    # Visualization + writes sensor_log.jsonl
│   ├── sensor_log.jsonl          # ← earthquake events (runtime)
│   ├── hardware/                 # Arduino serial interface
│   └── streaming/                # Docker variant
├── drone/                        # Drone damage simulation
│   ├── mock_damage.py            # Writes drone_log.jsonl
│   └── drone_log.jsonl           # ← damage reports (runtime)
├── models/                       # LSTM model + training
│   ├── lstm_earthquake_model.h5
│   ├── lstm_scaler.pkl
│   ├── train_lstm_real.py
│   ├── train_lstm_csv.py
│   └── earthquake_simulator/
└── hypothesis_generator/         # LLM geological analysis
```
