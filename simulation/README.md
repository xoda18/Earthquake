# JASS Earthquake Response — All Commands

## Live Demo (Real Sensor)

Open 3 terminals:

```bash
# Terminal 1 — web map
python3 web/server.py
# Open http://localhost:8080

# Terminal 2 — drone listener (waits for earthquake)
python3 drone/listener.py

# Terminal 3 — sensor + LSTM + visualization
python3 detection/realtime_visualizer.py --port /dev/tty.usbmodemXXXX
```

Now **shake the sensor** — when LSTM detects earthquake:
1. Posts `earthquake_sensor` to blackboard
2. Posts `drone_dispatch` to blackboard
3. Drone listener picks it up → flies → posts `damage_zone`
4. Map shows red zones in Paphos

To find your port: `ls /dev/tty.usb*`

---

## Simulated Demo (No Sensor)

```bash
# Terminal 1 — web map
python3 web/server.py

# Terminal 2 — full simulation (earthquake → drone → damage → rescuers)
python3 simulation/run_full_demo.py
python3 simulation/run_full_demo.py --fast   # 3x speed
```

---

## Individual Components

### Earthquake Detection (Sensor + LSTM)

```bash
# Real-time with visualization (posts to blackboard + dispatches drone)
python3 detection/realtime_visualizer.py --port /dev/tty.usbmodemXXXX

# Real-time CLI only
python3 detection/realtime_predict.py --port /dev/tty.usbmodemXXXX

# Test on CSV data
python3 detection/realtime_predict.py --csv detection/accelerometer_data.csv

# Test model accuracy
python3 models/earthquake_simulator/test_model.py
```

### Drone

```bash
# Listener — polls blackboard, flies when dispatched
python3 drone/listener.py

# Manual dispatch
python3 drone/dispatch.py --lat 34.765 --lon 32.420 --pga 0.15

# Mock damage reports (without listener)
python3 drone/mock_damage.py --count 5
python3 drone/mock_damage.py --loop 10 --count 2
```

### Rescuers

```bash
# Dismiss one zone by eventId (visible in map popup)
python3 rescuers/dismiss_zone.py --id <eventId>

# Dismiss ALL zones
python3 rescuers/dismiss_zone.py --all
```

### Web Map

```bash
python3 web/server.py
# Open http://localhost:8080
```

### Model Training

```bash
# Train on real datasets (LEN-DB + Central Italy)
python3 models/train_lstm_real.py

# Train on synthetic data only
python3 models/train_lstm_csv.py
```

---

## Data Flow

```
Sensor (MPU6500)
  → LSTM detection
    → POST /blackboard (earthquake_sensor)
      → POST /blackboard (drone_dispatch)
        → Drone flies, scans
          → POST /blackboard (damage_zone)
            → Leaflet map shows red zones
              → Rescuers resolve
                → POST /blackboard (dismiss_zone)
                  → Map removes resolved zones
```

## Blackboard Message Types

| Type | Agent | Description |
|------|-------|-------------|
| `earthquake_sensor` | `earthquake` | LSTM detection result |
| `drone_dispatch` | `earthquake` | Command for drone to fly out |
| `drone_status` | `drone_agent` | Drone status (takeoff/scanning/landed) |
| `damage_zone` | `drone_agent` | Damage found at coordinates |
| `dismiss_zone` | `rescue_control` | Rescuers resolved this zone |
| `dismiss_all_zones` | `rescue_control` | Clear all zones from map |
