# Earthquake Detection System

Real-time earthquake detection using MPU6500 accelerometer + Arduino + LSTM neural network in Docker.

# For me

```bash
docker run --rm --device /dev/ttyACM0 earthquake-detector --profile table

#visualizing 

● Одна программа, всё параллельно внутри:

  cd ~/Desktop/NUP/JASS/Earthquake && source ../venv/bin/activate

  # Всё вместе: данные + детекция + blackboard + графики
  python3 detection/streaming/stream.py --detect --blackboard --viz

  # Без графиков
  python3 detection/streaming/stream.py --detect --blackboard

  # Только данные
  python3 detection/streaming/stream.py

  # Данные + графики (без blackboard)
  python3 detection/streaming/stream.py --viz

  # С сохранением в CSV
  python3 detection/streaming/stream.py --detect --blackboard --viz --save
   data.csv

  Флаги:
  - --detect — STA/LTA детекция
  - --blackboard — отправка на blackboard
  - --viz — живые графики (убрать флаг = без графиков)
  - --save FILE — сохранить CSV
  - --mode — earthquake / table_knock / optimized

✻ Churned for 1m 20s


```

## Setup

```bash
git clone https://github.com/JASS-2026-Cyprus/Earthquake.git
cd Earthquake
./setup.sh
```

This installs arduino-cli, uploads firmware, and builds the Docker image. Needs: Docker, Arduino Uno + MPU6500 sensor connected via USB.

## Run

```bash
# Table demo (recommended)
docker run --rm --device /dev/ttyACM0 earthquake-detector --profile table

# Default
docker run --rm --device /dev/ttyACM0 earthquake-detector

# Silent — only prints when earthquake detected
docker run --rm --device /dev/ttyACM0 earthquake-detector --profile table --mode magnitude
```


## Profiles

| Profile | Use case | Gain | Threshold |
|---------|----------|------|-----------|
| `default` | Balanced | 1x | 0.7 |
| `table` | Demo on table — knocks detected | 3x | 0.8 |
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
MPU6500 → Arduino (100Hz serial) → Docker (TensorFlow LSTM) → earthquake/quiet
```

1. Sensor reads acceleration at 100 Hz
2. Docker container calibrates gravity, removes it from signal
3. Gain amplifies the signal (profile-dependent)
4. LSTM classifies 1-second windows → probability 0–100%
5. If prob > threshold → EARTHQUAKE

## Docs

- [Streaming & Docker details](detection/streaming/README.md)
- [Troubleshooting](detection/streaming/TROUBLESHOOTING.md)
- [System prompt for ORB](earthquake_agent_system_prompt.md)

## If Arduino disconnects

```bash
./detection/streaming/setup.sh
```
