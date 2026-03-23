#!/bin/bash
# setup.sh — Upload sketch to Arduino and verify sensor data.
# Run this every time you reconnect the Arduino.
#
# Usage: ./demo/arduino-sensor/setup.sh [port]

set -e

PORT="${1:-/dev/ttyACM0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKETCH="$SCRIPT_DIR/sketch.ino"
TMP="/tmp/mpu_sketch_$$"

echo "=== Arduino MPU6500 Setup ==="
echo ""

# 1. Check port
echo "[1/4] Checking port $PORT..."
if [ ! -e "$PORT" ]; then
    echo "ERROR: $PORT not found."
    echo "Available ports:"
    ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || echo "  (none)"
    exit 1
fi
echo "  OK: $PORT found"

# 2. Fix permissions
echo "[2/4] Fixing permissions..."
sudo chmod 666 "$PORT"
echo "  OK"

# 3. Compile + upload
echo "[3/4] Uploading sketch..."
mkdir -p "$TMP"
cp "$SKETCH" "$TMP/$(basename $TMP).ino"
arduino-cli compile --fqbn arduino:avr:uno "$TMP" 2>&1 | tail -1
arduino-cli upload -p "$PORT" --fqbn arduino:avr:uno "$TMP" 2>&1 | tail -1
rm -rf "$TMP"
echo "  OK"

# 4. Verify
echo "[4/4] Verifying sensor..."
sleep 3
stty -F "$PORT" 115200 raw -echo
OUTPUT=$(timeout 4 cat "$PORT" 2>/dev/null | head -5)

if echo "$OUTPUT" | grep -q "connection failed\|WHO_AM_I=0xFF"; then
    echo ""
    echo "ERROR: Sensor not responding. Check wiring:"
    echo "  VCC → 3.3V  |  GND → GND  |  SDA → A4  |  SCL → A5"
    exit 1
fi

if echo "$OUTPUT" | grep -qE "^[0-9]+,"; then
    echo "  OK: Data flowing"
    echo ""
    echo "$OUTPUT" | head -3
else
    echo "  WARN: No data yet — try replugging USB"
fi

echo ""
echo "=== Done ==="
echo ""
echo "Stream data:"
echo "  source ../venv/bin/activate"
echo "  python3 demo/arduino-sensor/stream.py --save sensor_data.csv"
echo ""
echo "Analyze saved CSV:"
echo "  python3 demo/detect_earthquake.py sensor_data.csv"
