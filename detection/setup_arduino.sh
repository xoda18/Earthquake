#!/bin/bash
# setup_arduino.sh — One-command Arduino MPU6500 setup
# Compiles, uploads sketch, fixes permissions, and verifies sensor data.
#
# Usage: ./demo/setup_arduino.sh [port]
#   port defaults to /dev/ttyACM0

set -e

PORT="${1:-/dev/ttyACM0}"
SKETCH_DIR="/tmp/mpu6500_sketch_$$"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INO_FILE="$SCRIPT_DIR/hardware/arduino_mpu6050.ino"

echo "=== Arduino MPU6500 Setup ==="
echo ""

# Step 1: Check Arduino is connected
echo "[1/5] Checking serial port..."
if [ ! -e "$PORT" ]; then
    echo "ERROR: $PORT not found. Is the Arduino plugged in via USB?"
    echo "Available ports:"
    ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || echo "  (none found)"
    exit 1
fi
echo "  Found: $PORT"

# Step 2: Fix permissions
echo "[2/5] Fixing serial port permissions..."
if [ ! -r "$PORT" ] || [ ! -w "$PORT" ]; then
    echo "  Need sudo to set permissions on $PORT"
    sudo chmod 666 "$PORT"
fi
echo "  Permissions OK"

# Step 3: Compile sketch
echo "[3/5] Compiling sketch..."
mkdir -p "$SKETCH_DIR"
cp "$INO_FILE" "$SKETCH_DIR/$(basename "$SKETCH_DIR").ino"
arduino-cli compile --fqbn arduino:avr:uno "$SKETCH_DIR" 2>&1 | tail -2
echo "  Compiled OK"

# Step 4: Upload
echo "[4/5] Uploading to Arduino..."
arduino-cli upload -p "$PORT" --fqbn arduino:avr:uno "$SKETCH_DIR" 2>&1 | tail -1
echo "  Uploaded OK"

# Step 5: Verify data
echo "[5/5] Verifying sensor data (5 seconds)..."
sleep 3
stty -F "$PORT" 115200 raw -echo
LINES=$(timeout 5 cat "$PORT" 2>/dev/null | head -20)

if echo "$LINES" | grep -q "connection failed"; then
    echo ""
    echo "ERROR: Sensor not responding! Check wiring:"
    echo "  VCC → 3.3V"
    echo "  GND → GND"
    echo "  SDA → A4"
    echo "  SCL → A5"
    rm -rf "$SKETCH_DIR"
    exit 1
fi

SAMPLE_COUNT=$(echo "$LINES" | grep -c ",")
if [ "$SAMPLE_COUNT" -lt 3 ]; then
    echo "ERROR: No data received. Try unplugging and replugging USB."
    rm -rf "$SKETCH_DIR"
    exit 1
fi

echo ""
echo "  Sample data:"
echo "$LINES" | head -5
echo "  ..."

# Cleanup
rm -rf "$SKETCH_DIR"

echo ""
echo "=== SUCCESS ==="
echo ""
echo "Arduino is streaming data on $PORT at 115200 baud."
echo ""
echo "Run the live monitor:"
echo "  python3 demo/live_sensor.py --port $PORT --save sensor_data.csv"
echo ""
echo "Analyze saved data:"
echo "  python3 demo/detect_earthquake.py sensor_data.csv"
