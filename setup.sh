#!/bin/bash
# setup.sh — Full setup for Earthquake Detection System
#
# Installs everything needed, uploads Arduino firmware, builds Docker image.
# Run this once on a fresh machine, or after cloning the repo.
#
# Requirements: Linux, USB port, Arduino Uno + MPU6500 sensor connected
#
# Usage:
#   ./setup.sh              # full setup
#   ./setup.sh --skip-arduino   # skip Arduino upload (no hardware connected)
#   ./setup.sh --docker-only    # only rebuild Docker image

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SKIP_ARDUINO=false
DOCKER_ONLY=false

for arg in "$@"; do
    case $arg in
        --skip-arduino) SKIP_ARDUINO=true ;;
        --docker-only)  DOCKER_ONLY=true ;;
    esac
done

echo "============================================"
echo "  JASS Earthquake Detection — Full Setup"
echo "============================================"
echo ""

# ── Step 1: Check prerequisites ──────────────────────────────────────────────

echo "[1/5] Checking prerequisites..."

if ! command -v docker &>/dev/null; then
    echo "ERROR: Docker not installed."
    echo "  Install: https://docs.docker.com/engine/install/"
    echo "  Or: sudo pacman -S docker   (Arch/Manjaro)"
    echo "      sudo apt install docker.io   (Ubuntu/Debian)"
    exit 1
fi
echo "  Docker: $(docker --version | head -1)"

if ! command -v arduino-cli &>/dev/null; then
    echo "  arduino-cli: NOT FOUND"
    if [ "$SKIP_ARDUINO" = false ] && [ "$DOCKER_ONLY" = false ]; then
        echo "  Installing arduino-cli..."
        if command -v pacman &>/dev/null; then
            sudo pacman -S --noconfirm arduino
        elif command -v apt &>/dev/null; then
            curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh
            export PATH="$PATH:$HOME/bin"
        else
            echo "  ERROR: Cannot auto-install. See https://arduino.github.io/arduino-cli/installation/"
            exit 1
        fi
    fi
else
    echo "  arduino-cli: $(arduino-cli version | head -1)"
fi

echo "  OK"
echo ""

# ── Step 2: Arduino core + libraries ─────────────────────────────────────────

if [ "$DOCKER_ONLY" = false ] && [ "$SKIP_ARDUINO" = false ]; then
    echo "[2/5] Installing Arduino core and libraries..."
    arduino-cli core install arduino:avr 2>/dev/null || true
    echo "  OK"
    echo ""
else
    echo "[2/5] Skipping Arduino libraries"
    echo ""
fi

# ── Step 3: Upload firmware to Arduino ───────────────────────────────────────

if [ "$DOCKER_ONLY" = false ] && [ "$SKIP_ARDUINO" = false ]; then
    echo "[3/5] Uploading firmware to Arduino..."

    PORT=""
    for p in /dev/ttyACM* /dev/ttyUSB*; do
        if [ -e "$p" ]; then
            PORT="$p"
            break
        fi
    done

    if [ -z "$PORT" ]; then
        echo "  WARNING: No Arduino found on USB. Skipping upload."
        echo "  Connect Arduino and run: ./detection/streaming/setup.sh"
    else
        echo "  Found: $PORT"

        # Fix permissions
        if [ ! -w "$PORT" ]; then
            echo "  Fixing permissions (needs sudo)..."
            sudo chmod 666 "$PORT"
        fi

        # Compile and upload
        SKETCH_DIR="/tmp/eq_sketch_$$"
        mkdir -p "$SKETCH_DIR"
        cp detection/streaming/sketch.ino "$SKETCH_DIR/$(basename $SKETCH_DIR).ino"
        arduino-cli compile --fqbn arduino:avr:uno "$SKETCH_DIR" 2>&1 | tail -1
        arduino-cli upload -p "$PORT" --fqbn arduino:avr:uno "$SKETCH_DIR" 2>&1 | tail -1
        rm -rf "$SKETCH_DIR"

        # Verify
        sleep 3
        stty -F "$PORT" 115200 raw -echo 2>/dev/null || true
        DATA=$(timeout 4 cat "$PORT" 2>/dev/null | head -3)
        if echo "$DATA" | grep -qE "^[0-9]+,"; then
            echo "  Sensor data verified!"
        else
            echo "  WARNING: No data from sensor. Check wiring."
            echo "  See: detection/streaming/TROUBLESHOOTING.md"
        fi
    fi
    echo ""
else
    echo "[3/5] Skipping Arduino upload"
    echo ""
fi

# ── Step 4: Build Docker image ───────────────────────────────────────────────

echo "[4/5] Building Docker image (this takes ~2 min first time)..."
docker build -f detection/streaming/Dockerfile -t earthquake-detector . 2>&1 | tail -3
echo "  OK"
echo ""

# ── Step 5: Verify ───────────────────────────────────────────────────────────

echo "[5/5] Verifying..."
docker image inspect earthquake-detector > /dev/null 2>&1
echo "  Docker image: earthquake-detector OK"

if [ -e "$PORT" ]; then
    echo "  Arduino: $PORT OK"
fi

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Run earthquake detection:"
echo ""
echo "  # Default mode"
echo "  docker run --rm --device /dev/ttyACM0 earthquake-detector"
echo ""
echo "  # Table demo (recommended)"
echo "  docker run --rm --device /dev/ttyACM0 earthquake-detector --profile table"
echo ""
echo "  # Silent — only prints earthquakes"
echo "  docker run --rm --device /dev/ttyACM0 earthquake-detector --profile table --mode magnitude"
echo ""
echo "Profiles: default, table, sensitive, earthquake"
echo "See: detection/streaming/README.md for full docs"
echo ""
echo "If Arduino disconnected, reconnect and run:"
echo "  ./detection/streaming/setup.sh"
