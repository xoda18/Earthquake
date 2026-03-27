#!/usr/bin/env bash
set -e

# ==========================================================================
#   ./launch.sh misha                       — VLM with NVIDIA GPU
#   ./launch.sh alex                        — VLM on Mac MPS GPU (native) + rest in Docker
#   ./launch.sh valik --vlm-ip <MISHA_IP>  — everything in Docker, VLM on Misha's GPU
#   ./launch.sh all                         — everything in Docker (CPU, no GPU)
# ==========================================================================

DIR="$(cd "$(dirname "$0")" && pwd)"
ROLE="${1:-}"
shift 2>/dev/null || true

VLM_IP=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --vlm-ip) VLM_IP="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

case "$ROLE" in

misha)
    docker compose --profile misha up --build
    ;;

valik)
    if [ -z "$VLM_IP" ]; then
        echo "Usage: ./launch.sh valik --vlm-ip <MISHA_IP>"
        echo "Example: ./launch.sh valik --vlm-ip 192.168.1.125"
        exit 1
    fi

    echo "=== Valik → Misha's VLM at ${VLM_IP}:5060 ==="
    echo ""

    echo "Checking VLM connection..."
    if curl -s --connect-timeout 3 "http://${VLM_IP}:5060/health" > /dev/null 2>&1; then
        echo "VLM is reachable"
    else
        echo "WARNING: Cannot reach VLM at ${VLM_IP}:5060 — make sure Misha ran ./launch.sh misha"
    fi
    echo ""

    VLM_URL="http://${VLM_IP}:5060" docker compose --profile valik up --build
    ;;

alex)
    echo "=== Alex's Mac setup (local VLM with MPS GPU) ==="
    echo ""

    if ! python3 -c "import fastapi" 2>/dev/null; then
        echo "Installing Python dependencies..."
        pip3 install fastapi "uvicorn[standard]" python-multipart pillow \
            torch transformers accelerate requests piexif
    fi

    echo "[1/2] Starting VLM server on port 5060 (MPS GPU)..."
    cd "$DIR/vlm"
    uvicorn server:app --host 0.0.0.0 --port 5060 &
    VLM_PID=$!
    cd "$DIR"

    echo "     Waiting for model to load..."
    for i in $(seq 1 120); do
        if curl -s http://localhost:5060/health > /dev/null 2>&1; then
            echo "     VLM ready!"
            break
        fi
        sleep 2
    done

    echo "[2/2] Starting Docker services..."
    VLM_URL="http://host.docker.internal:5060" docker compose --profile alex up --build

    kill $VLM_PID 2>/dev/null
    ;;

all)
    docker compose --profile all up --build
    ;;

*)
    echo "Usage:"
    echo "  ./launch.sh misha                       — VLM with NVIDIA GPU"
    echo "  ./launch.sh alex                        — VLM on Mac MPS (native) + rest in Docker"
    echo "  ./launch.sh valik --vlm-ip <MISHA_IP>  — everything in Docker, VLM on Misha's GPU"
    echo "  ./launch.sh all                         — everything in Docker (CPU, no GPU)"
    ;;
esac
