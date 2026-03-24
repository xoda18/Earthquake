#!/bin/bash
# run_all.sh — Run everything in parallel: streaming + LSTM + visualizer
#
# Usage:
#   ./detection/streaming/run_all.sh                # stream + Docker LSTM
#   ./detection/streaming/run_all.sh --visualize    # + live graphs
#
# Ctrl+C stops all processes.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CSV_FILE="/tmp/sensor_live.csv"
PROFILE="${PROFILE:-table}"
VISUALIZE=false

for arg in "$@"; do
    case $arg in
        --visualize) VISUALIZE=true ;;
    esac
done

cd "$REPO_DIR"
source ../venv/bin/activate 2>/dev/null || true

echo "=== Earthquake Detection System ==="
echo "  CSV:       $CSV_FILE"
echo "  Profile:   $PROFILE"
echo "  Visualize: $VISUALIZE"
echo ""

cleanup() {
    echo ""
    echo "Stopping all processes..."
    [ -n "$STREAM_PID" ] && kill "$STREAM_PID" 2>/dev/null
    [ -n "$DOCKER_PID" ] && kill "$DOCKER_PID" 2>/dev/null
    [ -n "$VIS_PID" ] && kill "$VIS_PID" 2>/dev/null
    [ -n "$TAIL_PID" ] && kill "$TAIL_PID" 2>/dev/null
    wait 2>/dev/null
    echo "Done."
    exit 0
}
trap cleanup EXIT INT TERM

# 1. Stream sensor data to CSV
echo "[1/3] Starting sensor → $CSV_FILE"
python3 detection/streaming/stream.py --save "$CSV_FILE" --rate 5 &
STREAM_PID=$!
sleep 2

# 2. Docker LSTM reads from tail -f (no serial access needed)
echo "[2/3] Starting Docker LSTM (profile=$PROFILE)"
tail -f "$CSV_FILE" | docker run --rm -i earthquake-detector --stdin --profile "$PROFILE" &
DOCKER_PID=$!

# 3. Visualizer reads same CSV
if [ "$VISUALIZE" = true ]; then
    echo "[3/3] Starting visualizer"
    (cd detection && python3 realtime_visualizer.py --from-csv "$CSV_FILE" --duration 600 --mode table_knock) &
    VIS_PID=$!
else
    echo "[3/3] Visualizer: off (add --visualize to enable)"
fi

echo ""
echo "Running. Ctrl+C to stop."
echo ""

wait $STREAM_PID
