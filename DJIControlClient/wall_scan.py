import cv2
import os
import sys
import time
import threading
from datetime import datetime

import requests
from DJIControlClient import DJIControlClient

ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "")

# ---- Configuration ----
DRONE_IP = "192.168.1.119"
DRONE_PORT = 8080

# MediaMTX RTSP stream (lowest latency for OpenCV)
# If RTSP doesn't work, try: "http://localhost:8888/live/webcam/index.m3u8"
STREAM_URL = "rtsp://localhost:8554/live/webcam"

RISE_HEIGHT = 2.0
MOVE_DISTANCE = 0.5       # metres to the right between shots
NUM_PHOTOS = 4
SETTLE_TIME = 5           # seconds after each drone movement
STREAM_CATCHUP_TIME = 25  # seconds to let the stream catch up to real-time


class LatestFrameCapture:
    """Continuously reads the stream in a background thread,
    always keeping only the latest frame. This way .read()
    never returns a stale buffered frame."""

    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open stream: {url}")

        self.lock = threading.Lock()
        self.frame = None
        self.ret = False
        self.running = True

        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

        # Wait until we get the first frame
        deadline = time.time() + 10
        while time.time() < deadline:
            with self.lock:
                if self.ret:
                    break
            time.sleep(0.1)
        else:
            self.release()
            raise RuntimeError("Stream opened but no frames received within 10s")

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=5)
        self.cap.release()


def capture_and_save(stream, path):
    """Grab the latest frame from the stream and save it."""
    ret, frame = stream.read()
    if not ret or frame is None:
        print(f"  ERROR: failed to capture frame")
        return False
    cv2.imwrite(path, frame)
    print(f"  Saved: {path}  ({frame.shape[1]}x{frame.shape[0]})")
    return True


def test_stream():
    """Quick test: connect to the stream, grab one frame, save it."""
    print(f"Connecting to stream: {STREAM_URL}")
    stream = LatestFrameCapture(STREAM_URL)
    print("Stream connected. Waiting 3s for a fresh frame...")
    time.sleep(3)

    os.makedirs("scans", exist_ok=True)
    path = os.path.join("scans", "test_frame.jpg")
    capture_and_save(stream, path)

    stream.release()
    print("Done. Check the saved image to verify quality.")


def scan_wall():
    """Full wall-scanning flight."""

    # Output folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join("scans", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Images will be saved to: {output_dir}")

    # Connect to video stream
    print(f"Connecting to stream: {STREAM_URL}")
    stream = LatestFrameCapture(STREAM_URL)
    print("Stream OK")

    # # Connect to drone
    # print(f"Connecting to drone at {DRONE_IP}:{DRONE_PORT}...")
    # client = DJIControlClient(DRONE_IP, DRONE_PORT)
    # print("Drone connected")

    try:
        # # Take off
        # print("\n[1] Taking off...")
        # client.takeOff()
        # time.sleep(10)

        # # Rise to scanning height
        # print(f"[2] Rising {RISE_HEIGHT}m...")
        # client.moveUp(RISE_HEIGHT)
        # time.sleep(SETTLE_TIME)

        # # Capture 4 images, moving right between each
        # for i in range(NUM_PHOTOS):
        #     print(f"\n--- Photo {i + 1}/{NUM_PHOTOS} ---")
        #
        #     print(f"  Moving right {MOVE_DISTANCE}m...")
        #     client.moveRight(MOVE_DISTANCE)
        #     time.sleep(SETTLE_TIME)
        #
        #     print(f"  Waiting {STREAM_CATCHUP_TIME}s for stream to catch up...")
        #     time.sleep(STREAM_CATCHUP_TIME)
        #
        #     photo_path = os.path.join(output_dir, f"wall_{i + 1}.jpg")
        #     capture_and_save(stream, photo_path)

        # Just capture one shot to test
        print("Waiting 3s for a fresh frame...")
        time.sleep(3)
        photo_path = os.path.join(output_dir, "test_shot.jpg")
        capture_and_save(stream, photo_path)

        # # Return to start position
        # total_distance = NUM_PHOTOS * MOVE_DISTANCE
        # print(f"\n[3] Returning: moving left {total_distance}m...")
        # client.moveLeft(total_distance)
        # time.sleep(SETTLE_TIME)

        # # Land
        # print("[4] Landing...")
        # client.land()
        # time.sleep(5)
        # client.confirmLanding()
        # print("Landed successfully")

    except Exception as e:
        print(f"\nERROR: {e}")
    finally:
        stream.release()

    print(f"\nDone! Image saved in: {output_dir}")

    # Notify orchestrator that drone scan is complete
    if ORCHESTRATOR_URL:
        try:
            requests.post(f"{ORCHESTRATOR_URL}/step/done", json={
                "step": "drone_scan",
                "status": "success",
                "detail": f"images_dir={output_dir}",
            }, timeout=5)
            print("[orchestrator] Notified: drone_scan done")
        except Exception as e:
            print(f"[orchestrator] Failed to notify: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test-stream":
        test_stream()
    else:
        scan_wall()
