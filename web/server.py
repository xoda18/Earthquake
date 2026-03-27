"""
Web dashboard server — serves index.html + proxies Supabase (bypasses CORS).

Usage:
    python web/server.py
    # Open http://localhost:8080
"""

import http.server
import json
import os
import ssl
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import supabase_client as sb

PORT = 8080
WEB_DIR = os.path.dirname(os.path.abspath(__file__))
LIVE_FILE = os.path.join(WEB_DIR, "..", "detection", "sensor_live.json")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        if self.path == "/api/live":
            self.serve_live()
        elif self.path == "/api/sensor":
            self.json_response(sb.select("sensor_events", limit=50))
        elif self.path == "/api/drone":
            self.json_response(sb.select("drone_reports", limit=50))
        elif self.path.startswith("/api/sensor?since="):
            epoch = float(self.path.split("since=")[1])
            self.json_response(sb.select("sensor_events", limit=50, since_epoch=epoch))
        elif self.path.startswith("/api/drone?since="):
            epoch = float(self.path.split("since=")[1])
            self.json_response(sb.select("drone_reports", limit=50, since_epoch=epoch))
        else:
            super().do_GET()

    def serve_live(self):
        try:
            with open(LIVE_FILE, "r") as f:
                body = f.read().encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self.json_response({"error": "sensor not running"})
        except Exception:
            self.json_response({"error": "read error"})

    def json_response(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # silent


if __name__ == "__main__":
    print(f"Dashboard: http://localhost:{PORT}")
    server = http.server.HTTPServer(("", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
