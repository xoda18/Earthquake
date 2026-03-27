"""
Web dashboard server — serves index.html + proxies Supabase.

All data comes from Supabase:
  /api/live    — latest sensor stream (from sensor_stream table)
  /api/sensor  — earthquake detections (from sensor_events table)
  /api/drone   — damage reports (from drone_reports table)

Usage:
    python web/server.py
    # Open http://localhost:8080
"""

import http.server
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import supabase_client as sb

PORT = 8080
WEB_DIR = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        if self.path == "/api/live":
            # Read from local file first (fast, ~1ms)
            live_file = os.path.join(WEB_DIR, "..", "detection", "sensor_live.json")
            try:
                with open(live_file, "r") as f:
                    self.raw_json(f.read())
                return
            except FileNotFoundError:
                pass
            # Fallback to Supabase (slow, ~300ms)
            rows = sb.select("sensor_stream", order_by="created_at", limit=1)
            if rows and rows[0].get("data"):
                data = rows[0]["data"]
                self.raw_json(data if isinstance(data, str) else json.dumps(data))
            else:
                self.json_response({"error": "no live data"})
        elif self.path == "/api/sensor":
            self.json_response(sb.select("sensor_events", limit=50))
        elif self.path == "/api/drone":
            self.json_response(sb.select("drone_reports", limit=50))
        else:
            super().do_GET()

    def do_DELETE(self):
        # DELETE /api/drone?id=xxx
        if self.path.startswith("/api/drone?id="):
            rid = self.path.split("id=")[1]
            ok = sb.delete("drone_reports", "event_id", rid)
            if not ok:
                ok = sb.delete("drone_reports", "id", rid)
            self.json_response({"ok": ok})
        else:
            self.json_response({"error": "unknown"})

    def raw_json(self, text):
        body = text.encode() if isinstance(text, str) else json.dumps(text).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def json_response(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    print(f"Dashboard: http://localhost:{PORT}")
    print("Data source: Supabase")
    server = http.server.HTTPServer(("", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
