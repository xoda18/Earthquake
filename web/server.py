"""
Web server: serves index.html + proxies blackboard API (bypasses CORS/SSL).

Usage:
    python web/server.py
    # Open http://localhost:8080
"""

import http.server
import json
import os
import requests as req

PORT = 8080
BLACKBOARD_URL = "https://blackboard.jass.school/blackboard"
WEB_DIR = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        if self.path == "/api/blackboard":
            self.proxy_blackboard()
        else:
            super().do_GET()

    def proxy_blackboard(self):
        try:
            resp = req.get(BLACKBOARD_URL, timeout=5, verify=False)
            self.send_response(resp.status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(resp.content)
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def log_message(self, format, *args):
        # Suppress noisy logs, only show errors
        if "502" in str(args):
            super().log_message(format, *args)


if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    print(f"Serving on http://localhost:{PORT}")
    print(f"Proxying {BLACKBOARD_URL} → /api/blackboard")
    server = http.server.HTTPServer(("", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
