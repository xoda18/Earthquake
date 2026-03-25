"""
Local web server — serves dashboard + proxies Supabase API.

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
        if self.path.startswith("/api/"):
            self.proxy_supabase()
        else:
            super().do_GET()

    def proxy_supabase(self):
        """Proxy /api/<table>?params → Supabase REST API."""
        try:
            # Parse path and query
            path = self.path[5:]  # strip /api/
            table = path.split("?")[0]
            query = path.split("?")[1] if "?" in path else ""

            # Parse query params
            params = {}
            if query:
                for pair in query.split("&"):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        params[k] = v

            import requests
            resp = requests.get(
                f"{sb.SUPABASE_URL}/rest/v1/{table}",
                headers={
                    "apikey": sb.SUPABASE_KEY,
                    "Authorization": f"Bearer {sb.SUPABASE_KEY}",
                },
                params=params,
                timeout=5,
            )

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
        if "502" in str(args) or "500" in str(args):
            super().log_message(format, *args)


if __name__ == "__main__":
    print(f"JASS Earthquake Monitor")
    print(f"http://localhost:{PORT}")
    print(f"Supabase: {sb.SUPABASE_URL}")
    server = http.server.HTTPServer(("", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
