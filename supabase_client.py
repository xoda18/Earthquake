"""
supabase_client.py
Shared Supabase REST client — used by sensor, drone, and swarm tools.

Uses plain requests (no extra dependencies).

Requires environment variables:
    SUPABASE_URL  — e.g. https://xxxx.supabase.co
    SUPABASE_KEY  — anon or service_role key
"""

import os
import requests

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation",
}


def _check_config():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("WARNING: SUPABASE_URL or SUPABASE_KEY not set. Export them first.")
        return False
    return True


def insert(table: str, row: dict) -> bool:
    """Insert a row into a Supabase table. Returns True on success."""
    if not _check_config():
        return False
    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers=_HEADERS,
            json=row,
            timeout=5,
        )
        if resp.status_code in (200, 201):
            return True
        print(f"Supabase insert error: {resp.status_code} {resp.text[:200]}")
        return False
    except Exception as e:
        print(f"Supabase error: {e}")
        return False


def select(table: str, order_by: str = "created_at", limit: int = 10, since_epoch: float = 0) -> list:
    """Select recent rows from a Supabase table. Returns list of dicts."""
    if not _check_config():
        return []
    try:
        params = {
            "order": f"{order_by}.desc",
            "limit": str(limit),
        }
        if since_epoch > 0:
            params["epoch"] = f"gt.{since_epoch}"

        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers={**_HEADERS, "Prefer": ""},
            params=params,
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json()
        print(f"Supabase select error: {resp.status_code} {resp.text[:200]}")
        return []
    except Exception as e:
        print(f"Supabase error: {e}")
        return []
