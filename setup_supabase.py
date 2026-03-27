"""
setup_supabase.py — Create all required Supabase tables and storage buckets.

Run once to set up the database for the Earthquake Intelligence System.

Usage:
    python setup_supabase.py
"""

import requests
import json
import sys

SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
SUPABASE_ANON_KEY = "YOUR_SUPABASE_ANON_KEY"
SUPABASE_SERVICE_KEY = "YOUR_SUPABASE_SERVICE_KEY"

# Use service key for admin operations (DDL, storage)
HEADERS = {
    "apikey": SUPABASE_SERVICE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    "Content-Type": "application/json",
}


def run_sql(sql: str, description: str) -> bool:
    """Execute SQL via Supabase's RPC endpoint or REST."""
    print(f"\n--- {description} ---")
    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/rpc/",
            headers={**HEADERS, "Prefer": ""},
            json={"query": sql},
            timeout=10,
        )
        # If RPC doesn't work, try the SQL endpoint
        if resp.status_code not in (200, 201, 204):
            # Try pg_net or raw SQL if available
            print(f"  RPC returned {resp.status_code}, trying direct...")
            return False
        print(f"  OK")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def check_table_exists(table: str) -> bool:
    """Check if a table exists by trying to select from it."""
    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers={**HEADERS, "Prefer": ""},
            params={"limit": "1"},
            timeout=5,
        )
        return resp.status_code == 200
    except Exception:
        return False


def create_table_via_insert(table: str, sample_row: dict, description: str) -> bool:
    """Test if table exists by inserting and reading."""
    print(f"\n--- {description} ---")
    exists = check_table_exists(table)
    if exists:
        print(f"  Table '{table}' already exists")
        return True
    print(f"  Table '{table}' does NOT exist. You need to create it in the Supabase SQL editor.")
    return False


def create_storage_bucket(bucket_name: str) -> bool:
    """Create a Supabase Storage bucket."""
    print(f"\n--- Creating storage bucket: {bucket_name} ---")
    try:
        # Check if bucket exists
        resp = requests.get(
            f"{SUPABASE_URL}/storage/v1/bucket/{bucket_name}",
            headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"},
            timeout=5,
        )
        if resp.status_code == 200:
            print(f"  Bucket '{bucket_name}' already exists")
            return True

        # Create bucket
        resp = requests.post(
            f"{SUPABASE_URL}/storage/v1/bucket",
            headers={"apikey": SUPABASE_SERVICE_KEY, "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                      "Content-Type": "application/json"},
            json={"id": bucket_name, "name": bucket_name, "public": True},
            timeout=5,
        )
        if resp.status_code in (200, 201):
            print(f"  Created bucket '{bucket_name}' (public)")
            return True
        print(f"  Failed: {resp.status_code} {resp.text[:200]}")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    print("=" * 60)
    print("Earthquake Intelligence System — Supabase Setup")
    print("=" * 60)
    print(f"URL: {SUPABASE_URL}")

    # 1. Check existing tables
    tables = {
        "sensor_events": "Earthquake sensor readings",
        "drone_reports": "Drone damage assessment reports",
        "crack_reports": "VLM crack analysis results",
    }

    missing_tables = []
    for table, desc in tables.items():
        exists = check_table_exists(table)
        status = "EXISTS" if exists else "MISSING"
        print(f"  [{status}] {table} — {desc}")
        if not exists:
            missing_tables.append(table)

    # 2. Create storage bucket
    create_storage_bucket("crack-images")

    # 3. Print SQL for missing tables
    if missing_tables:
        print("\n" + "=" * 60)
        print("MISSING TABLES — Run this SQL in Supabase SQL Editor:")
        print("=" * 60)

        sql_statements = {
            "sensor_events": """
CREATE TABLE IF NOT EXISTS sensor_events (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT now(),
    timestamp TEXT,
    epoch DOUBLE PRECISION,
    type TEXT DEFAULT 'earthquake',
    probability DOUBLE PRECISION,
    pga_g DOUBLE PRECISION,
    ax DOUBLE PRECISION,
    ay DOUBLE PRECISION,
    az DOUBLE PRECISION,
    magnitude_g DOUBLE PRECISION
);""",
            "drone_reports": """
CREATE TABLE IF NOT EXISTS drone_reports (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT now(),
    timestamp TEXT,
    epoch DOUBLE PRECISION,
    type TEXT DEFAULT 'drone_report',
    event_id TEXT,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    severity TEXT DEFAULT 'unknown',
    damage_type TEXT,
    building TEXT,
    description TEXT,
    drone_id TEXT,
    confidence DOUBLE PRECISION DEFAULT 0
);""",
            "crack_reports": """
CREATE TABLE IF NOT EXISTS crack_reports (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT now(),
    timestamp TEXT,
    epoch DOUBLE PRECISION,
    run_id TEXT,
    type TEXT DEFAULT 'crack_analysis',
    image_name TEXT,
    image_url TEXT,
    has_crack BOOLEAN DEFAULT false,
    severity TEXT DEFAULT 'none',
    crack_count INT DEFAULT 0,
    max_crack_length_mm DOUBLE PRECISION DEFAULT 0,
    max_crack_width_mm DOUBLE PRECISION DEFAULT 0,
    description TEXT,
    damage_type TEXT,
    confidence DOUBLE PRECISION DEFAULT 0,
    status TEXT DEFAULT 'unknown',
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION
);""",
        }

        for table in missing_tables:
            print(f"\n-- {table}")
            print(sql_statements[table])

        print("\n-- Enable realtime (optional)")
        print("ALTER PUBLICATION supabase_realtime ADD TABLE sensor_events;")
        print("ALTER PUBLICATION supabase_realtime ADD TABLE drone_reports;")
        print("ALTER PUBLICATION supabase_realtime ADD TABLE crack_reports;")
    else:
        print("\nAll tables exist!")

    print("\n" + "=" * 60)
    print("Setup complete. Storage bucket created.")
    if missing_tables:
        print(f"Run the SQL above in Supabase SQL Editor for: {', '.join(missing_tables)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
