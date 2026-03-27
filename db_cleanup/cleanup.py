"""
db_cleanup/cleanup.py — Remove old drone images from Supabase, keeping only the two most recent versions.

After cleanup, notifies the orchestrator so the next pipeline step (image_diff) can run.

Usage:
    python cleanup.py
    python cleanup.py --keep 2 --table drone_images
"""

import argparse
import os
import sys

import requests

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://YOUR_PROJECT.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "YOUR_SUPABASE_SERVICE_KEY")

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation",
}


def get_all_images(table: str) -> list[dict]:
    """Fetch all image records, newest first."""
    resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers={**HEADERS, "Prefer": ""},
        params={"order": "created_at.desc", "limit": "1000"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def delete_rows(table: str, ids: list) -> int:
    """Delete rows by ID. Returns count deleted."""
    if not ids:
        return 0
    # Supabase REST: DELETE with ?id=in.(id1,id2,...)
    id_list = ",".join(str(i) for i in ids)
    resp = requests.delete(
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers=HEADERS,
        params={"id": f"in.({id_list})"},
        timeout=10,
    )
    resp.raise_for_status()
    return len(ids)


def cleanup_images(table: str, keep: int) -> dict:
    """Remove old images, keeping only the `keep` most recent versions per group."""
    rows = get_all_images(table)
    total = len(rows)

    if total <= keep:
        print(f"Only {total} image(s) found, nothing to delete (keep={keep}).")
        return {"total": total, "deleted": 0, "kept": total}

    # Keep the first `keep` rows (newest), delete the rest
    to_keep = rows[:keep]
    to_delete = rows[keep:]

    ids_to_delete = [r["id"] for r in to_delete if "id" in r]
    deleted = delete_rows(table, ids_to_delete)

    print(f"Cleanup: {total} total, kept {len(to_keep)}, deleted {deleted}")
    return {"total": total, "deleted": deleted, "kept": len(to_keep)}


def main():
    parser = argparse.ArgumentParser(description="Clean up old drone images from Supabase")
    parser.add_argument("--table", default="drone_images", help="Supabase table name")
    parser.add_argument("--keep", type=int, default=2, help="Number of recent versions to keep")
    args = parser.parse_args()

    print(f"DB Cleanup: table={args.table}, keep={args.keep}")
    result = cleanup_images(args.table, args.keep)

    # Notify orchestrator
    try:
        sys.path.insert(0, "/app")
        from notify import notify_done
        notify_done("db_cleanup", detail=f"deleted={result['deleted']} kept={result['kept']}")
    except Exception as e:
        print(f"Could not notify orchestrator: {e}")


if __name__ == "__main__":
    main()
