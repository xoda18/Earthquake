"""
orchestrator/notify.py — Lightweight helper for containers to notify the orchestrator.

Each container calls `notify_done()` when finished. This POSTs to the
orchestrator, which then launches the next step in the pipeline.

Usage:
    from notify import notify_done
    # ... do work ...
    notify_done("image_diff", detail="processed 3 pairs")
"""

import os
import requests

ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:5050")


def notify_done(step: str, status: str = "success", detail: str = ""):
    """Notify the orchestrator that this step is complete."""
    run_id = os.getenv("RUN_ID", "")
    payload = {
        "step": step,
        "status": status,
        "detail": detail,
        "run_id": run_id or None,
    }
    url = f"{ORCHESTRATOR_URL}/step/done"
    try:
        resp = requests.post(url, json=payload, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        print(f"[orchestrator] step='{step}' done → next='{data.get('next')}'")
        return data
    except Exception as e:
        print(f"[orchestrator] Failed to notify: {e}")
        return None
