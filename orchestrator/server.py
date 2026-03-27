"""
orchestrator/server.py — Pipeline orchestrator for the Earthquake Intelligence System.

Receives POST /step/done from each container when it finishes,
then launches the next Docker container in the pipeline.

Pipeline:
  earthquake_detected → drone_scan → image_crop → vlm_analysis → db_cleanup

Usage:
    uvicorn server:app --host 0.0.0.0 --port 5050
"""

import logging
import os
import time
from datetime import datetime, timezone

import docker
import requests as http_client
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("orchestrator")

app = FastAPI(title="Earthquake Pipeline Orchestrator")
client = docker.from_env()

# ---------------------------------------------------------------------------
# Pipeline definition: step → next step
# ---------------------------------------------------------------------------
PIPELINE = {
    "earthquake_detected": "drone_scan",
    "drone_scan":          "image_crop",
    "image_crop":          "vlm_analysis",
    "vlm_analysis":        "db_cleanup",
    "db_cleanup":          None,
}

# Docker image + run config for each step
STEP_CONFIG = {
    "drone_scan": {
        "image": "drone-scanner",
        "command": ["python", "wall_scan.py"],
        "environment": {
            "ORCHESTRATOR_URL": os.getenv("ORCHESTRATOR_URL", "http://orchestrator:5050"),
        },
        "network_mode": "host",
    },
    "image_crop": {
        "image": "image-cropper",
        "command": ["python", "crop.py"],
        "environment": {
            "ORCHESTRATOR_URL": os.getenv("ORCHESTRATOR_URL", "http://orchestrator:5050"),
        },
        "volumes": {
            os.getenv("IMAGE_CROP_INPUT", "./crack_detection/image_diff/input"): {
                "bind": "/app/input", "mode": "rw",
            },
            os.getenv("IMAGE_CROP_OUTPUT", "./crack_detection/image_diff/output_cropped"): {
                "bind": "/app/output", "mode": "rw",
            },
        },
    },
    "vlm_analysis": {
        # VLM runs as a long-lived server — not launched as a container.
        # Instead, the orchestrator sends an HTTP POST to trigger analysis.
        "mode": "http",
        "url": os.getenv("VLM_URL", "http://vlm-analyzer:5060") + "/run",
    },
    "db_cleanup": {
        "image": "db-cleanup",
        "command": ["python", "cleanup.py"],
        "environment": {
            "ORCHESTRATOR_URL": os.getenv("ORCHESTRATOR_URL", "http://orchestrator:5050"),
        },
    },
}

# ---------------------------------------------------------------------------
# In-memory pipeline state
# ---------------------------------------------------------------------------
pipeline_runs: list[dict] = []


class StepDone(BaseModel):
    step: str
    status: str = "success"
    detail: str = ""
    run_id: str | None = None


class TriggerRequest(BaseModel):
    step: str
    run_id: str | None = None


# ---------------------------------------------------------------------------
# Container launcher
# ---------------------------------------------------------------------------
def launch_step(step: str, run_id: str) -> str | None:
    """Launch the next pipeline step. Either starts a container or makes an HTTP call.

    Returns container ID (for container steps) or "http_ok" (for HTTP steps).
    """
    cfg = STEP_CONFIG.get(step)
    if cfg is None:
        log.warning(f"No config for step '{step}', skipping.")
        return None

    # HTTP mode: call an already-running service instead of launching a container
    if cfg.get("mode") == "http":
        return _trigger_http_step(step, run_id, cfg["url"])

    return _launch_container(step, run_id, cfg)


def _trigger_http_step(step: str, run_id: str, url: str) -> str | None:
    """Send an HTTP POST to trigger a step on an already-running service."""
    try:
        resp = http_client.post(url, json={"run_id": run_id}, timeout=10)
        resp.raise_for_status()
        log.info(f"Triggered [{step}] via HTTP {url} run={run_id}")
        return "http_ok"
    except Exception as e:
        log.error(f"HTTP trigger failed [{step}] {url}: {e}")
        return None


def _launch_container(step: str, run_id: str, cfg: dict) -> str | None:
    """Launch a Docker container for the given step. Returns container ID."""
    env = {**(cfg.get("environment") or {}), "RUN_ID": run_id, "STEP_NAME": step}
    kwargs = {
        "image": cfg["image"],
        "command": cfg.get("command"),
        "environment": env,
        "detach": True,
        "remove": True,
        "name": f"{step}_{run_id}",
    }

    if cfg.get("network_mode"):
        kwargs["network_mode"] = cfg["network_mode"]
    if cfg.get("volumes"):
        kwargs["volumes"] = cfg["volumes"]
    if cfg.get("devices"):
        kwargs["devices"] = cfg["devices"]

    try:
        container = client.containers.run(**kwargs)
        log.info(f"Launched [{step}] container={container.short_id} run={run_id}")
        return container.short_id
    except docker.errors.ImageNotFound:
        log.error(f"Image '{cfg['image']}' not found. Build it first.")
        return None
    except Exception as e:
        log.error(f"Failed to launch [{step}]: {e}")
        return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.post("/pipeline/trigger")
def trigger_pipeline(req: TriggerRequest):
    """Manually trigger a specific pipeline step (e.g., from UI button)."""
    if req.step not in STEP_CONFIG:
        raise HTTPException(400, f"Unknown step '{req.step}'. Valid: {list(STEP_CONFIG.keys())}")

    run_id = req.run_id or f"run_{int(time.time())}"
    pipeline_runs.append({
        "run_id": run_id,
        "started": datetime.now(timezone.utc).isoformat(),
        "trigger": req.step,
        "steps": [],
    })

    cid = launch_step(req.step, run_id)
    return {"run_id": run_id, "step": req.step, "container": cid}


@app.post("/step/done")
def step_done(body: StepDone):
    """Called by a container when it finishes its work."""
    step = body.step
    run_id = body.run_id or f"run_{int(time.time())}"

    log.info(f"Step done: [{step}] status={body.status} run={run_id} detail={body.detail}")

    # Record completion
    run = next((r for r in pipeline_runs if r["run_id"] == run_id), None)
    if run is None:
        run = {"run_id": run_id, "started": datetime.now(timezone.utc).isoformat(),
               "trigger": step, "steps": []}
        pipeline_runs.append(run)
    run["steps"].append({
        "step": step, "status": body.status, "detail": body.detail,
        "completed": datetime.now(timezone.utc).isoformat(),
    })

    # If failed, stop the pipeline
    if body.status != "success":
        log.warning(f"Step [{step}] failed — pipeline halted for run={run_id}")
        return {"next": None, "reason": "step_failed"}

    # Determine and launch next step
    next_step = PIPELINE.get(step)
    if next_step is None:
        log.info(f"Pipeline complete for run={run_id}")
        return {"next": None, "reason": "pipeline_complete"}

    cid = launch_step(next_step, run_id)
    return {"next": next_step, "container": cid}


@app.get("/pipeline/status")
def pipeline_status():
    """Show recent pipeline runs."""
    return {"runs": pipeline_runs[-20:]}


@app.get("/pipeline/steps")
def pipeline_steps():
    """Show the pipeline definition."""
    return {"pipeline": PIPELINE, "steps": list(STEP_CONFIG.keys())}
