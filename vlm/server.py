"""
vlm/server.py — VLM crack analysis API server using LLaVA damage recognition.

Receives images via POST /analyze, runs them through the LLaVA-based
DamageAnalyzer, saves images to Supabase Storage, writes analysis to
crack_reports table, compares with previous run, and sends GPS alerts.

Always-running service (port 5060).

Usage:
    uvicorn server:app --host 0.0.0.0 --port 5060
"""

import glob
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from io import BytesIO

import requests
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from pydantic import BaseModel

# Add parent dir so we can import VLM_damage_recognition
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from VLM_damage_recognition import DamageAnalyzer, ImageProcessor, SupabaseReporter
from compare import compare_runs

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("vlm")

app = FastAPI(title="VLM Crack Analyzer")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://orchestrator:5050")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://YOUR_PROJECT.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "YOUR_SUPABASE_SERVICE_KEY")
STORAGE_BUCKET = os.environ.get("STORAGE_BUCKET", "crack-images")
ALERT_URL = os.environ.get("ALERT_URL", "https://blackboard.jass.school/blackboard")
AGENT_NAME = "CrackAnalyzer"
QUANTIZE = os.environ.get("QUANTIZE", "false").lower() == "true"
MODEL_ID = os.environ.get("MODEL_ID", "llava-hf/llava-1.5-7b-hf")

SUPABASE_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation",
}

# ---------------------------------------------------------------------------
# Load model at startup
# ---------------------------------------------------------------------------
analyzer: DamageAnalyzer | None = None
reporter: SupabaseReporter | None = None


@app.on_event("startup")
def load_model():
    global analyzer, reporter
    log.info(f"Loading LLaVA model: {MODEL_ID} (quantize={QUANTIZE})")
    analyzer = DamageAnalyzer(quantize=QUANTIZE, model_id=MODEL_ID)
    reporter = SupabaseReporter(use_supabase=True)
    log.info("Model loaded and ready.")


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------
def upload_image_to_storage(filename: str, image_bytes: bytes) -> str | None:
    """Upload image to Supabase Storage. Returns public URL."""
    path = f"{int(time.time())}_{filename}"
    try:
        resp = requests.post(
            f"{SUPABASE_URL}/storage/v1/object/{STORAGE_BUCKET}/{path}",
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "image/jpeg",
            },
            data=image_bytes,
            timeout=10,
        )
        if resp.status_code in (200, 201):
            url = f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/{path}"
            log.info(f"Uploaded {filename} → {url}")
            return url
        log.warning(f"Storage upload failed: {resp.status_code} {resp.text[:200]}")
        return None
    except Exception as e:
        log.warning(f"Storage upload error: {e}")
        return None


def save_crack_report(report: dict, run_id: str, image_url: str | None) -> bool:
    """Save VLM analysis result to crack_reports table."""
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "epoch": time.time(),
        "run_id": run_id,
        "type": "crack_analysis",
        "image_name": report.get("file", ""),
        "image_url": image_url,
        "has_crack": report.get("severity", "none") != "none",
        "severity": report.get("severity", "unknown"),
        "crack_count": len(report.get("cracks", [])),
        "max_crack_length_mm": max(
            (c.get("measurements", {}).get("length_mm", 0) for c in report.get("cracks", [])),
            default=0,
        ),
        "max_crack_width_mm": max(
            (c.get("measurements", {}).get("width_mm", 0) for c in report.get("cracks", [])),
            default=0,
        ),
        "description": report.get("description", ""),
        "damage_type": report.get("damage_type", ""),
        "confidence": report.get("confidence", 0),
        "status": report.get("status", "unknown"),
        "lat": report.get("lat", 0),
        "lon": report.get("lon", 0),
    }
    try:
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/crack_reports",
            headers=SUPABASE_HEADERS,
            json=row,
            timeout=5,
        )
        return resp.status_code in (200, 201)
    except Exception as e:
        log.warning(f"DB error: {e}")
        return False


def get_previous_run(current_run_id: str) -> list[dict]:
    """Fetch the most recent previous run's reports."""
    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/crack_reports",
            headers={**SUPABASE_HEADERS, "Prefer": ""},
            params={"order": "epoch.desc", "limit": "50", "run_id": f"neq.{current_run_id}"},
            timeout=5,
        )
        if resp.status_code != 200:
            return []
        rows = resp.json()
        if not rows:
            return []
        prev_run_id = rows[0].get("run_id")
        return [r for r in rows if r.get("run_id") == prev_run_id]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------
def send_alert(report: dict, comparison: dict | None):
    """Send GPS-tagged alert to blackboard."""
    content = {
        "image": report.get("file", ""),
        "severity": report.get("severity", "unknown"),
        "damage_type": report.get("damage_type", ""),
        "crack_count": len(report.get("cracks", [])),
        "description": report.get("description", ""),
        "status": report.get("status", "unknown"),
        "lat": report.get("lat", 0),
        "lon": report.get("lon", 0),
        "confidence": report.get("confidence", 0),
        "timestamp": time.time(),
    }
    if comparison:
        content["changes"] = comparison

    severity_confidence = {"low": 0.3, "moderate": 0.6, "high": 0.85, "critical": 0.95}

    try:
        requests.post(ALERT_URL, json={
            "agent": AGENT_NAME,
            "type": "crack_alert",
            "content": json.dumps(content),
            "confidence": severity_confidence.get(report.get("severity", ""), 0.5),
        }, timeout=5)
        log.info(f"Alert sent: {report.get('file')} severity={report.get('severity')} "
                 f"GPS=({report.get('lat')}, {report.get('lon')})")
    except Exception as e:
        log.warning(f"Alert failed: {e}")


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "model_loaded": analyzer is not None,
        "time": datetime.now(timezone.utc).isoformat(),
    }


class AnalyzeResponse(BaseModel):
    image_name: str
    image_url: str | None = None
    analysis: dict
    comparison: dict | None = None
    alert_sent: bool = False


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(image: UploadFile = File(...), run_id: str = ""):
    """Analyze a single image for structural damage.

    1. Upload image to Supabase Storage
    2. Run through LLaVA DamageAnalyzer
    3. Save report to Supabase (crack_reports + drone_reports)
    4. Compare with previous run
    5. Send GPS alert if damage found
    """
    image_bytes = await image.read()
    filename = image.filename or "unknown.jpg"
    run_id = run_id or f"run_{int(time.time())}"

    log.info(f"Analyzing {filename} (run={run_id}, {len(image_bytes)} bytes)")

    # 1. Upload to Supabase Storage
    image_url = upload_image_to_storage(filename, image_bytes)

    # 2. Run LLaVA analysis
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    lat, lon = ImageProcessor.generate_paphos_coordinates()
    report = analyzer.analyze_image(pil_image, lat, lon)
    report["file"] = filename

    # 3. Save to both tables
    save_crack_report(report, run_id, image_url)
    reporter.write_report(report)

    # 4. Compare with previous run
    prev_reports = get_previous_run(run_id)
    comparison = None
    if prev_reports:
        # Build a comparable dict from the VLM report
        current_comparable = [{
            "image_name": filename,
            "has_crack": report.get("severity", "none") != "none",
            "severity": report.get("severity", "unknown"),
            "crack_count": len(report.get("cracks", [])),
            "max_crack_length_mm": max(
                (c.get("measurements", {}).get("length_mm", 0) for c in report.get("cracks", [])),
                default=0,
            ),
            "max_crack_width_mm": max(
                (c.get("measurements", {}).get("width_mm", 0) for c in report.get("cracks", [])),
                default=0,
            ),
        }]
        comparison = compare_runs(current_comparable, prev_reports)

    # 5. Alert if damage found
    alert_sent = False
    if report.get("severity", "none") not in ("none", "unknown"):
        send_alert(report, comparison)
        alert_sent = True

    return AnalyzeResponse(
        image_name=filename,
        image_url=image_url,
        analysis=report,
        comparison=comparison,
        alert_sent=alert_sent,
    )


class BatchRequest(BaseModel):
    run_id: str = ""
    input_dir: str = "/app/input"


class BatchResponse(BaseModel):
    run_id: str
    images_analyzed: int
    cracks_found: int
    comparison: dict | None = None


@app.post("/run", response_model=BatchResponse)
async def run_batch(req: BatchRequest):
    """Process all images in a directory. Called by the orchestrator."""
    run_id = req.run_id or f"run_{int(time.time())}"
    input_dir = req.input_dir

    # Load images using the VLM module's processor
    image_list = ImageProcessor.batch_load_images(input_dir)
    log.info(f"Batch run: {len(image_list)} images in {input_dir} (run={run_id})")

    reports = analyzer.analyze_batch(image_list, drone_id="JASS-DRONE-01")

    # Save each report
    cracks_found = 0
    current_comparables = []
    for i, report in enumerate(reports):
        file_path = image_list[i][0]
        filename = os.path.basename(file_path)
        report["file"] = filename

        # Upload image to storage
        with open(file_path, "rb") as f:
            image_url = upload_image_to_storage(filename, f.read())

        # Save to DB
        save_crack_report(report, run_id, image_url)
        reporter.write_report(report)

        has_crack = report.get("severity", "none") not in ("none", "unknown")
        if has_crack:
            cracks_found += 1

        current_comparables.append({
            "image_name": filename,
            "has_crack": has_crack,
            "severity": report.get("severity", "unknown"),
            "crack_count": len(report.get("cracks", [])),
            "max_crack_length_mm": max(
                (c.get("measurements", {}).get("length_mm", 0) for c in report.get("cracks", [])),
                default=0,
            ),
            "max_crack_width_mm": max(
                (c.get("measurements", {}).get("width_mm", 0) for c in report.get("cracks", [])),
                default=0,
            ),
        })

    # Compare with previous run
    prev_reports = get_previous_run(run_id)
    comparison = None
    if prev_reports:
        comparison = compare_runs(current_comparables, prev_reports)
        log.info(f"Run comparison:\n{json.dumps(comparison, indent=2)}")

    # Send alerts
    for report in reports:
        if report.get("severity", "none") not in ("none", "unknown"):
            send_alert(report, comparison)

    # Notify orchestrator
    try:
        requests.post(f"{ORCHESTRATOR_URL}/step/done", json={
            "step": "vlm_analysis",
            "status": "success",
            "detail": f"analyzed={len(reports)} cracks={cracks_found}",
            "run_id": run_id,
        }, timeout=5)
        log.info("Notified orchestrator: vlm_analysis done")
    except Exception as e:
        log.warning(f"Failed to notify orchestrator: {e}")

    return BatchResponse(
        run_id=run_id,
        images_analyzed=len(reports),
        cracks_found=cracks_found,
        comparison=comparison,
    )
