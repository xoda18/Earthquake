# Pipeline Orchestrator

A FastAPI server that chains the Earthquake Intelligence System's Docker containers into an automated pipeline. Each step POSTs to the orchestrator when it finishes, and the orchestrator launches the next one.

## Pipeline

```
earthquake_detected → drone_scan → image_crop → vlm_analysis → db_cleanup
```

| Step | Service | Type | What it does |
|------|---------|------|--------------|
| `earthquake_detected` | `earthquake-detector` | Container (continuous) | LSTM classifies seismic data from Arduino sensor. |
| `drone_scan` | `drone-scanner` | Container (on-demand) | DJI drone captures 4 wall photos. |
| `image_crop` | `image-cropper` | Container (on-demand) | Detects AprilTags, crops images to tag region. No diff. |
| `vlm_analysis` | `vlm-analyzer` | HTTP service (always-on) | Sends cropped images to VLM, saves images to Supabase Storage, writes crack analysis to DB, compares with previous run, sends GPS alert. |
| `db_cleanup` | `db-cleanup` | Container (on-demand) | Removes old data from Supabase, keeps 2 versions. Runs last. |

## Architecture

```
┌──────────────────┐     ┌──────────────────┐
│   Orchestrator    │     │   VLM Analyzer    │
│   port 5050       │────▶│   port 5060       │
│   launches steps  │     │   always running  │
└────────┬─────────┘     └──────────────────┘
         │                   │
         │ POST /step/done   │ POST /run (from orchestrator)
         │◀──────────────────│ POST /analyze (direct image upload)
         │                   │
         │ launches containers for:
         ├──▶ drone-scanner
         ├──▶ image-cropper
         └──▶ db-cleanup
```

Steps 1, 2, 4 run as **on-demand containers** (launched by the orchestrator, exit when done).
Step 3 (VLM) runs as an **always-on HTTP service** — the orchestrator sends `POST /run` to trigger it.

## VLM Analyzer

The VLM is a FastAPI server with two endpoints:

### `POST /analyze` — Single image

Upload one image for analysis. Returns results immediately.

```bash
curl -X POST http://localhost:5060/analyze \
  -F "image=@photo.jpg" \
  -F "run_id=run_123"
```

Response:
```json
{
  "image_name": "photo.jpg",
  "image_url": "https://...supabase.co/storage/v1/object/public/crack-images/...",
  "analysis": {
    "has_crack": true,
    "severity": "moderate",
    "crack_count": 2,
    "max_crack_length_mm": 45.3,
    "max_crack_width_mm": 1.2,
    "description": "Two diagonal cracks in upper quadrant"
  },
  "comparison": {
    "summary": "+1 new crack(s); severity low→moderate; max length +12.0mm",
    "status": "worsened",
    "crack_count": {"previous": 1, "current": 2, "delta": 1},
    "severity": {"previous": "low", "current": "moderate", "changed": true},
    "max_length_mm": {"previous": 33.3, "current": 45.3, "delta": 12.0},
    "max_width_mm": {"previous": 0.8, "current": 1.2, "delta": 0.4},
    "new_cracks": ["photo.jpg"],
    "resolved_cracks": []
  },
  "alert_sent": true
}
```

### `POST /run` — Batch (called by orchestrator)

Processes all images in `/app/input`. Called automatically when the pipeline reaches the VLM step.

### What happens on each analysis

1. **Image uploaded to Supabase Storage** (`crack-images` bucket)
2. **Image sent to VLM** for crack analysis (or placeholder mock if `VLM_ENDPOINT` not set)
3. **Results saved** to Supabase `crack_reports` table
4. **Previous run fetched** from DB and **compared programmatically**:
   - Crack count delta
   - Severity change
   - Max crack length/width deltas
   - New cracks vs resolved cracks
   - Overall status: `worsened` / `improved` / `stable` / `new`
5. **GPS alert sent** to blackboard if crack detected (fake Paphos coordinates)

### Comparison logic

No LLM needed — pure JSON diff. The comparison computes:

| Field | How |
|-------|-----|
| `crack_count` | `current - previous` |
| `severity` | Ordered: none < low < moderate < high < critical |
| `max_length_mm` | Delta in mm |
| `max_width_mm` | Delta in mm |
| `new_cracks` | Images with cracks now that didn't have them before |
| `resolved_cracks` | Images that had cracks before but don't now |
| `status` | `worsened` if severity/count/length increased, `improved` if decreased, `stable` otherwise |

## Running

```bash
# Build everything
docker compose build

# Start always-on services (orchestrator + VLM)
docker compose up orchestrator vlm-analyzer -d

# Start earthquake detector
docker compose up earthquake-detector -d

# Trigger pipeline manually
curl -X POST http://localhost:5050/pipeline/trigger \
  -H "Content-Type: application/json" \
  -d '{"step": "drone_scan"}'

# Or send a single image directly to VLM
curl -X POST http://localhost:5060/analyze \
  -F "image=@wall_photo.jpg"

# Check pipeline status
curl http://localhost:5050/pipeline/status

# Check VLM health
curl http://localhost:5060/health
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ORCHESTRATOR_URL` | `http://orchestrator:5050` | How containers reach the orchestrator |
| `VLM_URL` | `http://vlm-analyzer:5060` | How orchestrator reaches VLM |
| `VLM_ENDPOINT` | _(empty = placeholder)_ | External VLM API for actual analysis |
| `SUPABASE_URL` | `https://YOUR_PROJECT.supabase.co` | Supabase project |
| `SUPABASE_KEY` | _(hardcoded)_ | Supabase API key |
| `STORAGE_BUCKET` | `crack-images` | Supabase Storage bucket for images |
| `ALERT_URL` | `https://blackboard.jass.school/blackboard` | Where to send crack alerts |

## Files

```
orchestrator/
├── server.py          # Pipeline orchestrator (port 5050)
├── notify.py          # Helper for containers to POST back
├── Dockerfile
└── requirements.txt

vlm/
├── server.py          # VLM API server (port 5060)
├── compare.py         # Programmatic run-to-run comparison
└── Dockerfile

crack_detection/image_diff/
├── crop.py            # AprilTag crop-only script
├── Dockerfile.crop    # Container for cropping
└── pipeline/          # Shared preprocessing

db_cleanup/
├── cleanup.py         # Supabase cleanup (runs last)
└── Dockerfile
```
