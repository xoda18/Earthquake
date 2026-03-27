## Pipeline Architecture

```
┌──────────────────────┐
│   Orchestrator       │  ← FastAPI server (port 5050)
│   POST /step/done    │  ← receives completion signals
│   POST /pipeline/trigger │ ← manual trigger (e.g. UI button)
└──────────┬───────────┘
           │  launches next Docker container
           │
    ┌──────┴──────────────────────────────────────────┐
    │                                                  │
    ▼                                                  │
┌────────────────┐   POST /step/done              ┌────┴───────────┐
│ Earthquake     │──────────────────────────────→  │ Orchestrator   │
│ Detector       │  step=earthquake_detected       │                │
│ (continuous)   │                                 └────┬───────────┘
└────────────────┘                                      │ launches
                                                        ▼
┌────────────────┐   POST /step/done              ┌────────────────┐
│ Drone Scanner  │──────────────────────────────→  │ Orchestrator   │
│ 4 photos       │  step=drone_scan                │                │
│ upload to DB   │                                 └────┬───────────┘
└────────────────┘                                      │ launches
                                                        ▼
┌────────────────┐   POST /step/done              ┌────────────────┐
│ DB Cleanup     │──────────────────────────────→  │ Orchestrator   │
│ keep 2 versions│  step=db_cleanup                │                │
└────────────────┘                                 └────┬───────────┘
                                                        │ launches
                                                        ▼
┌────────────────┐   POST /step/done              ┌────────────────┐
│ Image Diff     │──────────────────────────────→  │ Orchestrator   │
│ crack detection│  step=image_diff                │                │
│ → output_resized                                 └────┬───────────┘
└────────────────┘                                      │ launches
                                                        ▼
┌────────────────┐   POST /step/done
│ VLM Analyzer   │──────────────────────────────→  Pipeline Complete
│ crack analysis │  step=vlm_analysis
└────────────────┘
```

## Containers

| # | Container | Image | Trigger | Notifies |
|---|-----------|-------|---------|----------|
| 0 | Earthquake Detector | `earthquake-detector` | Runs continuously | `earthquake_detected` |
| 1 | Drone Scanner | `drone-scanner` | Orchestrator / UI button | `drone_scan` |
| 2 | DB Cleanup | `db-cleanup` | Orchestrator | `db_cleanup` |
| 3 | Image Diff | `crack-detector` | Orchestrator | `image_diff` |
| 4 | VLM Analyzer | `vlm-analyzer` | Orchestrator | `vlm_analysis` |

## Usage

```bash
# Build all containers
docker compose build

# Start orchestrator (always running)
docker compose up orchestrator -d

# Start earthquake detector (continuous)
docker compose up earthquake-detector -d

# Trigger pipeline manually (e.g. from UI "launch drone" button)
curl -X POST http://localhost:5050/pipeline/trigger \
  -H "Content-Type: application/json" \
  -d '{"step": "drone_scan"}'

# Check pipeline status
curl http://localhost:5050/pipeline/status

# Check available steps
curl http://localhost:5050/pipeline/steps
```

## How it works

1. Each container does its job, then sends `POST /step/done` to the orchestrator
2. The orchestrator looks up the next step in the pipeline and launches that container
3. The pipeline chain: `earthquake_detected → drone_scan → db_cleanup → image_diff → vlm_analysis`
4. If a step fails (`status: "failed"`), the pipeline halts
5. The UI can trigger any step manually via `POST /pipeline/trigger`
