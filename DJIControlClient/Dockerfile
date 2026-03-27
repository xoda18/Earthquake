FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir requests

COPY DJIControlClient.py wall_scan.py ./
COPY scans/ ./scans/

CMD ["python", "wall_scan.py", "--simulate"]
