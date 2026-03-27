FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir opencv-python requests

WORKDIR /app
COPY DJIControlClient.py wall_scan.py ./

CMD ["python", "wall_scan.py"]
