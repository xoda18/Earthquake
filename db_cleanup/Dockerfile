FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir requests

COPY db_cleanup/cleanup.py .
COPY orchestrator/notify.py .

CMD ["python", "cleanup.py"]
