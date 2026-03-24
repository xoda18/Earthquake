FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    tensorflow-cpu \
    scikit-learn \
    pyserial \
    numpy \
    scipy

# Build from repo root: docker build -f detection/streaming/Dockerfile .
COPY models/lstm_earthquake_model.h5 /app/model.h5
COPY models/lstm_scaler.pkl /app/scaler.pkl
COPY detection/streaming/infer.py /app/infer.py

ENTRYPOINT ["python", "-u", "infer.py"]
