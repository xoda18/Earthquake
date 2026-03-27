FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    tensorflow-cpu \
    numpy \
    pyserial \
    requests \
    scikit-learn

COPY detection/streaming/infer.py .
COPY models/lstm_earthquake_model.h5 model.h5
COPY models/lstm_scaler.pkl scaler.pkl

CMD ["python", "infer.py"]
