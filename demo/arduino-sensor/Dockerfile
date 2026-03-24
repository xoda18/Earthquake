FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    tensorflow-cpu \
    scikit-learn \
    pyserial \
    numpy \
    scipy

COPY lstm_earthquake_model.h5 /app/model.h5
COPY lstm_scaler.pkl /app/scaler.pkl
COPY infer.py /app/infer.py

ENTRYPOINT ["python", "-u", "infer.py"]
