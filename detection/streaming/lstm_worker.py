"""
lstm_worker.py — LSTM inference worker. Reads (3,100) windows from stdin, writes probabilities to stdout.

Runs with Python 3.11 + TensorFlow. Communicates via JSON lines over stdin/stdout.

Input (one JSON per line):  {"window": [[ax0,ay0,az0], [ax1,ay1,az1], ...]}  (100 rows)
Output (one JSON per line): {"probability": 0.85, "label": "EARTHQUAKE"}
"""

import sys
import os
import json
import pickle
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow import keras

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
def main():
    threshold = 0.5

    model = keras.models.load_model(os.path.join(MODEL_DIR, "lstm_earthquake_model.h5"), compile=False)
    with open(os.path.join(MODEL_DIR, "lstm_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    print(json.dumps({"status": "ready"}), flush=True)

    for line in sys.stdin:
        try:
            msg = json.loads(line)
            if "threshold" in msg:
                threshold = msg["threshold"]
                continue

            window = np.array(msg["window"], dtype=np.float32)  # (100, 3)
            X = window.T.reshape(1, 3, 100)
            X_flat = X.reshape(-1, 100)
            X_flat = scaler.transform(X_flat)
            X = X_flat.reshape(1, 3, 100)
            prob = float(model.predict(X, verbose=0)[0][0])
            label = "EARTHQUAKE" if prob > threshold else "quiet"
            print(json.dumps({"probability": prob, "label": label}), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
