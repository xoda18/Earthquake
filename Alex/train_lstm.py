import h5py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt

print("📊 Loading data...")

# Window parameters
WINDOW_SIZE = 100  # Samples per window (1.0 sec at 100 Hz — matches MPU6500 sensor)
STRIDE = 50        # Window stride (to create more samples)

# ============= LOAD DATA =============
def load_waveforms(filepath, label):
    """Load data from HDF5 and create windows"""
    with h5py.File(filepath, 'r') as f:
        data = f['data/bucket0'][:]  # (328, 3, 25001)

    X_windows = []
    y_labels = []

    for i in range(len(data)):
        waveform = data[i]  # (3, 25001)

        # Create windows with stride
        for start in range(0, waveform.shape[1] - WINDOW_SIZE, STRIDE):
            window = waveform[:, start:start + WINDOW_SIZE]  # (3, 100)
            X_windows.append(window)
            y_labels.append(label)

    return np.array(X_windows), np.array(y_labels)

# Load earthquakes (class 1)
print("📍 Loading earthquakes...")
X_eq, y_eq = load_waveforms('Dataset_Central/dataset_earthquakes/waveforms.hdf5', label=1)
print(f"   Earthquake windows: {len(X_eq)}")

# Load noise (class 0)
print("📍 Loading background noise...")
X_noise, y_noise = load_waveforms('Dataset_Central/dataset_noise/waveforms.hdf5', label=0)
print(f"   Noise windows: {len(X_noise)}")

# Combine
X = np.vstack([X_eq, X_noise])  # (N_samples, 3, 100)
y = np.hstack([y_eq, y_noise])

print(f"\n✅ Total samples: {len(X)}")
print(f"   Class 0 (noise): {(y == 0).sum()}")
print(f"   Class 1 (earthquake): {(y == 1).sum()}")

# ============= NORMALIZE DATA =============
print("\n🔧 Normalizing data...")
X_reshaped = X.reshape(-1, WINDOW_SIZE)  # (N_samples * 3, 100)
scaler = StandardScaler()
X_reshaped = scaler.fit_transform(X_reshaped)
X = X_reshaped.reshape(-1, 3, WINDOW_SIZE)

# ============= SPLIT INTO TRAIN AND TEST =============
print("📊 Splitting into train (90%) and test (10%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)
print(f"   Train: {len(X_train)} samples")
print(f"   Test: {len(X_test)} samples")

# ============= BUILD LSTM NEURAL NETWORK =============
print("\n🧠 Building LSTM neural network for time series...")
model = keras.Sequential([
    keras.layers.LSTM(64, activation='relu', input_shape=(3, WINDOW_SIZE), return_sequences=True),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# ============= TRAIN =============
print("\n⏳ Training model (this will take a few minutes)...")
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# ============= TEST =============
print("\n📈 Testing model on new data...")
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print(f"\n✅ RESULTS ON TEST DATA (10%):")
print(f"   Accuracy:  {accuracy * 100:.2f}%")
print(f"   Recall:    {recall * 100:.2f}%")
print(f"   Precision: {precision * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
print(f"\n🎯 Confusion matrix:")
print(f"   TN (noise correct):        {cm[0,0]}")
print(f"   FP (false earthquake):     {cm[0,1]}")
print(f"   FN (missed earthquake):    {cm[1,0]}")
print(f"   TP (earthquake correct):   {cm[1,1]}")

# ============= SAVE =============
print("\n💾 Saving model...")
model.save('lstm_earthquake_model.h5')
print("✅ Model saved to lstm_earthquake_model.h5")

import pickle
with open('lstm_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved to lstm_scaler.pkl")

# ============= PLOT =============
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss During Training')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy During Training')

plt.tight_layout()
plt.savefig('lstm_training_history.png', dpi=100)
print("✅ Plot saved to lstm_training_history.png")

print("\n🎉 Training complete!")
