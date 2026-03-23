import h5py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt

print("📊 Загружаю данные...")

# Параметры окна
WINDOW_SIZE = 100  # Отсчётов на окно (примерно 0.4 сек при 250 Гц)
STRIDE = 50        # Сдвиг окна (для создания больше примеров)

# ============= ЗАГРУЗИТЬ ДАННЫЕ =============
def load_waveforms(filepath, label):
    """Загружает данные из HDF5 и создаёт окна"""
    with h5py.File(filepath, 'r') as f:
        data = f['data/bucket0'][:]  # (328, 3, 25001)

    X_windows = []
    y_labels = []

    for i in range(len(data)):
        waveform = data[i]  # (3, 25001)

        # Создаём окна с stride
        for start in range(0, waveform.shape[1] - WINDOW_SIZE, STRIDE):
            window = waveform[:, start:start + WINDOW_SIZE]  # (3, 100)
            X_windows.append(window)
            y_labels.append(label)

    return np.array(X_windows), np.array(y_labels)

# Загрузить землетрясения (класс 1)
print("📍 Загружаю землетрясения...")
X_eq, y_eq = load_waveforms('Dataset_Central/dataset_earthquakes/waveforms.hdf5', label=1)
print(f"   Окон землетрясений: {len(X_eq)}")

# Загрузить шум (класс 0)
print("📍 Загружаю фоновый шум...")
X_noise, y_noise = load_waveforms('Dataset_Central/dataset_noise/waveforms.hdf5', label=0)
print(f"   Окон шума: {len(X_noise)}")

# Объединить
X = np.vstack([X_eq, X_noise])  # (N_samples, 3, 100)
y = np.hstack([y_eq, y_noise])

print(f"\n✅ Всего примеров: {len(X)}")
print(f"   Класс 0 (шум): {(y == 0).sum()}")
print(f"   Класс 1 (землетрясение): {(y == 1).sum()}")

# ============= НОРМАЛИЗОВАТЬ ДАННЫЕ =============
print("\n🔧 Нормализую данные...")
X_reshaped = X.reshape(-1, WINDOW_SIZE)  # (N_samples * 3, 100)
scaler = StandardScaler()
X_reshaped = scaler.fit_transform(X_reshaped)
X = X_reshaped.reshape(-1, 3, WINDOW_SIZE)

# ============= РАЗДЕЛИТЬ НА ОБУЧЕНИЕ И ТЕСТ =============
print("📊 Разделяю на обучение (90%) и тест (10%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)
print(f"   Обучение: {len(X_train)} примеров")
print(f"   Тест: {len(X_test)} примеров")

# ============= ПОСТРОИТЬ LSTM НЕЙРОСЕТЬ =============
print("\n🧠 Строю LSTM нейросеть для временных рядов...")
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

# ============= ОБУЧИТЬ =============
print("\n⏳ Обучаю модель (это займёт несколько минут)...")
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# ============= ТЕСТИРОВАТЬ =============
print("\n📈 Тестирую модель на новых данных...")
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print(f"\n✅ РЕЗУЛЬТАТЫ НА ТЕСТОВЫХ ДАННЫХ (10%):")
print(f"   Точность (Accuracy):      {accuracy * 100:.2f}%")
print(f"   Полнота (Recall):         {recall * 100:.2f}%")
print(f"   Аккуратность (Precision): {precision * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
print(f"\n🎯 Матрица ошибок:")
print(f"   Правильно класс 0 (шум):           {cm[0,0]}")
print(f"   Неправильно как класс 1:           {cm[0,1]}")
print(f"   Неправильно как класс 0:           {cm[1,0]}")
print(f"   Правильно класс 1 (землетрясение): {cm[1,1]}")

# ============= СОХРАНИТЬ =============
print("\n💾 Сохраняю модель...")
model.save('lstm_earthquake_model.h5')
print("✅ Модель сохранена в lstm_earthquake_model.h5")

import pickle
with open('lstm_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Нормализатор сохранён в lstm_scaler.pkl")

# ============= ГРАФИК =============
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
print("✅ График сохранён в lstm_training_history.png")

print("\n🎉 Обучение завершено!")
