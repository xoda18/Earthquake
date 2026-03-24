"""
Test trained model on independently generated data
"""
import pandas as pd
import numpy as np
import pickle
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import sys

print("🧪 MODEL TESTING ON GENERATED DATA\n")

# Window parameters (same as during training)
WINDOW_SIZE = 100
STRIDE = 50

# Load model
print("🧠 Loading model...")
model = keras.models.load_model('../lstm_earthquake_model.h5')
with open('../lstm_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load all CSV files
csv_files = ['earthquake_noise.csv', 'earthquake_weak.csv',
             'earthquake_moderate.csv', 'earthquake_strong.csv']

results_summary = []
all_predictions = []
all_true_labels = []

print(f"\n{'File':<25} {'True':<15} {'Predicted':<15} {'Probability':<15} {'Status'}")
print("=" * 85)

for csv_file in csv_files:
    # Load data
    df = pd.read_csv(csv_file)
    data = df[['x', 'y', 'z']].values
    true_label = df['label'].iloc[0]

    # Создать окна
    windows = []
    for start in range(0, len(data) - WINDOW_SIZE, STRIDE):
        window = data[start:start + WINDOW_SIZE].T
        windows.append(window)

    X = np.array(windows)

    # Нормализовать
    X_reshaped = X.reshape(-1, WINDOW_SIZE)
    X_reshaped = scaler.transform(X_reshaped)
    X = X_reshaped.reshape(-1, 3, WINDOW_SIZE)

    # Предсказать
    predictions = model.predict(X, verbose=0)
    mean_prob = predictions.mean()

    # Определить класс
    pred_label = 1 if mean_prob > 0.5 else 0
    true_class = "EARTHQUAKE" if true_label == 1 else "QUIET"
    pred_class = "EARTHQUAKE" if pred_label == 1 else "QUIET"
    status = "✅" if pred_label == true_label else "❌"

    # Сохранить результаты
    results_summary.append({
        'файл': csv_file,
        'истина': true_label,
        'предсказано': pred_label,
        'вероятность': mean_prob
    })

    all_predictions.append(pred_label)
    all_true_labels.append(true_label)

    print(f"{csv_file:<25} {true_class:<15} {pred_class:<15} {mean_prob:>6.1%}{'':<8} {status}")

# Overall metrics
print("\n" + "=" * 85)
print("📊 OVERALL METRICS")
print("=" * 85)

y_true = np.array(all_true_labels)
y_pred = np.array(all_predictions)

accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy:  {accuracy:.1%}")
print(f"Recall:    {recall:.1%}")
print(f"Precision: {precision:.1%}")
print(f"F1-Score:  {f1:.1%}")

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion matrix:")
print(f"  TN={tn} (quiet correct)      FP={fp} (false earthquake)")
print(f"  FN={fn} (missed)             TP={tp} (earthquake correct)")

if accuracy == 1.0:
    print("\n🎉 PERFECT! Model is 100% accurate!")
elif accuracy >= 0.75:
    print("\n✅ GOOD! Model works reliably")
elif accuracy >= 0.5:
    print("\n⚠️ AVERAGE. Needs improvement")
else:
    print("\n❌ POOR. Retraining required")

print("\n🎉 Testing complete!")
