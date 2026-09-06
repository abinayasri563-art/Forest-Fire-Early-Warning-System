import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "model_data.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "forest_fire_model.pkl"
)

SCREENSHOT_DIR = os.path.join(
    BASE_DIR,
    "screenshots"
)

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_FILE)

features = [
    "Temperature",
    "RH",
    "Ws",
    "Rain",
    "FFMC",
    "DMC",
    "DC",
    "ISI",
    "BUI",
    "FWI"
]

X = df[features]
y = df["Fire"]

# Same train/test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load trained model
model = joblib.load(MODEL_FILE)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("====================================")
print("MODEL EVALUATION")
print("====================================")

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Not Fire", "Fire"]
    )
)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Create confusion matrix graph
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Not Fire", "Fire"],
    yticklabels=["Not Fire", "Fire"]
)

plt.title("Forest Fire Prediction - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    os.path.join(
        SCREENSHOT_DIR,
        "05_confusion_matrix.png"
    )
)

plt.show()

print("\nEvaluation screenshot saved:")
print(
    os.path.join(
        SCREENSHOT_DIR,
        "05_confusion_matrix.png"
    )
)

print("====================================")