import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

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

# Load dataset
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

# Load trained model
model = joblib.load(MODEL_FILE)

# Get feature importance
importance = model.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

# Sort
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("====================================")
print("FEATURE IMPORTANCE")
print("====================================")

print("\nFeature Importance:")
print(importance_df)

# Plot
plt.figure(figsize=(9, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Forest Fire Prediction - Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

output_file = os.path.join(
    SCREENSHOT_DIR,
    "06_feature_importance.png"
)

plt.savefig(output_file)

plt.show()

print("\nGraph saved:")
print(output_file)

print("====================================")