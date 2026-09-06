import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "forest_fire_cleaned.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "model_data.csv"
)

# Load cleaned dataset
df = pd.read_csv(INPUT_FILE)

# Features used for prediction
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

# Target
target = "Fire"

# Select required columns
model_df = df[features + [target]].copy()

# Remove missing values
model_df = model_df.dropna()

# Save model dataset
model_df.to_csv(OUTPUT_FILE, index=False)

print("====================================")
print("MODEL DATA PREPARED")
print("====================================")

print("\nModel Dataset Shape:")
print(model_df.shape)

print("\nFeatures:")
print(features)

print("\nTarget:")
print(target)

print("\nTarget Distribution:")
print(model_df[target].value_counts())

print("\nFirst 5 Rows:")
print(model_df.head())

print("\nSaved File:")
print(OUTPUT_FILE)

print("====================================")