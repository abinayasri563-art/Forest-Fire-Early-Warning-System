import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "forest_fire_parsed.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "forest_fire_cleaned.csv"
)

# Load dataset
df = pd.read_csv(INPUT_FILE)

print("Original Shape:", df.shape)

# Remove unnecessary spaces from column names
df.columns = df.columns.str.strip()

# Clean class names
df["Classes"] = df["Classes"].str.strip().str.lower()

# Convert numeric columns
numeric_columns = [
    "day",
    "month",
    "year",
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

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove missing values
df = df.dropna()

# Create target column
df["Fire"] = df["Classes"].map({
    "fire": 1,
    "not fire": 0
})

# Remove invalid target rows
df = df.dropna(subset=["Fire"])

df["Fire"] = df["Fire"].astype(int)

# Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\n================================")
print("DATASET CLEANED SUCCESSFULLY")
print("================================")

print("\nCleaned Shape:")
print(df.shape)

print("\nClass Distribution:")
print(df["Classes"].value_counts())

print("\nFire Distribution:")
print(df["Fire"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSaved File:")
print(OUTPUT_FILE)

print("================================")