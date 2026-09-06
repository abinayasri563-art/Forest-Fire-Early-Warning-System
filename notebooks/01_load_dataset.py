
import csv
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "Algerian_forest_fires_dataset_UPDATE.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "forest_fire_parsed.csv"
)

columns = [
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
    "FWI",
    "Classes"
]

rows = []

with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as file:
    reader = csv.reader(file)

    for raw_row in reader:

        row = [value.strip() for value in raw_row]

        if not row or not any(row):
            continue

        # Skip region headings
        text = " ".join(row).lower()

        if "bejaia region dataset" in text:
            continue

        if "sidi-bel abbes region dataset" in text:
            continue

        # Skip column heading
        if row[0].lower() == "day":
            continue

        # Data rows must contain 14 values
        if len(row) != 14:
            continue

        try:
            day = int(row[0])
            month = int(row[1])
            year = int(row[2])
        except ValueError:
            continue

        if year != 2012:
            continue

        class_value = row[13].strip().lower()

        if class_value not in ["fire", "not fire"]:
            continue

        rows.append([
            day,
            month,
            year,
            row[3],
            row[4],
            row[5],
            row[6],
            row[7],
            row[8],
            row[9],
            row[10],
            row[11],
            row[12],
            class_value
        ])

df = pd.DataFrame(rows, columns=columns)

numeric_columns = columns[:-1]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna()
df = df.reset_index(drop=True)

df.to_csv(OUTPUT_FILE, index=False)

print("====================================")
print("FOREST FIRE DATASET LOADED")
print("====================================")

print("\nDataset Shape:")
print(df.shape)

print("\nClass Values:")
print(df["Classes"].value_counts())

print("\nFirst 5 Rows:")
print(df.head())

print("\nSaved File:")
print(OUTPUT_FILE)

print("====================================")