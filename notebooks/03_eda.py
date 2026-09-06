import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "forest_fire_cleaned.csv"
)

SCREENSHOT_DIR = os.path.join(
    BASE_DIR,
    "screenshots"
)

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Load data
df = pd.read_csv(INPUT_FILE)

print("Dataset Shape:", df.shape)
print("\nDataset Preview:")
print(df.head())

# --------------------------------
# Graph 1: Fire Distribution
# --------------------------------
plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Classes"
)

plt.title("Forest Fire Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Records")
plt.tight_layout()

plt.savefig(
    os.path.join(
        SCREENSHOT_DIR,
        "01_fire_distribution.png"
    )
)

plt.show()


# --------------------------------
# Graph 2: Temperature vs Fire
# --------------------------------
plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Classes",
    y="Temperature"
)

plt.title("Temperature vs Forest Fire")
plt.xlabel("Class")
plt.ylabel("Temperature")

plt.tight_layout()

plt.savefig(
    os.path.join(
        SCREENSHOT_DIR,
        "02_temperature_vs_fire.png"
    )
)

plt.show()


# --------------------------------
# Graph 3: Rain vs Fire
# --------------------------------
plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Classes",
    y="Rain"
)

plt.title("Rainfall vs Forest Fire")
plt.xlabel("Class")
plt.ylabel("Rainfall")

plt.tight_layout()

plt.savefig(
    os.path.join(
        SCREENSHOT_DIR,
        "03_rain_vs_fire.png"
    )
)

plt.show()


# --------------------------------
# Graph 4: Correlation Heatmap
# --------------------------------
plt.figure(figsize=(10, 7))

numeric_df = df.select_dtypes(include="number")

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()

plt.savefig(
    os.path.join(
        SCREENSHOT_DIR,
        "04_correlation_heatmap.png"
    )
)

plt.show()


print("\n================================")
print("EDA COMPLETED SUCCESSFULLY")
print("================================")

print("\nGraphs saved in:")
print(SCREENSHOT_DIR)