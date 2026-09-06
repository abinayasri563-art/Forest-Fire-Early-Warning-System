import os
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "forest_fire_model.pkl"
)

# Load trained model
model = joblib.load(MODEL_FILE)

print("====================================")
print("FOREST FIRE EARLY WARNING SYSTEM")
print("====================================")

print("\nEnter sensor values:")

temperature = float(input("Temperature: "))
rh = float(input("Relative Humidity (RH): "))
ws = float(input("Wind Speed (Ws): "))
rain = float(input("Rain: "))
ffmc = float(input("FFMC: "))
dmc = float(input("DMC: "))
dc = float(input("DC: "))
isi = float(input("ISI: "))
bui = float(input("BUI: "))
fwi = float(input("FWI: "))

# Create input data
input_data = pd.DataFrame([[
    temperature,
    rh,
    ws,
    rain,
    ffmc,
    dmc,
    dc,
    isi,
    bui,
    fwi
]], columns=[
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
])

# Prediction
prediction = model.predict(input_data)[0]

# Probability
probability = model.predict_proba(input_data)[0]

fire_probability = probability[1] * 100

print("\n====================================")

if prediction == 1:
    print("🔥 WARNING: FOREST FIRE RISK DETECTED")
else:
    print("✅ STATUS: NO FIRE RISK DETECTED")

print(f"Fire Probability: {fire_probability:.2f}%")

print("====================================")