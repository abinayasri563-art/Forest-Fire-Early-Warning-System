import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Forest Fire Early Warning System",
    page_icon="🔥",
    layout="wide"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "forest_fire_model.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "model_data.csv"
)

# ============================================================
# FEATURES
# ============================================================

FEATURES = [
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

# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error("❌ Model file not found.")

    st.write("Expected file:")
    st.code(MODEL_PATH)

    st.info(
        "Run 05_train_model.py first to create the model."
    )

    st.stop()

try:

    model = joblib.load(MODEL_PATH)

except Exception as e:

    st.error("❌ Model loading failed.")

    st.code(str(e))

    st.stop()

# ============================================================
# SESSION STATE
# ============================================================

if "sensor_history" not in st.session_state:
    st.session_state.sensor_history = []

if "probability_history" not in st.session_state:
    st.session_state.probability_history = []

# ============================================================
# TITLE
# ============================================================

st.title("🔥 Forest Fire Early Warning System")

st.subheader(
    "Machine Learning Based Forest Fire Risk Monitoring"
)

st.write(
    "This system monitors environmental conditions "
    "and predicts the possibility of forest fire."
)

st.divider()

# ============================================================
# LIVE MONITORING
# ============================================================

st.header("📡 Live Environmental Monitoring")

# Simulated sensor values

temperature = np.random.uniform(25, 45)
humidity = np.random.uniform(20, 80)
wind_speed = np.random.uniform(5, 30)
rain = np.random.uniform(0, 10)

ffmc = np.random.uniform(40, 95)
dmc = np.random.uniform(1, 60)
dc = np.random.uniform(5, 200)
isi = np.random.uniform(0, 20)
bui = np.random.uniform(1, 80)
fwi = np.random.uniform(0, 30)

# ============================================================
# DISPLAY SENSOR VALUES
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🌡️ Temperature",
        f"{temperature:.1f} °C"
    )

with col2:
    st.metric(
        "💧 Humidity",
        f"{humidity:.1f} %"
    )

with col3:
    st.metric(
        "💨 Wind Speed",
        f"{wind_speed:.1f}"
    )

with col4:
    st.metric(
        "🌧️ Rain",
        f"{rain:.2f} mm"
    )

# ============================================================
# CREATE LIVE DATA
# ============================================================

live_data = pd.DataFrame(
    {
        "Temperature": [temperature],
        "RH": [humidity],
        "Ws": [wind_speed],
        "Rain": [rain],
        "FFMC": [ffmc],
        "DMC": [dmc],
        "DC": [dc],
        "ISI": [isi],
        "BUI": [bui],
        "FWI": [fwi]
    }
)

# Make sure column order is correct

live_data = live_data[FEATURES]

# ============================================================
# LIVE PREDICTION
# ============================================================

try:

    prediction = model.predict(
        live_data
    )[0]

    probability = (
        model.predict_proba(
            live_data
        )[0][1] * 100
    )

except Exception as e:

    st.error("❌ Prediction failed.")

    st.code(str(e))

    st.stop()

# ============================================================
# RISK LEVEL
# ============================================================

if probability >= 70:

    risk_level = "HIGH RISK 🔴"

elif probability >= 40:

    risk_level = "MEDIUM RISK 🟠"

else:

    risk_level = "LOW RISK 🟢"

# ============================================================
# PREDICTION RESULT
# ============================================================

st.header("🚨 Fire Risk Prediction")

r1, r2, r3 = st.columns(3)

with r1:

    st.metric(
        "Fire Probability",
        f"{probability:.2f}%"
    )

with r2:

    st.metric(
        "Risk Level",
        risk_level
    )

with r3:

    if prediction == 1:

        st.metric(
            "Prediction",
            "🔥 FIRE"
        )

    else:

        st.metric(
            "Prediction",
            "✅ NO FIRE"
        )

# ============================================================
# WARNING
# ============================================================

if probability >= 70:

    st.error(
        "🚨 HIGH FIRE RISK! Immediate attention is recommended."
    )

elif probability >= 40:

    st.warning(
        "⚠️ MEDIUM FIRE RISK. Continue monitoring."
    )

else:

    st.success(
        "✅ LOW FIRE RISK. Conditions are relatively safe."
    )

# ============================================================
# SAVE HISTORY
# ============================================================

st.session_state.sensor_history.append(
    {
        "Temperature": temperature,
        "Humidity": humidity,
        "Wind Speed": wind_speed
    }
)

st.session_state.probability_history.append(
    probability
)

# Keep only 20 readings

if len(st.session_state.sensor_history) > 20:

    st.session_state.sensor_history.pop(0)

if len(st.session_state.probability_history) > 20:

    st.session_state.probability_history.pop(0)

# ============================================================
# SENSOR HISTORY
# ============================================================

st.divider()

st.header("📊 Sensor History")

history_df = pd.DataFrame(
    st.session_state.sensor_history
)

if not history_df.empty:

    st.line_chart(history_df)

# ============================================================
# PROBABILITY HISTORY
# ============================================================

st.header("📈 Fire Probability History")

probability_df = pd.DataFrame(
    {
        "Fire Probability (%)":
        st.session_state.probability_history
    }
)

if not probability_df.empty:

    st.line_chart(probability_df)

# ============================================================
# MANUAL PREDICTION
# ============================================================

st.divider()

st.header("🧪 Manual Fire Risk Prediction")

st.write(
    "Enter the environmental values and check the fire risk."
)

# ------------------------------------------------------------
# ROW 1
# ------------------------------------------------------------

a1, a2, a3 = st.columns(3)

with a1:

    temperature_input = st.number_input(
        "Temperature (°C)",
        0.0,
        60.0,
        30.0
    )

with a2:

    rh_input = st.number_input(
        "Relative Humidity (%)",
        0.0,
        100.0,
        50.0
    )

with a3:

    ws_input = st.number_input(
        "Wind Speed",
        0.0,
        100.0,
        15.0
    )

# ------------------------------------------------------------
# ROW 2
# ------------------------------------------------------------

b1, b2, b3 = st.columns(3)

with b1:

    rain_input = st.number_input(
        "Rain (mm)",
        0.0,
        100.0,
        0.0
    )

with b2:

    ffmc_input = st.number_input(
        "FFMC",
        0.0,
        120.0,
        70.0
    )

with b3:

    dmc_input = st.number_input(
        "DMC",
        0.0,
        200.0,
        10.0
    )

# ------------------------------------------------------------
# ROW 3
# ------------------------------------------------------------

c1, c2, c3 = st.columns(3)

with c1:

    dc_input = st.number_input(
        "DC",
        0.0,
        1000.0,
        50.0
    )

with c2:

    isi_input = st.number_input(
        "ISI",
        0.0,
        100.0,
        5.0
    )

with c3:

    bui_input = st.number_input(
        "BUI",
        0.0,
        300.0,
        15.0
    )

# ------------------------------------------------------------
# FWI
# ------------------------------------------------------------

fwi_input = st.number_input(
    "FWI",
    0.0,
    100.0,
    5.0
)

# ============================================================
# MANUAL PREDICTION BUTTON
# ============================================================

if st.button(
    "🔍 Check Fire Risk",
    use_container_width=True
):

    manual_data = pd.DataFrame(
        {
            "Temperature": [temperature_input],
            "RH": [rh_input],
            "Ws": [ws_input],
            "Rain": [rain_input],
            "FFMC": [ffmc_input],
            "DMC": [dmc_input],
            "DC": [dc_input],
            "ISI": [isi_input],
            "BUI": [bui_input],
            "FWI": [fwi_input]
        }
    )

    manual_data = manual_data[FEATURES]

    try:

        manual_prediction = model.predict(
            manual_data
        )[0]

        manual_probability = (
            model.predict_proba(
                manual_data
            )[0][1] * 100
        )

        if manual_probability >= 70:

            manual_risk = "HIGH RISK 🔴"

        elif manual_probability >= 40:

            manual_risk = "MEDIUM RISK 🟠"

        else:

            manual_risk = "LOW RISK 🟢"

        st.subheader(
            "📋 Manual Prediction Result"
        )

        x1, x2, x3 = st.columns(3)

        with x1:

            st.metric(
                "Fire Probability",
                f"{manual_probability:.2f}%"
            )

        with x2:

            st.metric(
                "Risk Level",
                manual_risk
            )

        with x3:

            if manual_prediction == 1:

                st.metric(
                    "Prediction",
                    "🔥 FIRE"
                )

            else:

                st.metric(
                    "Prediction",
                    "✅ NO FIRE"
                )

        if manual_probability >= 70:

            st.error(
                "🚨 HIGH FIRE RISK detected!"
            )

        elif manual_probability >= 40:

            st.warning(
                "⚠️ MEDIUM FIRE RISK detected."
            )

        else:

            st.success(
                "✅ LOW FIRE RISK detected."
            )

    except Exception as e:

        st.error(
            "❌ Manual prediction failed."
        )

        st.code(str(e))

# ============================================================
# DATASET INFORMATION
# ============================================================

st.divider()

st.header("📚 Dataset Information")

if os.path.exists(DATA_PATH):

    try:

        dataset = pd.read_csv(
            DATA_PATH
        )

        total_records = len(dataset)

        if "Fire" in dataset.columns:

            fire_records = int(
                dataset["Fire"].sum()
            )

            no_fire_records = (
                total_records -
                fire_records
            )

        else:

            fire_records = 0
            no_fire_records = 0

        d1, d2, d3 = st.columns(3)

        with d1:

            st.metric(
                "Total Records",
                total_records
            )

        with d2:

            st.metric(
                "🔥 Fire Records",
                fire_records
            )

        with d3:

            st.metric(
                "✅ No-Fire Records",
                no_fire_records
            )

    except Exception as e:

        st.warning(
            "Dataset could not be loaded."
        )

        st.code(str(e))

else:

    st.warning(
        "model_data.csv not found."
    )

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.divider()

st.header("📊 Important Fire Risk Factors")

try:

    importance_values = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "Feature": FEATURES,
            "Importance": importance_values
        }
    )

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    st.bar_chart(
        importance_df.set_index(
            "Feature"
        )
    )

except Exception as e:

    st.warning(
        "Feature importance could not be displayed."
    )

    st.code(str(e))

# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.header("🤖 Machine Learning Model")

st.write(
    "**Algorithm:** Random Forest Classifier"
)

st.write(
    "**Purpose:** Forest fire classification"
)

st.write(
    "**Input Features:**"
)

st.write(
    ", ".join(FEATURES)
)

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🔥 Forest Fire Early Warning System | "
    "Machine Learning + Streamlit"
)

st.caption(
    "Note: Live sensor values are simulated "
    "for demonstration purposes."
)