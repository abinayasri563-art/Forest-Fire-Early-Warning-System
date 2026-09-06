# 🔥 Forest Fire Early Warning & Prediction System

## 📌 Project Overview

The **Forest Fire Early Warning & Prediction System** is a Machine Learning based application designed to predict the risk of forest fires using environmental and fire-weather data.

The system analyzes parameters such as temperature, relative humidity, wind speed, rainfall, FFMC, DMC, DC, ISI, BUI, and FWI to determine whether a forest fire is likely to occur.

The project also provides a **real-time Streamlit dashboard** for monitoring environmental conditions and displaying fire-risk predictions.

## 🎯 Objectives

* Predict forest fire occurrence using Machine Learning.
* Analyze important environmental conditions related to forest fires.
* Monitor temperature, humidity, wind speed, and rainfall.
* Classify fire risk into Low, Medium, and High risk levels.
* Provide manual fire-risk prediction using user-entered values.
* Display sensor trends through interactive graphs.
* Identify the most important features influencing fire prediction.

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Joblib**
* **Streamlit**
* **Random Forest Classifier**

## 📊 Dataset

The project uses the **Algerian Forest Fires Dataset**.

The dataset contains environmental and fire-weather information from two regions of Algeria:

* Bejaia Region
* Sidi Bel-Abbes Region

The dataset includes parameters such as:

* Temperature
* Relative Humidity (RH)
* Wind Speed (Ws)
* Rain
* FFMC
* DMC
* DC
* ISI
* BUI
* FWI
* Fire/Not Fire class

## 🧠 Machine Learning Model

A **Random Forest Classifier** is used for forest fire prediction.

### Input Features

```text
Temperature
RH
Ws
Rain
FFMC
DMC
DC
ISI
BUI
FWI
```

### Output

The model predicts:

```text
Fire
Not Fire
```

The Streamlit application converts the prediction probability into three risk levels:

* 🟢 **Low Risk** — below 40%
* 🟡 **Medium Risk** — 40% to 69%
* 🔴 **High Risk** — 70% and above

## 📁 Project Structure

```text
Forest_Fire_Early Warning/
│
├── data/
│   ├── Algerian_forest_fires_dataset_UPDATE.csv
│   ├── forest_fire_cleaned.csv
│   ├── forest_fire_model_data.csv
│   ├── forest_fire_parsed.csv
│   └── model_data.csv
│
├── notebooks/
│   ├── 01_load_dataset.py
│   ├── 02_clean_dataset.py
│   ├── 03_eda.py
│   ├── 04_prepare_model.py
│   ├── 05_train_model.py
│   ├── 06_evaluate_model.py
│   └── 07_feature_importance.py
│
├── models/
│   └── forest_fire_model.pkl
│
├── app/
│   ├── predict.py
│   ├── streamlit_app.py
│   └── .gitignore
│
├── screenshots/
│   ├── 01_fire_distribution.png
│   ├── 05_confusion_matrix.png
│   └── 06_feature_importance.png
│
└── README.md
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/Forest-Fire-Early-Warning-System.git
```

Go to the project directory:

```bash
cd Forest-Fire-Early-Warning-System
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment on Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
```

## ▶️ Run the Streamlit Application

Run:

```bash
streamlit run app/streamlit_app.py
```

The application opens in the browser and displays the Forest Fire Early Warning dashboard.

## 🔬 Run the Machine Learning Pipeline

### 1. Load and parse the dataset

```bash
python notebooks/01_load_dataset.py
```

### 2. Clean the dataset

```bash
python notebooks/02_clean_dataset.py
```

### 3. Perform Exploratory Data Analysis

```bash
python notebooks/03_eda.py
```

### 4. Prepare model data

```bash
python notebooks/04_prepare_model.py
```

### 5. Train the Random Forest model

```bash
python notebooks/05_train_model.py
```

### 6. Evaluate the model

```bash
python notebooks/06_evaluate_model.py
```

### 7. Generate feature importance

```bash
python notebooks/07_feature_importance.py
```

## 📈 Application Features

### Real-Time Monitoring

The dashboard simulates environmental sensor readings and continuously displays:

* Temperature
* Humidity
* Wind Speed
* Rainfall
* Fire probability
* Current fire-risk level

### Manual Prediction

Users can enter environmental values manually and check the predicted fire risk.

### Risk Classification

The application provides a simple risk indicator:

```text
LOW RISK
MEDIUM RISK
HIGH RISK
```

### Data Visualization

The project includes visualizations for:

* Fire vs. Not Fire distribution
* Temperature vs. Fire
* Rain vs. Fire
* Correlation between features
* Confusion matrix
* Feature importance

## 📸 Screenshots

Project screenshots are available in the `screenshots/` folder.

## 🌲 Applications

This system can be used as a prototype for:

* Forest monitoring systems
* Environmental monitoring
* Early warning systems
* IoT-based fire detection
* Disaster management
* Smart environmental monitoring

## 🚀 Future Enhancements

Future versions can include:

* Real IoT sensors instead of simulated sensor values.
* GPS-based forest location monitoring.
* SMS/email alerts for high-risk conditions.
* Cloud database integration.
* Real-time weather API integration.
* Deployment on cloud platforms.
* Mobile application integration.
* Multiple forest monitoring locations.

