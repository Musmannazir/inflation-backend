import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import datetime
import numpy as np
import uvicorn

from fastapi import FastAPI

app = FastAPI()

@app.get("/features")
def features_test():
    return {"DEPLOYMENT": "RENDER IS RUNNING NEW CODE"}

# --------------------------
# Initialize FastAPI app
# --------------------------
app = FastAPI(title="Pakistan Inflation Predictor", version="2.1")

# --------------------------
# Load trained model
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully")

# --------------------------
# Input schema
# --------------------------
class InflationInput(BaseModel):
    t1: float
    t2: float
    t3: float
    CPI_MoM: float | None = None
    WPI_MoM: float | None = None
    SPI_MoM: float | None = None

# --------------------------
# Prediction endpoint
# --------------------------
@app.post("/predict")
def predict(data: InflationInput):

    # --------------------------
    # Create base feature dict
    # --------------------------
    features = {}

    # Inflation lags
    features["inflation_lag_1"] = data.t1
    features["inflation_lag_2"] = data.t2
    features["inflation_lag_3"] = data.t3
    features["inflation_rolling_mean_3"] = (data.t1 + data.t2 + data.t3) / 3

    # Component values (default 0)
    CPI = data.CPI_MoM or 0.0
    WPI = data.WPI_MoM or 0.0
    SPI = data.SPI_MoM or 0.0

    for name, val in [("CPI_MoM", CPI), ("WPI_MoM", WPI), ("SPI_MoM", SPI)]:
        features[f"{name}_lag_1"] = val
        features[f"{name}_lag_2"] = val
        features[f"{name}_lag_3"] = val
        features[f"{name}_rolling_mean_3"] = val

    # Seasonal features
    month = datetime.datetime.now().month
    features["month_sin"] = np.sin(2 * np.pi * month / 12)
    features["month_cos"] = np.cos(2 * np.pi * month / 12)

    # --------------------------
    # Convert to DataFrame
    # --------------------------
    df = pd.DataFrame([features])

    # --------------------------
    # 🔐 ALIGN FEATURES WITH MODEL (CRITICAL FIX)
    # --------------------------
    model_features = model.get_booster().feature_names

    for col in model_features:
        if col not in df.columns:
            df[col] = 0.0

    df = df[model_features]

    print("MODEL EXPECTS:", model_features)
    print("SENDING:", df.columns.tolist())

    # --------------------------
    # Predict
    # --------------------------
    prediction = float(model.predict(df)[0])

    return {
        "predicted_inflation_next_month": round(prediction, 2)
    }
    
@app.get("/features")
def get_model_features():
    return {
        "model_features": model.get_booster().feature_names
    }

# --------------------------
# Health check
# --------------------------
@app.get("/")
def root():
    return {"status": "Pakistan Inflation Predictor API running"}

# --------------------------
# Local run
# --------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
