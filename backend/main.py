from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import math
import os

app = FastAPI()

# ----------------------------
# Load model
# ----------------------------
MODEL_PATH = os.path.join("model", "model.pkl")
model = joblib.load(MODEL_PATH)

# EXACT feature order (from /features)
FEATURE_ORDER = [
    "inflation_lag_1",
    "inflation_lag_2",
    "inflation_lag_3",
    "CPI_MoM_lag_1",
    "CPI_MoM_lag_2",
    "CPI_MoM_lag_3",
    "WPI_MoM_lag_1",
    "WPI_MoM_lag_2",
    "WPI_MoM_lag_3",
    "SPI_MoM_lag_1",
    "SPI_MoM_lag_2",
    "SPI_MoM_lag_3",
    "inflation_rolling_mean_3",
    "CPI_MoM_rolling_mean_3",
    "WPI_MoM_rolling_mean_3",
    "SPI_MoM_rolling_mean_3",
    "month_sin",
    "month_cos",
]

# ----------------------------
# Request schema
# ----------------------------
class PredictRequest(BaseModel):
    t1: float
    t2: float
    t3: float
    CPI_MoM: float
    WPI_MoM: float
    SPI_MoM: float
    month: int  # 1–12

# ----------------------------
# Health check
# ----------------------------
@app.get("/")
def health():
    return {"status": "Pakistan Inflation Predictor API running"}

# ----------------------------
# Debug endpoint (keep it)
# ----------------------------
@app.get("/features")
def get_model_features():
    return {"model_features": FEATURE_ORDER}

# ----------------------------
# Prediction endpoint
# ----------------------------
@app.post("/predict")
def predict(req: PredictRequest):
    # Rolling means
    inflation_rm3 = np.mean([req.t1, req.t2, req.t3])
    cpi_rm3 = req.CPI_MoM
    wpi_rm3 = req.WPI_MoM
    spi_rm3 = req.SPI_MoM

    # Seasonality
    month_sin = math.sin(2 * math.pi * req.month / 12)
    month_cos = math.cos(2 * math.pi * req.month / 12)

    # Build row EXACTLY in model order
    row = {
        "inflation_lag_1": req.t1,
        "inflation_lag_2": req.t2,
        "inflation_lag_3": req.t3,

        "CPI_MoM_lag_1": req.CPI_MoM,
        "CPI_MoM_lag_2": req.CPI_MoM,
        "CPI_MoM_lag_3": req.CPI_MoM,

        "WPI_MoM_lag_1": req.WPI_MoM,
        "WPI_MoM_lag_2": req.WPI_MoM,
        "WPI_MoM_lag_3": req.WPI_MoM,

        "SPI_MoM_lag_1": req.SPI_MoM,
        "SPI_MoM_lag_2": req.SPI_MoM,
        "SPI_MoM_lag_3": req.SPI_MoM,

        "inflation_rolling_mean_3": inflation_rm3,
        "CPI_MoM_rolling_mean_3": cpi_rm3,
        "WPI_MoM_rolling_mean_3": wpi_rm3,
        "SPI_MoM_rolling_mean_3": spi_rm3,

        "month_sin": month_sin,
        "month_cos": month_cos,
    }

    df = pd.DataFrame([row], columns=FEATURE_ORDER)

    prediction = float(model.predict(df)[0])

    return {
        "predicted_inflation": round(prediction, 2)
    }

