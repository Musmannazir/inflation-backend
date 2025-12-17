from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import math
import os

app = FastAPI()

# ----------------------------
# 1. Enable CORS (Fixes 403 Error)
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# ----------------------------
# 2. Load Models (Regression + Clustering)
# ----------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

# Regression Model (Inflation Prediction)
model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))

# Clustering Models (Risk Analysis)
kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_cluster.pkl"))
rank_mapping = joblib.load(os.path.join(MODEL_DIR, "cluster_mapping.pkl"))

# EXACT feature order for Regression
FEATURE_ORDER = [
    "inflation_lag_1", "inflation_lag_2", "inflation_lag_3",
    "CPI_MoM_lag_1", "CPI_MoM_lag_2", "CPI_MoM_lag_3",
    "WPI_MoM_lag_1", "WPI_MoM_lag_2", "WPI_MoM_lag_3",
    "SPI_MoM_lag_1", "SPI_MoM_lag_2", "SPI_MoM_lag_3",
    "inflation_rolling_mean_3",
    "CPI_MoM_rolling_mean_3", "WPI_MoM_rolling_mean_3", "SPI_MoM_rolling_mean_3",
    "month_sin", "month_cos",
]

# ----------------------------
# Request Schema
# ----------------------------
class PredictRequest(BaseModel):
    t1: float       # Inflation Last Month
    t2: float       # Inflation 2 Months Ago
    t3: float       # Inflation 3 Months Ago
    CPI_MoM: float
    WPI_MoM: float
    SPI_MoM: float
    month: int      # 1–12

# ----------------------------
# Endpoints
# ----------------------------
@app.get("/")
def health():
    return {"status": "Pakistan Inflation Predictor API running"}

@app.post("/predict")
def predict(req: PredictRequest):
    # --- TASK 1: REGRESSION (Predicting the Number) ---
    
    # Calculate Rolling Means
    inflation_rm3 = np.mean([req.t1, req.t2, req.t3])
    
    # Seasonality
    month_sin = math.sin(2 * math.pi * req.month / 12)
    month_cos = math.cos(2 * math.pi * req.month / 12)

    # Build row for Regression Model
    row = {
        "inflation_lag_1": req.t1,
        "inflation_lag_2": req.t2,
        "inflation_lag_3": req.t3,
        "CPI_MoM_lag_1": req.CPI_MoM, "CPI_MoM_lag_2": req.CPI_MoM, "CPI_MoM_lag_3": req.CPI_MoM,
        "WPI_MoM_lag_1": req.WPI_MoM, "WPI_MoM_lag_2": req.WPI_MoM, "WPI_MoM_lag_3": req.WPI_MoM,
        "SPI_MoM_lag_1": req.SPI_MoM, "SPI_MoM_lag_2": req.SPI_MoM, "SPI_MoM_lag_3": req.SPI_MoM,
        "inflation_rolling_mean_3": inflation_rm3,
        "CPI_MoM_rolling_mean_3": req.CPI_MoM,
        "WPI_MoM_rolling_mean_3": req.WPI_MoM,
        "SPI_MoM_rolling_mean_3": req.SPI_MoM,
        "month_sin": month_sin,
        "month_cos": month_cos,
    }

    df = pd.DataFrame([row], columns=FEATURE_ORDER)
    pred_inflation = float(model.predict(df)[0])

    # --- TASK 2: CLUSTERING (Risk Analysis) ---
    
    # Prepare input for K-Means: [inflation, CPI, WPI, SPI]
    cluster_input = np.array([[req.t1, req.CPI_MoM, req.WPI_MoM, req.SPI_MoM]])
    
    # Scale the input (Crucial for K-Means)
    cluster_input_scaled = scaler.transform(cluster_input)
    
    # Get Cluster ID
    cluster_id = int(kmeans.predict(cluster_input_scaled)[0])
    
    # Map Cluster ID to Risk Level (0=Low, 1=Med, 2=High) using our trained mapping
    risk_level = rank_mapping.get(cluster_id, 1)
    
    # Define readable labels
    risk_labels = {
        0: "Stable (Low Risk)",
        1: "Warning (Moderate Risk)",
        2: "Crisis (High Risk)"
    }
    status_text = risk_labels.get(risk_level, "Unknown")

    return {
        "predicted_inflation": round(pred_inflation, 2),
        "cluster_status": status_text,
        "risk_level": risk_level
    }