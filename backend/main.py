import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import datetime
import numpy as np
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Pakistan Inflation Predictor", version="2.0")

# --------------------------
# Load trained model
# --------------------------
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "model/model.pkl")

# Force-check model existence
if not os.path.exists(model_path):
    # Try alternate path just in case
    alternative_path = "model/model.pkl"
    if os.path.exists(alternative_path):
        model_path = alternative_path
    else:
        print(f"DEBUG: Current Dir: {os.getcwd()}")
        print(f"DEBUG: Files: {os.listdir(os.getcwd())}")
        raise FileNotFoundError(f"Trained model not found at {model_path}")

try:
    model = joblib.load(model_path)
    print(f"SUCCESS: Loaded model from {model_path}")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load model. {e}")
    raise e

# --------------------------
# Input schema
# --------------------------
class InflationInput(BaseModel):
    t1: float
    t2: float
    t3: float
    CPI_MoM: float = None
    WPI_MoM: float = None
    SPI_MoM: float = None

# --------------------------
# Prediction endpoint
# --------------------------
@app.post("/predict")
def predict(data: InflationInput):
    # Prepare DataFrame
    df = pd.DataFrame()

    # 1. Add Inflation Lags
    df['inflation_lag_1'] = [data.t1]
    df['inflation_lag_2'] = [data.t2]
    df['inflation_lag_3'] = [data.t3]

    # 2. Add Component Lags
    for col, val in zip(['CPI_MoM', 'WPI_MoM', 'SPI_MoM'], 
                        [data.CPI_MoM, data.WPI_MoM, data.SPI_MoM]):
        if val is None: val = 0.0
        df[f'{col}_lag_1'] = [val]
        df[f'{col}_lag_2'] = [val]
        df[f'{col}_lag_3'] = [val]

    # 3. Add Rolling Means
    df['inflation_rolling_mean_3'] = [(data.t1 + data.t2 + data.t3)/3]
    
    for col, val in zip(['CPI_MoM', 'WPI_MoM', 'SPI_MoM'], 
                        [data.CPI_MoM, data.WPI_MoM, data.SPI_MoM]):
        if val is None: val = 0.0
        df[f'{col}_rolling_mean_3'] = [val]

    # 4. Add Seasonal Features
    current_month = datetime.datetime.now().month
    df['month_sin'] = [np.sin(2 * np.pi * current_month / 12)]
    df['month_cos'] = [np.cos(2 * np.pi * current_month / 12)]

    # ---------------------------------------------------------
    # CRITICAL FIX: REORDER COLUMNS (Must match Model Training)
    # ---------------------------------------------------------
    expected_order = [
        'inflation_lag_1', 'inflation_lag_2', 'inflation_lag_3', 
        'CPI_MoM_lag_1', 'CPI_MoM_lag_2', 'CPI_MoM_lag_3', 
        'WPI_MoM_lag_1', 'WPI_MoM_lag_2', 'WPI_MoM_lag_3', 
        'SPI_MoM_lag_1', 'SPI_MoM_lag_2', 'SPI_MoM_lag_3', 
        'inflation_rolling_mean_3', 
        'CPI_MoM_rolling_mean_3', 'WPI_MoM_rolling_mean_3', 'SPI_MoM_rolling_mean_3', 
        'month_sin', 'month_cos'
    ]
    
    # Force the DataFrame to follow this exact order
    df = df[expected_order]

    # DEBUG PRINT to prove new code is running
    print("DEBUG: Columns sent to model:", df.columns.tolist())

    # Predict
    prediction = model.predict(df)[0]

    return {"predicted_inflation_next_month": round(float(prediction), 2)}

@app.get("/")
def read_root():
    return {"message": "Pakistan Inflation Predictor API is running."}

# --------------------------
# STARTUP COMMAND
# --------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting server on 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)