import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import datetime
import numpy as np
import uvicorn # Import this at the top

# Initialize FastAPI app
app = FastAPI(title="Pakistan Inflation Predictor", version="2.0")

# --------------------------
# Load trained model
# --------------------------
# Robust path finding to handle different server environments
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "model/model.pkl")

# Fallback: specific check if we are inside the backend folder already
if not os.path.exists(model_path):
    # Try looking just in the local folder
    alternative_path = "model/model.pkl"
    if os.path.exists(alternative_path):
        model_path = alternative_path
    else:
        # Print directory contents to help with debugging logs if it fails
        print(f"DEBUG: Current Directory: {os.getcwd()}")
        print(f"DEBUG: Directory Contents: {os.listdir(os.getcwd())}")
        raise FileNotFoundError(f"Trained model not found at {model_path}")

model = joblib.load(model_path)
print(f"Loaded model from {model_path}")

# --------------------------
# Input schema
# --------------------------
class InflationInput(BaseModel):
    # Only last 3 months of inflation rates are required
    t1: float  # Last month
    t2: float  # 2 months ago
    t3: float  # 3 months ago
    CPI_MoM: float = None
    WPI_MoM: float = None
    SPI_MoM: float = None

# --------------------------
# Prediction endpoint
# --------------------------
@app.post("/predict")
def predict(data: InflationInput):
    # Prepare DataFrame with all required features
    df = pd.DataFrame()

    # Lag features (inflation)
    df['inflation_lag_1'] = [data.t1]
    df['inflation_lag_2'] = [data.t2]
    df['inflation_lag_3'] = [data.t3]

    # Rolling mean
    df['inflation_rolling_mean_3'] = [(data.t1 + data.t2 + data.t3)/3]

    # If monthly CPI/WPI/SPI are provided, create lag & rolling for them too
    # Otherwise fill with zeros
    for col, val in zip(['CPI_MoM', 'WPI_MoM', 'SPI_MoM'], 
                        [data.CPI_MoM, data.WPI_MoM, data.SPI_MoM]):
        if val is None:
            val = 0.0
        df[f'{col}_lag_1'] = [val]
        df[f'{col}_lag_2'] = [val]
        df[f'{col}_lag_3'] = [val]
        df[f'{col}_rolling_mean_3'] = [val]

    # Add seasonal features (month sin/cos)
    current_month = datetime.datetime.now().month
    df['month_sin'] = [np.sin(2 * np.pi * current_month / 12)]
    df['month_cos'] = [np.cos(2 * np.pi * current_month / 12)]

    # Predict
    prediction = model.predict(df)[0]

    # Return response
    return {"predicted_inflation_next_month": round(float(prediction), 2)}

# --------------------------
# Root endpoint for testing
# --------------------------
@app.get("/")
def read_root():
    return {"message": "Pakistan Inflation Predictor API is running."}

# --------------------------
# STARTUP COMMAND (THIS WAS MISSING)
# --------------------------
if __name__ == "__main__":
    # Railway sets the PORT environment variable. We MUST use it.
    port = int(os.environ.get("PORT", 8080))
    # Host must be "0.0.0.0" to allow outside connections
    uvicorn.run(app, host="0.0.0.0", port=port)