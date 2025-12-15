import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import datetime
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="Pakistan Inflation Predictor", version="2.0")

# --------------------------
# Load trained model
# --------------------------
# Use strict path joining to find the model inside the backend folder
model_path = os.path.join(os.path.dirname(__file__), "model/model.pkl")

if not os.path.exists(model_path):
    # Fallback: check if we are already inside the model folder (sometimes happens in containers)
    if os.path.exists("model/model.pkl"):
        model_path = "model/model.pkl"
    else:
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
# SERVER STARTUP (CRITICAL FOR RAILWAY)
# --------------------------
if __name__ == "__main__":
    import uvicorn
    # Railway provides the PORT environment variable.
    # We must listen on 0.0.0.0 to be accessible from the outside.
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)