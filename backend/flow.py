from prefect import flow, task
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib
import os

# --- TASK 1: LOAD DATA ---
@task(name="Load Data", retries=3)
def load_data():
    # Adjust path to where your data lives in the repo
    data_path = os.path.join(os.path.dirname(__file__), "data", "pakistan_cpi.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")
    
    df = pd.read_csv(data_path)
    df.rename(columns={'Date': 'date', 'Inflation_Rate': 'inflation'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    return df

# --- TASK 2: FEATURE ENGINEERING ---
@task(name="Create Features")
def create_features(df: pd.DataFrame):
    df = df.sort_values('date').reset_index(drop=True)
    cols_to_lag = ['inflation', 'CPI_MoM', 'WPI_MoM', 'SPI_MoM']
    
    # Lag & Rolling Features
    for col in cols_to_lag:
        for lag in range(1, 4):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        df[f'{col}_rolling_mean_3'] = df[col].shift(1).rolling(3, min_periods=1).mean()
        
    # Seasonality
    df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)
    
    return df.dropna().reset_index(drop=True)

# --- TASK 3: TRAIN MODELS ---
@task(name="Train Regression & Clustering")
def train_models(df: pd.DataFrame):
    # Prepare Regression Data
    feature_cols = [c for c in df.columns if '_lag_' in c or '_rolling_' in c or 'month_' in c]
    X = df[feature_cols]
    y = df['inflation']
    
    # Train XGBoost
    reg_model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05)
    reg_model.fit(X, y)
    
    # Prepare Clustering Data
    cluster_cols = ['inflation', 'CPI_MoM', 'WPI_MoM', 'SPI_MoM']
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df[cluster_cols])
    
    # Train K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_cluster)
    
    return reg_model, kmeans, scaler, X, y

# --- TASK 4: EVALUATE ---
@task(name="Evaluate Model")
def evaluate_model(model, X, y):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    print(f"✅ Model Performance MAE: {mae:.4f}")
    return mae

# --- MAIN FLOW ---
@flow(name="Inflation Training Pipeline")
def training_pipeline():
    print("🚀 Starting Training Pipeline...")
    
    raw_data = load_data()
    processed_data = create_features(raw_data)
    
    reg_model, kmeans, scaler, X, y = train_models(processed_data)
    
    mae = evaluate_model(reg_model, X, y)
    
    # In a real scenario, you would save the models here using joblib
    print("🎉 Pipeline Finished Successfully!")

if __name__ == "__main__":
    training_pipeline()