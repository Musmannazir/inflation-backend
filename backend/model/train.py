import os
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# --------------------------
# Feature engineering (updated to use multiple features & seasonal info)
# --------------------------
def create_features(df):
    df = df.copy()
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Columns to create lag/rolling features for
    cols_to_lag = ['inflation', 'CPI_MoM', 'WPI_MoM', 'SPI_MoM']
    
    # Lag features (1,2,3 months)
    for col in cols_to_lag:
        for lag in range(1, 4):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Rolling mean (3 months)
    for col in cols_to_lag:
        df[f'{col}_rolling_mean_3'] = df[col].shift(1).rolling(3, min_periods=1).mean()
    
    # Seasonal features (month sin/cos)
    df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)
    
    # Drop rows with NaNs created by shift/rolling
    df = df.dropna().reset_index(drop=True)
    
    return df

# --------------------------
# Load dataset
# --------------------------
data_folder = os.path.join(os.path.dirname(__file__), "../data")
csv_file = os.path.join(data_folder, "pakistan_cpi.csv")

if not os.path.exists(csv_file):
    raise ValueError(f"CSV file not found: {csv_file}")

df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()  # Remove whitespace

# Standardize column names
df.rename(columns={'Date': 'date', 'Inflation_Rate': 'inflation'}, inplace=True)

# Convert types and drop invalid rows
df['date'] = pd.to_datetime(df['date'], errors='coerce')
numeric_cols = ['inflation', 'CPI_MoM', 'WPI_MoM', 'SPI_MoM', 'CPI_YoY', 'WPI_YoY', 'SPI_YoY']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=['date', 'inflation']).reset_index(drop=True)

print(f"Loaded {len(df)} valid rows from {csv_file}")
print(df.head())

# --------------------------
# Feature engineering
# --------------------------
df_feat = create_features(df)
print(f"Rows after feature creation: {len(df_feat)}")
print(df_feat.head())

# --------------------------
# Prepare features & target
# --------------------------
# Use all lag, rolling, and seasonal features
feature_cols = [col for col in df_feat.columns if '_lag_' in col or '_rolling_mean' in col or 'month_' in col]
X = df_feat[feature_cols]
y = df_feat['inflation']

print("Feature matrix shape:", X.shape)
print("Target vector shape:", y.shape)

# --------------------------
# Train-test split (time-based)
# --------------------------
split = int(len(df_feat) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --------------------------
# Train model
# --------------------------
model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# --------------------------
# Evaluate
# --------------------------
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print("MAE on test set:", mae)

# --------------------------
# Save model
# --------------------------
model_path = os.path.join(os.path.dirname(__file__), "../model/model.pkl")
joblib.dump(model, model_path)
print(f"Model saved at {model_path}")
