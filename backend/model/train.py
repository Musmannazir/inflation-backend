import os
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import numpy as np

# --------------------------
# Feature engineering
# --------------------------
def create_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    cols_to_lag = ['inflation', 'CPI_MoM', 'WPI_MoM', 'SPI_MoM']
    
    # Lag features
    for col in cols_to_lag:
        for lag in range(1, 4):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Rolling mean
    for col in cols_to_lag:
        df[f'{col}_rolling_mean_3'] = df[col].shift(1).rolling(3, min_periods=1).mean()
    
    # Seasonal features
    df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)
    
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
df.columns = df.columns.str.strip()
df.rename(columns={'Date': 'date', 'Inflation_Rate': 'inflation'}, inplace=True)

df['date'] = pd.to_datetime(df['date'], errors='coerce')
numeric_cols = ['inflation', 'CPI_MoM', 'WPI_MoM', 'SPI_MoM']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=['date', 'inflation']).reset_index(drop=True)

print(f"Loaded {len(df)} rows.")

# --------------------------
# TASK 1: REGRESSION (XGBoost)
# --------------------------
print("\n--- Training Regression Model (Task 1) ---")
df_feat = create_features(df)
feature_cols = [col for col in df_feat.columns if '_lag_' in col or '_rolling_mean' in col or 'month_' in col]
X = df_feat[feature_cols]
y = df_feat['inflation']

split = int(len(df_feat) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

reg_model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
reg_model.fit(X_train, y_train)

preds = reg_model.predict(X_test)
print(f"Regression MAE: {mean_absolute_error(y_test, preds)}")

# --------------------------
# TASK 2: CLUSTERING (K-Means)
# --------------------------
print("\n--- Training Clustering Model (Task 2) ---")
# We cluster based on the core indicators to define "Economic State"
cluster_cols = ['inflation', 'CPI_MoM', 'WPI_MoM', 'SPI_MoM']
X_cluster = df[cluster_cols].dropna()

# Scaling is mandatory for K-Means
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

# Train K-Means (3 Clusters: Low, Medium, High Risk)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_cluster_scaled)

# Logic to map Cluster ID -> Meaningful Label (Low/Med/High)
# We calculate the average inflation for each cluster center to rank them.
centers = scaler.inverse_transform(kmeans.cluster_centers_)
inflation_idx = cluster_cols.index('inflation')
cluster_avg_inflation = centers[:, inflation_idx]

# Create a mapping dictionary: {cluster_id: 0 (Low), 1 (Med), 2 (High)}
sorted_indices = np.argsort(cluster_avg_inflation) # Returns indices sorted by inflation value
rank_mapping = {original_id: rank for rank, original_id in enumerate(sorted_indices)}

print(f"Cluster Ranks (0=Low, 2=High): {rank_mapping}")

# --------------------------
# Save Everything
# --------------------------
model_dir = os.path.join(os.path.dirname(__file__), "../model")
joblib.dump(reg_model, os.path.join(model_dir, "model.pkl"))
joblib.dump(kmeans, os.path.join(model_dir, "kmeans.pkl"))
joblib.dump(scaler, os.path.join(model_dir, "scaler_cluster.pkl"))
joblib.dump(rank_mapping, os.path.join(model_dir, "cluster_mapping.pkl"))

print("All models saved successfully!")