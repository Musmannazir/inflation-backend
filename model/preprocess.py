import pandas as pd
import numpy as np

def create_features(df):
    df = df.copy()
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # List of columns to create lag and rolling features for
    cols_to_lag = ['inflation', 'CPI_MoM', 'WPI_MoM', 'SPI_MoM']
    
    # Create lag features (last 3 months)
    for col in cols_to_lag:
        for lag in range(1, 4):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Create rolling mean features (3 months)
    for col in cols_to_lag:
        df[f'{col}_rolling_mean_3'] = df[col].rolling(window=3).mean()
    
    # Optional: Add seasonal features for month
    df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)
    
    # Drop rows with NaN values (first 3 rows due to lag/rolling)
    df = df.dropna().reset_index(drop=True)
    
    return df
