import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def load_and_clean_data(file_path: Path) -> pd.DataFrame:
    """Loads raw crypto data and ensures continuous time index."""
    df = pd.read_csv(file_path)
    # Binance timestamps are in ms
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('timestamp').set_index('timestamp')
    
    # Forward fill any missing minutes to prevent gaps in sequences
    df = df.resample('1min').ffill()
    return df

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generates stationary features suitable for Deep Learning."""
    features = pd.DataFrame(index=df.index)
    
    # 1. Log Returns (Stationary price representation)
    features['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # 2. Volatility (Rolling standard deviation of returns)
    features['volatility_15m'] = features['log_return'].rolling(window=15).std()
    features['volatility_60m'] = features['log_return'].rolling(window=60).std()
    
    # 3. Volume and Order Flow Features
    # Taker buy base asset volume is how much was bought by market orders
    # If taker_buy > 0.5 * volume, buyers are aggressive
    taker_buy = df['taker_buy_base_asset_volume']
    total_vol = df['volume']
    features['buy_pressure'] = np.where(total_vol > 0, taker_buy / total_vol, 0.5)
    
    # Normalize volume (z-score against recent 60-min history)
    features['volume_zscore'] = (total_vol - total_vol.rolling(60).mean()) / (total_vol.rolling(60).std() + 1e-8)
    
    # Drop rows with NaNs caused by the rolling windows
    features = features.dropna()
    return features

def scale_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Scales data strictly fitting on train to avoid look-ahead bias."""
    scaler = StandardScaler()
    
    # Fit only on training data
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_df), 
        index=train_df.index, 
        columns=train_df.columns
    )
    
    # Transform test data using the train scaler
    test_scaled = pd.DataFrame(
        scaler.transform(test_df), 
        index=test_df.index, 
        columns=test_df.columns
    )
    
    return train_scaled, test_scaled, scaler
