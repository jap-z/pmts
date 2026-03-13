import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import os
import argparse
from tqdm import tqdm

def create_features_and_labels(input_file='data/BTC_USDT_6h.csv', output_dir='data', window_size=28, forecast_horizon=4):
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, index_col=0, parse_dates=True)
    df = df.sort_index()

    # Calculate log returns relative to previous close to capture the shape accurately
    # Adding a small epsilon to volume to avoid log(0)
    print("Calculating log returns...")
    prev_close = df['close'].shift(1)
    
    # We use log(current / previous_close) for price features
    features_df = pd.DataFrame(index=df.index)
    features_df['log_open'] = np.log(df['open'] / prev_close)
    features_df['log_high'] = np.log(df['high'] / prev_close)
    features_df['log_low'] = np.log(df['low'] / prev_close)
    features_df['log_close'] = np.log(df['close'] / prev_close)
    features_df['log_vol'] = np.log((df['volume'] + 1e-8) / (df['volume'].shift(1) + 1e-8))

    # Drop the first row since it will have NaNs from the shift
    features_df = features_df.dropna()
    df = df.loc[features_df.index]

    print(f"Generating {window_size}-candle windows with {forecast_horizon}-candle forward labels...")
    
    features = features_df.values
    closes = df['close'].values
    timestamps = df.index.values

    X = []
    y_close_ret = []
    y_max_ret = []
    y_min_ret = []
    valid_timestamps = []

    # Pre-instantiate scaler for performance (we will fit per sample)
    scaler = RobustScaler()

    # Iterate through the dataset
    num_samples = len(features) - window_size - forecast_horizon
    
    for i in tqdm(range(num_samples), desc="Processing windows"):
        # The 28-candle window
        window = features[i : i + window_size]
        
        # Sample Normalization: Robust Scaling per window across the time axis
        # This normalizes the volatility of the specific week, isolating the "shape"
        window_scaled = scaler.fit_transform(window)
        
        X.append(window_scaled)
        
        # The last close price inside the window
        base_close = closes[i + window_size - 1]
        
        # The future window (next 4 candles)
        future_closes = closes[i + window_size : i + window_size + forecast_horizon]
        future_highs = df['high'].values[i + window_size : i + window_size + forecast_horizon]
        future_lows = df['low'].values[i + window_size : i + window_size + forecast_horizon]
        
        # Calculate truth labels (percentage returns)
        # 1. 24h Close Return
        close_ret = (future_closes[-1] - base_close) / base_close
        # 2. 24h Max Excursion Up
        max_ret = (np.max(future_highs) - base_close) / base_close
        # 3. 24h Max Excursion Down
        min_ret = (np.min(future_lows) - base_close) / base_close
        
        y_close_ret.append(close_ret)
        y_max_ret.append(max_ret)
        y_min_ret.append(min_ret)
        
        # Record the timestamp of the LAST candle in the window
        valid_timestamps.append(timestamps[i + window_size - 1])

    X = np.array(X, dtype=np.float32)
    y_close_ret = np.array(y_close_ret, dtype=np.float32)
    y_max_ret = np.array(y_max_ret, dtype=np.float32)
    y_min_ret = np.array(y_min_ret, dtype=np.float32)
    valid_timestamps = np.array(valid_timestamps)

    print(f"Generated {len(X)} samples.")
    print(f"X shape: {X.shape}") # Expected: (N, 28, 5)
    
    # Save datasets
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    out_file = os.path.join(output_dir, 'btc_6h_features.npz')
    np.savez_compressed(
        out_file, 
        X=X, 
        y_close_ret=y_close_ret, 
        y_max_ret=y_max_ret,
        y_min_ret=y_min_ret,
        timestamps=valid_timestamps
    )
    print(f"Data successfully saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate features and labels for PMTS")
    parser.add_argument('--input', type=str, default='data/BTC_USDT_6h.csv', help='Input CSV file')
    parser.add_argument('--window', type=int, default=28, help='Window size (e.g. 28 candles = 7 days at 6h)')
    parser.add_argument('--horizon', type=int, default=4, help='Forecast horizon (e.g. 4 candles = 24h)')
    
    args = parser.parse_args()
    create_features_and_labels(input_file=args.input, window_size=args.window, forecast_horizon=args.horizon)
