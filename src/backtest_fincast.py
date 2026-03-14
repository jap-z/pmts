import sys
import os
import torch
import pandas as pd
import numpy as np
from types import SimpleNamespace
from tqdm import tqdm

# Add fincast_src to path
sys.path.append(os.path.join(os.getcwd(), 'fincast_src'))

from tools.inference_utils import FinCast_Inference

def run_fincast_weekly_backtest(test_days=2):
    print(f"--- FinCast Foundation Model: {test_days}-Day Backtest ---")
    
    config = SimpleNamespace()
    config.backend = "cpu"
    config.model_path = "fincast_model/v1.pth"
    config.model_version = "v1"
    config.data_path = "data/BTC_USDT_6h.csv"
    config.data_frequency = "6h"
    config.context_len = 512
    config.horizon_len = 4 # Predict 24h
    config.all_data = True # Important: this enables sliding windows
    config.columns_target = ['close']
    config.series_norm = True
    config.batch_size = 1
    config.forecast_mode = "mean"
    config.save_output = False
    config.plt_outputs = False

    try:
        # 1. Load the full dataset to get ground truth and normalization params
        raw_df = pd.read_csv(config.data_path, index_col=0, parse_dates=True).sort_index()
        series = pd.to_numeric(raw_df['close'], errors="coerce").dropna().to_numpy(dtype=np.float32)
        mu = float(series.mean())
        sigma = float(series.std(ddof=0))
        if sigma == 0.0: sigma = 1.0
        
        # 2. Setup FinCast
        print("Loading Model...")
        fincast = FinCast_Inference(config)
        
        # 3. Filter for the last week of test samples
        test_samples_count = test_days * 4 # 4 candles per day
        print(f"Running backtest on the last {test_samples_count} windows...")
        
        # Optimization: We don't want to run inference on 12,000 windows if we only want 28.
        # We manually override the dataset's index records to only include the last N.
        if fincast.inference_dataset.sliding_windows:
            fincast.inference_dataset.index_records = fincast.inference_dataset.index_records[-test_samples_count:]
        
        preds, mapping, _ = fincast.run_inference()
        
        # mapping contains metadata for every window
        results = []
        
        # Calculate returns for the last week
        for i in range(len(mapping)):
            meta = mapping.iloc[i]
            # window_end is the index in the raw series
            end_idx = int(meta['window_end'])
            
            # Reality check: what actually happened in the next 4 candles?
            if end_idx + 4 >= len(series):
                continue
                
            base_price = series[end_idx]
            actual_future_price = series[end_idx + 4]
            actual_return = (actual_future_price - base_price) / base_price
            
            # Model prediction (Inverse Scaled)
            pred_norm = preds[i][-1] # Get the 24h prediction (last step of horizon)
            pred_abs = (pred_norm * sigma) + mu
            pred_return = (pred_abs - base_price) / base_price
            
            results.append({
                'time': raw_df.index[end_idx],
                'actual': actual_return,
                'pred': pred_return,
                'correct': (actual_return > 0 and pred_return > 0) or (actual_return < 0 and pred_return < 0)
            })

        # 4. Aggregate Results
        print("\n--- RESULTS ---")
        wins = sum(1 for r in results if r['correct'])
        total = len(results)
        
        if total > 0:
            print(f"Total Prediction Points: {total}")
            print(f"Win Rate: {(wins/total)*100:.1f}%")
            
            # Simple PnL (long only if pred > 0, short only if pred < 0)
            pnl = sum(r['actual'] if r['pred'] > 0 else -r['actual'] for r in results)
            print(f"Cumulative PnL: {pnl*100:.2f}%")
        else:
            print("No testable samples found in the last week.")
            
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_fincast_weekly_backtest()
