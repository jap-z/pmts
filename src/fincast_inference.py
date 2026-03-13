import sys
import os
import torch
import pandas as pd
import numpy as np
from types import SimpleNamespace

# Add fincast_src to path
sys.path.append(os.path.join(os.getcwd(), 'fincast_src'))

from tools.inference_utils import FinCast_Inference

def run_fincast_btc_test():
    print("--- FinCast Foundation Model: BTC 6h Test ---")
    
    # Configuration based on research notebook
    config = SimpleNamespace()
    config.backend = "cpu"
    config.model_path = "fincast_model/v1.pth"
    config.model_version = "v1"
    
    # We use our actual 10-year dataset
    config.data_path = "data/BTC_USDT_6h.csv"
    config.data_frequency = "6h"
    config.context_len = 512
    config.horizon_len = 4
    
    config.all_data = False
    config.columns_target = ['close']
    config.series_norm = True
    config.batch_size = 1
    config.forecast_mode = "mean"
    config.save_output = False
    config.plt_outputs = False

    try:
        # Load raw data to calculate the exact same normalization parameters used by the Dataset class
        raw_df = pd.read_csv(config.data_path)
        # The dataset class drops NaNs and then normalizes
        series = pd.to_numeric(raw_df['close'], errors="coerce").dropna().to_numpy(dtype=np.float32)
        mu = float(series.mean())
        sigma = float(series.std(ddof=0))
        if sigma == 0.0: sigma = 1.0
        
        print(f"Loading 1B Parameter weights...")
        fincast = FinCast_Inference(config)
        
        print(f"Running inference...")
        preds, mapping, _ = fincast.run_inference()
        
        print("\n--- INFERENCE COMPLETE ---")
        last_price = series[-1]
        last_time = raw_df.iloc[-1, 0] # Assuming first col is timestamp
        
        print(f"Base Time: {last_time}")
        print(f"Current Price: ${last_price:.2f}")
        print(f"Normalization Params: mu={mu:.2f}, sigma={sigma:.2f}")
        
        # In v1, preds is [ [step1, step2, ... stepN] ]
        forecast_norm = preds[0]
        
        print("\nPredicted next 24 hours (6h intervals):")
        for i, val_norm in enumerate(forecast_norm):
            # Inverse Transform: y = y_norm * sigma + mu
            val_abs = (val_norm * sigma) + mu
            
            diff = val_abs - last_price
            pct = (diff / last_price) * 100
            print(f" +{(i+1)*6}h: ${val_abs:.2f} ({pct:+.2f}%)")
            
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_fincast_btc_test()
