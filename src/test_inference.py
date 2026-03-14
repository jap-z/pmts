import sys
import os
import torch
import pandas as pd
import numpy as np
from types import SimpleNamespace

# Add fincast_src to path
sys.path.append(os.path.join(os.getcwd(), 'fincast_src'))
from tools.inference_utils import FinCast_Inference, get_forecasts_f

def test_inference_manual():
    print("Testing manual inference pass...")
    config = SimpleNamespace()
    config.backend = "cpu"
    config.model_path = "fincast_model/v1.pth"
    config.model_version = "v1"
    config.data_path = "data/BTC_USDT_1d.csv"
    config.data_frequency = "1d"
    config.context_len = 128
    config.horizon_len = 7
    config.all_data = False
    config.columns_target = ['close']
    config.series_norm = False
    config.batch_size = 1
    config.forecast_mode = "mean"
    config.save_output = False
    config.plt_outputs = False
    config.num_experts = 4
    config.gating_top_n = 2
    config.load_from_compile = True

    fincast = FinCast_Inference(config)
    ds = fincast.inference_dataset
    loader = fincast._make_inference_loader(batch_size=1, num_workers=0)
    model = fincast.model_api
    
    print("Starting loop...")
    with torch.inference_mode():
        for x_ctx, x_pad, freq, _x_fut, meta in loader:
            print(f"Sensing window: {meta[0]['window_end']}")
            preds_out, _ = get_forecasts_f(model, x_ctx, freq=freq)
            print(f"Prediction: {preds_out[0][0]}")
            break
    print("Test finished successfully.")

if __name__ == "__main__":
    test_inference_manual()
