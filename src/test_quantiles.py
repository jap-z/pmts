import sys
import os
import torch
import pandas as pd
import numpy as np
from types import SimpleNamespace

# Add fincast_src to path
sys.path.append(os.path.join(os.getcwd(), 'fincast_src'))
from tools.inference_utils import FinCast_Inference, get_forecasts_f

def test_quantiles():
    print("Testing FinCast Quantile Outputs...")
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
    config.forecast_mode = "median" # Switch to median to see if it's different
    config.save_output = False
    config.plt_outputs = False
    config.num_experts = 4
    config.gating_top_n = 2
    config.load_from_compile = True

    fincast = FinCast_Inference(config)
    ds = fincast.inference_dataset
    loader = fincast._make_inference_loader(batch_size=1, num_workers=0)
    model = fincast.model_api
    
    with torch.inference_mode():
        for x_ctx, x_pad, freq, _x_fut, meta in loader:
            print(f"Context shape: {x_ctx.shape}")
            print(f"Last 5 context prices: {x_ctx[0, -5:].numpy()}")
            
            preds_out, full_pred = get_forecasts_f(model, x_ctx, freq=freq)
            
            out_np = preds_out.detach().cpu().numpy() if isinstance(preds_out, torch.Tensor) else np.asarray(preds_out)
            full_np = full_pred.detach().cpu().numpy() if isinstance(full_pred, torch.Tensor) else np.asarray(full_pred)
            
            print(f"\nMedian Prediction (next 7 days):")
            print(out_np[0])
            
            print(f"\nFull Quantile Output Shape: {full_np.shape}")
            # Shape should be [Batch, Horizon, 1 + Quantiles]
            # Index 0 is mean, 1-9 are quantiles
            print("Mean (Index 0):", full_np[0, :, 0])
            print("10th Percentile (Pessimistic):", full_np[0, :, 1])
            print("50th Percentile (Median):", full_np[0, :, 5])
            print("90th Percentile (Optimistic):", full_np[0, :, 9])
            
            break

if __name__ == "__main__":
    test_quantiles()
