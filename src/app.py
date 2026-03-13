import sys
import os
import time
import json
import torch
import pandas as pd
import numpy as np
from types import SimpleNamespace
from flask import Flask, Response, send_from_directory, jsonify
from flask_cors import CORS

# Add fincast_src to path
sys.path.append(os.path.join(os.getcwd(), 'fincast_src'))
from tools.inference_utils import FinCast_Inference

app = Flask(__name__, static_folder='../public')
CORS(app)

# Global model holder so we only load it once
fincast_model = None
raw_df = None
sigma = 1.0
mu = 0.0

def init_model():
    global fincast_model, raw_df, sigma, mu
    if fincast_model is not None:
        return

    print("Initializing FinCast 1D Model...")
    config = SimpleNamespace()
    config.backend = "cpu"
    config.model_path = "fincast_model/v1.pth"
    config.model_version = "v1"
    config.data_path = "data/BTC_USDT_1d.csv"
    config.data_frequency = "1d" # Changed to 1d
    config.context_len = 128     # 4 months of context
    config.horizon_len = 7       # 1 week prediction
    config.all_data = True       # Sliding windows
    config.columns_target = ['close']
    config.series_norm = True
    config.batch_size = 1
    config.forecast_mode = "mean"
    config.save_output = False
    config.plt_outputs = False

    raw_df = pd.read_csv(config.data_path, index_col=0, parse_dates=True).sort_index()
    series = pd.to_numeric(raw_df['close'], errors="coerce").dropna().to_numpy(dtype=np.float32)
    mu = float(series.mean())
    sigma = float(series.std(ddof=0))
    if sigma == 0.0: sigma = 1.0

    fincast_model = FinCast_Inference(config)
    print("Model initialized.")

@app.route('/pmts/')
def index():
    return send_from_directory('../public', 'index.html')

@app.route('/pmts/stream_backtest')
def stream_backtest():
    init_model()
    
    def generate():
        test_days = 365
        print(f"Starting stream backtest for last {test_days} days...")
        
        # We need to test the last 365 days. 
        # For a 1d dataset, 1 window = 1 day. So we need the last 365 index records.
        if fincast_model.inference_dataset.sliding_windows:
            all_records = fincast_model.inference_dataset.index_records
            original_records = list(all_records) # Backup
            fincast_model.inference_dataset.index_records = all_records[-test_days:]
            
        # Run inference on all 365 windows at once? 
        # No, that blocks the thread and we can't stream.
        # Actually, run_inference() does a loop inside. We can't easily yield from inside it.
        # Let's write a custom inference loop here so we can yield after every prediction!
        
        ds = fincast_model.inference_dataset
        loader = fincast_model._make_inference_loader(batch_size=1, num_workers=0)
        model = fincast_model.model_api
        
        series_data = pd.to_numeric(raw_df['close'], errors="coerce").dropna().to_numpy(dtype=np.float32)
        timestamps = raw_df.index
        
        wins = 0
        total = 0
        pnl = 0.0
        
        with torch.inference_mode():
            for i, (x_ctx, x_pad, freq, _x_fut, meta) in enumerate(loader):
                # x_ctx is the normalized context window
                from tools.inference_utils import get_forecasts_f
                preds_out, _ = get_forecasts_f(model, x_ctx, freq=freq)
                
                if isinstance(preds_out, torch.Tensor):
                    out_np = preds_out.detach().cpu().numpy()
                else:
                    out_np = np.asarray(preds_out)
                    
                if out_np.ndim == 1: out_np = out_np[None, :]
                
                # We have batch_size=1, so we take index 0
                pred_norm = out_np[0] 
                pred_abs = (pred_norm * sigma) + mu
                
                # Get metadata
                m = meta[0]
                end_idx = int(m['window_end'])
                start_idx = int(m['window_start'])
                
                base_price = series_data[end_idx]
                
                # Extract the exact context prices for the chart
                context_prices = series_data[start_idx : end_idx + 1].tolist()
                context_times = [str(t) for t in timestamps[start_idx : end_idx + 1]]
                
                # Extract actual future prices
                horizon = 7
                if end_idx + horizon >= len(series_data):
                    break # Skip if we don't have future data
                    
                actual_future_prices = series_data[end_idx + 1 : end_idx + horizon + 1].tolist()
                future_times = [str(t) for t in timestamps[end_idx + 1 : end_idx + horizon + 1]]
                
                predicted_prices = pred_abs.tolist()
                
                # Trade logic (predicting day 7 relative to base)
                actual_return = (actual_future_prices[-1] - base_price) / base_price
                pred_return = (predicted_prices[-1] - base_price) / base_price
                
                # Signal thresholds
                trade_taken = False
                won = False
                trade_pnl = 0.0
                signal = "NEUTRAL"
                
                if pred_return > 0.01: # Predict >1% up
                    trade_taken = True
                    signal = "LONG"
                    trade_pnl = actual_return
                    won = actual_return > 0
                elif pred_return < -0.01: # Predict >1% down
                    trade_taken = True
                    signal = "SHORT"
                    trade_pnl = -actual_return
                    won = actual_return < 0
                    
                if trade_taken:
                    total += 1
                    pnl += trade_pnl
                    if won: wins += 1
                
                win_rate = (wins / total * 100) if total > 0 else 0
                
                data = {
                    "step": i + 1,
                    "total_steps": test_days,
                    "timestamp": str(timestamps[end_idx]),
                    "base_price": float(base_price),
                    "context_times": context_times[-30:], # Only send last 30 days of context for chart
                    "context_prices": context_prices[-30:],
                    "future_times": future_times,
                    "actual_prices": actual_future_prices,
                    "predicted_prices": predicted_prices,
                    "signal": signal,
                    "trade_taken": trade_taken,
                    "won": won,
                    "trade_pnl": float(trade_pnl),
                    "total_trades": total,
                    "win_rate": float(win_rate),
                    "cumulative_pnl": float(pnl)
                }
                
                yield f"data: {json.dumps(data)}\n\n"
                
        # Restore records
        if fincast_model.inference_dataset.sliding_windows:
            fincast_model.inference_dataset.index_records = original_records

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
