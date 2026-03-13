import numpy as np
import pandas as pd
from src.predictor_v1_enhanced import PatternPredictor
from tqdm import tqdm
import argparse

def run_backtest(test_days=365):
    print(f"--- Walk-Forward Backtest (Last {test_days} Days) [V1 Model + Enhanced Logic] ---")
    
    # Load V1 embeddings
    data = np.load('data/v1_backup/btc_6h_embeddings.npz', allow_pickle=True)
    embeddings = data['embeddings']
    timestamps = pd.to_datetime(data['timestamps'])
    y_close_ret = data['y_close_ret']
    
    # We pass the main CSV for regime calculation (it has more data but that's fine, it matches by timestamp)
    predictor = PatternPredictor(raw_data_path='data/BTC_USDT_6h.csv', embedding_path='data/v1_backup/btc_6h_embeddings.npz')
    
    end_time = timestamps[-1]
    cutoff_time = end_time - pd.Timedelta(days=test_days)
    
    print(f"Cutoff Time (Test start): {cutoff_time}")
    
    test_indices = np.where(timestamps >= cutoff_time)[0]
    print(f"Total Test Samples (6-hour windows): {len(test_indices)}\n")
    
    long_trades = []
    short_trades = []
    
    # Parameters for signal generation
    # We can be slightly looser here since the distance-weighting and strict regime filtering gives higher confidence
    UP_PROB_THRESHOLD = 0.70
    DOWN_PROB_THRESHOLD = 0.30
    
    for idx in tqdm(test_indices, desc="Backtesting"):
        query_vec = embeddings[idx]
        query_ts = timestamps[idx]
        actual_return = y_close_ret[idx]
        
        pred = predictor.get_prediction(
            query_vec, 
            query_timestamp=query_ts, 
            k=10, 
            filter_regime=True, 
            max_timestamp=query_ts
        )
        
        if not pred:
            continue
            
        prob_up = pred['prob_up']
        
        if prob_up >= UP_PROB_THRESHOLD:
            long_trades.append({
                'timestamp': query_ts,
                'prob_up': prob_up,
                'actual_return': actual_return,
                'won': actual_return > 0
            })
        elif prob_up <= DOWN_PROB_THRESHOLD:
            short_trades.append({
                'timestamp': query_ts,
                'prob_down': 1 - prob_up,
                'actual_return': -actual_return,
                'won': actual_return < 0
            })

    print("\n\n--- Backtest Results ---")
    total_longs = len(long_trades)
    total_shorts = len(short_trades)
    total_trades = total_longs + total_shorts
    
    if total_trades == 0:
        print("No trades generated with the current thresholds.")
        return
        
    long_wins = sum(1 for t in long_trades if t['won'])
    short_wins = sum(1 for t in short_trades if t['won'])
    
    long_pnl = sum(t['actual_return'] for t in long_trades)
    short_pnl = sum(t['actual_return'] for t in short_trades)
    total_pnl = long_pnl + short_pnl
    
    print(f"Total Signals Fired: {total_trades}")
    print(f"LONG Signals: {total_longs}")
    if total_longs > 0:
        print(f"  Long Win Rate: {long_wins/total_longs*100:.1f}%")
        print(f"  Long PnL: +{long_pnl*100:.2f}%")
        
    print(f"SHORT Signals: {total_shorts}")
    if total_shorts > 0:
        print(f"  Short Win Rate: {short_wins/total_shorts*100:.1f}%")
        print(f"  Short PnL: +{short_pnl*100:.2f}%")
        
    overall_win_rate = (long_wins + short_wins) / total_trades
    print(f"\nOVERALL WIN RATE: {overall_win_rate*100:.1f}%")
    print(f"OVERALL CUMULATIVE PnL (Unleveraged): +{total_pnl*100:.2f}%")
    
if __name__ == "__main__":
    run_backtest()
