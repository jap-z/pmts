import numpy as np
import pandas as pd
from src.predictor import PatternPredictor
from tqdm import tqdm
import argparse

def run_backtest(test_days=365):
    print(f"--- Walk-Forward Backtest (Last {test_days} Days) ---")
    
    predictor = PatternPredictor()
    data = np.load('data/btc_6h_embeddings.npz', allow_pickle=True)
    
    embeddings = data['embeddings']
    timestamps = pd.to_datetime(data['timestamps'])
    y_close_ret = data['y_close_ret']
    
    end_time = timestamps[-1]
    cutoff_time = end_time - pd.Timedelta(days=test_days)
    
    print(f"Cutoff Time (Test start): {cutoff_time}")
    
    # Identify test indices
    test_indices = np.where(timestamps >= cutoff_time)[0]
    print(f"Total Test Samples (6-hour windows): {len(test_indices)}\n")
    
    long_trades = []
    short_trades = []
    
    # Parameters for signal generation (Tuned for extremely high confidence)
    UP_PROB_THRESHOLD = 0.80
    DOWN_PROB_THRESHOLD = 0.20
    
    for idx in tqdm(test_indices, desc="Backtesting"):
        query_vec = embeddings[idx]
        query_ts = timestamps[idx]
        actual_return = y_close_ret[idx]
        
        # Max timestamp ensures we don't look at the future (or the current window itself)
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
        
        # Determine signal
        if prob_up >= UP_PROB_THRESHOLD:
            # LONG SIGNAL
            long_trades.append({
                'timestamp': query_ts,
                'prob_up': prob_up,
                'actual_return': actual_return,
                'won': actual_return > 0
            })
        elif prob_up <= DOWN_PROB_THRESHOLD:
            # SHORT SIGNAL
            short_trades.append({
                'timestamp': query_ts,
                'prob_down': 1 - prob_up,
                'actual_return': -actual_return, # Invert return for short
                'won': actual_return < 0
            })

    # --- Print Results ---
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=365, help='Days to backtest')
    args = parser.parse_args()
    
    run_backtest(test_days=args.days)
