import numpy as np
import pandas as pd
from src.predictor_v1_enhanced import PatternPredictor
from tqdm import tqdm
import argparse
from collections import defaultdict

def run_weekly_backtest(test_days=365):
    print(f"--- Weekly Breakdown Walk-Forward Backtest (Last {test_days} Days) [V1 Model + Enhanced Logic] ---")
    
    # Load V1 embeddings
    data = np.load('data/v1_backup/btc_6h_embeddings.npz', allow_pickle=True)
    embeddings = data['embeddings']
    timestamps = pd.to_datetime(data['timestamps'])
    y_close_ret = data['y_close_ret']
    
    predictor = PatternPredictor(raw_data_path='data/BTC_USDT_6h.csv', embedding_path='data/v1_backup/btc_6h_embeddings.npz')
    
    end_time = timestamps[-1]
    cutoff_time = end_time - pd.Timedelta(days=test_days)
    
    test_indices = np.where(timestamps >= cutoff_time)[0]
    print(f"Total Test Samples (6-hour windows): {len(test_indices)}\n")
    
    UP_PROB_THRESHOLD = 0.70
    DOWN_PROB_THRESHOLD = 0.30
    
    # Data structure to hold weekly performance
    weekly_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0})
    
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
        
        # Get ISO calendar year and week number
        year, week, _ = query_ts.isocalendar()
        week_key = f"{year}-W{week:02d}"
        
        trade_taken = False
        trade_pnl = 0.0
        trade_won = False
        
        if prob_up >= UP_PROB_THRESHOLD:
            trade_taken = True
            trade_pnl = actual_return
            trade_won = actual_return > 0
        elif prob_up <= DOWN_PROB_THRESHOLD:
            trade_taken = True
            trade_pnl = -actual_return
            trade_won = actual_return < 0
            
        if trade_taken:
            weekly_stats[week_key]['trades'] += 1
            weekly_stats[week_key]['pnl'] += trade_pnl
            if trade_won:
                weekly_stats[week_key]['wins'] += 1

    print("\n\n--- Weekly Performance Breakdown ---")
    
    winning_weeks = 0
    losing_weeks = 0
    breakeven_weeks = 0
    
    max_weekly_loss = 0.0
    max_weekly_gain = 0.0
    
    for week_key in sorted(weekly_stats.keys()):
        stats = weekly_stats[week_key]
        if stats['trades'] == 0:
            continue
            
        win_rate = (stats['wins'] / stats['trades']) * 100
        pnl_pct = stats['pnl'] * 100
        
        if pnl_pct > 0:
            winning_weeks += 1
            status = "🟢 WIN "
            max_weekly_gain = max(max_weekly_gain, pnl_pct)
        elif pnl_pct < 0:
            losing_weeks += 1
            status = "🔴 LOSS"
            max_weekly_loss = min(max_weekly_loss, pnl_pct)
        else:
            breakeven_weeks += 1
            status = "⚪ EVEN"
            
        print(f"Week {week_key} | {status} | Trades: {stats['trades']:2d} | WinRate: {win_rate:5.1f}% | PnL: {pnl_pct:+6.2f}%")
        
    print("\n--- Summary ---")
    print(f"Total Weeks Traded: {winning_weeks + losing_weeks + breakeven_weeks}")
    print(f"Winning Weeks: {winning_weeks}")
    print(f"Losing Weeks:  {losing_weeks}")
    print(f"Max Weekly Gain: +{max_weekly_gain:.2f}%")
    print(f"Max Weekly Drawdown: {max_weekly_loss:.2f}%")
    
if __name__ == "__main__":
    run_weekly_backtest()
