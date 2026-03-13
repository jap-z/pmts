import numpy as np
import pandas as pd
from src.vector_db import VectorDatabase
import os
import argparse

class PatternPredictor:
    def __init__(self, raw_data_path='data/BTC_USDT_6h.csv', embedding_path='data/btc_6h_embeddings.npz'):
        self.vdb = VectorDatabase()
        self.vdb.load_data(embedding_path)
        
        # Load raw data for regime detection (SMA)
        self.df = pd.read_csv(raw_data_path, index_col=0, parse_dates=True)
        self.df = self.df.sort_index()
        
        # Calculate 200-period SMA for Regime Detection (Bull/Bear)
        self.df['sma_200'] = self.df['close'].rolling(window=200).mean()
        self.df['regime'] = np.where(self.df['close'] > self.df['sma_200'], 'Bull', 'Bear')
        
        # Map timestamps to regimes for fast lookup
        self.regime_map = self.df['regime'].to_dict()

    def get_prediction(self, query_vec, query_timestamp, k=20, filter_regime=True):
        """
        Calculates probability and expected return based on historical matches.
        """
        # 1. Get raw similarity matches (larger K because we might filter some out)
        raw_matches = self.vdb.query(query_vec, k=k*2)
        
        current_regime = self.regime_map.get(pd.to_datetime(query_timestamp), 'Unknown')
        
        filtered_matches = []
        for m in raw_matches:
            match_ts = pd.to_datetime(m['timestamp'])
            match_regime = self.regime_map.get(match_ts, 'Unknown')
            
            if filter_regime:
                # ONLY keep matches that occurred in the same market regime
                if match_regime == current_regime:
                    filtered_matches.append(m)
            else:
                filtered_matches.append(m)
                
            if len(filtered_matches) >= k:
                break
        
        if not filtered_matches:
            return None

        # 2. Aggregate Outcomes
        returns = [m['next_24h_return'] for m in filtered_matches]
        max_ups = [m['next_24h_max'] for m in filtered_matches]
        min_downs = [m['next_24h_min'] for m in filtered_matches]
        
        prob_up = sum(1 for r in returns if r > 0) / len(returns)
        avg_return = np.mean(returns)
        exp_max = np.mean(max_ups)
        exp_min = np.mean(min_downs)
        
        return {
            'regime': current_regime,
            'matches_found': len(filtered_matches),
            'prob_up': float(prob_up),
            'avg_return': float(avg_return),
            'expected_max': float(exp_max),
            'expected_min': float(exp_min),
            'confidence': float(filtered_matches[0]['score']), # Top match similarity
            'top_matches': filtered_matches[:3]
        }

if __name__ == "__main__":
    predictor = PatternPredictor()
    
    # Test on a random recent sample
    data = np.load('data/btc_6h_embeddings.npz')
    test_idx = np.random.randint(len(data['embeddings']) - 100, len(data['embeddings']))
    
    test_vec = data['embeddings'][test_idx]
    test_ts = data['timestamps'][test_idx]
    
    print(f"\n--- Pattern Matching Prediction Engine ---")
    print(f"Querying pattern at: {test_ts}")
    
    pred = predictor.get_prediction(test_vec, test_ts, k=10)
    
    if pred:
        print(f"Current Regime: {pred['regime']}")
        print(f"Confidence (Top Match): {pred['confidence']:.4f}")
        print(f"Probability of UP move (24h): {pred['prob_up']*100:.1f}%")
        print(f"Average Expected Return: {pred['avg_return']*100:.2f}%")
        print(f"Expected Max Excursion: +{pred['expected_max']*100:.2f}%")
        print(f"Expected Min Excursion: {pred['expected_min']*100:.2f}%")
        
        signal = "NEUTRAL"
        if pred['prob_up'] >= 0.7: signal = "STRONG LONG"
        elif pred['prob_up'] <= 0.3: signal = "STRONG SHORT"
        
        print(f"\nFinal Signal: [{signal}]")
    else:
        print("No suitable historical matches found in this regime.")
