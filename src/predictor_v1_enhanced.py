import numpy as np
import pandas as pd
from src.vector_db import VectorDatabase
import os
import argparse

class PatternPredictor:
    def __init__(self, raw_data_path='data/BTC_USDT_6h.csv', embedding_path='data/v1_backup/btc_6h_embeddings.npz'):
        # For this test, we load the V1 backup model (which is 256D)
        self.vdb = VectorDatabase(embedding_dim=256)
        self.vdb.load_data(embedding_path)
        
        # Load raw data for regime detection (SMA and ATR)
        self.df = pd.read_csv(raw_data_path, index_col=0, parse_dates=True)
        self.df = self.df.sort_index()
        
        # 1. Trend Regime (200 SMA)
        self.df['sma_200'] = self.df['close'].rolling(window=200).mean()
        self.df['trend'] = np.where(self.df['close'] > self.df['sma_200'], 'Bull', 'Bear')
        
        # 2. Volatility Regime (ATR)
        # Calculate True Range
        high_low = self.df['high'] - self.df['low']
        high_close = np.abs(self.df['high'] - self.df['close'].shift())
        low_close = np.abs(self.df['low'] - self.df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        # 14-period ATR
        self.df['atr_14'] = true_range.rolling(14).mean()
        # 100-period SMA of ATR to determine if current vol is relatively high or low
        self.df['atr_100_sma'] = self.df['atr_14'].rolling(100).mean()
        
        self.df['volatility'] = np.where(self.df['atr_14'] > self.df['atr_100_sma'], 'HighVol', 'LowVol')
        
        # Combine regimes
        self.df['regime'] = self.df['trend'] + '_' + self.df['volatility']
        
        # Map timestamps to regimes for fast lookup
        self.regime_map = self.df['regime'].to_dict()

    def get_prediction(self, query_vec, query_timestamp, k=10, filter_regime=True, max_timestamp=None):
        """
        Calculates probability and expected return based on historical matches using Distance-Weighted K-NN.
        """
        # 1. Get raw similarity matches (larger K because we might filter some out)
        raw_matches = self.vdb.query(query_vec, k=k*10) # Heavy filtering requires deeper initial search
        
        current_regime = self.regime_map.get(pd.to_datetime(query_timestamp), 'Unknown')
        
        filtered_matches = []
        for m in raw_matches:
            match_ts = pd.to_datetime(m['timestamp'])
            
            # Prevent lookahead bias and matching with itself
            if max_timestamp is not None and match_ts >= pd.to_datetime(max_timestamp):
                continue
            elif max_timestamp is None and match_ts == pd.to_datetime(query_timestamp):
                continue
                
            match_regime = self.regime_map.get(match_ts, 'Unknown')
            
            if filter_regime:
                # ONLY keep matches that occurred in the exact same trend & volatility regime
                if match_regime == current_regime:
                    filtered_matches.append(m)
            else:
                filtered_matches.append(m)
                
            if len(filtered_matches) >= k:
                break
        
        if not filtered_matches:
            return None

        # 2. Distance-Weighted Aggregation
        # Instead of simple counting, we weight each match's outcome by its cosine similarity.
        # We square the similarity to exponentially reward highly similar patterns.
        
        total_weight = 0.0
        weighted_up_votes = 0.0
        
        weighted_return_sum = 0.0
        weighted_max_sum = 0.0
        weighted_min_sum = 0.0
        
        for m in filtered_matches:
            # Cosine similarity is [-1, 1]. We only care about positive highly correlated matches.
            # Using max(0.01, score)^3 to heavily favor exact geometric twins
            weight = max(0.01, m['score']) ** 3 
            
            total_weight += weight
            
            ret = m['next_24h_return']
            if ret > 0:
                weighted_up_votes += weight
                
            weighted_return_sum += ret * weight
            weighted_max_sum += m['next_24h_max'] * weight
            weighted_min_sum += m['next_24h_min'] * weight
            
        prob_up = weighted_up_votes / total_weight if total_weight > 0 else 0.5
        avg_return = weighted_return_sum / total_weight if total_weight > 0 else 0
        exp_max = weighted_max_sum / total_weight if total_weight > 0 else 0
        exp_min = weighted_min_sum / total_weight if total_weight > 0 else 0
        
        return {
            'regime': current_regime,
            'matches_found': len(filtered_matches),
            'prob_up': float(prob_up),
            'avg_return': float(avg_return),
            'expected_max': float(exp_max),
            'expected_min': float(exp_min),
            'confidence': float(filtered_matches[0]['score']), 
            'top_matches': filtered_matches[:3]
        }
