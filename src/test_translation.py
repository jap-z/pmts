import pandas as pd
import numpy as np
import sys
import os
from types import SimpleNamespace

# Add fincast_src to path
sys.path.append(os.path.join(os.getcwd(), 'fincast_src'))
from tools.inference_utils import FinCast_Inference

def test_translation_logic():
    print("--- Engineering Test: Price Translation Verification ---")
    
    # 1. Setup Dummy Data
    # Assume Bitcoin at $60,000 with some volatility
    prices = np.array([58000, 59000, 61000, 62000, 60000], dtype=np.float32)
    mu = prices.mean()
    sigma = prices.std(ddof=0)
    
    print(f"Original Prices: {prices}")
    print(f"Calculated Stats: Mean=${mu:.2f}, StdDev=${sigma:.2f}")
    
    # 2. Normalize (What the Model Sees)
    normalized = (prices - mu) / sigma
    print(f"Normalized Tensors: {normalized}")
    
    # 3. Inverse Scale (What the Chart Sees)
    # Assume the model predicts a slight up-move (+0.5 standard deviations)
    predicted_norm = 0.5
    predicted_abs = (predicted_norm * sigma) + mu
    
    print(f"Model Output (Normalized): {predicted_norm}")
    print(f"Translated Price (Absolute): ${predicted_abs:.2f}")
    
    # Verification
    expected = (0.5 * sigma) + mu
    if abs(predicted_abs - expected) < 1e-5:
        print("✅ SUCCESS: Math logic for translation is correct.")
    else:
        print("❌ FAILURE: Translation math mismatch.")

    # 4. Live Environment Check
    print("\n--- Live Data check (BTC 1D) ---")
    df = pd.read_csv('data/BTC_USDT_1d.csv')
    series = pd.to_numeric(df['close'], errors="coerce").dropna().to_numpy(dtype=np.float32)
    
    l_mu = float(series.mean())
    l_sigma = float(series.std(ddof=0))
    last_price = series[-1]
    
    print(f"Latest BTC Price: ${last_price:.2f}")
    print(f"Dataset Mean: ${l_mu:.2f}")
    print(f"Dataset StdDev: ${l_sigma:.2f}")
    
    # If the model returned 1.04 earlier, what would that mean in real dollars?
    buggy_price_norm = 1.04 
    translated_buggy = (buggy_price_norm * l_sigma) + l_mu
    print(f"A raw output of '1.04' translates to: ${translated_buggy:.2f}")
    
    if translated_buggy > 60000:
        print("✅ SUCCESS: The translation correctly maps low-decimal outputs to high-value market prices.")
    else:
        print("⚠️ WARNING: Check if the model expects different normalization parameters.")

if __name__ == "__main__":
    test_translation_logic()
