import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
import os
import argparse
from tqdm import tqdm

def fetch_historical_data(symbol='BTC/USDT', timeframe='6h', years=5, output_dir='data'):
    # Initialize exchange
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })

    # Calculate start time
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=years * 365)
    start_timestamp = int(start_date.timestamp() * 1000)

    print(f"Fetching {symbol} {timeframe} data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    all_ohlcv = []
    current_timestamp = start_timestamp

    # 5 years of 6h candles = 5 * 365 * 4 = 7300 candles
    # binance limit is 1000 per request
    expected_requests = (years * 365 * 4) // 1000 + 1
    
    with tqdm(total=expected_requests, desc="Fetching data chunks") as pbar:
        while True:
            try:
                # Fetch OHLCV data
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_timestamp, limit=1000)
                
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                
                # Get the timestamp of the last candle fetched and add one timeframe duration
                # 6 hours in milliseconds = 6 * 60 * 60 * 1000 = 21600000
                timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
                current_timestamp = ohlcv[-1][0] + timeframe_ms
                
                pbar.update(1)
                
                if current_timestamp >= int(end_date.timestamp() * 1000):
                    break
                    
            except Exception as e:
                print(f"Error fetching data: {e}")
                time.sleep(5)  # Wait and retry on error
                
    # Create DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Remove duplicates if any
    df = df[~df.index.duplicated(keep='first')]
    
    # Sanitization
    print("Sanitizing data...")
    # Reindex to fill any missing candles with NaN
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='6h')
    df = df.reindex(full_idx)
    
    # Forward-fill missing close prices
    df['close'] = df['close'].ffill()
    
    # For missing open, high, low, use the ffilled close price
    df['open'] = df['open'].fillna(df['close'])
    df['high'] = df['high'].fillna(df['close'])
    df['low'] = df['low'].fillna(df['close'])
    df['volume'] = df['volume'].fillna(0)
    
    # Basic outlier check (e.g. price drops to 0)
    outliers = df[df['close'] <= 0]
    if not outliers.empty:
        print(f"Warning: Found {len(outliers)} candles with price <= 0. Dropping them.")
        df = df[df['close'] > 0]
        
    print(f"Total candles fetched and sanitized: {len(df)}")
    
    # Save to CSV
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename = f"{symbol.replace('/', '_')}_{timeframe}.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath)
    print(f"Data saved to {filepath}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical crypto data")
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading pair symbol')
    parser.add_argument('--timeframe', type=str, default='6h', help='Candle timeframe')
    parser.add_argument('--years', type=int, default=5, help='Years of history to fetch')
    
    args = parser.parse_args()
    fetch_historical_data(symbol=args.symbol, timeframe=args.timeframe, years=args.years)
