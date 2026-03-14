import sys
import os
import time
import json
import torch
import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace
from flask import Flask, Response, send_from_directory, jsonify, request
from flask_cors import CORS
import ccxt
import yfinance as yf

# Add fincast_src to path
sys.path.append(os.path.join(os.getcwd(), 'fincast_src'))
from tools.inference_utils import FinCast_Inference

app = Flask(__name__, static_folder='../public')
CORS(app)

# Custom JSON Encoder to handle Numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Global model holder
fincast_model = None
active_streams = set()

def fetch_data_for_backtest(symbol, timeframe, total_candles_needed, csv_path, provider='binance'):
    print(f"Fetching {total_candles_needed} candles of {timeframe} for {symbol} via {provider}...")
    
    if provider == 'binance':
        exchange = ccxt.binance()
        limit = 1000
        all_ohlcv = []
        
        timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
        start_timestamp = int(datetime.utcnow().timestamp() * 1000) - (total_candles_needed * timeframe_ms)
        
        current_timestamp = start_timestamp
        while len(all_ohlcv) < total_candles_needed:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_timestamp, limit=limit)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                current_timestamp = ohlcv[-1][0] + timeframe_ms
                time.sleep(0.1)
            except Exception as e:
                print("CCXT fetch error:", e)
                time.sleep(2)
        
        all_ohlcv = all_ohlcv[-total_candles_needed:]
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
    elif provider == 'yahoo':
        yf_map = {'1m': '1m', '5m': '5m', '15m': '15min', '1h': '1h', '4h': '1h', '1d': '1d'}
        yf_tf = yf_map.get(timeframe, '1d')
        start_date = datetime.now() - timedelta(days=int(total_candles_needed * 1.5) + 10) if timeframe == '1d' else datetime.now() - timedelta(days=60)
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, interval=yf_tf)
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index, utc=True)
        
    df = df[~df.index.duplicated(keep='first')]
    tf_map = {'1m': '1min', '5m': '5min', '15m': '15min', '1h': '1h', '4h': '4h', '6h': '6h', '12h': '12h', '1d': '1D'}
    pd_freq = tf_map.get(timeframe, timeframe)
    
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=pd_freq)
    df = df.reindex(full_idx)
    df['close'] = df['close'].ffill()
    df['open'] = df['open'].fillna(df['close'])
    df['high'] = df['high'].fillna(df['close'])
    df['low'] = df['low'].fillna(df['close'])
    df['volume'] = df['volume'].fillna(0)
    
    df.to_csv(csv_path)
    return df

def get_loaded_model():
    global fincast_model
    if fincast_model is not None:
        return fincast_model

    config = SimpleNamespace()
    config.backend = "cpu"
    config.model_path = "fincast_model/v1.pth"
    config.model_version = "v1"
    config.data_path = "data/BTC_USDT_1d.csv" 
    config.data_frequency = "1d" 
    config.context_len = 128     
    config.horizon_len = 7       
    config.all_data = True       
    config.columns_target = ['close']
    config.series_norm = False       
    config.batch_size = 1
    config.forecast_mode = "mean"
    config.save_output = False
    config.plt_outputs = False
    config.num_experts = 4
    config.gating_top_n = 2
    config.load_from_compile = True

    fincast_model = FinCast_Inference(config)
    print("Model initialized.")
    return fincast_model

@app.route('/pmts/')
def index():
    return send_from_directory('../public', 'index.html')

@app.route('/pmts/stop')
def stop_all():
    active_streams.clear()
    return jsonify({"status": "ok", "message": "All backtests stopped."})

@app.route('/pmts/stream_backtest')
def stream_backtest():
    duration_val = int(request.args.get('duration', 365))
    duration_unit = request.args.get('unit', 'days')
    timeframe = request.args.get('timeframe', '1d')
    direction = request.args.get('direction', 'both')
    symbol = request.args.get('symbol', 'BTC/USDT')
    provider = request.args.get('provider', 'binance')
    
    initial_capital = float(request.args.get('capital', 10000.0))
    leverage = float(request.args.get('leverage', 1.0))
    sl_pct = float(request.args.get('sl', 5.0)) / 100.0
    tp_pct = float(request.args.get('tp', 10.0)) / 100.0
    fee_pct = float(request.args.get('fee', 0.1)) / 100.0
    
    stream_id = str(uuid.uuid4())
    active_streams.add(stream_id)
    
    def generate():
        original_records = None
        model_wrapper = None
        try:
            yield f"data: {json.dumps({'type': 'status', 'msg': f'Connecting to FinCast Engine for {symbol}...', 'stream_id': stream_id})}\n\n"
            
            tf_hours_map = {'1m': 1/60, '5m': 5/60, '15m': 15/60, '1h': 1, '4h': 4, '6h': 6, '12h': 12, '1d': 24}
            tf_hours = tf_hours_map.get(timeframe, 24)
            
            if duration_unit == 'minutes': total_hours = duration_val / 60
            elif duration_unit == 'hours': total_hours = duration_val
            elif duration_unit == 'days': total_hours = duration_val * 24
            elif duration_unit == 'months': total_hours = duration_val * 24 * 30
            elif duration_unit == 'years': total_hours = duration_val * 24 * 365
            else: total_hours = duration_val * 24

            test_steps = int(total_hours / tf_hours)
            if test_steps <= 0: test_steps = 1
            
            context_len = 128
            horizon_len = 7
            total_candles_needed = test_steps + context_len + horizon_len + 10
            
            yield f"data: {json.dumps({'type': 'status', 'msg': f'Fetching {total_candles_needed} candles of {timeframe} data...'})}\n\n"
            
            clean_symbol = symbol.replace('/', '_').replace('^', '')
            csv_path = f"data/temp_{clean_symbol}_{timeframe}.csv"
            fetch_data_for_backtest(symbol, timeframe, total_candles_needed, csv_path, provider=provider)
            
            secondary_tf = '5m' if tf_hours >= 1 else '1m'
            secondary_csv_path = f"data/temp_{clean_symbol}_{secondary_tf}_micro.csv"
            sec_mult = tf_hours / (5/60 if secondary_tf == '5m' else 1/60)
            sec_needed = int(total_candles_needed * sec_mult)
            
            yield f"data: {json.dumps({'type': 'status', 'msg': f'Fetching {sec_needed} candles of {secondary_tf} for micro-tick simulation...'})}\n\n"
            fetch_data_for_backtest(symbol, secondary_tf, sec_needed, secondary_csv_path, provider=provider)
            
            sec_df = pd.read_csv(secondary_csv_path, index_col=0, parse_dates=True).sort_index()
            sec_df.index = pd.to_datetime(sec_df.index, utc=True)

            model_wrapper = get_loaded_model()
            from data_tools.Inference_dataset import TimeSeriesDataset_SingleCSV_Inference
            from tools.inference_utils import freq_reader_inference
            
            model_wrapper.inference_freq = freq_reader_inference(timeframe)
            model_wrapper.inference_dataset = TimeSeriesDataset_SingleCSV_Inference(
                csv_path=csv_path, context_length=context_len, freq_type=model_wrapper.inference_freq,
                columns=['close'], first_c_date=True, series_norm=False, dropna=True, sliding_windows=True, return_meta=True
            )
            
            df_clean = pd.read_csv(csv_path).dropna(axis=0).reset_index(drop=True)
            series_close = pd.to_numeric(df_clean['close']).to_numpy(dtype=np.float32)
            timestamps = pd.to_datetime(df_clean.iloc[:, 0], utc=True).tolist()

            if model_wrapper.inference_dataset.sliding_windows:
                all_records = model_wrapper.inference_dataset.index_records
                original_records = list(all_records)
                testable_records = [r for r in all_records if r[1] + context_len - 1 + horizon_len < len(series_close)]
                model_wrapper.inference_dataset.index_records = testable_records[-test_steps:]
                test_steps = len(model_wrapper.inference_dataset.index_records)
                
            loader = model_wrapper._make_inference_loader(batch_size=1, num_workers=0)
            model = model_wrapper.model_api
            
            wins, total_trades, pnl_sum = 0, 0, 0.0
            account_balance = initial_capital
            peak_balance = initial_capital
            max_drawdown = 0.0
            
            with torch.inference_mode():
                for i, (x_ctx, x_pad, freq, _x_fut, meta) in enumerate(loader):
                    if stream_id not in active_streams: break

                    yield f"data: {json.dumps({'type': 'status', 'msg': f'Crunching math for window {i+1} / {test_steps}...'})}\n\n"
                    
                    from tools.inference_utils import get_forecasts_f
                    preds_out, full_out = get_forecasts_f(model, x_ctx, freq=freq)
                    
                    out_np = preds_out.detach().cpu().numpy() if isinstance(preds_out, torch.Tensor) else np.asarray(preds_out)
                    if out_np.ndim == 1: out_np = out_np[None, :]
                    pred_abs = out_np[0] 
                    
                    full_np = full_out.detach().cpu().numpy() if isinstance(full_out, torch.Tensor) else np.asarray(full_out)
                    q10_abs = full_np[0, :, 1]
                    q90_abs = full_np[0, :, 9]
                    
                    m = meta[0]
                    end_idx, start_idx = int(m['window_end']), int(m['window_start'])
                    base_price = series_close[end_idx]
                    
                    entry_time = timestamps[end_idx]
                    exit_time = timestamps[end_idx + horizon_len]
                    
                    actual_future_prices = series_close[end_idx + 1 : end_idx + horizon_len + 1]
                    future_times = [str(t) for t in timestamps[end_idx + 1 : end_idx + horizon_len + 1]]
                    
                    micro_slice = sec_df.loc[entry_time : exit_time]
                    micro_highs = micro_slice['high'].values
                    micro_lows = micro_slice['low'].values
                    micro_closes = micro_slice['close'].values
                    
                    pred_return = (pred_abs[-1] - base_price) / base_price
                    
                    trade_taken, trade_won, trade_pnl, signal, exit_reason = False, False, 0.0, "NEUTRAL", "TIMED"
                    
                    if pred_return > 0.01 and direction in ['both', 'long']:
                        trade_taken, signal = True, "LONG"
                        tp_price = base_price * (1 + tp_pct)
                        sl_price = base_price * (1 - sl_pct)
                        
                        for tick_idx in range(len(micro_highs)):
                            if micro_lows[tick_idx] <= sl_price:
                                trade_pnl = -sl_pct
                                exit_reason = "STOP LOSS"
                                break
                            if micro_highs[tick_idx] >= tp_price:
                                trade_pnl = tp_pct
                                exit_reason = "TAKE PROFIT"
                                trade_won = True
                                break
                        else:
                            final_px = micro_closes[-1] if len(micro_closes) > 0 else actual_future_prices[-1]
                            trade_pnl = (final_px - base_price) / base_price
                            trade_won = trade_pnl > 0
                            
                    elif pred_return < -0.01 and direction in ['both', 'short']:
                        trade_taken, signal = True, "SHORT"
                        tp_price = base_price * (1 - tp_pct)
                        sl_price = base_price * (1 + sl_pct)
                        
                        for tick_idx in range(len(micro_highs)):
                            if micro_highs[tick_idx] >= sl_price:
                                trade_pnl = -sl_pct
                                exit_reason = "STOP LOSS"
                                break
                            if micro_lows[tick_idx] <= tp_price:
                                trade_pnl = tp_pct
                                exit_reason = "TAKE PROFIT"
                                trade_won = True
                                break
                        else:
                            final_px = micro_closes[-1] if len(micro_closes) > 0 else actual_future_prices[-1]
                            trade_pnl = (base_price - final_px) / base_price
                            trade_won = trade_pnl > 0

                    if trade_taken:
                        gross_pnl = trade_pnl * leverage
                        net_pnl = gross_pnl - (fee_pct * 2 * leverage)
                        
                        account_balance *= (1 + net_pnl)
                        total_trades += 1
                        pnl_sum += net_pnl
                        if trade_won: wins += 1
                        
                        if account_balance > peak_balance: peak_balance = account_balance
                        dd = (peak_balance - account_balance) / peak_balance
                        if dd > max_drawdown: max_drawdown = dd

                    yield f"data: {json.dumps({
                        'step': i + 1, 'total_steps': test_steps, 'timestamp': str(timestamps[end_idx]),
                        'base_price': base_price, 'context_times': [str(t) for t in timestamps[start_idx : end_idx + 1]][-30:], 
                        'context_prices': series_close[start_idx : end_idx + 1][-30:], 
                        'future_times': future_times,
                        'actual_prices': actual_future_prices, 
                        'predicted_prices': pred_abs,
                        'q10_prices': q10_abs,
                        'q90_prices': q90_abs,
                        'signal': signal, 'trade_taken': trade_taken, 'won': trade_won,
                        'trade_pnl': trade_pnl if not trade_taken else net_pnl,
                        'exit_reason': exit_reason,
                        'total_trades': total_trades, 'win_rate': (wins/total_trades*100) if total_trades > 0 else 0,
                        'account_balance': account_balance, 'max_drawdown': max_drawdown * 100
                    }, cls=NumpyEncoder)}\n\n"
        except GeneratorExit: pass
        except Exception as e:
            print(f"Error in stream: {e}")
        finally:
            if stream_id in active_streams: active_streams.remove(stream_id)
            if model_wrapper and original_records: model_wrapper.inference_dataset.index_records = original_records

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
