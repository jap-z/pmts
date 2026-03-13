# Pattern Matching Trading System (PMTS)
## Implementation & Engineering Roadmap

This document outlines the end-to-end architecture and phased execution plan for building a semantic pattern-matching trading engine using Time Series Embeddings (TS2Vec) and Vector Similarity Search (FAISS).

---

## Phase 1: Environment & Data Ingestion
*Goal: Establish the Python backend and harvest high-quality, continuous historical market data.*

1. **Environment Setup**
   - [x] Initialize Python virtual environment (`venv`).
   - [x] Install core data science libraries: `pandas`, `numpy`, `scikit-learn`.
   - [x] Install ML and Vector libraries: `torch` (PyTorch), `faiss-cpu`.
   - [x] Install data retrieval tools: `ccxt` (for Binance/KuCoin API) or `yfinance`.
2. **Historical Data Pipeline**
   - [x] Build an ingestion script to pull 5-10 years of OHLCV (Open, High, Low, Close, Volume) data.
   - [x] **Resolution:** 6-hour candles.
   - [x] **Target Asset:** Start with a high-liquidity asset (e.g., BTC/USDT or SPY) to minimize noise.
3. **Data Sanitization**
   - [x] Handle missing candles (forward-fill close prices for zero-volume periods).
   - [x] Flag and remove extreme API outlier spikes.
   - *Status:* **COMPLETED**. Established Python 3.12 venv, created `src/data_ingestion.py`, and successfully harvested 5 years (~7,300 candles) of BTC/USDT 6h data from Binance.

---

## Phase 2: Feature Engineering & Normalization
*Goal: Transform absolute dollar values into relative percentage representations to avoid the "Scale Swamp."*

1. **Window Generation**
   - [x] Implement a rolling window function to slice the continuous time series into discrete **7-day periods** (28 candles per window).
   - [x] Use a stride of 1 candle (rolling forward every 6 hours).
2. **Mathematical Transformation**
   - [x] Convert absolute prices to **Log Returns**: `ln(Price_t / Price_t-1)`.
   - [x] Apply **Robust Scaling** across each window individually (Sample Normalization) to center the data while dampening the effect of flash crashes/pumps.
3. **Labeling the "Truth"**
   - [x] For every 7-day window, calculate the *forward return* (e.g., What is the maximum excursion and close price of the next 4 candles / 24 hours?).
   - [x] This becomes the target label for evaluating historical similarity.
   - *Status:* **COMPLETED**. Created `src/feature_engineering.py`. Calculated Log Returns for OHLCV, generated 7,267 28-candle windows with 4-candle forecast horizons, and applied RobustScaler per window. Saved as `btc_6h_features.npz`.

---

## Phase 3: The Embedding Encoder (The Brain)
*Goal: Compress the 28-candle windows into dense semantic vectors.*

1. **Model Selection**
   - [x] Implement or import a pre-trained **TS2Vec** (Time Series to Vector) model architecture in PyTorch.
   - [x] *Alternative for rapid prototyping:* Use an established Time-Series Foundation Model (TSFM) like Chronos or Moment if API access is viable, otherwise stick to local TS2Vec.
2. **Temporal Encoding (Time2Vec)**
   - [x] Inject periodic sine-wave encodings so the model mathematically recognizes daily market cycles (Asian vs. NY sessions).
3. **Encoding Generation**
   - [x] Run all historical 7-day windows through the encoder to generate a master matrix of 320-dimensional embedding vectors.
   - *Status:* **COMPLETED**. Created `src/encoder.py`. Built a bidirectional LSTM Autoencoder combined with a custom `Time2Vec` module for temporal awareness. Trained for 20 epochs to reconstruct the 28-candle windows. Successfully compressed all 7,267 windows into 128-dimensional vectors and saved them to `data/btc_6h_embeddings.npz`.

---

## Phase 4: Vector Database & Similarity Search
*Goal: Build the "Memory Bank" to instantly find historical twins.*

1. **FAISS Indexing**
   - [x] Initialize a local `faiss.IndexFlatIP` (Inner Product) or `IndexFlatL2` for exact similarity search.
   - [x] Insert all historical embeddings into the FAISS index.
2. **The K-NN Retrieval Engine**
   - [x] Build the query function: Given a "Current" 7-day embedding, ask FAISS to return the Top `K=10` closest historical matches.
   - [x] Define the distance metric (Cosine Similarity is preferred to match pattern shape regardless of absolute volatility magnitude).
   - *Status:* **COMPLETED**. Created `src/vector_db.py`. Implemented FAISS with `IndexFlatIP` combined with L2 normalization to achieve Cosine Similarity search. Successfully indexed 7,267 vectors and verified that it can find "historical twins" in milliseconds.

---

## Phase 5: The Predictive Logic & Regime Filtering
*Goal: Turn similar historical shapes into an actionable trading signal.*

1. **Outcome Aggregation**
   - Extract the forward returns (the "Truth" labels from Phase 2) of the Top 10 historical matches.
   - Calculate the Probability of an Up-Move: e.g., if 8/10 matches pumped, P(Up) = 80%.
   - Calculate Expected Return (mean of the 10 forward returns).
2. **Market Regime Detection (Crucial)**
   - Implement a basic regime filter (e.g., classifying history into High-Vol/Bull, Low-Vol/Bear using VIX or simple Moving Average states).
   - *Rule:* The system must drop historical matches that occurred in a fundamentally different macroeconomic regime than the current live market.

---

## Phase 6: Backtesting & Live Simulation
*Goal: Prove the statistical edge without risking capital.*

1. **Out-of-Sample Walk-Forward Test**
   - Hide the last 1 year of data from the FAISS database.
   - Run the system chronologically through the hidden year, making predictions based only on the data available prior to that specific timestamp.
2. **Performance Metrics**
   - Track directional accuracy (did it guess up/down correctly?).
   - Track Sharpe Ratio and Max Drawdown of the generated signals.
3. **Live Webhook Integration**
   - Once validated, wrap the Python engine in a FastAPI or Flask server.
   - Expose an endpoint to ingest live 6-hour candles, run the embedding query, and fire a buy/sell signal to a webhook (or directly to Discord via OpenClaw).