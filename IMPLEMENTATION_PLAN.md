# Pattern Matching Trading System (PMTS)
## Implementation & Engineering Roadmap

This document outlines the end-to-end architecture and phased execution plan for building a semantic pattern-matching trading engine using Time Series Embeddings (TS2Vec) and Vector Similarity Search (FAISS).

---

## Phase 1: Environment & Data Ingestion
*Goal: Establish the Python backend and harvest high-quality, continuous historical market data.*

1. **Environment Setup**
   - Initialize Python virtual environment (`venv`).
   - Install core data science libraries: `pandas`, `numpy`, `scikit-learn`.
   - Install ML and Vector libraries: `torch` (PyTorch), `faiss-cpu`.
   - Install data retrieval tools: `ccxt` (for Binance/KuCoin API) or `yfinance`.
2. **Historical Data Pipeline**
   - Build an ingestion script to pull 5-10 years of OHLCV (Open, High, Low, Close, Volume) data.
   - **Resolution:** 6-hour candles.
   - **Target Asset:** Start with a high-liquidity asset (e.g., BTC/USDT or SPY) to minimize noise.
3. **Data Sanitization**
   - Handle missing candles (forward-fill close prices for zero-volume periods).
   - Flag and remove extreme API outlier spikes.

---

## Phase 2: Feature Engineering & Normalization
*Goal: Transform absolute dollar values into relative percentage representations to avoid the "Scale Swamp."*

1. **Window Generation**
   - Implement a rolling window function to slice the continuous time series into discrete **7-day periods** (28 candles per window).
   - Use a stride of 1 candle (rolling forward every 6 hours).
2. **Mathematical Transformation**
   - Convert absolute prices to **Log Returns**: `ln(Price_t / Price_t-1)`.
   - Apply **Robust Scaling** across each window individually (Sample Normalization) to center the data while dampening the effect of flash crashes/pumps.
3. **Labeling the "Truth"**
   - For every 7-day window, calculate the *forward return* (e.g., What is the maximum excursion and close price of the next 4 candles / 24 hours?).
   - This becomes the target label for evaluating historical similarity.

---

## Phase 3: The Embedding Encoder (The Brain)
*Goal: Compress the 28-candle windows into dense semantic vectors.*

1. **Model Selection**
   - Implement or import a pre-trained **TS2Vec** (Time Series to Vector) model architecture in PyTorch.
   - *Alternative for rapid prototyping:* Use an established Time-Series Foundation Model (TSFM) like Chronos or Moment if API access is viable, otherwise stick to local TS2Vec.
2. **Temporal Encoding (Time2Vec)**
   - Inject periodic sine-wave encodings so the model mathematically recognizes daily market cycles (Asian vs. NY sessions).
3. **Encoding Generation**
   - Run all historical 7-day windows through the encoder to generate a master matrix of 320-dimensional embedding vectors.

---

## Phase 4: Vector Database & Similarity Search
*Goal: Build the "Memory Bank" to instantly find historical twins.*

1. **FAISS Indexing**
   - Initialize a local `faiss.IndexFlatIP` (Inner Product) or `IndexFlatL2` for exact similarity search.
   - Insert all historical embeddings into the FAISS index.
2. **The K-NN Retrieval Engine**
   - Build the query function: Given a "Current" 7-day embedding, ask FAISS to return the Top `K=10` closest historical matches.
   - Define the distance metric (Cosine Similarity is preferred to match pattern shape regardless of absolute volatility magnitude).

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