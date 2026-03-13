# Pattern Matching Trading System (PMTS) - FinCast Edition
## Implementation & Engineering Roadmap

This document outlines the pivot toward utilizing the 1B parameter FinCast Foundation Model for zero-shot time-series forecasting.

---

## Phase 1: Environment & Data Ingestion
*Goal: Establish the Python backend and harvest historical market data.*
- [x] Initialize Python 3.12 virtual environment.
- [x] Build `src/data_ingestion.py` to pull 10 years of BTC/USDT 6h candles.
- [x] Successfully harvested 12,521 candles.

## Phase 2: Feature Engineering (Light)
*Goal: Prepare data for foundation model consumption.*
- [x] Calculated Log Returns and prepared normalized windows.
- [x] Note: FinCast handles internal normalization, but we maintain raw CSV for inverse scaling.

## Phase 3: Time Series Foundation Model (FinCast)
*Goal: Leverage a 1B parameter pre-trained transformer for zero-shot forecasting.*
- [x] Clone official FinCast repository and install heavy dependencies (PyTorch, JAX, TensorFlow, Einops).
- [x] Download 1B parameter weights (`v1.pth`) from HuggingFace.
- [x] Implement inference wrapper `src/fincast_inference.py`.
- [x] Successfully ran zero-shot inference on BTC data. Verified realistic USD price outputs via Inverse Scaling.

---

## Phase 4: Foundation Backtesting
*Goal: Prove the predictive power of the foundation model.*
- [ ] Implement `src/backtest_fincast.py`.
- [ ] Step through 365 days of 6h data (1,460 windows).
- [ ] For each window, perform a blind FinCast prediction (next 24h).
- [ ] Calculate Win Rate and PnL.

---

## Phase 5: Live Execution Strategy
*Goal: Deploy the model to broadcast live signals.*
- [ ] Define entry/exit thresholds based on FinCast quantile probabilities.
- [ ] Wrap inference in a FastAPI server.
- [ ] Fire "Strong Long/Short" alerts to Discord every 6 hours via OpenClaw webhook.

---

## Phase 6: Risk Management & Multi-Asset Expansion
*Goal: Scaling the system.*
- [ ] Simulate real-world trading costs (fees + slippage).
- [ ] Run zero-shot backtests on ETH, SOL, and SPY to verify foundation model universality.

---
*Document maintained by Big Clawd - 2026-03-13*
