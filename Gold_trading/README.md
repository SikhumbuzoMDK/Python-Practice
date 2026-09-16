# Gold Macro Intelligence & Backtesting Application

A production-ready Python application that analyses, backtests, scores, and
forecasts Gold (XAUUSD) using macroeconomic and market data from FRED and
Yahoo Finance.

## Features
- Downloads FRED macroeconomic data + Yahoo Finance market data (with local caching)
- Computes a 0–100 **Gold Macro Score** from six weighted fundamental drivers
- Generates **Daily / Weekly / Monthly / Yearly** signals with confidence levels
- Runs a **walk-forward backtest** (2005–present) with no lookahead bias
- Detects **market regimes** and measures historical gold performance by regime
- Produces an explainable report + an interactive Streamlit dashboard

## Setup
```bash
pip install -r requirements.txt
