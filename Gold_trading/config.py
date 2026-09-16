"""
config.py
=========
Central configuration for the Gold Macro Intelligence & Backtesting Application.

Holds the FRED API key, FRED series mapping, Yahoo Finance symbols, macro-score
weights, classification thresholds, signal thresholds, and filesystem paths.
"""
from __future__ import annotations

import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
REPORT_DIR = DATA_DIR / "reports"
for _d in (DATA_DIR, CACHE_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# FRED API
# ----------------------------------------------------------------------------
# SECURITY: in production read from env var / secrets manager.
FRED_API_KEY = os.environ.get("FRED_API_KEY", "c2fd4c1708984ab087ccde7d39409823")
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"

# ----------------------------------------------------------------------------
# FRED Series Mapping  (logical_name -> FRED series id)
# ----------------------------------------------------------------------------
FRED_SERIES = {
    "DXY": "DTWEXBGS",          # US Dollar Index (broad)
    "DGS10": "DGS10",           # 10-Year Treasury Yield
    "CPI": "CPIAUCSL",          # CPI (All Urban)
    "CORE_CPI": "CPILFESL",     # Core CPI
    "CORE_PCE": "PCEPILFE",     # Core PCE inflation
    "FEDFUNDS": "FEDFUNDS",     # Effective Fed Funds Rate
    "UNRATE": "UNRATE",         # Unemployment Rate
    "PAYEMS": "PAYEMS",         # Nonfarm Payrolls (level)
    "INDPRO": "INDPRO",         # Industrial Production
}

# Publication lag (business days) applied when building the panel to avoid
# using future data that would not have been known at the time.
FRED_LAG = {
    "DXY": 0,
    "DGS10": 0,
    "CPI": 20, "CORE_CPI": 20, "CORE_PCE": 25,
    "FEDFUNDS": 5, "UNRATE": 10, "PAYEMS": 10, "INDPRO": 20,
}

# ----------------------------------------------------------------------------
# Yahoo Finance Symbols
# ----------------------------------------------------------------------------
YAHOO_SYMBOLS = {
    "GOLD": "GC=F",
    "GOLD_ALT": "XAUUSD=X",
    "DXY": "DX-Y.NYB",
    "TLT": "TLT",       # Long Treasury ETF (yield proxy / risk proxy)
    "GLD": "GLD",       # Gold ETF (sentiment proxy)
}

# ----------------------------------------------------------------------------
# Macro Score Weights (must sum to 1.0)
# ----------------------------------------------------------------------------
MACRO_WEIGHTS = {
    "real_yield": 0.30,
    "dxy": 0.25,
    "fed": 0.15,
    "inflation": 0.10,
    "labor": 0.10,
    "ism": 0.10,
}

# ----------------------------------------------------------------------------
# Macro Score Classification  (lower_bound, label)
# ----------------------------------------------------------------------------
SCORE_CLASSIFICATION = [
    (90, "Strong Bullish Gold"),
    (75, "Bullish Gold"),
    (60, "Moderately Bullish"),
    (40, "Neutral"),
    (25, "Bearish"),
    (0,  "Strong Bearish"),
]

# ----------------------------------------------------------------------------
# Signal thresholds  (lower_bound, label) on a numeric -100..100 score
# ----------------------------------------------------------------------------
SIGNAL_THRESHOLDS = [
    (50, "Strong Buy"),
    (20, "Buy"),
    (-20, "Neutral"),
    (-50, "Sell"),
    (-100, "Strong Sell"),
]

# ----------------------------------------------------------------------------
# Backtest parameters
# ----------------------------------------------------------------------------
BACKTEST_START = "2005-01-01"
FORWARD_HORIZONS = {"1M": 21, "3M": 63, "6M": 126, "12M": 252}

# Rolling window (in business days) used for no-lookahead z-scores.
Z_WINDOW_DAYS = 252
Z_MIN_OBS = 60
