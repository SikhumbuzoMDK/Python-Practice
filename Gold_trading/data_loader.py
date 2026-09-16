"""
data_loader.py
==============
Downloads and caches macroeconomic data from FRED and market data from Yahoo
Finance, then combines them into a single daily-frequency panel.

Features:
  - FRED client via requests (portable, no extra dependency required).
  - yfinance for market data (with graceful fallback if unavailable).
  - Local CSV caching to the data/cache directory.
  - Publication-lag shift so stale future values are never used.
  - Offline mode: load from cache / generate synthetic data if no internet.
"""
from __future__ import annotations

import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import (
    FRED_API_KEY, FRED_API_URL, FRED_SERIES, FRED_LAG,
    YAHOO_SYMBOLS, CACHE_DIR, BACKTEST_START,
)

logger = logging.getLogger(__name__)


def _fred_request(series_id: str, start: str, end: str) -> pd.Series:
    """Fetch a single FRED series via the official API (requests)."""
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    resp = requests.get(FRED_API_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json().get("observations", [])
    s = pd.Series(
        {obs["date"]: obs["value"] for obs in data if obs["value"] != "."},
        name=series_id,
    )
    s.index = pd.to_datetime(s.index)
    s = s.astype(float).sort_index()
    return s


def _fetch_fred_series(series_id: str, start: str, end: str) -> pd.Series:
    """Fetch a FRED series, trying fredapi first then requests fallback."""
    try:
        from fredapi import Fred  # optional dependency
        fred = Fred(api_key=FRED_API_KEY)
        return fred.get_series(series_id, observation_start=start, observation_end=end)
    except Exception:  # pragma: no cover - fallback to requests
        return _fred_request(series_id, start, end)


def fetch_fred_panel(start: str = BACKTEST_START, end: str = None) -> pd.DataFrame:
    """Download all configured FRED series and return a monthly index DataFrame."""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    panel = pd.DataFrame()
    for name, sid in FRED_SERIES.items():
        try:
            s = _fetch_fred_series(sid, start, end)
            panel[name] = s
            time.sleep(0.2)  # be polite to the API
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to fetch FRED %s (%s): %s", name, sid, exc)
    return panel.sort_index()


def fetch_yahoo(symbols: dict | None = None, start: str = BACKTEST_START,
                end: str = None) -> pd.DataFrame:
    """Download daily Yahoo Finance data and return a DataFrame of adjusted closes."""
    import yfinance as yf

    symbols = symbols or YAHOO_SYMBOLS
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    out = pd.DataFrame()
    for name, ticker in symbols.items():
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False)
            if df.empty:
                continue
            close = df["Close"]
            if isinstance(close, pd.DataFrame):  # yfinance may return multi-col
                close = close.iloc[:, 0]
            out[name] = close
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to fetch Yahoo %s (%s): %s", name, ticker, exc)
    return out.sort_index()


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.csv"


def _save_cache(df: pd.DataFrame, name: str) -> None:
    if df is not None and not df.empty:
        df.to_csv(_cache_path(name))


def _load_cache(name: str) -> pd.DataFrame | None:
    p = _cache_path(name)
    if p.exists():
        return pd.read_csv(p, index_col=0, parse_dates=True)
    return None


def _synthetic_fred(start: str = BACKTEST_START) -> pd.DataFrame:
    """Deterministic synthetic macro panel for offline development/testing."""
    idx = pd.date_range(start, pd.Timestamp.today(), freq="M")
    n = len(idx)
    rng = np.random.default_rng(42)
    df = pd.DataFrame(index=idx)
    df["DXY"] = 95 + np.cumsum(rng.normal(0, 0.5, n))
    df["DGS10"] = 3.0 + np.cumsum(rng.normal(0, 0.08, n))
    df["CPI"] = 250 + np.cumsum(rng.normal(0, 0.3, n))
    df["CORE_CPI"] = 250 + np.cumsum(rng.normal(0, 0.25, n))
    df["CORE_PCE"] = 100 + np.cumsum(rng.normal(0, 0.1, n))
    df["FEDFUNDS"] = 2.0 + np.cumsum(rng.normal(0, 0.05, n))
    df["UNRATE"] = 5.0 + np.cumsum(rng.normal(0, 0.05, n))
    df["PAYEMS"] = 130000 + np.cumsum(rng.normal(0, 60, n))
    df["INDPRO"] = 100 + np.cumsum(rng.normal(0, 0.3, n))
    return df


def _synthetic_yahoo(start: str = BACKTEST_START) -> pd.DataFrame:
    """Deterministic synthetic market panel for offline development/testing."""
    idx = pd.bdate_range(start, pd.Timestamp.today())
    n = len(idx)
    rng = np.random.default_rng(7)
    gold = 400 * np.exp(np.cumsum(rng.normal(0.0002, 0.011, n)))
    df = pd.DataFrame(index=idx)
    df["GOLD"] = gold
    df["GOLD_ALT"] = gold
    df["DXY"] = 95 + np.cumsum(rng.normal(0, 0.1, n))
    df["TLT"] = 100 * np.exp(np.cumsum(rng.normal(0.0000, 0.006, n)))
    df["GLD"] = gold / 10
    return df


def load_panel(offline: bool = False, force_refresh: bool = False) -> pd.DataFrame:
    """
    Load and combine FRED + Yahoo into a single daily panel.

    - Caches raw FRED and Yahoo frames to disk.
    - In offline mode (or on failure) falls back to cached or synthetic data.
    - Resolves the DXY name collision between Yahoo (DX-Y.NYB) and FRED
      (DTWEXBGS) by renaming the Yahoo column to DXY_YH.
    """
    end = pd.Timestamp.today().strftime("%Y-%m-%d")

    if force_refresh or not offline:
        try:
            fred = fetch_fred_panel(end=end)
            yahoo = fetch_yahoo(end=end)
            if not fred.empty:
                _save_cache(fred, "fred")
            if not yahoo.empty:
                _save_cache(yahoo, "yahoo")
        except Exception as exc:  # pragma: no cover
            logger.warning("Live download failed (%s). Falling back to cache/synthetic.", exc)
            fred = _load_cache("fred")
            yahoo = _load_cache("yahoo")
    else:
        fred = _load_cache("fred")
        yahoo = _load_cache("yahoo")

    if fred is None or fred.empty:
        fred = _synthetic_fred()
        logger.info("Using synthetic FRED data (offline).")
    if yahoo is None or yahoo.empty:
        yahoo = _synthetic_yahoo()
        logger.info("Using synthetic Yahoo data (offline).")

    # Resolve DXY name collision: keep FRED broad index as DXY, rename Yahoo to DXY_YH
    if "DXY" in yahoo.columns and "DXY" in fred.columns:
        yahoo = yahoo.rename(columns={"DXY": "DXY_YH"})

    # Apply publication lag to FRED series (shift forward in index time)
    fred_lagged = fred.copy()
    for col, lag in FRED_LAG.items():
        if col in fred_lagged:
            fred_lagged[col] = fred_lagged[col].shift(lag)

    # Resample FRED (monthly) to daily and forward-fill.
    fred_daily = fred_lagged.resample("D").ffill()
    yahoo_daily = yahoo.resample("D").ffill()

    panel = yahoo_daily.join(fred_daily, how="outer").sort_index()
    panel = panel.loc[pd.Timestamp(BACKTEST_START):end]
    panel = panel.ffill().dropna(how="all")
    _save_cache(panel, "panel")
    return panel
