from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone, timedelta
from typing import Optional, List

# Supported instruments mapping friendly names to yfinance symbols
SUPPORTED_INSTRUMENTS = {
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X',
    'USDJPY': 'USDJPY=X',
    'AUDUSD': 'AUDUSD=X',
    'XAUUSD': 'GC=F',  # Gold futures
}

STANDARD_COLUMNS = ['Time', 'Open', 'High', 'Low', 'Close']

# Interval -> candidate periods to try, in order of preference (longest first).
# FX pairs have limited intraday history on yfinance, so we try progressively
# shorter periods until data is returned.
INTERVAL_PERIODS = {
    '1m': ['7d', '5d', '1d'],
    '5m': ['60d', '30d', '1mo'],
    '15m': ['60d', '30d', '1mo'],
    '30m': ['60d', '30d', '1mo'],
    '1h': ['730d', '1y', '60d', '30d'],
    '1d': ['max', '5y', '2y', '1y'],
    '1wk': ['max', '5y', '2y'],
    '1mo': ['max', '5y'],
}


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw yfinance dataframe to the standard Time/OHLC format."""
    if df is None or df.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    # Handle MultiIndex columns (multiple tickers)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ['Open', 'High', 'Low', 'Close']
    for col in required:
        if col not in df.columns:
            return pd.DataFrame(columns=STANDARD_COLUMNS)

    df = df.dropna(subset=required)

    result = df[required].copy()
    result.index = pd.to_datetime(result.index)
    result.index.name = 'Time'
    result = result.reset_index()
    if 'index' in result.columns:
        result = result.rename(columns={'index': 'Time'})
    result = result[STANDARD_COLUMNS]
    result = result.sort_values('Time').reset_index(drop=True)
    return result


def fetch_ohlcv(symbol: str, interval: str = '1h',
                start: Optional[str] = None, end: Optional[str] = None,
                period: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch OHLCV data from yfinance for the maximum available history,
    with automatic fallback to progressively shorter periods.

    - If `period` is provided, tries that period first.
    - Else if start/end provided, uses those explicit dates.
    - Else fetches using period-based approach (most reliable for FX intraday).
    """
    yf_symbol = SUPPORTED_INSTRUMENTS.get(symbol, symbol)

    # Determine candidate periods to try
    if period:
        candidates = [period]
    else:
        candidates = INTERVAL_PERIODS.get(interval, ['60d'])

    last_error = None
    for candidate in candidates:
        try:
            df = yf.download(yf_symbol, period=candidate, interval=interval, progress=False)
            if df is not None and not df.empty:
                result = _standardize(df)
                if len(result) > 0:
                    return result
        except Exception as e:
            last_error = str(e)
            continue

    # If period-based all failed and explicit dates were given, try date range
    if start and end:
        try:
            df = yf.download(yf_symbol, start=start, end=end, interval=interval, progress=False)
            if df is not None and not df.empty:
                result = _standardize(df)
                if len(result) > 0:
                    return result
        except Exception as e:
            last_error = str(e)

    raise ValueError(
        f"Failed to fetch OHLCV for {symbol} ({yf_symbol}) at interval {interval}. "
        f"yfinance may not have this much history for this instrument/interval. "
        f"Last error: {last_error}"
    )


def generate_synthetic_data(symbol: str, n: int = 5000, seed: int = 42,
                            freq: str = 'h') -> pd.DataFrame:
    """Generate synthetic OHLCV data via geometric Brownian motion."""
    np.random.seed(seed)

    if 'XAU' in symbol or symbol == 'XAUUSD':
        base_price = 1800.0
    elif symbol == 'USDJPY':
        base_price = 140.0
    elif symbol == 'AUDUSD':
        base_price = 0.65
    else:
        base_price = 1.1

    mu = 0.00005
    sigma = 0.001
    dt = 1.0

    returns = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n)
    closes = base_price * np.exp(np.cumsum(returns))

    opens = np.empty(n)
    opens[0] = base_price
    opens[1:] = closes[:-1]

    highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, sigma * 0.3, n)) * closes
    lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, sigma * 0.3, n)) * closes

    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    end_time = datetime.now(timezone.utc)
    times = pd.date_range(end=end_time, periods=n, freq=freq)

    return pd.DataFrame({
        'Time': times,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': closes,
    })


def load_uploaded_csv(file_path: str) -> pd.DataFrame:
    """Load an uploaded CSV and standardize it (supports original 'Time (EET)' format)."""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {str(e)}")

    if 'Time (EET)' in df.columns:
        time_col = 'Time (EET)'
    elif 'Time' in df.columns:
        time_col = 'Time'
    else:
        raise ValueError("Column 'Time (EET)' or 'Time' not found in CSV.")

    required_cols = ['Open', 'High', 'Low', 'Close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV.")

    parsed = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col]).copy()
    df['Time'] = parsed.dropna()

    if df['Time'].dt.tz is None:
        df['Time'] = df['Time'].dt.tz_localize('Etc/GMT-2').dt.tz_convert('UTC')
    else:
        df['Time'] = df['Time'].dt.tz_convert('UTC')

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    result = df[STANDARD_COLUMNS].copy()
    result = result.sort_values('Time').reset_index(drop=True)
    return result
