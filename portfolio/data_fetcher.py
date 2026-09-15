from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone, timedelta
from typing import Optional

# Supported instruments mapping friendly names to yfinance symbols
SUPPORTED_INSTRUMENTS = {
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X',
    'USDJPY': 'USDJPY=X',
    'AUDUSD': 'AUDUSD=X',
    'XAUUSD': 'GC=F',  # Gold futures
}

STANDARD_COLUMNS = ['Time', 'Open', 'High', 'Low', 'Close']

# yfinance maximum history for each interval (approximate limits)
YFINANCE_MAX_PERIOD = {
    '1m': '7d',
    '5m': '60d',
    '15m': '60d',
    '30m': '60d',
    '1h': '730d',
    '1d': 'max',
    '1wk': 'max',
    '1mo': 'max',
}

# Number of days covered by each interval's max period
INTERVAL_MAX_DAYS = {
    '1m': 7,
    '5m': 60,
    '15m': 60,
    '30m': 60,
    '1h': 730,
    '1d': 3650,
    '1wk': 3650,
    '1mo': 3650,
}


def max_history_start(interval: str, end_date: datetime = None) -> str:
    """Return the earliest feasible start date for the given interval."""
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    days = INTERVAL_MAX_DAYS.get(interval, 3650)
    start = end_date - timedelta(days=days)
    return start.strftime('%Y-%m-%d')


def fetch_ohlcv(symbol: str, interval: str = '1h',
                start: Optional[str] = None, end: Optional[str] = None,
                period: Optional[str] = None) -> pd.DataFrame:
    """Fetch OHLCV data from yfinance for the maximum available history.

    - If `period` is given, uses yfinance period (e.g. '1y', 'max').
    - Else if start/end given, uses those dates.
    - Else fetches from max history for the interval up to today.
    The current signal is derived from the latest bar (today).
    """
    yf_symbol = SUPPORTED_INSTRUMENTS.get(symbol, symbol)

    try:
        if period:
            df = yf.download(yf_symbol, period=period, interval=interval, progress=False)
        else:
            if end is None:
                end = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            if start is None:
                start = max_history_start(interval, datetime.now(timezone.utc))
            df = yf.download(yf_symbol, start=start, end=end, interval=interval, progress=False)

        if df.empty:
            raise ValueError(f"No data returned for symbol '{symbol}' ({yf_symbol}).")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ['Open', 'High', 'Low', 'Close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}'.")

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

    except Exception as e:
        raise ValueError(f"Failed to fetch OHLCV for {symbol} ({yf_symbol}): {str(e)}")


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
