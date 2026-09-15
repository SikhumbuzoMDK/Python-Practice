import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone
from typing import Optional, Tuple

# Supported instruments mapping friendly names to yfinance symbols
SUPPORTED_INSTRUMENTS = {
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X',
    'USDJPY': 'USDJPY=X',
    'AUDUSD': 'AUDUSD=X',
    'XAUUSD': 'GC=F'  # Gold futures
}

STANDARD_COLUMNS = ['Time', 'Open', 'High', 'Low', 'Close']


def fetch_ohlcv(symbol: str, start: str, end: str, interval: str = '1h') -> pd.DataFrame:
    """
    Fetch OHLCV data from yfinance for a given date range.

    Parameters
    ----------
    symbol : str
        Instrument symbol (friendly name or yfinance symbol).
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.
    interval : str, default='1h'
        Interval for OHLCV data (e.g., '1h', '1d', '15m').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Time, Open, High, Low, Close.
    """
    yf_symbol = SUPPORTED_INSTRUMENTS.get(symbol, symbol)

    try:
        df = yf.download(yf_symbol, start=start, end=end, interval=interval, progress=False)

        if df.empty:
            raise ValueError(f"No data returned for symbol '{symbol}' ({yf_symbol}).")

        # Handle MultiIndex columns (multiple tickers usually)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ['Open', 'High', 'Low', 'Close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}'.")

        df = df.dropna(subset=required)

        # Standardize: Time as a column, not index
        result = df[required].copy()
        result.index = pd.to_datetime(result.index)
        result.index.name = 'Time'
        result = result.reset_index()   # creates 'Time' column since index is named
        result = result[STANDARD_COLUMNS]
        result = result.sort_values('Time').reset_index(drop=True)

        return result

    except Exception as e:
        raise ValueError(f"Failed to fetch OHLCV for {symbol} ({yf_symbol}): {str(e)}")


def generate_synthetic_data(symbol: str, n: int = 5000, seed: int = 42, freq: str = 'h') -> pd.DataFrame:
    """
    Generate synthetic OHLCV data via geometric Brownian motion.

    Parameters
    ----------
    symbol : str
        Instrument name (used for reference/base price).
    n : int, default=5000
        Number of data points.
    seed : int, default=42
        Random seed for reproducibility.
    freq : str, default='h'
        Timestamp frequency ('h' for hourly, '15min' for 15-min, etc.).

    Returns
    -------
    pd.DataFrame
        Synthetic OHLCV data with columns Time, Open, High, Low, Close.
    """
    np.random.seed(seed)

    if 'XAU' in symbol or symbol == 'XAUUSD':
        base_price = 1800.0   # Gold
    elif symbol == 'USDJPY':
        base_price = 140.0    # JPY
    elif symbol == 'AUDUSD':
        base_price = 0.65     # AUD/USD
    else:
        base_price = 1.1      # EUR/USD, GBP/USD

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
    """
    Load an uploaded CSV and standardize it (supports original 'Time (EET)' format).

    Returns a DataFrame with columns: Time, Open, High, Low, Close.
    """
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

    # Parse time, handle possible timezone-aware input safely
    parsed = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col]).copy()
    df['Time'] = parsed.dropna()

    # Convert EET (UTC+2) to UTC if naive
    if df['Time'].dt.tz is None:
        df['Time'] = df['Time'].dt.tz_localize('Etc/GMT-2').dt.tz_convert('UTC')
    else:
        df['Time'] = df['Time'].dt.tz_convert('UTC')

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    result = df[STANDARD_COLUMNS].copy()
    result = result.sort_values('Time').reset_index(drop=True)

    return result
