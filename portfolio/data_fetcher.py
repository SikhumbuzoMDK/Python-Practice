import pandas as pd\
import numpy as np\
import yfinance as yf\
from datetime import datetime, timezone\
from typing import Optional, Tuple\
import os\
\
# Supported instruments mapping friendly names to yfinance symbols\
SUPPORTED_INSTRUMENTS = {\
'EURUSD': 'EURUSD=X',\
'GBPUSD': 'GBPUSD=X',\
'USDJPY': 'USDJPY=X',\
'AUDUSD': 'AUDUSD=X',\
'XAUUSD': 'GC=F' # Gold futures\
}\
\
STANDARD_COLUMNS = ['Time', 'Open', 'High', 'Low', 'Close']\
\
\
def fetch_ohlcv(symbol: str, start: str, end: str, interval: str = '1h') -> pd.DataFrame:\
"""\
Fetch OHLCV data from yfinance.\
\
Parameters\
----------\
symbol : str\
Instrument symbol (friendly name or yfinance symbol). If a friendly name\
is provided (e.g., 'EURUSD'), it will be converted to yfinance symbol.\
start : str\
Start date in 'YYYY-MM-DD' format.\
end : str\
End date in 'YYYY-MM-DD' format.\
interval : str, default='1h'\
Interval for OHLCV data (e.g., '1h', '1d', '15m').\
\
Returns\
-------\
pd.DataFrame\
DataFrame with columns: Time, Open, High, Low, Close.\
\
Raises\
------\
ValueError\
If symbol is not supported or data cannot be fetched.\
"""\
# Resolve symbol to yfinance symbol\
yf_symbol = SUPPORTED_INSTRUMENTS.get(symbol, symbol)\
\
try:\
# Fetch data using yfinance\
df = yf.download(yf_symbol, start=start, end=end, interval=interval, progress=False)\
\
if df.empty:\
raise ValueError(f"No data returned for symbol '{symbol}' ({yf_symbol}) in the given date range.")\
\
# Check if MultiIndex columns (common with yfinance when multiple tickers)\
if isinstance(df.columns, pd.MultiIndex):\
df.columns = df.columns.get_level_values(0)\
\
# Ensure required columns exist\
required = ['Open', 'High', 'Low', 'Close']\
for col in required:\
if col not in df.columns:\
raise ValueError(f"Missing column '{col}' in fetched data.")\
\
# Drop rows with NaN in OHLC columns\
df = df.dropna(subset=required)\
\
# Standardize index name\
df.index.name = 'Time'\
\
# Convert index to datetime if not already\
if not isinstance(df.index, pd.DatetimeIndex):\
df.index = pd.to_datetime(df.index)\
\
# Sort chronologically\
df = df.sort_index()\
\
# Return standardized dataframe with Time column (reset index)\
result = df[required].copy()\
result.index = pd.to_datetime(result.index)\
result.reset_index(inplace=True)\
result.rename(columns={'index': 'Time'}, inplace=True)\
\
return result\
\
except Exception as e:\
raise ValueError(f"Failed to fetch OHLCV data for {symbol} ({yf_symbol}): {str(e)}")\
\
\
def generate_synthetic_data(symbol: str, n: int = 5000, seed: int = 42) -> pd.DataFrame:\
"""\
Generate synthetic OHLCV data using geometric Brownian motion.\
\
Parameters\
----------\
symbol : str\
Instrument name (used only for reference).\
n : int, default=5000\
Number of data points to generate.\
seed : int, default=42\
Random seed for reproducibility.\
\
Returns\
-------\
pd.DataFrame\
Synthetic OHLCV data with columns: Time, Open, High, Low, Close.\
"""\
np.random.seed(seed)\
\
# Base price level depends on instrument (rough approximation)\
base_price = 1.0\
if 'XAU' in symbol or symbol == 'XAUUSD':\
base_price = 1800.0 # Gold approx\
elif symbol in ['EURUSD', 'GBPUSD']:\
base_price = 1.1 # EUR/USD approx\
elif symbol == 'USDJPY':\
base_price = 140.0 # JPY approx\
elif symbol == 'AUDUSD':\
base_price = 0.65 # AUD/USD approx\
\
# Parameters for GBM\
mu = 0.00005 # small drift per step\
sigma = 0.001 # volatility per step\
\
# Generate log returns (geometric Brownian motion)\
dt = 1.0\
returns = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n)\
\
# Generate close prices by cumulative product\
closes = base_price * np.exp(np.cumsum(returns))\
\
# Generate open prices (previous close, first uses base price)\
opens = np.empty(n)\
opens[0] = base_price\
opens[1:] = closes[:-1]\
\
# Generate realistic high/low based on open and close\
# Random intra-bar ranges proportional to price volatility\
intrabar_range = np.abs(np.random.normal(0, sigma * 0.5, n)) * closes\
\
highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, sigma * 0.3, n)) * closes\
lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, sigma * 0.3, n)) * closes\
\
# Ensure high >= max(open, close) and low <= min(open, close)\
highs = np.maximum(highs, np.maximum(opens, closes))\
lows = np.minimum(lows, np.minimum(opens, closes))\
\
# Generate timestamps (hourly intervals going back from now)\
end_time = datetime.now(timezone.utc)\
times = pd.date_range(end=end_time, periods=n, freq='h')\
\
# Create DataFrame\
df = pd.DataFrame({\
'Time': times,\
'Open': opens,\
'High': highs,\
'Low': lows,\
'Close': closes\
})\
\
return df\
\
\
def load_uploaded_csv(file_path: str) -> pd.DataFrame:\
"""\
Load an uploaded CSV file and standardize its format.\
\
Parameters\
----------\
file_path : str\
Path to the uploaded CSV file.\
\
Returns\
-------\
pd.DataFrame\
Standardized DataFrame with columns: Time, Open, High, Low, Close.\
\
Raises\
------\
ValueError\
If required columns are missing or parsing fails.\
"""\
try:\
# Read the CSV\
df = pd.read_csv(file_path)\
except Exception as e:\
raise ValueError(f"Failed to read CSV file: {str(e)}")\
\
# Check for required columns\
expected_time_col = 'Time (EET)'\
required_cols = ['Open', 'High', 'Low', 'Close']\
\
if expected_time_col not in df.columns:\
# Also accept 'Time' as fallback\
if 'Time' in df.columns:\
time_col = 'Time'\
else:\
raise ValueError(f"Column '{expected_time_col}' or 'Time' not found in CSV.")\
else:\
time_col = expected_time_col\
\
for col in required_cols:\
if col not in df.columns:\
raise ValueError(f"Required column '{col}' not found in CSV.")\
\
# Parse time column - handle EET timezone (UTC+2)\
df[time_col] = pd.to_datetime(df[time_col], errors='coerce')\
\
# Drop rows with invalid time\
df = df.dropna(subset=[time_col])\
\
# Convert EET to UTC (EET = UTC+2, no DST adjustment for simplicity)\
df[time_col] = df[time_col].dt.tz_localize('Etc/GMT-2').dt.tz_convert('UTC')\
\
# Rename time column to 'Time'\
df.rename(columns={time_col: 'Time'}, inplace=True)\
\
# Standardize output columns\
for col in STANDARD_COLUMNS:\
if col not in df.columns:\
df[col] = pd.NA\
\
result = df[STANDARD_COLUMNS].copy()\
\
# Convert numeric columns to float\
for col in STANDARD_COLUMNS[1:]:\
result[col] = pd.to_numeric(result[col], errors='coerce')\
\
# Sort by time\
result = result.sort_values('Time')\
\
return result
