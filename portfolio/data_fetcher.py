"""\
Data fetching module for the trading strategy backtester.\
\
This module provides functions for fetching OHLCV data from Yahoo Finance,\
generating synthetic data, and mapping supported instruments to their\
yfinance ticker symbols. It includes automatic handling of the maximum\
historical period available for each interval.\
"""\
\
from __future__ import annotations\
\
import datetime as dt\
from typing import Optional\
\
import numpy as np\
import pandas as pd\
import yfinance as yf\
\
# Map supported instruments to yfinance ticker symbols.\
SUPPORTED_INSTRUMENTS = {\
"EURUSD": "EURUSD=X",\
"GBPUSD": "GBPUSD=X",\
"USDJPY": "USDJPY=X",\
"AUDUSD": "AUDUSD=X",\
"XAUUSD": "GC=F",\
}\
\
# Map interval to the maximum period yfinance allows.\
YFINANCE_MAX_PERIOD = {\
"1m": "7d",\
"5m": "60d",\
"15m": "60d",\
"30m": "60d",\
"1h": "730d",\
"1d": "max",\
"1wk": "max",\
"1mo": "max",\
}\
\
# Standard column names for the OHLCV DataFrame.\
STANDARD_COLUMNS = ["Time", "Open", "High", "Low", "Close"]\
\
# Interval to timedelta mapping for calculating historical start dates.\
INTERVAL_TO_TIMEDELTA = {\
"1m": dt.timedelta(days=7),\
"5m": dt.timedelta(days=60),\
"15m": dt.timedelta(days=60),\
"30m": dt.timedelta(days=60),\
"1h": dt.timedelta(days=730),\
"1d": dt.timedelta(days=365 * 30), # Roughly max for daily interval\
"1wk": dt.timedelta(weeks=52 * 30),\
"1mo": dt.timedelta(days=365 * 30),\
}\
\
\
def max_history_start(interval: str, end_date: dt.date) -> dt.date:\
"""\
Return the earliest start date for the given interval.\
\
The start date is calculated by subtracting the appropriate timedelta\
from the end date based on the interval.\
\
Args:\
interval (str): The trading interval (e.g. '1h', '1d').\
end_date (dt.date): The end date (typically today).\
\
Returns:\
dt.date: The earliest start date for the interval.\
"""\
if interval not in INTERVAL_TO_TIMEDELTA:\
raise ValueError(f"Unsupported interval for max_history_start: {interval}")\
delta = INTERVAL_TO_TIMEDELTA[interval]\
return end_date - delta\
\
\
def fetch_ohlcv(\
symbol: str,\
interval: str = "1h",\
period: Optional[str] = None,\
start: Optional[str] = None,\
end: Optional[str] = None,\
) -> pd.DataFrame:\
"""\
Fetch OHLCV data from Yahoo Finance.\
\
The function fetches data using the following precedence:\
1. If a yfinance 'period' is provided, it is used directly.\
2. Otherwise, if start/end dates are provided, they are used.\
3. Otherwise, the maximum historical range for the given interval is used,\
from the earliest date yfinance allows for the interval to today.\
\
Args:\
symbol (str): The trading instrument symbol (e.g., 'EURUSD', 'XAUUSD').\
interval (str): The data interval (e.g., '1m', '1h', '1d').\
period (Optional[str]): A yfinance period string (e.g., '1y', '2y', 'max').\
start (Optional[str]): Start date string in YYYY-MM-DD format.\
end (Optional[str]): End date string in YYYY-MM-DD format.\
\
Returns:\
pd.DataFrame: A DataFrame with columns [Time, Open, High, Low, Close].\
"""\
if symbol not in SUPPORTED_INSTRUMENTS:\
raise ValueError(f"Unsupported instrument: {symbol}")\
\
ticker = SUPPORTED_INSTRUMENTS[symbol]\
\
# Determine the fetch parameters based on precedence.\
if period is not None:\
history_kwargs: dict = {"period": period}\
elif start is not None or end is not None:\
history_kwargs = {"start": start, "end": end}\
else:\
# Use the maximum history available for the interval.\
today = dt.date.today()\
start_date = max_history_start(interval, today)\
history_kwargs = {"start": start_date.strftime("%Y-%m-%d"), "end": today.strftime("%Y-%m-%d")}\
\
try:\
data = yf.download(ticker, interval=interval, auto_adjust=False, progress=False, **history_kwargs)\
except Exception as e:\
raise RuntimeError(f"Failed to fetch data for {symbol}: {e}") from e\
\
if data is None or data.empty:\
raise ValueError(f"No data returned for {symbol} at interval {interval}")\
\
# Ensure one-level column names if multi-index columns are present.\
if isinstance(data.columns, pd.MultiIndex):\
data.columns = data.columns.get_level_values(0)\
\
# Standardize and reset the index to produce a reliable 'Time' column.\
data = data.reset_index()\
data = data.rename(columns={"Date": "Time", "Datetime": "Time", "index": "Time"})\
\
if "Time" not in data.columns:\
data.index.name = "Time"\
data = data.reset_index()\
\
# Select and reorder standard columns.\
available_columns = [col for col in STANDARD_COLUMNS if col in data.columns]\
if "Time" not in available_columns:\
raise ValueError("Time column is missing after processing")\
\
data = data[available_columns].copy()\
\
# Convert Time column to datetime and set as index for consistent handling.\
data["Time"] = pd.to_datetime(data["Time"])\
data = data.set_index("Time")\
return data\
\
\
def generate_synthetic_data(\
start_date: str,\
end_date: str,\
freq: str = "1h",\
seed: int = 42,\
) -> pd.DataFrame:\
"""\
Generate synthetic OHLCV data for testing purposes.\
\
Args:\
start_date (str): Start date string in YYYY-MM-DD format.\
end_date (str): End date string in YYYY-MM-DD format.\
freq (str): Frequency string for pandas date_range (e.g., '1h', '1d').\
seed (int): Random seed for reproducibility.\
\
Returns:\
pd.DataFrame: A DataFrame with columns [Time, Open, High, Low, Close].\
"""\
rng = np.random.default_rng(seed)\
date_range = pd.date_range(start=start_date, end=end_date, freq=freq)\
\
if len(date_range) == 0:\
raise ValueError("Empty date range for synthetic data generation")\
\
# Generate random price series.\
n = len(date_range)\
base_price = 1.0 if "USD" not in start_date else 1800.0 # Simple heuristic\
returns = rng.normal(0, 0.001, n)\
close = base_price * np.exp(np.cumsum(returns))\
open_prices = close * (1 + rng.normal(0, 0.0005, n))\
high = np.maximum(open_prices, close) * (1 + np.abs(rng.normal(0, 0.001, n)))\
low = np.minimum(open_prices, close) * (1 - np.abs(rng.normal(0, 0.001, n)))\
\
data = pd.DataFrame(\
{\
"Time": date_range,\
"Open": open_prices,\
"High": high,\
"Low": low,\
"Close": close,\
}\
)\
\
return data\
\
\
def convert_to_eet(df: pd.DataFrame) -> pd.DataFrame:\
"""\
Convert the Time index of the DataFrame to EET (Eastern European Time).\
\
The function handles both naive and timezone-aware datetimes safely by\
localizing naive datetimes to UTC before converting to EET.\
\
Args:\
df (pd.DataFrame): DataFrame with a Time index or column.\
\
Returns:\
pd.DataFrame: DataFrame with Time index converted to EET.\
"""\
if df.index.name == "Time" or "Time" in df.columns:\
time_series = df.index if df.index.name == "Time" else df["Time"]\
else:\
# Assume the index is the time column if no 'Time' column exists.\
time_series = df.index\
\
# Ensure the time series is a pandas Series.\
if not isinstance(time_series, pd.Series):\
time_series = pd.Series(time_series)\
\
# Handle timezone conversion safely.\
if time_series.dt.tz is None:\
# Localize naive datetimes to UTC before converting.\
time_series = time_series.dt.tz_localize("UTC")\
else:\
time_series = time_series.dt.tz_convert("UTC")\
\
# Convert to EET (UTC+2, or UTC+3 in summer for Eastern European Time).\
eet = time_series.dt.tz_convert("Europe/Athens") # Uses EET/EEST properly\
eet = eet.dt.tz_localize(None) # Remove tz info for display purposes\
\
# Rebuild the DataFrame with the converted time.\
result_df = df.copy()\
if df.index.name == "Time":\
result_df.index = eet\
else:\
result_df.loc[:, "Time"] = eet\
\
return result_df\
\
\
def get_latest_bar(df: pd.DataFrame) -> pd.Series:\
"""\
Get the latest bar from the data, which is used for the current signal.\
\
Args:\
df (pd.DataFrame): OHLCV DataFrame with Time index.\
\
Returns:\
pd.Series: The latest data as a Series.\
"""\
return df.iloc[-1]
