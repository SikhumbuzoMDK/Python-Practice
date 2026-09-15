import pandas as pd\
import numpy as np\
from datetime import datetime, timedelta\
import os\
\
# Define instrument configurations\
INSTRUMENTS = {\
'EURUSD': {'start_price': 1.0850, 'volatility': 0.00035},\
'GBPUSD': {'start_price': 1.2800, 'volatility': 0.00040},\
'USDJPY': {'start_price': 152.50, 'volatility': 0.045},\
'AUDUSD': {'start_price': 0.6550, 'volatility': 0.00030},\
'XAUUSD': {'start_price': 2750.00, 'volatility': 1.20}\
}\
\
def generate_ohlc_data(symbol, start_price, volatility, start_date, end_date):\
"""\
Generate synthetic OHLC data using geometric Brownian motion / random walk\
\
Parameters:\
symbol (str): Instrument symbol\
start_price (float): Starting price\
volatility (float): Per-bar volatility\
start_date (datetime): Start date\
end_date (datetime): End date\
\
Returns:\
pd.DataFrame: DataFrame with Time (EET), Open, High, Low, Close columns\
"""\
\
# Generate timestamps for 15-minute bars\
# 96 bars per day (24 hours * 60 minutes / 15 minutes)\
time_index = pd.date_range(start=start_date, end=end_date, freq='15min')\
\
# Remove weekends (Saturday and Sunday)\
# Forex market is closed on weekends\
time_index = time_index[time_index.dayofweek < 5]\
\
n_bars = len(time_index)\
\
# Generate random walk returns\
# Using geometric Brownian motion\
# Returns ~ N(0, volatility^2)\
returns = np.random.normal(0, volatility, n_bars)\
\
# Calculate close prices\
close_prices = start_price * np.exp(np.cumsum(returns))\
\
# Clip prices to realistic ranges\
range_clips = {\
'EURUSD': (1.05, 1.12),\
'GBPUSD': (1.25, 1.35),\
'USDJPY': (145, 160),\
'AUDUSD': (0.62, 0.70),\
'XAUUSD': (2500, 3000)\
}\
\
min_price, max_price = range_clips.get(symbol, (0, float('inf')))\
close_prices = np.clip(close_prices, min_price, max_price)\
\
# Calculate opens (previous close, with slight gap)\
open_prices = np.zeros(n_bars)\
open_prices[0] = start_price\
open_prices[1:] = close_prices[:-1] + np.random.normal(0, volatility * 0.3, n_bars - 1)\
\
# Calculate high and low\
# High should be max(open, close) + some random amount\
# Low should be min(open, close) - some random amount\
high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.normal(0, volatility * 0.5, n_bars))\
low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.normal(0, volatility * 0.5, n_bars))\
\
# Ensure the price ranges are maintained\
high_prices = np.clip(high_prices, min_price, max_price)\
low_prices = np.clip(low_prices, min_price, max_price)\
\
# Create DataFrame\
df = pd.DataFrame({\
'Time (EET)': time_index.strftime('%Y.%m.%d %H:%M:%S'),\
'Open': open_prices,\
'High': high_prices,\
'Low': low_prices,\
'Close': close_prices\
})\
\
# Round prices appropriately based on instrument\
decimal_places = 5 if symbol in ['EURUSD', 'GBPUSD', 'AUDUSD'] else (3 if symbol == 'USDJPY' else 2)\
\
for col in ['Open', 'High', 'Low', 'Close']:\
df[col] = df[col].round(decimal_places)\
\
# Final validation: ensure OHLC relationships are correct\
df['High'] = df[['Open', 'Close', 'High']].max(axis=1)\
df['Low'] = df[['Open', 'Close', 'Low']].min(axis=1)\
\
return df\
\
def save_instrument_data(symbol, start_price, volatility):\
"""\
Generate and save data for a single instrument\
\
Parameters:\
symbol (str): Instrument symbol\
start_price (float): Starting price\
volatility (float): Per-bar volatility\
"""\
\
# Define date range for 2025\
start_date = datetime(2025, 1, 1, 0, 0, 0)\
end_date = datetime(2025, 12, 31, 23, 45, 0)\
\
print(f"Generating data for {symbol}...")\
\
# Generate data\
df = generate_ohlc_data(symbol, start_price, volatility, start_date, end_date)\
\
# Create filename matching original naming convention\
filename = f"{symbol}_15 Mins_Bid_2025.01.01_2025.12.31.csv"\
\
# Save to CSV\
df.to_csv(filename, index=False)\
\
print(f"✓ Saved {filename} - {len(df)} bars generated")\
print(f" Price range: {df['Low'].min():.5f} - {df['High'].max():.5f}")\
\
return filename\
\
def main():\
"""\
Main function to generate sample data for all instruments\
"""\
print("=" * 60)\
print("SAMPLE DATA GENERATOR FOR BACKTEST STRATEGY")\
print("=" * 60)\
print()\
\
files_created = []\
\
for symbol, config in INSTRUMENTS.items():\
filename = save_instrument_data(symbol, config['start_price'], config['volatility'])\
files_created.append(filename)\
\
print()\
print("=" * 60)\
print("GENERATION COMPLETE")\
print("=" * 60)\
print(f"Created {len(files_created)} CSV files:")\
for f in files_created:\
print(f" • {f}")\
print()\
print("Data format: Time (EET), Open, High, Low, Close")\
print("Timeframe: 15 minutes")\
print("Period: January 1, 2025 - December 31, 2025")\
print("Weekends excluded (market closed)")\
\
if __name__ == "__main__":\
main()
