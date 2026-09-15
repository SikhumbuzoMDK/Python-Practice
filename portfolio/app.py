import streamlit as st\
import pandas as pd\
import numpy as np\
import plotly.express as px\
from datetime import datetime\
\
from data_fetcher import (\
fetch_ohlcv,\
generate_synthetic_data,\
SUPPORTED_INSTRUMENTS,\
INTERVAL_PERIODS,\
)\
\
# -----------------------------------------------------------------------------\
# Common Utilities\
# -----------------------------------------------------------------------------\
\
def compute_metrics(trades):\
"""Compute all metrics numerically. Returns dict with all numeric values."""\
if trades is None or len(trades) == 0:\
return {\
'total_trades': 0,\
'win_rate': 0.0,\
'total_r': 0.0,\
'avg_r': 0.0,\
'profit_factor': 0.0,\
'max_drawdown': 0.0,\
}\
\
total_trades = len(trades)\
wins = trades[trades['R'] > 0]\
win_rate = len(wins) / total_trades if total_trades else 0.0\
total_r = trades['R'].sum()\
avg_r = trades['R'].mean()\
\
losses = trades[trades['R'] < 0]['R'].sum()\
profits = trades[trades['R'] > 0]['R'].sum()\
profit_factor = abs(profits / losses) if losses != 0 else float('inf')\
\
equity = trades['Equity'].values\
if len(equity) > 0:\
peak = np.maximum.accumulate(equity)\
drawdown = (equity - peak) / peak\
max_drawdown = drawdown.min()\
else:\
max_drawdown = 0.0\
\
return {\
'total_trades': total_trades,\
'win_rate': win_rate,\
'total_r': total_r,\
'avg_r': avg_r,\
'profit_factor': profit_factor,\
'max_drawdown': max_drawdown,\
}\
\
\
def format_metrics_for_display(metrics):\
"""Format metrics for display with strings/percentages."""\
formatted = {}\
for k, v in metrics.items():\
if k == 'win_rate':\
formatted[k] = f"{v*100:.1f}%"\
elif k == 'max_drawdown':\
formatted[k] = f"{v*100:.2f}%"\
elif k in ['total_r', 'avg_r', 'profit_factor']:\
formatted[k] = f"{v:.2f}"\
else:\
formatted[k] = f"{v}"\
return formatted\
\
\
def position_sizing(account, risk_pct, entry, stop):\
"""Compute position size in units based on risk percentage."""\
if entry is None or stop is None:\
return 0.0\
risk_per_trade = account * (risk_pct / 100.0)\
risk_per_share = abs(entry - stop)\
if risk_per_share == 0:\
return 0.0\
return risk_per_trade / risk_per_share\
\
\
# -----------------------------------------------------------------------------\
# Strategy A: SMA-Slope Pullback\
# -----------------------------------------------------------------------------\
\
def run_backtest_sma_slope(data, p):\
"""Run SMA-Slope Pullback backtest."""\
df = data.copy()\
df['SMA'] = df['Close'].rolling(p['SMA_PERIOD']).mean()\
df['SMA_Slope'] = df['SMA'].diff(p['SLOPE_LOOKBACK'])\
\
df['TR'] = np.maximum(df['High'] - df['Low'],\
np.maximum(abs(df['High'] - df['Close'].shift()),\
abs(df['Low'] - df['Close'].shift())))\
df['ATR'] = df['TR'].rolling(p['ATR_PERIOD']).mean()\
\
\
trades = []\
state = 'LOOKING'\
swing_high = None\
swing_low = None\
pull_extreme = None\
pullback_bars = 0\
entry = sl = tp = 0.0\
raw_sl = 0.0\
direction = 0\
\
equity = 0.0\
risk_per_trade = 0.0\
\
for i in range(p['SMA_PERIOD'] + p['SLOPE_LOOKBACK'] + p['ATR_PERIOD'], len(df)):\
c = df['Close'].iloc[i]\
h = df['High'].iloc[i]\
l = df['Low'].iloc[i]\
sma = df['SMA'].iloc[i]\
slope = df['SMA_Slope'].iloc[i]\
atr = df['ATR'].iloc[i]\
\
pc = df['Close'].iloc[i-1]\
psma = df['SMA'].iloc[i-1]\
\
if state == 'LOOKING':\
if pc < psma and c > sma:\
state = 'BULL_IMPULSE'\
swing_high = h\
elif pc > psma and c < sma:\
state = 'BEAR_IMPULSE'\
swing_low = l\
\
elif state == 'BULL_IMPULSE':\
if h > swing_high:\
swing_high = h\
if l <= sma and slope >= p['SLOPE_THRESHOLD']:\
state = 'BULL_PULLBACK'\
pull_extreme = l\
pullback_bars = 0\
elif slope < p['SLOPE_THRESHOLD'] * p['INVALIDATE_MULT']:\
state = 'LOOKING'\
\
elif state == 'BULL_PULLBACK':\
if l < pull_extreme:\
pull_extreme = l\
if l < swing_low or slope < p['SLOPE_THRESHOLD'] * p['INVALIDATE_MULT']:\
state = 'LOOKING'\
else:\
pullback_bars += 1\
if c > sma and pullback_bars >= p['MIN_PULLBACK_BARS']:\
entry = c\
raw_sl = pull_extreme - atr * p['SL_BUFFER_MULT']\
risk = max(entry - raw_sl, p['MIN_RISK_PRICE'])\
sl = entry - risk\
tp = entry + risk * p['RR_TARGET']\
direction = 1\
state = 'IN_TRADE'\
risk_per_trade = risk\
equity = df['Close'].iloc[:i+1].sum() # placeholder; will be updated below\
\
elif state == 'BEAR_IMPULSE':\
if l < swing_low:\
swing_low = l\
if h >= sma and slope <= -p['SLOPE_THRESHOLD']:\
state = 'BEAR_PULLBACK'\
pull_extreme = h\
pullback_bars = 0\
elif slope > -p['SLOPE_THRESHOLD'] * p['INVALIDATE_MULT']:\
state = 'LOOKING'\
\
elif state == 'BEAR_PULLBACK':\
if h > pull_extreme:\
pull_extreme = h\
if h > swing_high or slope > -p['SLOPE_THRESHOLD'] * p['INVALIDATE_MULT']:\
state = 'LOOKING'\
else:\
pullback_bars += 1\
if c < sma and pullback_bars >= p['MIN_PULLBACK_BARS']:\
entry = c\
raw_sl = pull_extreme + atr * p['SL_BUFFER_MULT']\
risk = max(raw_sl - entry, p['MIN_RISK_PRICE'])\
sl = entry + risk\
tp = entry - risk * p['RR_TARGET']\
direction = -1\
state = 'IN_TRADE'\
risk_per_trade = risk\
equity = df['Close'].iloc[:i+1].sum() # placeholder; will be updated below\
\
elif state == 'IN_TRADE':\
if direction == 1:\
if l <= sl:\
exit_price = sl\
exit_time = df.index[i]\
entry_time = df.index[entry_idx] if 'entry_idx' in locals() else None\
r = (exit_price - entry) / risk_per_trade\
trades.append({\
'Entry Time': entry_time,\
'Exit Time': exit_time,\
'Direction': 'LONG',\
'Entry': entry,\
'Stop': sl,\
'Target': tp,\
'Exit': exit_price,\
'R': r,\
'Equity': exit_price if len(trades)==0 else trades[-1]['Equity'] + (exit_price - entry)\
})\
state = 'LOOKING'\
elif h >= tp:\
exit_price = tp\
exit_time = df.index[i]\
entry_time = df.index[entry_idx] if 'entry_idx' in locals() else None\
r = (exit_price - entry) / risk_per_trade\
trades.append({\
'Entry Time': entry_time,\
'Exit Time': exit_time,\
'Direction': 'LONG',\
'Entry': entry,\
'Stop': sl,\
'Target': tp,\
'Exit': exit_price,\
'R': r,\
'Equity': exit_price if len(trades)==0 else trades[-1]['Equity'] + (exit_price - entry)\
})\
state = 'LOOKING'\
elif direction == -1:\
if h >= sl:\
exit_price = sl\
exit_time = df.index[i]\
entry_time = df.index[entry_idx] if 'entry_idx' in locals() else None\
r = (entry - exit_price) / risk_per_trade\
trades.append({\
'Entry Time': entry_time,\
'Exit Time': exit_time,\
'Direction': 'SHORT',\
'Entry': entry,\
'Stop': sl,\
'Target': tp,\
'Exit': exit_price,\
'R': r,\
'Equity': exit_price if len(trades)==0 else trades[-1]['Equity'] - (entry - exit_price)\
})\
state = 'LOOKING'\
elif l <= tp:\
exit_price = tp\
exit_time = df.index[i]\
entry_time = df.index[entry_idx] if 'entry_idx' in locals() else None\
r = (entry - exit_price) / risk_per_trade\
trades.append({\
'Entry Time': entry_time,\
'Exit Time': exit_time,\
'Direction': 'SHORT',\
'Entry': entry,\
'Stop': sl,\
'Target': tp,\
'Exit': exit_price,\
'R': r,\
'Equity': exit_price if len(trades)==0 else trades[-1]['Equity'] - (entry - exit_price)\
})\
state = 'LOOKING'\
\
# track entry_idx for trade time\
if state == 'IN_TRADE' and ('entry_idx' not in locals() or entry_idx is None):\
entry_idx = i\
\
if trades:\
trades_df = pd.DataFrame(trades)\
# Fix Entry Time if None - use previous bar\
trades_df['Entry Time'] = trades_df['Entry Time'].fillna(df.index[0])\
# recompute equity properly starting from initial capital\
equity_curve = []\
current_equity = 10000\
for idx, row in trades_df.iterrows():\
current_equity += row['R'] * 100 # assume risk = 100 per trade for simplicity\
equity_curve.append(current_equity)\
trades_df['Equity'] = equity_curve\
return trades_df\
else:\
return pd.DataFrame(columns=['Entry Time', 'Exit Time', 'Direction', 'Entry', 'Stop', 'Target', 'Exit', 'R', 'Equity'])\
\
\
def compute_signal_sma_slope(data, p):\
"""Compute current signal for SMA-Slope Pullback."""\
df = data.copy()\
df['SMA'] = df['Close'].rolling(p['SMA_PERIOD']).mean()\
df['SMA_Slope'] = df['SMA'].diff(p['SLOPE_LOOKBACK'])\
\
df['TR'] = np.maximum(df['High'] - df['Low'],\
np.maximum(abs(df['High'] - df['Close'].shift()),\
abs(df['Low'] - df['Close'].shift())))\
df['ATR'] = df['TR'].rolling(p['ATR_PERIOD']).mean()\
\
last = df.iloc[-1]\
prev = df.iloc[-2]\
c = last['Close']\
h = last['High']\
l = last['Low']\
sma = last['SMA']\
slope = last['SMA_Slope']\
atr = last['ATR']\
pc = prev['Close']\
psma = prev['SMA']\
\
if pc < psma and c > sma:\
return 'BULL_IMPULSE'\
elif pc > psma and c < sma:\
return 'BEAR_IMPULSE'\
elif c > sma and slope >= p['SLOPE_THRESHOLD']:\
return 'BULL_PULLBACK_READY'\
elif c < sma and slope <= -p['SLOPE_THRESHOLD']:\
return 'BEAR_PULLBACK_READY'\
else:\
return 'NEUTRAL'\
\
\
PARAMS_SMA_SLOPE = {\
'Base Case': {\
'SMA_PERIOD': 20,\
'SLOPE_LOOKBACK': 5,\
'SLOPE_THRESHOLD': 0.5,\
'INVALIDATE_MULT': 0.8,\
'MIN_PULLBACK_BARS': 3,\
'SL_BUFFER_MULT': 1.5,\
'MIN_RISK_PRICE': 0.5,\
'RR_TARGET': 2.0,\
'ATR_PERIOD': 14\
},\
'Loose Filter': {\
'SMA_PERIOD': 20,\
'SLOPE_LOOKBACK': 5,\
'SLOPE_THRESHOLD': 0.2,\
'INVALIDATE_MULT': 0.5,\
'MIN_PULLBACK_BARS': 2,\
'SL_BUFFER_MULT': 1.0,\
'MIN_RISK_PRICE': 0.3,\
'RR_TARGET': 2.0,\
'ATR_PERIOD': 14\
}\
}\
\
\
# -----------------------------------------------------------------------------\
# Strategy B: BOS/CHoCH Fibonacci\
# -----------------------------------------------------------------------------\
\
def find_swing_points(data, window=10):\
"""Find swing highs and lows in data."""\
swing_highs = []\
swing_lows = []\
\
for i in range(window, len(data) - window):\
# Swing High\
if data['High'].iloc[i] == max(data['High'].iloc[i-window:i+window+1]):\
swing_highs.append({'index': i, 'price': data['High'].iloc[i]})\
# Swing Low\
if data['Low'].iloc[i] == min(data['Low'].iloc[i-window:i+window+1]):\
swing_lows.append({'index': i, 'price': data['Low'].iloc[i]})\
\
return swing_highs, swing_lows\
\
\
def calculate_fibonacci_levels(high_price, low_price):\
"""Calculate Fibonacci levels."""\
diff = high_price - low_price\
return {\
'0%': low_price,\
'23.6%': low_price + diff * 0.236,\
'38.2%': low_price + diff * 0.382,\
'50%': low_price + diff * 0.5,\
'61.8%': low_price + diff * 0.618,\
'78.6%': low_price + diff * 0.786,\
'100%': high_price,\
'161.8%': low_price + diff * 1.618\
}\
\
\
def run_backtest_fib(data, window=10):\
"""Run BOS/CHoCH Fibonacci backtest."""\
swing_highs, swing_lows = find_swing_points(data, window)\
\
trades = []\
\
for sh in swing_highs:\
for sl in swing_lows:\
if sh['price'] > sl['price'] and sh['index'] > sl['index']:\
fib = calculate_fibonacci_levels(sh['price'], sl['price'])\
recent_idx = max(sh['index'], sl['index'])\
\
if recent_idx + 1 < len(data):\
recent_data = data.iloc[recent_idx+1:]\
if len(recent_data) == 0:\
continue\
\
# BOS: price breaks above swing high\
if recent_data['Close'].max() > sh['price']:\
entry = fib['61.8%']\
stop = fib['0%']\
target = fib['161.8%']\
risk = abs(entry - stop)\
if risk > 0:\
# Simulate trade outcome\
for j in range(recent_idx+1, len(data)):\
bar = data.iloc[j]\
if bar['Low'] <= stop:\
exit_price = stop\
r = (exit_price - entry) / risk\
trades.append({\
'Entry Time': data.index[recent_idx+1],\
'Exit Time': data.index[j],\
'Direction': 'LONG',\
'Entry': entry,\
'Stop': stop,\
'Target': target,\
'Exit': exit_price,\
'R': r,\
'Equity': 10000 + len(trades) * 100 * r\
})\
break\
if bar['High'] >= target:\
exit_price = target\
r = (exit_price - entry) / risk\
trades.append({\
'Entry Time': data.index[recent_idx+1],\
'Exit Time': data.index[j],\
'Direction': 'LONG',\
'Entry': entry,\
'Stop': stop,\
'Target': target,\
'Exit': exit_price,\
'R': r,\
'Equity': 10000 + len(trades) * 100 * r\
})\
break\
\
# CHoCH: price breaks below swing low then recovers above 38.2%\
elif recent_data['Close'].min() < sl['price'] and recent_data['Close'].iloc[-1] > fib['38.2%']:\
entry = fib['78.6%']\
stop = fib['0%']\
target = fib['161.8%']\
risk = abs(entry - stop)\
if risk > 0:\
# Simulate trade outcome\
for j in range(recent_idx+1, len(data)):\
bar = data.iloc[j]\
if bar['Low'] <= stop:\
exit_price = stop\
r = (exit_price - entry) / risk\
trades.append({\
'Entry Time': data.index[recent_idx+1],\
'Exit Time': data.index[j],\
'Direction': 'LONG',\
'Entry': entry,\
'Stop': stop,\
'Target': target,\
'Exit': exit_price,\
'R': r,\
'Equity': 10000 + len(trades) * 100 * r\
})\
break\
if bar['High'] >= target:\
exit_price = target\
r = (exit_price - entry) / risk\
trades.append({\
'Entry Time': data.index[recent_idx+1],\
'Exit Time': data.index[j],\
'Direction': 'LONG',\
'Entry': entry,\
'Stop': stop,\
'Target': target,\
'Exit': exit_price,\
'R': r,\
'Equity': 10000 + len(trades) * 100 * r\
})\
break\
\
if trades:\
return pd.DataFrame(trades)\
else:\
return pd.DataFrame(columns=['Entry Time', 'Exit Time', 'Direction', 'Entry', 'Stop', 'Target', 'Exit', 'R', 'Equity'])\
\
\
def compute_signal_fib(data, window=10):\
"""Compute current signal for BOS/CHoCH Fibonacci."""\
swing_highs, swing_lows = find_swing_points(data, window)\
\
for sh in swing_highs:\
for sl in swing_lows:\
if sh['price'] > sl['price'] and sh['index'] > sl['index']:\
fib = calculate_fibonacci_levels(sh['price'], sl['price'])\
recent_idx = max(sh['index'], sl['index'])\
\
if recent_idx + 1 < len(data):\
recent_data = data.iloc[recent_idx+1:]\
\
# Check if recent close broke above swing high (BOS)\
if recent_data['Close'].max() > sh['price']:\
return {\
'signal': 'LONG_BOS',\
'entry': fib['61.8%'],\
'stop': fib['0%'],\
'target': fib['161.8%']\
}\
\
# Check CHoCH pattern\
if recent_data['Close'].min() < sl['price'] and recent_data['Close'].iloc[-1] > fib['38.2%']:\
return {\
'signal': 'LONG_CHOCH',\
'entry': fib['78.6%'],\
'stop': fib['0%'],\
'target': fib['161.8%']\
}\
\
return {'signal': 'NEUTRAL', 'entry': None, 'stop': None, 'target': None}\
\
\
# -----------------------------------------------------------------------------\
# Strategy C: School Run\
# -----------------------------------------------------------------------------\
\
def run_backtest_school_run(data, rr_target=2.0, buffer=0.0002):\
"""Run School Run backtest (daily opening breakout)."""\
df = data.copy()\
df['Date'] = df.index.date\
\
trades = []\
\
for day, day_data in df.groupby('Date'):\
if len(day_data) < 3:\
continue\
\
signal_bar = day_data.iloc[1] # 2nd candle\
buy_price = signal_bar['High'] + buffer * signal_bar['Close']\
sell_price = signal_bar['Low'] - buffer * signal_bar['Close']\
\
entered = False\
direction = None\
entry_price = None\
stop = None\
target = None\
entry_time = None\
\
for idx, row in day_data.iloc[2:].iterrows():\
if not entered:\
if row['High'] >= buy_price:\
entered = True\
direction = 'LONG'\
entry_price = buy_price\
stop = signal_bar['Low']\
risk = entry_price - stop\
target = entry_price + risk * rr_target\
entry_time = idx\
elif row['Low'] <= sell_price:\
entered = True\
direction = 'SHORT'\
entry_price = sell_price\
stop = signal_bar['High']\
risk = stop - entry_price\
target = entry_price - risk * rr_target\
entry_time = idx\
else:\
# Exit logic\
if direction == 'LONG':\
if row['Low'] <= stop:\
exit_price = stop\
r = (exit_price - entry_price) / risk\
trades.append({\
'Entry Time': entry_time,\
'Exit Time': idx,\
'Direction': 'LONG',\
'Entry': entry_price,\
'Stop': stop,\
'Target': target,\
'Exit': exit_price,\
'R': r,\
'Equity': 10000 + len(trades) * 100 * r\
})\
break\
if row['High'] >= target:\
exit_price = target\
r = (exit_price - entry_price) / risk\
trades.append({\
'Entry Time': entry_time,\
'Exit Time': idx,\
'Direction': 'LONG',\
'Entry': entry_price,\
'Stop': stop,\
'Target': target,\
'Exit': exit_price,\
'R': r,\
'Equity': 10000 + len(trades) * 100 * r\
})\
break\
elif direction == 'SHORT':\
if row['High'] >= stop:\
exit_price = stop\
r = (entry_price - exit_price) / risk\
trades.append({\
'Entry Time': entry_time,\
'Exit Time': idx,\
'Direction': 'SHORT',\
'Entry': entry_price,\
'Stop': stop,\
'Target': target,\
'Exit': exit_price,\
'R': r,\
'Equity': 10000 + len(trades) * 100 * r\
})\
break\
if row['Low'] <= target:\
exit_price = target\
r = (entry_price - exit_price) / risk\
trades.append({\
'Entry Time': entry_time,\
'Exit Time': idx,\
'Direction': 'SHORT',\
'Entry': entry_price,\
'Stop': stop,\
'Target': target,\
'Exit': exit_price,\
'R': r,\
'Equity': 10000 + len(trades) * 100 * r\
})\
break\
\
if trades:\
return pd.DataFrame(trades)\
else:\
return pd.DataFrame(columns=['Entry Time', 'Exit Time', 'Direction', 'Entry', 'Stop', 'Target', 'Exit', 'R', 'Equity'])\
\
\
def compute_signal_school_run(data, buffer=0.0002):\
"""Compute current signal for School Run."""\
df = data.copy()\
df['Date'] = df.index.date\
\
today = df['Date'].iloc[-1]\
today_data = df[df['Date'] == today]\
\
if len(today_data) < 3:\
return {'signal': 'NO_TRADE', 'entry': None, 'stop': None, 'target': None}\
\
signal_bar = today_data.iloc[1]\
buy_price = signal_bar['High'] + buffer * signal_bar['Close']\
sell_price = signal_bar['Low'] - buffer * signal_bar['Close']\
\
last = today_data.iloc[-1]\
\
if last['High'] >= buy_price:\
return {\
'signal': 'LONG_SETUP',\
'entry': buy_price,\
'stop': signal_bar['Low'],\
'target': buy_price + (buy_price - signal_bar['Low']) * 2.0\
}\
elif last['Low'] <= sell_price:\
return {\
'signal': 'SHORT_SETUP',\
'entry': sell_price,\
'stop': signal_bar['High'],\
'target': sell_price - (signal_bar['High'] - sell_price) * 2.0\
}\
else:\
return {'signal': 'WAITING', 'entry': None, 'stop': None, 'target': None}\
\
\
# -----------------------------------------------------------------------------\
# Data Fetching (cached)\
# -----------------------------------------------------------------------------\
\
@st.cache_data(ttl=3600)\
def cached_fetch(instrument, interval, source):\
"""Fetch data with caching."""\
try:\
data = fetch_ohlcv(instrument, interval)\
return data\
except Exception as e:\
if source == 'Synthetic':\
data = generate_synthetic_data(instrument, interval)\
return data\
st.error(f"Error fetching data: {e}")\
return None\
\
\
# -----------------------------------------------------------------------------\
# Main App\
# -----------------------------------------------------------------------------\
\
def main():\
st.set_page_config(page_title="Trading Dashboard", layout="wide")\
\
st.title("📈 Multi-Strategy Trading Dashboard")\
st.markdown("Analyze multiple trading strategies on live or synthetic data.")\
\
# Sidebar config\
st.sidebar.header("Strategy Settings")\
\
strategy_choice = st.sidebar.selectbox(\
"Select Strategy",\
['SMA-Slope Pullback', 'BOS/CHoCH Fibonacci', 'School Run']\
)\
\
instrument = st.sidebar.selectbox("Instrument", SUPPORTED_INSTRUMENTS)\
interval = st.sidebar.selectbox("Interval", INTERVAL_PERIODS)\
\
scenario = 'Base Case'\
if strategy_choice == 'SMA-Slope Pullback':\
scenario = st.sidebar.selectbox("Scenario", ['Base Case', 'Loose Filter'])\
\
sma_period = st.sidebar.number_input("SMA Period", min_value=5, max_value=200, value=20)\
rr_target = st.sidebar.number_input("RR Target", min_value=1.0, max_value=10.0, value=2.0, step=0.1)\
initial_capital = st.sidebar.number_input("Initial Capital", min_value=1000, max_value=1000000, value=10000, step=1000)\
risk_pct = st.sidebar.number_input("Risk %", min_value=0.5, max_value=10.0, value=1.0, step=0.5)\
data_source = st.sidebar.selectbox("Data Source", ['Live', 'Synthetic'])\
\
run_button = st.sidebar.button("▶ Run Analysis")\
\
if run_button:\
# Fetch data\
data = cached_fetch(instrument, interval, data_source)\
if data is None:\
st.error("Failed to load data. Please check your settings.")\
return\
\
# Display data info\
st.caption(f"Data Range: {data.index.min()} to {data.index.max()} | {len(data)} bars")\
\
# Run selected strategy\
if strategy_choice == 'SMA-Slope Pullback':\
params = PARAMS_SMA_SLOPE[scenario].copy()\
params['SMA_PERIOD'] = sma_period\
params['RR_TARGET'] = rr_target\
trades = run_backtest_sma_slope(data, params)\
signal = compute_signal_sma_slope(data, params)\
elif strategy_choice == 'BOS/CHoCH Fibonacci':\
trades = run_backtest_fib(data, window=10)\
signal_info = compute_signal_fib(data, window=10)\
signal = signal_info['signal']\
elif strategy_choice == 'School Run':\
trades = run_backtest_school_run(data, rr_target=rr_target)\
signal_info = compute_signal_school_run(data)\
signal = signal_info['signal']\
else:\
trades = None\
signal = 'UNKNOWN'\
\
metrics = compute_metrics(trades)\
formatted_metrics = format_metrics_for_display(metrics)\
\
# Top metric cards\
col1, col2, col3, col4 = st.columns(4)\
\
with col1:\
st.metric("Current Signal", signal)\
with col2:\
st.metric("Current Price", f"{data['Close'].iloc[-1]:.2f}")\
with col3:\
st.metric("Total Trades", formatted_metrics['total_trades'])\
with col4:\
st.metric("Total R", formatted_metrics['total_r'])\
\
# Current setup details\
st.subheader("Current Setup")\
if signal not in ['NEUTRAL', 'NO_TRADE', 'WAITING']:\
if strategy_choice == 'SMA-Slope Pullback':\
st.info(f"State: {signal}")\
else:\
entry = signal_info['entry'] if 'signal_info' in locals() else None\
stop = signal_info['stop'] if 'signal_info' in locals() else None\
target = signal_info['target'] if 'signal_info' in locals() else None\
if entry and stop:\
col_entry, col_stop, col_target = st.columns(3)\
col_entry.metric("Entry", f"{entry:.2f}")\
col_stop.metric("Stop", f"{stop:.2f}")\
col_target.metric("Target", f"{target:.2f}")\
\
# Position sizing\
size = position_sizing(initial_capital, risk_pct, entry, stop)\
st.write(f"**Position Size:** {size:.2f} units (risk {risk_pct}% of capital)")\
else:\
st.info("No active setup. Waiting for signal.")\
\
# Tabs section\
tab_perf, tab_trades, tab_summary = st.tabs(["Performance", "Trade Log", "Summary"])\
\
with tab_perf:\
if len(trades) > 0:\
# Equity curve\
equity_curve = pd.DataFrame({\
'Trade': range(1, len(trades) + 1),\
'Equity': trades['Equity'].values\
})\
fig = px.line(\
equity_curve,\
x='Trade',\
y='Equity',\
title="Equity Curve",\
markers=True\
)\
fig.update_layout(width='stretch', height=400)\
st.plotly_chart(fig)\
\
# R distribution histogram\
fig_hist = px.histogram(\
trades,\
x='R',\
nbins=20,\
title="R Multiple Distribution"\
)\
fig_hist.update_layout(width='stretch', height=350)\
st.plotly_chart(fig_hist)\
else:\
st.warning("No trades generated. Try different parameters or data.")\
\
with tab_trades:\
if len(trades) > 0:\
st.dataframe(trades)\
csv = trades.to_csv(index=False)\
st.download_button(\
label="Download Trade Log CSV",\
data=csv,\
file_name=f"trade_log_{strategy_choice.replace(' ', '_')}.csv",\
mime="text/csv"\
)\
else:\
st.info("No trades available.")\
\
with tab_summary:\
if len(trades) > 0:\
summary_df = pd.DataFrame([formatted_metrics])\
st.dataframe(summary_df)\
csv = summary_df.to_csv(index=False)\
st.download_button(\
label="Download Summary CSV",\
data=csv,\
file_name="summary.csv",\
mime="text/csv"\
)\
else:\
st.info("No metrics to display.")\
\
else:\
st.info("Configure the strategy settings on the left and click **▶ Run Analysis** to start.")\
\
\
if __name__ == "__main__":\
main()
