"""\
Pullback-to-SMA State Machine Strategy Engine\
\
This module implements a standalone Python strategy engine for a\
pullback-to-SMA state machine algorithm. It includes functions for\
running backtests, computing live signals, analyzing trade metrics,\
and calculating position sizes.\
\
Author: AI Assistant\
Date: 2024\
"""\
\
import numpy as np\
import pandas as pd\
\
\
# -----------------------------------------------------------------------------\
# State Machine Definitions\
# -----------------------------------------------------------------------------\
\
LOOKING = 'LOOKING'\
BULL = 'BULL'\
PULLBACK = 'PULLBACK'\
BEAR = 'BEAR'\
SHORT_PULLBACK = 'SHORT_PULLBACK'\
\
LONG = 'LONG'\
SHORT = 'SHORT'\
WAIT = 'WAIT'\
\
\
# -----------------------------------------------------------------------------\
# Core Backtest Engine\
# -----------------------------------------------------------------------------\
\
def run_backtest(df, sma_period=50, rr=5):\
"""\
Run the pullback-to-SMA state machine backtest.\
\
Parameters\
----------\
df : pandas.DataFrame\
DataFrame with columns [Time, Open, High, Low, Close].\
sma_period : int, default 50\
Period for the Simple Moving Average.\
rr : float, default 5\
Risk-to-Reward ratio for trade targets.\
\
Returns\
-------\
pandas.DataFrame\
Trades DataFrame with columns:\
Entry Time, Exit Time, Direction, Entry, Stop, Target, Exit, R, Equity\
"""\
# Validate input\
required_cols = ['Time', 'Open', 'High', 'Low', 'Close']\
for col in required_cols:\
if col not in df.columns:\
raise ValueError(f"Missing required column: {col}")\
if len(df) < sma_period + 5:\
raise ValueError(f"Not enough data. Need at least {sma_period + 5} rows, got {len(df)}")\
\
# Create a working copy\
data = df.copy().reset_index(drop=True)\
\
# Calculate SMA\
data['SMA'] = data['Close'].rolling(window=sma_period).mean()\
\
# Initialize state machine variables\
state = LOOKING\
swing_high = np.nan\
swing_low = np.nan\
pull_low = np.nan\
pull_high = np.nan\
\
# Trade management variables\
in_trade = False\
trade_direction = None\
entry_price = np.nan\
stop_price = np.nan\
target_price = np.nan\
entry_time = None\
entry_equity = None\
risk_per_unit = np.nan\
\
# Store for equity curve\
trades = []\
\
# Starting equity\
equity = 10000.0 # Base equity assumption; can be adjusted\
\
# Iterate through bars (start after SMA is valid)\
for i in range(sma_period, len(data)):\
row = data.iloc[i]\
prev_row = data.iloc[i - 1]\
\
# Get current values\
o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']\
sma = row['SMA']\
prev_close = prev_row['Close']\
prev_sma = prev_row['SMA']\
\
# -----------------------------------------------------------------\
# State Machine Logic\
# -----------------------------------------------------------------\
\
# ---- BULL IMPULSE ----\
if state == LOOKING:\
# Check for bull impulse: prev.Close < prev.SMA and close > sma\
if prev_close < prev_sma and c > sma:\
state = BULL\
swing_low = prev_row['Low']\
swing_high = h\
\
# Check for bear impulse: prev.Close > prev.SMA and close < sma\
elif prev_close > prev_sma and c < sma:\
state = BEAR\
swing_high = prev_row['High']\
swing_low = l\
\
# ---- BULL state management ----\
elif state == BULL:\
# Update swing high if new high\
if h > swing_high:\
swing_high = h\
\
# Check for pullback condition\
if l <= sma:\
state = PULLBACK\
pull_low = l\
\
# ---- PULLBACK (long) state management ----\
elif state == PULLBACK:\
# Update pullback low\
if l < pull_low:\
pull_low = l\
\
# Check if pullback invalidated (price falls below swing low)\
if l < swing_low:\
state = LOOKING\
continue\
\
# Check for long entry signal\
midpoint = (swing_high + pull_low) / 2.0\
sma_change = sma - prev_sma # Check if SMA is rising\
\
if c > midpoint and sma_change > 0:\
# Calculate trade parameters\
risk = midpoint - pull_low\
entry = c\
stop = pull_low\
target = entry + rr * (entry - stop)\
\
# Enter trade\
in_trade = True\
trade_direction = 'LONG'\
entry_price = entry\
stop_price = stop\
target_price = target\
entry_time = row['Time']\
risk_per_unit = entry_price - stop_price\
entry_equity = equity\
\
# Reset state after entry\
state = LOOKING\
\
# ---- BEAR state management ----\
elif state == BEAR:\
# Update swing low if new low\
if l < swing_low:\
swing_low = l\
\
# Check for pullback condition\
if h >= sma:\
state = SHORT_PULLBACK\
pull_high = h\
\
# ---- SHORT_PULLBACK (short) state management ----\
elif state == SHORT_PULLBACK:\
# Update pullback high\
if h > pull_high:\
pull_high = h\
\
# Check if pullback invalidated (price rises above swing high)\
if h > swing_high:\
state = LOOKING\
continue\
\
# Check for short entry signal\
midpoint = (swing_low + pull_high) / 2.0\
sma_change = sma - prev_sma # Check if SMA is falling\
\
if c < midpoint and sma_change < 0:\
# Calculate trade parameters\
risk = pull_high - midpoint\
entry = c\
stop = pull_high\
target = entry - rr * (stop - entry)\
\
# Enter trade\
in_trade = True\
trade_direction = 'SHORT'\
entry_price = entry\
stop_price = stop\
target_price = target\
entry_time = row['Time']\
risk_per_unit = stop_price - entry_price\
entry_equity = equity\
\
# Reset state after entry\
state = LOOKING\
\
# -----------------------------------------------------------------\
# Position Management\
# -----------------------------------------------------------------\
\
if in_trade:\
# Check for stop loss or take profit\
exit_price = np.nan\
exit_reason = None\
\
if trade_direction == 'LONG':\
if l <= stop_price:\
exit_price = stop_price\
exit_reason = 'STOP'\
elif h >= target_price:\
exit_price = target_price\
exit_reason = 'TARGET'\
else:\
# Move stop to breakeven after +1R\
if h >= entry_price + (entry_price - stop_price):\
stop_price = entry_price\
elif trade_direction == 'SHORT':\
if h >= stop_price:\
exit_price = stop_price\
exit_reason = 'STOP'\
elif l <= target_price:\
exit_price = target_price\
exit_reason = 'TARGET'\
else:\
# Move stop to breakeven after +1R\
if l <= entry_price - (stop_price - entry_price):\
stop_price = entry_price\
\
# If exit triggered, record trade\
if exit_reason is not None:\
# Calculate R multiple\
if trade_direction == 'LONG':\
r_multiple = (exit_price - entry_price) / (entry_price - stop_price_target_initial(trade_direction, entry_price, stop_price))\
else:\
r_multiple = (entry_price - exit_price) / (stop_price_target_initial(trade_direction, entry_price, stop_price))\
\
# Use initial risk (before breakeven move) for R calculation\
# Note: We need to track initial risk separately\
# For simplicity, we recompute based on original entry/stop\
# But stop may have moved... Let's fix this:\
\
# Actually, let's compute R based on the original stop\
# We'll store original stop\
# (Simplification: R is based on initial risk)\
# We'll do this properly below\
\
in_trade = False\
# We'll handle R calculation in a separate pass below\
# For now, store raw values\
\
# Record trade\
trades.append({\
'Entry Time': entry_time,\
'Exit Time': row['Time'],\
'Direction': trade_direction,\
'Entry': entry_price,\
'Stop': stop_price,\
'Target': target_price,\
'Exit': exit_price,\
'R': np.nan, # Will compute below\
'Equity': equity\
})\
\
# Second pass: Compute R multiples and equity properly\
# This is needed because we moved stops but want R based on initial risk\
# Restart from scratch with proper tracking\
\
trades = []\
state = LOOKING\
in_trade = False\
trade_direction = None\
entry_price = np.nan\
stop_price = np.nan\
target_price = np.nan\
entry_time = None\
initial_stop = np.nan\
equity = 10000.0\
\
for i in range(sma_period, len(data)):\
row = data.iloc[i]\
prev_row = data.iloc[i - 1]\
\
o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']\
sma = row['SMA']\
prev_close = prev_row['Close']\
prev_sma = prev_row['SMA']\
\
# STATE MACHINE (same as above)\
if state == LOOKING:\
if prev_close < prev_sma and c > sma:\
state = BULL\
swing_low = prev_row['Low']\
swing_high = h\
elif prev_close > prev_sma and c < sma:\
state = BEAR\
swing_high = prev_row['High']\
swing_low = l\
\
elif state == BULL:\
if h > swing_high:\
swing_high = h\
if l <= sma:\
state = PULLBACK\
pull_low = l\
\
elif state == PULLBACK:\
if l < pull_low:\
pull_low = l\
if l < swing_low:\
state = LOOKING\
continue\
midpoint = (swing_high + pull_low) / 2.0\
sma_change = sma - prev_sma\
if c > midpoint and sma_change > 0:\
risk = midpoint - pull_low\
entry = c\
stop = pull_low\
target = entry + rr * (entry - stop)\
in_trade = True\
trade_direction = 'LONG'\
entry_price = entry\
stop_price = stop\
target_price = target\
entry_time = row['Time']\
initial_stop = stop\
state = LOOKING\
\
elif state == BEAR:\
if l < swing_low:\
swing_low = l\
if h >= sma:\
state = SHORT_PULLBACK\
pull_high = h\
\
elif state == SHORT_PULLBACK:\
if h > pull_high:\
pull_high = h\
if h > swing_high:\
state = LOOKING\
continue\
midpoint = (swing_low + pull_high) / 2.0\
sma_change = sma - prev_sma\
if c < midpoint and sma_change < 0:\
risk = pull_high - midpoint\
entry = c\
stop = pull_high\
target = entry - rr * (stop - entry)\
in_trade = True\
trade_direction = 'SHORT'\
entry_price = entry\
stop_price = stop\
target_price = target\
entry_time = row['Time']\
initial_stop = stop\
state = LOOKING\
\
# POSITION MANAGEMENT\
if in_trade:\
exit_price = np.nan\
exit_reason = None\
\
if trade_direction == 'LONG':\
if l <= stop_price:\
exit_price = stop_price\
exit_reason = 'STOP'\
elif h >= target_price:\
exit_price = target_price\
exit_reason = 'TARGET'\
else:\
if h >= entry_price + (entry_price - initial_stop):\
stop_price = entry_price\
elif trade_direction == 'SHORT':\
if h >= stop_price:\
exit_price = stop_price\
exit_reason = 'STOP'\
elif l <= target_price:\
exit_price = target_price\
exit_reason = 'TARGET'\
else:\
if l <= entry_price - (initial_stop - entry_price):\
stop_price = entry_price\
\
if exit_reason is not None:\
# Compute R multiple\
if trade_direction == 'LONG':\
r_multiple = (exit_price - entry_price) / (entry_price - initial_stop)\
else:\
r_multiple = (entry_price - exit_price) / (initial_stop - entry_price)\
\
# Update equity\
equity += r_multiple * 100 # Assuming $100 risk per trade\
\
trades.append({\
'Entry Time': entry_time,\
'Exit Time': row['Time'],\
'Direction': trade_direction,\
'Entry': entry_price,\
'Stop': initial_stop,\
'Target': target_price,\
'Exit': exit_price,\
'R': r_multiple,\
'Equity': equity\
})\
\
in_trade = False\
\
return pd.DataFrame(trades)\
\
\
def stop_price_target_initial(direction, entry, current_stop):\
"""Helper function to get initial stop (kept for compatibility)."""\
return current_stop\
\
\
# -----------------------------------------------------------------------------\
# Signal Computation\
# -----------------------------------------------------------------------------\
\
def compute_signal(df, sma_period=50, rr=5):\
"""\
Compute the current signal based on the latest data.\
\
Parameters\
----------\
df : pandas.DataFrame\
DataFrame with columns [Time, Open, High, Low, Close].\
sma_period : int, default 50\
Period for the Simple Moving Average.\
rr : float, default 5\
Risk-to-Reward ratio.\
\
Returns\
-------\
dict\
Dictionary with state, signal, swing levels, entry/stop/target, etc.\
"""\
required_cols = ['Time', 'Open', 'High', 'Low', 'Close']\
for col in required_cols:\
if col not in df.columns:\
raise ValueError(f"Missing required column: {col}")\
if len(df) < sma_period + 5:\
raise ValueError(f"Not enough data. Need at least {sma_period + 5} rows, got {len(df)}")\
\
data = df.copy().reset_index(drop=True)\
data['SMA'] = data['Close'].rolling(window=sma_period).mean()\
\
# Run state machine up to last bar WITHOUT executing last bar\
# We'll replicate the logic but stop at last bar\
state = LOOKING\
swing_high = np.nan\
swing_low = np.nan\
pull_low = np.nan\
pull_high = np.nan\
\
for i in range(sma_period, len(data) - 1): # Stop at second-to-last bar\
row = data.iloc[i]\
prev_row = data.iloc[i - 1]\
\
o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']\
sma = row['SMA']\
prev_close = prev_row['Close']\
prev_sma = prev_row['SMA']\
\
if state == LOOKING:\
if prev_close < prev_sma and c > sma:\
state = BULL\
swing_low = prev_row['Low']\
swing_high = h\
elif prev_close > prev_sma and c < sma:\
state = BEAR\
swing_high = prev_row['High']\
swing_low = l\
\
elif state == BULL:\
if h > swing_high:\
swing_high = h\
if l <= sma:\
state = PULLBACK\
pull_low = l\
\
elif state == PULLBACK:\
if l < pull_low:\
pull_low = l\
if l < swing_low:\
state = LOOKING\
continue\
midpoint = (swing_high + pull_low) / 2.0\
sma_change = sma - prev_sma\
if c > midpoint and sma_change > 0:\
risk = midpoint - pull_low\
entry = c\
stop = pull_low\
target = entry + rr * (entry - stop)\
state = LOOKING\
\
elif state == BEAR:\
if l < swing_low:\
swing_low = l\
if h >= sma:\
state = SHORT_PULLBACK\
pull_high = h\
\
elif state == SHORT_PULLBACK:\
if h > pull_high:\
pull_high = h\
if h > swing_high:\
state = LOOKING\
continue\
midpoint = (swing_low + pull_high) / 2.0\
sma_change = sma - prev_sma\
if c < midpoint and sma_change < 0:\
risk = pull_high - midpoint\
entry = c\
stop = pull_high\
target = entry - rr * (stop - entry)\
state = LOOKING\
\
# Now process the LAST bar for signal generation (without executing trade)\
last_idx = len(data) - 1\
row = data.iloc[last_idx]\
prev_row = data.iloc[last_idx - 1]\
\
o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']\
sma = row['SMA']\
prev_close = prev_row['Close']\
prev_sma = prev_row['SMA']\
\
signal = WAIT\
entry = np.nan\
stop = np.nan\
target = np.nan\
\
# Process last bar for signal determination\
if state == LOOKING:\
if prev_close < prev_sma and c > sma:\
state = BULL\
swing_low = prev_row['Low']\
swing_high = h\
elif prev_close > prev_sma and c < sma:\
state = BEAR\
swing_high = prev_row['High']\
swing_low = l\
\
elif state == BULL:\
if h > swing_high:\
swing_high = h\
if l <= sma:\
state = PULLBACK\
pull_low = l\
\
elif state == PULLBACK:\
if l < pull_low:\
pull_low = l\
if l < swing_low:\
state = LOOKING\
else:\
midpoint = (swing_high + pull_low) / 2.0\
sma_change = sma - prev_sma\
if c > midpoint and sma_change > 0:\
risk = midpoint - pull_low\
entry = c\
stop = pull_low\
target = entry + rr * (entry - stop)\
signal = LONG\
\
elif state == BEAR:\
if l < swing_low:\
swing_low = l\
if h >= sma:\
state = SHORT_PULLBACK\
pull_high = h\
\
elif state == SHORT_PULLBACK:\
if h > pull_high:\
pull_high = h\
if h > swing_high:\
state = LOOKING\
else:\
midpoint = (swing_low + pull_high) / 2.0\
sma_change = sma - prev_sma\
if c < midpoint and sma_change < 0:\
risk = pull_high - midpoint\
entry = c\
stop = pull_high\
target = entry - rr * (stop - entry)\
signal = SHORT\
\
return {\
'state': state,\
'signal': signal,\
'swing_high': swing_high if not np.isnan(swing_high) else None,\
'swing_low': swing_low if not np.isnan(swing_low) else None,\
'pull_low': pull_low if not np.isnan(pull_low) else None,\
'pull_high': pull_high if not np.isnan(pull_high) else None,\
'entry': entry if not np.isnan(entry) else None,\
'stop': stop if not np.isnan(stop) else None,\
'target': target if not np.isnan(target) else None,\
'current_price': c,\
'sma_value': sma\
}\
\
\
# -----------------------------------------------------------------------------\
# Metrics Computation\
# -----------------------------------------------------------------------------\
\
def compute_metrics(trades_df):\
"""\
Compute performance metrics from trades DataFrame.\
\
Parameters\
----------\
trades_df : pandas.DataFrame\
Trades DataFrame from run_backtest().\
\
Returns\
-------\
dict\
Dictionary with metrics.\
"""\
if trades_df is None or len(trades_df) == 0:\
return {\
'total_trades': 0,\
'wins': 0,\
'losses': 0,\
'win_rate': 0.0,\
'avg_R': 0.0,\
'total_R': 0.0,\
'profit_factor': 0.0,\
'max_drawdown': 0.0,\
'best_trade': 0.0,\
'worst_trade': 0.0\
}\
\
total_trades = len(trades_df)\
wins = int((trades_df['R'] > 0).sum())\
losses = int((trades_df['R'] < 0).sum())\
win_rate = wins / total_trades if total_trades > 0 else 0.0\
avg_R = trades_df['R'].mean()\
total_R = trades_df['R'].sum()\
\
# Profit factor\
gross_profit = trades_df.loc[trades_df['R'] > 0, 'R'].sum()\
gross_loss = abs(trades_df.loc[trades_df['R'] < 0, 'R'].sum())\
profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')\
\
# Max drawdown from equity curve\
equity_curve = trades_df['Equity'].values\
running_max = np.maximum.accumulate(equity_curve)\
drawdown = running_max - equity_curve\
max_drawdown = drawdown.max() if len(drawdown) > 0 else 0.0\
\
best_trade = trades_df['R'].max() if len(trades_df) > 0 else 0.0\
worst_trade = trades_df['R'].min() if len(trades_df) > 0 else 0.0\
\
return {\
'total_trades': total_trades,\
'wins': wins,\
'losses': losses,\
'win_rate': win_rate,\
'avg_R': avg_R,\
'total_R': total_R,\
'profit_factor': profit_factor,\
'max_drawdown': max_drawdown,\
'best_trade': best_trade,\
'worst_trade': worst_trade\
}\
\
\
# -----------------------------------------------------------------------------\
# Position Sizing\
# -----------------------------------------------------------------------------\
\
def position_sizing(account_size, risk_percent, entry, stop):\
"""\
Calculate position size based on account size and risk.\
\
Parameters\
----------\
account_size : float\
Total account size in currency units.\
risk_percent : float, default 1\
Percentage of account to risk per trade (e.g., 1 for 1%).\
entry : float\
Entry price.\
stop : float\
Stop loss price.\
\
Returns\
-------\
dict\
Dictionary with position size (units), notional value, and risk amount.\
"""\
risk_amount = account_size * (risk_percent / 100.0)\
stop_distance = abs(entry - stop)\
\
if stop_distance == 0:\
raise ValueError("Stop distance cannot be zero.")\
\
# Position size in units\
units = risk_amount / stop_distance\
notional = units * entry\
\
return {\
'units': units,\
'notional': notional,\
'risk_amount': risk_amount\
}\
\
\
# -----------------------------------------------------------------------------\
# Example Usage\
# -----------------------------------------------------------------------------\
\
if __name__ == "__main__":\
# Generate sample data for demonstration\
np.random.seed(42)\
n = 500\
dates = pd.date_range(start='2023-01-01', periods=n, freq='D')\
\
# Simulate price data with trend and noise\
price = 100 + np.cumsum(np.random.randn(n) * 0.5)\
open_prices = price + np.random.randn(n) * 0.1\
high_prices = np.maximum(open_prices, price) + np.abs(np.random.randn(n) * 0.2)\
low_prices = np.minimum(open_prices, price) - np.abs(np.random.randn(n) * 0.2)\
close_prices = price\
\
df = pd.DataFrame({\
'Time': dates,\
'Open': open_prices,\
'High': high_prices,\
'Low': low_prices,\
'Close': close_prices\
})\
\
print("=== Example Usage ===")\
print(f"Data shape: {df.shape}")\
\
# Run backtest\
print("\nRunning backtest...")\
trades = run_backtest(df, sma_period=50, rr=5)\
print(f"Number of trades: {len(trades)}")\
if len(trades) > 0:\
print(trades.head())\
\
# Compute metrics\
print("\n=== Trade Metrics ===")\
metrics = compute_metrics(trades)\
for key, value in metrics.items():\
print(f" {key}: {value:.4f}" if isinstance(value, float) else f" {key}: {value}")\
\
# Compute signal from latest data\
print("\n=== Latest Signal ===")\
signal = compute_signal(df, sma_period=50, rr=5)\
for key, value in signal.items():\
print(f" {key}: {value}")\
\
# Position sizing example\
print("\n=== Position Sizing ===")\
sizing = position_sizing(account_size=10000, risk_percent=1, entry=100, stop=98)\
print(f" Units: {sizing['units']:.2f}")\
print(f" Notional: {sizing['notional']:.2f}")\
print(f" Risk amount: {sizing['risk_amount']:.2f}")
