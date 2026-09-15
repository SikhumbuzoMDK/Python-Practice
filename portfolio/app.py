import streamlit as st\
import pandas as pd\
import numpy as np\
import plotly.express as px\
import plotly.graph_objects as go\
from datetime import datetime, timedelta\
import yfinance as yf\
\
# ============================================================\
# CONSTANTS\
# ============================================================\
SUPPORTED_INSTRUMENTS = {\
"EURUSD": {"symbol": "EURUSD=X", "default_price": 1.08},\
"GBPUSD": {"symbol": "GBPUSD=X", "default_price": 1.26},\
"USDJPY": {"symbol": "USDJPY=X", "default_price": 150.5},\
"AUDUSD": {"symbol": "AUDUSD=X", "default_price": 0.66},\
"XAUUSD": {"symbol": "GC=F", "default_price": 2300.0},\
}\
\
# ============================================================\
# DATA LOADING\
# ============================================================\
def fetch_ohlcv(symbol, start_date, end_date):\
"""Fetch OHLCV data for a symbol from yfinance.\
\
Args:\
symbol (str): The yfinance ticker symbol.\
start_date (str or datetime.date): Start date for data.\
end_date (str or datetime.date): End date for data.\
\
Returns:\
pd.DataFrame: DataFrame with columns Open, High, Low, Close, Volume.\
"""\
try:\
data = yf.download(symbol, start=start_date, end=end_date, progress=False)\
if data is None or data.empty:\
raise ValueError("No data returned from yfinance")\
# Flatten MultiIndex columns if necessary\
if isinstance(data.columns, pd.MultiIndex):\
data.columns = data.columns.get_level_values(0)\
data = data.rename(columns=str.title)\
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]\
data.index = pd.to_datetime(data.index)\
data = data.dropna()\
return data\
except Exception as e:\
st.warning(f"Failed to fetch data from yfinance for {symbol}: {e}. Using synthetic data.")\
return generate_synthetic_data(symbol, start_date, end_date)\
\
\
def generate_synthetic_data(symbol, start_date, end_date):\
"""Generate synthetic OHLCV data when yfinance is unavailable.\
\
Args:\
symbol (str): The ticker symbol (used for default price).\
start_date (str or datetime.date): Start date.\
end_date (str or datetime.date): End date.\
\
Returns:\
pd.DataFrame: Synthetic OHLCV DataFrame.\
"""\
# Determine default price from constant map, fallback to 100 if not found\
default_price = 100.0\
for instr_name, info in SUPPORTED_INSTRUMENTS.items():\
if info["symbol"] == symbol:\
default_price = info["default_price"]\
break\
\
np.random.seed(42) # reproducible synthetic data\
days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days\
periods = max(days, 1)\
dates = pd.date_range(start=start_date, end=end_date, freq='D')\
if len(dates) < 2:\
dates = pd.date_range(end=end_date, periods=250, freq='D')\
\
# Random walk with drift\
returns = np.random.normal(0.0001, 0.008, len(dates))\
close = default_price * np.exp(np.cumsum(returns))\
\
open_prices = close * (1 + np.random.normal(0, 0.002, len(dates)))\
high = np.maximum(open_prices, close) * (1 + np.abs(np.random.normal(0, 0.003, len(dates))))\
low = np.minimum(open_prices, close) * (1 - np.abs(np.random.normal(0, 0.003, len(dates))))\
volume = np.random.randint(1000, 10000, len(dates))\
\
df = pd.DataFrame({\
"Open": open_prices,\
"High": high,\
"Low": low,\
"Close": close,\
"Volume": volume.astype(float),\
}, index=dates)\
return df\
\
\
# ============================================================\
# STRATEGY FUNCTIONS\
# ============================================================\
def compute_signal(row, sma, prev_close, prev_sma):\
"""Compute trading signal based on pullback-to-SMA state machine.\
\
Args:\
row (pd.Series): Current OHLC row.\
sma (float): Current SMA value.\
prev_close (float): Previous close price.\
prev_sma (float): Previous SMA value.\
\
Returns:\
dict: Signal information.\
"""\
close = row['Close']\
high = row['High']\
low = row['Low']\
sma_change = sma - prev_sma\
\
signal = {"direction": None, "midpoint": None, "swing_ref": None, "triggered": False}\
\
# BULL IMPULSE: price crosses above SMA from below\
if prev_close < prev_sma and close > sma:\
signal["direction"] = "bull"\
signal["swing_ref"] = high # track swing_high\
signal["triggered"] = True\
\
# BEAR IMPULSE: price crosses below SMA from above\
elif prev_close > prev_sma and close < sma:\
signal["direction"] = "bear"\
signal["swing_ref"] = low # track swing_low\
signal["triggered"] = True\
\
return signal\
\
\
def position_sizing(equity, risk_pct, stop_distance, default_price):\
"""Calculate position size based on risk percentage.\
\
Args:\
equity (float): Current account equity.\
risk_pct (float): Risk percentage (e.g., 0.01 for 1%).\
stop_distance (float): Distance from entry to stop loss.\
default_price (float): Current instrument price reference.\
\
Returns:\
dict: Position sizing details (units, risk_amount, notional).\
"""\
risk_amount = equity * risk_pct\
if stop_distance <= 0:\
return {"units": 0, "risk_amount": 0, "notional": 0}\
units = risk_amount / stop_distance\
notional = units * default_price\
return {\
"units": units,\
"risk_amount": risk_amount,\
"notional": notional\
}\
\
\
def run_backtest(data, sma_period=50, rr_ratio=5.0, risk_pct=0.01, initial_capital=100000):\
"""Run the pullback-to-SMA trading strategy backtest.\
\
Args:\
data (pd.DataFrame): OHLCV data.\
sma_period (int): SMA period for trend filter.\
rr_ratio (float): Risk-reward ratio (target = RR * risk).\
risk_pct (float): Risk per trade as fraction of equity.\
initial_capital (float): Starting equity.\
\
Returns:\
tuple: (trades_df, equity_curve, metrics, final_signal)\
"""\
df = data.copy()\
df['SMA'] = df['Close'].rolling(sma_period).mean()\
df = df.dropna()\
\
if len(df) < 5:\
return pd.DataFrame(), pd.DataFrame(), {}, None\
\
trades = []\
position = None # dict with entry, stop, target, direction, entry_idx, swing_ref, pull_low/pull_high\
equity = initial_capital\
equity_curve = []\
\
# State machine variables\
pending_signal = None\
\
for i in range(1, len(df)):\
row = df.iloc[i]\
prev_row = df.iloc[i - 1]\
sma = row['SMA']\
prev_sma = prev_row['SMA']\
close = row['Close']\
high = row['High']\
low = row['Low']\
\
# If in position, manage the trade\
if position is not None:\
# Move stop to breakeven after +1R\
risk = abs(position['entry'] - position['stop'])\
if position['direction'] == 'bull':\
if close >= position['entry'] + risk:\
position['stop'] = position['entry'] # breakeven\
# Check stop loss\
if low <= position['stop']:\
exit_price = position['stop']\
pnl = exit_price - position['entry']\
trades.append({\
"entry_time": df.index[position['entry_idx']],\
"exit_time": df.index[i],\
"direction": position['direction'],\
"entry": position['entry'],\
"stop": position['stop'],\
"target": position['target'],\
"exit": exit_price,\
"pnl": pnl,\
"r_multiple": pnl / risk if risk > 0 else 0,\
"exit_reason": "stop_loss",\
"equity_after": equity + pnl,\
})\
equity += pnl\
position = None\
# Check take profit\
elif high >= position['target']:\
exit_price = position['target']\
pnl = exit_price - position['entry']\
trades.append({\
"entry_time": df.index[position['entry_idx']],\
"exit_time": df.index[i],\
"direction": position['direction'],\
"entry": position['entry'],\
"stop": position['stop'],\
"target": position['target'],\
"exit": exit_price,\
"pnl": pnl,\
"r_multiple": pnl / risk if risk > 0 else 0,\
"exit_reason": "take_profit",\
"equity_after": equity + pnl,\
})\
equity += pnl\
position = None\
\
elif position['direction'] == 'bear':\
if close <= position['entry'] - risk:\
position['stop'] = position['entry'] # breakeven\
# Check stop loss\
if high >= position['stop']:\
exit_price = position['stop']\
pnl = position['entry'] - exit_price\
trades.append({\
"entry_time": df.index[position['entry_idx']],\
"exit_time": df.index[i],\
"direction": position['direction'],\
"entry": position['entry'],\
"stop": position['stop'],\
"target": position['target'],\
"exit": exit_price,\
"pnl": pnl,\
"r_multiple": pnl / risk if risk > 0 else 0,\
"exit_reason": "stop_loss",\
"equity_after": equity + pnl,\
})\
equity += pnl\
position = None\
# Check take profit\
elif low <= position['target']:\
exit_price = position['target']\
pnl = position['entry'] - exit_price\
trades.append({\
"entry_time": df.index[position['entry_idx']],\
"exit_time": df.index[i],\
"direction": position['direction'],\
"entry": position['entry'],\
"stop": position['stop'],\
"target": position['target'],\
"exit": exit_price,\
"pnl": pnl,\
"r_multiple": pnl / risk if risk > 0 else 0,\
"exit_reason": "take_profit",\
"equity_after": equity + pnl,\
})\
equity += pnl\
position = None\
\
equity_curve.append({"date": df.index[i], "equity": equity})\
continue\
\
# Detect new impulse signal\
signal = compute_signal(row, sma, prev_row['Close'], prev_sma)\
if signal["triggered"]:\
pending_signal = signal\
\
# If we have a pending signal and price pulls back to SMA, set pullback reference\
if pending_signal is not None:\
if pending_signal["direction"] == "bull":\
# Track swing high after impulse\
if high > pending_signal["swing_ref"]:\
pending_signal["swing_ref"] = high\
# Pullback: price touches or crosses below SMA\
if low <= sma:\
pending_signal["pull_low"] = low\
# Entry: close > midpoint and SMA rising\
midpoint = (pending_signal["swing_ref"] + pending_signal["pull_low"]) / 2\
if close > midpoint and sma_change > 0:\
entry = close\
stop = pending_signal["pull_low"]\
target = entry + rr_ratio * (entry - stop)\
position = {\
"direction": "bull",\
"entry": entry,\
"stop": stop,\
"target": target,\
"entry_idx": i,\
}\
pending_signal = None\
\
elif pending_signal["direction"] == "bear":\
# Track swing low after impulse\
if low < pending_signal["swing_ref"]:\
pending_signal["swing_ref"] = low\
# Pullback: price touches or crosses above SMA\
if high >= sma:\
pending_signal["pull_high"] = high\
# Entry: close < midpoint and SMA falling\
midpoint = (pending_signal["pull_high"] + pending_signal["swing_ref"]) / 2\
if close < midpoint and sma_change < 0:\
entry = close\
stop = pending_signal["pull_high"]\
target = entry - rr_ratio * (stop - entry)\
position = {\
"direction": "bear",\
"entry": entry,\
"stop": stop,\
"target": target,\
"entry_idx": i,\
}\
pending_signal = None\
\
equity_curve.append({"date": df.index[i], "equity": equity})\
\
trades_df = pd.DataFrame(trades)\
equity_df = pd.DataFrame(equity_curve)\
\
metrics = compute_metrics(trades_df, equity_df, initial_capital)\
\
final_signal = None\
if position is not None:\
final_signal = position\
elif pending_signal is not None:\
final_signal = pending_signal\
\
return trades_df, equity_df, metrics, final_signal\
\
\
def compute_metrics(trades_df, equity_df, initial_capital):\
"""Compute performance metrics from backtest results.\
\
Args:\
trades_df (pd.DataFrame): DataFrame of trades.\
equity_df (pd.DataFrame): Equity curve DataFrame.\
initial_capital (float): Starting capital.\
\
Returns:\
dict: Dictionary of metrics.\
"""\
if trades_df is None or trades_df.empty:\
return {\
"total_trades": 0,\
"win_rate": 0,\
"profit_factor": 0,\
"avg_win": 0,\
"avg_loss": 0,\
"max_drawdown": 0,\
"final_equity": initial_capital,\
"total_return_pct": 0,\
"sharpe_ratio": 0,\
}\
\
total_trades = len(trades_df)\
wins = trades_df[trades_df['pnl'] > 0]\
losses = trades_df[trades_df['pnl'] <= 0]\
win_rate = len(wins) / total_trades * 100\
\
total_win = wins['pnl'].sum() if not wins.empty else 0\
total_loss = abs(losses['pnl'].sum()) if not losses.empty else 0\
profit_factor = total_win / total_loss if total_loss > 0 else float('inf')\
\
avg_win = wins['pnl'].mean() if not wins.empty else 0\
avg_loss = losses['pnl'].mean() if not losses.empty else 0\
\
if equity_df is not None and not equity_df.empty:\
equity = equity_df['equity'].values\
peak = np.maximum.accumulate(equity)\
drawdown = (peak - equity) / peak\
max_drawdown = drawdown.max()\
final_equity = equity[-1]\
total_return_pct = (final_equity - initial_capital) / initial_capital * 100\
\
# Sharpe ratio (simplified)\
daily_returns = np.diff(equity) / equity[:-1]\
sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0\
else:\
final_equity = initial_capital\
total_return_pct = 0\
max_drawdown = 0\
sharpe = 0\
\
return {\
"total_trades": total_trades,\
"win_rate": win_rate,\
"profit_factor": profit_factor,\
"avg_win": avg_win,\
"avg_loss": avg_loss,\
"max_drawdown": max_drawdown,\
"final_equity": final_equity,\
"total_return_pct": total_return_pct,\
"sharpe_ratio": sharpe,\
}\
\
\
# ============================================================\
# STREAMLIT APP\
# ============================================================\
st.set_page_config(\
page_title="Pullback-to-SMA Trading Strategy Dashboard",\
layout="wide",\
)\
\
st.title("📈 Pullback-to-SMA Trading Strategy Dashboard")\
st.markdown("---")\
st.markdown("""\
This dashboard implements a **pullback-to-SMA trading strategy** for trading 5 instruments including gold (XAUUSD).\
The strategy identifies strong impulsive moves (BULL or BEAR), waits for a pullback to the Simple Moving Average,\
and enters in the direction of the original impulse when price resumes the trend.\
\
**Instruments:** EURUSD, GBPUSD, USDJPY, AUDUSD, XAUUSD\
""")\
\
# ============================================================\
# SIDEBAR\
# ============================================================\
with st.sidebar:\
st.header("⚙️ Configuration")\
\
# Instrument selection\
selected_instrument = st.selectbox(\
"Select Instrument",\
list(SUPPORTED_INSTRUMENTS.keys()),\
)\
instr_info = SUPPORTED_INSTRUMENTS[selected_instrument]\
symbol = instr_info["symbol"]\
default_price = instr_info["default_price"]\
\
# SMA period\
sma_period = st.slider("SMA Period", min_value=10, max_value=200, value=50, step=5)\
\
# RR ratio\
rr_ratio = st.number_input("Risk-Reward Ratio", min_value=1.0, max_value=20.0, value=5.0, step=0.5)\
\
# Initial capital\
initial_capital = st.number_input("Initial Capital (R)", min_value=10000, value=100000, step=10000)\
\
# Risk percentage\
risk_pct = st.slider("Risk per Trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1) / 100\
\
# Data source\
data_source = st.radio("Data Source", ["yfinance", "synthetic"])\
\
# Date range\
today = datetime.now()\
default_start = today - timedelta(days=365)\
start_date = st.date_input("Start Date", value=default_start)\
end_date = st.date_input("End Date", value=today)\
\
run_button = st.button("🚀 Run Backtest", type="primary")\
\
# ============================================================\
# CACHED DATA LOADING\
# ============================================================\
@st.cache_data(ttl=3600)\
def load_data_cached(symbol, start_date, end_date, data_source):\
"""Wrapper to cache data loading."""\
if data_source == "yfinance":\
return fetch_ohlcv(symbol, start_date, end_date)\
else:\
return generate_synthetic_data(symbol, start_date, end_date)\
\
# ============================================================\
# MAIN APP LOGIC\
# ============================================================\
if run_button:\
with st.spinner(f"Loading data for {selected_instrument}..."):\
data = load_data_cached(symbol, start_date, end_date, data_source)\
\
if data is None or data.empty:\
st.error("No data available for the selected parameters.")\
else:\
st.success(f"Loaded {len(data)} data points for {selected_instrument} ({symbol})")\
\
# Run backtest\
trades_df, equity_df, metrics, final_signal = run_backtest(\
data,\
sma_period=sma_period,\
rr_ratio=rr_ratio,\
risk_pct=risk_pct,\
initial_capital=initial_capital,\
)\
\
# ============================================================\
# CURRENT SIGNAL DISPLAY\
# ============================================================\
st.header("🎯 Current Signal")\
\
col1, col2, col3 = st.columns(3)\
\
if final_signal is not None:\
with col1:\
st.metric("Signal Direction", str(final_signal.get("direction", "N/A")).upper())\
with col2:\
st.metric("Signal Type", "Active Position" if "entry" in final_signal else "Pending Setup")\
with col3:\
if "entry" in final_signal:\
st.metric("Entry Price", f"{final_signal['entry']:.4f}")\
else:\
st.metric("Ref Price", f"{final_signal.get('swing_ref', 0):.4f}")\
\
st.info(\
f"**Current Setup:** {final_signal.get('direction', '').upper()} setup detected. "\
f"Reference level: {final_signal.get('swing_ref', 'N/A'):.4f}" if isinstance(final_signal.get('swing_ref'), (int, float)) else f"**Current Setup:** {final_signal.get('direction', '').upper()} setup detected."\
)\
else:\
st.info("No active signal at the moment.")\
\
# ============================================================\
# TABS\
# ============================================================\
tab1, tab2, tab3, tab4 = st.tabs(["Performance Dashboard", "Trade Log", "Summary", "Position Sizing"])\
\
# Tab 1: Performance Dashboard\
with tab1:\
st.subheader("📊 Performance Dashboard")\
\
if not equity_df.empty:\
# Equity curve\
st.markdown("**Equity Curve**")\
fig_equity = px.line(equity_df, x="date", y="equity", title=f"Equity Curve for {selected_instrument}")\
fig_equity.update_layout(\
xaxis_title="Date",\
yaxis_title="Equity (R)",\
height=400,\
)\
st.plotly_chart(fig_equity, use_container_width=True)\
\
# R distribution histogram\
if not trades_df.empty and 'r_multiple' in trades_df.columns:\
st.markdown("**R-Multiple Distribution**")\
fig_hist = px.histogram(\
trades_df,\
x="r_multiple",\
nbins=20,\
title="Distribution of Trade R-Multiples",\
color_discrete_sequence=['#2196F3'],\
)\
fig_hist.update_layout(\
xaxis_title="R-Multiple",\
yaxis_title="Frequency",\
height=300,\
)\
st.plotly_chart(fig_hist, use_container_width=True)\
else:\
st.info("No trades executed yet.")\
else:\
st.warning("No equity data available.")\
\
# Tab 2: Trade Log\
with tab2:\
st.subheader("📋 Trade Log")\
\
if not trades_df.empty:\
st.dataframe(trades_df, use_container_width=True)\
\
# Download button for trade log CSV\
csv_trades = trades_df.to_csv(index=False).encode('utf-8')\
st.download_button(\
label="📥 Download Trade Log (CSV)",\
data=csv_trades,\
file_name=f"trade_log_{selected_instrument}.csv",\
mime="text/csv",\
)\
else:\
st.info("No trades executed yet.")\
\
# Tab 3: Summary\
with tab3:\
st.subheader("📑 Summary Metrics")\
\
if metrics:\
metrics_df = pd.DataFrame({\
"Metric": [\
"Total Trades",\
"Win Rate (%)",\
"Profit Factor",\
"Avg Win",\
"Avg Loss",\
"Max Drawdown",\
"Final Equity",\
"Total Return (%)",\
"Sharpe Ratio",\
],\
"Value": [\
metrics["total_trades"],\
f"{metrics['win_rate']:.2f}",\
f"{metrics['profit_factor']:.2f}",\
f"{metrics['avg_win']:.2f}",\
f"{metrics['avg_loss']:.2f}",\
f"{metrics['max_drawdown']*100:.2f}%",\
f"R {metrics['final_equity']:,.2f}",\
f"{metrics['total_return_pct']:.2f}%",\
f"{metrics['sharpe_ratio']:.2f}",\
],\
})\
\
st.table(metrics_df)\
\
# Download summary CSV\
csv_summary = metrics_df.to_csv(index=False).encode('utf-8')\
st.download_button(\
label="📥 Download Summary (CSV)",\
data=csv_summary,\
file_name=f"summary_{selected_instrument}.csv",\
mime="text/csv",\
)\
else:\
st.warning("No metrics available.")\
\
# Tab 4: Position Sizing\
with tab4:\
st.subheader("💠 Position Sizing Calculator")\
\
if not trades_df.empty:\
# Calculate position sizing based on latest trade setup stats\
st.markdown("""\
**Position Sizing Based on Risk Model**\
\
The position size is calculated as:\
- Risk Amount = Equity × Risk %\
- Units = Risk Amount / Stop Distance\
- Notional = Units × Current Price\
""")\
\
# Use the most recent entry if available, else use default price\
latest_close = data['Close'].iloc[-1]\
\
# Show current position sizing for the last trade entry if exists\
if not trades_df.empty:\
last_trade = trades_df.iloc[-1]\
entry_price = last_trade['entry']\
stop_price = last_trade['stop']\
stop_distance = abs(entry_price - stop_price)\
\
sizing = position_sizing(\
metrics.get("final_equity", initial_capital),\
risk_pct,\
stop_distance,\
latest_close,\
)\
\
cols = st.columns(4)\
with cols[0]:\
st.metric("Risk Amount", f"R {sizing['risk_amount']:,.2f}")\
with cols[1]:\
st.metric("Units to Trade", f"{sizing['units']:.2f}")\
with cols[2]:\
st.metric("Notional Value", f"R {sizing['notional']:,.2f}")\
with cols[3]:\
st.metric("Stop Distance", f"{stop_distance:.4f}")\
\
st.markdown("---")\
st.markdown("**Example Position Sizing for This Trade:**")\
st.code(f"""\
Equity: R {metrics.get('final_equity', initial_capital):,.2f}\
Risk %: {risk_pct*100:.2f}%\
Risk Amount: R {sizing['risk_amount']:,.2f}\
Entry: {entry_price:.4f}\
Stop: {stop_price:.4f}\
Stop Distance: {stop_distance:.4f}\
Units: {sizing['units']:.2f}\
Notional: R {sizing['notional']:,.2f}\
""")\
\
# Download equity curve CSV\
if not equity_df.empty:\
csv_equity = equity_df.to_csv(index=False).encode('utf-8')\
st.download_button(\
label="📥 Download Equity Curve (CSV)",\
data=csv_equity,\
file_name=f"equity_curve_{selected_instrument}.csv",\
mime="text/csv",\
)\
\
else:\
st.info("Run backtest to see position sizing details.")\
\
else:\
# Show initial info\
st.info("👈 Configure the strategy and click **Run Backtest** to start.")\
\
""\
}
