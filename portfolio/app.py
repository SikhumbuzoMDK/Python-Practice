"""\
Pullback-to-SMA Trading Strategy Dashboard\
\
A complete Streamlit application for backtesting and live signal generation\
for a pullback-to-Simple Moving Average (SMA) trading strategy.\
"""\
\
import streamlit as st\
import pandas as pd\
import numpy as np\
import plotly.graph_objects as go\
from plotly.subplots import make_subplots\
from datetime import datetime, timedelta\
import yfinance as yf\
import io\
import warnings\
warnings.filterwarnings('ignore')\
\
# ============================================================================\
# STRATEGY LOGIC MODULE (importable)\
# ============================================================================\
\
# Mapping of instruments to yfinance symbols\
INSTRUMENT_SYMBOLS = {\
'EURUSD': 'EURUSD=X',\
'GBPUSD': 'GBPUSD=X',\
'USDJPY': 'USDJPY=X',\
'AUDUSD': 'AUDUSD=X',\
'XAUUSD': 'GC=F' # Gold futures as proxy\
}\
\
def generate_synthetic_data(days=500, seed=42):\
"""Generate synthetic OHLC data for demo purposes.\
\
Args:\
days (int): Number of days of data to generate\
seed (int): Random seed for reproducibility\
\
Returns:\
pd.DataFrame: DataFrame with Time, Open, High, Low, Close columns\
"""\
np.random.seed(seed)\
end_date = datetime.now()\
start_date = end_date - timedelta(days=days)\
dates = pd.date_range(start=start_date, end=end_date, freq='D')\
\
# Generate price with trend and volatility\
returns = np.random.normal(0.0002, 0.01, len(dates))\
close = 100 * np.exp(np.cumsum(returns))\
\
# Generate OHLC\
open_price = np.roll(close, 1)\
open_price[0] = close[0] * 0.999\
high = np.maximum(open_price, close) * (1 + np.random.uniform(0, 0.005, len(dates)))\
low = np.minimum(open_price, close) * (1 - np.random.uniform(0, 0.005, len(dates)))\
\
df = pd.DataFrame({\
'Time': dates.strftime('%Y-%m-%d'),\
'Open': open_price.round(4),\
'High': high.round(4),\
'Low': low.round(4),\
'Close': close.round(4)\
})\
return df\
\
def fetch_historical_data(symbol, start_date, end_date):\
"""Fetch historical data from yfinance.\
\
Args:\
symbol (str): yfinance symbol\
start_date (str): Start date in 'YYYY-MM-DD' format\
end_date (str): End date in 'YYYY-MM-DD' format\
\
Returns:\
pd.DataFrame: DataFrame with Time, Open, High, Low, Close columns\
"""\
try:\
ticker = yf.Ticker(symbol)\
data = ticker.history(start=start_date, end=end_date)\
\
if len(data) == 0:\
return None\
\
# Rename to match expected format\
df = data.copy()\
df['Time'] = df.index.strftime('%Y-%m-%d')\
df = df.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close'})\
df = df[['Time', 'Open', 'High', 'Low', 'Close']]\
return df.reset_index(drop=True)\
\
except Exception as e:\
st.warning(f"Error fetching data for {symbol}: {str(e)}")\
return None\
\
def calculate_sma(close, period):\
"""Calculate Simple Moving Average.\
\
Args:\
close (pd.Series): Close prices\
period (int): SMA period\
\
Returns:\
pd.Series: SMA values\
"""\
return close.rolling(window=period).mean()\
\
def run_backtest(df, sma_period=50, risk_reward=5, initial_capital=100000):\
"""Run backtest for pullback-to-SMA strategy.\
\
Args:\
df (pd.DataFrame): OHLC data\
sma_period (int): SMA period\
risk_reward (float): Risk:Reward ratio\
initial_capital (float): Initial account capital\
\
Returns:\
tuple: (trades_df, equity_curve_df, metrics_dict, signals_df)\
"""\
df = df.copy()\
\
# Calculate SMA\
df['SMA'] = calculate_sma(df['Close'], sma_period)\
\
# State machine variables\
trades = []\
signals = []\
current_state = 'WAIT'\
swing_high = None\
swing_low = None\
stop_price = None\
entry_price = None\
target_price = None\
position = None # 'LONG' or 'SHORT'\
\
for i in range(1, len(df)):\
row = df.iloc[i]\
prev_row = df.iloc[i - 1]\
\
# Initialize signal\
signal_dict = {\
'Date': row['Time'],\
'Close': row['Close'],\
'SMA': row['SMA'] if pd.notna(row['SMA']) else None,\
'State': current_state,\
'Signal': 'WAIT'\
}\
\
# Skip if SMA is NaN\
if pd.isna(row['SMA']) or pd.isna(prev_row['SMA']):\
signals.append(signal_dict)\
continue\
\
sma_change = row['SMA'] - prev_row['SMA']\
\
# LONG IMPULSE\
if current_state == 'WAIT' and prev_row['Close'] < prev_row['SMA'] and row['Close'] > row['SMA']:\
current_state = 'LONG_IMPULSE'\
swing_high = row['High']\
signal_dict['Signal'] = 'LONG'\
signal_dict['State'] = 'LONG_IMPULSE'\
\
# SHORT IMPULSE\
elif current_state == 'WAIT' and prev_row['Close'] > prev_row['SMA'] and row['Close'] < row['SMA']:\
current_state = 'SHORT_IMPULSE'\
swing_low = row['Low']\
signal_dict['Signal'] = 'SHORT'\
signal_dict['State'] = 'SHORT_IMPULSE'\
\
# LONG IMPULSE TRACKING\
elif current_state == 'LONG_IMPULSE':\
# Update swing high\
if row['High'] > swing_high:\
swing_high = row['High']\
\
# Check for pullback (low touches SMA)\
if row['Low'] <= row['SMA']:\
current_state = 'LONG_PULLBACK'\
signal_dict['State'] = 'LONG_PULLBACK'\
\
# Check exit if close drops well below SMA\
elif row['Close'] < row['SMA'] * 0.98:\
current_state = 'WAIT'\
signal_dict['State'] = current_state\
\
elif current_state == 'LONG_PULLBACK':\
# Track pullback low\
if 'pull_low' not in signal_dict:\
signal_dict['pull_low'] = None\
\
# Entry condition: close > midpoint and positive sma_change\
pull_low = row['Low'] if row['Low'] < row['SMA'] else row['Low']\
midpoint = (swing_high + row['SMA']) / 2\
\
if row['Close'] > midpoint and sma_change > 0:\
# ENTER LONG\
entry_price = row['Close']\
stop_price = pull_low\
target_price = entry_price + risk_reward * (entry_price - stop_price)\
position = 'LONG'\
current_state = 'LONG_POSITION'\
signal_dict['Signal'] = 'LONG_ENTRY'\
signal_dict['State'] = 'LONG_POSITION'\
signal_dict['Entry'] = entry_price\
signal_dict['Stop'] = stop_price\
signal_dict['Target'] = target_price\
\
# Check exit if close drops well below SMA\
elif row['Close'] < row['SMA'] * 0.95:\
current_state = 'WAIT'\
signal_dict['State'] = current_state\
\
elif current_state == 'SHORT_IMPULSE':\
# Update swing low\
if row['Low'] < swing_low:\
swing_low = row['Low']\
\
# Check for pullback (high touches SMA)\
if row['High'] >= row['SMA']:\
current_state = 'SHORT_PULLBACK'\
signal_dict['State'] = 'SHORT_PULLBACK'\
\
# Check exit if close rises well above SMA\
elif row['Close'] > row['SMA'] * 1.02:\
current_state = 'WAIT'\
signal_dict['State'] = current_state\
\
elif current_state == 'SHORT_PULLBACK':\
# Track pullback high\
if 'pull_high' not in signal_dict:\
signal_dict['pull_high'] = None\
\
# Entry condition: close < midpoint and negative sma_change\
pull_high = row['High'] if row['High'] > row['SMA'] else row['High']\
midpoint = (swing_low + row['SMA']) / 2\
\
if row['Close'] < midpoint and sma_change < 0:\
# ENTER SHORT\
entry_price = row['Close']\
stop_price = pull_high\
target_price = entry_price - risk_reward * (stop_price - entry_price)\
position = 'SHORT'\
current_state = 'SHORT_POSITION'\
signal_dict['Signal'] = 'SHORT_ENTRY'\
signal_dict['State'] = 'SHORT_POSITION'\
signal_dict['Entry'] = entry_price\
signal_dict['Stop'] = stop_price\
signal_dict['Target'] = target_price\
\
# Check exit if close rises well above SMA\
elif row['Close'] > row['SMA'] * 1.05:\
current_state = 'WAIT'\
signal_dict['State'] = current_state\
\
# POSITION MANAGEMENT\
elif current_state in ['LONG_POSITION', 'SHORT_POSITION']:\
# Check for breakeven after +1R\
if position == 'LONG':\
risk = entry_price - stop_price\
breakeven_price = entry_price + risk # +1R\
\
# Move stop to breakeven\
if row['High'] >= breakeven_price:\
stop_price = entry_price\
signal_dict['Stop_Updated'] = True\
\
# Check exit conditions\
if row['Low'] <= stop_price:\
# STOP HIT\
exit_price = stop_price\
r_multiple = (exit_price - entry_price) / risk\
trades.append({\
'Date_Entry': signal_dict['Date'],\
'Date_Exit': row['Time'],\
'Direction': 'LONG',\
'Entry': entry_price,\
'Stop': stop_price,\
'Target': target_price,\
'Exit': exit_price,\
'R_Multiple': r_multiple,\
'Result': 'STOP'\
})\
current_state = 'WAIT'\
position = None\
signal_dict['Signal'] = 'STOP_HIT'\
elif row['High'] >= target_price:\
# TARGET HIT\
exit_price = target_price\
r_multiple = risk_reward\
trades.append({\
'Date_Entry': signal_dict['Date'],\
'Date_Exit': row['Time'],\
'Direction': 'LONG',\
'Entry': entry_price,\
'Stop': stop_price,\
'Target': target_price,\
'Exit': exit_price,\
'R_Multiple': r_multiple,\
'Result': 'TARGET'\
})\
current_state = 'WAIT'\
position = None\
signal_dict['Signal'] = 'TARGET_HIT'\
\
elif position == 'SHORT':\
risk = stop_price - entry_price\
breakeven_price = entry_price - risk # -1R\
\
# Move stop to breakeven\
if row['Low'] <= breakeven_price:\
stop_price = entry_price\
signal_dict['Stop_Updated'] = True\
\
# Check exit conditions\
if row['High'] >= stop_price:\
# STOP HIT\
exit_price = stop_price\
r_multiple = (entry_price - exit_price) / risk\
trades.append({\
'Date_Entry': signal_dict['Date'],\
'Date_Exit': row['Time'],\
'Direction': 'SHORT',\
'Entry': entry_price,\
'Stop': stop_price,\
'Target': target_price,\
'Exit': exit_price,\
'R_Multiple': r_multiple,\
'Result': 'STOP'\
})\
current_state = 'WAIT'\
position = None\
signal_dict['Signal'] = 'STOP_HIT'\
elif row['Low'] <= target_price:\
# TARGET HIT\
exit_price = target_price\
r_multiple = risk_reward\
trades.append({\
'Date_Entry': signal_dict['Date'],\
'Date_Exit': row['Time'],\
'Direction': 'SHORT',\
'Entry': entry_price,\
'Stop': stop_price,\
'Target': target_price,\
'Exit': exit_price,\
'R_Multiple': r_multiple,\
'Result': 'TARGET'\
})\
current_state = 'WAIT'\
position = None\
signal_dict['Signal'] = 'TARGET_HIT'\
\
# Update signal state\
signal_dict['Entry'] = entry_price if entry_price else None\
signal_dict['Stop'] = stop_price if stop_price else None\
signal_dict['Target'] = target_price if target_price else None\
signal_dict['SMA_Change'] = sma_change\
signals.append(signal_dict)\
\
# Convert to DataFrames\
trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=[\
'Date_Entry', 'Date_Exit', 'Direction', 'Entry', 'Stop', 'Target', 'Exit', 'R_Multiple', 'Result'\
])\
\
# Build equity curve\
equity_curve = []\
equity = initial_capital\
for trade in trades:\
equity += trade['R_Multiple'] * 1000 # Assuming $1000 risk per trade (1% of $100k)\
equity_curve.append({\
'Date': trade['Date_Exit'],\
'Equity': equity,\
'R_Multiple': trade['R_Multiple']\
})\
\
# Add initial capital to equity curve\
if equity_curve:\
equity_curve = [{'Date': df.iloc[0]['Time'], 'Equity': initial_capital, 'R_Multiple': 0}] + equity_curve\
equity_df = pd.DataFrame(equity_curve) if equity_curve else pd.DataFrame(columns=['Date', 'Equity', 'R_Multiple'])\
\
# Calculate metrics\
metrics = {}\
if len(trades_df) > 0:\
# Total R\
total_r = trades_df['R_Multiple'].sum()\
\
# Win rate\
wins = trades_df[(trades_df['R_Multiple'] > 0)].shape[0]\
win_rate = wins / len(trades_df) * 100 if len(trades_df) > 0 else 0\
\
# Average R\
avg_r = trades_df['R_Multiple'].mean()\
\
# Max drawdown\
equity_series = equity_df['Equity']\
max_drawdown = 0\
peak = equity_series.iloc[0]\
for price in equity_series:\
if price > peak:\
peak = price\
dd = (peak - price) / peak * 100\
if dd > max_drawdown:\
max_drawdown = dd\
\
# Profit factor\
gross_profit = trades_df[trades_df['R_Multiple'] > 0]['R_Multiple'].sum()\
gross_loss = abs(trades_df[trades_df['R_Multiple'] < 0]['R_Multiple'].sum())\
profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')\
\
metrics = {\
'Total_Trades': len(trades_df),\
'Win_Rate': win_rate,\
'Average_R': avg_r,\
'Total_R': total_r,\
'Max_Drawdown': max_drawdown,\
'Profit_Factor': profit_factor\
}\
else:\
metrics = {\
'Total_Trades': 0,\
'Win_Rate': 0,\
'Average_R': 0,\
'Total_R': 0,\
'Max_Drawdown': 0,\
'Profit_Factor': 0\
}\
\
# Convert signals to DataFrame\
signals_df = pd.DataFrame(signals)\
\
return trades_df, equity_df, metrics, signals_df\
\
def compute_signal(df, sma_period=50, risk_reward=5):\
"""Compute current signal from latest data.\
\
Args:\
df (pd.DataFrame): OHLC data\
sma_period (int): SMA period\
risk_reward (float): Risk:Reward ratio\
\
Returns:\
dict: Dictionary with current signal information\
"""\
df = df.copy()\
df['SMA'] = calculate_sma(df['Close'], sma_period)\
\
if len(df) < sma_period + 2:\
return {'Signal': 'WAIT', 'Reason': 'Not enough data'}\
\
# Get latest values\
last_row = df.iloc[-1]\
prev_row = df.iloc[-2]\
\
sma_change = last_row['SMA'] - prev_row['SMA']\
\
# Check conditions\
if prev_row['Close'] < prev_row['SMA'] and last_row['Close'] > last_row['SMA']:\
return {\
'Signal': 'LONG',\
'Reason': 'Bullish impulse detected',\
'SMA_Change': sma_change\
}\
elif prev_row['Close'] > prev_row['SMA'] and last_row['Close'] < last_row['SMA']:\
return {\
'Signal': 'SHORT',\
'Reason': 'Bearish impulse detected',\
'SMA_Change': sma_change\
}\
elif pd.notna(last_row['SMA']):\
# Check pullback conditions\
if last_row['Low'] <= last_row['SMA'] and sma_change > 0:\
return {\
'Signal': 'LONG_PULLBACK',\
'Reason': 'Pullback to SMA in uptrend',\
'SMA_Change': sma_change\
}\
elif last_row['High'] >= last_row['SMA'] and sma_change < 0:\
return {\
'Signal': 'SHORT_PULLBACK',\
'Reason': 'Pullback to SMA in downtrend',\
'SMA_Change': sma_change\
}\
\
return {\
'Signal': 'WAIT',\
'Reason': 'No setup found',\
'SMA_Change': sma_change\
}\
\
\
# ============================================================================\
# STREAMLIT APPLICATION\
# ============================================================================\
\
def main():\
st.set_page_config(\
page_title="Pullback-to-SMA Strategy Dashboard",\
page_icon="📈",\
layout="wide"\
)\
\
st.title("📊 Pullback-to-SMA Trading Strategy Dashboard")\
st.markdown("""\
**Strategy Description:**\
- Entry: Price breaks SMA (impulse), waits for pullback, enters on momentum resumption\
- Exit: Risk-based targets with breakeven after +1R\
- Risk:Reward: Configurable (default 1:5)\
""")\
\
# Sidebar\
with st.sidebar:\
st.header("⚙️ Configuration")\
\
# Data source selection\
st.subheader("Data Source")\
data_source = st.radio(\
"Choose data source:",\
["Live (yfinance)", "Upload CSV", "Synthetic Data"],\
key="data_source"\
)\
\
# Instrument selection (for live data or synthetic)\
if data_source != "Upload CSV":\
instrument = st.selectbox(\
"Instrument",\
list(INSTRUMENT_SYMBOLS.keys()),\
key="instrument"\
)\
\
# Upload file section\
if data_source == "Upload CSV":\
uploaded_file = st.file_uploader(\
"Upload CSV file",\
type=['csv'],\
help="Format: Time (EET), Open, High, Low, Close"\
)\
\
# Strategy parameters\
st.subheader("Strategy Parameters")\
sma_period = st.slider(\
"SMA Period",\
min_value=10, max_value=200, value=50, step=5,\
key="sma_period"\
)\
\
risk_reward = st.slider(\
"Risk:Reward Ratio",\
min_value=1.0, max_value=10.0, value=5.0, step=0.5,\
key="risk_reward"\
)\
\
# Backtest parameters\
st.subheader("Backtest Parameters")\
initial_capital = st.number_input(\
"Initial Capital ($)",\
min_value=1000, max_value=10000000, value=100000, step=1000,\
key="initial_capital"\
)\
\
if data_source == "Live (yfinance)":\
default_end = datetime.now()\
default_start = default_end - timedelta(days=365*2)\
start_date = st.date_input("Start Date", default_start)\
end_date = st.date_input("End Date", default_end)\
\
# Run button\
run_backtest_btn = st.button("🚀 Run Backtest", type="primary", width="100%")\
\
# Main content\
if run_backtest_btn:\
# Load data based on source\
df = None\
\
if data_source == "Live (yfinance)":\
symbol = INSTRUMENT_SYMBOLS[instrument]\
st.info(f"Fetching data for {instrument} ({symbol})...")\
df = fetch_historical_data(symbol, str(start_date), str(end_date))\
\
# Fallback to synthetic if no data\
if df is None or len(df) == 0:\
st.warning(f"No data available for {instrument}. Using synthetic data instead.")\
df = generate_synthetic_data()\
\
elif data_source == "Upload CSV":\
if uploaded_file is not None:\
try:\
df = pd.read_csv(uploaded_file)\
st.success("CSV loaded successfully")\
\
# Validate columns\
required_cols = ['Time', 'Open', 'High', 'Low', 'Close']\
if not all(col in df.columns for col in required_cols):\
st.error(f"CSV must contain columns: {', '.join(required_cols)}")\
st.stop()\
\
except Exception as e:\
st.error(f"Error loading CSV: {str(e)}")\
st.stop()\
else:\
st.warning("Please upload a CSV file or select another data source.")\
st.stop()\
\
else: # Synthetic Data\
st.info("Using synthetic data for demonstration...")\
df = generate_synthetic_data(days=500)\
\
# Run backtest\
trades_df, equity_df, metrics, signals_df = run_backtest(\
df, sma_period, risk_reward, initial_capital\
)\
\
# Compute current signal\
current_signal = compute_signal(df, sma_period, risk_reward)\
\
# Display results\
# ============\
\
# Header section\
col1, col2, col3 = st.columns(3)\
with col1:\
st.metric(\
"Current Signal",\
current_signal['Signal'],\
delta=current_signal['Reason']\
)\
with col2:\
st.metric(\
"Total Trades",\
f"{metrics['Total_Trades']}"\
)\
with col3:\
st.metric(\
"Total R",\
f"{metrics['Total_R']:.2f}"\
)\
\
# Performance metrics section\
st.subheader("📈 Performance Metrics")\
mcol1, mcol2, mcol3, mcol4, mcol5, mcol6 = st.columns(6)\
with mcol1:\
st.metric("Win Rate", f"{metrics['Win_Rate']:.1f}%")\
with mcol2:\
st.metric("Average R", f"{metrics['Average_R']:.2f}")\
with mcol3:\
st.metric("Total Trades", str(metrics['Total_Trades']))\
with mcol4:\
st.metric("Profit Factor", f"{metrics['Profit_Factor']:.2f}" if metrics['Profit_Factor'] != float('inf') else "∞")\
with mcol5:\
st.metric("Max Drawdown", f"{metrics['Max_Drawdown']:.1f}%")\
with mcol6:\
st.metric("Initial Capital", f"${initial_capital:,.0f}")\
\
# Position sizing info\
st.info(f"💡 Position Sizing: Risking 1% of ${initial_capital:,} = ${initial_capital * 0.01:,.2f} per trade")\
\
# Price and signals chart\
st.subheader("📊 Price Chart with Signals")\
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,\
row_heights=[0.7, 0.3],\
subplot_titles=("Price & SMA", "Equity Curve"))\
\
# Add candlestick chart\
fig.add_trace(go.Candlestick(\
x=df['Time'],\
open=df['Open'], high=df['High'],\
low=df['Low'], close=df['Close'],\
name='OHLC'\
), row=1, col=1)\
\
# Add SMA line\
fig.add_trace(go.Scatter(\
x=df['Time'],\
y=df['SMA'],\
name=f'SMA({sma_period})',\
line=dict(color='orange', width=2)\
), row=1, col=1)\
\
# Add trade markers\
if len(trades_df) > 0:\
# Long entries\
long_trades = trades_df[trades_df['Direction'] == 'LONG']\
fig.add_trace(go.Scatter(\
x=long_trades['Date_Entry'],\
y=long_trades['Entry'],\
mode='markers',\
marker=dict(symbol='triangle-up', size=10, color='green'),\
name='Long Entry'\
), row=1, col=1)\
\
# Short entries\
short_trades = trades_df[trades_df['Direction'] == 'SHORT']\
fig.add_trace(go.Scatter(\
x=short_trades['Date_Entry'],\
y=short_trades['Entry'],\
mode='markers',\
marker=dict(symbol='triangle-down', size=10, color='red'),\
name='Short Entry'\
), row=1, col=1)\
\
# Add equity curve\
if len(equity_df) > 0:\
fig.add_trace(go.Scatter(\
x=equity_df['Date'],\
y=equity_df['Equity'],\
name='Equity Curve',\
line=dict(color='blue', width=2)\
), row=2, col=1)\
\
# Update layout\
fig.update_layout(\
title=f"{instrument if data_source != 'Upload CSV' else 'Uploaded Data'} - Price & Equity",\
xaxis_rangeslider_visible=False,\
height=600,\
template='plotly_dark'\
)\
\
st.plotly_chart(fig, use_container_width=True)\
\
# Signal table\
st.subheader("🎯 Signal Table")\
st.dataframe(signals_df.tail(20), use_container_width=True)\
\
# Trades table\
if len(trades_df) > 0:\
st.subheader("📋 Trade Log")\
st.dataframe(trades_df, use_container_width=True)\
\
# R distribution histogram\
st.subheader("📊 R-Multiple Distribution")\
fig_r = go.Figure(data=[go.Histogram(\
x=trades_df['R_Multiple'],\
nbinsx=20,\
marker_color='skyblue'\
)])\
fig_r.update_layout(\
title="Distribution of R-Multiples",\
xaxis_title="R Multiple",\
yaxis_title="Frequency",\
template='plotly_dark'\
)\
st.plotly_chart(fig_r, use_container_width=True)\
\
# Monthly performance\
st.subheader("📅 Monthly Performance")\
trades_df['Date_Exit'] = pd.to_datetime(trades_df['Date_Exit'])\
trades_df['Month'] = trades_df['Date_Exit'].dt.strftime('%Y-%m')\
monthly_perf = trades_df.groupby('Month')['R_Multiple'].sum().reset_index()\
monthly_perf.columns = ['Month', 'Total R']\
st.dataframe(monthly_perf, use_container_width=True)\
\
# Download buttons\
st.subheader("💾 Download Data")\
dlcol1, dlcol2, dlcol3 = st.columns(3)\
\
if len(trades_df) > 0:\
with dlcol1:\
csv_trades = trades_df.to_csv(index=False)\
st.download_button(\
"📥 Download Trade Log",\
csv_trades,\
file_name="trade_log.csv",\
mime="text/csv"\
)\
\
if len(equity_df) > 0:\
with dlcol2:\
csv_equity = equity_df.to_csv(index=False)\
st.download_button(\
"📥 Download Equity Curve",\
csv_equity,\
file_name="equity_curve.csv",\
mime="text/csv"\
)\
\
with dlcol3:\
metrics_df = pd.DataFrame([metrics])\
csv_metrics = metrics_df.to_csv(index=False)\
st.download_button(\
"📥 Download Summary",\
csv_metrics,\
file_name="summary.csv",\
mime="text/csv"\
)\
\
# Show information about current setup\
st.subheader("ℹ️ Current Setup Information")\
st.write(f"Current signal: **{current_signal['Signal']}**")\
st.write(f"Reason: {current_signal['Reason']}")\
\
else:\
# Show welcome/instructions\
st.info("👈 Configure settings in the sidebar and click 'Run Backtest' to begin.")\
\
# Show a preview of instruments\
st.subheader("Available Instruments")\
for inst, symbol in INSTRUMENT_SYMBOLS.items():\
st.write(f"- **{inst}** (yfinance: {symbol})")\
\
\
if __name__ == "__main__":\
main()
