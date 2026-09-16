import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

from data_fetcher import (
    fetch_ohlcv,
    generate_synthetic_data,
    SUPPORTED_INSTRUMENTS,
    INTERVAL_PERIODS,
)

st.set_page_config(page_title="Multi-Strategy Trading Dashboard", layout="wide")

STRATEGIES = ["SMA-Slope Pullback", "BOS/CHoCH Fibonacci", "School Run"]


# ===== SHARED HELPERS =====
def compute_metrics(trades):
    if trades is None or len(trades) == 0:
        return {"Trades": 0, "Wins": 0, "Losses": 0, "Win Rate": 0.0, "Average R": 0.0, "Total R": 0.0, "Profit Factor": 0.0, "Max Drawdown (R)": 0.0}
    wins = int((trades["R"] > 0).sum())
    losses = int((trades["R"] < 0).sum())
    total_r = float(trades["R"].sum())
    avg_r = float(trades["R"].mean())
    win_rate = float((trades["R"] > 0).mean())
    gross_profit = float(trades[trades["R"] > 0]["R"].sum())
    gross_loss = float(abs(trades[trades["R"] < 0]["R"].sum()))
    pf = gross_profit / gross_loss if gross_loss > 0 else 0.0
    eq = trades["Equity"]
    dd = eq - eq.cummax()
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0
    return {"Trades": len(trades), "Wins": wins, "Losses": losses, "Win Rate": win_rate, "Average R": avg_r, "Total R": total_r, "Profit Factor": pf, "Max Drawdown (R)": max_dd}


def format_metrics_for_display(m):
    return pd.DataFrame({"Metric": list(m.keys()), "Value": [m["Trades"], m["Wins"], m["Losses"], f"{m['Win Rate']:.2%}", m["Average R"], m["Total R"], m["Profit Factor"], m["Max Drawdown (R)"]]})


def position_sizing(account, risk_pct, entry, stop):
    risk_amount = account * (risk_pct / 100.0)
    sd = abs(entry - stop)
    if sd == 0 or np.isnan(sd):
        return {"risk_amount": risk_amount, "units": 0, "notional": 0}
    units = risk_amount / sd
    return {"risk_amount": risk_amount, "units": units, "notional": units * entry}


# ===== STRATEGY A: SMA-SLOPE PULLBACK =====
PARAMS = {
    "Base Case": dict(SLOPE_THRESHOLD=0.00025, TOLERANCE_MULT=0.10, INVALIDATE_MULT=0.60, SL_BUFFER_MULT=0.10, MIN_PULLBACK_BARS=2, RR_TARGET=2.0, MIN_RISK_PRICE=0.0003, SMA_PERIOD=50, SLOPE_LOOKBACK=5, ATR_PERIOD=14, ENTRY_MODE="sma_reclaim", BREAKEVEN_AT_R=1.0),
    "Loose Filter": dict(SLOPE_THRESHOLD=0.00010, TOLERANCE_MULT=0.10, INVALIDATE_MULT=0.60, SL_BUFFER_MULT=0.10, MIN_PULLBACK_BARS=1, RR_TARGET=2.0, MIN_RISK_PRICE=0.0003, SMA_PERIOD=50, SLOPE_LOOKBACK=5, ATR_PERIOD=14, ENTRY_MODE="sma_reclaim", BREAKEVEN_AT_R=1.0),
}


def _prepare_sma_slope(df, p):
    df = df.copy()
    df["SMA"] = df["Close"].rolling(p["SMA_PERIOD"]).mean()
    df["SMA_Slope"] = df["SMA"].diff(p["SLOPE_LOOKBACK"])
    df["PrevClose"] = df["Close"].shift(1)
    df["TR"] = np.maximum(df["High"] - df["Low"], np.maximum(abs(df["High"] - df["PrevClose"]), abs(df["Low"] - df["PrevClose"])))
    df["ATR"] = df["TR"].rolling(p["ATR_PERIOD"]).mean()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def run_backtest_sma_slope(data, p):
    df = _prepare_sma_slope(data, p)
    state = "LOOKING"; trades = []
    swing_high = -np.inf; swing_low = np.inf; pull_extreme = np.nan; pullback_bars = 0
    entry = sl = tp = risk = 0; entry_time = None; direction = None
    for i in range(1, len(df)):
        row = df.iloc[i]; prev = df.iloc[i-1]
        c, h, l, sma, slope, atr = row["Close"], row["High"], row["Low"], row["SMA"], row["SMA_Slope"], row["ATR"]
        pc, psma = prev["Close"], prev["SMA"]
        if state == "IN_TRADE":
            if direction == "LONG":
                if l <= sl:
                    r = (sl - entry) / risk; trades.append([entry_time, row["Time"], "LONG", entry, sl, tp, sl, round(r, 2)]); state = "LOOKING"; continue
                elif h >= tp:
                    trades.append([entry_time, row["Time"], "LONG", entry, sl, tp, tp, p["RR_TARGET"]]); state = "LOOKING"; continue
                if h >= entry + (risk * p["BREAKEVEN_AT_R"]): sl = max(sl, entry)
            elif direction == "SHORT":
                if h >= sl:
                    r = (entry - sl) / risk; trades.append([entry_time, row["Time"], "SHORT", entry, sl, tp, sl, round(r, 2)]); state = "LOOKING"; continue
                elif l <= tp:
                    trades.append([entry_time, row["Time"], "SHORT", entry, sl, tp, tp, p["RR_TARGET"]]); state = "LOOKING"; continue
                if l <= entry - (risk * p["BREAKEVEN_AT_R"]): sl = min(sl, entry)
            continue
        if state == "LOOKING":
            if pc < psma and c > sma: state = "BULL_IMPULSE"; swing_high = h; swing_low = l
            elif pc > psma and c < sma: state = "BEAR_IMPULSE"; swing_high = h; swing_low = l
        elif state == "BULL_IMPULSE":
            swing_high = max(swing_high, h)
            if l <= sma:
                if slope >= p["SLOPE_THRESHOLD"]: state = "BULL_PULLBACK"; pull_extreme = l; pullback_bars = 1
                else: state = "LOOKING"
        elif state == "BULL_PULLBACK":
            pullback_bars += 1; pull_extreme = min(pull_extreme, l)
            if l < swing_low or slope < (p["SLOPE_THRESHOLD"] * p["INVALIDATE_MULT"]): state = "LOOKING"; continue
            if c > sma:
                if pullback_bars >= p["MIN_PULLBACK_BARS"]:
                    entry = c; raw_sl = pull_extreme - (atr * p["SL_BUFFER_MULT"]); risk = max(entry - raw_sl, p["MIN_RISK_PRICE"]); sl = entry - risk; tp = entry + (risk * p["RR_TARGET"]); direction = "LONG"; entry_time = row["Time"]; state = "IN_TRADE"
                else: state = "LOOKING"
        elif state == "BEAR_IMPULSE":
            swing_low = min(swing_low, l)
            if h >= sma:
                if slope <= -p["SLOPE_THRESHOLD"]: state = "BEAR_PULLBACK"; pull_extreme = h; pullback_bars = 1
                else: state = "LOOKING"
        elif state == "BEAR_PULLBACK":
            pullback_bars += 1; pull_extreme = max(pull_extreme, h)
            if h > swing_high or slope > -(p["SLOPE_THRESHOLD"] * p["INVALIDATE_MULT"]): state = "LOOKING"; continue
            if c < sma:
                if pullback_bars >= p["MIN_PULLBACK_BARS"]:
                    entry = c; raw_sl = pull_extreme + (atr * p["SL_BUFFER_MULT"]); risk = max(raw_sl - entry, p["MIN_RISK_PRICE"]); sl = entry + risk; tp = entry - (risk * p["RR_TARGET"]); direction = "SHORT"; entry_time = row["Time"]; state = "IN_TRADE"
                else: state = "LOOKING"
    cols = ["Entry Time", "Exit Time", "Direction", "Entry", "Stop", "Target", "Exit", "R"]
    td = pd.DataFrame(trades, columns=cols)
    if not td.empty: td["Equity"] = td["R"].cumsum()
    return td


def compute_signal_sma_slope(data, p):
    df = _prepare_sma_slope(data, p)
    state = "LOOKING"; swing_high = -np.inf; swing_low = np.inf; pull_extreme = np.nan; pullback_bars = 0
    entry = sl = tp = risk = 0; direction = None
    for i in range(1, len(df)):
        row = df.iloc[i]; prev = df.iloc[i-1]
        c, h, l, sma, slope, atr = row["Close"], row["High"], row["Low"], row["SMA"], row["SMA_Slope"], row["ATR"]
        pc, psma = prev["Close"], prev["SMA"]
        if state == "IN_TRADE":
            if direction == "LONG":
                if l <= sl or h >= tp: state = "LOOKING"; continue
                if h >= entry + (risk * p["BREAKEVEN_AT_R"]): sl = max(sl, entry)
            elif direction == "SHORT":
                if h >= sl or l <= tp: state = "LOOKING"; continue
                if l <= entry - (risk * p["BREAKEVEN_AT_R"]): sl = min(sl, entry)
            continue
        if state == "LOOKING":
            if pc < psma and c > sma: state = "BULL_IMPULSE"; swing_high = h; swing_low = l
            elif pc > psma and c < sma: state = "BEAR_IMPULSE"; swing_high = h; swing_low = l
        elif state == "BULL_IMPULSE":
            swing_high = max(swing_high, h)
            if l <= sma:
                if slope >= p["SLOPE_THRESHOLD"]: state = "BULL_PULLBACK"; pull_extreme = l; pullback_bars = 1
                else: state = "LOOKING"
        elif state == "BULL_PULLBACK":
            pullback_bars += 1; pull_extreme = min(pull_extreme, l)
            if l < swing_low or slope < (p["SLOPE_THRESHOLD"] * p["INVALIDATE_MULT"]): state = "LOOKING"; continue
            if c > sma:
                if pullback_bars >= p["MIN_PULLBACK_BARS"]:
                    entry = c; raw_sl = pull_extreme - (atr * p["SL_BUFFER_MULT"]); risk = max(entry - raw_sl, p["MIN_RISK_PRICE"]); sl = entry - risk; tp = entry + (risk * p["RR_TARGET"]); direction = "LONG"; state = "IN_TRADE"
                else: state = "LOOKING"
        elif state == "BEAR_IMPULSE":
            swing_low = min(swing_low, l)
            if h >= sma:
                if slope <= -p["SLOPE_THRESHOLD"]: state = "BEAR_PULLBACK"; pull_extreme = h; pullback_bars = 1
                else: state = "LOOKING"
        elif state == "BEAR_PULLBACK":
            pullback_bars += 1; pull_extreme = max(pull_extreme, h)
            if h > swing_high or slope > -(p["SLOPE_THRESHOLD"] * p["INVALIDATE_MULT"]): state = "LOOKING"; continue
            if c < sma:
                if pullback_bars >= p["MIN_PULLBACK_BARS"]:
                    entry = c; raw_sl = pull_extreme + (atr * p["SL_BUFFER_MULT"]); risk = max(raw_sl - entry, p["MIN_RISK_PRICE"]); sl = entry + risk; tp = entry - (risk * p["RR_TARGET"]); direction = "SHORT"; state = "IN_TRADE"
                else: state = "LOOKING"
    sig = "WAIT"
    if state == "IN_TRADE": sig = "LONG" if direction == "LONG" else "SHORT"
    cp = df.iloc[-1]["Close"]
    return {"state": state, "signal": sig, "current_price": cp, "sma": df.iloc[-1]["SMA"], "entry": entry, "stop": sl, "target": tp}


# ===== STRATEGY B: BOS/CHoCH FIBONACCI =====
def find_swing_points(data, window=10):
    highs = []; lows = []
    for i in range(window, len(data) - window):
        if data["High"].iloc[i] == data["High"].iloc[i-window:i+window+1].max():
            highs.append({"index": i, "price": data["High"].iloc[i], "time": data["Time"].iloc[i]})
        if data["Low"].iloc[i] == data["Low"].iloc[i-window:i+window+1].min():
            lows.append({"index": i, "price": data["Low"].iloc[i], "time": data["Time"].iloc[i]})
    return highs, lows


def calculate_fibonacci_levels(high_price, low_price):
    diff = high_price - low_price
    return {"0%": low_price, "23.6%": low_price + 0.236*diff, "38.2%": low_price + 0.382*diff, "50%": low_price + 0.5*diff, "61.8%": low_price + 0.618*diff, "78.6%": low_price + 0.786*diff, "100%": high_price, "161.8%": high_price + 0.618*diff}


def run_backtest_fib(data, window=10):
    df = data.copy().reset_index(drop=True)
    if len(df) < window * 2 + 20: return pd.DataFrame()
    swh, swl = find_swing_points(df, window)
    trades = []
    for h in swh:
        for l in swl:
            if h["price"] <= l["price"]: continue
            if h["index"] < l["index"]: continue
            recent_idx = max(h["index"], l["index"])
            if recent_idx + 20 >= len(df): continue
            fib = calculate_fibonacci_levels(h["price"], l["price"])
            recent = df.iloc[recent_idx:recent_idx+20]
            setup_type = None
            if recent["Close"].max() > h["price"]: setup_type = "BOS"
            elif recent["Close"].min() < l["price"] and recent["Close"].iloc[-1] > fib["38.2%"]: setup_type = "CHoCH"
            if setup_type is None: continue
            entry = fib["61.8%"] if setup_type == "BOS" else fib["78.6%"]
            stop = fib["0%"]
            target = fib["161.8%"]
            direction = "LONG"
            risk = abs(entry - stop)
            if risk <= 0: continue
            # scan forward for outcome
            exit_price = None; exit_time = None; exit_r = 0
            for j in range(recent_idx, len(df)):
                row = df.iloc[j]
                if direction == "LONG":
                    if row["Low"] <= stop: exit_price = stop; exit_time = row["Time"]; exit_r = (stop - entry) / risk; break
                    if row["High"] >= target: exit_price = target; exit_time = row["Time"]; exit_r = (target - entry) / risk; break
            if exit_price is None: continue
            trades.append([df.iloc[recent_idx]["Time"], exit_time, direction, entry, stop, target, exit_price, round(exit_r, 2)])
    cols = ["Entry Time", "Exit Time", "Direction", "Entry", "Stop", "Target", "Exit", "R"]
    td = pd.DataFrame(trades, columns=cols)
    if not td.empty: td["Equity"] = td["R"].cumsum()
    return td


def compute_signal_fib(data, window=10):
    df = data.copy().reset_index(drop=True)
    if len(df) < window * 2 + 20:
        return {"signal": "WAIT", "current_price": data.iloc[-1]["Close"] if len(data) else None, "entry": np.nan, "stop": np.nan, "target": np.nan}
    swh, swl = find_swing_points(df, window)
    if not swh or not swl: return {"signal": "WAIT", "current_price": df.iloc[-1]["Close"], "entry": np.nan, "stop": np.nan, "target": np.nan}
    latest_high = max(swh, key=lambda x: x["index"])
    latest_low = max(swl, key=lambda x: x["index"])
    fib = calculate_fibonacci_levels(latest_high["price"], latest_low["price"])
    cp = df.iloc[-1]["Close"]
    sig = "WAIT"; entry = np.nan; stop = np.nan; target = np.nan
    if cp > fib["61.8%"]: sig = "LONG"; entry = fib["61.8%"]; stop = fib["0%"]; target = fib["161.8%"]
    elif cp > fib["38.2%"]: sig = "CHoCH"; entry = fib["78.6%"]; stop = fib["0%"]; target = fib["161.8%"]
    return {"signal": sig, "current_price": cp, "entry": entry, "stop": stop, "target": target}


# ===== STRATEGY C: SCHOOL RUN =====
def run_backtest_school_run(data, rr_target=2.0, buffer=0.0002):
    df = data.copy().reset_index(drop=True)
    if len(df) == 0: return pd.DataFrame()
    df["Day"] = df["Time"].dt.date
    trades = []
    for day, g in df.groupby("Day"):
        g = g.reset_index(drop=True)
        if len(g) < 3: continue
        signal_bar = g.iloc[1]  # 2nd candle as signal bar
        buy = signal_bar["High"] + buffer
        sell = signal_bar["Low"] - buffer
        sig_high = signal_bar["High"]; sig_low = signal_bar["Low"]
        entry_time = signal_bar["Time"]
        # LONG trigger
        for j in range(2, len(g)):
            row = g.iloc[j]
            if row["High"] >= buy:
                entry = buy; stop = sig_low; risk = abs(entry - stop)
                if risk <= 0: continue
                target = entry + risk * rr_target
                direction = "LONG"; exit_price = None; exit_time = None; exit_r = 0
                for k in range(j, len(g)):
                    r2 = g.iloc[k]
                    if direction == "LONG":
                        if r2["Low"] <= stop: exit_price = stop; exit_time = r2["Time"]; exit_r = (stop - entry) / risk; break
                        if r2["High"] >= target: exit_price = target; exit_time = r2["Time"]; exit_r = (target - entry) / risk; break
                if exit_price is not None:
                    trades.append([entry_time, exit_time, direction, entry, stop, target, exit_price, round(exit_r, 2)])
                break
            if row["Low"] <= sell:
                entry = sell; stop = sig_high; risk = abs(entry - stop)
                if risk <= 0: continue
                target = entry - risk * rr_target
                direction = "SHORT"; exit_price = None; exit_time = None; exit_r = 0
                for k in range(j, len(g)):
                    r2 = g.iloc[k]
                    if direction == "SHORT":
                        if r2["High"] >= stop: exit_price = stop; exit_time = r2["Time"]; exit_r = (entry - stop) / risk; break
                        if r2["Low"] <= target: exit_price = target; exit_time = r2["Time"]; exit_r = (entry - target) / risk; break
                if exit_price is not None:
                    trades.append([entry_time, exit_time, direction, entry, stop, target, exit_price, round(exit_r, 2)])
                break
    cols = ["Entry Time", "Exit Time", "Direction", "Entry", "Stop", "Target", "Exit", "R"]
    td = pd.DataFrame(trades, columns=cols)
    if not td.empty: td["Equity"] = td["R"].cumsum()
    return td


def compute_signal_school_run(data, rr_target=2.0, buffer=0.0002):
    df = data.copy().reset_index(drop=True)
    if len(df) == 0: return {"signal": "WAIT", "current_price": None, "entry": np.nan, "stop": np.nan, "target": np.nan}
    today = df.iloc[-1]["Time"].date()
    g = df[df["Time"].dt.date == today]
    if len(g) < 2: return {"signal": "WAIT", "current_price": df.iloc[-1]["Close"], "entry": np.nan, "stop": np.nan, "target": np.nan}
    g = g.reset_index(drop=True)
    signal_bar = g.iloc[1]
    buy = signal_bar["High"] + buffer; sell = signal_bar["Low"] - buffer
    cp = df.iloc[-1]["Close"]
    sig = "WAIT"; entry = np.nan; stop = np.nan; target = np.nan
    if cp > buy:
        sig = "LONG"; entry = buy; stop = signal_bar["Low"]; target = entry + abs(entry - stop) * rr_target
    elif cp < sell:
        sig = "SHORT"; entry = sell; stop = signal_bar["High"]; target = entry - abs(entry - stop) * rr_target
    return {"signal": sig, "current_price": cp, "entry": entry, "stop": stop, "target": target}


# ===== CACHED FETCH =====
@st.cache_data(ttl=1800, show_spinner=False)
def cached_fetch(symbol, interval, source):
    if source == "Live (yfinance)":
        return fetch_ohlcv(symbol, interval=interval)
    else:
        freq = "h"
        if interval == "15m": freq = "15min"
        elif interval == "30m": freq = "30min"
        elif interval == "1d": freq = "D"
        return generate_synthetic_data(symbol, freq=freq)


# ===== UI =====
st.title("Multi-Strategy Trading Dashboard")
st.markdown("Choose a strategy, backtest against maximum yfinance history, and view today's live signal. Initial account **R100,000**.")

with st.sidebar:
    st.header("Settings")
    strategy = st.selectbox("Strategy", STRATEGIES)
    instrument = st.selectbox("Instrument", list(SUPPORTED_INSTRUMENTS.keys()))
    interval = st.selectbox("Interval", ["15m", "30m", "1h", "1d"], index=2)
    if strategy == "SMA-Slope Pullback":
        scenario = st.selectbox("Scenario", list(PARAMS.keys()))
    sma_period = st.number_input("SMA Period", min_value=10, max_value=200, value=50, step=1)
    rr_target = st.number_input("RR Target", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
    initial_capital = st.number_input("Initial Capital (R)", min_value=1000, value=100000, step=1000)
    risk_pct = st.number_input("Risk per Trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    data_source = st.radio("Data Source", ["Live (yfinance)", "Synthetic"])
    run_btn = st.button("Run Backtest")

if run_btn:
    df = cached_fetch(instrument, interval, data_source)
    if df is None or len(df) == 0:
        st.error("Failed to fetch data. Try another interval or use Synthetic data.")
    else:
        if strategy == "SMA-Slope Pullback":
            p = dict(PARAMS[scenario]); p["SMA_PERIOD"] = int(sma_period); p["RR_TARGET"] = float(rr_target)
            trades = run_backtest_sma_slope(df, p); signal = compute_signal_sma_slope(df, p)
        elif strategy == "BOS/CHoCH Fibonacci":
            trades = run_backtest_fib(df, window=10); signal = compute_signal_fib(df, window=10)
        else:
            trades = run_backtest_school_run(df, rr_target=float(rr_target)); signal = compute_signal_school_run(df, rr_target=float(rr_target))

        metrics = compute_metrics(trades)
        data_start = df["Time"].min(); data_end = df["Time"].max()
        st.caption(f"Backtest data: **{data_start} → {data_end}** ({len(df)} bars, interval {interval}). Current signal from latest bar ({data_end}).")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Signal", signal["signal"])
        col2.metric("Current Price", round(signal["current_price"], 4) if signal["current_price"] is not None else "N/A")
        col3.metric("Total Trades", str(metrics["Trades"]))
        col4.metric("Total R", str(round(metrics["Total R"], 2)))

        st.subheader("Current Setup (Today)")
        if signal["signal"] != "WAIT" and not np.isnan(signal["entry"]):
            s1, s2, s3 = st.columns(3)
            s1.metric("Entry", round(signal["entry"], 4)); s2.metric("Stop", round(signal["stop"], 4)); s3.metric("Target", round(signal["target"], 4))
        else:
            st.info("No active trade setup for today.")

        st.subheader("Position Sizing (R100,000 account)")
        if signal["signal"] != "WAIT" and not np.isnan(signal["entry"]) and not np.isnan(signal["stop"]):
            sizing = position_sizing(initial_capital, risk_pct, signal["entry"], signal["stop"])
            p1, p2, p3 = st.columns(3)
            p1.metric("Risk Amount (R)", round(sizing["risk_amount"], 2)); p2.metric("Units", round(sizing["units"], 2)); p3.metric("Notional (R)", round(sizing["notional"], 2))
        else:
            st.info("Position sizing appears once a signal is active.")

        tab1, tab2, tab3 = st.tabs(["Performance Dashboard", "Trade Log", "Summary"])
        with tab1:
            st.subheader("Equity Curve")
            if len(trades) > 0:
                eq_df = trades[["Entry Time", "Equity"]].copy()
                initial_eq = pd.DataFrame({"Entry Time": [eq_df.iloc[0]["Entry Time"]], "Equity": [0]})
                eq_df = pd.concat([initial_eq, eq_df], ignore_index=True)
                fig = px.line(eq_df, x="Entry Time", y="Equity", markers=True, title="Equity Curve (R multiples)")
                st.plotly_chart(fig, width="stretch")
                fig2 = px.histogram(trades, x="R", nbins=40, title="Trade R-Distribution")
                st.plotly_chart(fig2, width="stretch")
            else:
                st.warning("No trades generated for the current parameters.")
        with tab2:
            st.subheader("Trade Log")
            if len(trades) > 0:
                st.dataframe(trades)
                csv = trades.to_csv(index=False)
                st.download_button("Download Trade Log CSV", data=csv, file_name="trade_log.csv", mime="text/csv")
            else:
                st.warning("No trades to display.")
        with tab3:
            st.subheader("Summary Metrics")
            display_df = format_metrics_for_display(metrics)
            st.dataframe(display_df)
            summary_csv = display_df.to_csv(index=False)
            st.download_button("Download Summary CSV", data=summary_csv, file_name="summary.csv", mime="text/csv")
else:
    st.info("Adjust settings in the sidebar and click **Run Backtest** to see results.")
