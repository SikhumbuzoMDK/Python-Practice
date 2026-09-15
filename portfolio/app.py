import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# Import from data_fetcher module
from data_fetcher import (
    fetch_ohlcv,
    generate_synthetic_data,
    SUPPORTED_INSTRUMENTS,
    INTERVAL_MAX_DAYS,
)

st.set_page_config(page_title="Pullback-to-SMA Strategy Dashboard", layout="wide")


# ==================== STRATEGY FUNCTIONS ====================
def run_backtest(df, sma_period=50, rr=5):
    cols = ["Entry Time", "Exit Time", "Direction", "Entry", "Stop", "Target", "Exit", "R", "Equity"]
    if df is None or len(df) < sma_period + 5:
        return pd.DataFrame(columns=cols)
    df = df.copy()
    df["SMA"] = df["Close"].rolling(sma_period).mean()

    state = "LOOKING"
    trades = []
    position = None
    swing_high = np.nan
    swing_low = np.nan
    pull_high = np.nan
    pull_low = np.nan
    entry = stop = target = np.nan
    entry_time = None
    sma_start = np.nan
    impulse_index = 0

    for i in range(sma_period + 5, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        close = row["Close"]
        high = row["High"]
        low = row["Low"]
        sma = row["SMA"]
        if np.isnan(sma):
            continue

        if position == "LONG":
            initial_risk = entry - stop
            if initial_risk == 0 or np.isnan(initial_risk):
                position = None
                state = "LOOKING"
                continue
            if stop < entry and high >= entry + initial_risk:
                stop = entry
            if low <= stop:
                exit_price = stop
                r_val = (exit_price - entry) / initial_risk
                trades.append([entry_time, row["Time"], "LONG", entry, stop, target, exit_price, round(r_val, 2)])
                position = None
                state = "LOOKING"
                continue
            elif high >= target:
                trades.append([entry_time, row["Time"], "LONG", entry, stop, target, target, rr])
                position = None
                state = "LOOKING"
                continue
        elif position == "SHORT":
            initial_risk = stop - entry
            if initial_risk == 0 or np.isnan(initial_risk):
                position = None
                state = "LOOKING"
                continue
            if stop > entry and low <= entry - initial_risk:
                stop = entry
            if high >= stop:
                exit_price = stop
                r_val = (entry - exit_price) / initial_risk
                trades.append([entry_time, row["Time"], "SHORT", entry, stop, target, exit_price, round(r_val, 2)])
                position = None
                state = "LOOKING"
                continue
            elif low <= target:
                trades.append([entry_time, row["Time"], "SHORT", entry, stop, target, target, rr])
                position = None
                state = "LOOKING"
                continue

        if position is None:
            if state == "LOOKING":
                if prev["Close"] < prev["SMA"] and close > sma:
                    state = "BULL"
                    swing_low = prev["Low"]
                    swing_high = high
                    impulse_index = i
            elif state == "BULL":
                if high > swing_high:
                    swing_high = high
                if low <= sma:
                    state = "PULLBACK"
                    pull_low = low
                    sma_start = df.iloc[impulse_index]["SMA"]
            elif state == "PULLBACK":
                if low < pull_low:
                    pull_low = low
                if low < swing_low:
                    state = "LOOKING"
                    continue
                sma_change = sma - sma_start
                midpoint = (swing_high + pull_low) / 2
                if close > midpoint and sma_change > 0:
                    risk = midpoint - pull_low
                    if risk <= 0:
                        state = "LOOKING"
                        continue
                    entry = close
                    stop = pull_low
                    target = entry + rr * (entry - stop)
                    position = "LONG"
                    entry_time = row["Time"]

            if state == "LOOKING":
                if prev["Close"] > prev["SMA"] and close < sma:
                    state = "BEAR"
                    swing_high = prev["High"]
                    swing_low = low
                    impulse_index = i
            elif state == "BEAR":
                if low < swing_low:
                    swing_low = low
                if high >= sma:
                    state = "SHORT_PULLBACK"
                    pull_high = high
                    sma_start = df.iloc[impulse_index]["SMA"]
            elif state == "SHORT_PULLBACK":
                if high > pull_high:
                    pull_high = high
                if high > swing_high:
                    state = "LOOKING"
                    continue
                sma_change = sma - sma_start
                midpoint = (pull_high + swing_low) / 2
                if close < midpoint and sma_change < 0:
                    risk = pull_high - midpoint
                    if risk <= 0:
                        state = "LOOKING"
                        continue
                    entry = close
                    stop = pull_high
                    target = entry - rr * (stop - entry)
                    position = "SHORT"
                    entry_time = row["Time"]

    if len(trades) == 0:
        return pd.DataFrame(columns=cols)
    trades = pd.DataFrame(trades, columns=["Entry Time", "Exit Time", "Direction", "Entry", "Stop", "Target", "Exit", "R"])
    trades["Equity"] = trades["R"].cumsum()
    return trades


def compute_signal(df, sma_period=50, rr=5):
    if df is None or len(df) < sma_period + 5:
        return {"state": "LOOKING", "signal": "WAIT", "current_price": None, "sma": None}
    df = df.copy()
    df["SMA"] = df["Close"].rolling(sma_period).mean()

    state = "LOOKING"
    position = None
    swing_high = np.nan
    swing_low = np.nan
    pull_high = np.nan
    pull_low = np.nan
    entry = stop = target = np.nan
    sma_start = np.nan
    impulse_index = 0

    for i in range(sma_period + 5, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        close = row["Close"]
        high = row["High"]
        low = row["Low"]
        sma = row["SMA"]
        if np.isnan(sma):
            continue

        if position == "LONG":
            initial_risk = entry - stop
            if initial_risk == 0 or np.isnan(initial_risk):
                position = None
                state = "LOOKING"
                continue
            if stop < entry and high >= entry + initial_risk:
                stop = entry
            if low <= stop or high >= target:
                position = None
                state = "LOOKING"
                continue
        elif position == "SHORT":
            initial_risk = stop - entry
            if initial_risk == 0 or np.isnan(initial_risk):
                position = None
                state = "LOOKING"
                continue
            if stop > entry and low <= entry - initial_risk:
                stop = entry
            if high >= stop or low <= target:
                position = None
                state = "LOOKING"
                continue

        if position is None:
            if state == "LOOKING":
                if prev["Close"] < prev["SMA"] and close > sma:
                    state = "BULL"
                    swing_low = prev["Low"]
                    swing_high = high
                    impulse_index = i
            elif state == "BULL":
                if high > swing_high:
                    swing_high = high
                if low <= sma:
                    state = "PULLBACK"
                    pull_low = low
                    sma_start = df.iloc[impulse_index]["SMA"]
            elif state == "PULLBACK":
                if low < pull_low:
                    pull_low = low
                if low < swing_low:
                    state = "LOOKING"
                    continue
                sma_change = sma - sma_start
                midpoint = (swing_high + pull_low) / 2
                if close > midpoint and sma_change > 0:
                    risk = midpoint - pull_low
                    if risk <= 0:
                        state = "LOOKING"
                        continue
                    entry = close
                    stop = pull_low
                    target = entry + rr * (entry - stop)
                    position = "LONG"
            if state == "LOOKING":
                if prev["Close"] > prev["SMA"] and close < sma:
                    state = "BEAR"
                    swing_high = prev["High"]
                    swing_low = low
                    impulse_index = i
            elif state == "BEAR":
                if low < swing_low:
                    swing_low = low
                if high >= sma:
                    state = "SHORT_PULLBACK"
                    pull_high = high
                    sma_start = df.iloc[impulse_index]["SMA"]
            elif state == "SHORT_PULLBACK":
                if high > pull_high:
                    pull_high = high
                if high > swing_high:
                    state = "LOOKING"
                    continue
                sma_change = sma - sma_start
                midpoint = (pull_high + swing_low) / 2
                if close < midpoint and sma_change < 0:
                    risk = pull_high - midpoint
                    if risk <= 0:
                        state = "LOOKING"
                        continue
                    entry = close
                    stop = pull_high
                    target = entry - rr * (stop - entry)
                    position = "SHORT"

    current_price = df.iloc[-1]["Close"]
    sma_val = df.iloc[-1]["SMA"]
    sig = "WAIT"
    if position == "LONG":
        sig = "LONG"
    elif position == "SHORT":
        sig = "SHORT"
    return {
        "state": state,
        "signal": sig,
        "current_price": current_price,
        "sma": sma_val,
        "entry": entry,
        "stop": stop,
        "target": target,
    }


def compute_metrics(trades):
    # All-numeric values so pyarrow can serialize cleanly.
    if trades is None or len(trades) == 0:
        return {
            "Trades": 0,
            "Wins": 0,
            "Losses": 0,
            "Win Rate": 0.0,
            "Average R": 0.0,
            "Total R": 0.0,
            "Profit Factor": 0.0,
            "Max Drawdown (R)": 0.0,
        }
    wins = int((trades["R"] > 0).sum())
    losses = int((trades["R"] < 0).sum())
    total_r = float(trades["R"].sum())
    avg_r = float(trades["R"].mean())
    win_rate = float((trades["R"] > 0).mean())
    gross_profit = float(trades[trades["R"] > 0]["R"].sum())
    gross_loss = float(abs(trades[trades["R"] < 0]["R"].sum()))
    pf = gross_profit / gross_loss if gross_loss > 0 else 0.0
    eq = trades["Equity"]
    drawdowns = eq - eq.cummax()
    max_dd = float(drawdowns.min()) if len(drawdowns) > 0 else 0.0
    return {
        "Trades": len(trades),
        "Wins": wins,
        "Losses": losses,
        "Win Rate": win_rate,
        "Average R": avg_r,
        "Total R": total_r,
        "Profit Factor": pf,
        "Max Drawdown (R)": max_dd,
    }


def format_metrics_for_display(metrics):
    return pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Value": [
            metrics["Trades"],
            metrics["Wins"],
            metrics["Losses"],
            f"{metrics['Win Rate']:.2%}",
            metrics["Average R"],
            metrics["Total R"],
            metrics["Profit Factor"],
            metrics["Max Drawdown (R)"],
        ],
    })


def position_sizing(account, risk_pct, entry, stop):
    risk_amount = account * (risk_pct / 100.0)
    stop_distance = abs(entry - stop)
    if stop_distance == 0 or np.isnan(stop_distance):
        return {"risk_amount": risk_amount, "units": 0, "notional": 0}
    units = risk_amount / stop_distance
    notional = units * entry
    return {"risk_amount": risk_amount, "units": units, "notional": notional}


# ==================== CACHED DATA FETCH ====================
@st.cache_data(ttl=1800, show_spinner=False)
def cached_fetch(symbol, interval, source):
    if source == "Live (yfinance)":
        return fetch_ohlcv(symbol, interval=interval)
    else:
        freq = "h"
        if interval == "15m":
            freq = "15min"
        elif interval == "30m":
            freq = "30min"
        elif interval == "1d":
            freq = "D"
        return generate_synthetic_data(symbol, freq=freq)


# ==================== UI ====================
st.title("Pullback-to-SMA Strategy Dashboard")
st.markdown("**Backtest:** maximum yfinance history for the selected interval. **Current signal:** today's latest bar.")

with st.sidebar:
    st.header("Settings")
    instrument = st.selectbox("Instrument", list(SUPPORTED_INSTRUMENTS.keys()))
    sma_period = st.number_input("SMA Period", min_value=10, max_value=200, value=50, step=1)
    rr = st.number_input("Risk:Reward", min_value=1, max_value=20, value=5, step=1)
    interval = st.selectbox("Interval", ["15m", "30m", "1h", "1d"], index=2)
    initial_capital = st.number_input("Initial Capital (R)", min_value=1000, value=100000, step=1000)
    risk_pct = st.number_input("Risk per Trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    data_source = st.radio("Data Source", ["Live (yfinance)", "Synthetic"])
    run_btn = st.button("Run Backtest")

if run_btn:
    df = cached_fetch(instrument, interval, data_source)

    if df is None or len(df) == 0:
        st.error("Failed to fetch data. Try another interval or use Synthetic data.")
    else:
        trades = run_backtest(df, sma_period=sma_period, rr=rr)
        signal = compute_signal(df, sma_period=sma_period, rr=rr)
        metrics = compute_metrics(trades)

        data_start = df["Time"].min()
        data_end = df["Time"].max()
        st.caption(
            f"Backtest data: **{data_start} → {data_end}** "
            f"({len(df)} bars, interval {interval}). "
            f"Current signal derived from latest bar ({data_end})."
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Signal", signal["signal"])
        col2.metric("Current Price", round(signal["current_price"], 4) if signal["current_price"] is not None else "N/A")
        col3.metric("Total Trades", str(metrics["Trades"]))
        col4.metric("Total R", str(metrics["Total R"]))

        st.subheader("Current Setup (Today)")
        if signal["signal"] != "WAIT":
            s1, s2, s3 = st.columns(3)
            s1.metric("Entry", round(signal["entry"], 4) if not np.isnan(signal["entry"]) else "N/A")
            s2.metric("Stop", round(signal["stop"], 4) if not np.isnan(signal["stop"]) else "N/A")
            s3.metric("Target", round(signal["target"], 4) if not np.isnan(signal["target"]) else "N/A")
        else:
            st.info("No active trade setup for today. Strategy is waiting for a valid impulse-pullback signal.")

        st.subheader("Position Sizing (R100,000 account)")
        if signal["signal"] != "WAIT" and not np.isnan(signal["entry"]) and not np.isnan(signal["stop"]):
            sizing = position_sizing(initial_capital, risk_pct, signal["entry"], signal["stop"])
            p1, p2, p3 = st.columns(3)
            p1.metric("Risk Amount (R)", round(sizing["risk_amount"], 2))
            p2.metric("Units", round(sizing["units"], 2))
            p3.metric("Notional (R)", round(sizing["notional"], 2))
        else:
            st.info("Position sizing will appear once a signal is active.")

        tab1, tab2, tab3 = st.tabs(["Performance Dashboard", "Trade Log", "Summary"])

        with tab1:
            st.subheader("Equity Curve")
            if len(trades) > 0:
                eq_df = trades[["Entry Time", "Equity"]].copy()
                initial_eq = pd.DataFrame({"Entry Time": [eq_df.iloc[0]["Entry Time"]], "Equity": [0]})
                eq_df = pd.concat([initial_eq, eq_df], ignore_index=True)
                fig = px.line(eq_df, x="Entry Time", y="Equity", markers=True, title="Equity Curve (R multiples)")
                st.plotly_chart(fig, width="stretch")

                st.subheader("R Distribution")
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
    st.info("Adjust settings in the sidebar and click **Run Backtest** to fetch max-history data, generate trades, and see today's signal.")
