import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# Import from data_fetcher module
from data_fetcher import (
    fetch_ohlcv,
    generate_synthetic_data,
    SUPPORTED_INSTRUMENTS,
    INTERVAL_PERIODS,
)

st.set_page_config(page_title="SMA-Slope Pullback Strategy Dashboard", layout="wide")


# ==================== PARAMS ====================
PARAMS = {
    "Base Case": dict(
        SLOPE_THRESHOLD=0.00025, TOLERANCE_MULT=0.10, INVALIDATE_MULT=0.60,
        SL_BUFFER_MULT=0.10, MIN_PULLBACK_BARS=2, RR_TARGET=2.0,
        MIN_RISK_PRICE=0.0003, SMA_PERIOD=50, SLOPE_LOOKBACK=5, ATR_PERIOD=14,
        ENTRY_MODE="sma_reclaim", BREAKEVEN_AT_R=1.0,
    ),
    "Loose Filter": dict(
        SLOPE_THRESHOLD=0.00010, TOLERANCE_MULT=0.10, INVALIDATE_MULT=0.60,
        SL_BUFFER_MULT=0.10, MIN_PULLBACK_BARS=1, RR_TARGET=2.0,
        MIN_RISK_PRICE=0.0003, SMA_PERIOD=50, SLOPE_LOOKBACK=5, ATR_PERIOD=14,
        ENTRY_MODE="sma_reclaim", BREAKEVEN_AT_R=1.0,
    ),
}


# ==================== NEW STRATEGY ENGINE ====================
def run_backtest(data, p):
    df = data.copy()

    # Indicators
    df["SMA"] = df["Close"].rolling(p["SMA_PERIOD"]).mean()
    df["SMA_Slope"] = df["SMA"].diff(p["SLOPE_LOOKBACK"])

    # True Range & ATR
    df["PrevClose"] = df["Close"].shift(1)
    df["TR"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(abs(df["High"] - df["PrevClose"]), abs(df["Low"] - df["PrevClose"])),
    )
    df["ATR"] = df["TR"].rolling(p["ATR_PERIOD"]).mean()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    state = "LOOKING"
    trades = []
    swing_high = -np.inf
    swing_low = np.inf
    pull_extreme = np.nan
    pullback_bars = 0
    entry = sl = tp = risk = 0
    entry_time = None
    direction = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        c, h, l, sma, slope, atr = row["Close"], row["High"], row["Low"], row["SMA"], row["SMA_Slope"], row["ATR"]
        pc, psma = prev["Close"], prev["SMA"]

        # --- TRADE MANAGEMENT ---
        if state == "IN_TRADE":
            if direction == "LONG":
                if l <= sl:
                    r = (sl - entry) / risk
                    trades.append([entry_time, row["Time"], "LONG", entry, sl, tp, sl, round(r, 2)])
                    state = "LOOKING"
                    continue
                elif h >= tp:
                    trades.append([entry_time, row["Time"], "LONG", entry, sl, tp, tp, p["RR_TARGET"]])
                    state = "LOOKING"
                    continue
                if h >= entry + (risk * p["BREAKEVEN_AT_R"]):
                    sl = max(sl, entry)
            elif direction == "SHORT":
                if h >= sl:
                    r = (entry - sl) / risk
                    trades.append([entry_time, row["Time"], "SHORT", entry, sl, tp, sl, round(r, 2)])
                    state = "LOOKING"
                    continue
                elif l <= tp:
                    trades.append([entry_time, row["Time"], "SHORT", entry, sl, tp, tp, p["RR_TARGET"]])
                    state = "LOOKING"
                    continue
                if l <= entry - (risk * p["BREAKEVEN_AT_R"]):
                    sl = min(sl, entry)
            continue

        # --- SETUP LOGIC ---
        if state == "LOOKING":
            if pc < psma and c > sma:
                state = "BULL_IMPULSE"
                swing_high = h
                swing_low = l
            elif pc > psma and c < sma:
                state = "BEAR_IMPULSE"
                swing_high = h
                swing_low = l
        # BULL
        elif state == "BULL_IMPULSE":
            swing_high = max(swing_high, h)
            if l <= sma:
                if slope >= p["SLOPE_THRESHOLD"]:
                    state = "BULL_PULLBACK"
                    pull_extreme = l
                    pullback_bars = 1
                else:
                    state = "LOOKING"
        elif state == "BULL_PULLBACK":
            pullback_bars += 1
            pull_extreme = min(pull_extreme, l)
            if l < swing_low or slope < (p["SLOPE_THRESHOLD"] * p["INVALIDATE_MULT"]):
                state = "LOOKING"
                continue
            if c > sma:
                if pullback_bars >= p["MIN_PULLBACK_BARS"]:
                    entry = c
                    raw_sl = pull_extreme - (atr * p["SL_BUFFER_MULT"])
                    risk = max(entry - raw_sl, p["MIN_RISK_PRICE"])
                    sl = entry - risk
                    tp = entry + (risk * p["RR_TARGET"])
                    direction = "LONG"
                    entry_time = row["Time"]
                    state = "IN_TRADE"
                else:
                    state = "LOOKING"
        # BEAR
        elif state == "BEAR_IMPULSE":
            swing_low = min(swing_low, l)
            if h >= sma:
                if slope <= -p["SLOPE_THRESHOLD"]:
                    state = "BEAR_PULLBACK"
                    pull_extreme = h
                    pullback_bars = 1
                else:
                    state = "LOOKING"
        elif state == "BEAR_PULLBACK":
            pullback_bars += 1
            pull_extreme = max(pull_extreme, h)
            if h > swing_high or slope > -(p["SLOPE_THRESHOLD"] * p["INVALIDATE_MULT"]):
                state = "LOOKING"
                continue
            if c < sma:
                if pullback_bars >= p["MIN_PULLBACK_BARS"]:
                    entry = c
                    raw_sl = pull_extreme + (atr * p["SL_BUFFER_MULT"])
                    risk = max(raw_sl - entry, p["MIN_RISK_PRICE"])
                    sl = entry + risk
                    tp = entry - (risk * p["RR_TARGET"])
                    direction = "SHORT"
                    entry_time = row["Time"]
                    state = "IN_TRADE"
                else:
                    state = "LOOKING"

    trade_cols = ["Entry Time", "Exit Time", "Direction", "Entry", "Stop", "Target", "Exit", "R"]
    trades_df = pd.DataFrame(trades, columns=trade_cols)
    if not trades_df.empty:
        trades_df["Equity"] = trades_df["R"].cumsum()
    return trades_df


def compute_signal(data, p):
    """Compute current signal using the new strategy logic up to the latest bar."""
    df = data.copy()
    df["SMA"] = df["Close"].rolling(p["SMA_PERIOD"]).mean()
    df["SMA_Slope"] = df["SMA"].diff(p["SLOPE_LOOKBACK"])
    df["PrevClose"] = df["Close"].shift(1)
    df["TR"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(abs(df["High"] - df["PrevClose"]), abs(df["Low"] - df["PrevClose"])),
    )
    df["ATR"] = df["TR"].rolling(p["ATR_PERIOD"]).mean()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    state = "LOOKING"
    swing_high = -np.inf
    swing_low = np.inf
    pull_extreme = np.nan
    pullback_bars = 0
    entry = sl = tp = risk = 0
    direction = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        c, h, l, sma, slope, atr = row["Close"], row["High"], row["Low"], row["SMA"], row["SMA_Slope"], row["ATR"]
        pc, psma = prev["Close"], prev["SMA"]

        if state == "IN_TRADE":
            if direction == "LONG":
                if l <= sl or h >= tp:
                    state = "LOOKING"
                    continue
                if h >= entry + (risk * p["BREAKEVEN_AT_R"]):
                    sl = max(sl, entry)
            elif direction == "SHORT":
                if h >= sl or l <= tp:
                    state = "LOOKING"
                    continue
                if l <= entry - (risk * p["BREAKEVEN_AT_R"]):
                    sl = min(sl, entry)
            continue

        if state == "LOOKING":
            if pc < psma and c > sma:
                state = "BULL_IMPULSE"
                swing_high = h
                swing_low = l
            elif pc > psma and c < sma:
                state = "BEAR_IMPULSE"
                swing_high = h
                swing_low = l
        elif state == "BULL_IMPULSE":
            swing_high = max(swing_high, h)
            if l <= sma:
                if slope >= p["SLOPE_THRESHOLD"]:
                    state = "BULL_PULLBACK"
                    pull_extreme = l
                    pullback_bars = 1
                else:
                    state = "LOOKING"
        elif state == "BULL_PULLBACK":
            pullback_bars += 1
            pull_extreme = min(pull_extreme, l)
            if l < swing_low or slope < (p["SLOPE_THRESHOLD"] * p["INVALIDATE_MULT"]):
                state = "LOOKING"
                continue
            if c > sma:
                if pullback_bars >= p["MIN_PULLBACK_BARS"]:
                    entry = c
                    raw_sl = pull_extreme - (atr * p["SL_BUFFER_MULT"])
                    risk = max(entry - raw_sl, p["MIN_RISK_PRICE"])
                    sl = entry - risk
                    tp = entry + (risk * p["RR_TARGET"])
                    direction = "LONG"
                    state = "IN_TRADE"
                else:
                    state = "LOOKING"
        elif state == "BEAR_IMPULSE":
            swing_low = min(swing_low, l)
            if h >= sma:
                if slope <= -p["SLOPE_THRESHOLD"]:
                    state = "BEAR_PULLBACK"
                    pull_extreme = h
                    pullback_bars = 1
                else:
                    state = "LOOKING"
        elif state == "BEAR_PULLBACK":
            pullback_bars += 1
            pull_extreme = max(pull_extreme, h)
            if h > swing_high or slope > -(p["SLOPE_THRESHOLD"] * p["INVALIDATE_MULT"]):
                state = "LOOKING"
                continue
            if c < sma:
                if pullback_bars >= p["MIN_PULLBACK_BARS"]:
                    entry = c
                    raw_sl = pull_extreme + (atr * p["SL_BUFFER_MULT"])
                    risk = max(raw_sl - entry, p["MIN_RISK_PRICE"])
                    sl = entry + risk
                    tp = entry - (risk * p["RR_TARGET"])
                    direction = "SHORT"
                    state = "IN_TRADE"

    sig = "WAIT"
    if state == "IN_TRADE":
        sig = "LONG" if direction == "LONG" else "SHORT"
    current_price = df.iloc[-1]["Close"]
    sma_val = df.iloc[-1]["SMA"]
    return {
        "state": state,
        "signal": sig,
        "current_price": current_price,
        "sma": sma_val,
        "entry": entry,
        "stop": sl,
        "target": tp,
    }


# ==================== METRICS & SIZING ====================
def compute_metrics(trades):
    if trades is None or len(trades) == 0:
        return {
            "Trades": 0, "Wins": 0, "Losses": 0, "Win Rate": 0.0,
            "Average R": 0.0, "Total R": 0.0, "Profit Factor": 0.0, "Max Drawdown (R)": 0.0,
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
        "Trades": len(trades), "Wins": wins, "Losses": losses, "Win Rate": win_rate,
        "Average R": avg_r, "Total R": total_r, "Profit Factor": pf, "Max Drawdown (R)": max_dd,
    }


def format_metrics_for_display(metrics):
    return pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Value": [
            metrics["Trades"], metrics["Wins"], metrics["Losses"],
            f"{metrics['Win Rate']:.2%}", metrics["Average R"], metrics["Total R"],
            metrics["Profit Factor"], metrics["Max Drawdown (R)"],
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


# ==================== CACHED FETCH ====================
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
st.title("SMA-Slope Pullback Strategy Dashboard")
st.markdown("**Strategy:** Impulse + pullback to 50-SMA with slope filter, ATR-scaled stops, and 2R target. Initial account **R100,000**.")

with st.sidebar:
    st.header("Settings")
    instrument = st.selectbox("Instrument", list(SUPPORTED_INSTRUMENTS.keys()))
    scenario = st.selectbox("Scenario", list(PARAMS.keys()))
    interval = st.selectbox("Interval", ["15m", "30m", "1h", "1d"], index=2)
    sma_period = st.number_input("SMA Period", min_value=10, max_value=200, value=50, step=1)
    rr_target = st.number_input("RR Target", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
    initial_capital = st.number_input("Initial Capital (R)", min_value=1000, value=100000, step=1000)
    risk_pct = st.number_input("Risk per Trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    data_source = st.radio("Data Source", ["Live (yfinance)", "Synthetic"])
    run_btn = st.button("Run Backtest")

if run_btn:
    p = dict(PARAMS[scenario])
    p["SMA_PERIOD"] = int(sma_period)
    p["RR_TARGET"] = float(rr_target)

    df = cached_fetch(instrument, interval, data_source)

    if df is None or len(df) == 0:
        st.error("Failed to fetch data. Try another interval or use Synthetic data.")
    else:
        trades = run_backtest(df, p)
        signal = compute_signal(df, p)
        metrics = compute_metrics(trades)

        data_start = df["Time"].min()
        data_end = df["Time"].max()
        st.caption(
            f"Backtest data: **{data_start} → {data_end}** ({len(df)} bars, interval {interval}). "
            f"Current signal derived from latest bar ({data_end})."
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Signal", signal["signal"])
        col2.metric("Current Price", round(signal["current_price"], 4) if signal["current_price"] is not None else "N/A")
        col3.metric("Total Trades", str(metrics["Trades"]))
        col4.metric("Total R", str(round(metrics["Total R"], 2)))

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
