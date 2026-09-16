"""
dashboard.py
============
Streamlit interactive dashboard for the Gold Macro Intelligence application.

Run with:  streamlit run dashboard.py

Pages:
  1. Market Overview
  2. Gold Dashboard
  3. Macro Dashboard
  4. FRED Data Explorer
  5. Signal History
  6. Backtest Results
  7. Economic Regime Monitor
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_loader import load_panel
from feature_engineering import build_features, get_economic_summary
from macro_score import compute_macro_score, classify_score, component_table
from signals import generate_all_signals
from regime import detect_regime, gold_performance_by_regime
from backtest import run_all_backtests, backtest_series

st.set_page_config(page_title="Gold Macro Intelligence", layout="wide")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Loading data...")
def load():
    panel = load_panel(offline=False, force_refresh=False)
    f = build_features(panel)
    f["macro_score"] = compute_macro_score(f)
    sig = generate_all_signals(f)
    f = f.join(sig)
    f["regime"] = detect_regime(f)
    return panel, f

panel, f = load()
latest = f.iloc[-1]

# Sidebar navigation
st.sidebar.title("Gold Macro Intelligence")
page = st.sidebar.radio("Navigate", [
    "Market Overview", "Gold Dashboard", "Macro Dashboard",
    "FRED Data Explorer", "Signal History", "Backtest Results",
    "Economic Regime Monitor",
])

# ---------------------------------------------------------------------------
def _chart(df, y, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[y], mode="lines", name=y))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=y)
    return fig

# ---------------------------------------------------------------------------
if page == "Market Overview":
    st.header("Market Overview")
    st.metric("Current Gold Price", f"${latest.get('GOLD', 0):,.2f}")
    st.metric("Macro Score", f"{latest['macro_score']:.1f}",
              help=classify_score(latest["macro_score"]))
    st.metric("Regime", latest["regime"])
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_chart(f, "GOLD", "Gold Price (GC=F)"), use_container_width=True)
    with c2:
        st.plotly_chart(_chart(f, "macro_score", "Gold Macro Score"), use_container_width=True)

elif page == "Gold Dashboard":
    st.header("Gold Dashboard")
    st.plotly_chart(_chart(f, "GOLD", "Gold Price"), use_container_width=True)
    st.plotly_chart(_chart(f, "gold_ret20", "20-Day Gold Momentum"), use_container_width=True)
    st.plotly_chart(_chart(f, "gold_vol20", "20-Day Volatility"), use_container_width=True)

elif page == "Macro Dashboard":
    st.header("Macro Dashboard")
    st.subheader("Macro Score Components")
    st.dataframe(component_table(f).round(2))
    cols = st.columns(3)
    for i, col in enumerate(["real_yield", "DXY", "CPI_YoY"]):
        if col in f:
            with cols[i]:
                st.plotly_chart(_chart(f, col, col), use_container_width=True)
    st.plotly_chart(_chart(f, "FEDFUNDS", "Fed Funds Rate"), use_container_width=True)

elif page == "FRED Data Explorer":
    st.header("FRED Data Explorer")
    series = st.selectbox("Select series", [c for c in f.columns if c in
        ["DXY","DGS10","CPI","CORE_CPI","CORE_PCE","FEDFUNDS","UNRATE","PAYEMS","INDPRO","real_yield","CPI_YoY"]])
    if series:
        st.plotly_chart(_chart(f, series, series), use_container_width=True)
        econ = get_economic_summary(f, [series])
        st.dataframe(econ)

elif page == "Signal History":
    st.header("Signal History")
    lookback = st.selectbox("Lookback", [30, 90, 180, 365, 730, 3650])
    sub = f.tail(lookback)
    for sig_col, conf_col in [("daily_signal","daily_conf"),("weekly_signal","weekly_conf"),
                              ("monthly_signal","monthly_conf"),("yearly_signal","yearly_conf")]:
        st.subheader(sig_col.replace("_signal","").title())
        st.plotly_chart(_chart(sub, sig_col.replace("_signal","_score"), sig_col), use_container_width=True)
        st.metric("Current", sub[sig_col].iloc[-1], f"Confidence {sub[conf_col].iloc[-1]:.0f}%")

elif page == "Backtest Results":
    st.header("Backtest Results")
    bt = run_all_backtests(panel)
    st.dataframe(bt.round(4))
    horizon = st.selectbox("Horizon", ["1M","3M","6M","12M"])
    bt_series = backtest_series(panel, horizon)
    st.plotly_chart(_chart(bt_series, "macro_score", f"Macro Score ({horizon})"), use_container_width=True)
    st.plotly_chart(_chart(bt_series, "weekly_score", "Weekly Signal Score"), use_container_width=True)

elif page == "Economic Regime Monitor":
    st.header("Economic Regime Monitor")
    st.plotly_chart(_chart(f, "macro_score", "Macro Score"), use_container_width=True)
    st.subheader("Gold Performance by Regime (6M)")
    perf = gold_performance_by_regime(f, horizon=126)
    st.dataframe(perf.round(4))
    st.bar_chart(perf["avg_fwd_return"].dropna())
