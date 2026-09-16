"""
feature_engineering.py
======================
Computes economic calculations and feature engineering for the macro panel.

For every economic series we compute: current value, 1M/3M/6M/12M changes,
z-score, trend direction, acceleration, and momentum. Also builds gold-specific
and regime-related features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import z_score, rolling_z, trend_direction, acceleration, momentum, diff


def compute_economic_calculations(df: pd.DataFrame,
                                  series: list[str],
                                  no_lookahead: bool = False) -> pd.DataFrame:
    """
    For each economic series, compute the requested set of transformations.

    no_lookahead=True uses rolling z-scores (for backtesting); otherwise uses
    full-sample z-scores (for live analysis).
    """
    out = df.copy()
    for col in series:
        if col not in df:
            continue
        s = df[col]
        out[f"{col}_mom1"] = diff(s, 1)
        out[f"{col}_chg3"] = diff(s, 3)
        out[f"{col}_chg6"] = diff(s, 6)
        out[f"{col}_chg12"] = diff(s, 12)
        out[f"{col}_yoy"] = s.pct_change(12) * 100 if s.min() > 0 else diff(s, 12)
        if no_lookahead:
            out[f"{col}_zscore"] = rolling_z(s, window=252, min_obs=60)
        else:
            out[f"{col}_zscore"] = z_score(s)
        out[f"{col}_trend"] = trend_direction(s, 12)
        out[f"{col}_accel"] = acceleration(s, 3)
        out[f"{col}_momentum"] = momentum(s, 12)
    return out


def build_features(panel: pd.DataFrame, no_lookahead: bool = False) -> pd.DataFrame:
    """
    Build the master feature dataframe used by the scoring, signal and backtest
    engines.
    """
    f = panel.copy()

    # --- Derived macro variables -------------------------------------------
    f["CPI_YoY"] = f["CPI"].pct_change(12) * 100
    f["real_yield"] = f["DGS10"] - f["CPI_YoY"]
    f["CORE_PCE_YoY"] = f["CORE_PCE"].pct_change(12) * 100
    f["CORE_CPI_YoY"] = f["CORE_CPI"].pct_change(12) * 100
    f["NFP_change"] = f["PAYEMS"].diff(1)
    f["fed_real"] = f["FEDFUNDS"] - f["CPI_YoY"]
    f["INDPRO_YoY"] = f["INDPRO"].pct_change(12) * 100

    # --- Gold / market features -------------------------------------------
    if "GOLD" in f:
        f["gold_ret"] = f["GOLD"].pct_change()
        f["gold_ret20"] = f["GOLD"].pct_change(20)
        f["gold_ret60"] = f["GOLD"].pct_change(60)
        f["gold_sma20"] = f["GOLD"].rolling(20).mean()
        f["gold_sma50"] = f["GOLD"].rolling(50).mean()
        f["gold_sma200"] = f["GOLD"].rolling(200).mean()
        f["gold_atr14"] = f["GOLD"].diff().abs().rolling(14).mean()
        f["gold_vol20"] = f["gold_ret"].rolling(20).std()

    if "DXY" in f:
        f["dxy_ret"] = f["DXY"].pct_change()
        f["dxy_ret60"] = f["DXY"].pct_change(60)

    if "DGS10" in f:
        f["yield_chg3"] = f["DGS10"].diff(3)
        f["real_yield_chg3"] = f["real_yield"].diff(3)
        f["yield_trend"] = trend_direction(f["DGS10"], 60)

    if "TLT" in f:
        f["tlt_ret"] = f["TLT"].pct_change()

    # --- Regime indicators --------------------------------------------------
    f["risk_off"] = (f["gold_vol20"] > f["gold_vol20"].rolling(252).median()).astype(float)
    f["inflation_rising"] = (f["CPI_YoY"] > f["CPI_YoY"].rolling(12).mean()).astype(float)
    f["growth_slowing"] = (f["INDPRO_YoY"] < f["INDPRO_YoY"].rolling(12).mean()).astype(float)
    f["fed_easing"] = (f["FEDFUNDS"] < f["FEDFUNDS"].rolling(6).mean()).astype(float)

    return f


def get_economic_summary(f: pd.DataFrame, series: list[str]) -> pd.DataFrame:
    """
    Return the latest economic calculations table for the report (current,
    MoM, 3M, 6M, 12M, z-score, trend, acceleration, momentum).

    Self-contained: computes changes directly from each series so it does not
    depend on pre-computed columns.
    """
    rows = []
    for col in series:
        if col not in f:
            continue
        s = f[col].dropna()
        if s.empty:
            continue
        rows.append({
            "Series": col,
            "Current": s.iloc[-1],
            "1M": s.diff(1).iloc[-1],
            "3M": s.diff(3).iloc[-1],
            "6M": s.diff(6).iloc[-1],
            "12M": s.diff(12).iloc[-1],
            "Z-Score": z_score(s).iloc[-1],
            "Trend": trend_direction(s, 12).iloc[-1],
            "Accel": acceleration(s, 3).iloc[-1],
            "Momentum": momentum(s, 12).iloc[-1],
        })
    return pd.DataFrame(rows)
