"""
regime.py
=========
Detects the prevailing macro/market regime and measures how gold has
historically performed in each regime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import rolling_z


REGIMES = [
    "Risk-On",
    "Risk-Off",
    "Inflationary Expansion",
    "Deflationary Slowdown",
    "Stagflation",
    "Recession",
    "Recovery",
]


def detect_regime(f: pd.DataFrame) -> pd.Series:
    """
    Classify each day into one of the defined regimes.

    Uses:
      - Growth (INDPRO YoY)
      - Inflation (CPI YoY)
      - Policy (Fed Funds vs its rolling mean)
      - Risk sentiment (gold volatility / TLT trend)
    """
    g = f.copy()
    growth = g["INDPRO_YoY"] if "INDPRO_YoY" in g else g["INDPRO"].pct_change(12) * 100
    inflation = g["CPI_YoY"]
    fed = g["FEDFUNDS"]

    growth_up = growth > growth.rolling(12).mean()
    inf_up = inflation > inflation.rolling(12).mean()
    fed_easing = fed < fed.rolling(6).mean()

    # Risk sentiment proxy: gold volatility high + TLT up => risk-off
    risk_off = g["gold_vol20"] > g["gold_vol20"].rolling(252).median()

    regime = np.where(
        growth_up & inf_up & ~fed_easing, "Inflationary Expansion",
        np.where(
            growth_up & inf_up & fed_easing, "Recovery",
            np.where(
                ~growth_up & inf_up & ~fed_easing, "Stagflation",
                np.where(
                    ~growth_up & inf_up & fed_easing, "Risk-Off",
                    np.where(
                        ~growth_up & ~inf_up & fed_easing, "Recession",
                        np.where(
                            ~growth_up & ~inf_up & ~fed_easing, "Deflationary Slowdown",
                            np.where(
                                growth_up & ~inf_up, "Recovery",
                                "Risk-On",
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    # Overlay risk sentiment: if risk_off, mark as Risk-Off unless stagflation.
    regime = np.where(risk_off & (regime == "Risk-On"), "Risk-Off", regime)
    return pd.Series(regime, index=g.index)


def gold_performance_by_regime(f: pd.DataFrame, horizon: int = 63) -> pd.DataFrame:
    """
    Compute average forward returns of gold conditional on the regime at the
    time of the signal.
    """
    f = f.copy()
    f["regime"] = detect_regime(f)
    f[f"fwd_{horizon}"] = f["GOLD"].shift(-horizon) / f["GOLD"] - 1
    grp = f.groupby("regime")[f"fwd_{horizon}"].agg(["mean", "median", "count"])
    grp.columns = ["avg_fwd_return", "median_fwd_return", "count"]
    return grp
