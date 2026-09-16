"""
signals.py
==========
Generates Daily, Weekly, Monthly and Yearly Gold signals with confidence scores.

Each signal is produced by a weighted combination of continuous (tanh-scaled)
sub-signals, producing a numeric score in [-100, 100] that is mapped to
Strong Buy / Buy / Neutral / Sell / Strong Sell.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import SIGNAL_THRESHOLDS
from utils import classify, rolling_z, z_score


def _tanh(x: pd.Series, scale: float = 1.0) -> pd.Series:
    """Squash a feature into [-1, 1] then scale by `scale`."""
    return np.tanh(x) * scale


def _clean(s: pd.Series) -> pd.Series:
    return s.replace([np.inf, -np.inf], np.nan).fillna(0)


def daily_signal(f: pd.DataFrame, no_lookahead: bool = False) -> pd.DataFrame:
    """
    Daily signal uses: price momentum, ATR/volatility, trend structure, and
    recent macro-score changes.
    """
    g = f.copy()
    mom = z_score(g["gold_ret20"]) if not no_lookahead else rolling_z(g["gold_ret20"], 252, 60)
    trend = ((g["GOLD"] - g["gold_sma50"]) / g["gold_sma50"])
    ltrend = ((g["gold_sma50"] - g["gold_sma200"]) / g["gold_sma200"])
    macro_chg = z_score(g["macro_score"].diff(20)) if "macro_score" in g else pd.Series(0, index=g.index)

    score = (
        40 * _tanh(_clean(mom))
        + 25 * _tanh(_clean(trend), 5)
        + 15 * _tanh(_clean(ltrend), 5)
        + 20 * _tanh(_clean(macro_chg))
    )
    # Volatility filter reduces confidence (not direction)
    vol_penalty = np.clip(1 - g["gold_vol20"] / g["gold_vol20"].rolling(252).median().replace(0, np.nan), 0.3, 1.0)
    conf = np.clip(50 + np.abs(score), 50, 100) * vol_penalty

    out = pd.DataFrame(index=g.index)
    out["daily_score"] = score
    out["daily_signal"] = score.apply(lambda v: classify(v, SIGNAL_THRESHOLDS))
    out["daily_conf"] = conf
    return out


def weekly_signal(f: pd.DataFrame, no_lookahead: bool = False) -> pd.DataFrame:
    """Weekly signal uses trend structure, macro score, yield trend, DXY trend."""
    g = f.copy()
    tr_struct = (g["GOLD"] - g["gold_sma200"]) / g["gold_sma200"]
    macro_lvl = z_score(g["macro_score"]) if not no_lookahead else rolling_z(g["macro_score"], 252, 60)
    yield_t = g["yield_chg3"].rolling(5).mean()
    dxy_t = g["dxy_ret60"].rolling(5).mean()

    score = (
        30 * _tanh(_clean(tr_struct), 5)
        + 20 * _tanh(_clean(macro_lvl))
        + 25 * _tanh(_clean(-yield_t), 5)
        + 25 * _tanh(_clean(-dxy_t), 5)
    )
    conf = np.clip(50 + np.abs(score), 50, 100)
    out = pd.DataFrame(index=g.index)
    out["weekly_score"] = score
    out["weekly_signal"] = score.apply(lambda v: classify(v, SIGNAL_THRESHOLDS))
    out["weekly_conf"] = conf
    return out


def monthly_signal(f: pd.DataFrame, no_lookahead: bool = False) -> pd.DataFrame:
    """Monthly signal uses inflation regime, Fed regime, ISM trend, real yields."""
    g = f.copy()
    inf_r = g["CPI_YoY"].diff(12)
    fed_r = g["FEDFUNDS"].diff(6)
    ism_t = (g["INDPRO"].pct_change(12) * 100).diff(6)
    real_lvl = z_score(g["real_yield"]) if not no_lookahead else rolling_z(g["real_yield"], 252, 60)

    score = (
        20 * _tanh(_clean(inf_r))
        + 25 * _tanh(_clean(-fed_r))
        + 20 * _tanh(_clean(-ism_t))
        + 35 * _tanh(_clean(-real_lvl))
    )
    conf = np.clip(50 + np.abs(score), 50, 100)
    out = pd.DataFrame(index=g.index)
    out["monthly_score"] = score
    out["monthly_signal"] = score.apply(lambda v: classify(v, SIGNAL_THRESHOLDS))
    out["monthly_conf"] = conf
    return out


def yearly_signal(f: pd.DataFrame, no_lookahead: bool = False) -> pd.DataFrame:
    """Yearly signal uses macro cycle, growth cycle, inflation cycle, policy cycle."""
    g = f.copy()
    growth_cyc = (g["INDPRO"].pct_change(12) * 100).diff(24)
    inf_cyc = g["CPI_YoY"].diff(24)
    pol_cyc = g["FEDFUNDS"].diff(24)
    real_lvl = z_score(g["real_yield"]) if not no_lookahead else rolling_z(g["real_yield"], 252, 60)

    score = (
        30 * _tanh(_clean(-growth_cyc))
        + 25 * _tanh(_clean(inf_cyc))
        + 25 * _tanh(_clean(-pol_cyc))
        + 20 * _tanh(_clean(-real_lvl))
    )
    conf = np.clip(50 + np.abs(score), 50, 100)
    out = pd.DataFrame(index=g.index)
    out["yearly_score"] = score
    out["yearly_signal"] = score.apply(lambda v: classify(v, SIGNAL_THRESHOLDS))
    out["yearly_conf"] = conf
    return out


def generate_all_signals(f: pd.DataFrame, no_lookahead: bool = False) -> pd.DataFrame:
    """Generate all four signal frames and concatenate them."""
    parts = [daily_signal(f, no_lookahead),
             weekly_signal(f, no_lookahead),
             monthly_signal(f, no_lookahead),
             yearly_signal(f, no_lookahead)]
    return pd.concat(parts, axis=1)
