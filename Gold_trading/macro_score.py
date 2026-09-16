"""
macro_score.py
==============
Computes the Gold Macro Score (0-100) from six weighted fundamental drivers.

Each component is scored on a 0-100 'bullishness-for-gold' scale using the
rolling z-score of the relevant directional change. Weights are defined in
config.MACRO_WEIGHTS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import MACRO_WEIGHTS, SCORE_CLASSIFICATION
from utils import rolling_z, z_score, classify


def _component(directional_change: pd.Series, no_lookahead: bool = False) -> pd.Series:
    """
    Convert a directional change z-score to a 0-100 bullishness score.
    Positive directional_change => bullish gold => score > 50.
    """
    if no_lookahead:
        z = rolling_z(directional_change, 252, 60)
    else:
        z = z_score(directional_change)
    return np.clip(50 + 25 * z, 0, 100)


def compute_components(f: pd.DataFrame, no_lookahead: bool = False) -> pd.DataFrame:
    """Return a DataFrame of the six component scores (0-100 each)."""
    comps = pd.DataFrame(index=f.index)

    # Real Yield: rising real yield is bearish gold -> invert the change.
    comps["real_yield"] = _component(-f["real_yield"].diff(3), no_lookahead)

    # DXY: rising dollar is bearish gold -> invert the change.
    comps["dxy"] = _component(-f["DXY"].diff(3), no_lookahead)

    # Fed: rising Fed Funds is bearish gold -> invert the change.
    comps["fed"] = _component(-f["FEDFUNDS"].diff(3), no_lookahead)

    # Inflation: rising inflation is bullish gold (hedge).
    comps["inflation"] = _component(f["CPI_YoY"].diff(3), no_lookahead)

    # Labor: rising unemployment is bullish gold (Fed-easing expectations).
    comps["labor"] = _component(f["UNRATE"].diff(3), no_lookahead)

    # ISM: falling/contracting manufacturing is bullish gold (stagflation hedge).
    # We proxy ISM with industrial production growth momentum.
    ism_proxy = f["INDPRO"].pct_change(12) * 100
    comps["ism"] = _component(-ism_proxy.diff(3), no_lookahead)

    return comps


def compute_macro_score(f: pd.DataFrame, no_lookahead: bool = False) -> pd.Series:
    """
    Compute the weighted Gold Macro Score (0-100).

    Returns a Series aligned to the feature index.
    """
    comps = compute_components(f, no_lookahead)
    score = sum(MACRO_WEIGHTS[k] * comps[k] for k in MACRO_WEIGHTS)
    score = score.clip(0, 100)
    score.name = "macro_score"
    return score


def classify_score(score: float) -> str:
    """Map a macro score to a human-readable classification."""
    return classify(score, SCORE_CLASSIFICATION)


def component_table(f: pd.DataFrame, no_lookahead: bool = False) -> pd.DataFrame:
    """Return the latest component scores and weights for reporting."""
    comps = compute_components(f, no_lookahead).iloc[[-1]].T
    comps.columns = ["score"]
    comps["weight"] = pd.Series(MACRO_WEIGHTS)
    comps["contribution"] = comps["score"] * comps["weight"]
    comps = comps.sort_values("contribution", ascending=False)
    return comps
