"""
utils.py
========
Statistic and transformation helpers shared across the application.

Includes z-score / rolling z-score, trend direction, acceleration, momentum,
safe division, and threshold-based classification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def z_score(series: pd.Series) -> pd.Series:
    """Standard z-score over the full sample (for analysis, NOT backtest)."""
    return (series - series.mean()) / series.std()


def rolling_z(series: pd.Series, window: int = 252, min_obs: int = 60) -> pd.Series:
    """
    Rolling z-score that uses only past data (no lookahead).

    Used for the walk-forward backtest to avoid lookahead bias.
    """
    mu = series.rolling(window, min_periods=min_obs).mean()
    sd = series.rolling(window, min_periods=min_obs).std()
    return (series - mu) / sd.replace(0, np.nan)


def pct_change(series: pd.Series, period: int) -> pd.Series:
    """Percentage change over `period` observations."""
    return series.pct_change(period)


def diff(series: pd.Series, period: int) -> pd.Series:
    """Absolute difference over `period` observations."""
    return series.diff(period)


def trend_direction(series: pd.Series, window: int = 12) -> pd.Series:
    """
    Trend direction: +1 if current > rolling mean, -1 if below, 0 if flat.
    """
    mean = series.rolling(window, min_periods=max(3, window // 2)).mean()
    return np.sign(series - mean).fillna(0)


def acceleration(series: pd.Series, period: int = 3) -> pd.Series:
    """Second difference (change in change) — indicates accelerating momentum."""
    return series.diff(period).diff(period)


def momentum(series: pd.Series, period: int = 12) -> pd.Series:
    """Momentum = series / (series shifted by period) - 1."""
    return series / series.shift(period) - 1


def classify(value: float, thresholds) -> str:
    """
    Map a numeric value to a label using an ordered list of (lower_bound, label).
    Returns the label whose lower_bound is the largest <= value.
    """
    label = thresholds[-1][1]
    for lower, lbl in sorted(thresholds, key=lambda x: x[0]):
        if value >= lower:
            label = lbl
    return label


def safe_div(numerator, denominator, default=np.nan):
    """Safe division that avoids ZeroDivisionError."""
    denominator = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numerator / denominator
    return np.where(np.isfinite(out), out, default)


def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown of an equity/price series."""
    if equity.empty:
        return 0.0
    cummax = equity.cummax()
    dd = equity / cummax - 1
    return float(dd.min())


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio of a returns series."""
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def profit_factor(returns: pd.Series) -> float:
    """Gross profit / gross loss. Inf if no losses."""
    r = returns.dropna()
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)
