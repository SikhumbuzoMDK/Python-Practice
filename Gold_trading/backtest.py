"""
backtest.py
===========
Walk-forward backtest engine.

For each historical date (from BACKTEST_START), it computes the macro score and
signals using ONLY data available up to that date (rolling z-scores, no
lookahead), then measures future Gold returns over 1M/3M/6M/12M. It evaluates
win rate, average return, Sharpe, max drawdown, profit factor, precision,
recall and accuracy.

Also builds strategy equity curves per signal timeframe (period).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import BACKTEST_START, FORWARD_HORIZONS
from feature_engineering import build_features
from macro_score import compute_macro_score
from signals import generate_all_signals
from utils import max_drawdown, sharpe_ratio, profit_factor


def _classification_metrics(signal_bool: pd.Series, positive: pd.Series) -> dict:
    """Accuracy / precision / recall treating 'signal on' as the positive class."""
    df = pd.DataFrame({"pred": signal_bool.astype(float),
                       "label": positive.astype(float)}).dropna()
    if df.empty:
        return {"accuracy": np.nan, "precision": np.nan, "recall": np.nan}
    pred = df["pred"].values
    lab = df["label"].values
    acc = (pred == lab).mean()
    tp = ((pred == 1) & (lab == 1)).sum()
    fp = ((pred == 1) & (lab == 0)).sum()
    fn = ((pred == 0) & (lab == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    return {"accuracy": acc, "precision": prec, "recall": rec}


def run_backtest(panel: pd.DataFrame, horizon_label: str = "3M") -> dict:
    """
    Run the walk-forward backtest on a chosen horizon.

    horizon_label: one of '1M', '3M', '6M', '12M'.
    """
    horizon = FORWARD_HORIZONS[horizon_label]
    f = build_features(panel, no_lookahead=True)

    # Compute macro score and signals with rolling z (no lookahead).
    f["macro_score"] = compute_macro_score(f, no_lookahead=True)
    sig = generate_all_signals(f, no_lookahead=True)
    f = f.join(sig)

    # Future gold return over the horizon.
    f["fwd_ret"] = f["GOLD"].shift(-horizon) / f["GOLD"] - 1

    # Use the weekly signal as the primary trading signal for the backtest.
    f["signal_on"] = f["weekly_score"] >= 20  # Buy / Strong Buy

    # Trading metrics: follow the signal (long when signal_on, flat otherwise).
    f["strat_ret"] = np.where(f["signal_on"].shift(1).fillna(False), f["gold_ret"], 0.0)
    equity = (1 + f["strat_ret"].fillna(0)).cumprod()

    # Subset to backtest period with sufficient warmup.
    bt = f.loc[f.index >= pd.Timestamp(BACKTEST_START) + pd.Timedelta(days=365)]
    bt = bt.dropna(subset=["fwd_ret"])

    signal_returns = bt.loc[bt["signal_on"], "fwd_ret"].dropna()
    metrics = {
        "horizon": horizon_label,
        "n_signal_days": int(len(signal_returns)),
        "win_rate": float((signal_returns > 0).mean()) if len(signal_returns) else np.nan,
        "avg_return": float(signal_returns.mean()) if len(signal_returns) else np.nan,
        "median_return": float(signal_returns.median()) if len(signal_returns) else np.nan,
        "sharpe": sharpe_ratio(f["strat_ret"].dropna()),
        "max_drawdown": max_drawdown(equity),
        "profit_factor": profit_factor(signal_returns),
    }
    metrics.update(_classification_metrics(bt["signal_on"], bt["fwd_ret"] > 0))
    return metrics


def run_all_backtests(panel: pd.DataFrame) -> pd.DataFrame:
    """Run backtests for all horizons and return a summary DataFrame."""
    rows = []
    for label in FORWARD_HORIZONS:
        rows.append(run_backtest(panel, label))
    return pd.DataFrame(rows)


def backtest_series(panel: pd.DataFrame, horizon_label: str = "3M") -> pd.DataFrame:
    """
    Return a time series of the backtest signal, score, and forward returns for
    plotting/dashboard use.
    """
    horizon = FORWARD_HORIZONS[horizon_label]
    f = build_features(panel, no_lookahead=True)
    f["macro_score"] = compute_macro_score(f, no_lookahead=True)
    sig = generate_all_signals(f, no_lookahead=True)
    f = f.join(sig)
    f["fwd_ret"] = f["GOLD"].shift(-horizon) / f["GOLD"] - 1
    return f


# ===========================================================================
# ADDED: Period Equity Curves
# ===========================================================================
def period_equity_curves(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Build strategy equity curves for each signal timeframe (period).

    For each timeframe (daily/weekly/monthly/yearly), the strategy is long
    when that timeframe's signal score >= 20 (Buy/Strong Buy) and flat
    otherwise, with positions entered on the NEXT bar to avoid lookahead.
    A buy-and-hold gold reference is included for comparison.

    Returns a DataFrame indexed by date with columns:
        daily, weekly, monthly, yearly, buy_hold
    """
    f = build_features(panel, no_lookahead=True)
    f["macro_score"] = compute_macro_score(f, no_lookahead=True)
    sig = generate_all_signals(f, no_lookahead=True)
    f = f.join(sig)

    curves = {}
    for score_col in ["daily_score", "weekly_score", "monthly_score", "yearly_score"]:
        f["signal_on"] = f[score_col] >= 20
        f["strat_ret"] = np.where(f["signal_on"].shift(1).fillna(False), f["gold_ret"], 0.0)
        curves[score_col.replace("_score", "")] = (1 + f["strat_ret"].fillna(0)).cumprod()

    # Buy-and-hold gold reference
    curves["buy_hold"] = (1 + f["gold_ret"].fillna(0)).cumprod()
    return pd.DataFrame(curves)


def equity_drawdown_series(equity: pd.Series) -> pd.Series:
    """Return the drawdown series (equity / cumulative max - 1)."""
    return equity / equity.cummax() - 1


def equity_summary(curves: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise each equity curve: final equity, max drawdown, CAGR, Sharpe.
    """
    rows = []
    for col in curves.columns:
        eq = curves[col].dropna()
        ret = eq.pct_change().dropna()
        rows.append({
            "Strategy": col,
            "Final_Equity": eq.iloc[-1],
            "Max_Drawdown": (eq / eq.cummax() - 1).min(),
            "CAGR": (eq.iloc[-1] ** (252 / len(eq)) - 1) if len(eq) else np.nan,
            "Sharpe": sharpe_ratio(ret),
        })
    return pd.DataFrame(rows)
