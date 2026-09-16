"""
app.py
======
Entry point for the Gold Macro Intelligence application.

Two modes, auto-detected:
  1. Streamlit app (deployment main module) -> renders the interactive dashboard.
  2. CLI (python app.py)                  -> generates the text report.

CLI usage:
    python app.py                 # generate report (uses cache, no refresh)
    python app.py --update        # force-refresh FRED + Yahoo data
    python app.py --offline       # use cached/synthetic data
    python app.py --backtest      # also run & print backtest metrics

Dashboard usage (deployment / manual):
    streamlit run app.py          # renders the interactive dashboard
"""
from __future__ import annotations

import argparse
import logging
from datetime import date

from config import REPORT_DIR


def _is_streamlit() -> bool:
    """Return True when the script is running under a Streamlit server."""
    try:
        from streamlit.runtime import exists
        return bool(exists())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Streamlit dashboard path
# ---------------------------------------------------------------------------
def _render_dashboard() -> None:
    from dashboard import render
    render()


# ---------------------------------------------------------------------------
# CLI report path
# ---------------------------------------------------------------------------
def _fmt_signal(v: str) -> str:
    return v.upper()


def _explain(f, comps, last) -> str:
    """Generate a plain-language explanation of the current signals."""
    parts = []
    score = last["macro_score"]
    parts.append(f"The Gold Macro Score is {score:.0f}, which is "
                 f"{classify_score(score)}. A higher score indicates "
                 f"conditions historically favourable to gold.")

    top = comps.head(3)
    for key, row in top.iterrows():
        parts.append(f"- The largest driver is '{key}' (weight {row['weight']:.0%}) "
                     f"scoring {row['score']:.0f}, contributing "
                     f"{row['contribution']:.1f} points to the total.")

    parts.append(f"The current regime is classified as {last['regime']}.")

    if last["daily_score"] >= 20:
        parts.append("Daily signal is bullish because gold price momentum and trend "
                     "structure are positive, with supportive recent macro-score change.")
    elif last["daily_score"] <= -20:
        parts.append("Daily signal is bearish because short-term momentum and trend are "
                     "negative, and recent macro change is unfavourable.")
    else:
        parts.append("Daily signal is neutral: momentum and trend are mixed.")

    if last["weekly_score"] >= 20:
        parts.append("Weekly signal is bullish driven by a positive trend structure, "
                     "elevated macro score, falling yields and a weakening dollar.")
    elif last["weekly_score"] <= -20:
        parts.append("Weekly signal is bearish driven by a negative trend structure, "
                     "lower macro score, rising yields and a strengthening dollar.")

    if last["monthly_score"] >= 20:
        parts.append("Monthly signal is bullish given an inflationary/fed-easing regime, "
                     "softening ISM and declining real yields.")
    elif last["monthly_score"] <= -20:
        parts.append("Monthly signal is bearish given a hawkish Fed, rising real yields "
                     "and firm manufacturing momentum.")

    if last["yearly_score"] >= 20:
        parts.append("Yearly signal is bullish reflecting an expansionary macro cycle, "
                     "rising inflation and an easing policy cycle.")
    elif last["yearly_score"] <= -20:
        parts.append("Yearly signal is bearish reflecting a tightening policy cycle, "
                     "disinflation and slowing growth.")
    return "\n".join(parts)


def generate_report(panel, include_backtest: bool = False) -> str:
    """Produce the mandated GOLD MACRO INTELLIGENCE REPORT text."""
    from feature_engineering import build_features
    from macro_score import compute_macro_score, classify_score, component_table
    from signals import generate_all_signals
    from regime import detect_regime
    from backtest import run_all_backtests

    f = build_features(panel)
    f["macro_score"] = compute_macro_score(f)
    sig = generate_all_signals(f)
    f = f.join(sig)
    f["regime"] = detect_regime(f)
    last = f.iloc[-1]

    lines = []
    lines.append("=" * 55)
    lines.append("GOLD MACRO INTELLIGENCE REPORT")
    lines.append("=" * 55)
    lines.append(f"Date:              {last.name.date()}")
    lines.append(f"Current Gold Price: ${last.get('GOLD', float('nan')):,.2f}")
    lines.append(f"Macro Score:       {last['macro_score']:.1f}  ({classify_score(last['macro_score'])})")
    lines.append(f"Current Regime:    {last['regime']}")
    lines.append("")

    lines.append("=" * 55)
    lines.append("MACRO ANALYSIS")
    lines.append("=" * 55)
    comps = component_table(f)
    comp_labels = {
        "real_yield": "Real Yields", "dxy": "DXY", "fed": "Fed Policy",
        "inflation": "Inflation", "labor": "Labor Market", "ism": "ISM",
    }
    for key, label in comp_labels.items():
        if key in comps.index:
            score = comps.loc[key, "score"]
            stance = "Bullish" if score >= 60 else ("Bearish" if score <= 40 else "Neutral")
            lines.append(f"{label:15s}: {stance}  (score {score:.0f})")
    lines.append("")

    lines.append("=" * 55)
    lines.append("SIGNALS")
    lines.append("=" * 55)
    for sig_col in ["daily_signal", "weekly_signal", "monthly_signal", "yearly_signal"]:
        lines.append(f"{sig_col.replace('_signal', '').title():15s}: {_fmt_signal(last[sig_col])}")
    lines.append("")

    lines.append("=" * 55)
    lines.append("CONFIDENCE")
    lines.append("=" * 55)
    for conf_col in ["daily_conf", "weekly_conf", "monthly_conf", "yearly_conf"]:
        lines.append(f"{conf_col.replace('_conf', '').title():15s}: {last[conf_col]:.0f}%")
    lines.append("")

    if include_backtest:
        bt = run_all_backtests(panel)
        best = bt.loc[bt["horizon"] == "3M"].iloc[0]
        lines.append("=" * 55)
        lines.append("BACKTEST PERFORMANCE  (3M horizon)")
        lines.append("=" * 55)
        lines.append(f"Accuracy:      {best['accuracy']:.1%}")
        lines.append(f"Win Rate:      {best['win_rate']:.1%}")
        lines.append(f"Sharpe Ratio:  {best['sharpe']:.2f}")
        lines.append(f"Average Return:{best['avg_return']:.2%}")
        lines.append(f"Max Drawdown:  {best['max_drawdown']:.1%}")
        lines.append("")

    lines.append("=" * 55)
    lines.append("EXPLANATION")
    lines.append("=" * 55)
    lines.append(_explain(f, comps, last))
    lines.append("")
    lines.append("DISCLAIMER: This is a research & decision-support tool. Not investment advice.")
    return "\n".join(lines)


def _cli_main() -> None:
    import logging
    from data_loader import load_panel

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Gold Macro Intelligence application")
    parser.add_argument("--update", action="store_true", help="Force-refresh FRED + Yahoo data")
    parser.add_argument("--offline", action="store_true", help="Use cached/synthetic data")
    parser.add_argument("--backtest", action="store_true", help="Include backtest performance in report")
    args = parser.parse_args()

    logger.info("Loading data (update=%s, offline=%s)...", args.update, args.offline)
    panel = load_panel(offline=args.offline, force_refresh=args.update)
    logger.info("Panel shape: %s rows x %s cols", panel.shape[0], panel.shape[1])

    report = generate_report(panel, include_backtest=args.backtest)
    print(report)

    out = REPORT_DIR / f"gold_macro_report_{date.today().isoformat()}.txt"
    out.write_text(report)
    logger.info("Report written to %s", out)


if _is_streamlit():
    _render_dashboard()
else:
    _cli_main()
