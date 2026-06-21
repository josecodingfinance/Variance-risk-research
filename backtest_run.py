#!/usr/bin/env python
"""
Backtest runner: long-gamma variance swap on SPX+VIX (2015-present).

Executes:
  source .venv/bin/activate && python backtest_run.py
"""

import logging
import pandas as pd

from src.data import load_data
from src.backtest import run_backtest
from src.risk import (
    cagr,
    annualized_vol,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    worst_day,
    worst_week,
    skewness,
    excess_kurtosis,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run backtest and print metrics."""
    logger.info("Loading SPX + VIX data (2015-01-01 to 2024-12-31)")
    spx_src, vix_src = load_data("2015-01-01", "2024-12-31")

    if spx_src.is_synthetic or vix_src.is_synthetic:
        logger.warning("⚠️ WARNING: Using synthetic data — not for real trading")

    logger.info(f"SPX: {len(spx_src.df)} trading days")
    logger.info(f"VIX: {len(vix_src.df)} trading days")

    # Run backtest: long gamma, $100k variance notional, 0.5 vol points bid-ask
    logger.info("Running backtest: Long Gamma, $100k notional, 0.5 vol points bid-ask")
    result = run_backtest(spx_src, vix_src, side=1, notional=100000, bid_ask_vol_points=0.5)

    pnl_series = result["pnl_series"].dropna()
    cumulative = result["cumulative"]
    pnl_gross = result["pnl_gross"].dropna()
    roll_costs = result["roll_costs"].dropna()

    logger.info(f"Trading days in backtest: {len(pnl_series)}")
    logger.info(f"Start date: {pnl_series.index[0]}")
    logger.info(f"End date: {pnl_series.index[-1]}")

    # Calculate years
    years = (pnl_series.index[-1] - pnl_series.index[0]).days / 365.25
    initial_capital = 100000

    # Compute metrics (with capital)
    metrics = {
        "Total P&L Gross ($)": pnl_gross.sum(),
        "Total P&L Net ($)": pnl_series.sum(),
        "Total Roll Costs ($)": roll_costs.sum(),
        "CAGR (%)": cagr(cumulative, years=years, initial_capital=initial_capital) * 100,
        "Annualized Vol (%)": annualized_vol(pnl_series) * 100,
        "Sharpe Ratio": sharpe_ratio(pnl_series),
        "Sortino Ratio": sortino_ratio(pnl_series),
        "Max Drawdown (%)": max_drawdown(cumulative, initial_capital=initial_capital) * 100,
        "Calmar Ratio": calmar_ratio(cumulative, years=years, initial_capital=initial_capital),
        "Worst Day ($)": worst_day(pnl_series),
        "Worst Week ($)": worst_week(pnl_series),
        "Skewness": skewness(pnl_series),
        "Excess Kurtosis": excess_kurtosis(pnl_series),
    }

    # Print report
    print("\n" + "=" * 60)
    print("GAMMA CAPTURE FUND — BACKTEST METRICS (2015-2024)")
    print("=" * 60)
    print(f"Strategy:      Long Gamma (Variance Swap)")
    print(f"Initial Capital: ${initial_capital:,.0f}")
    print(f"Notional:      $100,000 per point of variance")
    print(f"Bid-Ask:       0.5 vol points (monthly roll)")
    print(f"Period:        {years:.1f} years")
    print(f"Trading Days:  {len(pnl_series)}")
    print("=" * 60)

    for key, value in metrics.items():
        if "%" in key:
            print(f"{key:.<40} {value:>12.2f}%")
        else:
            print(f"{key:.<40} {value:>12,.2f}")

    print("=" * 60)
    print(f"Data Source:   {'SYNTHETIC' if spx_src.is_synthetic else 'REAL (yfinance)'}")
    print("=" * 60)

    # Tail analysis
    print("\nTAIL ANALYSIS:")
    print(f"  5th percentile P&L:  ${pnl_series.quantile(0.05):>10,.2f}")
    print(f"  95th percentile P&L: ${pnl_series.quantile(0.95):>10,.2f}")
    print(f"  % Positive Days:     {(pnl_series > 0).sum() / len(pnl_series) * 100:>10.1f}%")
    print(f"  Mean Daily P&L:      ${pnl_series.mean():>10,.2f}")
    print(f"  Median Daily P&L:    ${pnl_series.median():>10,.2f}")

    # Cost impact
    print("\nCOST IMPACT (Bruto vs Neto):")
    print(f"  Total Gross P&L:     ${pnl_gross.sum():>12,.2f}")
    print(f"  Total Roll Costs:    ${roll_costs.sum():>12,.2f}")
    print(f"  Total Net P&L:       ${pnl_series.sum():>12,.2f}")
    cost_pct = abs(roll_costs.sum()) / abs(pnl_gross.sum()) * 100 if pnl_gross.sum() != 0 else 0
    print(f"  Costs as % of Gross: {cost_pct:>12.2f}%")

    # Roll schedule summary
    roll_dates = [d for d in roll_costs.index if roll_costs.loc[d] != 0]
    print(f"\n  Roll Events: {len(roll_dates)} months")
    if len(roll_dates) > 0:
        print(f"  Avg Cost per Roll:   ${roll_costs[roll_costs > 0].mean():>12,.2f}")
        print(f"  First Roll: {roll_dates[0].strftime('%Y-%m-%d')}")
        print(f"  Last Roll:  {roll_dates[-1].strftime('%Y-%m-%d')}")
    print()


if __name__ == "__main__":
    main()
