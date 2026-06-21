#!/usr/bin/env python
"""
Evaluate HMM regime overlay strategy vs static baselines.

Compares:
1. HMM overlay (dynamic: short in calm, long in turbulent)
2. Static short gamma (always short)
3. Static long gamma (always long)

All out-of-sample, net of roll costs.
"""

import logging
import pandas as pd
import numpy as np
from src.data import load_data
from src.strategy import evaluate_regime_strategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run regime strategy evaluation."""
    logger.info("Loading SPX + VIX data (2015-01-01 to 2024-12-31)")
    spx_src, vix_src = load_data("2015-01-01", "2024-12-31")

    logger.info("Evaluating regime overlay strategy (walk-forward, annual refit)...")
    result = evaluate_regime_strategy(
        spx_src,
        vix_src,
        initial_capital=100000,
        bid_ask_vol_points=0.5,
        refit_freq="Y",
    )

    # Print results
    print("\n" + "=" * 80)
    print("REGIME OVERLAY STRATEGY — COMPARISON")
    print("=" * 80)
    print()

    print("PERFORMANCE METRICS (Out-of-Sample, Net of Roll Costs)")
    print("-" * 80)
    print(result["metrics"].to_string())
    print()

    print("CRASH ANALYSIS: Regime Before Key Vol Events")
    print("-" * 80)
    for date_str, analysis in result["crash_analysis"].items():
        regime = analysis["regime"]
        exposure = analysis["exposure"]
        exposure_label = "SHORT (sell vol)" if exposure < 0 else "LONG (buy vol)" if exposure > 0 else "FLAT"

        status = (
            "⚠️  RISKY" if (regime == "calm" and exposure < 0) else "✓ OK"
        )

        print(f"  {date_str} ({analysis['description']})")
        print(f"    Regime: {regime.upper()}")
        print(f"    Exposure: {exposure_label} {status}")
        print()

    # Additional diagnostics
    print("REGIME SEQUENCE (sample from 2020-03-01 to 2020-03-31)")
    print("-" * 80)
    regimes_march_2020 = result["regimes"]["2020-03-01":"2020-03-31"]
    if len(regimes_march_2020) > 0:
        print(regimes_march_2020.to_string())
    print()

    # Summary verdict
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    hmm_sharpe = result["metrics"].loc["Sharpe", "HMM Overlay"]
    short_sharpe = result["metrics"].loc["Sharpe", "Static Short"]
    long_sharpe = result["metrics"].loc["Sharpe", "Static Long"]

    hmm_cagr = result["metrics"].loc["CAGR (%)", "HMM Overlay"]
    short_cagr = result["metrics"].loc["CAGR (%)", "Static Short"]
    long_cagr = result["metrics"].loc["CAGR (%)", "Static Long"]

    print()
    print("Key Findings:")
    print(f"  • HMM Sharpe ({hmm_sharpe:.2f}) vs Short ({short_sharpe:.2f}), Long ({long_sharpe:.2f})")
    print(f"  • HMM CAGR ({hmm_cagr:.2f}%) vs Short ({short_cagr:.2f}%), Long ({long_cagr:.2f}%)")

    # Check Feb 2018 and Mar 2020
    feb_2018_regime = result["crash_analysis"].get("2018-02-03", {}).get("regime", "unknown")
    mar_2020_regime = result["crash_analysis"].get("2020-03-06", {}).get("regime", "unknown")

    print()
    print("Crash Diagnostics:")
    print(f"  • Feb 2018 precursor: Regime = {feb_2018_regime.upper()}")
    if feb_2018_regime == "calm":
        print("    ⚠️  Model was SHORT before spike — at risk of loss")
    else:
        print("    ✓ Model was LONG/FLAT before spike — safer position")

    print(f"  • Mar 2020 precursor: Regime = {mar_2020_regime.upper()}")
    if mar_2020_regime == "calm":
        print("    ⚠️  Model was SHORT before crash — at risk of loss")
    else:
        print("    ✓ Model was LONG/FLAT before crash — safer position")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
