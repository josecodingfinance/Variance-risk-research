#!/usr/bin/env python
"""
Honest HMM backtest: TRUE out-of-sample, excluding all training data (2015).
OOS window: 2016-01-04 onwards (after first refit is available).
"""

import logging
import pandas as pd
from src.data import load_data
from src.strategy.regime_overlay import evaluate_regime_strategy
from src.risk import sharpe_ratio, cagr, max_drawdown

logging.basicConfig(level=logging.CRITICAL)

print("=" * 80)
print("HONEST OUT-OF-SAMPLE BACKTEST (Excluding 2015 Training Data)")
print("=" * 80)
print()

# Load data
spx_src, vix_src = load_data("2015-01-01", "2024-12-31")

# Run full evaluation
result = evaluate_regime_strategy(
    spx_src, vix_src, initial_capital=100000, bid_ask_vol_points=0.5, refit_freq="Y"
)

# Get full series
cum_short = result["cumulative_short"]
cum_long = result["cumulative_long"]
cum_hmm = result["cumulative_hmm"]

# CRITICAL: Exclude 2015 (training period for 2016 refit)
oos_start = pd.Timestamp("2016-02-01")  # Safe margin after 2016-01-04 refit

cum_short_oos = cum_short[cum_short.index >= oos_start]
cum_long_oos = cum_long[cum_long.index >= oos_start]
cum_hmm_oos = cum_hmm[cum_hmm.index >= oos_start]

pnl_short_oos = result["pnl_short"][result["pnl_short"].index >= oos_start]
pnl_long_oos = result["pnl_long"][result["pnl_long"].index >= oos_start]
pnl_hmm_oos = result["pnl_hmm"][result["pnl_hmm"].index >= oos_start]

print(f"OOS Period: {oos_start.strftime('%Y-%m-%d')} to {cum_hmm_oos.index[-1].strftime('%Y-%m-%d')}")
print(f"OOS Trading Days: {len(pnl_hmm_oos)}")
print()

# Calculate metrics OOS
years_oos = (cum_hmm_oos.index[-1] - cum_hmm_oos.index[0]).days / 365.25

def calc_metrics_oos(pnl_series, cumulative):
    pnl_clean = pnl_series.dropna()
    cum_clean = cumulative.dropna()

    c = cagr(cum_clean, years=years_oos, initial_capital=100000)
    v = sharpe_ratio(pnl_clean)
    dd = max_drawdown(cum_clean, initial_capital=100000)

    return {
        "CAGR (%)": c * 100,
        "Sharpe": v,
        "Max DD (%)": dd * 100,
    }

metrics_short_oos = calc_metrics_oos(pnl_short_oos, cum_short_oos)
metrics_long_oos = calc_metrics_oos(pnl_long_oos, cum_long_oos)
metrics_hmm_oos = calc_metrics_oos(pnl_hmm_oos, cum_hmm_oos)

print("RESULTS (TRUE OUT-OF-SAMPLE, 2016-2024):")
print("-" * 80)
print(f"{'Metric':<15} {'Static Short':>15} {'Static Long':>15} {'HMM Overlay':>15}")
print("-" * 80)
for key in ["CAGR (%)", "Sharpe", "Max DD (%)"]:
    print(f"{key:<15} {metrics_short_oos[key]:>15.2f} {metrics_long_oos[key]:>15.2f} {metrics_hmm_oos[key]:>15.2f}")

print()
print("VERDICT:")
print("-" * 80)

beats_short = metrics_hmm_oos["Sharpe"] > metrics_short_oos["Sharpe"]
beats_long = metrics_hmm_oos["Sharpe"] > metrics_long_oos["Sharpe"]

print(f"HMM Sharpe ({metrics_hmm_oos['Sharpe']:.2f}) vs Short ({metrics_short_oos['Sharpe']:.2f}): {'✓ BEATS' if beats_short else '✗ LOSES'}")
print(f"HMM Sharpe ({metrics_hmm_oos['Sharpe']:.2f}) vs Long ({metrics_long_oos['Sharpe']:.2f}):  {'✓ BEATS' if beats_long else '✗ LOSES'}")
print()

if beats_short and beats_long:
    print("✓ HMM improves over BOTH baselines (out-of-sample)")
elif beats_short or beats_long:
    print("⚠️  HMM beats only one baseline (marginal improvement)")
else:
    print("❌ HMM loses to BOTH baselines (out-of-sample)")
    print("   → No value added by regime overlay")

print()
print("=" * 80)
print("COMPARISON: Full Period (2015-2024) vs OOS-Only (2016-2024)")
print("=" * 80)
print(f"Full period HMM Sharpe:  {result['metrics'].loc['Sharpe', 'HMM Overlay']:>7.2f}")
print(f"OOS-only HMM Sharpe:     {metrics_hmm_oos['Sharpe']:>7.2f}")
print()

if abs(metrics_hmm_oos['Sharpe'] - result['metrics'].loc['Sharpe', 'HMM Overlay']) > 0.2:
    print("⚠️  LARGE DIFFERENCE → 2015 training data was helping (data snooping)")
else:
    print("✓ Results stable → No major overfitting from 2015")

print()
