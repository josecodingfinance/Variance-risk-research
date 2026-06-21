#!/usr/bin/env python
"""Detailed HMM diagnostic: regime distribution, feature analysis, crash diagnostics."""

import logging
import pandas as pd
import numpy as np
from src.data import load_data
from src.strategy.regime_overlay import RegimeHMM

logging.basicConfig(level=logging.CRITICAL)

# Load data
spx_src, vix_src = load_data("2015-01-01", "2024-12-31")
spx = spx_src.df
vix = vix_src.df

# Align
common_idx = spx.index.intersection(vix.index)
spx = spx.loc[common_idx]
vix = vix.loc[common_idx]

# Fit HMM
hmm = RegimeHMM(n_components=2, refit_freq="Y")
regimes = hmm.fit_and_predict(spx["Close"], vix["Close"])

print("=" * 80)
print("HMM REGIME ANALYSIS")
print("=" * 80)
print()

# Distribution
print("REGIME DISTRIBUTION")
print("-" * 80)
regime_counts = regimes.value_counts()
total = len(regimes)
for regime, count in regime_counts.items():
    pct = count / total * 100
    print(f"  {regime.upper():12} : {count:4} days ({pct:5.1f}%)")
print()

# Transition matrix
print("REGIME TRANSITIONS (Monthly Count)")
print("-" * 80)
transitions = regimes.shift(1) != regimes
transition_count = transitions.sum()
print(f"  Total transitions: {transition_count} (~{transition_count / (total / 252):.0f} per year)")
print()

# Regime statistics (avg VIX, avg realized vol per regime)
print("REGIME STATISTICS")
print("-" * 80)
for regime in ["calm", "turbulent"]:
    mask = regimes == regime
    regime_dates = regimes[mask].index

    avg_vix = vix.loc[regime_dates, "Close"].mean()

    returns = np.log(spx["Close"] / spx["Close"].shift(1))
    avg_realized_vol = (
        returns[regime_dates]
        .rolling(21)
        .std()
        .mean()
    ) * np.sqrt(252)

    print(f"  {regime.upper():12}")
    print(f"    Avg VIX:         {avg_vix:>6.1f}")
    print(f"    Avg Realized Vol: {avg_realized_vol * 100:>6.1f}%")
    print()

# Crash diagnostics with full regime history around dates
crash_dates = [
    ("2018-02-05", "Feb-2018 Spike"),
    ("2020-03-16", "Mar-2020 COVID Low"),
]

print("DETAILED CRASH DIAGNOSTICS")
print("-" * 80)
for date_str, desc in crash_dates:
    date = pd.Timestamp(date_str)
    window = regimes[(regimes.index >= date - pd.Timedelta(days=20)) & (regimes.index <= date + pd.Timedelta(days=5))]

    print(f"\n  {desc} ({date_str}):")
    print(f"    Regime History (20 days before → 5 days after):")
    for d, r in window.items():
        marker = ">>> CRASH DATE <<<" if d.strftime("%Y-%m-%d") == date_str else ""
        print(f"      {d.strftime('%Y-%m-%d')} : {r.upper():12} {marker}")

    # Count regime in 5 days before crash
    before_window = regimes[(regimes.index >= date - pd.Timedelta(days=5)) & (regimes.index < date)]
    regime_before = before_window.mode()[0] if len(before_window) > 0 else "unknown"
    print(f"    → Regime in 5 days before: {regime_before.upper()}")
    if regime_before == "calm":
        print(f"      ⚠️  RISKY: Model was SHORT gamma before crash")
    else:
        print(f"      ✓ OK: Model was LONG gamma (or flat) before crash")

print()
print("=" * 80)
print("CONCLUSION")
print("=" * 80)
regime_stability = transition_count / total
if regime_stability < 0.05:  # < 5% transitions
    print(f"⚠️  Model is TOO STABLE ({regime_stability*100:.1f}% transitions)")
    print("    → May not be capturing regime changes effectively")
elif regime_stability > 0.20:  # > 20% transitions
    print(f"⚠️  Model is TOO VOLATILE ({regime_stability*100:.1f}% transitions)")
    print("    → May be overfitting to noise")
else:
    print(f"✓ Model regime transitions are reasonable ({regime_stability*100:.1f}%)")

print()
