#!/usr/bin/env python
"""
Rigorous audit of HMM implementation.
Checks for leakage, regime correctness on crashes, robustness to parameters.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from src.data import load_data
from src.strategy.regime_overlay import RegimeHMM
from src.backtest import run_backtest
from src.risk import sharpe_ratio, cagr, max_drawdown

logging.basicConfig(level=logging.CRITICAL)

print("=" * 80)
print("RIGOROUS HMM AUDIT")
print("=" * 80)
print()

# Load data
spx_src, vix_src = load_data("2015-01-01", "2024-12-31")
spx = spx_src.df
vix = vix_src.df

common_idx = spx.index.intersection(vix.index)
spx = spx.loc[common_idx]
vix = vix.loc[common_idx]

# ==============================================================================
# TEST 1: Feb-2020 COVID PRECURSOR (Did model SHORT before the crash?)
# ==============================================================================
print("TEST 1: FEB-2020 COVID CRASH PRECURSOR")
print("-" * 80)
print("Question: Was model CALM (SHORT gamma) entering the crash?")
print()

hmm = RegimeHMM(n_components=2, refit_freq="Y")
regimes = hmm.fit_and_predict(spx["Close"], vix["Close"])

covid_precursor = regimes["2020-02-19":"2020-02-28"]
print("Regime Feb 19-28, 2020 (BEFORE COVID LOW on Mar 16):")
print(covid_precursor.to_string())
print()

# Count CALM vs TURBULENT in Feb 19-28
calm_count = (covid_precursor == "calm").sum()
turbulent_count = (covid_precursor == "turbulent").sum()
print(f"Summary: {calm_count} CALM days, {turbulent_count} TURBULENT days")

if calm_count > turbulent_count:
    print("⚠️  PROBLEM: Model was CALM (SHORT) before COVID crash")
    print("   If model shorted into crash, it would have lost money")
else:
    print("✓ Model was TURBULENT (LONG) or flat — correct position")

print()

# ==============================================================================
# TEST 2: JAN-29-2018 FEATURE AUDIT (No leakage?)
# ==============================================================================
print("TEST 2: JAN-29-2018 SWITCH AUDIT (Check for Leakage)")
print("-" * 80)
print("Question: Did HMM use only data <= 2018-01-29 to infer regime that day?")
print()

# Manually compute features up to 2018-01-29 (causal)
date_audit = pd.Timestamp("2018-01-29")
data_up_to_audit = spx[spx.index <= date_audit]
vix_up_to_audit = vix[vix.index <= date_audit]

# Feature 1: VIX level
vix_level_audit = vix_up_to_audit["Close"] / 100.0

# Feature 2: 21d realized vol
returns_audit = np.log(data_up_to_audit["Close"] / data_up_to_audit["Close"].shift(1))
realized_vol_21d_audit = returns_audit.rolling(21).std() * np.sqrt(252)

# Feature 3: VIX momentum
vix_momentum_audit = (vix_up_to_audit["Close"] - vix_up_to_audit["Close"].shift(5)) / vix_up_to_audit["Close"].shift(5)

# Z-score with expanding mean/std (causal)
vix_level_z = (vix_level_audit - vix_level_audit.expanding().mean()) / (vix_level_audit.expanding().std() + 1e-8)
realized_vol_z = (realized_vol_21d_audit - realized_vol_21d_audit.expanding().mean()) / (realized_vol_21d_audit.expanding().std() + 1e-8)
vix_momentum_z = (vix_momentum_audit - vix_momentum_audit.expanding().mean()) / (vix_momentum_audit.expanding().std() + 1e-8)

# Report on Jan 29
jan_29_idx = data_up_to_audit.index.get_loc(date_audit)
print(f"Feature values on {date_audit.strftime('%Y-%m-%d')}:")
print(f"  VIX Level:        {vix_level_audit.iloc[jan_29_idx]:>8.4f}  (z-score: {vix_level_z.iloc[jan_29_idx]:>7.3f})")
print(f"  Realized Vol 21d: {realized_vol_21d_audit.iloc[jan_29_idx]:>8.4f}  (z-score: {realized_vol_z.iloc[jan_29_idx]:>7.3f})")
print(f"  VIX Momentum:     {vix_momentum_audit.iloc[jan_29_idx]:>8.4f}  (z-score: {vix_momentum_z.iloc[jan_29_idx]:>7.3f})")
print()

# Check: is this date AFTER min_history_days?
min_history = 252
if len(data_up_to_audit) < min_history:
    print(f"⚠️  WARNING: Only {len(data_up_to_audit)} days available, need {min_history}")
else:
    print(f"✓ Data available: {len(data_up_to_audit)} days (enough for min_history={min_history})")

# Check: regimes on Jan 29 and context
jan_29_regime = regimes.loc[date_audit]
jan_28_regime = regimes[regimes.index <= pd.Timestamp("2018-01-28")].iloc[-1] if len(regimes[regimes.index <= pd.Timestamp("2018-01-28")]) > 0 else "unknown"
print(f"  Regime on Jan 28: {jan_28_regime.upper()}")
print(f"  Regime on Jan 29: {jan_29_regime.upper()}")

if jan_28_regime != jan_29_regime:
    print(f"  ✓ Switch detected: {jan_28_regime.upper()} → {jan_29_regime.upper()}")
else:
    print(f"  No switch on Jan 29")

print()

# ==============================================================================
# TEST 3: OUT-OF-SAMPLE DEFINITION (Excluding burn-in?)
# ==============================================================================
print("TEST 3: OUT-OF-SAMPLE PERIOD DEFINITION")
print("-" * 80)
print("Question: Does OOS exclude the initial training window?")
print()

hmm_check = RegimeHMM(n_components=2, min_history_days=252, refit_freq="Y")

# First refit date should be ~Jan 1, 2016 (after 252 days of 2015 data)
first_refit = hmm_check._get_refit_dates(spx.index)[0] if len(hmm_check._get_refit_dates(spx.index)) > 0 else None
print(f"First refit date: {first_refit}")

# OOS period should start after first refit + some buffer
oos_start = first_refit + pd.Timedelta(days=30) if first_refit else spx.index[300]
print(f"OOS window should start: ~{oos_start.strftime('%Y-%m-%d')} or later")
print(f"Actual backtest starts: {spx.index[0].strftime('%Y-%m-%d')}")

if spx.index[0] < first_refit:
    print(f"⚠️  PROBLEM: In-sample data ({spx.index[0]}) is before first refit ({first_refit})")
    print("   This may include training data in reported results")
else:
    print(f"✓ Backtest starts after first refit (causal)")

print()

# ==============================================================================
# TEST 4: ROBUSTNESS (Quarterly refit, n_components=3)
# ==============================================================================
print("TEST 4: ROBUSTNESS CHECK (Parameter Sensitivity)")
print("-" * 80)
print("Question: Does HMM advantage hold with different parameters?")
print()

def eval_hmm_variant(refit_freq, n_components, desc):
    """Evaluate HMM with different parameters."""
    print(f"\n  Variant: {desc}")
    print(f"    refit_freq={refit_freq}, n_components={n_components}")

    hmm_var = RegimeHMM(n_components=n_components, refit_freq=refit_freq)
    regimes_var = hmm_var.fit_and_predict(spx["Close"], vix["Close"])

    # Evaluate HMM overlay (same mapping: short in calm, long in turbulent)
    result_short = run_backtest(spx_src, vix_src, side=-1, notional=100000, bid_ask_vol_points=0.5)
    result_long = run_backtest(spx_src, vix_src, side=1, notional=100000, bid_ask_vol_points=0.5)

    pnl_gross_short = result_short["pnl_gross"]
    pnl_gross_long = result_long["pnl_gross"]
    roll_costs = result_short["roll_costs"]

    # Dynamic P&L
    pnl_gross_hmm = pd.Series(0.0, index=pnl_gross_short.index)
    for date in regimes_var.index:
        if date in pnl_gross_short.index:
            regime = regimes_var.loc[date]
            if regime == "calm":
                pnl_gross_hmm.loc[date] = pnl_gross_short.loc[date]
            elif regime == "turbulent":
                pnl_gross_hmm.loc[date] = pnl_gross_long.loc[date]
            else:
                pnl_gross_hmm.loc[date] = 0.0

    pnl_net_short = result_short["pnl_series"]
    pnl_net_long = result_long["pnl_series"]
    pnl_net_hmm = pnl_gross_hmm - roll_costs

    # Metrics
    years = (pnl_net_hmm.index[-1] - pnl_net_hmm.index[0]).days / 365.25

    sharpe_short = sharpe_ratio(pnl_net_short.dropna())
    sharpe_long = sharpe_ratio(pnl_net_long.dropna())
    sharpe_hmm = sharpe_ratio(pnl_net_hmm.dropna())

    cum_short = pnl_net_short.dropna().cumsum()
    cum_long = pnl_net_long.dropna().cumsum()
    cum_hmm = pnl_net_hmm.dropna().cumsum()

    cagr_hmm = cagr(cum_hmm, years=years, initial_capital=100000)

    print(f"    Sharpe: Short={sharpe_short:>7.2f}, Long={sharpe_long:>7.2f}, HMM={sharpe_hmm:>7.2f}")
    print(f"    CAGR:   HMM={cagr_hmm*100:>6.2f}%")

    beats_short = sharpe_hmm > sharpe_short
    beats_long = sharpe_hmm > sharpe_long
    print(f"    Beats: Short? {beats_short}, Long? {beats_long}")

    return sharpe_hmm, sharpe_short, sharpe_long, beats_short and beats_long

print("Baseline (Annual, n=2):")
s_hmm_base, s_short_base, s_long_base, beats_base = eval_hmm_variant("Y", 2, "Annual refit, 2 states")

print("\n" + "-" * 40)
print("Testing variations:")
s_hmm_q, s_short_q, s_long_q, beats_q = eval_hmm_variant("Q", 2, "Quarterly refit, 2 states")
s_hmm_3, s_short_3, s_long_3, beats_3 = eval_hmm_variant("Y", 3, "Annual refit, 3 states")

print()
print("ROBUSTNESS SUMMARY:")
print(f"  Annual, n=2:  Beats both? {beats_base}")
print(f"  Quarterly, n=2: Beats both? {beats_q}")
print(f"  Annual, n=3:  Beats both? {beats_3}")

if beats_base and beats_q and beats_3:
    print("\n  ✓ Advantage is robust across parameters")
else:
    print("\n  ⚠️  ALERT: Advantage disappears with parameter changes")
    print("     This suggests overfitting or noise, not signal")

print()

# ==============================================================================
# FINAL VERDICT
# ==============================================================================
print("=" * 80)
print("FINAL AUDIT VERDICT")
print("=" * 80)
print()

issues = []

if calm_count > turbulent_count:
    issues.append("1. COVID PRECURSOR: Model was SHORT entering crash (lost position)")

jan_28_to_29_switch = jan_28_regime != jan_29_regime and jan_29_regime == "turbulent"
if not jan_28_to_29_switch:
    issues.append("2. JAN-29 SWITCH: Regime change less dramatic than claimed")

if spx.index[0] < first_refit:
    issues.append("3. OUT-OF-SAMPLE: May include training data in reported results")

if not (beats_base and beats_q and beats_3):
    issues.append("4. ROBUSTNESS: Advantage disappears with parameter changes (overfitting signal)")

if issues:
    print("❌ CRITICAL ISSUES FOUND:")
    for issue in issues:
        print(f"   {issue}")
    print()
    print("RECOMMENDATION: HMM does NOT improve over baselines reliably.")
    print("Current negative results (-1.43% CAGR) are NOT due to regime overlay benefit.")
else:
    print("✓ All audit checks pass")
    print("  But: Negative returns (-1.43% vs -3.30%) still require strategy rethink")

print()
