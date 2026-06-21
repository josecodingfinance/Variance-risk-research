# CLAUDE.md — Gamma Capture Fund

Context file for Claude Code. Read this before writing or editing anything in this repo.
Project root: `~/quant/fund`.

---

## 1. What this project is

A long-horizon, rigorously-built research platform for a **gamma capture / volatility
relative-value** strategy (Formulation A: variance swap), with room to grow into a
multi-strategy systematic book.

**This is a research and learning project first.** The goal is to *test honestly* whether
the strategy is profitable, not to assume it is. A pretty backtest is not a live edge.
We never conflate "good Sharpe in-sample" with "trade real money".

**Depth over breadth.** One component to a defensible, provable standard before the next.

---

## 2. The mathematical core (Formulation A — variance swap)

Daily P&L of a monthly-held variance swap, strike fixed between rolls:

    pnl_t = side * notional * (realized_var_t - strike_month)

- realized_var_t = (ln(S_t / S_{t-1}))^2  (close-to-close)
- strike_month   = (VIX_roll/100)^2 / 252, fixed from the roll date (ffill), no look-ahead
- side: +1 long-gamma (profit when realized > implied), -1 short-gamma
- NOT delta-hedging. NOT a daily-reset slice. The strike is held fixed within each month.

Roll costs charged ONLY on roll dates (1st business day of month), quoted in
**volatility points** and converted to variance via dVar/dsigma = 2*sigma.

---

## 3. Roadmap

1. Stage 1 — Long-gamma core on SPX + VIX. DONE & APPROVED.
2. Stage 2 — Short-gamma side + tail measurement. Implemented (side=-1 mirror verified).
3. Stage 3 — Regime overlay (HMM) deciding long vs short vs flat. <-- CURRENT.
4. Stage 4 — Multi-strategy expansion, allocator, factor risk decomposition, dashboard.

---

## 4. Current state (update at the end of each session)

- APPROVED, do NOT touch: src/core/variance_spread.py, src/risk/metrics.py,
  src/backtest/engine.py (monthly-held swap, fixed strike, vol-point costs).
- Key facts in approved code:
  - variance_spread.py: variance notional (NOT 0.5*Gamma*S^2); config is NOT read inside
    the core; variance_pnl_series aligns BEFORE shift and validates side in (-1,1).
  - metrics.py: CAGR and max_drawdown computed on equity = initial_capital + cumulative_pnl
    (NOT on base-zero cumulative P&L). calmar uses both.
  - engine.py: strike fixed between monthly rolls; costs in vol points (dVar/dsigma=2sigma);
    bid-ask param is bid_ask_vol_points.
- Stage 1 result (2015-2024, real data, $100k/variance point):
  Gross -$6,637 | Roll costs $21,844 | Net -$28,481 | CAGR -3.30% | MaxDD -28.59% |
  Sharpe -2.71. Long gamma bleeds (negative VRP for the long side); costs exceed the
  gross premium -> naive short is also net-negative. Verified: perfect long/short mirror,
  long gamma spikes up in Feb-2018 and Mar-2020.
- IN PROGRESS: src/strategy/regime_overlay.py (HMM).

---

## 5. Repo structure

fund/
├── CLAUDE.md
├── config.yaml              # DO NOT edit without asking
├── src/
│   ├── data/                # download + cache (DataWithSource), SPX/VIX
│   ├── core/                # realized_vol, implied_vol, variance_spread
│   ├── strategy/            # regime_overlay (HMM)  <-- current work
│   ├── backtest/            # engine (monthly-held swap)
│   └── risk/                # metrics, tail analysis
├── tests/                   # pytest — every module tested, hand-computed cases
├── notebooks/
└── outputs/

---

## 6. HMM regime overlay — MANDATORY guardrails (Stage 3)

A single violation here invalidates the result.

1. Causal. Regime at t uses only data <= t. Walk-forward: refit the HMM periodically
   (e.g. yearly) on data up to that date, infer regime with forward filtering — NOT
   smoothed Viterbi over the whole history.
2. Max 3 features: VIX level, 21-day realized vol, (optional) term slope. Start with
   n_components=2 (calm vs turbulent) for interpretability.
3. Causal scaling. Feature z-scores use expanding mean/std, never full-sample.
4. State identification. "Turbulent" = state with the higher mean VIX, recomputed at
   each refit (avoid label-switching). NEVER label states by realized returns.
5. Mapping by logic, a priori. Short gamma in calm, flat (or long) in turbulence.
   NOT optimized to maximize the backtest.
6. Evaluation. Out-of-sample only, net of roll costs (including the cost of switching
   regimes). Must beat BOTH static baselines (static short, static long) OOS net, or it
   adds nothing. Diagnostic: print the regime in the days BEFORE Feb-2018 and Mar-2020 —
   was it short going into the crash?

---

## 7. Tech stack & conventions

- Python 3.11+ in .venv. Never install globally.
- Libs: pandas, numpy, scipy, statsmodels, scikit-learn, hmmlearn, yfinance, matplotlib.
- Type hints + docstrings everywhere. pytest from day one; math functions get a test
  against a hand-computed value.
- Small, focused commits.

---

## 8. Quant honesty rules (non-negotiable)

- No look-ahead, anywhere (data, regime fitting, feature scaling).
- Always report the tail (max drawdown, worst week, skew/kurtosis), not just Sharpe.
- Costs are real and prominent. Always print gross vs net side by side.
- Synthetic data must be loudly labeled (DataWithSource.is_synthetic).
- If a result looks too good, say so and investigate.

---

## 9. Working agreement for Claude Code

- Do NOT edit config.yaml, or the approved files in section 4, without flagging it first.
- When implementing math, put the formula in the docstring and a test that verifies it.
- Ask before introducing a new dependency or a major architectural change.
- If a result looks too good, investigate rather than celebrate.
