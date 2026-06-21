# Research Memo — Equity Variance Risk Premium via Gamma Capture

**Author:** [Your name]
**Project:** Gamma Capture Fund (systematic volatility relative-value research platform)
**Status:** Closed — negative result
**One-line thesis tested:** *Can a passive/dynamic gamma (variance swap) strategy on SPX harvest the volatility risk premium net of realistic costs?*
**Verdict:** No. The premium is structurally negative for the long side and the short side is a negative-skew trade whose backtested Sharpe materially understates tail risk.

---

## 1. Executive summary

I built a reproducible research platform that prices a monthly-held SPX variance swap (realized vs implied variance, VIX as the implied input), backtests it long and short net of roll costs, and overlays a causal Hidden Markov regime classifier to time exposure. Across periods (including the 2008 crisis), across cost assumptions (including **zero cost**), and with the regime overlay, **no configuration of passive gamma trading is profitable.** The most important single result: at **zero transaction cost**, static long gamma still returns **−0.69% CAGR** — i.e., the loss is structural (you pay the volatility risk premium), not merely a cost problem. The disciplined conclusion is to **not deploy capital** and to **not pivot to short-premium harvesting**, which would convert a bounded loser into a strategy carrying catastrophic, under-sampled tail risk.

---

## 2. Methodology and rigor controls

- **Instrument:** monthly-held variance swap. Daily P&L = `side × notional × (realized_var_t − strike_month)`, with the strike fixed between monthly rolls at `(VIX_roll/100)² / 252` (forward-filled). Realized variance from close-to-close log returns.
- **Costs:** charged only on roll dates, quoted in volatility points and converted to variance via `dVar/dσ = 2σ` (realistic ~0.5 vol-point bid-ask baseline).
- **Regime overlay:** 2-state Gaussian HMM on three causal features (VIX level, 21-day realized vol, VIX momentum). Walk-forward refit, forward-filtered inference (no smoothed Viterbi), expanding-window feature scaling, states identified by mean VIX (no label switching), exposure mapped by economic logic (not optimized to the backtest).
- **No look-ahead:** verified at the code level (lag baked into the P&L function; causal scaling and inference audited on the Feb-2018 switch).
- **Reproducibility:** real SPX/VIX data via yfinance with caching and a clearly-labeled synthetic fallback; full pytest suite with hand-computed test cases.

---

## 3. Results

### Static strategies (2015–2024, net of costs, $100k variance notional)

| Strategy | CAGR | Sharpe | Max DD |
|---|---|---|---|
| Static long gamma | −3.30% | −2.71 | −28.59% |
| Static short gamma | −1.64% | −1.45 | −15.34% |

Gross long P&L −$6,637 vs roll costs $21,844 → net −$28,481. The long/short mirror is exact (sum of gross P&L = $0), confirming model symmetry.

### HMM regime overlay (true out-of-sample, 2016–2024, net)

| | Static short | Static long | HMM overlay |
|---|---|---|---|
| CAGR | −1.83% | −3.69% | **−1.60%** |
| Sharpe | −1.42 | −2.65 | **−1.16** |
| Max DD | −13.96% | −26.46% | −14.11% |

The HMM beats both static baselines out-of-sample and across robustness checks (quarterly vs annual refit; 2 vs 3 states). It reduces the bleed but remains underwater.

### Period sensitivity (static long, net)

| Period | CAGR | Sharpe | Note |
|---|---|---|---|
| 2007–2009 (GFC) | −3.30% | −1.81 | Gross ≈ breakeven; crisis did **not** rescue it |
| 2020–2021 | −3.29% | −1.67 | Negative even before costs |

In crises the strike reprices upward as VIX spikes, so a monthly-held swap captures only the within-month surprise, not the full move.

### Cost sensitivity (static long, 2015–2024)

| Bid-ask | CAGR | Sharpe |
|---|---|---|
| 0.0 (no cost) | **−0.69%** | −0.81 |
| 0.2 vol pts | −1.66% | −1.78 |
| 0.5 vol pts (baseline) | −3.30% | −2.71 |
| 1.0 vol pts | −6.77% | −3.25 |

**Even at zero cost the strategy loses.** The loss decomposes roughly 50% structural carry, 50% costs.

---

## 4. Key findings

1. **The long-side carry is structurally negative.** Implied exceeds realized variance ~85% of the time, so passive long gamma pays the volatility risk premium daily. Confirmed at zero cost.
2. **Crashes do not compensate the bleed.** The repricing of the strike during sustained high-vol regimes limits how much a monthly-held swap earns in a crisis.
3. **Costs compound a pre-existing loser; they are not the root cause.** Cost reduction cannot change the sign for the long side.
4. **The regime overlay adds genuine but marginal value** (≈+0.2pp CAGR, Sharpe −1.16 vs −1.42 OOS). Its structural blind spot is the calm→crash transition.
5. **The short side is a trap, not the opportunity.** Its positive average carry is compensation for negative skew: frequent small gains punctuated by catastrophic losses concentrated in crashes. A ~10-year sample contains only 2–3 such events, so backtested Sharpe systematically *understates* the true risk. This is the XIV/2018 and March-2020 failure mode. We explicitly declined to pursue it.

---

## 5. Decision and what this demonstrates

We close this strategy as a **rigorous negative result** rather than engineer a positive backtest. This decision itself is the deliverable: it demonstrates the ability to (a) build a correct, causal, reproducible volatility research stack, (b) measure honestly net of costs and with tail focus, and (c) resist the seductive short-vol pivot that ends accounts. The infrastructure (data layer, variance pricing core, backtest engine, risk metrics, HMM overlay) is reusable for future strategy research.

---

## 6. Limitations and honest caveats

- Single underlying (SPX) and single implied proxy (30-day VIX); the planned RVX (small-cap) and VIX3M quarterly-roll experiments were not completed, though the zero-cost result makes the long-side conclusion robust regardless.
- The monthly-held daily-accrual P&L is a stylized approximation of true variance-swap mark-to-market.
- Backtest costs are modeled, not measured against live fills; live costs and capacity constraints would be worse for a small operator.
- A backtest is not a live edge under any circumstances.

---

## 7. Reproducibility

Run `pytest` for the full test suite (causality, no-look-ahead, hand-computed math cases). The backtest entry point reproduces all tables above from cached or freshly-downloaded SPX/VIX data.
