# Gamma Capture Fund — Backtest Framework

Complete Stage 1 implementation: Long-gamma variance swap on SPX+VIX (2015-2024).

---

## Quick Start

### 1. Environment Setup

```bash
cd /home/jose/quant/fund
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Full Backtest (with metrics)

```bash
python backtest_run.py
```

**Output**: Strategy performance report (2015-2024) with:
- Total P&L, CAGR, annualized volatility
- Sharpe, Sortino, max drawdown, Calmar
- Worst day/week, skewness, excess kurtosis
- Tail analysis (5th/95th percentile, % positive days)

### 3. Run Test Suite

```bash
# All tests
pytest tests/ -v

# Coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Specific module
pytest tests/test_backtest_engine.py tests/test_risk_metrics.py -v
```

---

## Architecture

### Data Layer (`src/data/`)
- **`download.py`**: yfinance + local cache (keyed by ticker, start, end)
- **`types.py`**: `DataWithSource` wrapper (df + is_synthetic flag)
- Synthetic fallback with reproducible seed
- No interpolation of missing data

### Core (`src/core/`)
- **`realized_vol.py`**: Close-to-close + Parkinson estimators
- **`implied_vol.py`**: VIX → daily variance (no-look-ahead)
- **`variance_spread.py`**: P&L scalars + vectorized `variance_pnl_series()` (shift built-in)

### Backtest (`src/backtest/`)
- **`engine.py`**: Main runner
  - Loads SPX+VIX via `load_data()`
  - Computes realized var (r²) + implied var daily
  - Applies `variance_pnl_series()` (side: ±1, notional from config)
  - Deducts bid-ask costs (5bps configurable)
  - Returns P&L series + cumulative

### Risk (`src/risk/`)
- **`metrics.py`**: 10 tail-focused metrics
  - CAGR, vol, Sharpe, Sortino
  - Max drawdown, Calmar, worst day/week
  - Skewness, excess kurtosis (left tail + fat tails)

---

## Key Design Decisions

✅ **Formulación A (Variance Swap)**
- Constant variance notional (not dollar-gamma of vanilla option)
- No look-ahead: shift baked into `variance_pnl_series()`

✅ **Data Robustness**
- Synthetic flag survives all operations via `DataWithSource` dataclass
- Cache key includes (ticker, start, end) — prevents silently wrong ranges
- Synthetic data reproducible (seed-based)

✅ **Core Pure**
- Math functions take arguments, don't load config
- Backtest engine bridges to config.yaml

✅ **Tail Risk Focus**
- Not just Sharpe; includes Sortino, max_dd, Calmar, skew, kurtosis
- Worst-day/worst-week explicitly reported

---

## Config (`config.yaml`)

```yaml
variance:
  notional: 100000          # dollars per point of variance
costs:
  bid_ask_bps: 5            # bid-ask spread
```

---

## Test Coverage

- **102 tests pass** (93% coverage)
- **Data layer** (22 tests): cache, synthetic, alignment, reproducibility
- **Core** (34 tests): realized/implied vol, variance spread, series shift
- **Backtest** (10 tests): engine correctness, costs, side effects
- **Risk** (27 tests): metrics calculation, edge cases, hand-computed values
- **Integration**: Full backtest on 2515 trading days (2015-2024)

---

## Usage Example

```python
from src.data import load_data
from src.backtest import run_backtest
from src.risk import sharpe_ratio, max_drawdown

# Load real data
spx_src, vix_src = load_data("2015-01-01", "2024-12-31")

# Run backtest
result = run_backtest(spx_src, vix_src, side=1, notional=100000)

# Analyze
pnl = result["pnl_series"].dropna()
cumulative = result["cumulative"]

print(f"Sharpe: {sharpe_ratio(pnl):.2f}")
print(f"Max DD: {max_drawdown(cumulative):.2%}")
```

---

## Limitations & Next Steps

**Current (Stage 1)**:
- Long gamma only (static side)
- Fixed notional (no dynamic sizing)
- Daily rehedge assumption only

**Stage 2** (Planned):
- Short gamma + regime overlay (HMM)
- Dynamic rehedge scheduling
- Transaction cost modeling
- Multi-asset expansion

---

## Data Caching

Data is cached locally in `data/cache/` with MD5 keys:
```
data/cache/
├── GSPC_c3e1846c.pkl    # SPX (2015-2024)
└── VIX_f0e66bff.pkl     # VIX (2015-2024)
```

Delete cache to force re-download:
```bash
rm -rf data/cache/
```

---

## Troubleshooting

**Synthetic data fallback**:
- If yfinance is down, backtest uses reproducible synthetic data (clearly labeled)
- Check logs: `⚠️ WARNING: Using synthetic data`

**Cache mismatch**:
- If you request a different date range, a new cache is created (different hash key)

**Test failures**:
- Run `pytest tests/ -v --tb=short` for detailed error context
- Check system Python version: 3.11+ required

---

## References

See `CLAUDE.md` for:
- Mathematical formulas (variance swap framework)
- No-look-ahead bias requirements
- Tail risk reporting standards
- Stage roadmap
