# Gamma Capture Fund

Long-horizon research platform for a **gamma capture / volatility relative-value strategy**, with room to grow into a multi-strategy systematic book.

**Stage 1 (Current):** Long-gamma core on SPX + VIX with realized-vs-implied volatility.

## Quick Start

### 1. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run tests

```bash
pytest tests/ -v --tb=short
```

Check coverage:
```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## Project Structure

```
src/
├── data/       # Download + cache (SPX, VIX) via yfinance
├── core/       # Greeks, gamma P&L, rehedge (Stage 1+)
├── strategy/   # Long/short gamma, regime overlay (Stage 2+)
├── risk/       # Metrics, tail, drawdown analysis
└── backtest/   # Engine + walk-forward validation

tests/          # pytest test suite
notebooks/      # Exploration only (not source of truth)
```

## Key Principles

- **Type hints & docstrings** on all public functions
- **No look-ahead bias.** All parameters use only information available at that point in time.
- **Report the tail.** Max drawdown, worst week/day, VaR/CVaR, not just Sharpe.
- **Costs are real.** Transaction costs, slippage, and bid-ask modeled explicitly.
- **Test early.** Every new module ships with pytest tests.

See `STEPS.md` for full context and roadmap.
