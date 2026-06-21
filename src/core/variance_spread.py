"""Variance spread P&L calculations."""

import logging
import pandas as pd


logger = logging.getLogger(__name__)


def daily_variance_pnl(
    realized_var: float, implied_var_prev: float
) -> float:
    """
    Daily variance spread P&L (before position sizing).

    P&L = realized_var - implied_var_prev

    This is the core of the strategy: profit iff realized variance exceeds
    implied variance (implied vol was too low). The sign depends on whether
    we are long or short variance.

    Args:
        realized_var: Daily realized variance (e.g., r^2 where r is daily log-return)
        implied_var_prev: Lagged daily implied variance from VIX (t-1)
                         Must be lagged to avoid look-ahead bias.

    Returns:
        Variance spread in units of variance (not dollars)

    Note:
        - Uses lagged VIX (no look-ahead bias, per CLAUDE.md)
        - Sign interpretation:
          + positive: long-gamma wins (realized > implied)
          - negative: short-gamma wins (implied > realized)
    """
    return realized_var - implied_var_prev


def variance_pnl_notional(
    realized_var: float, implied_var_prev: float, notional: float
) -> float:
    """
    Daily variance spread P&L in dollars.

    P&L = (realized_var - implied_var_prev) * notional

    The notional represents the dollars per point of variance (constant by
    construction in a variance swap / log-contract). This is distinct from
    the dollar-gamma (0.5*Γ*S²) of a vanilla option, which varies daily.
    For a log-contract, dollar-gamma = 0.5*Γ*S² = 1 (constant), making
    the two concepts equivalent in that special case.

    Args:
        realized_var: Daily realized variance
        implied_var_prev: Lagged daily implied variance
        notional: Dollars per point of variance (required argument, not loaded from config)

    Returns:
        Daily P&L in dollars

    Note:
        - Notional is the constant variance exposure (variance swap framework).
        - For a delta-hedged position, the dollar variance exposure stays fixed
          regardless of underlying price level or time decay.
        - Core is pure; caller (backtest engine) provides notional from config.
    """
    return daily_variance_pnl(realized_var, implied_var_prev) * notional


def variance_pnl_series(
    realized_var: pd.Series,
    implied_var: pd.Series,
    side: int = 1,
    notional: float = 1.0,
) -> pd.Series:
    """
    Vectorized daily variance P&L with automatic lagging (no look-ahead).

    This function enforces the no-look-ahead rule at the code level:
    it shifts implied_var internally, so the caller cannot accidentally
    use tomorrow's information.

    Args:
        realized_var: Series of daily realized variance (e.g., log-return squared)
        implied_var: Series of daily implied variance (from lagged VIX close)
        side: +1 for long-gamma (profit when realized > implied),
              -1 for short-gamma (profit when implied > realized)
        notional: Dollars per point of variance

    Returns:
        Series of daily P&L with first row NaN (discarded from lagged shift).

    Formula:
        implied_prev_t = implied_var_{t-1}  (lagged)
        spread_t = realized_var_t - implied_prev_t
        pnl_t = side * spread_t * notional

    Raises:
        ValueError: If side not in (-1, 1)

    Note:
        - First row is NaN due to shift(1); typically discarded in analysis.
        - Indices aligned BEFORE shift to ensure clean lagging.
        - Shift happens inside; no need to remember to lag in caller code.
    """
    if side not in (-1, 1):
        raise ValueError(f"side must be -1 or +1, got {side}")

    # Align indices FIRST (inner join on common dates)
    common_idx = realized_var.index.intersection(implied_var.index)
    if common_idx.empty:
        logger.warning("No common dates between realized_var and implied_var")
        return pd.Series([], dtype=float)

    realized_aligned = realized_var.loc[common_idx]
    implied_aligned = implied_var.loc[common_idx]

    # THEN lag implied variance (yesterday's information, no look-ahead)
    implied_prev = implied_aligned.shift(1)

    # Variance spread: realized - lagged implied
    spread = realized_aligned - implied_prev

    # P&L with side: +1 long-gamma, -1 short-gamma
    pnl = side * spread * notional

    return pnl
