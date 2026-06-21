"""Realized volatility estimators."""

import numpy as np


def close_to_close_vol(returns: np.ndarray, periods: int = 252) -> float:
    """
    Close-to-close realized volatility from log-returns.

    Formula: sqrt(mean(r_t^2) * periods)

    Args:
        returns: Array of log-returns (e.g., np.log(price_t / price_{t-1}))
        periods: Annualization factor (default 252 for daily data)

    Returns:
        Annualized realized volatility as a float
    """
    if len(returns) == 0:
        return 0.0
    return float(np.sqrt(np.mean(returns**2) * periods))


def parkinson_vol(high: np.ndarray, low: np.ndarray, periods: int = 252) -> float:
    """
    Parkinson (1980) realized volatility from intraday high-low.

    Less sensitive to opening gaps than close-to-close. Formula:

        σ = sqrt(mean((1/(4*ln(2)) * ln(H/L))^2) * periods)

    Args:
        high: Array of intraday high prices (price levels, not returns)
        low: Array of intraday low prices (price levels, not returns)
        periods: Annualization factor (default 252 for daily data)

    Returns:
        Annualized realized volatility as a float
    """
    if len(high) != len(low) or len(high) == 0:
        return 0.0

    # Parkinson constant: 1 / (4 * ln(2))
    c = 1.0 / (4 * np.log(2))

    # hl_ratio = H/L for each period
    hl_ratio = high / low
    hl_log = np.log(hl_ratio)

    # Variance: mean((c * ln(H/L))^2)
    variance = np.mean((c * hl_log) ** 2)

    return float(np.sqrt(variance * periods))
