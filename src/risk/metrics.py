"""Risk metrics and tail analysis for strategy returns."""

import numpy as np
import pandas as pd
from scipy import stats


def cagr(cumulative_pnl: pd.Series | np.ndarray, years: float, initial_capital: float = 100000) -> float:
    """
    Compound Annual Growth Rate.

    CAGR = (ending_equity / initial_capital) ^ (1 / years) - 1

    Equity = initial_capital + cumulative_pnl

    Args:
        cumulative_pnl: Cumulative P&L series or array (starting at 0)
        years: Number of years
        initial_capital: Starting capital (default 100,000)

    Returns:
        CAGR as decimal (e.g., 0.15 for 15%)
    """
    if isinstance(cumulative_pnl, pd.Series):
        cumulative_pnl = cumulative_pnl.values

    if len(cumulative_pnl) == 0 or years <= 0:
        return 0.0

    ending_pnl = cumulative_pnl[-1]
    ending_equity = initial_capital + ending_pnl

    if ending_equity <= 0:
        return float("-inf")

    return float((ending_equity / initial_capital) ** (1.0 / years) - 1.0)


def annualized_vol(pnl_series: pd.Series | np.ndarray, periods: int = 252) -> float:
    """
    Annualized volatility of daily P&L.

    σ_annual = std(daily_pnl) * sqrt(periods)

    Args:
        pnl_series: Daily P&L (Series or array)
        periods: Annualization factor (default 252 for daily)

    Returns:
        Annualized volatility as decimal
    """
    if isinstance(pnl_series, pd.Series):
        pnl_series = pnl_series.dropna().values

    if len(pnl_series) < 2:
        return 0.0

    return float(np.std(pnl_series, ddof=1) * np.sqrt(periods))


def sharpe_ratio(
    pnl_series: pd.Series | np.ndarray, rf_rate: float = 0.0, periods: int = 252
) -> float:
    """
    Sharpe Ratio.

    Sharpe = (mean(pnl) - rf) / std(pnl) * sqrt(periods)

    Args:
        pnl_series: Daily P&L (Series or array)
        rf_rate: Annual risk-free rate (default 0.0)
        periods: Annualization factor (default 252)

    Returns:
        Sharpe ratio as decimal
    """
    if isinstance(pnl_series, pd.Series):
        pnl_series = pnl_series.dropna().values

    if len(pnl_series) < 2:
        return 0.0

    mean_pnl = np.mean(pnl_series)
    daily_rf = rf_rate / periods
    std_pnl = np.std(pnl_series, ddof=1)

    if std_pnl == 0:
        return float("inf") if mean_pnl > daily_rf else 0.0

    return float((mean_pnl - daily_rf) / std_pnl * np.sqrt(periods))


def sortino_ratio(
    pnl_series: pd.Series | np.ndarray,
    rf_rate: float = 0.0,
    target_return: float = 0.0,
    periods: int = 252,
) -> float:
    """
    Sortino Ratio (penalizes downside volatility only).

    Sortino = (mean(pnl) - rf) / downside_vol * sqrt(periods)

    where downside_vol = std(min(pnl - target, 0))

    Args:
        pnl_series: Daily P&L (Series or array)
        rf_rate: Annual risk-free rate (default 0.0)
        target_return: Target daily return (default 0.0)
        periods: Annualization factor (default 252)

    Returns:
        Sortino ratio as decimal
    """
    if isinstance(pnl_series, pd.Series):
        pnl_series = pnl_series.dropna().values

    if len(pnl_series) < 2:
        return 0.0

    mean_pnl = np.mean(pnl_series)
    daily_rf = rf_rate / periods
    daily_target = target_return / periods

    # Downside: negative deviations from target
    downside = np.minimum(pnl_series - daily_target, 0)
    downside_vol = np.std(downside, ddof=1)

    if downside_vol == 0:
        return float("inf") if mean_pnl > daily_rf else 0.0

    return float((mean_pnl - daily_rf) / downside_vol * np.sqrt(periods))


def max_drawdown(cumulative_pnl: pd.Series | np.ndarray, initial_capital: float = 100000) -> float:
    """
    Maximum Drawdown from peak.

    Equity = initial_capital + cumulative_pnl
    max_dd = (trough - peak) / peak

    Args:
        cumulative_pnl: Cumulative P&L (Series or array)
        initial_capital: Starting capital (default 100,000)

    Returns:
        Max drawdown as decimal (e.g., -0.25 for 25% loss)
    """
    if isinstance(cumulative_pnl, pd.Series):
        cumulative_pnl = cumulative_pnl.values

    if len(cumulative_pnl) == 0:
        return 0.0

    equity = initial_capital + cumulative_pnl
    cummax = np.maximum.accumulate(equity)
    drawdown = (equity - cummax) / cummax

    return float(np.min(drawdown))


def calmar_ratio(
    cumulative_pnl: pd.Series | np.ndarray,
    years: float = 1.0,
    initial_capital: float = 100000,
) -> float:
    """
    Calmar Ratio (Return / Max Drawdown).

    Calmar = CAGR / |max_drawdown|

    Args:
        cumulative_pnl: Cumulative P&L
        years: Number of years
        initial_capital: Starting capital (default 100,000)

    Returns:
        Calmar ratio as decimal
    """
    c_pnl = cagr(cumulative_pnl, years, initial_capital=initial_capital)
    max_dd = max_drawdown(cumulative_pnl, initial_capital=initial_capital)

    if max_dd == 0:
        return float("inf") if c_pnl > 0 else 0.0

    return float(c_pnl / abs(max_dd))


def worst_day(pnl_series: pd.Series | np.ndarray) -> float:
    """
    Worst single-day P&L.

    Args:
        pnl_series: Daily P&L

    Returns:
        Worst day P&L (most negative value)
    """
    if isinstance(pnl_series, pd.Series):
        pnl_series = pnl_series.dropna().values

    if len(pnl_series) == 0:
        return 0.0

    return float(np.min(pnl_series))


def worst_week(pnl_series: pd.Series) -> float:
    """
    Worst week P&L (rolling 5-day worst).

    Args:
        pnl_series: Daily P&L (Series)

    Returns:
        Worst 5-day cumulative return
    """
    if not isinstance(pnl_series, pd.Series):
        pnl_series = pd.Series(pnl_series)

    pnl_series = pnl_series.dropna()

    if len(pnl_series) < 5:
        return float(pnl_series.sum())

    # Rolling 5-day sum
    rolling_week = pnl_series.rolling(window=5).sum()
    return float(rolling_week.min())


def skewness(pnl_series: pd.Series | np.ndarray) -> float:
    """
    Skewness of returns (Fisher-Pearson coefficient).

    Negative = left tail risk; Positive = right tail potential.

    Args:
        pnl_series: Daily P&L

    Returns:
        Skewness as decimal
    """
    if isinstance(pnl_series, pd.Series):
        pnl_series = pnl_series.dropna().values

    if len(pnl_series) < 3:
        return 0.0

    return float(stats.skew(pnl_series))


def excess_kurtosis(pnl_series: pd.Series | np.ndarray) -> float:
    """
    Excess Kurtosis (Fisher definition).

    Excess kurtosis = Kurtosis - 3.
    Normal = 0; High = fat tails (more extreme events than normal).

    Args:
        pnl_series: Daily P&L

    Returns:
        Excess kurtosis as decimal
    """
    if isinstance(pnl_series, pd.Series):
        pnl_series = pnl_series.dropna().values

    if len(pnl_series) < 4:
        return 0.0

    return float(stats.kurtosis(pnl_series, fisher=True))
