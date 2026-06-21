"""Backtest engine for variance swap strategies."""

import logging
import numpy as np
import pandas as pd

from src.data.types import DataWithSource
from src.core import vix_to_daily_variance, variance_pnl_series


logger = logging.getLogger(__name__)


def identify_roll_dates(dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """
    Identify roll dates: 1st business day of each month.

    Args:
        dates: DatetimeIndex of trading dates (already business days)

    Returns:
        List of roll dates (1st business day of each month in the date range)
    """
    if dates.empty:
        return []

    roll_dates = []
    prev_month = None

    for date in dates:
        curr_month = (date.year, date.month)
        if prev_month is None or curr_month != prev_month:
            # First business day of new month
            roll_dates.append(date)
            prev_month = curr_month

    return roll_dates


def compute_roll_cost(prev_vix: float, curr_vix: float, notional: float, bid_ask_vol_points: float) -> float:
    """
    Cost to roll a variance swap (monthly).

    When rolling, we exit variance contract at prev_vol, enter at curr_vol.
    Bid-ask impact on vol → variance via dVar/dσ = 2σ.

    Cost in $ = bid_ask_vol * 2 * avg_vol * notional

    Args:
        prev_vix: VIX level at previous roll
        curr_vix: VIX level at current roll
        notional: Dollars per point of variance
        bid_ask_vol_points: Bid-ask spread in vol points (e.g., 0.5 = 0.5%)

    Returns:
        Roll cost in dollars (positive = cost to trader)
    """
    prev_vol = prev_vix / 100.0
    curr_vol = curr_vix / 100.0
    avg_vol = (prev_vol + curr_vol) / 2.0

    bid_ask_vol = bid_ask_vol_points / 100.0  # e.g., 0.5 vol points = 0.005
    dvar_from_bidask = bid_ask_vol * 2 * avg_vol  # Variance impact from bid-ask on vol
    cost = dvar_from_bidask * notional

    return cost


def run_backtest(
    spx_source: DataWithSource,
    vix_source: DataWithSource,
    side: int = 1,
    notional: float = 100000,
    bid_ask_vol_points: float = 0.5,
) -> dict:
    """
    Run variance swap backtest on SPX + VIX data (monthly roll mechanics).

    Model: Hold variance swap with FIXED STRIKE between roll dates.
    - Strike is set on 1st business day of each month at (VIX/100)^2 / 252
    - Strike is forward-filled (ffill) until next roll date
    - Daily gross P&L = side * notional * (realized_var - strike_fixed)
    - Roll costs charged only on roll dates (bid-ask in vol points → variance)
    - No daily look-ahead: strike locked in at roll date's VIX

    Args:
        spx_source: DataWithSource with SPX daily OHLCV data
        vix_source: DataWithSource with VIX daily OHLCV data
        side: +1 for long-gamma (profit when realized > fixed strike),
              -1 for short-gamma (profit when fixed strike > realized)
        notional: Dollars per point of variance
        bid_ask_vol_points: Bid-ask spread in vol points (e.g., 0.5 = 0.5% on vol)

    Returns:
        Dictionary with:
            {
                "pnl_series": Series of daily net P&L (after roll costs),
                "cumulative": Series of cumulative P&L,
                "pnl_gross": Series of daily gross P&L (mark-to-market vs fixed strike),
                "roll_costs": Series of roll costs (non-zero only on roll dates),
                "strike": Series of fixed strike (constant between rolls),
                "realized_var": Series of daily realized variance,
                "synthetic_warning": bool (True if data is synthetic),
            }

    Raises:
        ValueError: If data alignment results in empty index
    """
    if spx_source.is_synthetic or vix_source.is_synthetic:
        logger.warning("⚠️ Backtest on synthetic data — not for real trading")

    # Extract DataFrames
    spx = spx_source.df.copy()
    vix = vix_source.df.copy()

    # Align dates (inner join)
    common_idx = spx.index.intersection(vix.index)
    if common_idx.empty:
        raise ValueError("No common trading dates between SPX and VIX")

    spx = spx.loc[common_idx]
    vix = vix.loc[common_idx]

    # Daily realized variance: close-to-close log-returns squared
    returns = np.log(spx["Close"] / spx["Close"].shift(1))
    daily_var = returns ** 2

    # VIX close series
    vix_close = vix["Close"]

    # Identify roll dates (1st business day of each month)
    roll_dates = identify_roll_dates(common_idx)

    # Build fixed strike series: set at roll date, forward-fill until next roll
    strike_series = pd.Series(np.nan, index=common_idx)

    if not roll_dates:
        # No roll dates identified (e.g., less than 1 month of data), use first date as strike
        if len(common_idx) > 0:
            strike_value = (vix_close.iloc[0] / 100.0) ** 2 / 252.0
            strike_series.iloc[:] = strike_value
    else:
        for i, roll_date in enumerate(roll_dates):
            if roll_date in common_idx:
                roll_idx_pos = common_idx.get_loc(roll_date)
                curr_vix = vix_close.iloc[roll_idx_pos]
                strike_value = (curr_vix / 100.0) ** 2 / 252.0
                strike_series.iloc[roll_idx_pos:] = strike_value

    # Forward-fill to handle any gaps (in case of early gaps)
    strike_series = strike_series.ffill()
    # Back-fill if needed (handle any leading NaNs)
    strike_series = strike_series.bfill()

    # Gross P&L: Daily mark-to-market vs fixed strike
    # Daily P&L = notional * (realized_var_today - strike_fixed)
    # This is daily slicing against a fixed strike within each roll period
    pnl_gross = side * notional * (daily_var - strike_series)

    # Roll costs: only on roll dates (after first)
    roll_costs = pd.Series(0.0, index=common_idx)
    for i, roll_date in enumerate(roll_dates[1:], start=1):  # Skip first roll
        if roll_date in common_idx:
            roll_idx_pos = common_idx.get_loc(roll_date)
            prev_vix = vix_close.iloc[common_idx.get_loc(roll_dates[i - 1])]
            curr_vix = vix_close.iloc[roll_idx_pos]
            cost = compute_roll_cost(prev_vix, curr_vix, notional, bid_ask_vol_points)
            roll_costs.iloc[roll_idx_pos] = cost

    # Net P&L = Gross - Roll Costs
    pnl_net = pnl_gross - roll_costs

    # Cumulative P&L (drop first row NaN from shift)
    pnl_net_clean = pnl_net.dropna()
    if len(pnl_net_clean) > 0:
        cumulative = pnl_net_clean.cumsum()
        final_pnl = cumulative.iloc[-1]
    else:
        cumulative = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        final_pnl = 0.0

    logger.info(
        f"Backtest summary: {len(pnl_net_clean)} trading days, "
        f"final cumulative P&L: ${final_pnl:.2f}, "
        f"total roll costs: ${roll_costs.sum():.2f}, "
        f"num rolls: {len(roll_dates)}"
    )

    return {
        "pnl_series": pnl_net,
        "cumulative": cumulative,
        "pnl_gross": pnl_gross,
        "roll_costs": roll_costs,
        "strike": strike_series,
        "daily_var": daily_var,
        "synthetic_warning": spx_source.is_synthetic or vix_source.is_synthetic,
    }
