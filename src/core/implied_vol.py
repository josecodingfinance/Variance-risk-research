"""Implied volatility calculations from VIX."""


def vix_to_daily_variance(vix_close: float) -> float:
    """
    Convert VIX close to daily implied variance.

    The VIX is quoted as an annualized volatility (e.g., VIX = 20 means 20% annual vol).
    This function converts it to daily variance assuming 252 trading days/year.

    Formula: daily_variance = (VIX / 100)^2 / 252

    Args:
        vix_close: VIX close price (e.g., 20.0)

    Returns:
        Daily implied variance (e.g., ~1.5873e-4 for VIX=20)

    Limitations:
        - VIX^2 is not exactly the variance swap strike; it roughly approximates
          a 30-day realized variance contract, not 252-day.
        - This implementation assumes VIX is quoted as a 252-day annualized vol.
        - For more precision, use CBOE's formula (involves log-normal assumption).
    """
    if vix_close < 0:
        raise ValueError("VIX cannot be negative")

    # VIX is annualized volatility in percent
    # Convert to decimal: vix_close / 100
    # Square to get variance: (vix_close / 100)^2
    # De-annualize: divide by 252
    daily_variance = (vix_close / 100.0) ** 2 / 252.0

    return daily_variance
