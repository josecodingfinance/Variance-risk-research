"""Download and cache market data from yfinance with fallback to synthetic data."""

import logging
import pickle
import hashlib
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf

from .types import DataWithSource


logger = logging.getLogger(__name__)


def _ensure_cache_dir(cache_dir: str = "data/cache") -> Path:
    """Create cache directory if it doesn't exist."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def _get_cache_path(
    ticker: str, start: str, end: str, cache_dir: str = "data/cache"
) -> Path:
    """
    Return cache file path for a ticker + date range.

    Includes (ticker, start, end) in the key to prevent loading wrong date ranges.
    Uses MD5 hash to keep filenames short.
    """
    cache_dir_path = _ensure_cache_dir(cache_dir)
    key = f"{ticker}_{start}_{end}"
    key_hash = hashlib.md5(key.encode()).hexdigest()[:8]
    return cache_dir_path / f"{ticker}_{key_hash}.pkl"


def _validate_cache_metadata(df: pd.DataFrame, ticker: str, start: str, end: str) -> bool:
    """
    Validate that cached data matches requested date range.

    Returns False if date range mismatch detected.
    """
    if df.empty:
        return False
    df_start = df.index.min().strftime("%Y-%m-%d")
    df_end = df.index.max().strftime("%Y-%m-%d")
    # Check that cache covers requested range (within reason for business days)
    if df_start > start or df_end < end:
        logger.warning(
            f"Cache date mismatch for {ticker}: requested {start}:{end}, got {df_start}:{df_end}"
        )
        return False
    return True


def download_spx(
    start: str,
    end: str,
    cache_dir: str = "data/cache",
    use_cache: bool = True,
) -> DataWithSource:
    """
    Download SPX (^GSPC) daily OHLCV data.

    Falls back to synthetic data if download fails. All returns are never interpolated;
    missing rows are dropped.

    Args:
        start: Start date as 'YYYY-MM-DD'
        end: End date as 'YYYY-MM-DD'
        cache_dir: Cache directory for data
        use_cache: If True, load from cache; if False, force re-download

    Returns:
        DataWithSource(df, is_synthetic) — DataFrame + flag
    """
    ticker = "^GSPC"
    cache_path = _get_cache_path(ticker, start, end, cache_dir)

    # Try to load from cache
    if use_cache and cache_path.exists():
        try:
            logger.info(f"Loading {ticker} from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            if _validate_cache_metadata(data, ticker, start, end):
                return DataWithSource(df=data, is_synthetic=False)
            else:
                logger.info(f"Cache invalid for {ticker}. Re-downloading.")
        except Exception as e:
            logger.warning(f"Cache load failed for {ticker}: {e}. Re-downloading.")

    # Try to download
    try:
        logger.info(f"Downloading {ticker} from {start} to {end}")
        data = yf.download(ticker, start=start, end=end, progress=False)

        # Flatten MultiIndex columns if present (yfinance quirk for single ticker)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        # Drop rows with NaN returns (no interpolation)
        data = data.dropna()

        if data.empty:
            raise ValueError(f"No data returned for {ticker}")

        # Cache it
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"Cached {ticker} to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache {ticker}: {e}")

        return DataWithSource(df=data, is_synthetic=False)

    except Exception as e:
        logger.error(f"Download failed for {ticker}: {e}. Generating synthetic data.")
        synthetic_data = _generate_synthetic_spx(start=start, end=end)
        return DataWithSource(df=synthetic_data, is_synthetic=True)


def download_vix(
    start: str,
    end: str,
    cache_dir: str = "data/cache",
    use_cache: bool = True,
) -> DataWithSource:
    """
    Download VIX (^VIX) daily OHLCV data.

    Falls back to synthetic data if download fails. All returns are never interpolated;
    missing rows are dropped.

    Args:
        start: Start date as 'YYYY-MM-DD'
        end: End date as 'YYYY-MM-DD'
        cache_dir: Cache directory for data
        use_cache: If True, load from cache; if False, force re-download

    Returns:
        DataWithSource(df, is_synthetic) — DataFrame + flag
    """
    ticker = "^VIX"
    cache_path = _get_cache_path(ticker, start, end, cache_dir)

    # Try to load from cache
    if use_cache and cache_path.exists():
        try:
            logger.info(f"Loading {ticker} from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            if _validate_cache_metadata(data, ticker, start, end):
                return DataWithSource(df=data, is_synthetic=False)
            else:
                logger.info(f"Cache invalid for {ticker}. Re-downloading.")
        except Exception as e:
            logger.warning(f"Cache load failed for {ticker}: {e}. Re-downloading.")

    # Try to download
    try:
        logger.info(f"Downloading {ticker} from {start} to {end}")
        data = yf.download(ticker, start=start, end=end, progress=False)

        # Flatten MultiIndex columns if present (yfinance quirk for single ticker)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        # Drop rows with NaN (no interpolation)
        data = data.dropna()

        if data.empty:
            raise ValueError(f"No data returned for {ticker}")

        # Cache it
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"Cached {ticker} to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache {ticker}: {e}")

        return DataWithSource(df=data, is_synthetic=False)

    except Exception as e:
        logger.error(f"Download failed for {ticker}: {e}. Generating synthetic data.")
        synthetic_data = _generate_synthetic_vix(start=start, end=end)
        return DataWithSource(df=synthetic_data, is_synthetic=True)


def _generate_synthetic_spx(start: str, end: str) -> pd.DataFrame:
    """
    Generate synthetic SPX data with realistic statistics.

    CLEARLY labeled as synthetic. Never use for real trading.
    Replaces lost yfinance downloads so pipeline can run during dev.

    Seed is deterministic from (start, end) → reproducible for testing.
    OHLC is consistent: High >= max(O,C), Low <= min(O,C).
    """
    # Reproducible seed from params
    seed = hash((start, end, "spx")) % (2**32)
    rng = np.random.default_rng(seed)

    date_range = pd.date_range(start=start, end=end, freq="B")  # Business days
    n = len(date_range)

    # Realistic SPX-like returns: ~10% annual, ~15% vol
    daily_returns = rng.normal(loc=0.0004, scale=0.01, size=n)
    close_prices = 5000 * np.exp(np.cumsum(daily_returns))

    opens = close_prices * (1 + rng.normal(0, 0.003, n))

    # High: max(Open, Close) + intraday range
    highs = np.maximum(opens, close_prices) + np.abs(
        rng.normal(0, 0.005, n)
    ) * np.maximum(opens, close_prices)

    # Low: min(Open, Close) - intraday range
    lows = np.minimum(opens, close_prices) - np.abs(
        rng.normal(0, 0.005, n)
    ) * np.minimum(opens, close_prices)

    synthetic_data = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": close_prices,
            "Volume": rng.uniform(1e8, 5e8, n),
        },
        index=date_range,
    )

    logger.warning("⚠️ [SYNTHETIC - NO REAL DATA] for SPX")
    return synthetic_data


def _generate_synthetic_vix(start: str, end: str) -> pd.DataFrame:
    """
    Generate synthetic VIX data with realistic statistics.

    CLEARLY labeled as synthetic. Never use for real trading.
    VIX mean ~15-20, vol ~4-5, mean reversion, occasional spikes.

    Seed is deterministic from (start, end) → reproducible for testing.
    OHLC is consistent: High >= max(O,C), Low <= min(O,C).
    """
    # Reproducible seed from params
    seed = hash((start, end, "vix")) % (2**32)
    rng = np.random.default_rng(seed)

    date_range = pd.date_range(start=start, end=end, freq="B")  # Business days
    n = len(date_range)

    # Mean-reverting VIX-like series
    vix_values = np.zeros(n)
    vix_values[0] = 15
    for i in range(1, n):
        shock = rng.normal(0, 1.5)
        spike = 10 if rng.random() < 0.01 else 0
        vix_values[i] = 0.95 * vix_values[i - 1] + 0.05 * 15 + shock + spike

    vix_values = np.clip(vix_values, 8, 80)

    opens = vix_values * (1 + rng.normal(0, 0.02, n))

    # High: max(Open, Close) + intraday range
    highs = np.maximum(opens, vix_values) + np.abs(
        rng.normal(0, 0.03, n)
    ) * np.maximum(opens, vix_values)

    # Low: min(Open, Close) - intraday range
    lows = np.minimum(opens, vix_values) - np.abs(
        rng.normal(0, 0.03, n)
    ) * np.minimum(opens, vix_values)

    synthetic_data = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": vix_values,
            "Volume": rng.uniform(1e6, 5e7, n),
        },
        index=date_range,
    )

    logger.warning("⚠️ [SYNTHETIC - NO REAL DATA] for VIX")
    return synthetic_data


def load_data(
    start: str,
    end: str,
    cache_dir: str = "data/cache",
    use_cache: bool = True,
) -> tuple[DataWithSource, DataWithSource]:
    """
    Load SPX and VIX data in parallel.

    Args:
        start: Start date as 'YYYY-MM-DD'
        end: End date as 'YYYY-MM-DD'
        cache_dir: Cache directory
        use_cache: Use cache if available

    Returns:
        Tuple of (spx_data_source, vix_data_source)
        Both are DataWithSource objects. is_synthetic is True if either dataset is synthetic.
    """
    spx_source = download_spx(
        start=start, end=end, cache_dir=cache_dir, use_cache=use_cache
    )
    vix_source = download_vix(
        start=start, end=end, cache_dir=cache_dir, use_cache=use_cache
    )

    # Align dates (inner join on index)
    common_dates = spx_source.df.index.intersection(vix_source.df.index)
    logger.info(f"Aligned data: {len(common_dates)} trading days")

    # Mark as synthetic if either source is synthetic
    is_synthetic = spx_source.is_synthetic or vix_source.is_synthetic

    return (
        DataWithSource(df=spx_source.df.loc[common_dates], is_synthetic=is_synthetic),
        DataWithSource(df=vix_source.df.loc[common_dates], is_synthetic=is_synthetic),
    )
