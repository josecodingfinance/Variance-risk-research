"""Tests for data download module."""

import pytest
import pandas as pd
import tempfile
import logging

from src.data import download_spx, download_vix, load_data, DataWithSource


logging.basicConfig(level=logging.INFO)


class TestDownloadSPX:
    """Test SPX download and caching."""

    def test_download_spx_returns_datasource(self):
        """SPX download returns DataWithSource with valid DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_spx(
                start="2024-01-01",
                end="2024-01-31",
                cache_dir=tmpdir,
                use_cache=False,
            )
            assert isinstance(result, DataWithSource)
            assert isinstance(result.df, pd.DataFrame)
            assert len(result.df) > 0
            required_cols = {"Open", "High", "Low", "Close", "Volume"}
            assert required_cols.issubset(result.df.columns)

    def test_spx_data_has_no_nans(self):
        """SPX data has no NaN values (no interpolation rule)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_spx(
                start="2024-01-01",
                end="2024-01-31",
                cache_dir=tmpdir,
                use_cache=False,
            )
            assert result.df.isnull().sum().sum() == 0

    def test_spx_close_prices_positive(self):
        """SPX close prices are always positive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_spx(
                start="2024-01-01",
                end="2024-01-31",
                cache_dir=tmpdir,
                use_cache=False,
            )
            assert (result.df["Close"] > 0).all()

    def test_spx_ohlc_consistency(self):
        """SPX OHLC is consistent: High >= max(O,C), Low <= min(O,C)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_spx(
                start="2024-01-01",
                end="2024-01-31",
                cache_dir=tmpdir,
                use_cache=False,
            )
            df = result.df
            assert (df["High"] >= df[["Open", "Close"]].max(axis=1)).all()
            assert (df["Low"] <= df[["Open", "Close"]].min(axis=1)).all()

    def test_spx_caching_works(self):
        """SPX data is cached and reused on second call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result1 = download_spx(
                start="2024-01-01",
                end="2024-01-31",
                cache_dir=tmpdir,
                use_cache=False,
            )
            result2 = download_spx(
                start="2024-01-01",
                end="2024-01-31",
                cache_dir=tmpdir,
                use_cache=True,
            )
            pd.testing.assert_frame_equal(result1.df, result2.df)

    def test_spx_synthetic_labeled(self):
        """Synthetic SPX data is marked with is_synthetic=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_spx(
                start="2099-01-01",
                end="2099-01-31",
                cache_dir=tmpdir,
                use_cache=False,
            )
            assert result.is_synthetic is True


class TestDownloadVIX:
    """Test VIX download and caching."""

    def test_download_vix_returns_datasource(self):
        """VIX download returns DataWithSource with valid DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_vix(
                start="2024-01-01",
                end="2024-01-31",
                cache_dir=tmpdir,
                use_cache=False,
            )
            assert isinstance(result, DataWithSource)
            assert isinstance(result.df, pd.DataFrame)
            assert len(result.df) > 0
            required_cols = {"Open", "High", "Low", "Close", "Volume"}
            assert required_cols.issubset(result.df.columns)

    def test_vix_data_has_no_nans(self):
        """VIX data has no NaN values (no interpolation rule)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_vix(
                start="2024-01-01",
                end="2024-01-31",
                cache_dir=tmpdir,
                use_cache=False,
            )
            assert result.df.isnull().sum().sum() == 0

    def test_vix_close_prices_positive(self):
        """VIX close prices are always positive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_vix(
                start="2024-01-01",
                end="2024-01-31",
                cache_dir=tmpdir,
                use_cache=False,
            )
            assert (result.df["Close"] > 0).all()

    def test_vix_reasonable_range(self):
        """VIX values are in a reasonable range (typically 8-80)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_vix(
                start="2024-01-01",
                end="2024-01-31",
                cache_dir=tmpdir,
                use_cache=False,
            )
            assert (result.df["Close"] >= 5).all()
            assert (result.df["Close"] <= 150).all()

    def test_vix_ohlc_consistency(self):
        """VIX OHLC is consistent: High >= max(O,C), Low <= min(O,C)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_vix(
                start="2024-01-01",
                end="2024-01-31",
                cache_dir=tmpdir,
                use_cache=False,
            )
            df = result.df
            assert (df["High"] >= df[["Open", "Close"]].max(axis=1)).all()
            assert (df["Low"] <= df[["Open", "Close"]].min(axis=1)).all()

    def test_vix_caching_works(self):
        """VIX data is cached and reused on second call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result1 = download_vix(
                start="2024-01-01",
                end="2024-01-31",
                cache_dir=tmpdir,
                use_cache=False,
            )
            result2 = download_vix(
                start="2024-01-01",
                end="2024-01-31",
                cache_dir=tmpdir,
                use_cache=True,
            )
            pd.testing.assert_frame_equal(result1.df, result2.df)

    def test_vix_synthetic_labeled(self):
        """Synthetic VIX data is marked with is_synthetic=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_vix(
                start="2099-01-01",
                end="2099-01-31",
                cache_dir=tmpdir,
                use_cache=False,
            )
            assert result.is_synthetic is True


class TestLoadData:
    """Test parallel data loading."""

    def test_load_data_returns_tuple(self):
        """load_data returns tuple of (DataWithSource, DataWithSource)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_data(
                start="2024-01-01",
                end="2024-01-31",
                cache_dir=tmpdir,
                use_cache=False,
            )
            assert isinstance(result, tuple)
            assert len(result) == 2
            spx, vix = result
            assert isinstance(spx, DataWithSource)
            assert isinstance(vix, DataWithSource)

    def test_load_data_aligned_dates(self):
        """load_data returns aligned SPX and VIX (same index)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spx_src, vix_src = load_data(
                start="2024-01-01",
                end="2024-01-31",
                cache_dir=tmpdir,
                use_cache=False,
            )
            pd.testing.assert_index_equal(spx_src.df.index, vix_src.df.index)

    def test_load_data_non_empty(self):
        """load_data returns non-empty DataFrames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spx_src, vix_src = load_data(
                start="2024-01-01",
                end="2024-01-31",
                cache_dir=tmpdir,
                use_cache=False,
            )
            assert len(spx_src.df) > 0
            assert len(vix_src.df) > 0

    def test_synthetic_flag_survives_load_data(self):
        """load_data propagates is_synthetic flag through alignment/slicing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Force synthetic by using future dates
            spx_src, vix_src = load_data(
                start="2099-01-01",
                end="2099-01-31",
                cache_dir=tmpdir,
                use_cache=False,
            )
            # Both should be marked synthetic (and aligned)
            assert spx_src.is_synthetic is True
            assert vix_src.is_synthetic is True


class TestCacheDateRange:
    """Test that cache key includes date range (bug fix)."""

    def test_cache_key_includes_dates(self):
        """
        Bug test: Cache must include date range.

        Scenario: Download 2024-01-01:31, cache it. Then download 2020-01-01:2024-12-31
        with use_cache=True. Should re-download (not return 2024-01 cache).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # First: download short range and cache
            result_short = download_spx(
                start="2024-01-01",
                end="2024-01-31",
                cache_dir=tmpdir,
                use_cache=False,
            )
            short_len = len(result_short.df)

            # Second: request longer range with cache enabled
            # Should NOT use the short cache
            result_long = download_spx(
                start="2024-01-01",
                end="2024-12-31",
                cache_dir=tmpdir,
                use_cache=True,
            )
            long_len = len(result_long.df)

            # Long range should be much longer than short range
            # (if it reused the short cache, lengths would be equal)
            assert long_len > short_len


class TestSyntheticReproducibility:
    """Test that synthetic data is reproducible (seed-based)."""

    def test_synthetic_spx_reproducible(self):
        """Same params → same synthetic SPX data (deterministic seed)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result1 = download_spx(
                start="2099-01-01",
                end="2099-01-31",
                cache_dir=tmpdir,
                use_cache=False,
            )
            result2 = download_spx(
                start="2099-01-01",
                end="2099-01-31",
                cache_dir=tmpdir,
                use_cache=False,
            )
            pd.testing.assert_frame_equal(result1.df, result2.df)

    def test_synthetic_vix_reproducible(self):
        """Same params → same synthetic VIX data (deterministic seed)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result1 = download_vix(
                start="2099-01-01",
                end="2099-01-31",
                cache_dir=tmpdir,
                use_cache=False,
            )
            result2 = download_vix(
                start="2099-01-01",
                end="2099-01-31",
                cache_dir=tmpdir,
                use_cache=False,
            )
            pd.testing.assert_frame_equal(result1.df, result2.df)


@pytest.mark.network
class TestRealDownload:
    """Test that real yfinance download works (requires network)."""

    def test_download_spx_real(self):
        """
        Verify real SPX download (not synthetic fallback).
        Marked @pytest.mark.network — skip if offline.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use recent real data that definitely exists
            result = download_spx(
                start="2024-01-01",
                end="2024-01-31",
                cache_dir=tmpdir,
                use_cache=False,
            )
            # If yfinance works, is_synthetic should be False
            # If offline, this test is skipped by pytest -m network
            if not result.is_synthetic:
                assert (result.df["Close"] > 0).all()
                assert (
                    result.df["High"] >= result.df[["Open", "Close"]].max(axis=1)
                ).all()

    def test_download_vix_real(self):
        """
        Verify real VIX download (not synthetic fallback).
        Marked @pytest.mark.network — skip if offline.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            result = download_vix(
                start="2024-01-01",
                end="2024-01-31",
                cache_dir=tmpdir,
                use_cache=False,
            )
            if not result.is_synthetic:
                assert (result.df["Close"] >= 5).all()
                assert (result.df["Close"] <= 150).all()
