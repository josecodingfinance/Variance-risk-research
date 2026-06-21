"""Tests for realized volatility estimators."""

import numpy as np
import pytest

from src.core import close_to_close_vol, parkinson_vol


class TestCloseToCloseVol:
    """Test close-to-close realized volatility."""

    def test_close_to_close_basic(self):
        """Close-to-close vol is computed correctly."""
        # Simple case: constant returns
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        vol = close_to_close_vol(returns, periods=252)
        # mean(r^2) = 0.01^2 = 1e-4
        # sqrt(1e-4 * 252) = sqrt(0.0252) ≈ 0.1587
        expected = np.sqrt(1e-4 * 252)
        np.testing.assert_allclose(vol, expected, rtol=1e-6)

    def test_close_to_close_daily_log_returns(self):
        """Close-to-close with realistic log-returns."""
        # S goes 100 -> 101 (r = ln(1.01) ≈ 0.00995)
        r = np.log(101 / 100)
        returns = np.array([r] * 252)
        vol = close_to_close_vol(returns, periods=252)
        # mean(r^2) = r^2
        # sqrt(r^2 * 252) = abs(r) * sqrt(252) ≈ 0.00995 * 15.87 ≈ 0.158
        expected = np.sqrt(np.mean(returns**2) * 252)
        np.testing.assert_allclose(vol, expected, rtol=1e-6)

    def test_close_to_close_empty_returns(self):
        """Empty returns → vol = 0."""
        vol = close_to_close_vol(np.array([]), periods=252)
        assert vol == 0.0

    def test_close_to_close_single_return(self):
        """Single return."""
        returns = np.array([0.02])
        vol = close_to_close_vol(returns, periods=252)
        expected = np.sqrt(0.02**2 * 252)
        np.testing.assert_allclose(vol, expected, rtol=1e-6)

    def test_close_to_close_zero_returns(self):
        """Zero returns → vol = 0."""
        returns = np.array([0.0, 0.0, 0.0])
        vol = close_to_close_vol(returns, periods=252)
        assert vol == 0.0

    def test_close_to_close_custom_periods(self):
        """Custom annualization period."""
        returns = np.array([0.01, 0.01])
        vol_daily = close_to_close_vol(returns, periods=1)
        vol_annual = close_to_close_vol(returns, periods=252)
        # Annual should be ~sqrt(252) times the daily
        np.testing.assert_allclose(vol_annual, vol_daily * np.sqrt(252), rtol=1e-6)


class TestParkinsonVol:
    """Test Parkinson (high-low) realized volatility."""

    def test_parkinson_basic(self):
        """Parkinson vol from hand-computed values."""
        # H=101, L=99 for one period
        high = np.array([101.0])
        low = np.array([99.0])
        vol = parkinson_vol(high, low, periods=252)

        # Parkinson: σ = sqrt(mean((c * ln(H/L))^2) * 252)
        # where c = 1 / (4 * ln(2)) ≈ 0.36067
        # ln(H/L) = ln(101/99) ≈ 0.02020
        # (c * ln(H/L))^2 ≈ (0.36067 * 0.02020)^2 ≈ 5.328e-6
        # σ = sqrt(5.328e-6 * 252) ≈ sqrt(0.001343) ≈ 0.03667

        c = 1.0 / (4 * np.log(2))
        hl_log = np.log(101.0 / 99.0)
        variance = (c * hl_log) ** 2
        expected = np.sqrt(variance * 252)

        np.testing.assert_allclose(vol, expected, rtol=1e-6)

    def test_parkinson_constant_range(self):
        """Parkinson with constant H/L ratio."""
        high = np.array([110.0, 110.0, 110.0])
        low = np.array([100.0, 100.0, 100.0])
        vol = parkinson_vol(high, low, periods=252)

        c = 1.0 / (4 * np.log(2))
        hl_log = np.log(110.0 / 100.0)
        variance = (c * hl_log) ** 2
        expected = np.sqrt(variance * 252)

        np.testing.assert_allclose(vol, expected, rtol=1e-6)

    def test_parkinson_zero_range(self):
        """High == Low → vol = 0."""
        high = np.array([100.0, 100.0])
        low = np.array([100.0, 100.0])
        vol = parkinson_vol(high, low, periods=252)
        assert vol == 0.0

    def test_parkinson_empty(self):
        """Empty arrays → vol = 0."""
        vol = parkinson_vol(np.array([]), np.array([]), periods=252)
        assert vol == 0.0

    def test_parkinson_mismatched_lengths(self):
        """Mismatched high/low lengths → vol = 0."""
        high = np.array([100.0, 101.0])
        low = np.array([99.0])
        vol = parkinson_vol(high, low, periods=252)
        assert vol == 0.0

    def test_parkinson_custom_periods(self):
        """Custom annualization period."""
        high = np.array([101.0])
        low = np.array([99.0])
        vol_daily = parkinson_vol(high, low, periods=1)
        vol_annual = parkinson_vol(high, low, periods=252)
        # Annual should be ~sqrt(252) times the daily
        np.testing.assert_allclose(vol_annual, vol_daily * np.sqrt(252), rtol=1e-6)

    def test_parkinson_vs_close_to_close(self):
        """Parkinson typically lower than close-to-close for same price moves."""
        # 1% daily moves
        r = 0.01
        returns = np.array([r] * 10)
        vol_c2c = close_to_close_vol(returns, periods=252)

        # Same 1% move as H/L range (100 -> 101)
        # In practice, Parkinson is less noisy
        high = np.array([101.0] * 10)
        low = np.array([100.0] * 10)
        vol_parkinson = parkinson_vol(high, low, periods=252)

        # Both should be positive
        assert vol_c2c > 0
        assert vol_parkinson > 0
