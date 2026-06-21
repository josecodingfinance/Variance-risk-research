"""Tests for implied volatility calculations."""

import pytest

from src.core import vix_to_daily_variance


class TestVixToDailyVariance:
    """Test VIX to daily variance conversion."""

    def test_vix_20_to_daily_variance(self):
        """VIX=20 → daily variance ≈ 1.5873e-4 (hand-computed)."""
        vix = 20.0
        daily_var = vix_to_daily_variance(vix)

        # Formula: (VIX / 100)^2 / 252
        # (20 / 100)^2 / 252 = 0.04 / 252 ≈ 1.5873e-4
        expected = 0.04 / 252
        assert abs(daily_var - expected) < 1e-8

    def test_vix_10_to_daily_variance(self):
        """VIX=10 → daily variance = 0.01 / 252."""
        vix = 10.0
        daily_var = vix_to_daily_variance(vix)
        expected = 0.01 / 252
        assert abs(daily_var - expected) < 1e-10

    def test_vix_40_to_daily_variance(self):
        """VIX=40 → daily variance = 0.16 / 252."""
        vix = 40.0
        daily_var = vix_to_daily_variance(vix)
        expected = 0.16 / 252
        assert abs(daily_var - expected) < 1e-10

    def test_vix_zero(self):
        """VIX=0 → daily variance = 0."""
        vix = 0.0
        daily_var = vix_to_daily_variance(vix)
        assert daily_var == 0.0

    def test_vix_negative_raises(self):
        """VIX < 0 raises ValueError."""
        with pytest.raises(ValueError):
            vix_to_daily_variance(-5.0)

    def test_vix_returns_float(self):
        """Return type is float."""
        vix = 15.5
        daily_var = vix_to_daily_variance(vix)
        assert isinstance(daily_var, float)

    def test_vix_daily_variance_scales_with_vix_squared(self):
        """Daily variance scales as (VIX)^2."""
        vix1 = 10.0
        vix2 = 20.0
        var1 = vix_to_daily_variance(vix1)
        var2 = vix_to_daily_variance(vix2)

        # var2 / var1 should be (20/10)^2 = 4
        ratio = var2 / var1
        assert abs(ratio - 4.0) < 1e-10

    def test_vix_realistic_range(self):
        """Test across realistic VIX range (5-50)."""
        for vix in [5, 10, 15, 20, 25, 30, 40, 50]:
            daily_var = vix_to_daily_variance(vix)
            # Sanity check: should be small and positive
            assert 0 < daily_var < 0.01

    def test_vix_high_value(self):
        """Test with high VIX (e.g., crisis)."""
        vix = 80.0
        daily_var = vix_to_daily_variance(vix)
        expected = (80 / 100) ** 2 / 252
        assert abs(daily_var - expected) < 1e-10
