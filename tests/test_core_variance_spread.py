"""Tests for variance spread P&L calculations."""

import pytest

from src.core import daily_variance_pnl, variance_pnl_notional


class TestDailyVariancePnl:
    """Test daily variance spread P&L (before notional)."""

    def test_pnl_positive_when_realized_exceeds_implied(self):
        """P&L > 0 when realized vol > implied vol (long gamma wins)."""
        realized_var = 2e-4
        implied_var = 1e-4
        pnl = daily_variance_pnl(realized_var, implied_var)
        assert pnl > 0
        assert abs(pnl - 1e-4) < 1e-15

    def test_pnl_negative_when_implied_exceeds_realized(self):
        """P&L < 0 when implied vol > realized vol (short gamma wins)."""
        realized_var = 1e-4
        implied_var = 2e-4
        pnl = daily_variance_pnl(realized_var, implied_var)
        assert pnl < 0
        assert abs(pnl - (-1e-4)) < 1e-15

    def test_pnl_zero_when_equal(self):
        """P&L = 0 when realized == implied."""
        realized_var = 1.5e-4
        implied_var = 1.5e-4
        pnl = daily_variance_pnl(realized_var, implied_var)
        assert pnl == 0.0

    def test_pnl_hand_computed_scenario(self):
        """Hand-computed scenario from plan."""
        # S_100 -> S_101: r = ln(1.01) ≈ 0.00995
        r = 0.00995033085316
        realized_var = r ** 2  # ≈ 9.9009e-5
        # VIX=20: daily implied ≈ 1.5873e-4
        implied_var = 0.04 / 252  # ≈ 1.5873e-4
        # P&L = 9.9009e-5 - 1.5873e-4 ≈ -5.973e-5 (loss: realized < implied)
        pnl = daily_variance_pnl(realized_var, implied_var)
        expected = 9.9009e-5 - 1.5873e-4
        assert abs(pnl - expected) < 1e-8

    def test_pnl_zero_variances(self):
        """Both variances zero → P&L = 0."""
        pnl = daily_variance_pnl(0.0, 0.0)
        assert pnl == 0.0

    def test_pnl_linear_scaling(self):
        """P&L scales linearly with variance differences."""
        realized = 3e-4
        implied = 1e-4
        pnl1 = daily_variance_pnl(realized, implied)
        pnl2 = daily_variance_pnl(2 * realized, 2 * implied)
        assert abs(pnl2 - 2 * pnl1) < 1e-15


class TestVariancePnlNotional:
    """Test variance spread P&L with notional sizing."""

    def test_pnl_notional_explicit(self):
        """P&L with explicit notional."""
        realized_var = 2e-4
        implied_var = 1e-4
        notional = 100000
        pnl = variance_pnl_notional(realized_var, implied_var, notional=notional)
        # P&L = (2e-4 - 1e-4) * 100000 = 1e-4 * 100000 = 10
        expected = 1e-4 * 100000
        assert abs(pnl - expected) < 1e-6

    def test_pnl_notional_scaling(self):
        """P&L scales linearly with notional."""
        realized_var = 2e-4
        implied_var = 1e-4
        pnl1 = variance_pnl_notional(realized_var, implied_var, notional=100000)
        pnl2 = variance_pnl_notional(realized_var, implied_var, notional=200000)
        assert abs(pnl2 - 2 * pnl1) < 1e-6

    def test_pnl_notional_negative(self):
        """P&L can be negative."""
        realized_var = 1e-4
        implied_var = 2e-4
        pnl = variance_pnl_notional(realized_var, implied_var, notional=50000)
        assert pnl < 0
        # P&L = (1e-4 - 2e-4) * 50000 = -1e-4 * 50000 = -5
        expected = -1e-4 * 50000
        assert abs(pnl - expected) < 1e-6

    def test_pnl_notional_hand_computed(self):
        """Hand-computed P&L with notional."""
        realized_var = 9.9009e-5
        implied_var = 1.5873e-4
        notional = 100000
        pnl = variance_pnl_notional(realized_var, implied_var, notional=notional)
        # Spread = 9.9009e-5 - 1.5873e-4 = -5.973e-5
        # P&L = -5.973e-5 * 100000 ≈ -5.973
        expected = (realized_var - implied_var) * notional
        assert abs(pnl - expected) < 1e-6

    def test_pnl_notional_zero_notional(self):
        """Zero notional → zero P&L."""
        realized_var = 2e-4
        implied_var = 1e-4
        pnl = variance_pnl_notional(realized_var, implied_var, notional=0)
        assert pnl == 0.0

    def test_pnl_notional_large_notional(self):
        """Large notional for long gamma position."""
        realized_var = 2.5e-4
        implied_var = 1.5e-4
        notional = 1e6  # $1M notional
        pnl = variance_pnl_notional(realized_var, implied_var, notional=notional)
        # P&L = 1e-4 * 1e6 = 100
        expected = 1e-4 * 1e6
        assert abs(pnl - expected) < 1e-3
