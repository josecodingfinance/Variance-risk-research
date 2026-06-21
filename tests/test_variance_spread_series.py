"""Tests for vectorized variance spread P&L with automatic lagging."""

import pytest
import pandas as pd
import numpy as np

from src.core import variance_pnl_series


class TestVariancePnlSeries:
    """Test vectorized variance_pnl_series with shift."""

    def test_variance_pnl_series_basic(self):
        """Basic series P&L calculation."""
        realized_var = pd.Series([1e-4, 2e-4, 1.5e-4], index=pd.date_range("2024-01-01", periods=3))
        implied_var = pd.Series([1.5e-4, 1.5e-4, 1e-4], index=pd.date_range("2024-01-01", periods=3))

        pnl = variance_pnl_series(realized_var, implied_var, side=1, notional=100000)

        # First row is NaN (from shift)
        assert pd.isna(pnl.iloc[0])
        # Second: realized[1]=2e-4, implied_prev[1]=implied[0]=1.5e-4
        # spread = 2e-4 - 1.5e-4 = 0.5e-4
        # pnl = 1 * 0.5e-4 * 100000 = 5
        assert abs(pnl.iloc[1] - 5.0) < 1e-6
        # Third: realized[2]=1.5e-4, implied_prev[2]=implied[1]=1.5e-4
        # spread = 0, pnl = 0
        assert abs(pnl.iloc[2]) < 1e-10

    def test_variance_pnl_series_shift_lag(self):
        """First row is NaN due to shift."""
        realized_var = pd.Series([1e-4, 2e-4])
        implied_var = pd.Series([1e-4, 1e-4])

        pnl = variance_pnl_series(realized_var, implied_var, notional=1.0)

        assert pd.isna(pnl.iloc[0])
        assert not pd.isna(pnl.iloc[1])

    def test_variance_pnl_series_side_long(self):
        """Long-gamma (side=+1): profit when realized > implied."""
        realized_var = pd.Series([2e-4, 3e-4])
        implied_var = pd.Series([1e-4, 1e-4])

        pnl_long = variance_pnl_series(realized_var, implied_var, side=1, notional=100000)

        # Second row: (3e-4 - 1e-4) * 100000 = 20
        assert pnl_long.iloc[1] > 0

    def test_variance_pnl_series_side_short(self):
        """Short-gamma (side=-1): profit when implied > realized."""
        realized_var = pd.Series([1e-4, 2e-4])
        implied_var = pd.Series([2e-4, 2e-4])

        pnl_short = variance_pnl_series(realized_var, implied_var, side=-1, notional=100000)

        # Second row: -1 * (2e-4 - 2e-4) * 100000 = 0
        # But spread = 2e-4 - 2e-4 = 0, so pnl = 0
        # Let's use realized[1]=0.5e-4, implied[0]=2e-4
        realized_var2 = pd.Series([1e-4, 0.5e-4])
        implied_var2 = pd.Series([2e-4, 2e-4])
        pnl_short2 = variance_pnl_series(realized_var2, implied_var2, side=-1, notional=100000)
        # spread = 0.5e-4 - 2e-4 = -1.5e-4
        # pnl = -1 * (-1.5e-4) * 100000 = 15 (profit)
        assert pnl_short2.iloc[1] > 0

    def test_variance_pnl_series_notional_scaling(self):
        """P&L scales linearly with notional."""
        realized_var = pd.Series([2e-4, 3e-4])
        implied_var = pd.Series([1e-4, 1e-4])

        pnl1 = variance_pnl_series(realized_var, implied_var, notional=100000)
        pnl2 = variance_pnl_series(realized_var, implied_var, notional=200000)

        # pnl2 should be 2x pnl1 (ignoring NaN)
        assert abs(pnl2.iloc[1] - 2 * pnl1.iloc[1]) < 1e-6

    def test_variance_pnl_series_no_look_ahead(self):
        """Implied var is lagged; no use of tomorrow's VIX."""
        # If implied_var at day 2 is very different, it shouldn't affect day 1 P&L
        realized_var = pd.Series([1e-4, 2e-4, 3e-4])
        implied_var = pd.Series([1e-4, 1e-4, 10e-4])  # Jump at day 3

        pnl = variance_pnl_series(realized_var, implied_var, notional=1.0)

        # Day 2: spread = 2e-4 - 1e-4 = 1e-4, pnl = 1e-4
        # The jump at day 3 in implied_var doesn't affect day 2
        assert abs(pnl.iloc[1] - 1e-4) < 1e-15

    def test_variance_pnl_series_aligned_indices(self):
        """Mismatched indices are aligned (inner join)."""
        realized_var = pd.Series(
            [1e-4, 2e-4, 1.5e-4],
            index=pd.date_range("2024-01-01", periods=3),
        )
        # Missing one date
        implied_var = pd.Series(
            [1.5e-4, 1e-4],
            index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"]),
        )

        pnl = variance_pnl_series(realized_var, implied_var, notional=100000)

        # Only 2024-01-02 and 2024-01-03 are common
        # 2024-01-02: second row of pnl should be NaN (first after shift)
        # Actually, after alignment, only 2024-01-02 and 2024-01-03 exist
        # So pnl[0] = NaN (shift), pnl[1] should have a value
        assert len(pnl) == 2

    def test_variance_pnl_series_empty_indices(self):
        """No common indices → empty Series."""
        realized_var = pd.Series(
            [1e-4, 2e-4],
            index=pd.date_range("2024-01-01", periods=2),
        )
        implied_var = pd.Series(
            [1e-4, 1e-4],
            index=pd.date_range("2024-02-01", periods=2),
        )

        pnl = variance_pnl_series(realized_var, implied_var)

        assert len(pnl) == 0

    def test_variance_pnl_series_invalid_side(self):
        """Invalid side raises ValueError."""
        realized_var = pd.Series([1e-4, 2e-4])
        implied_var = pd.Series([1e-4, 1e-4])

        with pytest.raises(ValueError):
            variance_pnl_series(realized_var, implied_var, side=0)

        with pytest.raises(ValueError):
            variance_pnl_series(realized_var, implied_var, side=2)
        """Hand-computed scenario from plan."""
        # S_100 -> S_101: r = ln(1.01), r^2 ≈ 9.9009e-5
        # VIX=20: daily implied ≈ 1.5873e-4
        # Day 1: realized=1e-4, implied=1.5873e-4
        # Day 2: realized=9.9009e-5, implied_prev=1.5873e-4
        # spread = 9.9009e-5 - 1.5873e-4 = -5.973e-5
        # pnl = 1 * (-5.973e-5) * 100000 ≈ -5.973
        realized_var = pd.Series([1e-4, 9.9009e-5])
        implied_var = pd.Series([1.5873e-4, 1.5873e-4])

        pnl = variance_pnl_series(realized_var, implied_var, side=1, notional=100000)

        expected = -5.973
        assert abs(pnl.iloc[1] - expected) < 0.01
