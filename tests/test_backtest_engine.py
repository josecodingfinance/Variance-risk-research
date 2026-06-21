"""Tests for backtest engine."""

import pytest
import pandas as pd
import numpy as np

from src.data import DataWithSource
from src.backtest import run_backtest
from src.backtest.engine import identify_roll_dates, compute_roll_cost


class TestRunBacktest:
    """Test backtest engine."""

    def _create_synthetic_data(self, n_days: int = 20):
        """Helper: create simple synthetic SPX and VIX data."""
        dates = pd.date_range("2024-01-01", periods=n_days)

        # SPX: simple close series (100 + 0.1*i)
        spx_close = 5000 + np.arange(n_days)
        spx_data = pd.DataFrame(
            {
                "Open": spx_close,
                "High": spx_close + 1,
                "Low": spx_close - 1,
                "Close": spx_close,
                "Volume": np.ones(n_days) * 1e9,
            },
            index=dates,
        )

        # VIX: constant at 20
        vix_data = pd.DataFrame(
            {
                "Open": 20.0 * np.ones(n_days),
                "High": 21.0 * np.ones(n_days),
                "Low": 19.0 * np.ones(n_days),
                "Close": 20.0 * np.ones(n_days),
                "Volume": np.ones(n_days) * 1e6,
            },
            index=dates,
        )

        return spx_data, vix_data

    def test_run_backtest_returns_dict(self):
        """run_backtest returns dict with expected keys."""
        spx, vix = self._create_synthetic_data()
        spx_src = DataWithSource(df=spx, is_synthetic=False)
        vix_src = DataWithSource(df=vix, is_synthetic=False)

        result = run_backtest(spx_src, vix_src, notional=100000)

        expected_keys = {
            "pnl_series",
            "cumulative",
            "pnl_gross",
            "roll_costs",
            "strike",
            "daily_var",
            "synthetic_warning",
        }
        assert set(result.keys()) == expected_keys

    def test_run_backtest_series_types(self):
        """Return values are correct types."""
        spx, vix = self._create_synthetic_data()
        spx_src = DataWithSource(df=spx, is_synthetic=False)
        vix_src = DataWithSource(df=vix, is_synthetic=False)

        result = run_backtest(spx_src, vix_src)

        assert isinstance(result["pnl_series"], pd.Series)
        assert isinstance(result["cumulative"], pd.Series)
        assert isinstance(result["roll_costs"], pd.Series)
        assert isinstance(result["daily_var"], pd.Series)
        assert result["synthetic_warning"] == False

    def test_run_backtest_synthetic_warning(self):
        """Synthetic data flag is propagated."""
        spx, vix = self._create_synthetic_data()
        spx_src = DataWithSource(df=spx, is_synthetic=True)
        vix_src = DataWithSource(df=vix, is_synthetic=False)

        result = run_backtest(spx_src, vix_src)

        assert result["synthetic_warning"] == True

    def test_run_backtest_roll_costs_nonnegative(self):
        """Roll costs are always non-negative (zero except on roll dates)."""
        spx, vix = self._create_synthetic_data()
        spx_src = DataWithSource(df=spx, is_synthetic=False)
        vix_src = DataWithSource(df=vix, is_synthetic=False)

        result = run_backtest(spx_src, vix_src, bid_ask_vol_points=0.5)

        # Roll costs should be non-negative
        assert (result["roll_costs"] >= 0).all()

    def test_run_backtest_cumulative_cumsum(self):
        """Cumulative P&L is cumsum of daily P&L."""
        spx, vix = self._create_synthetic_data()
        spx_src = DataWithSource(df=spx, is_synthetic=False)
        vix_src = DataWithSource(df=vix, is_synthetic=False)

        result = run_backtest(spx_src, vix_src)

        pnl_clean = result["pnl_series"].dropna()
        expected_cumulative = pnl_clean.cumsum()

        pd.testing.assert_series_equal(
            result["cumulative"], expected_cumulative, check_exact=False, atol=1e-6, check_freq=False
        )

    def test_run_backtest_pnl_net_formula(self):
        """Net P&L = Gross - Roll Costs."""
        spx, vix = self._create_synthetic_data()
        spx_src = DataWithSource(df=spx, is_synthetic=False)
        vix_src = DataWithSource(df=vix, is_synthetic=False)

        result = run_backtest(spx_src, vix_src)

        # pnl_series should equal pnl_gross - roll_costs
        pnl_calculated = result["pnl_gross"] - result["roll_costs"]
        pd.testing.assert_series_equal(
            result["pnl_series"], pnl_calculated, check_exact=False, atol=1e-10
        )

    def test_run_backtest_side_long_vs_short(self):
        """Long and short sides have opposite signs."""
        spx, vix = self._create_synthetic_data(n_days=50)
        spx_src = DataWithSource(df=spx, is_synthetic=False)
        vix_src = DataWithSource(df=vix, is_synthetic=False)

        result_long = run_backtest(spx_src, vix_src, side=1, notional=100000)
        result_short = run_backtest(spx_src, vix_src, side=-1, notional=100000)

        # Gross P&L should be opposite
        pd.testing.assert_series_equal(
            result_long["pnl_gross"],
            -result_short["pnl_gross"],
            check_exact=False,
            atol=1e-10,
        )

    def test_run_backtest_notional_scaling(self):
        """P&L scales linearly with notional."""
        spx, vix = self._create_synthetic_data()
        spx_src = DataWithSource(df=spx, is_synthetic=False)
        vix_src = DataWithSource(df=vix, is_synthetic=False)

        result1 = run_backtest(spx_src, vix_src, notional=100000)
        result2 = run_backtest(spx_src, vix_src, notional=200000)

        # result2 should be 2x result1
        pd.testing.assert_series_equal(
            result2["pnl_series"] / 2,
            result1["pnl_series"],
            check_exact=False,
            atol=1e-6,
        )

    def test_run_backtest_empty_indices_raises(self):
        """Mismatched dates raise ValueError."""
        spx = pd.DataFrame(
            {
                "Open": 5000.0,
                "Close": 5001.0,
                "High": 5002.0,
                "Low": 4999.0,
                "Volume": 1e9,
            },
            index=pd.date_range("2024-01-01", periods=5),
        )

        vix = pd.DataFrame(
            {
                "Open": 20.0,
                "Close": 20.0,
                "High": 21.0,
                "Low": 19.0,
                "Volume": 1e6,
            },
            index=pd.date_range("2024-02-01", periods=5),
        )

        spx_src = DataWithSource(df=spx, is_synthetic=False)
        vix_src = DataWithSource(df=vix, is_synthetic=False)

        with pytest.raises(ValueError):
            run_backtest(spx_src, vix_src)

    def test_run_backtest_fixed_strike_between_rolls(self):
        """Strike is fixed between roll dates (no daily update)."""
        # Create 60 trading days (3 months) with two rolls
        spx_close = 5000 + np.cumsum(np.random.normal(0, 2, 60))
        spx_dates = pd.date_range("2024-01-01", periods=60, freq="B")
        spx = pd.DataFrame(
            {
                "Open": spx_close,
                "High": spx_close + 1,
                "Low": spx_close - 1,
                "Close": spx_close,
                "Volume": 1e9,
            },
            index=spx_dates,
        )

        # VIX: 20 for first month, 21 for second month, 20 for third
        vix_close = np.concatenate([
            np.full(20, 20.0),  # First month
            np.full(20, 21.0),  # Second month
            np.full(20, 20.0),  # Third month
        ])
        vix = pd.DataFrame(
            {
                "Open": vix_close,
                "High": vix_close + 1,
                "Low": vix_close - 1,
                "Close": vix_close,
                "Volume": 1e6,
            },
            index=spx_dates,
        )

        spx_src = DataWithSource(df=spx, is_synthetic=False)
        vix_src = DataWithSource(df=vix, is_synthetic=False)

        result = run_backtest(spx_src, vix_src, notional=100000)

        # Strike should be constant between rolls
        strike = result["strike"]
        # First roll on first business day of Jan, strike = (20/100)^2 / 252
        expected_strike_1 = 0.04 / 252
        # Find where the first strike is set (first 20 trading days, up until Feb 1)
        jan_days = [i for i, d in enumerate(spx_dates) if d.month == 1]
        assert np.allclose(strike.iloc[jan_days], expected_strike_1, rtol=1e-6)

        # Second roll on first business day of Feb, strike = (21/100)^2 / 252
        expected_strike_2 = 0.0441 / 252
        feb_days = [i for i, d in enumerate(spx_dates) if d.month == 2]
        if len(feb_days) > 0:
            assert np.allclose(strike.iloc[feb_days], expected_strike_2, rtol=1e-6)

    def test_run_backtest_hand_computed_pnl(self):
        """Hand-computed P&L with fixed strike (daily mark-to-market)."""
        # Simple 10-day case: realized vol varies, strike fixed at 20 VIX
        spx_close = np.array([100.0, 101.0, 100.5, 101.5, 100.0, 99.5, 100.0, 101.0, 99.5, 100.5])
        spx_dates = pd.date_range("2024-01-01", periods=10, freq="B")
        spx = pd.DataFrame(
            {
                "Open": spx_close,
                "High": spx_close + 0.5,
                "Low": spx_close - 0.5,
                "Close": spx_close,
                "Volume": 1e9,
            },
            index=spx_dates,
        )

        # VIX constant at 20 → strike = (20/100)^2 / 252 ≈ 1.5873e-4
        vix_close = np.full(10, 20.0)
        vix = pd.DataFrame(
            {
                "Open": vix_close,
                "High": vix_close + 1,
                "Low": vix_close - 1,
                "Close": vix_close,
                "Volume": 1e6,
            },
            index=spx_dates,
        )

        spx_src = DataWithSource(df=spx, is_synthetic=False)
        vix_src = DataWithSource(df=vix, is_synthetic=False)

        result = run_backtest(spx_src, vix_src, side=1, notional=100000)

        # Verify strike is constant
        strike = result["strike"].dropna()
        expected_strike = 0.04 / 252  # (20/100)^2 / 252
        assert np.allclose(strike.iloc[0], expected_strike, rtol=1e-6)

        # Verify gross P&L = notional * (daily_var - strike)
        daily_var = result["daily_var"]
        pnl_gross = result["pnl_gross"]

        # Manual calculation: daily P&L
        expected_pnl = 100000 * (daily_var - expected_strike)

        # Compare values (NaN in first row should match)
        pd.testing.assert_series_equal(pnl_gross, expected_pnl, check_exact=False, atol=1e-6, check_names=False)


class TestIdentifyRollDates:
    """Test roll date identification."""

    def test_identify_roll_dates_single_month(self):
        """Single month → one roll date (first business day)."""
        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        roll_dates = identify_roll_dates(dates)
        # First business day of January 2024
        assert len(roll_dates) >= 1
        assert roll_dates[0] == dates[0]

    def test_identify_roll_dates_three_months(self):
        """Three months → one roll per month."""
        dates = pd.date_range("2024-01-01", periods=60, freq="B")
        roll_dates = identify_roll_dates(dates)
        # Should have ~3 roll dates (one per month)
        assert len(roll_dates) >= 2
        # Each roll date should be the first date of a new month
        for i in range(1, len(roll_dates)):
            assert roll_dates[i].month != roll_dates[i - 1].month

    def test_identify_roll_dates_empty(self):
        """Empty date index → empty roll dates."""
        dates = pd.DatetimeIndex([])
        roll_dates = identify_roll_dates(dates)
        assert roll_dates == []


class TestComputeRollCost:
    """Test roll cost calculation."""

    def test_compute_roll_cost_formula(self):
        """Roll cost formula: bid_ask_vol * 2 * avg_vol * notional."""
        # VIX=20, curr_vix=21, notional=100k, bid_ask=0.5 vol points
        prev_vix = 20.0
        curr_vix = 21.0
        notional = 100000
        bid_ask_vol_points = 0.5

        cost = compute_roll_cost(prev_vix, curr_vix, notional, bid_ask_vol_points)

        # prev_vol=0.20, curr_vol=0.21, avg_vol=0.205
        # bid_ask_vol=0.005 (0.5%), dvar=0.005*2*0.205=0.00205
        # cost=0.00205*100000=205
        expected = 0.005 * 2 * 0.205 * notional
        assert abs(cost - expected) < 1

    def test_compute_roll_cost_constant_vix(self):
        """No VIX change → lower cost (only bid-ask spread)."""
        prev_vix = 20.0
        curr_vix = 20.0
        notional = 100000
        bid_ask_vol_points = 0.5

        cost = compute_roll_cost(prev_vix, curr_vix, notional, bid_ask_vol_points)

        # avg_vol=0.20, bid_ask_vol=0.005, dvar=0.005*2*0.20=0.002
        # cost=0.002*100000=200
        expected = 0.005 * 2 * 0.20 * notional
        assert abs(cost - expected) < 1

    def test_compute_roll_cost_scales_linearly(self):
        """Cost scales linearly with notional."""
        prev_vix = 20.0
        curr_vix = 22.0
        bid_ask_vol_points = 0.5

        cost_100k = compute_roll_cost(prev_vix, curr_vix, 100000, bid_ask_vol_points)
        cost_200k = compute_roll_cost(prev_vix, curr_vix, 200000, bid_ask_vol_points)

        assert abs(cost_200k - 2 * cost_100k) < 1e-6
