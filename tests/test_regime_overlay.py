"""Tests for regime overlay HMM."""

import pytest
import pandas as pd
import numpy as np
from src.strategy import RegimeHMM


class TestRegimeHMM:
    """Test HMM regime detection with causality guarantees."""

    def _create_synthetic_market(self, n_days: int = 500):
        """Create synthetic SPX and VIX data with two regimes."""
        dates = pd.date_range("2020-01-01", periods=n_days, freq="B")

        # Create calm regime: low vol, stable prices
        calm_days = np.arange(0, n_days, 2)  # Every other week
        turbulent_days = np.arange(1, n_days, 2)

        # SPX: trend with regime-specific volatility
        spx_close = 5000 + np.cumsum(np.random.normal(0, 3, n_days))
        vix_close = np.zeros(n_days)

        for i in range(n_days):
            if i in calm_days:
                vix_close[i] = np.random.normal(15, 2)  # Low VIX, calm
            else:
                vix_close[i] = np.random.normal(25, 3)  # High VIX, turbulent

        vix_close = np.maximum(vix_close, 5)  # VIX floor

        spx = pd.DataFrame(
            {
                "Open": spx_close,
                "High": spx_close + 2,
                "Low": spx_close - 2,
                "Close": spx_close,
                "Volume": 1e9,
            },
            index=dates,
        )

        vix = pd.DataFrame(
            {
                "Open": vix_close,
                "High": vix_close + 1,
                "Low": np.maximum(vix_close - 1, 1),
                "Close": vix_close,
                "Volume": 1e6,
            },
            index=dates,
        )

        return spx, vix

    def test_hmm_initialization(self):
        """HMM initializes without errors."""
        hmm = RegimeHMM(n_components=2)
        assert hmm.n_components == 2
        assert hmm.refit_freq == "Y"

    def test_regime_prediction_shape(self):
        """Regime predictions have correct shape (may have unknown at start)."""
        spx, vix = self._create_synthetic_market(500)
        hmm = RegimeHMM(n_components=2, min_history_days=100)

        regimes = hmm.fit_and_predict(spx["Close"], vix["Close"])

        # Regimes has same index but may start with "unknown" during warmup
        assert len(regimes) <= len(spx)
        assert regimes.dtype == object  # String dtype

    def test_regime_values(self):
        """Regime values are valid ("calm", "turbulent", "unknown")."""
        spx, vix = self._create_synthetic_market(500)
        hmm = RegimeHMM(n_components=2, min_history_days=100)

        regimes = hmm.fit_and_predict(spx["Close"], vix["Close"])

        valid_regimes = {"calm", "turbulent", "unknown"}
        assert set(regimes.unique()).issubset(valid_regimes)

    def test_no_look_ahead_basic(self):
        """Regime at time t should not depend on future data (walk-forward check)."""
        spx, vix = self._create_synthetic_market(1000)

        # Fit 1: Full dataset
        hmm1 = RegimeHMM(n_components=2, min_history_days=100, refit_freq="Y")
        regimes1 = hmm1.fit_and_predict(spx["Close"], vix["Close"])

        # Fit 2: Only use first 600 days (does not include last 400 days)
        spx_short = spx.iloc[:600]
        vix_short = vix.iloc[:600]
        hmm2 = RegimeHMM(n_components=2, min_history_days=100, refit_freq="Y")
        regimes2 = hmm2.fit_and_predict(spx_short["Close"], vix_short["Close"])

        # Regimes in overlapping period should match (both use same data for that period)
        # Select indices that exist in both
        common_indices = regimes1.index.intersection(regimes2.index)
        overlap_start_idx = 200  # Skip warmup period
        common_indices = common_indices[overlap_start_idx:]

        if len(common_indices) > 0:
            overlap_regimes1 = regimes1.loc[common_indices]
            overlap_regimes2 = regimes2.loc[common_indices]

            # Filter out "unknown" regimes from warmup
            valid_mask = (overlap_regimes1 != "unknown") & (overlap_regimes2 != "unknown")
            if valid_mask.sum() > 0:
                match_rate = (overlap_regimes1[valid_mask] == overlap_regimes2[valid_mask]).sum() / valid_mask.sum()
                assert match_rate > 0.6, f"Look-ahead detected: match rate only {match_rate:.2f}"

    def test_regime_labels_by_vix(self):
        """High-VIX state is labeled 'turbulent', low-VIX state is 'calm'."""
        spx, vix = self._create_synthetic_market(500)
        hmm = RegimeHMM(n_components=2, min_history_days=100)

        regimes = hmm.fit_and_predict(spx["Close"], vix["Close"])

        # Check that "turbulent" regime has higher average VIX than "calm"
        calm_dates = regimes[regimes == "calm"].index
        turbulent_dates = regimes[regimes == "turbulent"].index

        if len(calm_dates) > 0 and len(turbulent_dates) > 0:
            calm_avg_vix = vix.loc[calm_dates, "Close"].mean()
            turbulent_avg_vix = vix.loc[turbulent_dates, "Close"].mean()

            assert (
                turbulent_avg_vix > calm_avg_vix
            ), "Turbulent regime should have higher VIX"

    def test_exposure_mapping(self):
        """Regime → exposure mapping is correct."""
        hmm = RegimeHMM()

        assert hmm.get_exposure("calm") == -1.0  # Short
        assert hmm.get_exposure("turbulent") == +1.0  # Long
        assert hmm.get_exposure("unknown") == 0.0  # Flat
        assert hmm.get_exposure("unknown", neutral_exposure=0.5) == 0.5  # Custom neutral

    def test_regime_at_date(self):
        """Can retrieve regime at specific date."""
        spx, vix = self._create_synthetic_market(500)
        hmm = RegimeHMM(n_components=2, min_history_days=100)

        regimes = hmm.fit_and_predict(spx["Close"], vix["Close"])

        # Get regime for a specific date
        test_date = regimes.index[250]
        regime = hmm.regime_at_date(regimes, test_date)

        assert regime in {"calm", "turbulent", "unknown"}

    def test_refit_frequency(self):
        """HMM refits according to specified frequency."""
        spx, vix = self._create_synthetic_market(1500)  # 6 years

        hmm_annual = RegimeHMM(n_components=2, refit_freq="Y", min_history_days=100)
        refit_dates = hmm_annual._get_refit_dates(spx.index)

        # Should have ~5-6 refits (annual) over ~6 years
        assert 4 <= len(refit_dates) <= 7

        hmm_quarterly = RegimeHMM(n_components=2, refit_freq="Q", min_history_days=100)
        refit_dates_q = hmm_quarterly._get_refit_dates(spx.index)

        # Should have ~20-25 refits (quarterly) over ~6 years
        assert len(refit_dates_q) > len(refit_dates)
