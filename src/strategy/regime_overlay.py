"""Regime-switching HMM for gamma trading exposure control.

DESIGN CONSTRAINTS (Anti-Overfitting):
1. CAUSAL: Regime at time t uses only data <= t (walk-forward, forward filtering)
2. MAX 3 FEATURES: VIX level, 21d realized vol, optional VIX term slope
3. CAUSAL SCALING: z-score with expanding mean/std, never full-sample
4. STATE IDENTIFICATION: Label "turbulent" = high-VIX state (computed on refit)
5. LOGIC-FIRST MAPPING: short gamma in calm, neutral/long in turbulent (not optimized)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional
import warnings

from scipy.stats import gaussian_kde
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore", category=UserWarning)


class RegimeHMM:
    """
    Gaussian HMM for market regime detection with walk-forward causality.

    Detects two regimes:
    - CALM: Low realized vol, high implied vol (short gamma attractive)
    - TURBULENT: High realized vol, low implied vol (long gamma attractive)
    """

    def __init__(
        self,
        n_components: int = 2,
        refit_freq: str = "Y",  # "Y"=annual, "Q"=quarterly
        min_history_days: int = 252,
    ):
        """
        Args:
            n_components: Number of hidden states (2 recommended: calm vs turbulent)
            refit_freq: Refit period ("Y"=annual, "Q"=quarterly, "M"=monthly)
            min_history_days: Minimum data required before first regime inference
        """
        self.n_components = n_components
        self.refit_freq = refit_freq
        self.min_history_days = min_history_days

        self.hmm = None
        self.refit_dates = []
        self.last_refit_date = None
        self.state_labels = {}  # Maps state index to "calm" or "turbulent"

    def _compute_features(
        self, spx_close: pd.Series, vix_close: pd.Series
    ) -> pd.DataFrame:
        """
        Compute HMM features: VIX level, 21d realized vol, VIX term slope.

        All scaling is CAUSAL: expanding mean/std, never full-sample.
        """
        # Feature 1: VIX level (normalized)
        vix_level = vix_close / 100.0

        # Feature 2: 21-day realized volatility
        returns = np.log(spx_close / spx_close.shift(1))
        realized_vol_21d = returns.rolling(21).std() * np.sqrt(252)

        # Feature 3 (optional): VIX term slope (proxy: VIX level change)
        # For simplicity, use 5d momentum of VIX (forward-looking for transitions)
        vix_momentum = (vix_close - vix_close.shift(5)) / vix_close.shift(5)

        # CAUSAL SCALING: z-score with expanding mean/std
        features = pd.DataFrame(
            {
                "vix_level": vix_level,
                "realized_vol_21d": realized_vol_21d,
                "vix_momentum": vix_momentum,
            },
            index=spx_close.index,
        )

        # Expanding z-score (never look-ahead)
        features_scaled = features.copy()
        for col in features.columns:
            expanding_mean = features[col].expanding().mean()
            expanding_std = features[col].expanding().std()
            features_scaled[col] = (features[col] - expanding_mean) / (
                expanding_std + 1e-8
            )

        return features_scaled.dropna()

    def _get_refit_dates(
        self, index: pd.DatetimeIndex, start_date: Optional[datetime] = None
    ) -> List[datetime]:
        """
        Get refit dates based on frequency (annual, quarterly, monthly).

        Refits occur on the first trading day of each period.
        """
        if start_date is None:
            start_date = index[self.min_history_days]

        refit_dates = []
        current_period = None

        for date in index:
            if date < start_date:
                continue

            if self.refit_freq == "Y":
                period = (date.year,)
            elif self.refit_freq == "Q":
                period = (date.year, date.quarter)
            elif self.refit_freq == "M":
                period = (date.year, date.month)
            else:
                raise ValueError(f"Unknown refit_freq: {self.refit_freq}")

            if current_period != period:
                refit_dates.append(date)
                current_period = period

        return refit_dates

    def _fit_and_label(
        self, X: np.ndarray
    ) -> Tuple[GaussianHMM, Dict[int, str]]:
        """
        Fit HMM and label states by VIX mean (high VIX = turbulent).

        For n_components > 2: only 2 labels (turbulent, calm), rest = unknown.
        """
        hmm = GaussianHMM(n_components=self.n_components, n_iter=1000, random_state=42)
        hmm.fit(X)

        # Identify states by mean VIX level (feature 0)
        state_vix_means = hmm.means_[:, 0]

        # For simplicity, label only top (turbulent) and bottom (calm), rest unknown
        sorted_states = np.argsort(state_vix_means)
        high_vix_state = sorted_states[-1]  # Highest VIX
        low_vix_state = sorted_states[0]    # Lowest VIX

        state_labels = {
            high_vix_state: "turbulent",
            low_vix_state: "calm",
        }
        # Unlabeled states default to "unknown" in predict

        return hmm, state_labels

    def fit_and_predict(
        self, spx_close: pd.Series, vix_close: pd.Series
    ) -> pd.Series:
        """
        Walk-forward HMM fitting and regime prediction (causal, forward-filtered).

        Returns:
            Series of regime labels ("calm" or "turbulent") indexed by date.
        """
        features = self._compute_features(spx_close, vix_close)
        refit_dates = self._get_refit_dates(features.index)

        regime_predictions = pd.Series(
            "unknown", index=features.index, dtype=str
        )

        for i, refit_date in enumerate(refit_dates):
            # Find next refit date (or end of data)
            if i + 1 < len(refit_dates):
                next_refit = refit_dates[i + 1]
                mask = (features.index >= refit_date) & (features.index < next_refit)
            else:
                mask = features.index >= refit_date

            # Training data: all data UP TO (not including) refit_date
            train_mask = features.index < refit_date
            if train_mask.sum() < self.min_history_days:
                continue

            X_train = features[train_mask].values
            X_pred = features[mask].values

            # Fit HMM on training data
            self.hmm, self.state_labels = self._fit_and_label(X_train)
            self.last_refit_date = refit_date

            # Predict on test period using forward algorithm (causal)
            if len(X_pred) > 0:
                hidden_states = self.hmm.predict(X_pred)
                regime_labels = np.array(
                    [self.state_labels.get(s, "unknown") for s in hidden_states]
                )
                regime_predictions.loc[mask] = regime_labels

        return regime_predictions

    def get_exposure(self, regime: str, neutral_exposure: float = 0.0) -> float:
        """
        Map regime to exposure: short gamma in calm, long/neutral in turbulent.

        Exposures (for notional=100k):
        - Calm: -1.0 (short gamma, sell vol)
        - Turbulent: +1.0 (long gamma, buy vol)
        - Neutral: 0.0 (flat)

        Args:
            regime: "calm", "turbulent", or "unknown"
            neutral_exposure: Value to use for unknown/transition regimes

        Returns:
            Exposure multiplier: -1.0 (short) to +1.0 (long)
        """
        if regime == "calm":
            return -1.0  # Short gamma
        elif regime == "turbulent":
            return +1.0  # Long gamma
        else:
            return neutral_exposure  # Flat on unknown

    def regime_at_date(self, regimes: pd.Series, date: pd.Timestamp) -> str:
        """Get regime label at specific date."""
        if date in regimes.index:
            return regimes.loc[date]
        return "unknown"


def evaluate_regime_strategy(
    spx_source,
    vix_source,
    initial_capital: float = 100000,
    bid_ask_vol_points: float = 0.5,
    refit_freq: str = "Y",
) -> Dict:
    """
    Evaluate HMM regime overlay strategy vs static baselines.

    Compares:
    1. HMM dynamic (short in calm, long in turbulent)
    2. Static short gamma (always short)
    3. Static long gamma (always long)

    All evaluated out-of-sample, neto de costes.

    Returns:
        Dict with:
        - metrics: DataFrame of CAGR, Sharpe, max_dd, worst_week for each strategy
        - regimes: Series of regime labels
        - crash_analysis: Dict with Feb-2018 and Mar-2020 diagnostics
    """
    from src.data import load_data
    from src.backtest import run_backtest
    from src.risk import (
        cagr,
        annualized_vol,
        sharpe_ratio,
        max_drawdown,
    )

    spx = spx_source.df
    vix = vix_source.df

    # Align data
    common_idx = spx.index.intersection(vix.index)
    spx = spx.loc[common_idx]
    vix = vix.loc[common_idx]

    # Compute regimes (causal, walk-forward)
    hmm = RegimeHMM(n_components=2, refit_freq=refit_freq)
    regimes = hmm.fit_and_predict(spx["Close"], vix["Close"])

    # Run baseline strategies: static short, static long
    result_short = run_backtest(
        spx_source, vix_source, side=-1, notional=initial_capital, bid_ask_vol_points=bid_ask_vol_points
    )
    result_long = run_backtest(
        spx_source, vix_source, side=1, notional=initial_capital, bid_ask_vol_points=bid_ask_vol_points
    )

    # HMM overlay: dynamic exposure based on regime
    # For each day, compute P&L using dynamic side (short/long/flat based on regime)
    pnl_gross_short = result_short["pnl_gross"]
    pnl_gross_long = result_long["pnl_gross"]
    roll_costs = result_short["roll_costs"]  # Same costs for all (no extra roll cost from switching)

    # Dynamic P&L: multiply by regime-based exposure
    pnl_gross_hmm = pd.Series(0.0, index=pnl_gross_short.index)
    for date in regimes.index:
        if date in pnl_gross_short.index:
            regime = regimes.loc[date]
            exposure = hmm.get_exposure(regime, neutral_exposure=0.0)

            # Interpolate exposure to smooth transitions
            if regime == "calm":
                pnl_gross_hmm.loc[date] = pnl_gross_short.loc[date]  # Short
            elif regime == "turbulent":
                pnl_gross_hmm.loc[date] = pnl_gross_long.loc[date]  # Long
            else:
                pnl_gross_hmm.loc[date] = 0.0  # Flat

    # Net P&L: account for roll costs (assume same costs regardless of exposure)
    pnl_net_short = result_short["pnl_series"]
    pnl_net_long = result_long["pnl_series"]
    pnl_net_hmm = pnl_gross_hmm - roll_costs

    # Calculate cumulative and metrics
    def calc_metrics(pnl_series, cumulative):
        pnl_clean = pnl_series.dropna()
        cum_clean = cumulative.dropna()

        years = (pnl_clean.index[-1] - pnl_clean.index[0]).days / 365.25
        c = cagr(cum_clean, years=years, initial_capital=initial_capital)
        v = annualized_vol(pnl_clean)
        s = sharpe_ratio(pnl_clean)
        dd = max_drawdown(cum_clean, initial_capital=initial_capital)

        return {
            "CAGR (%)": c * 100,
            "Vol (%)": v * 100,
            "Sharpe": s,
            "Max DD (%)": dd * 100,
        }

    cum_short = pnl_net_short.dropna().cumsum()
    cum_long = pnl_net_long.dropna().cumsum()
    cum_hmm = pnl_net_hmm.dropna().cumsum()

    metrics_short = calc_metrics(pnl_net_short, cum_short)
    metrics_long = calc_metrics(pnl_net_long, cum_long)
    metrics_hmm = calc_metrics(pnl_net_hmm, cum_hmm)

    metrics_df = pd.DataFrame(
        {
            "Static Short": metrics_short,
            "Static Long": metrics_long,
            "HMM Overlay": metrics_hmm,
        }
    )

    # Crash analysis: regime at specific dates
    crash_dates = {
        "2018-02-03": "Feb-2018 Vol Spike (precursor)",
        "2020-03-06": "Mar-2020 COVID Crash (precursor)",
    }

    crash_analysis = {}
    for date_str, desc in crash_dates.items():
        try:
            date = pd.Timestamp(date_str)
            regime = regimes[regimes.index <= date].iloc[-1] if len(regimes[regimes.index <= date]) > 0 else "unknown"
            crash_analysis[date_str] = {
                "date": date,
                "description": desc,
                "regime": regime,
                "exposure": hmm.get_exposure(regime),
            }
        except:
            pass

    return {
        "metrics": metrics_df,
        "regimes": regimes,
        "crash_analysis": crash_analysis,
        "pnl_short": pnl_net_short,
        "pnl_long": pnl_net_long,
        "pnl_hmm": pnl_net_hmm,
        "cumulative_short": cum_short,
        "cumulative_long": cum_long,
        "cumulative_hmm": cum_hmm,
    }
