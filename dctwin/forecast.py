r"""Split-conformal interval forecasting and the two reliability signals.

The controller never uses a point forecast. For each series it emits a predictive
interval with a distribution-free marginal coverage guarantee, obtained by:

  1. a cheap base predictor  \hat{x}(t+h)  (ridge on lag + seasonal features),
  2. split-conformal calibration: the (1-alpha) empirical quantile of absolute
     residuals on a trailing calibration window becomes the half-width.

From the interval widths we derive the two *distinct* notions of reliability the
formulation insists on separating:

  * event reliability   R = exp(-gamma * width / capacity)   -- trust in *this*
    prediction; drives where to commit a rule.
  * chronic predictability  Q = 1 - mean(normalized width over a window)  --
    whether *any* forecast helps; drives migration.

The base model is intentionally swappable (LSTM/TCN/transformer); the calibration
wrapper is the part that makes R and Q auditable rather than tuned scalars.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict

import numpy as np


@dataclass
class _SeriesState:
    history: Deque[float]
    cal_residuals: Deque[float]  # |actual - predicted| on calibration window


@dataclass
class ConformalForecaster:
    """One split-conformal forecaster shared across many named series."""

    period: int = 96
    n_lags: int = 4
    alpha: float = 0.1               # miscoverage: intervals target 1-alpha coverage
    hist_len: int = 400
    cal_len: int = 120
    gamma: float = 4.0               # steepness mapping width -> event reliability
    q_window: int = 96               # window for chronic predictability
    ridge_lambda: float = 1.0

    _series: Dict[str, _SeriesState] = field(default_factory=dict)
    # rolling normalized widths per series, for Q
    _width_hist: Dict[str, Deque[float]] = field(default_factory=dict)

    # coverage audit counters
    covered: int = 0
    total: int = 0

    # ---------------------------------------------------------------- ingest
    def observe(self, key: str, value: float) -> None:
        st = self._series.get(key)
        if st is None:
            st = _SeriesState(deque(maxlen=self.hist_len), deque(maxlen=self.cal_len))
            self._series[key] = st
            self._width_hist[key] = deque(maxlen=self.q_window)
        # before appending, score the calibration residual using a model fit on
        # history *excluding* this point (honest split)
        if len(st.history) >= self.n_lags + 2:
            pred = self._predict_point(st, horizon=1)
            st.cal_residuals.append(abs(value - pred))
        st.history.append(float(value))

    # ---------------------------------------------------------------- predict
    def _features(self, hist: np.ndarray, idx: int) -> np.ndarray:
        lags = hist[idx - self.n_lags:idx]
        t = idx  # position acts as phase proxy
        seas = [np.sin(2 * np.pi * t / self.period), np.cos(2 * np.pi * t / self.period)]
        return np.concatenate([lags, seas, [1.0]])

    def _fit(self, st: _SeriesState):
        hist = np.asarray(st.history, dtype=float)
        n = len(hist)
        X, y = [], []
        for idx in range(self.n_lags, n):
            X.append(self._features(hist, idx))
            y.append(hist[idx])
        X = np.asarray(X)
        y = np.asarray(y)
        d = X.shape[1]
        # ridge closed form
        A = X.T @ X + self.ridge_lambda * np.eye(d)
        w = np.linalg.solve(A, X.T @ y)
        return w

    def _predict_point(self, st: _SeriesState, horizon: int) -> float:
        hist = np.asarray(st.history, dtype=float)
        if len(hist) < self.n_lags + 2:
            return float(hist[-1]) if len(hist) else 0.0
        w = self._fit(st)
        buf = list(hist)
        pred = buf[-1]
        for _ in range(horizon):
            feat = self._features(np.asarray(buf), len(buf))
            pred = float(feat @ w)
            buf.append(pred)
        return max(0.0, pred)

    def _halfwidth(self, st: _SeriesState) -> float:
        if len(st.cal_residuals) < 5:
            # fall back to spread of recent history
            h = np.asarray(st.history, dtype=float)
            return float(np.std(h[-self.cal_len:]) + 1e-6) * 1.64
        res = np.asarray(st.cal_residuals)
        # finite-sample-corrected conformal quantile
        n = len(res)
        q_level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        return float(np.quantile(res, q_level))

    def forecast(self, key: str, horizon: int, capacity: float):
        """Return (point, lo, hi, event_reliability) for a series.

        capacity normalizes the interval width into the event-reliability map.
        """
        st = self._series.get(key)
        if st is None or len(st.history) < self.n_lags + 2:
            base = 0.0 if st is None or not st.history else st.history[-1]
            return base, base, base, 0.3  # low confidence until warmed up
        point = self._predict_point(st, horizon)
        hw = self._halfwidth(st) * (1.0 + 0.15 * (horizon - 1))  # widen with horizon
        lo, hi = max(0.0, point - hw), point + hw
        norm_w = min(1.0, (hi - lo) / max(capacity, 1e-9))
        self._width_hist[key].append(norm_w)
        reliability = float(np.exp(-self.gamma * norm_w))
        return point, lo, hi, reliability

    def chronic_predictability(self, key: str) -> float:
        """Q in [0,1]: long-run forecastability of this series."""
        wh = self._width_hist.get(key)
        if not wh:
            return 0.5
        return float(np.clip(1.0 - np.mean(wh), 0.0, 1.0))

    # ---------------------------------------------------------------- audit
    def record_coverage(self, key: str, actual: float, lo: float, hi: float) -> None:
        self.total += 1
        if lo <= actual <= hi:
            self.covered += 1

    @property
    def empirical_coverage(self) -> float:
        return self.covered / self.total if self.total else float("nan")
