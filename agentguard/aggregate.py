"""Fuse predictor sub-scores into one calibrated 0-1 risk.

Two modes, mirroring how ProactiveGuard shipped:

* **Heuristic** (default, zero deps): weighted noisy-OR over sub-scores.
  Ships day one; every number is explainable in one sentence.
* **Learned**: a trained :class:`agentguard.predictors.model.RiskModel`
  (the ported ProactiveGuard ensemble) replaces the noisy-OR.

Either way the raw score can be passed through a conformal-style
calibrator so the emitted number behaves like a probability. A risk score
is only as trustworthy as its calibration.
"""

from __future__ import annotations

import bisect
from typing import Dict, Mapping, Optional, Sequence

DEFAULT_WEIGHTS: Dict[str, float] = {
    "loop": 1.0,           # most common failure, near-deterministic signal
    "tool_cascade": 0.9,
    "budget_drift": 0.7,
    "semantic_drift": 0.6,  # noisier signal, weight it down
    "model": 1.0,           # the learned risk passes through unattenuated
}


class Aggregator:
    """Weighted noisy-OR fusion of sub-scores.

    risk = 1 - prod(1 - w_i * s_i)

    Properties that make this the right heuristic: one strong signal alone
    can drive risk high; several weak signals compound instead of being
    averaged away; and each predictor's contribution stays inspectable.
    """

    def __init__(
        self,
        weights: Optional[Mapping[str, float]] = None,
        calibrator: Optional["ConformalCalibrator"] = None,
    ):
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        self.calibrator = calibrator

    def fuse(self, subscores: Mapping[str, float]) -> float:
        survival = 1.0
        for name, score in subscores.items():
            w = self.weights.get(name, 0.8)
            survival *= 1.0 - max(0.0, min(1.0, w * score))
        raw = 1.0 - survival
        if self.calibrator is not None and self.calibrator.is_fitted:
            return self.calibrator.calibrate(raw)
        return raw


class ConformalCalibrator:
    """Split-conformal calibration of a raw risk score.

    Fit on raw scores from runs with known outcomes. The calibrated value
    for a new score is one minus the conformal p-value against the healthy
    population — i.e. the empirical fraction of *healthy* runs whose peak
    raw score stayed below it. A calibrated 0.95 therefore has an
    operational meaning: only ~5% of known-good runs ever looked this bad.

    Same conformal discipline as the trading-system ensemble this is
    lifted from: never emit a raw model score as if it were a probability.
    """

    def __init__(self) -> None:
        self._healthy_scores: list = []

    @property
    def is_fitted(self) -> bool:
        return len(self._healthy_scores) > 0

    def fit(self, scores: Sequence[float], failed: Sequence[bool]) -> "ConformalCalibrator":
        """``scores`` are per-run raw risk peaks, ``failed`` their outcomes."""
        self._healthy_scores = sorted(
            s for s, f in zip(scores, failed) if not f
        )
        return self

    def calibrate(self, raw: float) -> float:
        if not self._healthy_scores:
            return raw
        n = len(self._healthy_scores)
        below = bisect.bisect_right(self._healthy_scores, raw)
        # +1 smoothing keeps 0 and 1 unreachable from finite evidence
        return below / (n + 1)
