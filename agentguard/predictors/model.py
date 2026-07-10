"""The learned risk model — ProactiveGuard's architecture, re-targeted.

This is a direct port of the model from *ProactiveGuard: Deep
Learning-Based Predictive Failure Detection for Distributed Consensus
Systems* (Tripathi), moved from consensus-node metrics to agent-run
metrics. Ported pieces, same math:

* **learnable feature attention** — a sigmoid gate over input features,
  learned end-to-end, so the model itself reports which run metrics are
  discriminative (in the paper: WAL fsync latency, heartbeat latency;
  here: tool-error rate, loop score, token velocity)
* **residual connections** between equal-width hidden layers
* **focal loss** ``-(1-p_t)^2 * log(p_t)`` — failure precursors are the
  rare class, plain cross-entropy just learns to say "healthy"
* **mixup** augmentation + inverted dropout + Adam, from the same
  training loop
* **ensemble**: five bagged nets with different widths + a Random Forest
  counted at double weight, probabilities averaged

Deps: numpy (``pip install agentguard[model]``); scikit-learn optional —
without it the ensemble runs NN-only.

Classes: 0 = healthy, 1 = degraded (failure precursor), 2 = failing.
Risk emitted = 1 - P(healthy).
"""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ..telemetry import Step
from .base import Predictor
from .budget_drift import BudgetDriftPredictor
from .loop import LoopPredictor
from .tool_cascade import ToolCascadePredictor

NUM_CLASSES = 3
CLASS_NAMES = {0: "healthy", 1: "degraded", 2: "failing"}

# -- feature extraction ------------------------------------------------------
#
# The analogue of ProactiveGuard's 32-feature observation window. Inputs
# are the deterministic sub-signal trajectories plus raw run statistics
# over the recent window.

SIGNAL_NAMES = ("loop", "tool_cascade", "budget_drift")
WINDOW = 8
NUM_FEATURES = len(SIGNAL_NAMES) * 4 + 6  # per-signal stats + run stats


def extract_features(
    signal_trajectories: Sequence[Sequence[float]],
    error_flags: Sequence[bool],
    token_counts: Sequence[float],
    latencies: Sequence[float],
    n_steps: int,
) -> np.ndarray:
    """One (NUM_FEATURES,) vector from the tail of a run.

    ``signal_trajectories``: per-signal list of per-step sub-scores, in
    SIGNAL_NAMES order. Remaining args are per-step raw run metrics.
    """
    feats: List[float] = []
    for traj in signal_trajectories:
        recent = list(traj[-WINDOW:]) or [0.0]
        feats.append(recent[-1])                      # current
        feats.append(float(np.mean(recent)))          # window mean
        feats.append(float(np.max(recent)))           # window max
        slope = (recent[-1] - recent[0]) / max(1, len(recent) - 1)
        feats.append(slope)                           # window slope

    err = list(error_flags[-WINDOW:])
    feats.append(sum(err) / len(err) if err else 0.0)        # error rate

    tok = [t for t in token_counts[-WINDOW:] if t]
    early = [t for t in token_counts[: max(3, WINDOW // 2)] if t]
    if tok and early and np.mean(early) > 0:
        feats.append(min(5.0, float(np.mean(tok)) / float(np.mean(early))) / 5.0)
    else:
        feats.append(0.0)                                    # token velocity ratio

    lat = [l for l in latencies[-WINDOW:] if l]
    feats.append(min(1.0, float(np.mean(lat)) / 30.0) if lat else 0.0)
    feats.append(min(1.0, float(np.max(lat)) / 60.0) if lat else 0.0)

    feats.append(min(1.0, n_steps / 50.0))                   # run length
    feats.append(min(1.0, float(sum(err)) / 5.0) if err else 0.0)  # error count

    return np.asarray(feats, dtype=np.float64)


# -- the ported network ------------------------------------------------------


class ProactiveGuardNet:
    """Numpy MLP with learnable feature attention, residual connections and
    focal loss. Line-for-line port of ProactiveGuard's ImprovedProactiveGuard,
    minus nothing."""

    def __init__(self, n_classes: int = NUM_CLASSES, hidden_dims: Optional[List[int]] = None,
                 seed: int = 42):
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.seed = seed
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self.attention_weights: Optional[np.ndarray] = None
        self.training = False

    def _init_weights(self, input_dim: int) -> None:
        rng = np.random.RandomState(self.seed)
        dims = [input_dim] + self.hidden_dims + [self.n_classes]
        self.weights, self.biases = [], []
        for i in range(len(dims) - 1):
            limit = np.sqrt(6 / (dims[i] + dims[i + 1]))
            self.weights.append(rng.uniform(-limit, limit, (dims[i], dims[i + 1])))
            self.biases.append(np.zeros(dims[i + 1]))
        self.attention_weights = rng.randn(input_dim) * 0.1

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _softmax(x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def feature_attention(self) -> np.ndarray:
        """Learned per-feature gate — the model's own feature-importance."""
        return self._sigmoid(self.attention_weights)

    def _forward(self, X, rng=None):
        h = X * self.feature_attention()
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            h_new = self._relu(h @ W + b)
            if h.shape[1] == h_new.shape[1]:
                h_new = h_new + h * 0.1  # residual connection
            h = h_new
            if self.training and rng is not None:
                mask = rng.binomial(1, 0.7, h.shape) / 0.7  # inverted dropout
                h = h * mask
        return self._softmax(h @ self.weights[-1] + self.biases[-1])

    def fit(self, X, y, epochs=150, lr=0.01, batch_size=64):
        self._init_weights(X.shape[1])
        self.training = True
        rng = np.random.RandomState(self.seed)

        n = X.shape[0]
        best_loss, patience = float("inf"), 0
        m_w = [np.zeros_like(w) for w in self.weights]
        v_w = [np.zeros_like(w) for w in self.weights]
        m_b = [np.zeros_like(b) for b in self.biases]
        v_b = [np.zeros_like(b) for b in self.biases]
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for epoch in range(epochs):
            idx = rng.permutation(n)
            X_shuf, y_shuf = X[idx], y[idx]
            total_loss, n_batches = 0.0, 0

            for i in range(0, n, batch_size):
                Xb, yb = X_shuf[i : i + batch_size], y_shuf[i : i + batch_size]

                if rng.random() < 0.3:  # mixup augmentation
                    lam = rng.beta(0.4, 0.4)
                    perm = rng.permutation(len(Xb))
                    Xb = lam * Xb + (1 - lam) * Xb[perm]

                y_pred = self._forward(Xb, rng)
                y_oh = np.zeros((len(yb), self.n_classes))
                y_oh[np.arange(len(yb)), yb] = 1

                # focal loss: down-weight easy (well-classified) samples so
                # the rare failure-precursor class drives the gradient
                pt = np.sum(y_oh * y_pred, axis=1)
                total_loss += -np.mean((1 - pt) ** 2 * np.log(pt + 1e-7))
                n_batches += 1

                d_out = (y_pred - y_oh) / len(yb)

                activations = [Xb * self.feature_attention()]
                h = activations[0]
                for W, b in zip(self.weights[:-1], self.biases[:-1]):
                    h = self._relu(h @ W + b)
                    activations.append(h)

                delta = d_out
                for j in range(len(self.weights) - 1, -1, -1):
                    dW = activations[j].T @ delta + 0.001 * self.weights[j]
                    db = np.sum(delta, axis=0)

                    t = epoch * (n // batch_size + 1) + n_batches
                    m_w[j] = beta1 * m_w[j] + (1 - beta1) * dW
                    v_w[j] = beta2 * v_w[j] + (1 - beta2) * dW**2
                    self.weights[j] -= lr * (m_w[j] / (1 - beta1 ** (t + 1))) / (
                        np.sqrt(v_w[j] / (1 - beta2 ** (t + 1))) + eps
                    )
                    m_b[j] = beta1 * m_b[j] + (1 - beta1) * db
                    v_b[j] = beta2 * v_b[j] + (1 - beta2) * db**2
                    self.biases[j] -= lr * (m_b[j] / (1 - beta1 ** (t + 1))) / (
                        np.sqrt(v_b[j] / (1 - beta2 ** (t + 1))) + eps
                    )
                    if j > 0:
                        delta = delta @ self.weights[j].T * (activations[j] > 0)

            avg = total_loss / n_batches
            if avg < best_loss:
                best_loss, patience = avg, 0
            else:
                patience += 1
                if patience >= 20:
                    break  # early stopping

        self.training = False
        return self

    def predict_proba(self, X):
        self.training = False
        return self._forward(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class RiskEnsemble:
    """ProactiveGuard's ensemble: 5 bagged nets (different widths) + a
    Random Forest at double weight. Runs NN-only if sklearn is absent."""

    CONFIGS = [
        [256, 128, 64],
        [128, 64, 32],
        [512, 256, 128],
        [256, 256, 128, 64],
        [128, 128, 64],
    ]

    def __init__(self, n_classes: int = NUM_CLASSES, n_models: int = 5, seed: int = 42):
        self.n_classes = n_classes
        self.n_models = n_models
        self.seed = seed
        self.models: List[ProactiveGuardNet] = []
        self.rf = None

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        self.models = []
        for i in range(self.n_models):
            net = ProactiveGuardNet(self.n_classes, self.CONFIGS[i % len(self.CONFIGS)],
                                    seed=self.seed + i)
            idx = rng.choice(len(X), len(X), replace=True)  # bootstrap bag
            net.fit(X[idx], y[idx], epochs=100)
            self.models.append(net)

        try:
            from sklearn.ensemble import RandomForestClassifier

            self.rf = RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=self.seed, n_jobs=-1
            ).fit(X, y)
        except ImportError:
            self.rf = None
        return self

    def predict_proba(self, X):
        probs = np.zeros((len(X), self.n_classes))
        for net in self.models:
            probs += net.predict_proba(X)
        denom = len(self.models)
        if self.rf is not None:
            probs += self.rf.predict_proba(X) * 2
            denom += 2
        return probs / denom

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def risk(self, X) -> np.ndarray:
        """Scalar risk per row: 1 - P(healthy), i.e. the probability the
        run is off its healthy path (degraded or failing)."""
        p = self.predict_proba(X)
        return 1.0 - p[:, 0]

    def feature_attention(self) -> np.ndarray:
        """Mean learned attention across the nets — which features the
        ensemble found discriminative."""
        return np.mean([net.feature_attention() for net in self.models], axis=0)

    def save(self, path: str) -> None:
        with gzip.open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "RiskEnsemble":
        with gzip.open(path, "rb") as f:
            return pickle.load(f)


# -- live predictor wrapping a trained ensemble --------------------------------


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "risk_model.pkl"


class LearnedRiskPredictor(Predictor):
    """Runs the trained ProactiveGuard ensemble on the live step stream.

    Internally maintains its own deterministic sub-signal predictors (the
    model's inputs), extracts the windowed feature vector each step, and
    emits the ensemble's risk.
    """

    name = "model"

    def __init__(self, model: Optional[RiskEnsemble] = None,
                 model_path: Optional[str] = None):
        if model is None:
            path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
            if not path.exists():
                raise FileNotFoundError(
                    f"No trained risk model at {path}. Train one with "
                    f"`python -m agentguard.train` or pass model=/model_path=."
                )
            model = RiskEnsemble.load(str(path))
        self.model = model
        self._signals = [LoopPredictor(), ToolCascadePredictor(), BudgetDriftPredictor()]
        self.reset()

    def reset(self) -> None:
        for p in self._signals:
            p.reset()
        self._trajectories: List[List[float]] = [[] for _ in self._signals]
        self._errors: List[bool] = []
        self._tokens: List[float] = []
        self._latencies: List[float] = []

    def update(self, step: Step, history: List[Step]) -> float:
        for traj, p in zip(self._trajectories, self._signals):
            traj.append(p.update(step, history))
        self._errors.append(bool(step.error))
        self._tokens.append(float(step.tokens or 0))
        self._latencies.append(float(step.latency_s or 0))

        x = extract_features(
            self._trajectories, self._errors, self._tokens, self._latencies,
            n_steps=len(history),
        )
        return self._clip(float(self.model.risk(x[None, :])[0]))
