"""Train the learned risk model on labeled synthetic agent runs.

    python -m agentguard.train

Bootstraps exactly the way ProactiveGuard did before real cluster data
existed: generate scenario runs (healthy / loop / cascade / budget-blowout)
with gradual failure onset, label each step healthy -> degraded -> failing
by its distance from the failure, extract windowed features, train the
ensemble, report per-class precision/recall, save the model.

Replace ``generate_runs`` with your own traced runs when you have them —
the rest of the pipeline is unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np

from .predictors.budget_drift import BudgetDriftPredictor
from .predictors.loop import LoopPredictor
from .predictors.model import (
    DEFAULT_MODEL_PATH,
    NUM_FEATURES,
    RiskEnsemble,
    extract_features,
)
from .predictors.tool_cascade import ToolCascadePredictor
from .telemetry import Step

from .scenarios import (  # noqa: F401 — re-exported for backward compat
    DEGRADED,
    FAILING,
    HEALTHY,
    TOPICS,
    generate_runs,
)


# -- feature dataset --------------------------------------------------------------


def run_to_features(steps: List[Step]) -> np.ndarray:
    """Per-step feature matrix for one run (same pipeline as live inference)."""
    signals = [LoopPredictor(), ToolCascadePredictor(), BudgetDriftPredictor()]
    trajectories: List[List[float]] = [[] for _ in signals]
    errors: List[bool] = []
    tokens: List[float] = []
    latencies: List[float] = []
    history: List[Step] = []
    rows = []
    for step in steps:
        history.append(step)
        for traj, p in zip(trajectories, signals):
            traj.append(p.update(step, history))
        errors.append(bool(step.error))
        tokens.append(float(step.tokens or 0))
        latencies.append(float(step.latency_s or 0))
        rows.append(extract_features(trajectories, errors, tokens, latencies, len(history)))
    return np.stack(rows)


def build_dataset(n_per_type: int = 60, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for steps, labels in generate_runs(n_per_type, seed):
        xs.append(run_to_features(steps))
        ys.append(np.asarray(labels))
    return np.concatenate(xs), np.concatenate(ys)


# -- training ---------------------------------------------------------------------


def train(n_per_type: int = 60, seed: int = 42,
          out_path: str = str(DEFAULT_MODEL_PATH)) -> RiskEnsemble:
    print(f"generating scenario runs ({n_per_type} per failure type)...")
    X, y = build_dataset(n_per_type, seed)
    print(f"dataset: {X.shape[0]} steps x {NUM_FEATURES} features; "
          f"class counts: " + ", ".join(
              f"{name}={int((y == c).sum())}" for c, name in
              [(0, 'healthy'), (1, 'degraded'), (2, 'failing')]))

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    split = int(len(X) * 0.8)
    tr, te = idx[:split], idx[split:]

    print("training ensemble (5 attention-MLPs + Random Forest)...")
    model = RiskEnsemble(seed=seed).fit(X[tr], y[tr])

    pred = model.predict(X[te])
    print("\nholdout per-class precision/recall:")
    for c, name in [(0, "healthy"), (1, "degraded"), (2, "failing")]:
        tp = int(((pred == c) & (y[te] == c)).sum())
        fp = int(((pred == c) & (y[te] != c)).sum())
        fn = int(((pred != c) & (y[te] == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        print(f"  {name:<9} precision={prec:.3f} recall={rec:.3f} (n={int((y[te] == c).sum())})")

    # early-warning recall: of steps labeled degraded (precursor), how many
    # did the model flag as degraded-or-failing? this is the number that
    # matters for catching runs *before* they fail.
    mask = y[te] == DEGRADED
    if mask.any():
        flagged = (pred[mask] > 0).mean()
        print(f"  early-warning recall on precursors: {flagged:.3f}")

    att = model.feature_attention()
    order = np.argsort(att)[::-1][:6]
    from .predictors.model import SIGNAL_NAMES
    per_sig = [f"{s}_{stat}" for s in SIGNAL_NAMES for stat in ("cur", "mean", "max", "slope")]
    names = per_sig + ["error_rate", "token_velocity", "lat_mean", "lat_max",
                       "run_len", "error_count"]
    print("\nlearned feature attention (top 6):")
    for i in order:
        print(f"  {names[i]:<22} {att[i]:.3f}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    print(f"\nsaved model -> {out_path}")
    return model


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train the AgentGuard risk model")
    ap.add_argument("--runs-per-type", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    args = ap.parse_args()
    train(args.runs_per_type, args.seed, args.out)
