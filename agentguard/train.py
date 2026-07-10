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

import random
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np

from .adapters.raw import llm_output, tool_call, tool_result
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

HEALTHY, DEGRADED, FAILING = 0, 1, 2

TOPICS = [
    "quarterly sales", "user churn", "api latency", "carbon footprint",
    "vendor contracts", "onboarding funnel", "security incidents",
    "cloud spend", "support backlog", "release notes",
]


# -- synthetic scenario runs -----------------------------------------------------


def _healthy_steps(rng: random.Random, n_rounds: int) -> List[Step]:
    # NB: the loop predictor normalizes digits away, so healthy queries must
    # differ in *letters* — otherwise "report 3" and "report 7" on a repeated
    # topic look like a loop and poison the healthy class.
    steps = []
    for i in range(n_rounds):
        topic = rng.choice(TOPICS)
        uid = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))
        tokens = rng.randint(80, 140)
        steps.append(tool_call("search", {"q": f"{topic} {uid} report"}))
        steps.append(tool_result("search", f"results about {topic} {uid}",
                                 tokens=rng.randint(30, 60),
                                 error=rng.random() < 0.03))
        steps.append(llm_output(f"Progress on {topic}: found {uid}, next step differs.",
                                tokens=tokens, latency_s=rng.uniform(0.5, 2.0)))
    return steps


def generate_runs(n_per_type: int = 60, seed: int = 42) -> Iterator[Tuple[List[Step], List[int]]]:
    """Yield (steps, per-step labels) for each scenario run."""
    rng = random.Random(seed)

    for _ in range(n_per_type * 2):  # healthy is the majority class, as in prod
        steps = _healthy_steps(rng, rng.randint(4, 12))
        yield steps, [HEALTHY] * len(steps)

    for _ in range(n_per_type):  # loop failure
        prefix = _healthy_steps(rng, rng.randint(2, 5))
        labels = [HEALTHY] * len(prefix)
        steps = list(prefix)
        topic = rng.choice(TOPICS)
        n_cycles = rng.randint(4, 8)
        for c in range(n_cycles):
            steps.append(tool_call("search", {"q": f"{topic} more details"}))
            steps.append(llm_output(f"I still need more information about {topic}.",
                                    tokens=rng.randint(90, 130)))
            label = DEGRADED if c < 2 else FAILING
            labels.extend([label, label])
        yield steps, labels

    for _ in range(n_per_type):  # tool cascade failure
        prefix = _healthy_steps(rng, rng.randint(2, 5))
        labels = [HEALTHY] * len(prefix)
        steps = list(prefix)
        n_errs = rng.randint(4, 8)
        for e in range(n_errs):
            steps.append(tool_call("fetch", {"url": f"https://site-{rng.random():.4f}.com"}))
            steps.append(tool_result("fetch", "connection refused", error=True,
                                     latency_s=rng.uniform(5, 20)))
            label = DEGRADED if e < 2 else FAILING
            labels.extend([label, label])
        yield steps, labels

    for _ in range(n_per_type):  # budget blowout
        prefix = _healthy_steps(rng, 4)
        labels = [HEALTHY] * len(prefix)
        steps = list(prefix)
        base = 120
        for g in range(rng.randint(5, 9)):
            tokens = int(base * (2.0 + g * 1.5))
            steps.append(llm_output(f"Expanding context with everything about {rng.choice(TOPICS)} "
                                    + "…" * g, tokens=tokens, latency_s=rng.uniform(2, 8)))
            labels.append(DEGRADED if g < 2 else FAILING)
        yield steps, labels


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
