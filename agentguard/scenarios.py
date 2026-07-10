"""Synthetic scenario runs (stdlib-only).

Shared by the trainer (labeled dataset bootstrap, ProactiveGuard-style)
and the replay CLI's --demo mode. Replace ``generate_runs`` with your own
traced runs when you have them.
"""

from __future__ import annotations

import random
from typing import Iterator, List, Tuple

from .adapters.raw import llm_output, tool_call, tool_result
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
