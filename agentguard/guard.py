"""Guard + Watcher — the whole public API.

    from agentguard import Guard

    guard = Guard(predictors=["loop", "tool_cascade", "budget_drift"])

    with guard.watch() as w:
        for step in my_agent_loop():
            w.record(step)
            if w.risk > 0.8:
                w.intervene("halt")
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Union

from .aggregate import Aggregator
from .predictors.base import Predictor
from .telemetry import Step, StepKind

logger = logging.getLogger("agentguard")

INTERVENTIONS = ("halt", "reset_context", "escalate", "downgrade")


def _build_predictor(spec: Union[str, Predictor]) -> Predictor:
    if isinstance(spec, Predictor):
        return spec
    name = str(spec)
    if name == "loop":
        from .predictors.loop import LoopPredictor

        return LoopPredictor()
    if name == "tool_cascade":
        from .predictors.tool_cascade import ToolCascadePredictor

        return ToolCascadePredictor()
    if name == "budget_drift":
        from .predictors.budget_drift import BudgetDriftPredictor

        return BudgetDriftPredictor()
    if name == "semantic_drift":
        from .predictors.semantic_drift import SemanticDriftPredictor

        return SemanticDriftPredictor()
    if name == "retrieval_drift":
        from .predictors.retrieval_drift import RetrievalDriftPredictor

        return RetrievalDriftPredictor()
    if name == "grounding_gap":
        from .predictors.grounding_gap import GroundingGapPredictor

        return GroundingGapPredictor()
    if name == "goal_drift":
        from .predictors.goal_drift import GoalDriftPredictor

        return GoalDriftPredictor()
    if name == "model":
        from .predictors.model import LearnedRiskPredictor

        return LearnedRiskPredictor()
    raise ValueError(
        f"Unknown predictor {name!r}. Built-ins: loop, tool_cascade, "
        f"budget_drift, semantic_drift, retrieval_drift, grounding_gap, "
        f"goal_drift, model — or pass a Predictor instance."
    )


@dataclass
class Intervention:
    """A proposed intervention. AgentGuard proposes; the host app decides."""

    action: str
    risk: float
    step_index: int
    run_id: str
    subscores: Dict[str, float] = field(default_factory=dict)


class Guard:
    """Configuration holder; spawn a :class:`Watcher` per run via watch()."""

    def __init__(
        self,
        predictors: Optional[Sequence[Union[str, Predictor]]] = None,
        on_risk: Optional[Callable[["Watcher"], None]] = None,
        threshold: float = 0.8,
        aggregator: Optional[Aggregator] = None,
        memory: Optional[object] = None,
    ):
        self.predictor_specs = list(
            predictors if predictors is not None else ["loop", "tool_cascade", "budget_drift"]
        )
        self.on_risk = on_risk
        self.threshold = threshold
        self.aggregator = aggregator or Aggregator()
        self.memory = memory  # optional FailureMemory (agentguard.memory)

    def watch(self, run_id: Optional[str] = None, goal: Optional[str] = None) -> "Watcher":
        """``goal``: the task the run is trying to accomplish — enables
        goal-aware predictors like goal_drift."""
        return Watcher(self, run_id=run_id, goal=goal)


class Watcher:
    """Watches one agent run. Context manager; also usable bare."""

    def __init__(self, guard: Guard, run_id: Optional[str] = None,
                 goal: Optional[str] = None):
        self.guard = guard
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.goal = goal
        self.predictors: List[Predictor] = [
            _build_predictor(spec) for spec in guard.predictor_specs
        ]
        for p in self.predictors:
            p.reset()
            if hasattr(p, "bind"):
                p.bind(self)
        self.history: List[Step] = []
        self.subscores: Dict[str, float] = {p.name: 0.0 for p in self.predictors}
        self.risk_trajectory: List[float] = []
        self.subscore_trajectory: List[Dict[str, float]] = []
        self.interventions: List[Intervention] = []
        self._risk: float = 0.0
        self._above_threshold = False

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> "Watcher":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        memory = self.guard.memory
        if memory is not None and self.interventions:
            try:
                memory.add_run(self)
            except Exception:  # memory must never take a run down
                logger.exception("failure-memory add_run failed")

    # -- recording -----------------------------------------------------------

    def record(self, step: Union[Step, dict]) -> float:
        """Feed one step; returns the updated calibrated risk."""
        if not isinstance(step, Step):
            step = Step.from_dict(step)
        step.index = len(self.history)
        self.history.append(step)

        for p in self.predictors:
            try:
                self.subscores[p.name] = p.update(step, self.history)
            except Exception:  # a broken predictor must not kill the run
                logger.exception("predictor %s failed on step %d", p.name, step.index)

        self._risk = self.guard.aggregator.fuse(self.subscores)
        self.risk_trajectory.append(self._risk)
        self.subscore_trajectory.append(dict(self.subscores))

        if self._risk >= self.guard.threshold and not self._above_threshold:
            self._above_threshold = True
            if self.guard.on_risk is not None:
                self.guard.on_risk(self)
        elif self._risk < self.guard.threshold:
            self._above_threshold = False

        return self._risk

    @property
    def risk(self) -> float:
        """Current calibrated 0-1 risk."""
        return self._risk

    # -- intervention ----------------------------------------------------------

    def intervene(self, action: str = "halt") -> Intervention:
        """Propose an intervention. Logged and returned — never enforced.

        The host app owns the control flow (same human-in-the-loop
        discipline as a --fix flag that prints the fix instead of applying
        it). ``action``: halt | reset_context | escalate | downgrade.
        """
        if action not in INTERVENTIONS:
            raise ValueError(f"action must be one of {INTERVENTIONS}, got {action!r}")
        proposal = Intervention(
            action=action,
            risk=self._risk,
            step_index=len(self.history) - 1,
            run_id=self.run_id,
            subscores=dict(self.subscores),
        )
        self.interventions.append(proposal)
        logger.warning(
            "agentguard run=%s step=%d risk=%.2f -> proposing %s (%s)",
            self.run_id,
            proposal.step_index,
            self._risk,
            action,
            ", ".join(f"{k}={v:.2f}" for k, v in self.subscores.items() if v > 0.05),
        )
        return proposal

    # -- explainability -----------------------------------------------------

    def signature(self) -> Dict[str, object]:
        """Compact fingerprint of this run for the failure memory."""
        tools = [s.name for s in self.history if s.kind == StepKind.TOOL_CALL and s.name]
        return {
            "run_id": self.run_id,
            "steps": len(self.history),
            "peak_risk": max(self.risk_trajectory, default=0.0),
            "subscores": dict(self.subscores),
            "tool_sequence_tail": tools[-10:],
        }

    def explain(self, k: int = 3) -> List[dict]:
        """Retrieve the k most similar past failures from the attached
        failure memory (RAG over failure signatures). Empty without one."""
        memory = self.guard.memory
        if memory is None:
            return []
        return memory.similar_failures(self, k=k)
