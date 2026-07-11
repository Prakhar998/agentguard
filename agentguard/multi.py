"""MultiGuard — watch a *team* of agents and predict cascades.

Multi-agent failures are rarely isolated: a researcher loops, hands
garbage to a summarizer, the supervisor burns budget retrying both. This
is the same shape ProactiveGuard handled in consensus clusters — you
don't watch one node, you watch the cluster and how degradation spreads.

MultiGuard runs one Watcher per agent plus a propagation model over the
data-flow edges you declare:

    mg = MultiGuard(predictors=["loop", "tool_cascade", "budget_drift"])
    mg.add_edge("researcher", "summarizer")   # researcher feeds summarizer
    mg.add_edge("summarizer", "supervisor")

    mg.record("researcher", step)
    mg.effective_risk("summarizer")   # own risk + upstream contagion
    mg.system_risk                    # the run-the-team-off-a-cliff number

``effective_risk`` is a noisy-OR of the agent's own risk and its worst
upstream neighbor's risk attenuated by ``coupling`` — one hop, because
that's what you can defend in an incident review: *"the summarizer looked
fine on its own numbers, but 60% of its input came from an agent at risk
0.9."*
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Union

from .guard import Guard, Intervention, Watcher
from .predictors.base import Predictor
from .telemetry import Step


class MultiGuard:
    def __init__(
        self,
        predictors: Optional[Sequence[Union[str, Predictor]]] = None,
        threshold: float = 0.8,
        coupling: float = 0.5,
        guard: Optional[Guard] = None,
    ):
        self.guard = guard or Guard(predictors=predictors, threshold=threshold)
        self.coupling = coupling
        self.watchers: Dict[str, Watcher] = {}
        self._upstream: Dict[str, Set[str]] = {}

    # -- topology -----------------------------------------------------------

    def add_edge(self, upstream: str, downstream: str) -> "MultiGuard":
        """Declare that ``upstream``'s output feeds ``downstream``."""
        self._upstream.setdefault(downstream, set()).add(upstream)
        self.watcher(upstream)
        self.watcher(downstream)
        return self

    def watcher(self, agent_id: str) -> Watcher:
        if agent_id not in self.watchers:
            self.watchers[agent_id] = self.guard.watch(run_id=agent_id)
        return self.watchers[agent_id]

    @property
    def agents(self) -> List[str]:
        return list(self.watchers)

    # -- recording ------------------------------------------------------------

    def record(self, agent_id: str, step: Union[Step, dict]) -> float:
        """Feed one step for one agent; returns that agent's effective risk."""
        self.watcher(agent_id).record(step)
        return self.effective_risk(agent_id)

    # -- risk -------------------------------------------------------------------

    def risk(self, agent_id: str) -> float:
        """The agent's own risk, upstream ignored."""
        return self.watcher(agent_id).risk

    def effective_risk(self, agent_id: str) -> float:
        """Own risk fused (noisy-OR) with attenuated worst-upstream risk."""
        own = self.risk(agent_id)
        upstream = [
            self.watchers[u].risk
            for u in self._upstream.get(agent_id, ())
            if u in self.watchers
        ]
        contagion = self.coupling * max(upstream, default=0.0)
        return 1.0 - (1.0 - own) * (1.0 - contagion)

    @property
    def system_risk(self) -> float:
        """Noisy-OR across every agent's own risk: P(at least one agent
        is taking the run down)."""
        survival = 1.0
        for w in self.watchers.values():
            survival *= 1.0 - w.risk
        return 1.0 - survival

    # -- reporting / intervention ------------------------------------------------

    def report(self) -> Dict[str, dict]:
        return {
            agent_id: {
                "risk": round(w.risk, 4),
                "effective_risk": round(self.effective_risk(agent_id), 4),
                "steps": len(w.history),
                "subscores": {k: round(v, 4) for k, v in w.subscores.items()},
                "upstream": sorted(self._upstream.get(agent_id, ())),
            }
            for agent_id, w in self.watchers.items()
        }

    def worst_agent(self) -> Optional[str]:
        if not self.watchers:
            return None
        return max(self.watchers, key=lambda a: self.effective_risk(a))

    def intervene(self, agent_id: str, action: str = "halt") -> Intervention:
        return self.watcher(agent_id).intervene(action)

    def close(self) -> None:
        for w in self.watchers.values():
            w.close()
