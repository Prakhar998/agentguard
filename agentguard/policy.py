"""Intervention policies — make the proposals operational.

AgentGuard's contract is "propose, the host decides". A Policy is the
host's decision, written down once:

    from agentguard import Guard
    from agentguard.policy import Policy, Rule, SlackWebhook

    policy = Policy(
        rules=[
            Rule(signal="loop", threshold=0.8, action="reset_context"),
            Rule(signal="budget_drift", threshold=0.7, sustain=3, action="downgrade"),
            Rule(threshold=0.9, action="escalate"),   # fused risk, any cause
        ],
        notifiers=[SlackWebhook("https://hooks.slack.com/services/...")],
    )
    guard = Guard(predictors=[...], on_step=policy)

Each triggered rule records an Intervention on the Watcher (so the run's
history says what was decided and why) and notifies. Enforcement still
belongs to your loop: check ``watcher.interventions`` (or use the
LangChain/LangGraph adapters' ``auto_intervene`` to turn proposals into
halts). Notifiers never raise into the run.
"""

from __future__ import annotations

import json
import logging
import urllib.request
import weakref
from dataclasses import dataclass
from typing import Callable, List, Optional

from .guard import INTERVENTIONS, Intervention, Watcher

logger = logging.getLogger("agentguard")


@dataclass
class Rule:
    """When ``signal`` (a predictor name, or None for the fused risk) has
    been at/over ``threshold`` for ``sustain`` consecutive steps, propose
    ``action``. Fires ``max_fires`` times per run (default: once)."""

    action: str
    threshold: float = 0.8
    signal: Optional[str] = None
    sustain: int = 1
    max_fires: int = 1

    def __post_init__(self):
        if self.action not in INTERVENTIONS:
            raise ValueError(f"action must be one of {INTERVENTIONS}, got {self.action!r}")

    def value(self, watcher: Watcher) -> float:
        if self.signal is None:
            return watcher.risk
        return watcher.subscores.get(self.signal, 0.0)


class Policy:
    """Evaluate rules on every step; intervene and notify when one fires.

    Pass as ``Guard(on_step=policy)``. Per-watcher state (streaks, fire
    counts) is keyed by the watcher, so one Policy serves many runs.
    """

    def __init__(self, rules: List[Rule],
                 notifiers: Optional[List[Callable[[Watcher, Intervention, Rule], None]]] = None):
        self.rules = list(rules)
        self.notifiers = list(notifiers or [])
        # weak keys: state dies with its watcher (a plain id() key would be
        # reused by CPython after GC, leaking one run's fire-counts into the next)
        self._streaks: "weakref.WeakKeyDictionary[Watcher, List[int]]" = (
            weakref.WeakKeyDictionary())
        self._fired: "weakref.WeakKeyDictionary[Watcher, List[int]]" = (
            weakref.WeakKeyDictionary())

    def __call__(self, watcher: Watcher) -> None:
        streaks = self._streaks.setdefault(watcher, [0] * len(self.rules))
        fired = self._fired.setdefault(watcher, [0] * len(self.rules))

        for i, rule in enumerate(self.rules):
            if rule.value(watcher) >= rule.threshold:
                streaks[i] += 1
            else:
                streaks[i] = 0
                continue
            if streaks[i] >= rule.sustain and fired[i] < rule.max_fires:
                fired[i] += 1
                proposal = watcher.intervene(rule.action)
                for notify in self.notifiers:
                    try:
                        notify(watcher, proposal, rule)
                    except Exception:  # a down webhook must not take the run down
                        logger.exception("policy notifier failed")


# -- notifiers ---------------------------------------------------------------------


def _explanation(watcher: Watcher) -> List[dict]:
    memory = watcher.guard.memory
    if memory is None:
        return []
    try:
        return [
            {"run_id": m.get("run_id"), "similarity": round(m.get("similarity", 0.0), 3),
             "summary": m.get("summary")}
            for m in memory.similar_failures(watcher, k=3)
        ]
    except Exception:
        return []


class Webhook:
    """POST the intervention (risk, sub-scores, similar past failures) as
    JSON to any endpoint. ``transport`` is injectable for tests."""

    def __init__(self, url: str, timeout: float = 3.0,
                 transport: Optional[Callable[[str, bytes], None]] = None):
        self.url = url
        self.timeout = timeout
        self._transport = transport or self._post

    def _post(self, url: str, body: bytes) -> None:
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=self.timeout).read()

    def payload(self, watcher: Watcher, proposal: Intervention, rule: Rule) -> dict:
        return {
            "source": "agentguard",
            "run_id": watcher.run_id,
            "action": proposal.action,
            "risk": round(proposal.risk, 4),
            "step_index": proposal.step_index,
            "rule": {"signal": rule.signal or "risk", "threshold": rule.threshold,
                     "sustain": rule.sustain},
            "subscores": {k: round(v, 4) for k, v in proposal.subscores.items()},
            "similar_past_failures": _explanation(watcher),
        }

    def __call__(self, watcher: Watcher, proposal: Intervention, rule: Rule) -> None:
        self._transport(self.url, json.dumps(self.payload(watcher, proposal, rule)).encode())


class SlackWebhook(Webhook):
    """Same, shaped for a Slack incoming webhook."""

    def __call__(self, watcher: Watcher, proposal: Intervention, rule: Rule) -> None:
        signals = ", ".join(
            f"{k}={v:.2f}" for k, v in proposal.subscores.items() if v > 0.05) or "n/a"
        similar = _explanation(watcher)
        lines = [
            f":rotating_light: *AgentGuard* proposes `{proposal.action}` on run "
            f"`{watcher.run_id}` — risk {proposal.risk:.2f} at step {proposal.step_index}",
            f"signals: {signals}",
        ]
        if similar:
            lines.append(f"looks like {len(similar)} past failure(s), e.g. "
                         f"_{similar[0]['summary']}_")
        self._transport(self.url, json.dumps({"text": "\n".join(lines)}).encode())
