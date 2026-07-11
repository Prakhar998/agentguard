"""AgentGuard — predict LLM-agent-run failures while the run is live.

ProactiveGuard (predictive failure detection for distributed consensus),
re-targeted at LLM agent runs: watch the step stream, learn the failure
precursors, flag the run early enough to intervene.
"""

import logging

from .aggregate import Aggregator, ConformalCalibrator
from .guard import Guard, Intervention, Watcher
from .multi import MultiGuard
from .telemetry import Step, StepKind

__version__ = "0.4.1"

logging.getLogger("agentguard").addHandler(logging.NullHandler())

__all__ = [
    "Guard",
    "MultiGuard",
    "Watcher",
    "Intervention",
    "Step",
    "StepKind",
    "Aggregator",
    "ConformalCalibrator",
]
