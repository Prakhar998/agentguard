#!/usr/bin/env python3
"""AgentGuard watching an agent *team* — and predicting the cascade.

A supervisor delegates to two workers. The researcher falls into a
search loop; the writer looks healthy on its own numbers — but it is
consuming the researcher's output, so its *effective* risk climbs before
its own signals fire. This is the cluster-level view ProactiveGuard took
of consensus nodes, applied to an agent team.

Zero keys, zero services:

    python examples/multiagent_demo.py
"""

import time

from agentguard.adapters.raw import llm_output, tool_call, tool_result
from agentguard.multi import MultiGuard

GREEN, YELLOW, RED, DIM, BOLD, RESET = (
    "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[1m", "\033[0m",
)


def color(r):
    return GREEN if r < 0.5 else YELLOW if r < 0.8 else RED


def team_turns():
    """Yields (agent, step) — a supervisor/researcher/writer team where
    the researcher starts looping after two healthy rounds."""
    for i in range(2):  # healthy rounds
        uid = "abcdef"[i] * 3
        yield "supervisor", llm_output(f"Delegating subtopic {uid} to researcher.", tokens=60)
        yield "researcher", tool_call("search", {"q": f"subtopic {uid} sources"})
        yield "researcher", tool_result("search", f"sources about {uid}", tokens=40)
        yield "researcher", llm_output(f"Found sources on {uid}, handing off.", tokens=90)
        yield "writer", llm_output(f"Drafted the section about {uid}.", tokens=120)

    n = 0
    while True:  # researcher loops; writer keeps drafting from its output
        n += 1
        yield "researcher", tool_call("search", {"q": "subtopic final sources"})
        yield "researcher", llm_output("I still need more sources for this subtopic.",
                                       tokens=100)
        yield "writer", llm_output(f"Waiting on sources, padding the draft (rev {'x' * n}).",
                                   tokens=110)


def main():
    mg = MultiGuard(predictors=["loop", "tool_cascade", "budget_drift"], threshold=0.8)
    mg.add_edge("researcher", "writer")       # researcher's output feeds the writer
    mg.add_edge("writer", "supervisor")

    print(f"\n{BOLD}AgentGuard{RESET} watching a 3-agent team "
          f"{DIM}(researcher -> writer -> supervisor){RESET}\n")
    print(f"{DIM}{'turn':>4}  {'agent':<11} own    effective   (contagion from upstream){RESET}")

    for turn, (agent, step) in enumerate(team_turns()):
        mg.record(agent, step)
        own, eff = mg.risk(agent), mg.effective_risk(agent)
        note = ""
        if eff - own > 0.15:
            worst_up = max(mg._upstream.get(agent, ()), key=mg.risk, default="")
            note = f"{DIM}<- contagion from {worst_up} (risk {mg.risk(worst_up):.2f}){RESET}"
        print(f"{turn:>4}  {agent:<11} {color(own)}{own:.2f}{RESET}   "
              f"{color(eff)}{eff:.2f}{RESET}        {note}")
        time.sleep(0.12)

        if mg.system_risk > 0.85:
            worst = mg.worst_agent()
            mg.intervene(worst, "halt")
            print(f"\n{RED}{BOLD}⛔ system risk {mg.system_risk:.2f} — halting "
                  f"the {worst!r} agent{RESET}")
            for name, info in mg.report().items():
                print(f"   {name:<11} own={info['risk']:.2f} "
                      f"effective={info['effective_risk']:.2f} "
                      f"steps={info['steps']}")
            print(f"   the writer never tripped its own predictors — the cascade "
                  f"was visible\n   in effective risk two turns earlier.\n")
            break

    mg.close()


if __name__ == "__main__":
    main()
