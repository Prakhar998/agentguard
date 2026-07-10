#!/usr/bin/env python3
"""AgentGuard demo: catch a looping agent before it wastes the run.

A fake research agent starts healthy, then falls into the classic
search -> summarize -> search loop. Watch the risk climb and AgentGuard
propose a halt — ~20 steps and a pile of tokens before the agent would
have hit its own max-iterations wall.

Zero external services, zero API keys:

    python examples/failing_agent_demo.py
"""

import time

from agentguard import Guard
from agentguard.adapters.raw import llm_output, tool_call, tool_result

GREEN, YELLOW, RED, DIM, BOLD, RESET = (
    "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[1m", "\033[0m",
)


def fake_research_agent():
    """Yields (description, step) pairs. Healthy start, then a loop."""
    # -- healthy phase: the agent makes real progress -------------------
    healthy = [
        (tool_call("search", {"q": "LLM agent failure modes"}), "searches the web"),
        (tool_result("search", "10 results: reasoning loops, tool cascades..."), "gets results"),
        (llm_output("Found several failure modes. Reading the top result.", tokens=90), "thinks"),
        (tool_call("fetch", {"url": "example.com/agent-failures"}), "fetches a page"),
        (tool_result("fetch", "<html>...long article...</html>"), "gets the page"),
        (llm_output("The article lists loops as the top failure. Summarizing.", tokens=110), "thinks"),
    ]
    for step, desc in healthy:
        yield desc, step

    # -- failure phase: search -> summarize -> search, forever ----------
    n = 0
    while True:
        n += 1
        yield "searches again (same query!)", tool_call(
            "search", {"q": f"LLM agent failure modes page {n}"}
        )
        yield "gets the same results", tool_result(
            "search", "10 results: reasoning loops, tool cascades..."
        )
        yield "summarizes... again", llm_output(
            "I still need more information about failure modes. Let me search.",
            tokens=100 + 40 * n,  # context keeps growing — budget drift too
        )


def bar(risk: float, width: int = 22) -> str:
    filled = int(risk * width)
    color = GREEN if risk < 0.5 else YELLOW if risk < 0.8 else RED
    return f"{color}{'█' * filled}{DIM}{'░' * (width - filled)}{RESET}"


def main():
    guard = Guard(
        predictors=["loop", "tool_cascade", "budget_drift"],
        threshold=0.8,
    )

    print(f"\n{BOLD}AgentGuard{RESET} watching a research agent "
          f"{DIM}(threshold 0.8){RESET}\n")
    print(f"{DIM}{'step':>4}  {'agent action':<32} {'risk':>5}  meter{RESET}")

    tokens_spent = 0
    with guard.watch(run_id="demo") as w:
        for i, (desc, step) in enumerate(fake_research_agent()):
            tokens_spent += step.tokens or 0
            risk = w.record(step)

            print(f"{i:>4}  {desc:<32} {risk:>5.2f}  {bar(risk)}")
            time.sleep(0.15)  # so the GIF is watchable

            if risk > guard.threshold:
                proposal = w.intervene("halt")
                loop_score = w.subscores.get("loop", 0.0)
                print(f"\n{RED}{BOLD}⛔ risk {risk:.2f} > 0.8 — halting the run{RESET}")
                print(f"   dominant signal: {BOLD}loop{RESET} = {loop_score:.2f} "
                      f"{DIM}(search→summarize→search repeating){RESET}")
                print(f"   sub-scores: " + ", ".join(
                    f"{k}={v:.2f}" for k, v in w.subscores.items()))
                print(f"   caught at step {proposal.step_index} after "
                      f"~{tokens_spent} tokens — the agent's own max-iterations "
                      f"wall was 20+ steps away.\n")
                break
        else:  # pragma: no cover
            pass

    if not w.interventions:
        print("\nrun finished healthy (this should not happen in the demo)")


if __name__ == "__main__":
    main()
