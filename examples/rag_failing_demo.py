#!/usr/bin/env python3
"""AgentGuard catching a RAG agent going off the rails.

A fake support-bot answers questions over a tiny knowledge base. It starts
grounded, then degrades the way real RAG agents do: it re-retrieves the
same chunks hoping for a better answer, its queries wander past what the
corpus can answer, and its outputs drift away from the retrieved sources
(hallucination). Watch retrieval_drift and grounding_gap climb.

Zero keys, zero services (embeddings fall back to a local hasher):

    python examples/rag_failing_demo.py
"""

import time

from agentguard import Guard
from agentguard._embed import hash_embed
from agentguard.adapters.raw import llm_output, retrieval
from agentguard.predictors.grounding_gap import GroundingGapPredictor
from agentguard.predictors.retrieval_drift import RetrievalDriftPredictor

GREEN, YELLOW, RED, DIM, BOLD, RESET = (
    "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[1m", "\033[0m",
)

KB = {
    "pricing": "The enterprise plan costs $500 per month, including SSO and support.",
    "refunds": "Refunds are processed within 14 days of a written cancellation request.",
    "security": "All customer data is encrypted at rest (AES-256) and in transit (TLS 1.3).",
}


def fake_rag_agent():
    """Yields (description, step). Grounded start, then RAG-flavored decay."""
    healthy = [
        ("retrieves pricing docs", retrieval("enterprise plan monthly cost", [KB["pricing"]])),
        ("answers from sources", llm_output(
            "The enterprise plan costs $500 per month including SSO and support.", tokens=80)),
        ("retrieves refund policy", retrieval("refund cancellation window", [KB["refunds"]])),
        ("answers from sources", llm_output(
            "Refunds are processed within 14 days of a written cancellation request.", tokens=85)),
    ]
    for desc, step in healthy:
        yield desc, step

    n = 0
    while True:
        n += 1
        # re-retrieves the same chunks, hoping for a different answer
        yield "re-retrieves the same chunks", retrieval(
            f"pricing question attempt {n}", [KB["pricing"]]
        )
        # ...and drifts further from the sources each time
        hallucinations = [
            "The plan might also include a dedicated account manager, I believe.",
            "I think there is probably a lifetime deal with unlimited seats somewhere.",
            "Historically, medieval merchants negotiated bulk discounts on spice routes.",
            "The moon landing budget in 1969 was famously large compared to software.",
        ]
        yield "answer drifts from sources", llm_output(
            hallucinations[min(n - 1, len(hallucinations) - 1)], tokens=90 + 10 * n
        )


def bar(risk: float, width: int = 22) -> str:
    filled = int(risk * width)
    color = GREEN if risk < 0.5 else YELLOW if risk < 0.8 else RED
    return f"{color}{'█' * filled}{DIM}{'░' * (width - filled)}{RESET}"


def main():
    guard = Guard(
        predictors=[
            RetrievalDriftPredictor(embed_fn=hash_embed),
            GroundingGapPredictor(embed_fn=hash_embed),
        ],
        threshold=0.8,
    )

    print(f"\n{BOLD}AgentGuard{RESET} watching a RAG support bot "
          f"{DIM}(threshold 0.8){RESET}\n")
    print(f"{DIM}{'step':>4}  {'agent action':<30} {'risk':>5}  meter{RESET}")

    with guard.watch(run_id="rag-demo") as w:
        for i, (desc, step) in enumerate(fake_rag_agent()):
            risk = w.record(step)
            print(f"{i:>4}  {desc:<30} {risk:>5.2f}  {bar(risk)}")
            time.sleep(0.15)

            if risk > guard.threshold:
                w.intervene("reset_context")
                print(f"\n{RED}{BOLD}⛔ risk {risk:.2f} > 0.8 — proposing reset_context{RESET}")
                print("   sub-scores: " + ", ".join(
                    f"{k}={v:.2f}" for k, v in w.subscores.items()))
                print(f"   the bot is re-retrieving the same chunks and its answers "
                      f"are drifting\n   from the sources — classic hallucination "
                      f"spiral. Reset and re-anchor.\n")
                break


if __name__ == "__main__":
    main()
