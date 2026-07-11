# AgentGuard on real SWE-agent runs

Dataset: [`nebius/SWE-agent-trajectories`](https://huggingface.co/datasets/nebius/SWE-agent-trajectories) — 300 runs (100 resolved, 200 unresolved), deterministic predictors only (`loop,tool_cascade,budget_drift`), threshold 0.8.

| metric | value |
|---|---|
| flag rate on unresolved runs | **36%** |
| false-alarm rate on resolved runs | **12%** |
| mean early-warning lead (flagged runs) | **73.9 steps** |
| median lead | 41 steps |

## By how the unresolved run actually ended

| exit class | flagged | mean lead |
|---|---|---|
| clean submission (wrong patch) | 31/101 (31%) | 74.5 steps |
| exit_cost/context | 41/99 (41%) | 73.5 steps |

## Threshold sweep (pick your operating point)

| threshold | dysfunction flagged | wrong-patch flagged | false alarms (resolved) |
|---|---|---|---|
| 0.6 | 56% | 43% | 18% |
| 0.7 | 48% | 39% | 14% |
| 0.8 | 41% | 31% | 12% |
| 0.9 | 35% | 25% | 7% |

## Honest reading

- **Dysfunction vs wrong answers.** AgentGuard predicts *dysfunction* (loops, error cascades, budget blowout). Runs that died of `exit_cost`/`exit_context` are the dysfunction class — the flag rate there is the number that matters, and every step of lead time is unspent budget. A run that submitted a clean-but-wrong patch is often indistinguishable from a healthy run mid-flight, and the low flag rate on that class is expected, not a miss.
- Tokens are estimated (len/4); tool errors are inferred from observation text. Both proxies are stated in the converter.
- Reproduce: `python benchmarks/swe_agent_bench.py`.
