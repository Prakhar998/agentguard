# AgentGuard

**Predict LLM-agent-run failures in real time — and intervene before the tokens are spent.**

Every agent-observability tool today (LangSmith, Langfuse, Arize, Braintrust) is
*reactive*: it shows you the trace after the run burned $30 and forty steps going
in circles. AgentGuard is *predictive*: it watches the step stream live, scores
the risk that the run is going off the rails, and proposes an intervention
(halt / reset context / escalate / downgrade) **early enough for it to matter**.

```
step  agent action                      risk  meter
   5  thinks                            0.00  ░░░░░░░░░░░░░░░░░░░░░░
   9  searches again (same query!)      0.25  █████░░░░░░░░░░░░░░░░░
  12  searches again (same query!)      0.50  ███████████░░░░░░░░░░░
  15  searches again (same query!)      0.75  ████████████████░░░░░░
  18  searches again (same query!)      1.00  ██████████████████████

⛔ risk 1.00 > 0.8 — halting the run
   dominant signal: loop = 1.00 (search→summarize→search repeating)
```

## Where this comes from: ProactiveGuard

AgentGuard is a direct re-targeting of my published failure-prediction work,
**[ProactiveGuard: Deep Learning-Based Predictive Failure Detection for
Distributed Consensus Systems]** — same thesis, new domain.

ProactiveGuard's finding was that consensus-node failures don't happen out of
nowhere: they are preceded by observable degradation (rising WAL-fsync latency,
climbing heartbeat latency), and a model trained on those precursors can flag a
node *before* it fails, cutting per-failure downtime by 86.7% via graceful
leader handoff.

LLM agent runs degrade the same way. The failure precursors just have different
names:

| ProactiveGuard (consensus nodes) | AgentGuard (agent runs) |
|---|---|
| WAL fsync latency rising | tool-error rate rising |
| heartbeat latency climbing | step latency / token velocity climbing |
| leader-election churn | repeated tool-call cycles (loops) |
| log-replication lag | semantic drift — the run stops making progress |
| reactive timeout detectors | reactive trace viewers |
| predict → graceful handoff | predict → halt / reset / escalate / downgrade |

The learned risk model here is a port of the ProactiveGuard architecture, not a
lookalike: the same learnable per-feature attention gate, residual connections,
focal loss `-(1-p_t)² log(p_t)` for the rare-failure class imbalance, and the
same ensemble (five bagged attention-MLPs + a Random Forest at double weight).

## Install

```bash
pip install agentguard                 # zero-dependency core
pip install "agentguard[langchain]"    # + LangChain callback adapter
pip install "agentguard[embeddings]"   # + sentence-transformers drift detection
pip install "agentguard[memory]"       # + Chroma-backed failure memory
pip install "agentguard[model]"        # + the learned risk model (numpy/sklearn)
```

## The whole API is 8 lines

```python
from agentguard import Guard

guard = Guard(predictors=["loop", "tool_cascade", "budget_drift", "semantic_drift"])

with guard.watch() as w:
    for step in my_agent_loop():        # ANY agent — raw, LangChain, LlamaIndex
        w.record(step)                  # dict or Step; normalized internally
        if w.risk > 0.8:
            w.intervene("halt")         # stop before wasting 20 more steps / $30
```

`w.record` accepts a plain dict (`{"kind": "tool_call", "name": "search",
"content": {...}}`) or a `Step`. `w.risk` is the current calibrated 0–1 risk;
`w.subscores` breaks it down per predictor for explainability. `w.intervene`
**proposes** — it logs and returns an `Intervention`; your app owns the control
flow.

Try it now, no keys, no services:

```bash
python examples/failing_agent_demo.py
```

## Predictors

| predictor | type | what it catches |
|---|---|---|
| `loop` | deterministic | repeated tool calls with near-identical args; search→summarize→search cycles; the same LLM output recurring |
| `tool_cascade` | deterministic | tool errors clustering (one error is noise, three in five steps is a cascade) |
| `budget_drift` | deterministic | token velocity blowing past the run's own early baseline |
| `semantic_drift` | embeddings | the run stalling (restating itself) or oscillating (A→B→A) in embedding space instead of progressing |
| `retrieval_drift` | embeddings (RAG) | re-retrieval loops (same chunks again and again) and retrieval starvation (query→chunk relevance decaying) |
| `grounding_gap` | embeddings (RAG) | LLM outputs drifting away from the retrieved context — a hallucination precursor, scored live |
| `goal_drift` | embeddings | outputs wandering away from the stated goal (`guard.watch(goal="...")`) |
| `model` | learned | the ported ProactiveGuard ensemble fusing all sub-signals into one risk |

Sub-scores are fused with a weighted noisy-OR (one strong signal dominates;
several weak signals compound) and can be calibrated to a real probability with
split-conformal calibration (`ConformalCalibrator`) — a risk score is only as
trustworthy as its calibration.

### The learned model

```bash
pip install "agentguard[model]"
python -m agentguard.train        # bootstrap on labeled synthetic scenario runs
```

```python
guard = Guard(predictors=["model"], threshold=0.8)
```

Training mirrors the ProactiveGuard bootstrap: generate scenario runs
(healthy / loop / cascade / budget-blowout) with gradual failure onset, label
each step `healthy → degraded → failing` by distance-to-failure, train the
ensemble on windowed features. A pre-trained model ships with the package;
retrain on your own traced runs by swapping `generate_runs`.

The attention gate is inspectable — ask the model what it found predictive:

```
learned feature attention (top 6):
  loop_slope             0.515
  budget_drift_max       0.513
  error_rate             0.512
```

(In the paper, the analogous top features were WAL-fsync and heartbeat latency.)

## Guarding RAG pipelines

RAG agents fail in their own ways, all observable in embedding space while
the run is live. Emit a `RETRIEVAL` step (or let the LangChain adapter's
retriever hooks do it) and two predictors watch it:

```python
from agentguard import Guard
from agentguard.adapters.raw import retrieval, llm_output

guard = Guard(predictors=["retrieval_drift", "grounding_gap"])
with guard.watch() as w:
    w.record(retrieval(query, chunks))       # what the vector store returned
    w.record(llm_output(answer, tokens=n))   # graded against those sources
```

`grounding_gap` is groundedness made *predictive*: instead of a post-hoc
eval score, the gap between outputs and retrieved context is tracked
against the run's own best grounding, so the hallucination spiral is
flagged while a `reset_context` can still save the run. See it happen:

```bash
python examples/rag_failing_demo.py
```

```
   4  re-retrieves the same chunks    0.67  ██████████████░░░░░░░░
   5  answer drifts from sources      0.91  ████████████████████░░

⛔ risk 0.91 > 0.8 — proposing reset_context
   sub-scores: retrieval_drift=0.77, grounding_gap=1.00
```

## Backtest on your own traces

```bash
agentguard replay traces.jsonl        # your exported runs (JSONL, see below)
agentguard replay --demo              # synthesized runs, no file needed
```

Replays historical runs through the predictors and reports the number that
sells prediction — the same early-warning metrics as the ProactiveGuard
paper:

```
runs replayed: 75   threshold: 0.8
failed runs caught before the end: 100% (45 failed runs)
early-warning lead: mean 4.2 steps, median 4 steps before the run ended
false alarms on successful runs: 0% (30 success runs)
```

Trace format is one JSON object per line — steps plus optional outcome
lines (`{"run_id": "r1", "outcome": "failed"}`); anything you can't export
simply doesn't contribute. Use `--json` for machine-readable output and
`--predictors`/`--threshold` to test configurations against history before
changing production.

## LangChain adapter

```python
from agentguard import Guard
from agentguard.adapters.langchain import AgentGuardCallback, AgentGuardHalt

handler = AgentGuardCallback(Guard(), auto_intervene="halt")

try:
    agent.invoke({"input": "..."}, config={"callbacks": [handler]})
except AgentGuardHalt as halt:
    print(halt.watcher.subscores)   # why it was stopped
```

Every `on_tool_start/end/error` and `on_llm_end` becomes a Step — nothing else
about your agent changes. `python examples/langchain_demo.py` runs a real
LangChain tool-calling loop (scripted fake model, so it's keyless) and shows
AgentGuard stopping it 11 turns before the agent's own iteration cap.

## Failure memory (RAG over past failures)

```python
from agentguard.memory import FailureMemory

guard = Guard(memory=FailureMemory())   # in-memory store; Chroma via [memory]

# ... failed runs get stored on close; on the next risky run:
for match in watcher.explain(k=3):
    print(f"{match['similarity']:.2f}  {match['summary']}")
# 0.94  dominant signal loop; loop=1.00 ...; tool tail: search -> summarize -> search
```

Failure *signatures* (sub-score trajectory + tool-sequence tail) are embedded
into a vector store; a new high-risk run retrieves its nearest past failures so
the alert reads *"this looks like the 14 past runs that looped on
search→summarize→search"* — retrieval-augmented explanation.

Retrieval is **hybrid** by default: dense signature-text embeddings, a
*trajectory embedding* of the sub-score time series (so a slow-ramp loop
matches other slow-ramp loops by shape), and keyword overlap, fused with
reciprocal-rank fusion. Pick one with `mode="dense" | "trajectory" | "keyword"`.

Once enough failures accumulate, cluster them into a taxonomy:

```python
for cluster in memory.taxonomy():
    print(f"{cluster['size']} runs — {cluster['exemplar']}")
# 14 runs — dominant signal loop; loop=1.00 ...; tool tail: search -> summarize -> search
#  6 runs — dominant signal tool_cascade; ...; tool tail: fetch -> fetch -> fetch
```

Pure-python k-means over trajectory embeddings, k chosen by silhouette —
your agent's failure modes, named and counted.

## What AgentGuard is not

- **Not an agent framework.** One job: watch a run, predict failure, propose an
  intervention. Bring your own agent.
- **Not tracing/observability.** Use LangSmith/Langfuse for the post-mortem;
  AgentGuard exists so there are fewer post-mortems.
- **Not autonomous.** Interventions are proposed and logged; your code decides.

## Development

```bash
python -m unittest discover -s tests   # core tests need nothing but Python
```

MIT licensed.
