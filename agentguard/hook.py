"""Claude Code hook — guard a live coding-agent session.

Claude Code invokes hooks with a JSON event on stdin. This module turns
those events into Steps, replays the session through the deterministic
predictors (cheap — microseconds even at hundreds of steps), and answers
with the hook protocol when risk crosses the threshold: the same "propose,
human decides" discipline, expressed as a permission "ask".

Install (in your project's .claude/settings.json):

    {
      "hooks": {
        "PreToolUse": [{
          "matcher": "*",
          "hooks": [{"type": "command", "command": "agentguard hook"}]
        }],
        "PostToolUse": [{
          "matcher": "*",
          "hooks": [{"type": "command", "command": "agentguard hook"}]
        }]
      }
    }

Now a session that starts looping on the same failing command gets held
for confirmation instead of burning the afternoon:

    AgentGuard: risk 0.86 (loop=1.00) — this tool call repeats a cycle
    seen 4 times. Continue?
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import List, Optional

from .guard import Guard

STATE_DIR = os.path.join(tempfile.gettempdir(), "agentguard-hook")


def _state_path(session_id: str) -> str:
    os.makedirs(STATE_DIR, exist_ok=True)
    safe = "".join(c for c in session_id if c.isalnum() or c in "-_")[:64]
    return os.path.join(STATE_DIR, f"{safe or 'default'}.jsonl")


def event_to_step(event: dict) -> Optional[dict]:
    hook = event.get("hook_event_name", "")
    tool = event.get("tool_name", "tool")
    if hook == "PreToolUse":
        return {"kind": "tool_call", "name": tool,
                "content": event.get("tool_input")}
    if hook == "PostToolUse":
        response = event.get("tool_response")
        text = json.dumps(response, default=str)[:2000] if response is not None else ""
        error = '"is_error": true' in text.lower() or '"success": false' in text.lower()
        return {"kind": "tool_result", "name": tool, "content": text, "error": error}
    return None


def session_risk(path: str, predictors: List[str], threshold: float):
    guard = Guard(predictors=predictors, threshold=threshold)
    w = guard.watch()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        w.record(json.loads(line))
                    except (ValueError, KeyError):
                        continue
    return w


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="agentguard hook")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--predictors", default="loop,tool_cascade")
    ap.add_argument("--action", choices=["ask", "deny"], default="ask",
                    help="what to do at threshold on PreToolUse (default: ask the human)")
    args = ap.parse_args(argv)

    try:
        event = json.load(sys.stdin)
    except ValueError:
        return 0  # not a hook invocation; never break the host session

    step = event_to_step(event)
    if step is None:
        return 0

    path = _state_path(str(event.get("session_id", "default")))
    with open(path, "a") as f:
        f.write(json.dumps(step, default=str) + "\n")

    w = session_risk(path, args.predictors.split(","), args.threshold)
    if w.risk < args.threshold:
        return 0

    signals = ", ".join(f"{k}={v:.2f}" for k, v in w.subscores.items() if v > 0.05)
    reason = (f"AgentGuard: session risk {w.risk:.2f} >= {args.threshold} "
              f"({signals}) — the last {len(w.history)} tool events show a "
              f"repeating/failing pattern. Consider a different approach.")

    if event.get("hook_event_name") == "PreToolUse":
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": args.action,
                "permissionDecisionReason": reason,
            }
        }))
    else:  # PostToolUse: warn, don't block
        print(json.dumps({"systemMessage": reason}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
