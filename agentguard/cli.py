"""Console entry point: ``agentguard <command>``.

Commands:
  replay   backtest AgentGuard against historical traces (or --demo)
  train    train the learned risk model (needs agentguard[model])
"""

from __future__ import annotations

import sys
from typing import List, Optional

USAGE = """usage: agentguard <command> [args]

commands:
  replay   backtest against historical traces:  agentguard replay traces.jsonl
           or on synthesized runs:              agentguard replay --demo
  train    train the learned risk model:        agentguard train [--runs-per-type N]
"""


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0

    command, rest = argv[0], argv[1:]
    if command == "replay":
        from .replay import main as replay_main

        return replay_main(rest)
    if command == "train":
        try:
            from .train import train
        except ImportError as exc:
            print(f"train needs the model extra: pip install 'agentguard[model]' ({exc})")
            return 1
        import argparse

        ap = argparse.ArgumentParser(prog="agentguard train")
        ap.add_argument("--runs-per-type", type=int, default=60)
        ap.add_argument("--seed", type=int, default=42)
        ap.add_argument("--out", default=None)
        args = ap.parse_args(rest)
        kwargs = {"n_per_type": args.runs_per_type, "seed": args.seed}
        if args.out:
            kwargs["out_path"] = args.out
        train(**kwargs)
        return 0

    print(f"unknown command {command!r}\n\n{USAGE}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
