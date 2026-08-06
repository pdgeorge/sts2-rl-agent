"""Score one search configuration on the benchmark, keeping every fight.

Written because the comparison that matters is paired: both agents face the same
200 fights, and charging the fixture's own variance against the difference
between them hides real effects. Pairing needs each fight's result, not a
summary, so this writes them out.

    python scripts/bench_arm.py --label v2 --top-k 0   --out output/bench_v2.json
    python scripts/bench_arm.py --label v3 --top-k 5   --out output/bench_v3.json

Then:

    python scripts/bench_compare.py output/bench_v2.json output/bench_v3.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from sts2_env.search.benchmark import Summary, play
from sts2_env.search.situation import load_situations
from sts2_env.search.turn_search import SearchAgent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--benchmark", default="tests/fixtures/act1_combat_benchmark.json")
    parser.add_argument("--time-budget", type=float, default=3.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rollout-samples", type=int, default=3)
    parser.add_argument("--lookahead-turns", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    situations = load_situations(args.benchmark)
    if args.limit:
        situations = situations[:args.limit]

    agent = SearchAgent(
        time_budget=args.time_budget,
        top_k=args.top_k,
        rollout_samples=args.rollout_samples,
        lookahead_turns=args.lookahead_turns,
        name=args.label,
    )

    results = []
    started = time.perf_counter()
    for index, situation in enumerate(situations, 1):
        results.append(play(situation, agent))
        if index % 25 == 0:
            won = sum(r.won for r in results)
            print(f"  [{args.label}] {index}/{len(situations)} fights, "
                  f"win {won / len(results):.1%}, "
                  f"{time.perf_counter() - started:.0f}s", flush=True)

    summary = Summary(args.label, results)
    print(summary.report(), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "label": args.label,
        "config": vars(args),
        "summary": summary.to_dict(),
        "fights": [asdict(r) for r in results],
    }, indent=1), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
