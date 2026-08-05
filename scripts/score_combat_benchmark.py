"""Score an agent on the act 1 combat benchmark, or compare two.

    python scripts/score_combat_benchmark.py --model output/combat_v3_overnight/final_model.zip
    python scripts/score_combat_benchmark.py --random
    python scripts/score_combat_benchmark.py --model A.zip --against B.zip

The comparison form is the Phase 1 gate: it prints the difference between two
agents together with what 200 fights can actually resolve, so a two-point win
rate difference is reported as the noise it is rather than as progress.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from sts2_env.search.benchmark import ModelAgent, RandomAgent, compare, score
from sts2_env.search.situation import load_situations

DEFAULT_BENCHMARK = "tests/fixtures/act1_combat_benchmark.json"


def _agent(spec: str | None, *, stochastic: bool, seed: int):
    if spec is None or spec == "random":
        return RandomAgent(seed=seed)
    return ModelAgent(spec, deterministic=not stochastic)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a combat agent on real act 1 situations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=None, help="Combat model .zip to score")
    parser.add_argument("--random", action="store_true",
                        help="Score the random baseline instead")
    parser.add_argument("--against", "--baseline", dest="against", default=None,
                        help="Baseline to compare against ('random' or a .zip). "
                             "--model is the challenger; the comparison reads "
                             "baseline -> challenger.")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--limit", type=int, default=0,
                        help="Score only the first N situations (0 = all)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions rather than taking the argmax")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--json-out", default=None,
                        help="Write the summary as JSON as well")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.model is None and not args.random:
        parser.error("give --model, or --random for the baseline")

    situations = load_situations(args.benchmark)
    if args.limit:
        situations = situations[:args.limit]
    print(f"{len(situations)} situations from {args.benchmark}")

    first = _agent(None if args.random else args.model,
                   stochastic=args.stochastic, seed=args.seed)
    summary = score(situations, first, max_turns=args.max_turns)
    print(summary.report())

    if args.against:
        baseline_agent = _agent(args.against, stochastic=args.stochastic, seed=args.seed + 1)
        baseline = score(situations, baseline_agent, max_turns=args.max_turns)
        print(baseline.report())
        # baseline first: `compare` reads "a -> b", and the thing being judged is
        # the challenger, so it has to be b or the verdict describes the wrong one.
        print(compare(baseline, summary))

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(summary.to_dict(), indent=1),
                                       encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
