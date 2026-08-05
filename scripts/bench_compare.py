"""Compare two saved benchmark arms, paired fight by fight.

    python scripts/bench_compare.py output/bench_v2.json output/bench_v3.json

Reads what `bench_arm.py` wrote and reports the difference with the uncertainty
attached, because the question is never "is the number bigger" -- 200 fights
carries a standard error around 3 points -- but "is it bigger by more than the
measurement is worth".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sts2_env.search.benchmark import Result, Summary, compare


def load(path: str) -> Summary:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Summary(data["label"], [Result(**fight) for fight in data["fights"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("challenger")
    args = parser.parse_args()

    baseline = load(args.baseline)
    challenger = load(args.challenger)
    print(baseline.report())
    print(challenger.report())
    print(compare(baseline, challenger))


if __name__ == "__main__":
    main()
