"""Play the real game repeatedly and report the same numbers the simulator does.

The simulator says alpha reaches floor 9.7 and clears act 1 one time in a
hundred. Whether the live game agrees has never been measured, because a live
session ended at the first death: the mod starts runs back to back, but this side
exited on the first terminal message. Every live datapoint so far has been one
run, watched by hand, remembered informally.

So this keeps playing. Each finished run appends one JSON line to a log and
updates a running summary printed in the same shape as `eval_run_model.py`, so a
live number can be put beside a simulator number without arithmetic.

    python -m sts2_env.bridge.live_eval --model-path output/alpha/alpha_model.zip

Stop it with Ctrl-C; the summary prints on the way out and the log is already on
disk, flushed per run, so killing the game or the script keeps every run that
finished.

It deliberately shares `run_agent` rather than reimplementing the decision loop.
A second copy of that logic would drift from the first, and a bridge that
disagrees with itself about what an action means is the bug class that has cost
this project the most runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from sts2_env.bridge.agent_runner import run_agent

logger = logging.getLogger(__name__)

# Act 1's boss sits on floor 16, so clearing it means reaching 17.
ACT1_BOSS_FLOOR = 16
ACT1_CLEARED_FLOOR = ACT1_BOSS_FLOOR + 1
ACT3_START_FLOOR = 33


class LiveEvalRecorder:
    """Accumulates finished runs, writes them out, and reports the summary."""

    def __init__(self, log_path: Path | None, model_path: str):
        self.runs: list[dict[str, Any]] = []
        self.model_path = model_path
        self.started = time.monotonic()
        self._fh = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # Append: a crashed game or a restarted script should add to the
            # record, not silently replace yesterday's runs.
            self._fh = log_path.open("a", encoding="utf-8")

    def __call__(self, summary: dict[str, Any]) -> None:
        summary = dict(summary)
        summary["model"] = self.model_path
        summary["wall_clock"] = round(time.monotonic() - self.started, 1)
        self.runs.append(summary)

        if self._fh is not None:
            self._fh.write(json.dumps(summary, sort_keys=True) + "\n")
            # Flushed per run so a kill -9 keeps everything up to the last one.
            self._fh.flush()

        floor = summary.get("floor", 0)
        logger.info(
            "run %d: floor %s (%s), act %s, %s, hp %s, %ss  |  %s",
            summary.get("run", len(self.runs)), floor,
            summary.get("room_type", "?"), summary.get("act", "?"),
            summary.get("result", "?"), summary.get("run_hp", "?"),
            summary.get("seconds", "?"), self.one_line(),
        )

    def floors(self) -> list[int]:
        return [int(r.get("floor") or 0) for r in self.runs]

    def one_line(self) -> str:
        f = self.floors()
        if not f:
            return "no runs yet"
        cleared = sum(1 for x in f if x >= ACT1_CLEARED_FLOOR)
        return (f"{len(f)} runs, mean floor {statistics.mean(f):.1f}, "
                f"act 1 cleared {cleared}/{len(f)}")

    def report(self) -> str:
        f = self.floors()
        if not f:
            return "\nNo runs finished, so there is nothing to report.\n"

        n = len(f)
        reached = sum(1 for x in f if x >= ACT1_BOSS_FLOOR)
        cleared = sum(1 for x in f if x >= ACT1_CLEARED_FLOOR)
        act3 = sum(1 for x in f if x >= ACT3_START_FLOOR)
        results: dict[str, int] = {}
        for r in self.runs:
            key = str(r.get("result", "unknown"))
            results[key] = results.get(key, 0) + 1

        lines = [
            "",
            "=" * 58,
            f"model:    {self.model_path}",
            f"runs:     {n}  (live game)",
            f"outcomes: {results}",
            "",
            f"floors    mean {statistics.mean(f):.1f}   "
            f"median {statistics.median(f):.0f}   min {min(f)}   max {max(f)}",
            "",
            f"  reached the act 1 boss (f>={ACT1_BOSS_FLOOR})   "
            f"{reached:>3}/{n}  {reached / n:5.1%}",
            f"  CLEARED act 1          (f>={ACT1_CLEARED_FLOOR})   "
            f"{cleared:>3}/{n}  {cleared / n:5.1%}",
            f"  reached act 3          (f>={ACT3_START_FLOOR})   "
            f"{act3:>3}/{n}  {act3 / n:5.1%}",
            "",
        ]

        buckets = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 99)]
        lines.append("floor reached:")
        for lo, hi in buckets:
            c = sum(1 for x in f if lo <= x <= hi)
            if c:
                lines.append(f"  {lo:>3}-{hi:<3} {c:>4}  {'#' * min(40, c)}")

        # A live run is minutes, not milliseconds, so the cost of the next
        # datapoint is worth stating plainly.
        secs = [float(r.get("seconds") or 0) for r in self.runs]
        if any(secs):
            lines += ["", f"time      {statistics.mean(secs) / 60:.1f} min/run mean, "
                          f"{sum(secs) / 3600:.1f} h total"]

        # n is small live, so say how uncertain the headline number is rather
        # than letting a 3-run sample read like a measurement.
        if n >= 2:
            p = cleared / n
            se = (p * (1 - p) / n) ** 0.5
            lines += [f"act 1 clear rate {p:.1%} +/- {se:.1%} (1 se over {n} runs)"]
        lines += ["=" * 58, ""]
        return "\n".join(lines)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play STS2 live, repeatedly, and report act 1 clear rate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", required=True, help="Full-run model .zip")
    parser.add_argument("--runs", type=int, default=1000,
                        help="Stop after this many runs (Ctrl-C stops sooner)")
    parser.add_argument("--log", default="output/live_eval.jsonl",
                        help="JSONL file, appended to, one line per finished run")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9002)
    parser.add_argument("--speed", default="turbo",
                        choices=("turbo", "fast", "normal", "slow"))
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions instead of taking the argmax. Live "
                             "runs are few, and a deterministic policy replays "
                             "the same mistake on the same state every time.")
    parser.add_argument("--journal", default="output/live_journal.jsonl",
                        help="JSONL recording every room, fight, card played and "
                             "reward taken. The run log says a run reached floor "
                             "11; this says what happened on the way. Pass an "
                             "empty string to turn it off.")
    parser.add_argument("--tell-cyra", action="store_true",
                        help="Publish run milestones to cyra_brain over RabbitMQ. "
                             "Needs cyra_game reachable (CYRA_GAME_PATH) and a "
                             "broker; without either it logs once and plays on.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--combat-policy",
        default=None,
        help=(
            "Optional separate combat policy (.zip). When the main model is a full-run "
            "model, this overrides combat decisions so the main model only handles "
            "map, rewards, shop, rest, and events."
        ),
    )
    parser.add_argument(
        "--crash-log",
        default="output/crash_log.json",
        help="JSON file written when the game crashes or disconnects mid-run.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    recorder = LiveEvalRecorder(Path(args.log) if args.log else None, args.model_path)
    logger.info("Live eval: up to %d runs, logging to %s", args.runs, args.log)
    logger.info("Ctrl-C stops and prints the summary.")

    try:
        run_agent(
            model_path=args.model_path,
            host=args.host,
            port=args.port,
            deterministic=not args.stochastic,
            verbose=args.verbose,
            speed=args.speed,
            max_runs=args.runs,
            on_run_end=recorder,
            tell_cyra=args.tell_cyra,
            journal_path=args.journal or None,
            combat_policy_path=args.combat_policy,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    except Exception as exc:
        # Report what was measured before re-raising; a crash 40 runs in should
        # not throw away 40 runs of data.
        logger.exception("Live eval stopped by an error.")
        crash_info = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "model": args.model_path,
            "combat_policy": args.combat_policy,
            "runs_completed": len(recorder.runs),
            "last_run": recorder.runs[-1] if recorder.runs else None,
            "summary_one_line": recorder.one_line(),
        }
        try:
            Path(args.crash_log).parent.mkdir(parents=True, exist_ok=True)
            with open(args.crash_log, "w", encoding="utf-8") as fh:
                json.dump(crash_info, fh, indent=2, default=str)
            logger.info("Crash log written to %s", args.crash_log)
        except Exception:
            logger.exception("Could not write crash log.")
        print(recorder.report())
        recorder.close()
        sys.exit(1)

    print(recorder.report())
    recorder.close()


if __name__ == "__main__":
    main()
