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

# In the live game the act 1 boss room IS floor 17 -- not 16, which is where the
# simulator puts it. Counting "reached floor 17" as a clear therefore counted
# every boss DEATH as a win, and on 2026-08-05 it reported 20% cleared from 30
# runs in which the boss was never once beaten: all six were floor 17, room Boss,
# 0 HP, act 1.
#
# So a clear is not a floor at all. It is reaching act 2, which the run summary
# states outright and which no death can fake.
ACT1_BOSS_FLOOR = 17
ACT3_START_FLOOR = 33


def _reached_act_1_boss(run: dict[str, Any]) -> bool:
    if str(run.get("room_type", "")).upper() == "BOSS":
        return True
    return int(run.get("floor") or 0) >= ACT1_BOSS_FLOOR


def _cleared_act_1(run: dict[str, Any]) -> bool:
    """Past the act 1 boss, rather than merely standing in front of it."""
    act = run.get("act")
    if isinstance(act, int):
        return act >= 2
    # No act reported: fall back to being strictly beyond the boss floor, which
    # a death on the boss cannot satisfy.
    return int(run.get("floor") or 0) > ACT1_BOSS_FLOOR


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
        cleared = sum(1 for r in self.runs if _cleared_act_1(r))
        return (f"{len(f)} runs, mean floor {statistics.mean(f):.1f}, "
                f"act 1 cleared {cleared}/{len(f)}")

    def report(self) -> str:
        f = self.floors()
        if not f:
            return "\nNo runs finished, so there is nothing to report.\n"

        n = len(f)
        reached = sum(1 for r in self.runs if _reached_act_1_boss(r))
        cleared = sum(1 for r in self.runs if _cleared_act_1(r))
        died_on_boss = reached - cleared
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
            f"  reached the act 1 boss      "
            f"{reached:>3}/{n}  {reached / n:5.1%}",
            f"  ... and died to it          "
            f"{died_on_boss:>3}/{n}  {died_on_boss / n:5.1%}",
            f"  CLEARED act 1 (reached act 2)"
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

        # Every place the live game disagreed with the simulator. Printed with
        # the results rather than left in the log, because the search plans its
        # whole lookahead on the simulator's numbers and a session that found
        # three of these has three fights it was planning wrongly.
        from sts2_env.search.parity import disparity_summary

        found = disparity_summary()
        if found:
            lines += ["", f"simulator disparities ({len(found)} distinct):"]
            lines += [f"  {line}" for line in found]

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
                        choices=("turbo", "fast", "normal", "slow"),
                        help=(
                            "Game pacing; turbo is the default and the floor worth "
                            "having. Faster presets were measured and dropped: at "
                            "turbo the median gap between actions is 0.23s while the "
                            "search alone costs 0.08-0.52s, so animation is already "
                            "nearly free and the rest is not a multiplier's to give. "
                            "Animation speed was suspected of the Punch Off crash and "
                            "CLEARED. Note `normal` is NOT a way to disable that "
                            "patch: it sets the multiplier to 1.0 while the prefix "
                            "stays installed."
                        ))
    parser.add_argument("--stochastic", action="store_true",
                        help="Sample actions instead of taking the argmax. Live "
                             "runs are few, and a deterministic policy replays "
                             "the same mistake on the same state every time.")
    parser.add_argument("--journal", default="output/live_journal.jsonl",
                        help="JSONL recording every room, fight, card played and "
                             "reward taken. The run log says a run reached floor "
                             "11; this says what happened on the way. Pass an "
                             "empty string to turn it off.")
    parser.add_argument("--capture-raw", default=None,
                        help="JSONL of the raw states the mod sends, verbatim. The "
                             "journal records decisions and drops the state behind "
                             "them; this keeps whole states so the bridge parsers "
                             "can be replayed offline against real payloads. One "
                             "short --runs 1 session is enough to pin the protocol.")
    parser.add_argument("--capture-raw-per-type", type=int, default=25,
                        help="States kept per message type (default 25).")
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
        "--live-search",
        action="store_true",
        help=(
            "Use the SearchAgent turn planner for combat decisions instead of "
            "the trained model's argmax. Lifts boss win rate from 6.7%% to ~20%% "
            "on the harvested benchmark (docs/MODELS.md:120). Requires the "
            "Phase 1.1 mod patch from PR #6 to send encounter/seed fields; "
            "without it, the runner logs and falls back to END_TURN every "
            "combat step."
        ),
    )
    parser.add_argument(
        "--search-rollout-model",
        action="store_true",
        help=(
            "Roll out inside the search with the trained model rather than the "
            "block-then-attack heuristic. MODELS.md records four turns of "
            "lookahead scoring WORSE than two because the heuristic playout "
            "compounds its own errors and ranks Powers last, and names this as "
            "the next thing to try. Nearly free: the search spends about 3%% of "
            "its time budget."
        ),
    )
    parser.add_argument(
        "--console-log",
        default="output/live_console.log",
        help=(
            "Also write this side's log here, tracebacks included. Empty string "
            "disables. On by default because console-only output is how the "
            "LiveSearch failure tracebacks were lost."
        ),
    )
    parser.add_argument(
        "--seed",
        default=None,
        help=(
            "Force EVERY run onto this game seed, e.g. VHHTGKTPEZWF. Sent over "
            "the bridge, so it needs no game restart. Use it to reproduce one "
            "run (VHHTGKTPEZWF is the Punch Off crash) or to pair two arms of "
            "an A/B on identical maps -- unpaired live runs carry about +/-5.6%% "
            "at n=40, wider than most changes worth measuring."
        ),
    )
    parser.add_argument(
        "--crash-log",
        default="output/crash_log.json",
        help="JSON file written when the game crashes or disconnects mid-run.",
    )
    args = parser.parse_args()

    # CONSOLE *AND* FILE. Everything this side prints used to exist only in the
    # terminal, so a session's most useful output died with the scrollback. The
    # LiveSearch tracebacks were lost that way for weeks -- the message said the
    # simulator could not rebuild a state and never said which state, because
    # the traceback that would have said so was console-only.
    #
    # The game's own log rotates itself and the journal holds events, but
    # neither holds this side's exceptions. Default it on rather than rely on
    # remembering to pipe through tee.
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.console_log:
        Path(args.console_log).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.console_log, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    if args.console_log:
        logger.info("Console log also being written to %s", args.console_log)

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
            live_search=args.live_search,
            search_rollout_model=args.search_rollout_model,
            capture_raw_path=args.capture_raw or None,
            capture_raw_per_type=args.capture_raw_per_type,
            force_seed=args.seed,
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
