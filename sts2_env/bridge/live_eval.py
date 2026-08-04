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

# Clearing act 1 means being in act 2, and nothing else.
#
# This used to be `floor >= 17`, on the belief that the act 1 boss sat on floor 16.
# It does not -- floor 17 IS the boss room. Every run that reached it was counted
# as a clear, including one that died there at 0 HP and one that hung on the boss.
# A live batch reported "act 1 cleared 3/8" when the true figure was 0/8, and the
# same conflation reached docs/MODELS.md, where alpha's "roughly a quarter of runs
# now clear act 1" rests on a 16-20 floor bin that mostly means "reached the boss".
#
# The act number is recorded already, cannot be off by one, and says the thing we
# actually mean. Floor thresholds are kept only for the reach-the-boss statistic,
# which is a different and honestly-named quantity.
ACT1_BOSS_FLOOR = 17
ACT1_REACHED_BOSS_FLOOR = ACT1_BOSS_FLOOR
ACT3_START_FLOOR = 33


def deck_shape(deck: Any) -> dict[str, Any]:
    """The deck's SHAPE, recorded beside the run that played it.

    Three of the five numbers from Baalorlord's core-deckbuilding article, which
    describe what KIND of deck this is rather than which cards are in it. Stored
    per run because a decklist alone is not enough: it has to be rebuilt into
    card objects to mean anything, several past sessions recorded no decklist at
    all, and a run whose cards a future patch renames becomes unreadable while
    its shape stays perfectly comparable.

    Frontload and scaling damage -- the other two -- are deliberately absent.
    Splitting damage that way needs to know what a card does OVER TIME, which is
    the judgement the pilot cannot currently make while it scores 29 of 86 cards
    at zero. Faking it with a keyword list is how deck_features.py ended up with
    20 of 45 strings matching no card in this game.

    Never raises. A recorder that loses a finished run because a metric threw is
    worse than a recorder with a missing field.
    """
    if not deck:
        return {}
    try:
        from sts2_env.cards.factory import create_card
        from sts2_env.core.enums import CardId
        from sts2_env.evaluation.deck_metrics import (
            block_density,
            cycle_time,
            meaningful_upgrades,
            upgrade_density,
        )

        cards = []
        for entry in deck:
            name = entry.get("id") or entry.get("name") if isinstance(entry, dict) else entry
            if not isinstance(name, str):
                continue
            upgraded = name.endswith("+")
            if isinstance(entry, dict):
                upgraded = bool(entry.get("upgraded") or entry.get("is_upgraded"))
            # The bridge sends the game's id, which for some cards lacks the
            # `_CARD` suffix our enum carries -- FLAME_BARRIER against
            # FLAME_BARRIER_CARD. Without the fallback those cards silently drop
            # out of the shape, and a block card dropping out biases the one
            # number most likely to be acted on.
            base = name.rstrip("+").upper()
            for candidate in (base, f"{base}_CARD"):
                member = getattr(CardId, candidate, None)
                if member is None:
                    continue
                try:
                    cards.append(create_card(member, upgraded=upgraded))
                except Exception:  # noqa: BLE001 -- one bad card is not a lost run
                    pass
                break
        if not cards:
            return {}
        return {
            "block_density": round(block_density(cards), 3),
            "upgrade_density": round(upgrade_density(cards), 3),
            # The number that predicts the act 1 boss. Measured, 40 seeds:
            #   0 -> 7%   1 -> 17%   2 -> 30%   3 -> 66%   4 -> 79%
            # Live runs have been reaching the boss with two. Recorded on its own
            # because upgrade_density cannot express it -- an upgraded Strike
            # raises the density and is worth almost nothing.
            "meaningful_upgrades": meaningful_upgrades(cards),
            "cycle_time": round(cycle_time(cards), 2),
            "shape_cards_resolved": len(cards),
        }
    except Exception:  # noqa: BLE001
        logger.debug("deck shape unavailable", exc_info=True)
        return {}


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
        summary.update(deck_shape(summary.get("deck")))
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

    def _act1_cleared(self) -> int:
        """Runs that actually left act 1. Reaching the boss room is not clearing it."""
        return sum(1 for r in self.runs if int(r.get("act") or 1) >= 2)

    def one_line(self) -> str:
        f = self.floors()
        if not f:
            return "no runs yet"
        cleared = self._act1_cleared()
        reached = sum(1 for x in f if x >= ACT1_REACHED_BOSS_FLOOR)
        return (f"{len(f)} runs, mean floor {statistics.mean(f):.1f}, "
                f"reached boss {reached}/{len(f)}, act 1 cleared {cleared}/{len(f)}")

    def report(self) -> str:
        f = self.floors()
        if not f:
            return "\nNo runs finished, so there is nothing to report.\n"

        n = len(f)
        reached = sum(1 for x in f if x >= ACT1_BOSS_FLOOR)
        cleared = self._act1_cleared()
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
            f"  CLEARED act 1          (act>=2)  "
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
    parser.add_argument(
        "--measured-drafting", action="store_true",
        help="Choose card rewards by playing the deck with each candidate instead "
             "of asking the policy. Costs ~4s per reward and needs no training; "
             "the policy picked slot 2 in 19 of 28 live rewards and slot 0 in none.",
    )
    parser.add_argument(
        "--pilot-combat", action="store_true",
        help="Play fights with the evaluation pilot instead of the trained "
             "model. See agent_runner for the measurement behind it.",
    )
    parser.add_argument(
        "--record-replay", default=None,
        help="Save the real bridge states to this path while running. The states "
             "the game actually sends, for replaying decisions against instead "
             "of hand-written ones -- which is how a live HP field read as zero "
             "for a week while 4,700 synthetic-state tests passed.",
    )
    parser.add_argument(
        "--allow-layout-mismatch", action="store_true",
        help="Run a model trained against a different observation layout. It will "
             "read at least one column as something other than what it learned, so "
             "expect degraded play; use it to keep testing, not to measure.",
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
            combat_policy_path=args.combat_policy,
            measured_drafting=args.measured_drafting,
            pilot_combat=args.pilot_combat,
            record_replay_path=args.record_replay,
            allow_layout_mismatch=args.allow_layout_mismatch,
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
