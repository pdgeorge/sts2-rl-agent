"""Record the raw JSON the mod actually sends, so the parsers can be checked against it.

Phase 0.1 of `docs/GLM_ROADMAP_50P_ACT1.md` asked for exactly this and was
never done, and the cost of skipping it is on the record: every parser on the
bridge path -- ``from_bridge_state``, ``to_combat_mid_fight``,
``coerce_potion_id`` -- was written against a *guessed* payload shape, and two
of the three needed a live session and a bug-fix round to correct.

- PR #8: potion ids arrive UPPER_SNAKE (``STRENGTH_POTION``), not PascalCase.
  ``create_potion`` raised, the runner fell back to END_TURN every combat step,
  and the agent died on the first encounter of every run.
- PR #9: the kept local sim drifted from the live game and the search planned
  against a frozen fiction for eleven turns.

Neither was reachable from the unit tests, because the unit tests were built
from the same guess as the code. A recorded payload is the thing that breaks
that circle: it is the mod's answer rather than ours.

The journal deliberately records *decisions* -- what was offered, what was
chosen -- and drops the state behind them, which is right for run analysis and
useless for protocol work. This is the other half: whole states, verbatim,
unparsed, so that a fixture can be replayed through the parsers offline with
no game running.

Quotas rather than everything
-----------------------------
A 20-run session is tens of thousands of states, almost all of them
``combat_action`` from whichever fight ran longest. That is a large file that
answers one question. The capture instead keeps the first ``per_type`` states
of each message type, so one short session yields a sample containing every
*kind* of screen the mod emits -- which is what pins the spec. The totals of
what was seen and what was kept are written to the trailer, so a type that got
truncated says so rather than looking rare.

Usage::

    live_eval --capture-raw output/bridge_protocol_sample.jsonl --runs 1

Then, offline and with no game running::

    from sts2_env.bridge.raw_capture import load_capture
    states = load_capture("output/bridge_protocol_sample.jsonl")
    combat = [s for s in states if s.get("type") == "combat_action"]
    CombatSituation.from_bridge_state(combat[0]).to_combat_mid_fight(combat[0])
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_PER_TYPE = 25

#: Marks the trailer record. A plain state can never collide with this because
#: the mod's states are keyed by "type", not by this name.
TRAILER_KEY = "__capture_trailer__"


def _state_type(state: dict[str, Any]) -> str:
    """The message type, falling back to a stable placeholder.

    A state with no ``type`` is itself worth capturing -- an untyped screen is
    a protocol question -- so it gets a bucket rather than being dropped.
    """
    value = state.get("type")
    if value is None or value == "":
        return "<untyped>"
    return str(value)


class RawCapture:
    """Append raw bridge states to a JSONL file, quota'd per message type.

    Not a context manager on purpose: the runner's state loop can exit through
    a reconnect, a ``KeyboardInterrupt`` or a crash, and a capture that only
    lands its trailer on the clean path would be missing precisely when it is
    most wanted. Each state is written and flushed as it arrives, so the file
    is complete-up-to-the-crash at all times; ``close`` only adds the trailer.
    """

    def __init__(self, path: str | Path, *, per_type: int = DEFAULT_PER_TYPE):
        self.path = Path(path)
        self.per_type = per_type
        self.seen: Counter[str] = Counter()
        self.kept: Counter[str] = Counter()
        self._closed = False

        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate: a capture is a snapshot of one session's protocol, and
        # appending across sessions would silently mix two mod builds -- the
        # exact confusion this file exists to prevent.
        self._fh = self.path.open("w", encoding="utf-8")

    def observe(self, state: dict[str, Any]) -> bool:
        """Record ``state`` if its type is under quota. Returns whether it was kept.

        Never raises: a capture is a diagnostic, and a diagnostic that can kill
        a 20-run live session is worse than no diagnostic. A value JSON cannot
        represent is stringified (``default=str``) rather than costing the whole
        state -- real payloads arrive as JSON so this only fires on something
        impossible, and when it does the screen is still what you wanted to see.
        A state that defeats even that (a circular reference) is counted as seen,
        logged, and skipped.
        """
        if self._closed or not isinstance(state, dict):
            return False

        msg_type = _state_type(state)
        self.seen[msg_type] += 1
        if self.kept[msg_type] >= self.per_type:
            return False

        try:
            line = json.dumps(state, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            logger.warning(
                "raw capture: a %r state would not serialise; skipping it. "
                "The capture stays valid, this one state is absent.",
                msg_type,
            )
            return False

        self._fh.write(line + "\n")
        self._fh.flush()
        self.kept[msg_type] += 1

        if self.kept[msg_type] == self.per_type:
            logger.info(
                "raw capture: %r reached its quota of %d; further %r states "
                "are counted but not written.",
                msg_type, self.per_type, msg_type,
            )
        return True

    def close(self) -> None:
        """Write the trailer and close. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        trailer = {
            TRAILER_KEY: True,
            "per_type_quota": self.per_type,
            "seen": dict(self.seen),
            "kept": dict(self.kept),
        }
        try:
            self._fh.write(json.dumps(trailer) + "\n")
            self._fh.flush()
        finally:
            self._fh.close()

        logger.info(
            "raw capture: wrote %d states of %d types to %s (saw %d).",
            sum(self.kept.values()), len(self.kept), self.path, sum(self.seen.values()),
        )

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "types": len(self.kept),
            "kept": sum(self.kept.values()),
            "seen": sum(self.seen.values()),
            "per_type": dict(self.kept),
        }


def load_capture(path: str | Path) -> list[dict[str, Any]]:
    """Read a capture back as states, dropping the trailer.

    Tolerates a truncated final line: a session killed mid-write leaves one,
    and the states before it are still the point.
    """
    states: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "raw capture %s: line %d is not valid JSON (a truncated "
                    "final line is expected if the session was killed); "
                    "stopping here with %d states.", path, line_no, len(states),
                )
                break
            if isinstance(record, dict) and record.get(TRAILER_KEY):
                continue
            if isinstance(record, dict):
                states.append(record)
    return states


def load_trailer(path: str | Path) -> dict[str, Any] | None:
    """The capture's trailer, or None if the session never closed cleanly."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and record.get(TRAILER_KEY):
                return record
    return None
