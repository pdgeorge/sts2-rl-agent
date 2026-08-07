"""The four moments worth interrupting Cyra for, and nothing else.

A run is roughly 400 decisions. Sending them all would be noise, and in
cyra_brain a milestone *bypasses the reaction cooldown* -- it forces her to speak,
and the cooldown exists so she does not talk over herself. So the bar is high:

    run start        1 per run
    Ancient chosen   ~1 per act
    elite defeated   2-3 per act
    boss defeated    1 per act

Everything else is autopilot. She is not aware of the game between these, by
design: a character who plays on instinct and says so is honest, where one who
narrates every card play would be inventing reasoning the policy does not have.

The one real piece of introspection is how TORN the policy was -- the margin
between its top two choices. That is genuine, not confabulated. It is turned into
a phrase here and the number never leaves this module, because a softmax over
action logits is not a calibrated confidence: 0.75 does not mean "75% sure", and
an LLM handed 0.75 will say "I was 75% sure" and be wrong in a way that sounds
authoritative. Same rule STS2_DESIGN.md already sets for the value head.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Below this margin between the top two options, she was genuinely torn.
TORN_GAP = 0.15
# Above this, it was not close.
OBVIOUS_GAP = 0.50

ELITE_ROOMS = frozenset({"ELITE"})
BOSS_ROOMS = frozenset({"BOSS"})
# States that only appear once a fight is over and won. A loss goes to game_over
# instead, so seeing one of these after an elite room means she beat it.
POST_COMBAT_STATES = frozenset({"reward_screen", "card_reward", "card_bundle"})


def _canon(value: Any) -> str:
    return str(value or "").strip().upper().replace(" ", "_")


def gut_phrase(gap: float | None) -> str:
    """How she answers "why did you do that?" -- honestly, without numbers."""
    if gap is None:
        return "it felt right."
    if gap < TORN_GAP:
        return "honestly, could have gone either way."
    if gap >= OBVIOUS_GAP:
        return "that one was obvious to me."
    return "it felt like the better line."


class MilestoneWatcher:
    """Turns the bridge state stream into the handful of events worth sending.

    Stateful because the interesting things are transitions: "a reward screen
    appeared *and* we were in an elite room" is a win, where either alone is not.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._archetype: str | None = None
        """Called between runs. Without this, run 2 inherits run 1's room."""
        self._started = False
        self._room = ""
        self._act = 0
        self._pending_room = ""

    # -- helpers -------------------------------------------------------------

    def _observe(self, state: dict) -> None:
        room = _canon(state.get("room_type"))
        if room:
            # Remembered rather than read at reward time: the reward screen may
            # report a different room once the map advances.
            self._pending_room = self._room
            self._room = room
        act = state.get("act")
        if isinstance(act, int):
            self._act = act

    # -- the four events -----------------------------------------------------

    def run_start(self, state: dict) -> dict | None:
        if self._started:
            return None
        self._started = True
        character = state.get("character") or state.get("character_id")
        who = str(character).replace("_", " ").title() if character else "the Ironclad"
        ascension = state.get("ascension") or 0
        tail = f" on ascension {ascension}" if ascension else ""
        return _milestone(f"starting a new run as {who}{tail}.", "sts2_run_start")

    def map_choice(self, state: dict, chosen_index: int, gap: float | None) -> dict | None:
        """An Ancient on the map is the one routing choice worth mentioning."""
        nodes = state.get("nodes") or []
        if not (0 <= chosen_index < len(nodes)):
            return None
        if "ANCIENT" not in _canon(nodes[chosen_index].get("type")):
            return None
        return _milestone(
            f"heading for the Ancient. {gut_phrase(gap)}", "sts2_ancient")

    def archetype_chosen(self, archetype: str, confidence: float) -> dict | None:
        """Fires once a run, when the deck's direction stops being ambiguous.

        Worth having beyond the plumbing: it is the only moment in a run where
        she states a *plan*. Every other milestone reports something that
        happened to her -- a run started, an elite died, a boss fell. This is
        her deciding what she is trying to do, which is the first thing a
        viewer can hold her to.

        `confidence` is the per-card margin, not the raw one -- see
        DeckDirection.confidence. Passing the raw margin would have her sound
        certain about every deck simply because it had more cards in it.
        """
        if self._archetype is not None:
            return None
        self._archetype = archetype
        word = _ARCHETYPE_WORDS.get(archetype, archetype.replace("-", " "))
        return _milestone(
            f"building a {word} deck. {gut_phrase(confidence)}", "sts2_archetype")

    def combat_result(self, state: dict) -> dict | None:
        """A reward screen right after an elite or boss room means she won it."""
        state_type = str(state.get("type", ""))
        if state_type not in POST_COMBAT_STATES:
            return None

        room = self._room if self._room in (ELITE_ROOMS | BOSS_ROOMS) else self._pending_room
        if room not in (ELITE_ROOMS | BOSS_ROOMS):
            return None
        # Only fire once per fight; the reward and card_reward screens both arrive.
        self._pending_room = ""
        self._room = ""

        hp, max_hp = state.get("run_hp"), state.get("run_max_hp")
        how = ""
        if isinstance(hp, int) and isinstance(max_hp, int) and max_hp > 0:
            ratio = hp / max_hp
            if ratio <= 0.25:
                how = f" Barely -- {hp} health left."
            elif ratio >= 0.8:
                how = " Barely scratched."
            else:
                how = f" {hp} health left."

        if room in BOSS_ROOMS:
            return _milestone(f"beat the act {self._act} boss.{how}", "sts2_boss")
        return _milestone(f"took down an elite.{how}", "sts2_elite")

    def observe(self, state: dict) -> list[dict]:
        """Feed every state through here; returns the events worth sending."""
        out: list[dict] = []
        self._observe(state)
        start = self.run_start(state)
        if start is not None:
            out.append(start)
        result = self.combat_result(state)
        if result is not None:
            out.append(result)
        return out


#: Archetype slugs are internal; she should not say "block-scaling".
_ARCHETYPE_WORDS = {
    "strike-synergy": "strike",
    "block-scaling": "block",
    "strength": "strength",
    "exhaust": "exhaust",
    "bloodletting": "bloodletting",
}


def _milestone(text: str, kind: str) -> dict:
    """The payload shape cyra_brain reads.

    Only `text` and `tier` actually reach her -- cyra_brain's consumer discards
    the rest -- but the extra fields cost nothing and make the broker log
    readable when working out why she said something.
    """
    return {"text": text, "tier": "milestone", "kind": kind, "game": "sts2"}
