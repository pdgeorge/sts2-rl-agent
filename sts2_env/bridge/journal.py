"""A full record of what the agent actually did, one event per line.

`live_eval` writes one summary line per run: the floor it died on, the deck it
had at the end, how long it took. That is enough to say a run reached floor 11
and nothing at all about why it stopped there -- which card it took on floor 3,
whether it entered the elite at 30 HP or 70, whether it played its block or held
it. Every decision about what to fix next has been made without that.

So this records the run as it happens: rooms entered, fights started and how they
ended, every card played, every reward taken or skipped, every rest and shop and
event choice, and the HP either side of each fight.

TWO PROPERTIES IT IS BUILT FOR

It cannot miss an action. Rather than a call at each of the fourteen places the
runner sends something, the client is wrapped once and every outbound action is
recorded on the way through. A fifteenth decision site added later is journalled
without anyone remembering to journal it.

It cannot cost a run. Journalling is a file append inside a loop the game is
blocked on, so every write is wrapped: a full disk or a bad path costs the record
and never the run being measured. Same rule the Cyra seam already follows.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Which key holds the options a `choose` is picking from, per state type. Lets a
# recorded choice say "took Pommel Strike over Anger and Cleave" rather than
# "chose index 1", which is the difference between a log you can learn from and
# one you cannot.
_OPTION_KEYS = {
    "map_select": "nodes",
    "card_reward": "cards",
    "card_bundle": "bundles",
    "reward_screen": "options",
    "boss_relic": "relics",
    "shop": "options",
    "rest_site": "options",
    "event": "options",
}

# Tried in turn when the mapped key is absent. The mod has more than one name for
# "the things you may pick" and a journal that guesses one and gives up records a
# hundred decisions as `None`, which is worse than not recording them: it looks
# like data.
_OPTION_FALLBACKS = ("options", "cards", "nodes", "items", "relics", "bundles")


def _options_for(state: dict[str, Any]) -> list:
    state_type = str(state.get("type", ""))
    keys = (_OPTION_KEYS.get(state_type), *_OPTION_FALLBACKS)
    for key in keys:
        if key and isinstance(state.get(key), list) and state[key]:
            return state[key]
    return []

_COMBAT_STATES = {"combat_action", "combat"}

#: States that appear DURING a fight as an overlay and must not end it.
#:
#: A card-select or potion screen mid-combat is not the fight finishing, but the
#: naive "any non-combat state ends combat" rule counted it as one: the journal
#: wrote combat_end with 0 turns and 0 damage, then combat_start again when the
#: next combat_action arrived. 156 of 511 recorded act 1 boss "fights" were this
#: artefact -- 31% -- and it silently corrupted every per-fight statistic
#: computed from the journal, including a mean-turns figure quoted in analysis
#: on 2026-08-14 before the cause was found.
#:
#: Reach and clear rate are NOT affected: those are derived from combat_start
#: and run_end, and a duplicated combat_start does not change a boolean.
#: Only genuine mid-fight PROMPTS belong here. `card_reward`, `reward_screen`
#: and `card_bundle` legitimately follow a won fight and must still end it --
#: including them broke `test_leaving_a_fight_records_what_it_cost`, which is
#: the test doing its job.
_COMBAT_OVERLAY_STATES = {
    "card_select",
    "crystal_sphere",
}


def _describe(option: Any) -> Any:
    """A readable summary of one offered option.

    The bridge uses `id` for two different things, and taking it at face value
    makes the log useless exactly where it matters most. On a rest site or a card
    reward it names the thing (`HEAL`, `POMMEL_STRIKE`). In a shop or an event it
    is only the verb -- every purchase reads `buy_card` and every event choice
    reads `event_choice`, so a hundred logged decisions say nothing about what was
    bought or chosen. When `id` merely repeats `action`, the name is in `label`.
    """
    if not isinstance(option, dict):
        return option

    identifier = option.get("id")
    if identifier is not None and identifier != option.get("action"):
        if option.get("upgraded"):
            return f"{identifier}+"
        return identifier

    for key in ("card_id", "relic_id", "potion_id", "label", "name", "type", "option_id"):
        value = option.get(key)
        if value:
            return value
    return identifier if identifier is not None else option


def _deck_cards(state: dict[str, Any]) -> list[str] | None:
    """The deck as a compact list of card labels, `BASH+` meaning upgraded.

    THE BRIDGE HAS ALWAYS SENT THIS. `RlRunInfo.cs` attaches every card as
    `{id, upgraded}` and the journal recorded only `deck_size`, which is why
    500 pooled runs cannot answer the most important deck-building question
    there is -- which cards were in the decks that lost the boss. A count
    cannot be argued with; a list can.

    Labels rather than the raw dicts because this is written at every floor and
    at every elite and boss, and `["STRIKE_IRONCLAD", "BASH+"]` costs a third of
    what the dict form does while losing nothing anyone has asked for.
    """
    deck = state.get("deck")
    if not isinstance(deck, list) or not deck:
        return None
    labels: list[str] = []
    for card in deck:
        if isinstance(card, dict):
            name = str(card.get("id") or card.get("card_id") or "?")
            labels.append(name + "+" if card.get("upgraded") else name)
        elif card is not None:
            labels.append(str(card))
    return labels or None



class RunJournal:
    """Watches the state stream and the agent's actions, and writes both down."""

    def __init__(self, path: str | Path | None, *, model: str = "",
                 policy_version: str = "", git_sha: str = "",
                 on_event: "Callable[[dict[str, Any]], None] | None" = None) -> None:
        self._path = Path(path) if path else None
        self._fh = None
        self.model = model
        # Which policy played every run written here. Empty strings keep old
        # callers source-compatible; new callers always pass them, because a
        # journal that cannot say which weights produced it is the exact
        # failure PHASE_TWO section 3.2 exists to end.
        self.policy_version = policy_version
        self.git_sha = git_sha
        self._on_event = on_event

        if self._path is not None:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = self._path.open("a", encoding="utf-8")
            except Exception:
                # Deliberately broad. A bad path raises ValueError rather than
                # OSError, and the rule here is that no journal problem may cost
                # a run -- so the only correct response to any failure is to
                # carry on without a journal and say so once.
                logger.exception("Could not open the journal at %s; running without one",
                                 self._path)
                self._fh = None

        # Stamped once per process. The journal is appended across sessions and
        # the run counter restarts at 1 each time, so a file can hold runs
        # 1,2,3,1,2,3 -- and anything grouping by run number silently merges runs
        # that have nothing to do with each other. Seen in the first live session
        # to use this file.
        self.session = time.strftime("%Y%m%dT%H%M%S")
        self.run_index = 0
        self._state: dict[str, Any] = {}
        self._reset_run_state()

    # -- lifecycle ---------------------------------------------------------

    def _reset_run_state(self) -> None:
        self._started = False
        self._floor: Any = None
        self._room: Any = None
        self._act: Any = None
        self._in_combat = False
        self._combat_hp: int | None = None
        self._combat_floor: Any = None
        self._combat_room: Any = None
        self._combat_enemies: list = []
        self._combat_round: Any = None
        self._cards_this_combat = 0
        self._turns_this_combat = 0
        # WHICH AGENT PLAYED THIS FIGHT. Live search is disabled per-combat
        # after two exceptions and the fight silently continues on the trained
        # model, so "the run had search enabled" does not mean "the boss fight
        # was searched". That distinction is unrecoverable from the old logs
        # and is exactly where the 42-point boss gap could be hiding.
        self._searches_this_combat = 0
        self._search_failures_this_combat = 0
        # The run's own history, kept so `run_end` is self-contained: the path
        # walked, and the last deck and relics seen. Deriving these by scanning
        # backwards through a session's journal for the right `floor` rows is
        # possible but nobody does it, which is the same as not having them.
        self._path: list[dict[str, Any]] = []
        self._last_deck: list[str] | None = None
        self._last_relics: Any = None

    def start_run(self, run_index: int) -> None:
        self.run_index = run_index
        self._reset_run_state()

    # -- writing -----------------------------------------------------------

    def write(self, event: str, **fields: Any) -> None:
        """One event. Never raises: a lost line must not cost the run."""
        record = {
            "t": round(time.time(), 3),
            "session": self.session,
            "run": self.run_index,
            "event": event,
            "floor": self._floor,
        }
        record.update(fields)
        if self._fh is not None:
            try:
                self._fh.write(json.dumps(record, default=str) + "\n")
                self._fh.flush()
            except Exception:
                logger.debug("Journal write failed", exc_info=True)
        # The telemetry tap sees every record whether or not a journal file is
        # open, and its failures are its own: a broker problem must not reach
        # a run, and must not stop the journal line above from landing. The
        # stamp travels with the tap rather than into every file line, which
        # would repeat the version thousands of times per session.
        if self._on_event is not None:
            try:
                self._on_event({
                    **record,
                    "policy_version": self.policy_version,
                    "git_sha": self.git_sha,
                })
            except Exception:
                logger.debug("journal on_event tap failed", exc_info=True)

    # -- observing the stream ---------------------------------------------

    def observe(self, state: dict[str, Any]) -> None:
        """Feed every state through here, before the agent decides."""
        if not isinstance(state, dict):
            return
        self._state = state

        state_type = str(state.get("type", ""))
        floor = state.get("floor", self._floor)
        room = state.get("room_type", self._room)

        # Latched from whatever state carries them. The run-over screen does
        # not, so reading the deck at `run_end` alone would record nothing for
        # exactly the runs worth studying -- the ones that died.
        deck = _deck_cards(state)
        if deck is not None:
            self._last_deck = deck
        if state.get("relics"):
            self._last_relics = state.get("relics")

        if not self._started:
            self._started = True
            self._floor = floor
            self.write(
                "run_start",
                character=state.get("character") or state.get("character_id"),
                ascension=state.get("ascension", 0),
                model=self.model,
                policy_version=self.policy_version,
                git_sha=self.git_sha,
            )

        # Act transitions are one-way progress: when `act` increments, the
        # run has just cleared the previous act's boss. Recorded as its own
        # event so a clear can be derived from the journal alone, independent
        # of the run-end summary's `act_cleared` boolean. The first time we
        # see an act number is the run's starting act, not a clear, so it is
        # recorded silently.
        #
        # Tracked before the floor block below, so the act_clear event carries
        # the floor the boss was on -- the previous floor -- rather than the
        # floor the run crossed into. That is the diagnostically useful number:
        # "the boss on floor 17 was beaten", not "the run is now on floor 18".
        act = state.get("act")
        if isinstance(act, int):
            if isinstance(self._act, int) and act > self._act:
                self.write("act_clear", act_from=self._act, act_to=act,
                           floor=self._floor, room_type=self._room,
                           hp=state.get("run_hp"), max_hp=state.get("run_max_hp"),
                           deck=self._last_deck, relics=self._last_relics,
                           path=list(self._path))
                self._act = act
            elif not isinstance(self._act, int):
                self._act = act
            # act < self._act would be a bridge bug; ignored rather than
            # rewinding the tracker.

        if floor != self._floor and floor is not None:
            self._floor = floor
            self._path.append({
                "floor": floor, "room_type": room,
                "hp": state.get("run_hp"), "gold": state.get("gold"),
                "deck_size": state.get("deck_size"),
            })
            self.write("floor", room_type=room,
                       hp=state.get("run_hp"), max_hp=state.get("run_max_hp"),
                       gold=state.get("gold"), deck_size=state.get("deck_size"),
                       relics=state.get("relics"), deck=self._last_deck)
        if room is not None:
            self._room = room

        in_combat = state_type in _COMBAT_STATES
        if in_combat and not self._in_combat:
            self._begin_combat(state)
        elif self._in_combat and not in_combat:
            # An overlay is not the end of the fight. Ignore it and stay in
            # combat; the fight ends when a state arrives that could only
            # follow one.
            if state_type not in _COMBAT_OVERLAY_STATES:
                self._end_combat(state)
        elif in_combat:
            self._during_combat(state)

    def _begin_combat(self, state: dict[str, Any]) -> None:
        self._in_combat = True
        player = state.get("player") or {}
        self._combat_hp = player.get("hp", state.get("run_hp"))
        self._combat_floor = self._floor
        self._combat_room = self._room
        self._combat_round = state.get("round")
        self._cards_this_combat = 0
        self._turns_this_combat = 0
        # WHICH AGENT PLAYED THIS FIGHT. Live search is disabled per-combat
        # after two exceptions and the fight silently continues on the trained
        # model, so "the run had search enabled" does not mean "the boss fight
        # was searched". That distinction is unrecoverable from the old logs
        # and is exactly where the 42-point boss gap could be hiding.
        self._searches_this_combat = 0
        self._search_failures_this_combat = 0
        self._combat_enemies = [
            {"id": e.get("id"), "hp": e.get("hp"), "max_hp": e.get("max_hp")}
            for e in (state.get("enemies") or [])
        ]
        self.write(
            "combat_start",
            room_type=self._combat_room,
            hp=self._combat_hp,
            max_hp=player.get("max_hp", state.get("run_max_hp")),
            enemies=self._combat_enemies,
            deck_size=state.get("deck_size"),
            relics=state.get("relics"),
            deck=self._last_deck,
            potions=state.get("potion_slots"),
        )

    def _during_combat(self, state: dict[str, Any]) -> None:
        round_number = state.get("round")
        if round_number is not None and round_number != self._combat_round:
            self._combat_round = round_number
            self._turns_this_combat += 1
            player = state.get("player") or {}
            self.write(
                "turn",
                round=round_number,
                hp=player.get("hp"),
                block=player.get("block"),
                enemies=[
                    {"id": e.get("id"), "hp": e.get("hp")}
                    for e in (state.get("enemies") or [])
                ],
            )

    def _end_combat(self, state: dict[str, Any]) -> None:
        self._in_combat = False
        hp_after = state.get("run_hp")
        before = self._combat_hp
        self.write(
            "combat_end",
            room_type=self._combat_room,
            combat_floor=self._combat_floor,
            hp_before=before,
            hp_after=hp_after,
            damage_taken=(before - hp_after)
            if isinstance(before, int) and isinstance(hp_after, int)
            else None,
            cards_played=self._cards_this_combat,
            turns=self._turns_this_combat,
            searches=self._searches_this_combat,
            search_failures=self._search_failures_this_combat,
            played_by=("search" if self._searches_this_combat else "model"),
            enemies=self._combat_enemies,
        )

    def note_search(self, *, failed: bool = False) -> None:
        """Record that the search was asked for this combat's next action."""
        if failed:
            self._search_failures_this_combat += 1
        else:
            self._searches_this_combat += 1

    # -- observing the decisions ------------------------------------------

    def record_play_card(self, hand_index: int, target: Any) -> None:
        hand = self._state.get("hand") or []
        card = hand[hand_index] if 0 <= hand_index < len(hand) else None
        player = self._state.get("player") or {}
        self._cards_this_combat += 1
        self.write(
            "card_played",
            card=_describe(card),
            cost=(card or {}).get("cost") if isinstance(card, dict) else None,
            target=target,
            energy=player.get("energy"),
            hp=player.get("hp"),
            block=player.get("block"),
            round=self._state.get("round"),
            hand=[_describe(c) for c in hand],
        )

    def record_end_turn(self) -> None:
        player = self._state.get("player") or {}
        self.write(
            "end_turn",
            round=self._state.get("round"),
            hp=player.get("hp"),
            block=player.get("block"),
            energy_left=player.get("energy"),
            hand_left=[_describe(c) for c in (self._state.get("hand") or [])],
        )

    def record_potion(self, slot: int, target: Any = None) -> None:
        slots = self._state.get("potion_slots") or []
        potion = slots[slot] if 0 <= slot < len(slots) else None
        self.write("potion_used", potion=potion, slot=slot, target=target)

    def record_choice(self, index: int | list[int], skipped: bool = False) -> None:
        state_type = str(self._state.get("type", ""))
        described = [_describe(o) for o in _options_for(self._state)]

        chosen: Any = None
        if not skipped and isinstance(index, int) and 0 <= index < len(described):
            chosen = described[index]
        elif isinstance(index, list):
            chosen = [described[i] for i in index if 0 <= i < len(described)]

        self.write(
            "choice",
            screen=state_type,
            room_type=self._room,
            chosen=chosen,
            index=index,
            skipped=skipped,
            offered=described,
            hp=self._state.get("run_hp"),
            gold=self._state.get("gold"),
            deck_size=self._state.get("deck_size"),
            # Shop, removal, upgrade and event choices are all deck-building
            # decisions, and every one of them was recorded against a deck
            # SIZE. What was bought only means something next to what was
            # already held.
            deck=self._last_deck,
        )

    def record_run_end(self, summary: dict[str, Any]) -> None:
        """The run's outcome, WITH the deck, the relics and the path that made it.

        Previously this wrote `deck_size` and `relic_count` -- two integers
        describing a run whose whole shape was already in memory. A lost boss
        fight is only diagnosable next to what she brought to it, and the
        counts cannot be argued with.
        """
        fields = {k: v for k, v in summary.items() if k != "run"}
        fields.setdefault("deck", self._last_deck)
        fields.setdefault("relics", self._last_relics)
        fields.setdefault("path", list(self._path))
        self.write("run_end", **fields)

    # -- plumbing ----------------------------------------------------------

    def wrap(self, client: Any) -> Any:
        """A client that journals every action it sends.

        Wrapped once rather than called at each decision site, so a decision site
        added later cannot quietly go unrecorded.
        """
        return _JournallingClient(client, self)

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None


class _JournallingClient:
    """Proxies a BridgeClient, recording the actions that pass through it."""

    def __init__(self, client: Any, journal: RunJournal) -> None:
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_journal", journal)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._client, name, value)

    # -- the actions worth recording --------------------------------------

    def play_card(self, hand_index: int, target_index: Any = None, *args: Any, **kwargs: Any):
        self._journal.record_play_card(hand_index, target_index)
        return self._client.play_card(hand_index, target_index, *args, **kwargs)

    def end_turn(self, *args: Any, **kwargs: Any):
        self._journal.record_end_turn()
        return self._client.end_turn(*args, **kwargs)

    def use_potion(self, slot: int, *args: Any, **kwargs: Any):
        self._journal.record_potion(slot, kwargs.get("target_index"))
        return self._client.use_potion(slot, *args, **kwargs)

    def choose(self, index: int, *args: Any, **kwargs: Any):
        self._journal.record_choice(index)
        return self._client.choose(index, *args, **kwargs)

    def choose_many(self, indexes: list[int], *args: Any, **kwargs: Any):
        self._journal.record_choice(list(indexes))
        return self._client.choose_many(indexes, *args, **kwargs)

    def skip(self, *args: Any, **kwargs: Any):
        self._journal.record_choice(-1, skipped=True)
        return self._client.skip(*args, **kwargs)
