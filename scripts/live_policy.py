"""Run the LIVE non-combat heuristics offline, instead of an amateur stand-in.

Offline runs claimed to use "the same `agent_runner` heuristics the bridge uses
for map, rest and card rewards". They did not. They used
`harvest_combat_benchmark._noncombat_action`, whose own docstring calls it "a
plausible amateur's non-combat choice -- not an attempt at good play", and which
returns None for map selection entirely, so **the map was chosen at random**.

That mattered more than it sounds. The HP-gate sweep patched
`ROOM_MIN_HP_FRACTION` and every arm came back +0.0% +/- 0.0% -- an impossibly
clean result, and the giveaway: the gate was never consulted, because nothing
offline ever called `_pick_map_node`. It also means the `sim vs live` agreement
(32% simulated against 23% live) compared two different agents: live routes with
the HP gate, offline routed at random.

WHAT THIS DOES
--------------
Builds the bridge-shaped dict each live chooser expects out of the simulator's
own RunManager, calls the real chooser, and maps its answer back to a run_env
action index. Same functions the bridge path runs, so an offline result is about
the agent that actually ships.

WHERE IT STILL FALLS BACK
-------------------------
Shop, event, treasure and boss-relic keep the old heuristic for now. Map, rest
and card reward are the three that decide a run's shape -- routing, HP economy
and deck -- and are the ones the live agent has real logic for. The rest are
marked here rather than quietly left, so the next person knows the boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from sts2_env.bridge import agent_runner as _live  # noqa: E402
from sts2_env.run.run_manager import RunManager  # noqa: E402


def _boss_is_next(mgr) -> bool:
    """Is the very next room the act boss, per the simulator's own map?

    Computed rather than inferred from a floor number. The simulator does not
    count the Ancient/Neow room, so its floors run one behind the live game's
    and the live `floor + 1 in {17, 33, 50}` rule never fires here.
    """
    rs = getattr(mgr, "run_state", None)
    act_map = getattr(rs, "map", None)
    boss = getattr(act_map, "boss_point", None)
    coord = getattr(boss, "coord", None)
    boss_row = getattr(coord, "row", None)
    if boss_row is None:
        return False
    # `act_floor` is `coord.row + 1`, so the row just walked is act_floor - 1.
    current_row = int(getattr(rs, "act_floor", 0) or 0) - 1
    return current_row + 1 >= int(boss_row)


def _hp_fields(mgr) -> dict:
    """The HP/floor/act fields every live chooser reads off a bridge state."""
    rs = getattr(mgr, "run_state", None)
    player = getattr(rs, "player", None)
    return {
        "run_hp": int(getattr(player, "current_hp", 0) or 0),
        "run_max_hp": int(getattr(player, "max_hp", 0) or 0),
        "floor": int(getattr(rs, "total_floor", 0) or 0),
        "act": int(getattr(rs, "current_act_index", 0) or 0) + 1,
    }


def _deck(mgr) -> list[dict]:
    rs = getattr(mgr, "run_state", None)
    player = getattr(rs, "player", None)
    return [{"id": c.card_id.name, "upgraded": bool(getattr(c, "upgraded", False))}
            for c in getattr(player, "deck", [])]




#: The run action space puts the card-reward skip immediately after the three
#: card slots. `_pick_card_reward_index` returning None means "take nothing",
#: and without a slot to express that offline the decision cannot be replayed.
_CARD_REWARD_SKIP_OFFSET = 3


def _skip_action(run_mask) -> int | None:
    from sts2_env.gym_env.run_env import _CARD_RWD_START

    slot = _CARD_RWD_START + _CARD_REWARD_SKIP_OFFSET
    return int(slot) if slot < len(run_mask) and run_mask[slot] else None


def _skip_offered(run_mask) -> bool:
    return _skip_action(run_mask) is not None


def _bridge_options(actions: list) -> list[dict]:
    """Offline actions in the shape the live choosers read.

    Both sides use the same action strings, so this carries them through rather
    than mapping them, and passes `label`/`description`/`price` where present --
    the event heuristic reads the human text, and the shop one reads the price.
    """
    options = []
    for i, action in enumerate(actions):
        option = {
            "index": i,
            "action": action.get("action"),
            "enabled": bool(action.get("enabled", True)),
        }
        for key in ("label", "description", "option_id", "price", "card_id",
                    "relic_id", "potion_id", "id"):
            if action.get(key) is not None:
                option[key] = action[key]
        options.append(option)
    return options


def _delegate(mgr, actions, chooser, start: int, run_mask):
    """Ask a live chooser, and map its answer back to a run action index."""
    if not actions:
        return None
    state = {"options": _bridge_options(actions), "deck": _deck(mgr),
             **_hp_fields(mgr)}
    try:
        chosen = int(chooser(state))
    except Exception:
        return None
    chosen = max(0, min(chosen, len(actions) - 1))
    slot = start + chosen
    return int(slot) if slot < len(run_mask) and run_mask[slot] else None

def noncombat_action(mgr, phase: str, run_mask, rng, *, layout=None) -> int | None:
    """The live agent's non-combat choice, as a run_env action index."""
    from sts2_env.gym_env.run_env import (
        _BOSS_RELIC_START, _CARD_RWD_START, _EVENT_START, _MAP_START,
        _REST_START, _SHOP_START, _TREASURE_START,
    )
    from harvest_combat_benchmark import _noncombat_action as _fallback

    def offer(index: int) -> int | None:
        return int(index) if index < len(run_mask) and run_mask[index] else None

    actions = mgr.get_available_actions() or []

    # -- map: the one that was random, and the one the HP gate lives in -------
    if phase == RunManager.PHASE_MAP_CHOICE:
        moves = [a for a in actions if a.get("action") == "move"]
        if not moves:
            return None
        state = {
            "nodes": [{"index": i, "type": a.get("point_type")}
                      for i, a in enumerate(moves)],
            **_hp_fields(mgr),
        }
        chosen = _live._pick_map_node(state)
        return offer(_MAP_START + max(0, min(int(chosen), len(moves) - 1)))

    # -- rest: HEAL vs SMITH, the pre-boss decision ---------------------------
    if phase == RunManager.PHASE_REST_SITE:
        opts = [a for a in actions if a.get("action") == "rest_option"]
        if not opts:
            return None
        state = {
            "options": [{"index": i, "id": a.get("option_id") or a.get("id"),
                         "enabled": True}
                        for i, a in enumerate(opts)],
            "deck": _deck(mgr),
            "boss_is_next": _boss_is_next(mgr),
            **_hp_fields(mgr),
        }
        chosen = _live._pick_rest_option(state)
        return offer(_REST_START + max(0, min(int(chosen), len(opts) - 1)))

    # -- card reward: the LIVE chooser, not a copy of it ----------------------
    if phase == RunManager.PHASE_CARD_REWARD:
        picks = [a for a in actions if a.get("action") == "pick_card"]
        if picks:
            # WHY NOT ab_archetype_picking._pick_card_reward. It was a second
            # implementation, and the two had already drifted where it matters:
            # both rank with `card_quality.rank_cards`, but the live side skips
            # on `best_score < SKIP_THRESHOLD or deck_size > LARGE_DECK_SIZE`
            # while the copy skipped only on `best_score <= 0` and had no
            # deck-size rule at all. Deck size is what decides an act 1 boss
            # fight, so the offline agent was measuring a deckbuilder the live
            # agent is not.
            #
            # Keeping copies in step is not a plan; deleting them is.
            state = {
                "cards": [{"id": a.get("card_id")} for a in picks
                          if a.get("card_id")],
                "deck": _deck(mgr),
                "deck_size": len(_deck(mgr)),
                # The offline env exposes a skip slot, so offline can decline
                # too -- which is the point, since the live mod can now click
                # the game's Skip button.
                "can_skip": _skip_offered(run_mask),
                **_hp_fields(mgr),
            }
            chosen = _live._pick_card_reward_index(state)
            if chosen is None:
                skip = _skip_action(run_mask)
                if skip is not None:
                    return skip
                chosen = 0
            return offer(_CARD_RWD_START + max(0, min(int(chosen),
                                                      len(picks) - 1)))
        return _fallback(mgr, phase, run_mask, rng)

    # -- shop, event, treasure, boss relic: the live choosers, not a stand-in --
    #
    # These were the last four running `harvest_combat_benchmark._noncombat_action`,
    # whose own docstring calls it "a plausible amateur's non-combat choice --
    # not an attempt at good play". So offline was scoring an agent that shopped,
    # took events, opened chests and picked boss relics differently from the one
    # that ships, on every run.
    #
    # The delegation is mechanical because both sides already speak the same
    # action vocabulary -- buy_relic, buy_card, buy_potion, remove_card,
    # leave_shop, collect, pick_relic, event_choice are the identical strings on
    # the wire and in RunManager -- so the bridge shape is a relabelling of the
    # offline actions rather than a translation.
    table = {
        RunManager.PHASE_SHOP: (_live._pick_shop_option, _SHOP_START),
        RunManager.PHASE_TREASURE: (_live._pick_treasure_option, _TREASURE_START),
        RunManager.PHASE_BOSS_RELIC: (_live._pick_boss_relic_option, _BOSS_RELIC_START),
    }
    if phase in table:
        chooser, start = table[phase]
        chosen = _delegate(mgr, actions, chooser, start, run_mask)
        if chosen is not None:
            return chosen
        return _fallback(mgr, phase, run_mask, rng)

    if phase == RunManager.PHASE_EVENT:
        # `_pick_event_option` takes a `seen` map so it can refuse to take the
        # same paid option seven times -- Hot Baths re-presents itself and
        # charges more each time, which is what killed live run 1 (68 -> 41 HP
        # and still going). The counter lives on the manager so it survives the
        # per-decision calls within one event.
        seen = getattr(mgr, "_live_policy_events_seen", None)
        if seen is None:
            seen = {}
            setattr(mgr, "_live_policy_events_seen", seen)
        chosen = _delegate(mgr, actions, lambda st: _live._pick_event_option(st, seen),
                           _EVENT_START, run_mask)
        if chosen is not None:
            return chosen

    return _fallback(mgr, phase, run_mask, rng)
