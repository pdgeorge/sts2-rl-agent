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


def noncombat_action(mgr, phase: str, run_mask, rng, *, layout=None) -> int | None:
    """The live agent's non-combat choice, as a run_env action index."""
    from sts2_env.gym_env.run_env import _CARD_RWD_START, _MAP_START, _REST_START
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

    # -- card reward: quality x archetype fit --------------------------------
    if phase == RunManager.PHASE_CARD_REWARD:
        picks = [a for a in actions if a.get("action") == "pick_card"]
        if picks:
            from ab_archetype_picking import _pick_card_reward
            got = _pick_card_reward(mgr, run_mask, rng, True)
            if got is not None:
                return got
        return _fallback(mgr, phase, run_mask, rng)

    # -- shop / event / treasure / boss relic: still the old heuristic --------
    return _fallback(mgr, phase, run_mask, rng)
