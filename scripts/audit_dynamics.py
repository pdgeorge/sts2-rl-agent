"""Does the simulator RESOLVE a turn the way the game does?

    .venv/bin/python scripts/audit_dynamics.py --capture output/bridge_boss_fights_<tag>.jsonl

`audit_reconstruction.py` already asks whether a rebuilt position matches the
one the game described, field by field, and it comes back clean. That is the
static half. It cannot catch the thing that actually costs runs: the simulator
agreeing about the board and then disagreeing about what happens NEXT.

The search plans its whole lookahead on the simulator's dynamics. If a monster's
damage, its move order, or a card's effect differs, every line it scores is
scored against a fight that is not the one on screen -- which is precisely the
shape of the four act 3 parity bugs found on 2026-08-17, and of the ~26 points
of the boss gap that arrival HP and relics do not explain.

WHY THIS IS NEWLY POSSIBLE
--------------------------
Until 2026-08-17 the raw capture kept 25 floor-1 states and nothing else, so
there were no consecutive states to step between. It now keeps whole fights --
3,361 boss states over 85 fights in `boss_telemetry` -- and consecutive states
are exactly what a dynamics test needs: the game's own answer to "what happens
next" is sitting in the following record.

THREE TESTS, IN ORDER OF WHAT THEY COST TO GET WRONG
----------------------------------------------------
1. **NEXT MOVE.** Rebuild the position at the end of a round, end the turn, and
   ask the simulator's monster AI which move comes next. Compare to the
   `intent_move_id` the game reports on the following round. This is the state
   machine the lookahead depends on and it needs no damage arithmetic at all.

2. **ENEMY TURN.** Same step, but compare what the enemies DID: the player's HP
   after their attacks, and each enemy's HP and block. Card draw is not compared
   -- the bridge never sends draw pile ORDER, so the hand after a shuffle is
   unknowable and comparing it would report noise as divergence.

3. **CARD PLAY.** Not implemented. Inferring the play from the hand delta is
   unsound -- 11% of same-round transitions name a card that leaves the hand
   without being played (exhausted, discarded, moved by another card), and
   counting those would measure the inference rather than the simulator. It
   needs the journal's `card_played` keyed to the capture by (run, floor,
   round).

WHAT A CLEAN RESULT WOULD MEAN
------------------------------
That offline and live share a world model, and a paired offline A/B can be
trusted on any question whose mechanism lives inside a fight. That is worth far
more than any single experiment: 800 offline runs resolve +/-3.3 points in
hours, where 100 live runs resolve +/-9.3 in three.

A dirty result is worth more still, because each divergence names a constant.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Player HP is compared with a tolerance of zero -- it is the number the whole
#: funnel is made of, and "close" is how a 6 HP Burning Blood refund hid in the
#: chip figures for a week.
HP_TOLERANCE = 0


def _fights(capture: Path, room_types: set[str] | None, floor: int = 0) -> dict:
    """Captured combat states grouped per fight, in the order they arrived."""
    fights: dict[object, list] = defaultdict(list)
    with capture.open(encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                state = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(state, dict) or state.get("type") != "combat_action":
                continue
            if room_types and state.get("room_type") not in room_types:
                continue
            if floor and state.get("floor") != floor:
                continue
            key = (state.get("encounter_seed"), state.get("floor"))
            fights[key].append(state)
    return fights


def _played_card(before: dict, after: dict) -> str | None:
    """Which card left the hand between two states in the same round."""
    b = Counter(c.get("id") for c in (before.get("hand") or []))
    a = Counter(c.get("id") for c in (after.get("hand") or []))
    gone = b - a
    if sum(gone.values()) != 1:
        return None          # zero, or several: not a single unambiguous play
    return next(iter(gone))


def _enemies(state: dict) -> list:
    """Enemies BY POSITION, never by id.

    The Kin boss fields two KIN_FOLLOWERs. Keying a dict by id collapses them
    into one entry, and comparing each simulated follower against that single
    entry reports a divergence on every turn of every Kin fight -- which is what
    the first version of this script did, and it was the harness, not the game.
    Slot order is how the bridge and `to_combat` both address enemies.
    """
    return list(state.get("enemies") or [])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--rooms", default="",
                    help="comma-separated room types, e.g. Boss,Elite. Default: all")
    ap.add_argument("--limit", type=int, default=0, help="cap fights, for a quick look")
    ap.add_argument("--floor", type=int, default=0,
                    help="only this floor, e.g. 17 for the act 1 boss")
    ap.add_argument("--show", type=int, default=12, help="how many divergences to name")
    args = ap.parse_args()

    sys.path.insert(0, str(REPO))
    import sts2_env.cards  # noqa: F401
    import sts2_env.powers  # noqa: F401
    from sts2_env.search.situation import CombatSituation

    rooms = {r for r in args.rooms.split(",") if r} or None
    fights = _fights(Path(args.capture), rooms, args.floor)
    keys = list(fights)[: args.limit] if args.limit else list(fights)
    print(f"{len(keys)} captured fights, "
          f"{sum(len(fights[k]) for k in keys)} states\n")

    checked = Counter()
    diverged: dict[str, Counter] = defaultdict(Counter)
    unbuildable = 0

    for key in keys:
        states = fights[key]
        for before, after in zip(states, states[1:]):
            same_round = before.get("round") == after.get("round")
            try:
                combat = CombatSituation.from_bridge_state(before).to_combat_mid_fight(before)
            except Exception:
                unbuildable += 1
                continue

            if not same_round:
                # -- the enemies acted between these two states ---------------
                try:
                    combat.end_player_turn()
                except Exception as exc:  # noqa: BLE001
                    diverged["end_turn raised"][type(exc).__name__] += 1
                    continue

                want = _enemies(after)
                sim = list(combat.enemies)
                # Only compare when the rosters line up slot for slot. A death
                # compacts the game's list and not the simulator's, and a
                # misaligned comparison invents divergences rather than finding
                # them.
                if len(sim) == len(want):
                    for enemy, w in zip(sim, want):
                        mid = getattr(enemy, "monster_id", None)
                        if mid != w.get("id"):
                            aligned = False
                            break
                    else:
                        aligned = True
                else:
                    aligned = False

                if not aligned:
                    # The game compacts its list as monsters die and `to_combat`
                    # rebuilds the full opening roster, so the two disagree about
                    # slots the moment anything dies. Comparing across that
                    # invents divergences: the first version of this script
                    # reported 92 player-HP mismatches, 49 of them exactly -5,
                    # and they were this. Skip rather than guess a mapping.
                    checked["roster_misaligned"] += 1
                    continue

                checked["enemy_turn"] += 1
                got_hp = combat.player.current_hp
                want_hp = (after.get("player") or {}).get("hp")
                if want_hp is not None and abs(got_hp - want_hp) > HP_TOLERANCE:
                    lineup = "+".join(sorted(e.get("id") or "?" for e in _enemies(before)))
                    diverged["player hp after the enemy turn"][
                        f"{lineup} sim={got_hp} game={want_hp} (delta {got_hp - want_hp:+d})"] += 1

                if True:
                    for enemy, w in zip(sim, want):
                        mid = getattr(enemy, "monster_id", None)
                        if enemy.current_hp != w.get("hp"):
                            diverged["enemy hp after the enemy turn"][
                                f"{mid} sim={enemy.current_hp} game={w.get('hp')}"] += 1

                    checked["next_move"] += 1
                    for enemy, w in zip(sim, want):
                        mid = getattr(enemy, "monster_id", None)
                        ai = (getattr(combat, "enemy_ais", None) or {}).get(enemy.combat_id)
                        move = getattr(getattr(ai, "current_move", None), "state_id", None)
                        want_move = w.get("intent_move_id")
                        if move and want_move and move != want_move:
                            diverged["next move"][f"{mid} sim={move} game={want_move}"] += 1

            else:
                # -- CARD PLAY: not implemented, deliberately ------------------
                #
                # Inferring the play from the hand delta is not sound. 233 of
                # 2110 same-round transitions name a card that is not in the
                # rebuilt hand -- SETUP_STRIKE 92, RAGE 60, FLAME_BARRIER 23 --
                # and those are cards that LEAVE the hand without being played:
                # exhausted, discarded, or moved by another card's effect. A
                # test that counted those as divergence would be measuring its
                # own inference, which is how this file already produced one
                # false finding (see the roster note above).
                #
                # Doing it properly needs the action the agent actually sent.
                # The journal has it -- `card_played` carries card and target --
                # and the capture does not, so this wants the two keyed
                # together by (run, floor, round) rather than a guess from the
                # hand.
                checked["card_play_skipped"] += 1

    print("=" * 74)
    print(f"CHECKED   {dict(checked)}"
          + (f"   ({unbuildable} states would not rebuild)" if unbuildable else ""))
    print("=" * 74)
    if not any(diverged.values()):
        print("\n  No divergence. The simulator resolves these turns as the game does.")
    for kind, counter in diverged.items():
        total = sum(counter.values())
        base = checked.get("enemy_turn", 1) if "turn" in kind or "move" in kind else checked.get("card_play", 1)
        print(f"\n{kind.upper()}  {total} of {base}  ({100 * total / max(1, base):.1f}%)")
        for detail, count in counter.most_common(args.show):
            print(f"    x{count:<5} {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
