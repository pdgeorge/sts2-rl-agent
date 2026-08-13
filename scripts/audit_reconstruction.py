"""Does the live search plan against the board that is actually on screen?

    python scripts/audit_reconstruction.py

THE 26 POINTS
-------------
Live wins 26.5% of act 1 boss fights, offline 72%, on the same policy and the
same search. Arrival HP and relics explain about 19 of the 43 points. Decks are
equivalent, relic acquisition is not the gap, and the elite gate is a null. What
is left is the one thing offline never does: **reconstruct**.

Every live combat step, `LiveSearch` rebuilds the position from the bridge's
JSON -- `from_bridge_state(state).to_combat_mid_fight()` -- and searches THAT.
If reconstruction loses a power, misreads an intent, or drops a block, the
search plans a line that is optimal for a board that does not exist, and then
plays it into the real game. Offline never pays this cost because it searches a
position it built itself.

`from_bridge_state`'s own docstring already names the stakes: "a searcher that
clones a fight different from the one on screen is worse than useless -- it
looks correct and is not."

Nobody has ever checked.

WHAT THIS DOES
--------------
For every captured live combat state: reconstruct it, then compare the
reconstruction field by field against the JSON it was built from. The game told
us the answer in the same payload, so this is closed-book -- the same standard
`check_rng_parity_against_capture.py` holds itself to.

Compared, because the search consumes all of them:

  player   hp, block, energy, powers
  enemies  count, id, hp, block, powers, intent type and damage
  hand     size and contents

MONSTER HP IS READ, NOT ROLLED, AND THAT IS DELIBERATE
-------------------------------------------------------
`CombatState.cs:499` rolls enemy HP from `RunState.Rng.Niche`, a run-level
stream whose position depends on everything earlier in the run, so it cannot be
re-derived from an encounter seed no matter how faithful the generator is.
`from_bridge_state` therefore records the HP the game reported and `to_combat`
applies it. A mismatch here is a real defect in that path, not RNG divergence --
which is the mistake `diff_vs_real_engine.py` made on its first run.

WHAT A CLEAN RESULT MEANS
-------------------------
That reconstruction is faithful, the 26 points are somewhere else, and this
whole line of investigation closes. That is a real outcome and must be reported
as one rather than quietly reframed.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CAPTURE_GLOBS = ("output/bridge_protocol*.jsonl", "output/bridge_boss_fights*.jsonl")


def _power_name(raw) -> str:
    """Canonical power name, via the REAL resolver rather than a copy of it.

    The mod's slugify leaves a `_POWER` suffix that `_coerce_power_id` strips.
    An audit that upper-cases both sides instead reports every single power as a
    mismatch -- which is exactly what the first version of this script did, and
    it is the same mistake the identifier audit was written to stop.
    """
    from sts2_env.search.situation import _coerce_power_id
    pid = _coerce_power_id(str(raw))
    return getattr(pid, "name", str(raw).upper())


def _card_name(raw) -> str:
    """Canonical card name, via the REAL resolver.

    The game says SETUP_STRIKE where the enum member is SETUP_STRIKE_CARD, and
    `resolve_card_id` already reconciles them -- that suffix was the 68
    unplayable cards bug. Comparing raw strings instead reported every act 1
    boss hand as a mismatch and nearly had reconstruction blamed for the boss
    gap on the strength of it.
    """
    from sts2_env.search.situation import resolve_card_id
    cid = resolve_card_id(str(raw))
    return getattr(cid, "name", str(raw).upper())


def _norm_power(p) -> tuple[str, int]:
    if isinstance(p, dict):
        raw = p.get("id") or p.get("name") or "?"
        amt = p.get("amount", p.get("stacks", 0))
        try:
            amt = int(amt)
        except (TypeError, ValueError):
            amt = 0
        return (_power_name(raw), amt)
    return (_power_name(p), 0)


def _from_state(state: dict) -> dict:
    """What the GAME said, normalised.

    Player fields live under `player`, NOT at the top level, and the intent is a
    bare string with `intent_damage`/`intent_hits` as sibling keys on the enemy.
    Reading them the obvious way produced 322 phantom block mismatches and
    silently skipped the intent comparison altogether.
    """
    enemies = []
    for e in (state.get("enemies") or []):
        if not isinstance(e, dict):
            continue
        if e.get("is_alive") is False:
            continue
        enemies.append({
            "id": str(e.get("id") or "?").upper(),
            "hp": e.get("hp"),
            "block": e.get("block") or 0,
            "powers": sorted(_norm_power(p) for p in (e.get("powers") or [])),
            "intent": str(e.get("intent") or "").upper(),
            "intent_damage": e.get("intent_damage"),
            "intent_hits": e.get("intent_hits"),
        })
    player = state.get("player") or {}
    return {
        "player_hp": player.get("hp"),
        "player_block": player.get("block") or 0,
        "energy": player.get("energy"),
        "player_powers": sorted(_norm_power(p) for p in (player.get("powers") or [])),
        "hand": sorted(_card_name(c.get("id") if isinstance(c, dict) else c)
                       for c in (state.get("hand") or [])),
        "enemies": enemies,
    }


def _from_combat(combat) -> dict:
    """What OUR reconstruction believes, in the same shape."""
    def powers_of(entity):
        out = []
        for pid, power in (getattr(entity, "powers", {}) or {}).items():
            name = getattr(pid, "name", str(pid)).upper()
            amount = getattr(power, "amount", 0)
            try:
                amount = int(amount)
            except (TypeError, ValueError):
                amount = 0
            out.append((name, amount))
        return sorted(out)

    enemies = []
    for e in combat.enemies:
        if not getattr(e, "is_alive", True):
            continue
        intent = getattr(e, "intent", None)
        itype = (getattr(intent, "intent_type", None)
                 or getattr(intent, "type", None) or "")
        enemies.append({
            "id": str(getattr(e, "monster_id", "?")).upper(),
            "hp": int(getattr(e, "current_hp", 0)),
            "block": int(getattr(e, "block", 0) or 0),
            "powers": powers_of(e),
            "intent": str(getattr(itype, "name", itype) or "").upper(),
            "intent_damage": getattr(intent, "damage", None),
            "intent_hits": getattr(intent, "hits", None),
        })
    # Energy lives on `current_player_state`, NOT on `player` -- reading it off
    # `player` returns None, which scored 640 phantom mismatches of the form
    # "3 != 0" while situation.py:740 was assigning it correctly all along.
    p = combat.player
    cps = getattr(combat, "current_player_state", None) or p
    return {
        "player_hp": int(getattr(p, "current_hp", 0)),
        "player_block": int(getattr(p, "block", 0) or 0),
        "energy": int(getattr(cps, "energy", 0) or 0),
        "player_powers": powers_of(p),
        "hand": sorted(_card_name(getattr(getattr(c, "card_id", c), "name",
                                           getattr(c, "card_id", c)))
                       for c in (getattr(combat, "hand", None) or [])),
        "enemies": enemies,
    }


def _compare(game: dict, mine: dict) -> list[str]:
    """Field names that disagree. Missing on either side is not a mismatch."""
    bad = []
    for key in ("player_hp", "player_block", "energy"):
        g, m = game.get(key), mine.get(key)
        if g is None or m is None:
            continue
        if int(g) != int(m):
            bad.append(f"{key}({g}!={m})")
    if game.get("player_powers") and game["player_powers"] != mine["player_powers"]:
        bad.append("player_powers")
    if game.get("hand") and game["hand"] != mine["hand"]:
        bad.append("hand")

    ge, me = game["enemies"], mine["enemies"]
    if len(ge) != len(me):
        bad.append(f"enemy_count({len(ge)}!={len(me)})")
        return bad
    for i, (g, m) in enumerate(zip(ge, me)):
        if g["id"] != m["id"]:
            bad.append(f"e{i}.id({g['id']}!={m['id']})")
        if g["hp"] is not None and int(g["hp"]) != int(m["hp"]):
            bad.append(f"e{i}.hp({g['hp']}!={m['hp']})")
        if int(g["block"] or 0) != int(m["block"] or 0):
            bad.append(f"e{i}.block({g['block']}!={m['block']})")
        if g["powers"] != m["powers"]:
            bad.append(f"e{i}.powers({g['powers']}!={m['powers']})")
        if g["intent"] and m["intent"] and g["intent"] != m["intent"]:
            bad.append(f"e{i}.intent({g['intent']}!={m['intent']})")
        # THE FIELD THE SEARCH ACTUALLY SPENDS. `_incoming_damage` turns this
        # into how much block to hold; if it is wrong the agent mis-blocks every
        # turn of every fight, which is the right shape for a uniform boss gap.
        if (g["intent_damage"] is not None and m["intent_damage"] is not None
                and int(g["intent_damage"]) != int(m["intent_damage"])):
            bad.append(f"e{i}.intent_dmg({g['intent_damage']}!={m['intent_damage']})")
        if (g["intent_hits"] is not None and m["intent_hits"] is not None
                and int(g["intent_hits"]) != int(m["intent_hits"])):
            bad.append(f"e{i}.intent_hits({g['intent_hits']}!={m['intent_hits']})")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="output/audit_reconstruction.jsonl")
    args = ap.parse_args()

    import sts2_env.cards  # noqa: F401
    import sts2_env.powers  # noqa: F401
    from sts2_env.search.situation import CombatSituation

    states = []
    for pattern in CAPTURE_GLOBS:
        for path in glob.glob(str(REPO / pattern)):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(d, dict) and d.get("type") == "combat_action":
                        states.append(d)
    if args.limit:
        states = states[:args.limit]
    print(f"{len(states)} captured live combat states")
    print()

    fields = collections.Counter()
    raised = collections.Counter()
    ok = 0
    rows = []
    for state in states:
        try:
            situation = CombatSituation.from_bridge_state(state)
            combat = situation.to_combat_mid_fight(state)
        except Exception as exc:
            raised[type(exc).__name__] += 1
            continue
        try:
            bad = _compare(_from_state(state), _from_combat(combat))
        except Exception as exc:
            raised[f"compare:{type(exc).__name__}"] += 1
            continue
        if bad:
            for b in bad:
                fields[b.split("(")[0]] += 1
            rows.append({"floor": state.get("floor"), "mismatch": bad})
        else:
            ok += 1

    total = ok + len(rows)
    print("=" * 66)
    print(f"RECONSTRUCTION AUDIT   {ok}/{total} states reconstructed faithfully"
          + (f"  ({100 * ok / total:.0f}%)" if total else ""))
    print()
    if raised:
        print(f"  {sum(raised.values())} states could not be reconstructed at all:")
        for k, v in raised.most_common(8):
            print(f"     {v:>5}  {k}")
        print()
    if fields:
        print("  mismatching fields, by how often they disagree:")
        for k, v in fields.most_common(20):
            print(f"     {v:>5}  {k}")
    else:
        print("  no field mismatches. Reconstruction is faithful on this fixture,")
        print("  and the 26 points are NOT here.")
    print("=" * 66)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
