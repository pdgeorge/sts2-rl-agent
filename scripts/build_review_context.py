"""Generate the system prompt a reviewing model reads before every run.

    .venv/bin/python scripts/build_review_context.py

Writes `output/review_context.md`: what Slay the Spire 2 is, how act 1 works,
how this agent decides, and the real behaviour of every card and monster it can
meet -- all of it derived from the game's own source rather than written from
memory.

WHY IT IS BIG, AND WHY THAT IS FREE
-----------------------------------
`review_runs.py` sends the same system prompt on all 100 calls, and llama.cpp
caches an identical prompt prefix, so this is processed ONCE and reused for
every run after the first. A large reference costs one prompt evaluation and
then nothing, while the alternative -- a small prompt and a per-run card list --
pays unique tokens on every single call. Bigger here is genuinely cheaper.

WHY IT IS GENERATED AND NOT WRITTEN
-----------------------------------
An 8B has never seen this game. The card names overlap Slay the Spire 1 and the
effects do not, and a reference typed from memory would inherit exactly the
confusion it exists to prevent. Every number below comes from
`output/card_text.json` (which `generate_card_text.py` derives from the
decompile) and from the simulator's own monster factories -- the same
constants the agent plays with. Regenerate it after a game update.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: The pools an Ironclad run can draw from. Other characters' cards cannot
#: appear, and listing 400 of them would just be noise for the reviewer.
COLORS = ("ironclad", "colorless", "curse", "status", "event", "token")

MONSTER_MODULES = ("sts2_env.monsters.act1", "sts2_env.monsters.act1_weak",
                   "sts2_env.monsters.act2", "sts2_env.monsters.shared")

PREAMBLE = """# Slay the Spire 2 — reference for reviewing an agent's play

You are reviewing an AI agent playing **Slay the Spire 2**. Everything you need
is in this document; use it rather than what you remember.

## This is NOT Slay the Spire 1

Different game, different numbers, and many cards share a name with a Slay the
Spire 1 card while doing something else entirely. Ironclad's Defend gains 5
Block here. Bash costs 2 and applies a 50%-more-damage mark rather than
"Vulnerable 2". If a card below contradicts your memory, the card below is
right. Every entry in this document was generated from the game's own code.

## The run

A run is a sequence of floors. **Act 1 is 17 floors and ends with a boss.**
Clearing act 1 means beating that boss and reaching act 2.

- The Ironclad starts at **80 max HP** with 3 energy per turn.
- **HP does not refill between fights.** Damage taken on floor 3 is still gone
  at the boss on floor 17. This is why chip damage decides runs, and why a
  fight won at a cost can still lose the run.
- Rooms: `Monster` (ordinary fight), `Elite` (hard fight, drops a relic),
  `RestSite` (heal ~30% of max HP, or upgrade a card), `Shop`, `Treasure`,
  `Unknown` (an event, or any of the above), `Boss`.
- Between fights the player picks a card reward (or skips), buys from shops,
  and chooses which room to walk to next.

## A fight

- Each turn: draw 5, spend up to 3 energy, then end the turn and the enemies
  act. Block absorbs damage and is **lost at the start of your next turn** —
  block held over is block wasted.
- Enemies **telegraph** their next move. In the transcript that is shown in
  square brackets after their HP: `NIBBIT 45hp [ATTACK 12]` means it will deal
  12 damage when the turn ends. `[ATTACK 8x3]` is three hits of 8.
- Killing an enemy removes its damage for the rest of the fight. Blocking
  removes that damage once. This is why finishing an enemy is usually worth
  more than the raw numbers suggest.
- Powers persist for the fight. Relics persist for the run.

## How the agent decides

It does not use a neural policy for combat. Each turn it **enumerates every
legal ordering of its hand**, plays each ordering out on a copy, lets the
enemies respond, and scores the resulting position two turns ahead. It keeps
the best-scoring line.

That has two consequences worth remembering while reviewing:

- It is very good at sequencing and arithmetic within its horizon. Blaming it
  for "not seeing" a two-card combo this turn is usually wrong.
- It is blind past that horizon, and its score has **no term for potions held
  in reserve** — a potion in the belt is worth zero to it, so drinking one is
  free and any board improvement makes the line look better.

## Reading the transcript

```
=== f11 Elite | hp 62/80 | deck 19 | BYGONE_EFFIGY 108hp [ATTACK 16]
T1  hp 62  blk 0 | BYGONE_EFFIGY 108hp [ATTACK 16]
    played  BASH->0, STRIKE_IRONCLAD->0                     0.612  (replans: 2)
    passed  DEFEND_IRONCLAD, BASH->0                        0.608
    passed  STRIKE_IRONCLAD->0, STRIKE_IRONCLAD->0          0.571
    -> hp 62 to 46 over 2 turns, 3 cards
```

- `played` is what it actually did, taken from the game's own record.
- `passed` are lines it considered and rejected, with their scores.
- `->0` is the target enemy index.
- The score is its evaluation of the resulting position. Higher is better and
  roughly **1.0 is a full health bar**, so a gap of 0.004 is about a third of
  one HP — a coin flip, not a mistake. A gap of 0.1 is eight HP and means it
  was confident.
- `replans: N` is how many times it re-searched mid-turn.

## What a useful review looks like

Find where the run was decided. Be selective. Specifically worth looking for:

- Damage taken in ordinary fights that a different line would have avoided —
  small amounts, repeatedly, are what actually kills these runs.
- A kill available and not taken, where the dead enemy would have stopped
  attacking.
- Block gained that was never needed, or block short of the telegraphed hit.
- Potions drunk on trivial fights, or carried unused into a death.
- Card rewards taken that never got played, or skips that should not have been.
- Rest sites spent upgrading while at low HP, or healing while nearly full.

Do not report a rejected line that scored within about 0.01 of the played one
unless you can say concretely why the evaluation is wrong. Naming nothing in a
well-played run is a valid and useful answer.
"""


def _cards() -> str:
    path = REPO / "output" / "card_text.json"
    if not path.exists():
        return "\n(card reference unavailable: run scripts/generate_card_text.py)\n"
    data = json.loads(path.read_text(encoding="utf-8"))
    by_color: dict[str, list[str]] = {}
    for key, info in sorted(data.items()):
        color = info.get("color")
        if color not in COLORS:
            continue
        desc = " ".join(str(info.get("description") or "").split()) or "(no effect text)"
        by_color.setdefault(color, []).append(
            f"- **{info.get('name', key)}** ({info.get('type')}, cost "
            f"{info.get('cost')}, {info.get('rarity')}): {desc}")
    out = ["\n# Cards\n",
           "Every card an Ironclad run can hold. Costs and numbers are the "
           "unupgraded values; a `+` after a card name in a transcript means "
           "the upgraded version.\n"]
    for color in COLORS:
        if by_color.get(color):
            out.append(f"\n## {color.title()} cards\n")
            out.extend(by_color[color])
    return "\n".join(out)


def _monsters() -> str:
    from sts2_env.core.rng import Rng

    import sts2_env.cards  # noqa: F401
    import sts2_env.powers  # noqa: F401

    seen: dict[str, tuple[int, list[str]]] = {}
    for module_name in MONSTER_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001
            continue
        for name, func in vars(module).items():
            if not name.startswith("create_") or not callable(func):
                continue
            try:
                creature, ai = func(Rng(0))
            except Exception:  # noqa: BLE001
                continue
            monster_id = str(getattr(creature, "monster_id", "") or "")
            if not monster_id or monster_id in seen:
                continue
            moves = []
            for state_id, state in (getattr(ai, "states", None) or {}).items():
                for intent in getattr(state, "intents", None) or ():
                    dmg = getattr(intent, "damage", 0) or 0
                    hits = getattr(intent, "hits", 1) or 1
                    if dmg:
                        moves.append(f"{state_id} {dmg}" + (f"x{hits}" if hits > 1 else ""))
            seen[monster_id] = (getattr(creature, "max_hp", 0), moves)

    out = ["\n# Enemies\n",
           "HP is one roll of the range the game uses, so a live fight may "
           "differ by a few points. Damage figures are the base values before "
           "Strength, Weak or the target's Vulnerable are applied.\n"]
    for monster_id, (hp, moves) in sorted(seen.items()):
        move_text = ", ".join(moves[:6]) if moves else "no attacks (buffs/debuffs only)"
        out.append(f"- **{monster_id}** ({hp} HP): {move_text}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="output/review_context.md")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(REPO))

    text = PREAMBLE + _cards() + "\n" + _monsters() + "\n"
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

    print(f"wrote {path}  {len(text) / 1024:.0f} KB  (~{len(text) / 4 / 1000:.1f}k tokens)")
    print("Sent as the system prompt on every review call. llama.cpp caches an "
          "identical prefix,\nso it is evaluated once and reused for the other 99 runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
