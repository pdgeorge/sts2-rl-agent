"""One readable transcript per run, for a reviewer to read end to end.

    .venv/bin/python scripts/export_run_transcripts.py --tag boss_telemetry

Writes `output/transcripts/<tag>/run_001.md` .. `run_100.md`, plus a
`REVIEW_PROMPT.md` holding the contract a reviewing model should answer with.

WHY A TRANSCRIPT AND NOT THE JOURNAL
------------------------------------
The journal is the right shape for a script and the wrong shape for a reader.
One run is ~240 KB of JSONL, about 62k tokens, and 65% of those bytes are
`combat_options` rows repeating the same field names a few thousand times.
Rendered as a transcript the same run is ~30 KB, about 8k tokens -- a 7.9x
compression that loses nothing a reviewer needs, and fits any local model with
room to think in.

WHAT IS IN IT
-------------
Everything that was decided, with what was NOT chosen beside it:

- every fight, turn by turn: HP, block, energy, each enemy with its telegraphed
  intent, the line she played and the best few she passed over, with scores
- every card reward, shop, rest and map choice, with the options she declined
- the deck as it stood, and how the run ended

The rejected lines are the point. A log of what she played can only support
"that looks wrong"; a log of what she passed over supports "she had X and took
Y, and the evaluator scored them 0.002 apart".

ONE BLOCK PER TURN, NOT PER CARD
--------------------------------
`LiveSearch` re-searches after every card, so a five-card turn logs five
`combat_options` rows. They are not prefixes of each other -- the hand changes
underneath, so the plan is re-derived -- and printing all five triples the size
while saying the same thing five times.

So a turn renders as what she ACTUALLY played, taken from the `card_played`
events, which is ground truth and cannot disagree with the game; and beneath
it the alternatives from the FIRST search of that turn, which is the decision
that set the turn up. `replans: N` records how many times she re-searched, so a
reader can still see a turn that went off-plan.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: How many rejected lines to show under each decision. The log holds eight;
#: three is enough to see whether the choice was close or clear, and keeps a
#: boss turn to a few lines rather than a screenful.
SHOW_REJECTED = 3

#: Alternatives closer than this to the line she played are NOT shown.
#:
#: The searcher always plays its top-scored line, so a rejected line 0.002
#: behind is a position the evaluator had no opinion about -- roughly a sixth of
#: one HP. Telling a reviewer "ignore near-ties" does not work: on the tuesday
#: session 51% of the decisions it flagged were inside 0.005, having been told
#: in the prompt that those are coin flips. Not showing them is a design, where
#: asking is a plea.
#:
#: The turn still appears, with a note of how many lines were tied, so the fight
#: still reads as a fight and a genuinely close decision is visible as close.
MIN_GAP_TO_SHOW = 0.02

RULES_PRIMER = """## What you are reading

Slay the Spire 2, act 1: 17 floors ending in a boss. The player is the Ironclad,
80 max HP, 3 energy a turn. HP does NOT refill between fights -- damage taken in
a corridor fight is still gone at the boss, which is why chip damage decides
runs. Enemies telegraph their next move, shown in `[ ]` after their HP.

**You almost certainly do not know this game.** It is not Slay the Spire 1 and
the cards are different. Every card and monster in this run is listed below with
its real behaviour, taken from the game's own code. Use that list and do not
rely on what you remember about a card with a similar name.

The agent chooses by enumerating every legal ordering of its hand, playing each
one out on a copy, letting the enemies respond, and keeping the best. `played`
is the line it took; `passed` are lines it considered and rejected. The number
is its evaluation of the resulting position -- higher is better, and roughly
1.0 is a full health bar. A gap of 0.004 between two lines means the agent
considered them near-identical; a gap of 0.1 means it was confident.
"""

REVIEW_PROMPT = """# Reviewing a run

This file is for a human deciding whether the review pipeline is behaving. The
prompt the model actually receives is built by `scripts/review_runs.py` from
`output/review_context.md` plus its own task section -- there is deliberately
only one copy of it, because a second would drift.

    .venv/bin/python scripts/build_review_context.py      # the game reference
    .venv/bin/python scripts/review_runs.py --tag <tag>   # sends it

## How to read a transcript

```
=== f11 Elite | hp 62/80 | deck 19 | BYGONE_EFFIGY 108hp [ATTACK 16]
T1  hp 62  blk 0 | BYGONE_EFFIGY 108hp [ATTACK 16]
    played  BASH->0, STRIKE_IRONCLAD->0                     0.612  (replans: 2)
    passed  STRIKE_IRONCLAD->0, STRIKE_IRONCLAD->0          0.571
    -> hp 62 to 46 over 2 turns, 3 cards
```

`played` is what she did, from the game's own record. `passed` are lines the
search considered and rejected, with their scores -- but only those at least
MIN_GAP_TO_SHOW behind, because anything closer is a position the evaluator had
no opinion about. A turn whose alternatives were all that close says so instead.

## What a good review looks like

Specific, located, and willing to say a run was played well. Every claim carries
a floor, and a combat claim a turn, so it can be checked against the journal
afterwards -- `scripts/aggregate_run_reports.py` discards the ones that cannot.
"""


def _norm(line: str) -> str:
    """Tidy an action label for reading.

    Existing sessions logged anything that was not a card as `action:62`;
    decoding what can be decoded is better than showing the reader an integer.
    Sessions after 2026-08-17 name these properly at capture time.
    """
    m = re.fullmatch(r"action:(\d+)", line)
    if not m:
        return line
    from sts2_env.core.constants import (
        ACTION_END_TURN,
        POTION_ACTION_START,
        POTION_TARGET_OPTIONS,
    )
    action = int(m.group(1))
    if action == ACTION_END_TURN:
        return "END_TURN"
    if action >= POTION_ACTION_START:
        slot, target = divmod(action - POTION_ACTION_START, POTION_TARGET_OPTIONS)
        return f"potion[slot {slot}]" + (f"->{target - 1}" if target else "")
    return f"action:{action}"


def _enemy(e: dict) -> str:
    bits = f"{e.get('id')} {e.get('hp')}hp"
    if e.get("block"):
        bits += f" blk{e['block']}"
    intent = e.get("intent")
    if intent:
        dmg = e.get("intent_damage")
        hits = e.get("intent_hits") or 1
        bits += f" [{intent}"
        if dmg:
            bits += f" {dmg}" + (f"x{hits}" if hits and hits > 1 else "")
        bits += "]"
    return bits


def _card_reference() -> dict:
    """Card behaviour, generated from the decompile by `generate_card_text.py`.

    Grounding, not decoration. An 8B model has never seen this game -- the cards
    share names with Slay the Spire 1 and do not share effects -- so without
    this it reviews a run against a game it invented. Only the cards this run
    actually touched are emitted, which keeps the primer at about 1k tokens
    instead of dumping all 577.
    """
    path = REPO / "output" / "card_text.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _appendix(events: list[dict], cards: dict) -> str:
    """The cards and monsters THIS run met, with what they really do."""
    seen_cards: set[str] = set()
    seen_monsters: dict[str, int] = {}
    for e in events:
        if e.get("card"):
            seen_cards.add(str(e["card"]).rstrip("+"))
        if e.get("chosen") and isinstance(e.get("chosen"), str):
            seen_cards.add(e["chosen"].rstrip("+"))
        for offer in e.get("offered") or []:
            if isinstance(offer, str):
                seen_cards.add(offer.rstrip("+"))
        for enemy in e.get("enemies") or []:
            if isinstance(enemy, dict) and enemy.get("id"):
                seen_monsters.setdefault(str(enemy["id"]), enemy.get("max_hp") or enemy.get("hp") or 0)

    lines = [RULES_PRIMER, "### Cards in this run"]
    for name in sorted(seen_cards):
        info = cards.get(name) or cards.get(name + "_CARD")
        if not info:
            continue
        desc = " ".join(str(info.get("description", "")).split())
        lines.append(f"- **{info.get('name', name)}** ({info.get('type')}, "
                     f"cost {info.get('cost')}, {info.get('rarity')}): {desc}")
    if seen_monsters:
        lines.append("\n### Enemies in this run")
        for name, hp in sorted(seen_monsters.items()):
            lines.append(f"- {name} ({hp} HP)")
    lines.append("\n---\n")
    return "\n".join(lines)


def render(events: list[dict], cards: dict | None = None) -> str:
    out: list[str] = []
    if cards is not None:
        out.append(_appendix(events, cards))
    pending = _new_turn()

    for e in events:
        ev = e.get("event")

        if ev == "run_start":
            pending = _new_turn()
            out.append(f"# Run {e.get('run')}  ({e.get('character')}, "
                       f"ascension {e.get('ascension', 0)})")
            out.append(f"policy {e.get('policy_version')} @ {e.get('git_sha')}\n")

        elif ev == "combat_start":
            _flush(out, pending)
            ens = ", ".join(_enemy(x) for x in (e.get("enemies") or []))
            relics = ", ".join(e.get("relics") or [])
            out.append(f"\n=== f{e.get('floor')} {e.get('room_type')} | "
                       f"hp {e.get('hp')}/{e.get('max_hp')} | deck {e.get('deck_size')} | {ens}")
            if relics:
                out.append(f"    relics: {relics}")
            pots = [p for p in (e.get("potions") or []) if p]
            if pots:
                out.append(f"    potions: {', '.join(pots)}")

        elif ev == "turn":
            _flush(out, pending)
            ens = ", ".join(_enemy(x) for x in (e.get("enemies") or []))
            out.append(f"T{e.get('round')}  hp {e.get('hp')}  blk {e.get('block')} | {ens}")

        elif ev == "card_played":
            pending["played"].append(
                f"{e.get('card')}" + (f"->{e['target']}" if e.get("target") not in (None, -1) else ""))

        elif ev == "combat_options":
            pending["replans"] += 1
            if pending["alts"] is None:   # first search of this turn sets it up
                opts = e.get("options") or []
                pending["alts"] = [
                    (", ".join(_norm(x) for x in (o.get("line") or [])) or "END_TURN",
                     o.get("score"))
                    for o in opts if not o.get("chosen")
                ][:SHOW_REJECTED]
                chosen = next((o for o in opts if o.get("chosen")), None)
                pending["planned"] = chosen.get("score") if chosen else None

        elif ev == "end_turn":
            _flush(out, pending)

        elif ev == "combat_end":
            _flush(out, pending)
            out.append(f"    -> hp {e.get('hp_before')} to {e.get('hp_after')} "
                       f"over {e.get('turns')} turns, {e.get('cards_played')} cards")

        elif ev == "choice":
            offered = e.get("offered")
            skipped = " (SKIPPED)" if e.get("skipped") else ""
            out.append(f"  [{e.get('screen')} f{e.get('floor')}] took "
                       f"{e.get('chosen')}{skipped}   offered: {offered}")

        elif ev == "card_reward_options":
            scored = e.get("options") or e.get("scores")
            if scored:
                out.append(f"      scores: {scored}")

        elif ev == "potion_used":
            out.append(f"    POTION {e.get('potion')} (slot {e.get('slot')})")

        elif ev == "run_end":
            out.append(f"\n## RUN END: floor {e.get('floor')}, act {e.get('act')}, "
                       f"{e.get('room_type')}, hp {e.get('run_hp')}/{e.get('run_max_hp')}, "
                       f"killed by {e.get('death_enemy_id')}, "
                       f"deck {e.get('deck_size')}, relics {e.get('relic_count')}")
    return "\n".join(out) + "\n"


def _new_turn() -> dict:
    return {"played": [], "alts": None, "planned": None, "replans": 0}


def _flush(out: list[str], pending: dict) -> None:
    """Emit the accumulated turn, then reset it in place."""
    if pending["played"] or pending["alts"]:
        score = pending.get("planned")
        head = f"    played  {', '.join(pending['played']) or 'END_TURN'}"
        if score is not None:
            head += f"{'':<4}{score:>8.3f}"
        if pending["replans"] > 1:
            head += f"   (replans: {pending['replans']})"
        out.append(head)
        shown = 0
        near = 0
        for line, alt_score in pending["alts"] or []:
            if alt_score is not None and score is not None:
                if score - alt_score < MIN_GAP_TO_SHOW:
                    near += 1
                    continue
            out.append(f"    passed  {line}{'':<4}{alt_score:>8.3f}"
                       if alt_score is not None else f"    passed  {line}")
            shown += 1
        if near and not shown:
            out.append(f"    (no clear alternative -- {near} line(s) within "
                       f"{MIN_GAP_TO_SHOW} of what she played)")
    pending.update(_new_turn())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="session tag, e.g. boss_telemetry")
    ap.add_argument("--journal", default=None, help="defaults to output/live_journal_<tag>.jsonl")
    ap.add_argument("--out", default=None, help="defaults to output/transcripts/<tag>")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(REPO))

    journal = Path(args.journal or f"output/live_journal_{args.tag}.jsonl")
    outdir = Path(args.out or f"output/transcripts/{args.tag}")
    outdir.mkdir(parents=True, exist_ok=True)

    runs: dict[tuple, list[dict]] = defaultdict(list)
    with journal.open(encoding="utf-8") as fh:
        for raw in fh:
            try:
                e = json.loads(raw)
            except json.JSONDecodeError:
                continue
            runs[(e.get("session"), e.get("run"))].append(e)

    cards = _card_reference()
    if not cards:
        print("WARNING: output/card_text.json missing -- transcripts will have no "
              "card reference, and a small model will review a game it invented.\n"
              "  regenerate with: .venv/bin/python scripts/generate_card_text.py")
    (outdir / "REVIEW_PROMPT.md").write_text(REVIEW_PROMPT, encoding="utf-8")

    total = 0
    for index, key in enumerate(sorted(runs, key=lambda k: (str(k[0]), k[1] or 0)), 1):
        text = render(runs[key], cards)
        path = outdir / f"run_{index:03d}.md"
        path.write_text(text, encoding="utf-8")
        total += len(text)

    n = len(runs)
    print(f"{n} transcripts -> {outdir}")
    print(f"  mean {total / n / 1024:.0f} KB per run  (~{total / n / 4 / 1000:.0f}k tokens)")
    print(f"  total {total / 1024 / 1024:.1f} MB  (~{total / 4 / 1000:.0f}k tokens for all {n})")
    print(f"\nreview contract: {outdir / 'REVIEW_PROMPT.md'}")
    print("collect the JSON replies as output/reports/<tag>/run_NNN.json, then:")
    print(f"  .venv/bin/python scripts/aggregate_run_reports.py --tag {args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
