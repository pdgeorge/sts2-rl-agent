"""Where does our simulator disagree with the actual game?

    python scripts/diff_vs_real_engine.py --cli /home/pd/Bucket/development/sts2-cli

THE QUESTION THIS EXISTS FOR
----------------------------
Live wins 29% of act 1 boss fights. Offline wins 72%. Same policy, same search,
same seeds. About 19 of those 43 points are explained by arrival HP and relics;
**~26 are not**, and every hypothesis for them has died: decks are equivalent,
relics are not the gap, the elite gate is a null.

The remaining candidate is that our simulator is simply not the game. That has
never been checkable, because the decompile says what the game *says* and only
running it says what it *does*. `sts2-cli` runs the real `sts2.dll` headless at
~1 ms per step, which makes it an oracle for the first time.

WHY THE OPENING STATE, AND NOT A WHOLE FIGHT
--------------------------------------------
A turn-by-turn diff needs the draw order synchronised on both sides, or the
first shuffle diverges and every later difference is noise about card order
rather than about rules. That is worth building second.

What is worth building FIRST is the thing the search actually consumes. The
search picks its block from `_incoming_damage`, which reads enemy intents. If an
intent is 14 in the game and 11 here, the agent blocks 11 every single turn of
every single fight, loses 3 HP a turn it never accounted for, and no amount of
search quality recovers it. That is exactly the shape of a large, uniform boss
gap, and it is visible in the opening state alone.

The same applies to enemy max HP: nine of those were wrong once already, and a
boss with 15% more HP than modelled is a fight the search thinks it wins.

WHAT A DIFFERENCE HERE MEANS
----------------------------
Our simulator is wrong, in a way that has been silently steering the agent. That
is the impossibility class -- the one that produced all four of this project's
real wins, against zero from tuning.

WHAT NO DIFFERENCE MEANS
------------------------
Just as valuable, and it must not be spun as a disappointment. It would mean the
combat model is sound and the ~26 points are in the LIVE path -- the bridge, or
`to_combat_mid_fight` reconstruction -- which is a different investigation with a
much smaller search space.

DO NOT COMPARE ROLLED VALUES. THIS SCRIPT'S FIRST RESULT WAS A FALSE POSITIVE
-----------------------------------------------------------------------------
The first run reported SHRINKER_BEETLE_WEAK as a difference: real engine 38 max
HP, ours 40. It is not a difference. `ShrinkerBeetle.cs` declares
`MinInitialHp 38` and `MaxInitialHp 40`, our `act1_weak.py` carries the same 38
and 40, and the two implementations simply rolled different legal values from
the same range.

The two RNG streams cannot be aligned -- they are different implementations with
different draw orders -- so **any quantity the game rolls is untestable by direct
comparison** and will generate false positives forever. That includes initial
HP, which move a monster picks, and shuffle order.

What IS comparable is everything deterministic given the position:

  - intent DAMAGE and HITS for a given move (the CLI exposes `type`, `damage`
    and `hits` per intent, which is exactly what `_incoming_damage` consumes)
  - the result of applying a known action to a known state
  - card and power arithmetic

The robust shape for intents, since move SELECTION is still random: sample many
combats on both sides, collect `(monster_id, intent_type, damage, hits)` tuples,
and compare the SETS. A damage value the real engine produces that ours never
does is a real bug; a move appearing at a different frequency is not.

`enemy.intent` is None at combat construction on our side -- intents roll at turn
start -- so the intent comparison needs both sides advanced one turn first. That
is the next piece of work and the reason this file currently only diffs the
opening roster.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

#: Answer whatever the CLI asks until it is sitting on the map, so a combat can
#: be entered cleanly. Neow's follow-up screens (card_select, card_reward,
#: bundle_select) must be resolved or `enter_room` returns them instead.
MAX_DRIVE_STEPS = 40


class Cli:
    """One headless game process, spoken to over stdin/stdout JSON."""

    def __init__(self, root: Path, dotnet: str):
        proj = root / "src" / "Sts2Headless" / "Sts2Headless.csproj"
        self.p = subprocess.Popen(
            [dotnet, "run", "--project", str(proj), "--no-build"],
            cwd=str(root), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.read()  # ready

    def send(self, obj) -> dict | None:
        self.p.stdin.write(json.dumps(obj) + "\n")
        self.p.stdin.flush()
        return self.read()

    def read(self) -> dict | None:
        while True:
            line = self.p.stdout.readline()
            if not line:
                return None
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    def close(self):
        try:
            self.send({"cmd": "quit"})
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


def drive_to_map(cli: Cli, state: dict) -> dict | None:
    """Answer decisions until the map is reached.

    Neow is answered by TRYING each option until one advances. Four Neow options
    hang the harness -- they add content directly with no selection screen, so
    `Chosen()` never completes and the same event is re-presented. That is a
    known upstream bug, out of scope here, and retrying a different index walks
    around it.
    """
    stuck_options: set[int] = set()
    for _ in range(MAX_DRIVE_STEPS):
        if state is None:
            return None
        decision = state.get("decision")
        if decision == "map_select":
            return state
        if decision == "event_choice":
            options = state.get("options") or []
            nxt = next((i for i in range(len(options)) if i not in stuck_options),
                       None)
            if nxt is None:
                return None
            before = state.get("event_name")
            state = cli.send({"cmd": "action", "action": "choose_option",
                              "args": {"option_index": nxt}})
            if (state and state.get("decision") == "event_choice"
                    and state.get("event_name") == before):
                stuck_options.add(nxt)
            continue
        # Arg names are exact and unforgiving: `indices` is a COMMA-SEPARATED
        # STRING, and the reward/bundle keys are `card_index`/`bundle_index`.
        # Getting them wrong returns {"type": "error"} with no `decision`, which
        # reads as a hang rather than a mistake -- it cost an hour here.
        if decision == "card_select":
            n = max(int(state.get("min_select") or 1), 1)
            state = cli.send({"cmd": "action", "action": "select_cards",
                              "args": {"indices": ",".join(str(i) for i in range(n))}})
            continue
        if decision == "card_reward":
            state = cli.send({"cmd": "action", "action": "select_card_reward",
                              "args": {"card_index": 0}})
            continue
        if decision == "bundle_select":
            state = cli.send({"cmd": "action", "action": "select_bundle",
                              "args": {"bundle_index": 0}})
            continue
        if state.get("type") == "error":
            return None
        state = cli.send({"cmd": "action", "action": "proceed"})
    return None


def real_encounter(cli: Cli, encounter: str, seed: str) -> dict | None:
    """Opening state of `encounter` in the real engine."""
    state = cli.send({"cmd": "start_run", "character": "Ironclad", "seed": seed})
    state = drive_to_map(cli, state)
    if state is None:
        return None
    return cli.send({"cmd": "enter_room", "type": "combat",
                     "encounter": encounter})


def ours(setup_name: str) -> dict | None:
    """Opening state of the same encounter in our simulator."""
    import sts2_env.cards  # noqa: F401
    import sts2_env.powers  # noqa: F401
    from sts2_env.search.situation import CardRef, CombatSituation

    deck = tuple([CardRef(card_id="STRIKE_IRONCLAD", upgraded=False)] * 5
                 + [CardRef(card_id="DEFEND_IRONCLAD", upgraded=False)] * 4
                 + [CardRef(card_id="BASH", upgraded=False)])
    situation = CombatSituation(
        situation_id=setup_name, character_id="Ironclad",
        current_hp=80, max_hp=80, deck=deck, encounter=setup_name,
        encounter_seed=1, combat_seed=1, relics=(), room_type="MONSTER",
        act_floor=1, total_floor=1)
    try:
        combat = situation.to_combat()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "enemies": [
            {"id": str(getattr(e, "monster_id", getattr(e, "id", "?"))),
             "hp": int(getattr(e, "current_hp", 0)),
             "max_hp": int(getattr(e, "max_hp", 0))}
            for e in combat.enemies
        ],
    }


def norm_real(state: dict | None) -> list[dict]:
    if not state:
        return []
    out = []
    for e in (state.get("enemies") or []):
        out.append({
            "id": str(e.get("id") or e.get("name") or "?"),
            "hp": int(e.get("hp") or 0),
            "max_hp": int(e.get("max_hp") or e.get("hp") or 0),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cli", default="/home/pd/Bucket/development/sts2-cli")
    ap.add_argument("--dotnet", default="/home/pd/.dotnet/dotnet")
    ap.add_argument("--seed", default="diff_0")
    ap.add_argument("--encounters", default="",
                    help="Comma-separated setup names; default is all of act 1.")
    ap.add_argument("--out", default="output/diff_vs_real_engine.json")
    args = ap.parse_args()

    if args.encounters:
        setups = [s.strip() for s in args.encounters.split(",") if s.strip()]
    else:
        import sts2_env.encounters.act1 as act1
        setups = sorted(n for n in dir(act1) if n.startswith("setup_"))

    root = Path(args.cli)
    rows = []
    print(f"{len(setups)} act 1 encounters, ours against the real engine")
    print()
    for setup in setups:
        encounter = setup[len("setup_"):].upper()
        cli = Cli(root, args.dotnet)
        try:
            real = norm_real(real_encounter(cli, encounter, args.seed))
        finally:
            cli.close()
        mine = ours(setup)
        row = {"setup": setup, "encounter": encounter,
               "real": real, "ours": mine}
        rows.append(row)

        if mine and mine.get("error"):
            print(f"  {encounter:<34} OUR SIM RAISED  {mine['error'][:50]}")
            continue
        if not real:
            print(f"  {encounter:<34} real engine gave nothing (see json)")
            continue
        ours_list = (mine or {}).get("enemies") or []
        r_hp = [e["max_hp"] for e in real]
        o_hp = [e["max_hp"] for e in ours_list]
        flag = "OK " if r_hp == o_hp else "DIFF"
        print(f"  {encounter:<34} {flag}  real max_hp {r_hp}  ours {o_hp}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print()
    print(f"full detail written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
