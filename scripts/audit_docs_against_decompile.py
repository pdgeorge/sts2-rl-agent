"""Every claim the docs make about the game, checked against the decompile.

    .venv/bin/python scripts/audit_docs_against_decompile.py

WHY
---
Two documented "known issues" were wrong on 2026-08-19, and both were wrong the
same way: a claim about OUR code written up as a claim about the GAME.

  - `BATTLE_FRIEND_V1` was recorded as "unmodelled content the audit just
    caught". It was fully modelled; `monsters/factory.py` never imported the
    module it lived in.
  - "The mod's skip click does not consume the card reward" stood for five days
    and sent the fix to the wrong side of the bridge. The click worked. Not
    completing the reward is documented game behaviour --
    `PostAlternateCardRewardAction.EndSelectionAndDoNotCompleteReward`.

Both were caught by reading `decompiled/` rather than trusting the note. That is
a pattern, not two accidents, and it is worth a machine that does it on demand.

WHAT IT CHECKS
--------------
Anything in the docs that LOOKS like it names game code -- a `Foo.cs`, a
`Type.Member`, a bare PascalCase type in backticks -- must resolve somewhere in
`decompiled/`. A name that does not is one of:

  - stale: the game renamed or removed it, so the note describes a build that
    no longer exists
  - invented: the note reasoned about a mechanic and gave it a plausible name
  - ours: the note is about this repo's code, not the game's, and saying so in
    game-shaped language is what produced both failures above

It cannot tell those apart, and does not try. It narrows a few thousand lines of
prose to a list worth reading, which is the job.

WHAT IT DOES NOT CHECK
----------------------
Whether a name that DOES resolve is described correctly. `Orrery` exists and the
note about it could still be wrong. That needs reading, and the point of the
list is to make the reading finite.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DECOMPILED = REPO / "decompiled"

DOCS = ["docs/KNOWN_ISSUES.md", "WEEKEND_DECISIONS.md", "PHASE_TWO.md",
        "SCOREBOARD.md", "ROADMAP.md", "docs/MODELS.md", "HANDOFF.md",
        "docs/BRIDGE_OFFLINE_ALIGNMENT.md", "README.md"]

#: Names that look like game types but are ours, or are plain English. Listing
#: them beats loosening the pattern, which would hide the real misses.
IGNORE = frozenset("""
PolicyConfig EvalWeights SearchAgent CombatState CombatSituation RunManager
LiveSearch RawCapture RunJournal STS2RunEnv PowerInstance CardInstance
MonsterAI MoveState Creature Rng PhaseTwo ScoreBoard KnownIssues README
BridgeServer TODO NOTE WARNING IMPORTANT STOP MISS HIT CLOSED WITHDRAWN
GitHub Python JSON JSONL CSV HTML CUDA VRAM GPU CPU RAM API CLI URL
McNemar Wilson Bonferroni Cochran Armitage Ctrl Steam Godot Harmony
BaseLib MegaCrit Slay Spire Ironclad Silent Defect Necrobinder Regent
Cyra Claude Qwen Sonnet Opus RabbitMQ Ollama LM Studio
PHASE_TWO PROPOSAL ROADMAP SCOREBOARD WEEKEND_DECISIONS HANDOFF MODELS
MissingFieldException InvalidOperationException KeyError ValueError
""".split())

#: Our own code, in this repo and in the mod. Named here rather than pattern-
#: matched because the whole point is that ours and theirs LOOK alike in prose --
#: writing about our type as though it were the game's is the mistake this audit
#: exists to find, so the separation has to be explicit.
OURS = frozenset("""
CloneError RlCardSelector RlCombatHandler RlNonCombatRoomHandlers RlAutoSlayer
RlRewardsScreenHandler RlCardRewardScreenHandler RlShopRoomHandler RlMapHandler
SendStateAndWaitForActionAsync WaitForQuiescenceAsync PlayCardAndWaitAsync
RequestPreemptedException StateAdapter AnimationSpeedPatch WaitSpeedPatch
CombatSolver HeuristicCombatSolver FrozenSearchCombatSolver FrozenRLCombatSolver
HierarchicalRunEnv STS2CombatEnv STS2RunEnv ObservationLayoutMismatch
DeckDirection PowerId CardId RelicId RoomType IntentType PendingCardChoice
TransformCardsReward
""".split())

PATTERNS = [
    # `Foo.cs`
    (re.compile(r"`([A-Z][A-Za-z0-9_]*)\.cs`"), "file"),
    # decompiled/Some.Namespace/Foo.cs
    (re.compile(r"decompiled/[A-Za-z0-9_.]+/([A-Z][A-Za-z0-9_]*)\.cs"), "file"),
    # `Type.Member` or `Type.Member(...)`
    (re.compile(r"`([A-Z][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)"), "member"),
    # a lone backticked PascalCase word with no dot and no underscore
    (re.compile(r"`([A-Z][a-z]+(?:[A-Z][a-z0-9]+)+)`"), "type"),
]


def _grep(needle: str) -> bool:
    """Does this identifier appear anywhere in the decompile?"""
    out = subprocess.run(
        ["grep", "-rqlF", "--", needle, str(DECOMPILED)],
        capture_output=True, text=True, check=False)
    return out.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", nargs="*", default=None)
    args = ap.parse_args()

    if not DECOMPILED.is_dir():
        print(f"no decompile at {DECOMPILED}")
        return 1

    docs = args.docs or DOCS
    found: dict[str, list] = defaultdict(list)
    for rel in docs:
        path = REPO / rel
        if not path.is_file():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for pattern, kind in PATTERNS:
                for m in pattern.finditer(line):
                    name = m.group(1)
                    if name in IGNORE or name in OURS or len(name) < 4:
                        continue
                    found[name].append((rel, lineno, kind, line.strip()[:110]))

    print(f"{len(found)} distinct game-shaped names across {len(docs)} documents\n")
    missing = {n: hits for n, hits in sorted(found.items()) if not _grep(n)}
    print("=" * 74)
    print(f"NAMES THAT DO NOT APPEAR ANYWHERE IN THE DECOMPILE: {len(missing)}")
    print(f"  ({len(OURS)} known-ours names excluded; see OURS in this file -- "
          f"TransformCardsReward\n   is on that list because a doc named it as a "
          f"GAME reward type and it is ours)")
    print("=" * 74)
    for name, hits in missing.items():
        where = ", ".join(sorted({f"{r}:{l}" for r, l, _k, _t in hits}))
        print(f"\n  {name}   ({where})")
        print(f"    {hits[0][3]}")
    if not missing:
        print("\n  none -- every name the docs use resolves in the decompile.")
    print("\n" + "=" * 74)
    print("A name here is stale, invented, or ours-described-as-theirs. This\n"
          "cannot tell them apart; it makes the reading finite. A name that DOES\n"
          "resolve can still be described wrongly -- that needs eyes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
