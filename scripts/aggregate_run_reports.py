"""Turn 100 per-run reviews into one ranked report.

    .venv/bin/python scripts/aggregate_run_reports.py --tag boss_telemetry

Reads `output/reports/<tag>/run_NNN.json`, each one a reviewing model's reply
to `REVIEW_PROMPT.md`, and produces the thing worth reading: which mistakes
recur, how often, and what they are claimed to have cost.

WHY AN AGGREGATE AND NOT A HUNDRED REPORTS
------------------------------------------
One review of one run is an anecdote, and this project has a documented habit of
acting on those. A pattern across a hundred is a lead: "she over-blocks against
a telegraphed debuff" in 40 runs is worth a paired A/B, and the same claim in 2
runs is a reviewer noticing something once.

The ranking is frequency x claimed HP, so a small mistake made constantly
outranks a dramatic one made twice. That is the ordering the funnel cares
about: chip damage is 8.5 hallway fights a run.

WHAT IS CHECKED, AND WHAT IS NOT
--------------------------------
Every claim must carry a floor, and combat claims a turn, because a claim
without a location cannot be looked up in the journal afterwards. Claims are
also matched against how the run actually ENDED, so a report can be read
against ground truth rather than taken on trust -- a mistake claimed on floor
14 of a run that died on floor 9 did not happen.

Beyond that the claims are NOT filtered by score. A reviewer saying "the
evaluator preferred the worse line" is disagreeing with `evaluate.py`, and
that is the most interesting thing it can say, not a false positive to
suppress: prediction 10 showed more search does not help, so if there is
anything left in the fight it is in the scoring. Those are counted separately
rather than thrown away.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: A claim below this is noise by the reviewer's own admission; kept in the
#: tally but not in the headline, so one confident lead is not buried under
#: thirty maybes.
STRONG = {"high", "medium"}


def _load_json(text: str) -> dict | None:
    """Parse a reply, tolerating the fences models wrap JSON in."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


#: Words that carry no information about WHAT went wrong. Dropped before
#: clustering so "played Defend twice into a 6-damage intent" and "played
#: DEFEND_IRONCLAD into a damage intent" land in the same bucket -- the first
#: version keyed on the first six words and split that one claim across five
#: rows, which under-counts every real pattern and inflates the apparent
#: variety of them.
_FILLER = frozenset("""
a an the and or but into onto with without for from to of on at in by
played play plays playing used use uses using did does turn turn's card cards
twice thrice once again then when while that this it its her his their
was were is are be been being had has have having would should could
damage dmg hp intent enemy enemies point points instead rather than
one two three four five six seven eight nine ten
ironclad silent defect necrobinder regent watcher
move moves turn turns round rounds floor player agent she her
""".split())

#: Card names the model writes several ways. Canonicalised so the cluster key
#: does not depend on whether it wrote DEFEND_IRONCLAD, Defend Ironclad or
#: just Defend.
_ALIASES = {
    "defendironclad": "defend", "strikeironclad": "strike",
    "bashironclad": "bash", "defends": "defend", "strikes": "strike",
    "blocks": "block", "blocking": "block", "attacks": "attack",
    "kills": "kill", "killing": "kill", "blocked": "block",
}


def _claim_key(text: object) -> str:
    """A cluster key for one claim: what it is about, not how it was worded."""
    words = re.sub(r"[^a-z ]", " ", str(text or "").lower()).split()
    out = []
    for w in words:
        w = _ALIASES.get(w, w)
        if w in _FILLER or len(w) < 3:
            continue
        if w not in out:
            out.append(w)
    return " ".join(sorted(out)[:5]) or "(unlabelled)"


def _run_outcomes(journal: Path) -> dict[int, dict]:
    """How each run ended, keyed by TRANSCRIPT NUMBER, not by journal run index.

    A session restarts on a crash and the run counter restarts with it, so
    `tuesday` holds 103 runs under only 53 distinct indices. `export_run_transcripts`
    numbers its files 1..N over the runs sorted by (session, run), and the
    reports are named after those files -- so keying this table by the journal's
    bare `run` matched a report against a DIFFERENT run in another segment, and
    the "claim on a floor the run never reached" check was validating against
    the wrong ground truth entirely. Same ordering as the exporter, or the
    check is worse than not having one.
    """
    out: dict[int, dict] = {}
    if not journal.exists():
        return out
    ends: dict[tuple, dict] = {}
    seen: set = set()
    with journal.open(encoding="utf-8") as fh:
        for raw in fh:
            try:
                e = json.loads(raw)
            except json.JSONDecodeError:
                continue
            key = (e.get("session"), e.get("run"))
            seen.add(key)
            if e.get("event") == "run_end":
                ends[key] = {
                    "floor": e.get("floor"), "act": e.get("act"),
                    "room": e.get("room_type"), "killed_by": e.get("death_enemy_id"),
                    "cleared": bool(e.get("act_cleared")),
                }
    for index, key in enumerate(sorted(seen, key=lambda k: (str(k[0]), k[1] or 0)), 1):
        if key in ends:
            out[index] = ends[key]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reports", default=None, help="defaults to output/reports/<tag>")
    ap.add_argument("--journal", default=None,
                    help="defaults to output/live_journal_<tag>.jsonl")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    reports_dir = Path(args.reports or f"output/reports/{args.tag}")
    journal = Path(args.journal or f"output/live_journal_{args.tag}.jsonl")
    files = sorted(reports_dir.glob("run_*.json"))
    if not files:
        print(f"no reports in {reports_dir}. Export transcripts, review them, "
              f"and save each reply as run_NNN.json there.")
        return 1

    outcomes = _run_outcomes(journal)

    parsed, unparsed = [], []
    for path in files:
        data = _load_json(path.read_text(encoding="utf-8"))
        (parsed if data else unparsed).append((path, data))

    kinds = Counter()
    by_phrase: dict[str, list] = defaultdict(list)
    hp_by_kind = Counter()
    no_location = 0
    past_the_end = 0
    total_claims = 0
    empty_runs = 0
    good = Counter()

    for path, data in parsed:
        run = data.get("run")
        mistakes = data.get("mistakes") or []
        if not mistakes:
            empty_runs += 1
        for m in mistakes:
            total_claims += 1
            floor, kind = m.get("floor"), str(m.get("kind") or "unknown")
            conf = str(m.get("confidence") or "").lower()
            hp = m.get("cost_hp") or 0
            if floor is None:
                no_location += 1
                continue
            end = outcomes.get(run)
            if end and end.get("floor") is not None and floor > end["floor"]:
                past_the_end += 1
                continue
            kinds[kind] += 1
            hp_by_kind[kind] += hp if isinstance(hp, (int, float)) else 0
            by_phrase[_claim_key(m.get("did"))].append((run, floor, hp, conf))
        for g in data.get("good_plays") or []:
            good[" ".join(re.sub(r"[^a-z ]", "", str(g).lower()).split()[:6])] += 1

    n = len(parsed)
    print("=" * 74)
    print(f"{n} reports parsed"
          + (f", {len(unparsed)} UNPARSEABLE" if unparsed else "")
          + f"  |  {total_claims} claims  |  {empty_runs} runs reported clean")
    print("=" * 74)
    if no_location or past_the_end:
        print(f"  discarded: {no_location} without a location, "
              f"{past_the_end} on a floor the run never reached")

    print(f"\nBY KIND (claims, claimed HP)")
    for kind, count in kinds.most_common():
        print(f"  {kind:<16}{count:>5} claims{hp_by_kind[kind]:>8} HP   "
              f"{hp_by_kind[kind] / max(1, count):>5.1f} HP each")

    print(f"\nRECURRING, ranked by claims x claimed HP -- a pattern, not an anecdote")
    ranked = sorted(by_phrase.items(),
                    key=lambda kv: -(len(kv[1]) * (sum(x[2] for x in kv[1]) / max(1, len(kv[1])))))
    for phrase, hits in ranked[:args.top]:
        runs = sorted({h[0] for h in hits})
        strong = sum(1 for h in hits if h[3] in STRONG)
        hp = sum(h[2] for h in hits)
        print(f"  {len(hits):>3}x  {hp:>5} HP  ({strong} confident)  {phrase}")
        print(f"        runs {runs[:8]}{' ...' if len(runs) > 8 else ''}")

    if good:
        print(f"\nCALLED OUT AS GOOD")
        for phrase, count in good.most_common(5):
            print(f"  {count:>3}x  {phrase}")

    # Triage: which individual runs are worth opening. A ranked list of patterns
    # says what to change; this says which transcript to READ, which is the
    # other half of the question and the one an aggregate usually destroys.
    scored = []
    for path, data in parsed:
        run = data.get("run")
        ms = data.get("mistakes") or []
        hp = sum(m.get("cost_hp") or 0 for m in ms if isinstance(m.get("cost_hp"), (int, float)))
        end = outcomes.get(run) or {}
        scored.append((run, hp, len(ms), end.get("floor"), end.get("cleared"),
                       str(data.get("summary") or "")[:88]))

    worst = sorted((r for r in scored if r[1]), key=lambda r: -r[1])[:5]
    if worst:
        print(f"\nMOST EGREGIOUS -- open these transcripts first")
        for run, hp, n, floor, _cleared, summary in worst:
            print(f"  run {run:>3}  {hp:>4} HP over {n} claims, died floor {floor}")
            print(f"          {summary}")

    deepest = sorted((r for r in scored if r[3] is not None), key=lambda r: -r[3])[:5]
    if deepest:
        print(f"\nGOT FURTHEST -- what did these do right?")
        for run, hp, n, floor, cleared, summary in deepest:
            print(f"  run {run:>3}  floor {floor}"
                  f"{'  (cleared act 1)' if cleared else ''}"
                  f"  {n} claims, {hp} HP")
            print(f"          {summary}")

    clean = [r for r in scored if r[2] == 0 and r[4]]
    if clean:
        print(f"\n  {len(clean)} runs cleared act 1 with nothing flagged: "
              f"{[r[0] for r in clean][:12]}")

    print("\n" + "=" * 74)
    print("A claim appearing in 40 runs is a lead worth a paired A/B. In 2 runs it is\n"
          "a reviewer noticing something once. Rank by the first column, then go and\n"
          "read those runs' transcripts at the cited floor before building anything.")
    if unparsed:
        print(f"\nunparseable: {[p.name for p, _ in unparsed][:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
