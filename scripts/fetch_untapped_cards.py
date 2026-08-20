"""Pull the community card statistics from sts2.untapped.gg.

    .venv/bin/python scripts/fetch_untapped_cards.py --characters all
    .venv/bin/python scripts/fetch_untapped_cards.py --characters ironclad colorless
    .venv/bin/python scripts/fetch_untapped_cards.py --characters ironclad --parse-only

PERMISSION. pd contacted Untapped and they approved the bulk pull (2026-08-20).
The crawl is still deliberately gentle -- one request at a time, a delay between
them, an identifying User-Agent, and every page cached to disk so re-running the
PARSE never re-fetches. `--parse-only` re-reads the cache and touches the
network zero times; use it while iterating on the extraction.

WHAT IS BEING TAKEN, AND WHY THIS SOURCE IS WORTH HAVING
--------------------------------------------------------
Not a tier grade. Each card page carries, per act and per context (card reward,
shop, and so on):

    offered 2,500 times | Picked 92% | Act Winrate +7% | Run Winrate +6%

The winrates are DELTAS -- the lift in win rate for runs that took the card
against those that did not -- and the offer count is the sample size behind
them. That is the shape we need. `scripts/analyse_card_offers.py` measured what
our own 163 runs can resolve: about 392 runs CONTAINING a card to see a 10-point
effect, and roughly 1,565 for 5 points. We have nowhere near that per card and
will not soon. A source with thousands of offers per card is the only way to get
per-card information at all, which is why the prior is doing the work and our
own data is reduced to checking its ordering.

WHAT IT IS NOT
--------------
Human play, not Cyra's. She searches two turns ahead, plans nothing across
fights, and drafts by a fixed policy, so systematic disagreement is expected in
both directions -- scaling Powers probably over-rated for her, precise in-turn
sequencing under-rated. The pick rate is worth even less to us than the
winrate: it measures what humans LIKE, which is partly reputation. Treat the
delta as the signal, the pick rate as context, and validate the ordering against
our own runs before any of it reaches a policy.

The game is also in early access, so these numbers move with patches. The
decompile stays the authority on what a card DOES; this is only opinion about
what a card is WORTH.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "data/untapped/cache"
OUT = REPO / "data/untapped"

BASE = "https://sts2.untapped.gg"

#: Every card list the site publishes, read off its own navigation rather than
#: guessed. COLORLESS matters as much as IRONCLAD for us right now: those cards
#: are offered in an Ironclad run, so they are decisions the agent actually
#: makes. The class lists are for when it plays something other than Ironclad.
CHARACTERS = ("ironclad", "silent", "defect", "necrobinder", "regent", "colorless")
UA = "sts2-rl-agent research crawler (approved by Untapped, contact via repo)"

#: Seconds between requests. Slow on purpose, and slower than it needs to be.
#: The whole pull is ~93 pages and every one is cached, so a re-parse costs the
#: server nothing and a full refresh is a once-a-patch event -- there is no
#: reason at all to hurry someone else's machine for it. Untapped approved the
#: pull; that is a reason to be careful with it, not a licence to hammer.
DELAY = 5.0

#: A longer pause every so often, so a long crawl is not one unbroken stream of
#: requests even at the per-request delay.
BREATHER_EVERY = 20
BREATHER = 20.0


_fetched = 0


def _get(url: str, cache_name: str, *, refetch: bool = False,
         delay: float = DELAY) -> str:
    """Fetch, or return the cached copy. Only a real fetch costs a delay."""
    global _fetched
    path = CACHE / cache_name
    if path.exists() and not refetch:
        return path.read_text(encoding="utf-8")
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read().decode("utf-8", "replace")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    _fetched += 1
    time.sleep(delay)
    if _fetched % BREATHER_EVERY == 0:
        print(f"  ...{_fetched} fetched, pausing {BREATHER:.0f}s", flush=True)
        time.sleep(BREATHER)
    return body


def _slugs(character: str, *, refetch: bool = False,
           delay: float = DELAY) -> list[str]:
    html = _get(f"{BASE}/en/tier-list/cards/{character}",
                f"tier-list-{character}.html", refetch=refetch, delay=delay)
    return sorted(set(re.findall(r'href="/en/cards/([a-z0-9-]+)"', html)))


#: Every block carries one of these, so counting them in the raw text gives an
#: independent expectation for how many blocks the parser should return. The
#: two are compared on every run: see `_check_no_blocks_dropped`.
_MARKER = re.compile(r"\|(?:offered|seen|bought) [\d,]+ times\|")

#: The page is three sections -- Card Reward, Shop and Smith -- each with an
#: Act 1/2/3 block. Parsed by walking a tag-stripped stream and tracking the
#: current section, rather than with one big regex, because the first version
#: did use one and got two things wrong at once: it read the section from the
#: "offered N times" verb (so Shop rows were labelled the same as Card Reward
#: rows) and it required an Act Winrate, which ACT 3 DOES NOT HAVE. The act 3
#: blocks silently vanished and nothing said so.
_SECTION = re.compile(r"\|([A-Za-z][A-Za-z ]{1,20})\| Stats\|")
_BLOCK = re.compile(
    r"\|In Act (?P<act>\d)"
    r"\|(?P<verb>offered|seen|bought) (?P<n>[\d,]+) times"
    r"\|(?P<label>Picked|Bought|Upgraded)\|(?P<pct>[\d.]+)\|%"
    # Either a signed percentage or an em-dash. The dash is Untapped
    # SUPPRESSING the stat because the sample is too small, and demanding a
    # number here silently dropped 120 of 776 blocks -- biasing the dataset
    # toward exactly the well-sampled cards that least need a prior.
    r"(?:\|Act Winrate\|(?P<act_wr>[+\-][\d.]+%|—))?"
    r"\|Run Winrate\|(?P<run_wr>[+\-][\d.]+%|—)"
)
_NAME = re.compile(r"<title>([^<|\u2013]+)")


def _text_stream(html: str) -> str:
    """Tags collapsed to `|`. The class names are hashed per build, so anchoring
    on them would break on Untapped's next deploy; the visible text will not."""
    return re.sub(r"\|+", "|", re.sub(r"<[^>]+>", "|", html))


def _parse(slug: str, html: str) -> dict:
    stream = _text_stream(html)
    name = _NAME.search(html)

    # Where each section starts, so a block can be attributed to the section it
    # falls inside rather than to whichever verb happens to be nearest.
    sections = [(m.start(), m.group(1).strip()) for m in _SECTION.finditer(stream)]

    rows = []
    for match in _BLOCK.finditer(stream):
        section = "unknown"
        for start, title in sections:
            if start < match.start():
                section = title
            else:
                break
        def pct(raw):
            """A suppressed stat is None, not zero. Untapped prints an em-dash
            when it will not stand behind the number, and a measured 0% and a
            withheld one are different facts."""
            if raw is None or raw == "\u2014":
                return None
            return float(raw.rstrip("%"))

        act_wr = match.group("act_wr")
        rows.append({
            "section": section,
            "act": int(match.group("act")),
            "sample_verb": match.group("verb"),
            "n": int(match.group("n").replace(",", "")),
            "action": match.group("label"),
            "action_pct": float(match.group("pct")),
            # Absent on act 3 for every card. None, not 0.0 -- a missing
            # measurement and a measured zero are not the same thing.
            "act_winrate_delta": pct(act_wr),
            "run_winrate_delta": pct(match.group("run_wr")),
        })
    return {
        "slug": slug,
        "name": (name.group(1).strip() if name else slug),
        "blocks": rows,
    }


def _check_no_blocks_dropped(cards: list[dict]) -> list[tuple[str, int, int]]:
    """Every sample-size marker in the page must have produced a block.

    The parser's failure mode is silence: the regex stops matching, every
    request still returns 200, and the output is a smaller file that looks
    fine. Counting the markers independently is the only thing that catches it,
    and it already has -- an em-dash where a winrate should be cost 120 of 776
    blocks on the first full pull, all of them the small-sample cards.
    """
    mismatched = []
    for card in cards:
        path = CACHE / f"card-{card['slug']}.html"
        if not path.exists():
            continue
        stream = _text_stream(path.read_text(encoding="utf-8"))
        markers = len(_MARKER.findall(stream))
        if markers != len(card["blocks"]):
            mismatched.append((card["slug"], markers, len(card["blocks"])))
    return mismatched


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--characters", nargs="+", default=["ironclad"],
                    help=f"one or more of {', '.join(CHARACTERS)}, or 'all'")
    ap.add_argument("--parse-only", action="store_true",
                    help="re-parse the cache without touching the network")
    ap.add_argument("--refetch", action="store_true", help="ignore the cache")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--delay", type=float, default=DELAY,
                    help=f"seconds between real requests (default {DELAY}); "
                         f"cached pages cost nothing and are never delayed")
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)

    characters = list(CHARACTERS) if "all" in args.characters else args.characters
    unknown = [c for c in characters if c not in CHARACTERS]
    if unknown:
        raise SystemExit(f"unknown character(s) {unknown}; known: {list(CHARACTERS)}")

    for character in characters:
        _pull(character, args)
    return 0


def _pull(character: str, args) -> None:
    if args.parse_only:
        slugs = sorted(p.stem for p in CACHE.glob("card-*.html"))
        slugs = [s[len("card-"):] for s in slugs]
        if not slugs:
            raise SystemExit("nothing cached yet -- run without --parse-only first")
    else:
        slugs = _slugs(character, refetch=args.refetch, delay=args.delay)
        print(f"\n=== {character}: {len(slugs)} cards linked from the tier list")

    if args.limit:
        slugs = slugs[:args.limit]

    cards, empty = [], []
    for index, slug in enumerate(slugs, 1):
        try:
            html = _get(f"{BASE}/en/cards/{slug}", f"card-{slug}.html",
                        refetch=args.refetch and not args.parse_only,
                        delay=args.delay)
        except Exception as exc:  # noqa: BLE001 - one bad page is not fatal
            print(f"  [{index}/{len(slugs)}] {slug}: FAILED {exc}")
            continue
        card = _parse(slug, html)
        if not card["blocks"]:
            empty.append(slug)
        cards.append(card)
        if index % 10 == 0 or index == len(slugs):
            print(f"  [{index}/{len(slugs)}] {slug}: {len(card['blocks'])} blocks",
                  flush=True)

    dropped = _check_no_blocks_dropped(cards)

    out = OUT / f"cards_{character}.json"
    out.write_text(json.dumps(cards, indent=1) + "\n", encoding="utf-8")

    total = sum(len(c["blocks"]) for c in cards)
    print(f"\n{len(cards)} cards, {total} stat blocks -> {out}")
    if dropped:
        lost = sum(m - p for _, m, p in dropped)
        print(f"\n!! PARSER DROPPED {lost} BLOCKS across {len(dropped)} cards. The page "
              f"has a sample-size\n   marker the parser produced no row for, which "
              f"means the extraction is\n   incomplete in a way the file itself "
              f"cannot show. Fix before using.")
        for slug, m, p in dropped[:8]:
            print(f"     {slug:<24} markers={m} parsed={p}")
    else:
        print("  every sample-size marker produced a block (no silent drops)")
    if empty:
        print(f"NO BLOCKS PARSED for {len(empty)}: {', '.join(empty[:12])}"
              f"{' ...' if len(empty) > 12 else ''}")
        print("  A silent zero here is the failure mode that matters -- the site "
              "is server-rendered\n  with hashed class names, so a redeploy can "
              "break the regex while every request still 200s.")


if __name__ == "__main__":
    raise SystemExit(main())
