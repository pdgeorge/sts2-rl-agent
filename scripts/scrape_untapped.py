"""Pull per-card win rates from sts2.untapped.gg into a local table.

    python scripts/scrape_untapped.py --character ironclad
    python scripts/scrape_untapped.py --character ironclad --only TAUNT,ARMAMENTS

WHY THIS EXISTS

The battery answers "is this card good?" from eight simulated seeds. untapped
answers it from tens of thousands of real runs, split by the decision that was
actually being made -- card reward, shop, and smith are three different questions
about the same card and it reports all three.

Measured on the case that started this: act 1 card reward, run winrate

    TAUNT        offered 15,000    picked 60%    +1%
    FIEND_FIRE   offered    970    picked 67%    -1%

while our battery had FIEND_FIRE at +0.636 against TAUNT's +0.395. The battery is
not going to win that argument, so it should stop having it, and price deck fit
instead -- which is the thing untapped cannot see.

WHAT THE NUMBERS ARE, AND ARE NOT

A run winrate delta is conditioned on players who *picked* the card. A card
favoured by strong players, or one only taken when the deck already supports it,
looks better than it is. These are a prior, not ground truth, and the offer count
is carried alongside every row so a 970-sample number is not read as confidently
as a 27,000-sample one.

PATCH SURVIVABILITY

Re-run after a patch. The table is keyed on card slug rather than enum position,
so it survives the reordering that has broken this repo's card identity before,
and a card the scrape does not know simply has no prior rather than a wrong one.

Polite by construction: one request at a time with a delay, a real user agent,
and it skips anything already in the output file unless --refresh is passed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://sts2.untapped.gg/en/cards"
USER_AGENT = "sts2-rl-agent/0.1 (personal research; one request per second)"
DEFAULT_OUTPUT = Path("data/untapped_cards.json")
DELAY_SECONDS = 1.0

# Parsed off the stable substrings only. The CSS module names carry a build hash
# (`StatCell-module-scss-module__hPkuua__act`) which changes on any site rebuild,
# so matching the whole class name would rot silently and produce empty stats
# that look like "this card has no data".
_SECTION = re.compile(r'__title">[^<]*<span[^>]*>(Card Reward|Shop|Smith)</span>')
_CELL = re.compile(
    r'__act">In Act (\d+)</span>'
    r'<span class="[^"]*__offered">(?:offered|seen)\s+([\d,]+) times</span>'
)
_RATE = re.compile(r'__(?:pickRate|upgradeRate)"><span>(\w+)</span><strong>(\d+)<')
_WINRATE = re.compile(r'<span>(Act|Run) Winrate</span>.*?<strong>([+-]?\d+)%</strong>')


def _slugs(card_name: str) -> list[str]:
    """CardId name -> candidate untapped URL slugs, in order to try.

    FIEND_FIRE -> fiend-fire. Several of this repo's enum members carry a `_CARD`
    suffix that the site does not use (DEMON_FORM_CARD -> demon-form), which is
    why this returns candidates rather than one answer: 20 of 86 Ironclad cards
    were silently skipped as 404s before the suffix was stripped.
    """
    base = card_name.lower().replace("_", "-")
    candidates = [base]
    if base.endswith("-card"):
        candidates.insert(0, base[: -len("-card")])
    return candidates


def _fetch(slug: str, timeout: int = 30) -> str | None:
    request = urllib.request.Request(
        f"{BASE}/{slug}", headers={"User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def parse_card(html: str) -> dict:
    """Pull the three stat sections out of a card page.

    Returns ``{"card_reward": {act: {...}}, "shop": {...}, "smith": {...}}``.
    An absent section is absent rather than zeroed -- a card nobody has ever been
    offered in act 3 is not a card with a 0% win rate there.
    """
    # Each section appears twice: an empty shimmer placeholder rendered first,
    # then the real one streamed in later and swapped by a `$RC(...)` script. So
    # "keep the first occurrence" keeps the placeholder and silently yields a
    # card with no stats -- indistinguishable from a card nobody has played.
    # Keep whichever copy actually contains stat cells instead.
    sections: dict[str, str] = {}
    marks = [(m.start(), m.group(1)) for m in _SECTION.finditer(html)]
    for index, (start, name) in enumerate(marks):
        key = {"Card Reward": "card_reward", "Shop": "shop", "Smith": "smith"}[name]
        stop = marks[index + 1][0] if index + 1 < len(marks) else len(html)
        blob = html[start:stop]
        if len(_CELL.findall(blob)) > len(_CELL.findall(sections.get(key, ""))):
            sections[key] = blob

    parsed: dict[str, dict] = {}
    for key, blob in sections.items():
        cells = list(_CELL.finditer(blob))
        acts: dict[str, dict] = {}
        for index, cell in enumerate(cells):
            stop = cells[index + 1].start() if index + 1 < len(cells) else len(blob)
            body = blob[cell.start():stop]

            entry: dict = {"offered": int(cell.group(2).replace(",", ""))}
            rate = _RATE.search(body)
            if rate:
                entry["taken_pct"] = int(rate.group(2))
            for kind, value in _WINRATE.findall(body):
                entry[f"{kind.lower()}_winrate"] = int(value)
            acts[cell.group(1)] = entry
        if acts:
            parsed[key] = acts
    return parsed


def _character_cards(character: str) -> list[str]:
    """Every CardId this character's module registers an effect for."""
    from sts2_env.cards import registry  # noqa: F401  (populates the registry)

    module = __import__(f"sts2_env.cards.{character}", fromlist=["*"])
    source = Path(module.__file__).read_text()
    names = re.findall(r"@register_effect\(CardId\.(\w+)\)", source)
    return sorted(set(names))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--character", default="ironclad")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--only", help="comma-separated CardId names")
    parser.add_argument("--refresh", action="store_true",
                        help="re-fetch cards already present in the output")
    parser.add_argument("--delay", type=float, default=DELAY_SECONDS)
    args = parser.parse_args()

    if args.only:
        cards = [n.strip().upper() for n in args.only.split(",") if n.strip()]
    else:
        cards = _character_cards(args.character)

    table: dict = {}
    if args.output.is_file():
        table = json.loads(args.output.read_text())
    cards_table = table.setdefault("cards", {})

    missing, scraped = [], 0
    for index, name in enumerate(cards, 1):
        if name in cards_table and not args.refresh:
            continue
        stats, used, reached = None, None, False
        for slug in _slugs(name):
            print(f"[{index}/{len(cards)}] {name} -> {slug}", file=sys.stderr, flush=True)
            html = _fetch(slug)
            time.sleep(args.delay)
            if html is None:
                continue
            reached = True
            parsed = parse_card(html)
            if parsed:
                stats, used = parsed, slug
                break

        if stats:
            cards_table[name] = {"slug": used, **stats}
            scraped += 1
        elif reached:
            # Reached the page and understood nothing: either the card really has
            # no recorded games, or the markup moved. Distinguished from a 404 so
            # a site rebuild does not look like a pile of new cards.
            missing.append(f"{name} (page ok, no stats parsed)")
        else:
            missing.append(name)

    table["character"] = args.character
    table["source"] = BASE
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")

    print(f"\nwrote {len(cards_table)} cards to {args.output} ({scraped} new)",
          file=sys.stderr)
    if missing:
        print(f"no stats for {len(missing)}: {', '.join(missing[:20])}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
