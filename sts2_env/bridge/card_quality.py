"""Is this card worth adding to the deck?

The live agent used to answer with a four-word rule -- prefer a Power, else an
Attack, else a Skill -- and take the first card of that type on offer. Thirty
live runs of the result are in `output/live_journal.jsonl`: 183 card rewards,
none refused, and 61% of every card played was a basic Strike or Defend. It took
BLIGHT_STRIKE (8 damage for 1) over SUNDER (26 for 3) because Blight Strike was
listed first.

It does not have to guess. The bridge sends the card's id, and the simulator
knows what that card IS -- rarity, type, cost, damage, block -- for all 577 of
them. Asking it is both better than a keyword rule and free to keep correct: when
Mega Crit rebalances a card, the answer changes with the decompile, without
anyone editing a table here.

WHAT THIS DELIBERATELY IS NOT

It is not a tier list, and the numbers below are not claims about the metagame.
It is a floor: never take a curse, prefer 26 damage to 8, notice when a deck has
no way to defend itself. A real card rating -- measured by simulating the fights
that are coming, or supplied from outside -- belongs in CARD_RATINGS, which
overrides everything here.
"""

from __future__ import annotations

import logging
from typing import Any

from sts2_env.core.enums import CardId, CardRarity, CardType

logger = logging.getLogger(__name__)

# Hand-supplied ratings, id -> score, overriding everything computed below.
# This is where an external card rating goes. Empty by default: an empty table
# is honest about knowing nothing, where a half-remembered one would be wrong in
# ways nobody could see.
CARD_RATINGS: dict[str, float] = {}

# Never worth adding, whatever else the numbers say.
_REFUSE_TYPES = frozenset({CardType.CURSE, CardType.STATUS})
_REFUSE_RARITIES = frozenset({CardRarity.CURSE, CardRarity.STATUS, CardRarity.BASIC})

_RARITY_SCORE = {
    CardRarity.RARE: 2.0,
    CardRarity.UNCOMMON: 1.0,
    CardRarity.COMMON: 0.0,
}

# A deck needs a way to not die as well as a way to win. Below this share of
# cards that produce block, a block card is worth more than its numbers say.
BLOCK_DENSITY_TARGET = 0.25
# Scaling is how a deck beats a boss rather than a hallway. A deck with none is
# a deck that loses on floor 17, which is where all six boss attempts ended.
SCALING_BONUS = 1.5
BLOCK_NEED_BONUS = 1.0

# Below this, taking nothing is better -- when the game allows it. See
# `agent_runner._pick_card_reward_index` and the note there about the mod.
SKIP_THRESHOLD = 0.0


def _card_id(card: Any) -> CardId | None:
    name = card.get("id") if isinstance(card, dict) else card
    if not name:
        return None
    try:
        return CardId[str(name).rstrip("+")]
    except KeyError:
        return None


def _metadata(card_id: CardId, upgraded: bool = False):
    """Static metadata and a preview instance, at the requested upgrade state.

    `upgraded` used to be absent entirely, so every card was scored as its base
    version. That was tolerable while this only ranked card rewards -- the three
    on offer are rarely upgraded -- and became load-bearing the moment
    `upgrade_targets` asked "how much better would this card be upgraded?", a
    question whose answer was structurally 0.0 for all 564 numeric upgrades.
    """
    from sts2_env.cards.factory import card_metadata, card_preview

    if upgraded:
        from sts2_env.cards.factory import create_card

        return card_metadata(card_id), create_card(card_id, upgraded=True)
    return card_metadata(card_id), card_preview(card_id)


def _is_upgraded(card: Any) -> bool:
    if isinstance(card, dict):
        return bool(card.get("upgraded"))
    return bool(getattr(card, "upgraded", False))


def infer_character(deck: list[Any]) -> str | None:
    """Which character this deck belongs to, read off its starter cards.

    The card-reward state does not say who is playing, but a deck always holds
    STRIKE_<CHARACTER> and DEFEND_<CHARACTER>, and those names are exactly the
    ten basics across the five characters.
    """
    for entry in deck or []:
        card_id = _card_id(entry)
        if card_id is None:
            continue
        name = card_id.name.upper()
        if name.startswith(("STRIKE_", "DEFEND_")):
            return name.split("_", 1)[1]
    return None


def deck_shape(deck: list[Any]) -> dict[str, float]:
    """What the deck already has, so a card is judged against a need.

    A second Inflame is worth less than the first; a block card is worth more to
    a deck holding none. Judging cards in isolation is how a deck ends up as
    twenty attacks and no defence.
    """
    total = 0
    blockers = attacks = powers = 0
    for entry in deck or []:
        card_id = _card_id(entry)
        if card_id is None:
            continue
        try:
            meta, preview = _metadata(card_id)
        except Exception:
            continue
        total += 1
        if (preview.base_block or 0) > 0:
            blockers += 1
        if meta.card_type == CardType.ATTACK:
            attacks += 1
        if meta.card_type == CardType.POWER:
            powers += 1

    if not total:
        return {"size": 0, "block_density": 0.0, "attack_density": 0.0, "powers": 0}
    return {
        "size": total,
        "block_density": blockers / total,
        "attack_density": attacks / total,
        "powers": powers,
    }


def score_card(card: Any, deck: list[Any] | None = None) -> float:
    """How much this card is worth to this deck. Higher is better.

    Negative means the deck is better off without it.
    """
    card_id = _card_id(card)
    if card_id is None:
        # An id this build does not know. Neutral rather than refused: a new card
        # after a game update should not be treated as a curse.
        return 0.0

    if card_id.name in CARD_RATINGS:
        return CARD_RATINGS[card_id.name]

    try:
        meta, preview = _metadata(card_id, _is_upgraded(card))
    except Exception:
        logger.debug("No metadata for %s", card_id, exc_info=True)
        return 0.0

    if meta.card_type in _REFUSE_TYPES or meta.rarity in _REFUSE_RARITIES:
        return -10.0

    shape = deck_shape(deck or [])
    score = _RARITY_SCORE.get(meta.rarity, 0.0)

    cost = max(preview.cost, 0) or 1
    damage = preview.base_damage or 0
    block = preview.base_block or 0

    # Output per energy, which is what actually competes for a turn. A 3-cost
    # card is not three times better for costing three times as much.
    score += (damage + block) / (cost * 10.0)

    if meta.card_type == CardType.POWER:
        # Scaling wins boss fights and nothing else in a starter deck does.
        # Discounted once the deck already has some, because the second copy
        # has fewer turns left to pay off.
        score += SCALING_BONUS / (1 + shape["powers"])

    if block > 0 and shape["size"] and shape["block_density"] < BLOCK_DENSITY_TARGET:
        score += BLOCK_NEED_BONUS

    return score


ARCHETYPE_WEIGHT = 0.6
"""How much the deck's plan is allowed to move a card's score.

At 0.6 a perfectly on-plan card is worth 1.6x its raw quality and a perfectly
off-plan one 0.4x, so direction reorders cards of similar quality without ever
letting a weak on-theme card beat a strong off-theme one outright. That ordering
matters: the reference implementation this design came from picks purely by
similarity and its own description calls it "greedy similarity-based selection
... not an optimization solver", which would happily draft a coherent bad deck.
"""


ABSTAINED_QUALITY_SCALE = 2.0
"""What fit alone is worth when card quality abstains.

Calibrated against the quality scale rather than chosen: an ordinary playable
card scores 1.5-2.0, so a perfectly on-archetype card the quality scorer cannot
read lands in the same band rather than dominating or disappearing."""


def score_card_for_deck(
    card: Any,
    deck: list[Any] | None = None,
    direction: Any | None = None,
) -> float:
    """Quality, scaled by how well the card fits the deck's plan.

    `score_card` answers "is this a good card", which is most of the job and all
    of it before a plan exists. `direction` answers "does it belong in THIS
    deck" -- the thing card quality cannot see, and the reason a vulnerability
    payoff should outrank raw damage in a deck built on Vulnerable.

    Three regimes, because quality means three different things:

    * **negative** -- a curse or status. Refused however well it fits; letting
      fit soften a negative would make a well-themed curse look takeable.
    * **positive** -- scaled by fit. Direction reorders cards of similar
      quality without letting a weak on-theme card beat a strong off-theme one.
    * **zero** -- *no opinion*, not "worthless". `score_card` reads base damage
      and block, so a card whose effect is pure logic scores nothing: Body Slam
      ("damage equal to your Block") and Entrench ("double your Block") both
      come out 0.0 despite being core block-scaling cards. Multiplying that by
      fit leaves 0 and the deck would never take its own payoffs. So when
      quality abstains, fit decides.
    """
    quality = score_card(card, deck)
    if direction is None or quality < 0:
        return quality

    card_id = _card_id(card)
    if card_id is None:
        return quality
    try:
        fit = direction.fit(card_id.name)
    except Exception:  # noqa: BLE001 - a missing embedding is not a crash
        logger.debug("no archetype fit for %s", card_id, exc_info=True)
        return quality
    if quality == 0.0:
        return ABSTAINED_QUALITY_SCALE * fit
    return quality * (1.0 + ARCHETYPE_WEIGHT * fit)


def rank_cards(
    cards: list[Any],
    deck: list[Any] | None = None,
    direction: Any | None = None,
) -> list[tuple[float, int, Any]]:
    """(score, index, card), best first. Ties keep the offered order."""
    scored = [
        (score_card_for_deck(card, deck, direction), index, card)
        for index, card in enumerate(cards)
    ]
    return sorted(scored, key=lambda entry: (-entry[0], entry[1]))
