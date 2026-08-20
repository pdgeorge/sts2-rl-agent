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

import json
import logging
from pathlib import Path
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
    """The CardId for a bridge card, or None if this build really lacks it.

    Resolved rather than looked up directly. 68 of 600 CardId members are
    spelled `X_CARD` while the bridge sends `X`, so a raw lookup returned None
    for Barricade, Corruption, Colossus, Blur, Buffer and 63 others -- and this
    function decides CARD REWARDS. An unresolvable id cannot be scored, so every
    one of those was being judged blind at the reward screen.
    """
    name = card.get("id") if isinstance(card, dict) else card
    if not name:
        return None
    from sts2_env.search.situation import resolve_card_id

    return resolve_card_id(str(name))


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



# ---------------------------------------------------------------------------
# The community prior
# ---------------------------------------------------------------------------

#: Act 1 card-reward winrate DELTA per card, from sts2.untapped.gg, keyed by
#: CardId name. Loaded lazily and cached; missing file means the term is simply
#: unavailable and every card scores as it did before.
_PRIOR_PATH = Path(__file__).resolve().parents[2] / "data/untapped/act1_card_reward_prior.json"
_PRIOR: dict[str, dict] | None = None

#: Divides the raw winrate delta. The deltas run -18 to +16 while `score_card`
#: mostly lives in 0..5, so raw addition would not be a prior, it would BE the
#: score. At 10 a +14 card gains 1.4 -- enough to break the tie blocks the
#: formula produces (OFFERING, INFLAME and DEMON_FORM all score exactly 4.200)
#: without overturning a genuine quality gap.
PRIOR_SCALE = 10.0

#: The most the prior may move a card, either way. Same reasoning as
#: `EvalWeights.powers_cap`, which exists because an uncapped power term once
#: scored a sleeping elite at -1.2 and made the searcher refuse to attack: a
#: prior is a tiebreaker between cards of similar quality, not a replacement
#: for reading what the card does.
PRIOR_CAP = 1.5

#: Below this many observations the delta is not used. Untapped suppresses its
#: own numbers at small samples (an em-dash rather than a figure), so this only
#: catches the ones it publishes thinly.
PRIOR_MIN_N = 500


def _prior_table() -> dict[str, dict]:
    global _PRIOR
    if _PRIOR is None:
        try:
            _PRIOR = json.loads(_PRIOR_PATH.read_text(encoding="utf-8"))["cards"]
        except Exception:  # noqa: BLE001 - a missing prior is not a crash
            logger.debug("no card prior at %s", _PRIOR_PATH, exc_info=True)
            _PRIOR = {}
    return _PRIOR



def _active_prior_weight() -> float:
    """How much the community prior counts, from the active policy.

    Read through `policy_config` rather than a module constant, so an A/B is
    two JSON files and neither arm can see the other's value. `PHASE_TWO.md`
    3.1 records the alternative: 400 runs whose baseline arm did the opposite
    of its name. Defaults to 0.0, which makes `v001` bit-identical to the
    behaviour before the prior existed.
    """
    try:
        from sts2_env.policy_config import active_policy
        return float(getattr(active_policy(), "card_prior_weight", 0.0) or 0.0)
    except Exception:  # noqa: BLE001 - scoring must not depend on config loading
        return 0.0


def card_prior_bonus(card_id_name: str, weight: float) -> float:
    """The prior's contribution to a card's score, scaled, capped and gated.

    Zero weight -- the shipped default -- returns 0.0 without touching the
    table, so `v001` is bit-identical to the behaviour before this existed.
    """
    if not weight:
        return 0.0
    entry = _prior_table().get(card_id_name)
    if not entry or entry.get("n", 0) < PRIOR_MIN_N:
        return 0.0
    raw = weight * float(entry["wr"]) / PRIOR_SCALE
    return max(-PRIOR_CAP, min(PRIOR_CAP, raw))


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

    # The prior is added to the DERIVED score below rather than returned here,
    # so the deck-context terms still apply. Putting it in CARD_RATINGS instead
    # would have been the obvious place and is wrong: that lookup returns
    # immediately, switching off the block-density bonus, the diminishing
    # return on a second Power and everything else that reads the deck.
    prior = card_prior_bonus(card_id.name, _active_prior_weight())

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

    score += _effect_value(preview) / (cost * 10.0)

    # Cost, valued on its own rather than only as a divisor. Barricade's upgrade
    # is cost 3 -> 2 and nothing else, and dividing by cost cannot see that when
    # there is no damage or block to divide: 0/3 and 0/2 are both zero. A
    # cheaper card is better whatever else it does.
    score += (CHEAPNESS_REFERENCE_COST - cost) * CHEAPNESS_VALUE

    # Last, and capped, so it reorders cards this formula scored alike rather
    # than deciding on its own. Measured reason it is here at all: over 1,478
    # act 1 card rewards the agent took cards averaging -0.28 winrate delta
    # where the best offered averaged +3.37 and a RANDOM pick averaged -0.79.
    return score + prior


CHEAPNESS_REFERENCE_COST = 3.0
CHEAPNESS_VALUE = 0.15
"""What a point of energy cost is worth by itself.

Small: cost already divides the output terms, so this only has to cover the
case where there is nothing to divide. Set so a 1-cost card starts 0.3 ahead of
a 3-cost one, which is roughly the gap between them at equal effect."""


#: What a unit of each effect is worth, on the same scale as a point of damage.
#:
#: Grounded in what the cards actually carry: these are the `effect_vars` keys
#: the simulator produces, surveyed across all 577 cards, weighted by rough
#: parity with damage. Drawing a card is worth several points of damage; a turn
#: of Vulnerable multiplies everything that follows; energy buys a whole extra
#: card. Anything unlisted scores nothing, which is honest -- an unweighted key
#: is one nobody has thought about, not one worth zero.
#:
#: This exists because score_card read `(damage + block) / cost` and nothing
#: else, so Pommel Strike's extra card of draw was worth +0.10 and Uppercut's
#: doubled Weak and Vulnerable duration was worth exactly 0.00. Both were cited
#: as cards that go from fine to excellent when upgraded, and the scorer could
#: not see either upgrade.
_EFFECT_VALUE = {
    "cards": 6.0,        # a drawn card is roughly a card's worth of tempo
    "energy": 8.0,       # buys a card AND keeps the one in hand
    "damage": 1.0,
    "block": 1.0,
    "power": 5.0,        # a duration, mostly -- Uppercut's Weak and Vulnerable
    "vulnerable": 5.0,   # multiplies every attack that lands during it
    "weak": 4.0,
    "strength": 7.0,     # permanent, and compounds over a fight
    "dexterity": 5.0,
    "repeat": 4.0,
    "extra_damage": 2.0,
    "increase": 3.0,
    "poison_power": 4.0,
    "focus_power": 5.0,
    "strength_loss": 3.0,
    "hp_loss": -2.0,     # a real cost, not a benefit
    "doom": 2.0,
    "summon": 3.0,
    "forge": 2.0,
    "stars": 2.0,
    "gold": 0.05,
}


def quality_is_uninformative(card: Any) -> bool:
    """True when `score_card` has nothing real to go on for this card.

    A card whose effect is pure logic -- Body Slam ("damage equal to your
    Block"), Entrench ("double your Block") -- has no base damage, no base block
    and no scored effect vars. Whatever number `score_card` returns for it comes
    from rarity and cost, which say nothing about whether the deck wants it.

    Stated explicitly rather than inferred from `score == 0.0`. That test worked
    only by accident and stopped working the moment cheapness started
    contributing, silently dropping Body Slam from 1.10 to 0.40 in a deck built
    around it.
    """
    card_id = _card_id(card)
    if card_id is None:
        return True
    try:
        _, preview = _metadata(card_id, _is_upgraded(card))
    except Exception:
        return True
    return (
        not (preview.base_damage or 0)
        and not (preview.base_block or 0)
        and _effect_value(preview) == 0.0
    )


def _effect_value(preview: Any) -> float:
    """What a card's effects are worth beyond its damage and block.

    Deliberately reads `effect_vars` rather than trying to interpret behaviour:
    those are the numbers the simulator already derives from the decompile, so
    they stay correct across a rebalance without anyone editing a table here.
    """
    total = 0.0
    for name, amount in (getattr(preview, "effect_vars", None) or {}).items():
        weight = _EFFECT_VALUE.get(name)
        if weight is None:
            continue
        try:
            total += weight * float(amount)
        except (TypeError, ValueError):
            continue
    if getattr(preview, "exhausts", False):
        total -= 3.0        # once per combat is a real limit
    if getattr(preview, "is_innate", False):
        total += 3.0
    if getattr(preview, "is_retain", False):
        total += 2.0
    if getattr(preview, "is_ethereal", False):
        total -= 4.0
    return total


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
    * **uninformative** -- no damage, no block, no scored effects, so whatever
      number came back reflects rarity and cost rather than power. Body Slam
      ("damage equal to your Block") and Entrench ("double your Block") are both
      like this despite being core block-scaling cards, and a deck built around
      them would never take its own payoffs. When quality abstains, fit decides.
      Detected by `quality_is_uninformative` rather than by `score == 0.0`,
      which worked by accident until cheapness started contributing.
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
    if quality_is_uninformative(card):
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
