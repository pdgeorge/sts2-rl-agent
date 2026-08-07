"""Which card to upgrade, and which to transform away.

Step 8 of the Phase 5 build plan. Both are the card-reward scorer pointed at
your own deck rather than at an offer.

UPGRADE THE BIGGEST *GAIN*, NOT THE BIGGEST CARD
------------------------------------------------
Some cards go from fine to excellent when upgraded, and the change is not
proportional to how good they already were. Pommel Strike gains a card of draw
on top of its damage; Uppercut doubles the duration of Weak and Vulnerable;
Armaments stops upgrading one card and upgrades your whole hand. Bludgeon goes
from 32 damage to 42, which is a bigger number and a smaller change.

So the score is the difference:

    gain(X) = score(X⁺, deck − X) − score(X, deck − X)

Two details that are easy to get wrong and both matter:

* **The difference, not the absolute.** Ranking by `score(X⁺)` picks whichever
  upgraded card is most valuable, which is usually a card that was already
  strong -- and upgrading it buys little. An already-great card has no headroom.
* **`deck − X`.** The scorer reads deck shape, so leaving X in means it competes
  against its own contribution: a deck's only block card looks less needed
  because the deck has a block card, namely itself.

CARDS WHOSE UPGRADE CHANGES BEHAVIOUR SCORE A DELTA OF ZERO
------------------------------------------------------------
Armaments upgraded has identical cost, damage, block and flags -- the change
lives in an `if (base.IsUpgraded)` branch inside OnPlay, so nothing the
simulator previews differs and the delta is exactly 0.0. It is one of the
strongest upgrades in the game and would rank dead last.

`BEHAVIOURAL_UPGRADES` is those cards, derived by grepping the decompile for
`IsUpgraded` rather than hand-listed, so a patch that adds or removes one is
picked up by on_update.sh like everything else.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Cards whose upgrade changes what the card *does* rather than its numbers.
#: Derived: `grep -rl IsUpgraded decompiled/MegaCrit.Sts2.Core.Models.Cards/`.
#: 26 of 564 -- the rest have an `OnUpgrade` that bumps a value, which the
#: numeric delta already sees.
BEHAVIOURAL_UPGRADES = frozenset({
    "ARMAMENTS", "BEGONE", "CASCADE", "CHARGE", "COMPACT", "DARKNESS", "DIRGE",
    "ENLIGHTENMENT", "GUARDS", "HIDDEN_DAGGERS", "JACKPOT", "KNIFE_TRAP",
    "LARGESSE", "MALAISE", "MANIFEST_AUTHORITY", "MULTI_CAST", "PRIMAL_FORCE",
    "QUASAR", "REAVE", "SPINNER", "SPLASH", "STOKE", "STORM_OF_STEEL",
    "TEMPEST", "TESLA_COIL", "TRUE_GRIT",
})

BEHAVIOURAL_UPGRADE_BONUS = 1.5
"""What a behavioural upgrade is worth, on the card-quality scale.

An ordinary playable card scores 1.5-2.0 and a typical numeric upgrade gains a
few tenths, so this puts "the card starts doing something else" clearly above
"the card does more damage" without swamping the ranking.

Flat rather than per-card on purpose. The 26 have not been individually
assessed, and a hand-tuned table of 26 guesses would look more precise than it
is. If one of them turns out to matter more than the rest, measure it.
"""


def _card_id_name(card: Any) -> str | None:
    from sts2_env.bridge.card_quality import _card_id

    card_id = _card_id(card)
    return card_id.name if card_id is not None else None


def _without(deck: list[Any], index: int) -> list[Any]:
    return [c for i, c in enumerate(deck) if i != index]


def upgrade_gain(card: Any, deck: list[Any], index: int, direction: Any | None = None) -> float:
    """How much upgrading this copy would improve the deck."""
    from sts2_env.bridge.card_quality import score_card_for_deck

    name = _card_id_name(card)
    if name is None:
        return 0.0

    rest = _without(deck, index)
    base = dict(card) if isinstance(card, dict) else {"id": name}
    upgraded = dict(base)
    upgraded["upgraded"] = True

    numeric = (
        score_card_for_deck(upgraded, rest, direction)
        - score_card_for_deck(base, rest, direction)
    )
    behavioural = (
        BEHAVIOURAL_UPGRADE_BONUS
        if name.removesuffix("_CARD") in BEHAVIOURAL_UPGRADES
        else 0.0
    )
    return numeric + behavioural


def pick_upgrade_target(deck: list[Any], direction: Any | None = None) -> int | None:
    """Index of the card worth upgrading most, or None for an empty deck."""
    if not deck:
        return None
    gains = [(upgrade_gain(c, deck, i, direction), i) for i, c in enumerate(deck)]
    best_gain, best_index = max(gains, key=lambda g: (g[0], -g[1]))
    logger.debug("upgrade target: index %d, gain %.2f", best_index, best_gain)
    return best_index


def pick_transform_target(deck: list[Any], direction: Any | None = None) -> int | None:
    """Index of the card contributing least -- the "least meta card".

    `argmin` of the same scorer, with the same `deck − X` correction. Curses are
    excluded rather than preferred: transforming a curse yields another curse,
    sometimes a worse one, so it is the one card class never worth the roll.
    """
    from sts2_env.bridge.agent_runner import _is_curse
    from sts2_env.bridge.card_quality import score_card_for_deck

    candidates = [
        (score_card_for_deck(c, _without(deck, i), direction), i)
        for i, c in enumerate(deck)
        if not _is_curse(c)
    ]
    if not candidates:
        return None
    worst_score, worst_index = min(candidates, key=lambda s: (s[0], s[1]))
    logger.debug("transform target: index %d, score %.2f", worst_index, worst_score)
    return worst_index
