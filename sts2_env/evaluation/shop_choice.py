"""What to buy, priced in HP like everything else.

The heuristic this replaces is a fixed priority list: relic, then card, then
potion, then removal, then leave. It buys the first thing it can afford in that
order, which means it buys a bad card over a card removal every time, and it has
no way to decline a purchase that makes the deck worse.

Two of the five options can be measured directly with the battery, and they are
the two that change the deck:

    buy card X     = (hp_cost_now - hp_cost_with_X)    x fights_remaining
    remove card X  = (hp_cost_now - hp_cost_without_X) x fights_remaining

Card removal is the one the priority list handles worst. Removing a Strike from a
bloated deck, or a curse from any deck, is often the strongest purchase in the
shop, and the old order reaches it fourth -- after buying a relic, a card, and a
potion it may not want.

WHAT IS NOT MEASURED, AND WHY IT FALLS BACK

Relics and potions are not evaluated. The battery scores decks, and a relic's
value depends on interactions the pilot mostly cannot exercise. Guessing a number
for them would put a fabricated quantity next to two measured ones and let it win
comparisons it has not earned, so instead they are left to the existing priority
list and this module simply declines to have an opinion.

GOLD IS TREATED AS FREE

Anything affordable is considered at face value. Gold has no terminal value in a
run, so spending it on something useful is strictly better than dying with it --
but saving for a *later* shop has real option value, and this ignores that. It
will therefore overspend early. Worth revisiting only if runs start reaching
enough shops for it to matter; today most end before the second one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from sts2_env.core.constants import IRONCLAD_STARTING_HP
from sts2_env.evaluation.battery import Pilot
from sts2_env.evaluation.rest_choice import DEFAULT_FIGHTS_REMAINING, _hp_cost_per_fight

logger = logging.getLogger(__name__)

SEEDS_REQUIRED_NOTE = """Eight, and measured rather than chosen.

Ranking "buy Clumsy" against leaving, on a starter deck:

    seeds= 3   buy_curse +8.7   remove_strike  -0.5   -> buys the curse
    seeds= 8   buy_curse -4.4   remove_strike -10.7   -> leaves
    seeds=16   buy_curse -3.3   remove_strike  -6.1   -> leaves

Three seeds is confidently wrong, as in card_choice. It is worse here than there
because the per-fight difference is multiplied by fights_remaining, so the noise
is amplified along with the signal.

Also worth keeping: removing a Strike from a ten-card deck scores NEGATIVE.
Removal is good in bloated decks, not lean ones, and the priority list it
replaces would have taken it."""

BUDGET_NOTE = """Six seeds and four candidates per side, not eight and six.

Measured on a realistic 28-card deck, a shop decision took 14.5s against the
mod's 30s agent timeout -- thin enough that a larger deck or a fuller shop could
cross it, and a timed-out decision leaves the game waiting on an answer that
never comes. Two turbo sessions crashed with no log, and this was one of the
suspects.

Six seeds still ranks a curse below leaving (the eight-seed table below holds at
six); three does not. Four candidates a side is a real cap: it is chosen, not
sampled, so a good buy in slot five is never seen."""

MIN_WORTHWHILE_HP = 2.0
"""Below this, a purchase is noise rather than an improvement.

The measurement cannot resolve differences smaller than a couple of HP per act at
the sample sizes used live, so acting on one is acting on nothing.
"""


@dataclass(frozen=True)
class ShopOption:
    kind: str                 # "buy_card" | "remove_card" | "leave"
    hp_value: float
    index: int | None = None
    card: object | None = None

    @property
    def label(self) -> str:
        if self.kind == "leave":
            return "LEAVE"
        name = getattr(self.card, "name", self.card)
        return f"{self.kind.upper()} {name}"


def rank_shop_options(
    deck: Sequence,
    *,
    purchasable: Sequence = (),
    removable: Sequence = (),
    pilot: Pilot,
    floor: int = 1,
    max_hp: int = IRONCLAD_STARTING_HP,
    fights_remaining: int = DEFAULT_FIGHTS_REMAINING,
    seeds: Sequence[int] = (0, 1, 2, 3, 4, 5),
    max_considered: int = 4,
) -> list[ShopOption]:
    """Price the measurable shop options in HP, best first.

    `purchasable` and `removable` are `(index, card)` pairs so the caller can map
    a winner back to the action it has to send.

    Both lists are capped: each candidate costs a battery pass, and a shop with
    five cards plus a 30-card deck to remove from would otherwise mean 35 of them.
    """
    from sts2_env.evaluation.card_choice import tiers_for_floor

    tiers = tiers_for_floor(floor)
    baseline = _hp_cost_per_fight(deck, pilot, tiers, seeds, max_hp)

    options: list[ShopOption] = [ShopOption(kind="leave", hp_value=0.0)]

    for index, card in list(purchasable)[:max_considered]:
        cost = _hp_cost_per_fight(list(deck) + [card], pilot, tiers, seeds, max_hp)
        options.append(ShopOption(
            kind="buy_card",
            hp_value=(baseline - cost) * fights_remaining,
            index=index, card=card,
        ))

    for index, card in list(removable)[:max_considered]:
        remaining = [c for c in deck if c is not card]
        if len(remaining) == len(deck) or not remaining:
            continue
        cost = _hp_cost_per_fight(remaining, pilot, tiers, seeds, max_hp)
        options.append(ShopOption(
            kind="remove_card",
            hp_value=(baseline - cost) * fights_remaining,
            index=index, card=card,
        ))

    options.sort(key=lambda o: o.hp_value, reverse=True)
    logger.info(
        "shop: %s",
        "  ".join(f"{o.label}={o.hp_value:+.1f}hp" for o in options[:5]),
    )

    # Leaving beats any purchase that is within noise of doing nothing.
    if options[0].kind != "leave" and options[0].hp_value < MIN_WORTHWHILE_HP:
        logger.info(
            "shop: best purchase is worth %.1f hp, below the %.1f threshold; "
            "leaving instead", options[0].hp_value, MIN_WORTHWHILE_HP,
        )
        return [o for o in options if o.kind == "leave"] + options

    return options
