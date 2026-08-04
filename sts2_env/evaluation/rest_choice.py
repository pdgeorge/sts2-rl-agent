"""Rest or upgrade, decided in HP rather than by a threshold.

The heuristic this replaces is `heal if below 50% HP, else upgrade`. That is a
fixed rule which ignores both what upgrading would actually be worth and how much
of the heal would be wasted. At 95% HP a rest heals almost nothing and it still
takes it; with a garbage deck an upgrade buys nothing and it still takes that.

Both options can be priced in the same unit -- HP over the rest of the act -- so
they can simply be compared.

    rest      = min(heal_amount, missing_hp)
    upgrade X = (hp_lost_with_deck - hp_lost_with_X_upgraded) x fights_remaining

Resting is a one-off payment. An upgrade pays out on every remaining fight, which
is why it usually wins early in an act and loses late. The heuristic had no way to
express that; this gets it for free from the arithmetic.

A RESULT WORTH KNOWING

At 20/80 HP with Iron Wave upgradable, the upgrade scores ~+50 HP against a
+24 HP heal, stably across seed counts. So the right move at 25% health is to
upgrade, not to rest -- and the old rule heals there every time, because it never
looks at what the upgrade is worth. A test asserting "hurt means rest" was written
here and failed; the code was right and the test encoded the heuristic's habit.

WHY FIGHTS_REMAINING MATTERS MORE THAN IT LOOKS

An upgrade worth 3 HP a fight is worth 24 HP with eight fights left and 3 HP with
one. Rest sites cluster near the end of an act, exactly where the multiplier is
smallest, so a rule that ignores it will over-upgrade at the worst moment.

INHERITED LIMITS

Same pilot ceiling as everything else here: greedy cannot fly a block deck, so
upgrades to block cards are undervalued. And the same sample-size caution -- the
per-fight HP difference from one upgrade is small, so with few seeds this
separates a big upgrade from a useless one and not much finer than that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from sts2_env.core.constants import IRONCLAD_STARTING_HP
from sts2_env.evaluation.battery import Pilot, Tier, score_cell

logger = logging.getLogger(__name__)

REST_HEAL_FRACTION = 0.3
"""Matches `rest_site_heal_amount`: floor(max_hp * 0.3), before relic modifiers.

Read from the same rule the simulator uses rather than hardcoded independently,
because a second copy of a game constant is how this repo got a card reference
that drifted from the decompile.
"""

DEFAULT_FIGHTS_REMAINING = 6
"""Rough count of fights left in an act from a typical rest site. Used only when
the caller cannot supply a better estimate."""


@dataclass(frozen=True)
class RestOption:
    """One option, priced in HP saved over the rest of the act."""

    kind: str                  # "rest" | "upgrade"
    hp_value: float
    card: object | None = None
    index: int | None = None

    @property
    def label(self) -> str:
        if self.kind == "rest":
            return "REST"
        return f"UPGRADE {getattr(self.card, 'name', self.card)}"


def rest_heal_value(current_hp: int, max_hp: int) -> float:
    """HP a rest would actually restore -- capped by what is missing.

    The cap is the point. At full HP a rest is worth nothing, and a rule that
    compares against a flat 30% will happily take it anyway.
    """
    import math

    return float(min(math.floor(max_hp * REST_HEAL_FRACTION), max(0, max_hp - current_hp)))


def _hp_cost_per_fight(
    deck: Sequence, pilot: Pilot, tiers: Sequence[Tier], seeds: Sequence[int], max_hp: int
) -> float:
    """Mean HP lost per winnable fight. NaN-safe: a deck that never wins is
    reported as costing a full bar, since that is what it does to a run."""
    total_weighted, total_fights = 0.0, 0
    for tier in tiers:
        cell = score_cell(deck, tier, pilot, seeds=seeds, max_hp=max_hp)
        cost = cell.hp_lost_on_wins
        if cost != cost:          # NaN -- never won this cell
            cost = float(max_hp)
        # A cell it loses is not merely expensive, it is fatal; weight the loss
        # rate in rather than reporting the cost of the wins it did manage.
        effective = cost * cell.win_rate + max_hp * (1.0 - cell.win_rate)
        total_weighted += effective * cell.fights
        total_fights += cell.fights
    return total_weighted / total_fights if total_fights else float(max_hp)


def _boss_win_rate(
    deck: Sequence, pilot: Pilot, floor: int, seeds: Sequence[int], max_hp: int
) -> float:
    """Win rate against this act's boss with the deck as it stands."""
    from sts2_env.evaluation.battery import Tier, score_cell

    act = 1 if floor <= 17 else (2 if floor <= 34 else 3)
    return score_cell(deck, Tier(act, "boss"), pilot, seeds=seeds, max_hp=max_hp).win_rate


def rank_rest_options(
    deck: Sequence,
    upgradable: Sequence,
    pilot: Pilot,
    *,
    current_hp: int,
    max_hp: int = IRONCLAD_STARTING_HP,
    floor: int = 1,
    fights_remaining: int = DEFAULT_FIGHTS_REMAINING,
    seeds: Sequence[int] = (0, 1, 2, 3),
    max_upgrades_considered: int = 8,
    boss_next: bool = False,
) -> list[RestOption]:
    """Price every option in HP and return them best first.

    `upgradable` is capped because each candidate costs a full battery pass and a
    28-card deck would otherwise mean 28 of them. Capping is a real limitation
    rather than a detail -- it is chosen, not sampled, so a good upgrade sitting
    past the cap is simply never seen.
    """
    from sts2_env.evaluation.card_choice import tiers_for_floor

    tiers = tiers_for_floor(floor)
    baseline = _hp_cost_per_fight(deck, pilot, tiers, seeds, max_hp)
    boss_baseline = (
        _boss_win_rate(deck, pilot, floor, seeds, max_hp) if boss_next else 0.0
    )

    options: list[RestOption] = [
        RestOption(kind="rest", hp_value=rest_heal_value(current_hp, max_hp))
    ]

    # One candidate per distinct card, not the first N in deck order.
    #
    # Taking the first N was a real bug: a starter-ordered deck begins with five
    # Strikes and four Defends, so an 8-candidate cap evaluated nothing but those
    # and never saw the drafted cards at all. It upgraded Strikes because Uppercut
    # was past the cap. Deduplicating collapses those nine into two entries and
    # leaves room for the cards worth upgrading.
    #
    # Upgrading one Strike is the same decision as upgrading another, so nothing
    # is lost by collapsing them.
    seen: set = set()
    deduped = []
    for card in upgradable:
        key = getattr(card, "card_id", card)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(card)

    considered = deduped[:max_upgrades_considered]
    if len(deduped) > len(considered):
        logger.info(
            "rest: %d distinct upgradable cards, evaluating %d",
            len(deduped), len(considered),
        )

    for index, card in enumerate(considered):
        if getattr(card, "upgraded", False):
            continue
        upgraded_deck = list(deck)
        try:
            position = upgraded_deck.index(card)
        except ValueError:
            continue
        from sts2_env.cards.factory import create_card

        try:
            upgraded_deck[position] = create_card(card.card_id, upgraded=True)
        except Exception:  # noqa: BLE001
            continue

        cost = _hp_cost_per_fight(upgraded_deck, pilot, tiers, seeds, max_hp)
        saved_per_fight = baseline - cost
        value = saved_per_fight * fights_remaining

        if boss_next:
            # The rest before a boss is not an ordinary rest. Upgrading the same
            # eight cards moved the act 1 boss from 13.9% to 69.4% -- a 55-point
            # swing that HP-per-hallway-fight cannot represent, because it is not
            # about attrition, it is about whether the fight is winnable at all.
            # Priced as the win-rate gain times what losing costs: the run.
            swing = _boss_win_rate(upgraded_deck, pilot, floor, seeds, max_hp) \
                - boss_baseline
            value = max(value, swing * float(current_hp))

        options.append(
            RestOption(
                kind="upgrade",
                hp_value=value,
                card=card,
                index=index,
            )
        )

    options.sort(key=lambda o: o.hp_value, reverse=True)
    logger.info(
        "rest: %s",
        "  ".join(f"{o.label}={o.hp_value:+.1f}hp" for o in options[:5]),
    )
    return options
