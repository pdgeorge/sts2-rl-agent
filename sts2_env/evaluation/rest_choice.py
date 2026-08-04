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
from sts2_env.core.enums import CardId
from sts2_env.evaluation.battery import Pilot, Tier, score_cell

logger = logging.getLogger(__name__)

REST_HEAL_FRACTION = 0.3
"""Matches `rest_site_heal_amount`: floor(max_hp * 0.3), before relic modifiers.

Read from the same rule the simulator uses rather than hardcoded independently,
because a second copy of a game constant is how this repo got a card reference
that drifted from the decompile.
"""

LOW_PRIORITY_UPGRADES = frozenset(
    c for c in CardId
    if c.name.startswith("STRIKE_") or c.name.startswith("DEFEND_")
)
"""Every character's basic Strike and Defend, ranked strictly below resting.

Matched by prefix rather than listed, so the Silent and Defect get the same rule
without a second table to keep in step. Prefix, not substring: PERFECTED_STRIKE
and POMMEL_STRIKE are real cards and upgrading them is a real decision.

"Strike and Defend are the lowest priority to upgrade. If Strike and Defend are
your only cards to upgrade, rest instead." A Strike+ is +3 damage on a card the
deck is trying to stop drawing; the smith is better spent on a card that changes
what the deck can do, and if there is no such card the HP is worth more.
"""

DEFAULT_FIGHTS_REMAINING = 6
"""Fallback when the caller cannot supply a floor. Prefer `fights_remaining_at`."""

TOTAL_FLOORS = 48
FIGHT_FRACTION = 0.55
"""A run is ~48 floors and a little over half of them are fights.

WHY THIS REPLACES A FLAT 6

A flat 6 meant "the rest of this act", and it was never passed from the bridge --
every live rest decision used it. An upgrade is PERMANENT: taken on floor 8 it
pays out for every fight left in the RUN, not just the act. Valuing it over six
fights while a rest is valued over one made upgrades look about a third as good
as they are.

That is self-fulfilling, which is the part worth stating plainly. The model
expects to die inside the act, so it prices upgrades for a short run, so it
under-upgrades, so it dies inside the act. Measured over seven live runs the
agent reached a 7% upgrade density against Baalorlord's 33-50% for winning runs,
and the three runs with ZERO upgrades averaged floor 7.3 against 12.3 for the
four that had any.

Playing for the run we want rather than the run we expect is the standard
roguelike answer, and here it is also the only one that escapes the loop.
"""


UPGRADE_SHORTFALL_POINTS = 20.0
"""HP-equivalent value of closing the whole gap to a boss-ready deck.

Calibrated against both ends, not chosen freely.

At 60 it overturned the low-HP rest fix from an hour earlier -- a fully-short
deck at 20/80 hp got +28 of push and smithed at 25% health, where P(surviving
the next fight) is 46%. At 30 it still won there by 2 points, which is inside
this model's precision and on the wrong side of it. At 20:

    20/80 (46% survive)   REST +37   IRON_WAVE +31     rests
    32/80 (86% survive)   IRON_WAVE +58   REST +19     smiths
    68/80 (92% survive)   IRON_WAVE +79   REST +18     smiths

Banking a permanent upgrade you have a 46% chance of living to use is worse than
the heal that takes those odds to 92%.

The constant is a judgement call; the term's existence is not. rest_choice
prices upgrades one at a time, so a deck four real upgrades short of boss-ready
never presents a single urgent-looking upgrade -- see the 78%-vs-19% table in
deck_metrics.MEANINGFUL_UPGRADES_TARGET.
"""


def fights_remaining_at(floor: int) -> int:
    """Fights left in the RUN from this floor, not in the act."""
    return max(1, round((TOTAL_FLOORS - max(0, floor)) * FIGHT_FRACTION))


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


# Measured survival curve: P(surviving 2 hallway fights then an elite) as a
# function of starting HP fraction, 30 seeds per point, v3 pilot, 14-card deck.
#
#   30%  0%     40%  3%     50%  3%     60% 17%
#   70% 27%     80% 67%     90% 73%    100% 93%
#
# HP is strongly non-linear and the payoff is concentrated between 50% and 80%.
# Below half you are at 3% and eight more HP buys nothing; the steepest stretch is
# 70% -> 80%. This is pdgeorge's "get above 50% if you can", measured -- and the
# real cliff sits nearer 60-80% than at 50 exactly.
_SURVIVAL_CURVE: tuple[tuple[float, float], ...] = (
    (0.30, 0.00), (0.40, 0.03), (0.50, 0.03), (0.60, 0.17),
    (0.70, 0.27), (0.80, 0.67), (0.90, 0.73), (1.00, 0.93),
)


# Measured survival curve: P(winning THE NEXT hallway fight) by starting HP.
# 40 seeds per encounter, greedy pilot, on the real 20-card deck from the run
# that exposed the need for it.
#
#   10%  6%    20% 24%    30% 67%    40% 86%    49% 91%    69% 92%   100% 92%
#
# The curve above (`_SURVIVAL_CURVE`) asks about three fights, and at the bottom
# it is flat because three fights are hopeless from 9 hp AND from 18 hp. That
# flatness priced "almost certainly dead" the same as "certainly dead", and the
# agent upgraded PERFECTED_STRIKE at 9/91 hp with a fight next -- rest was worth
# 6.8 against the upgrade's 18.0. A greed call, straight out of the arithmetic.
#
# One fight is the question that actually matters at low HP, and it is enormous
# there: healing 9 -> 36 moves survival from 6% to 86%. It saturates by ~50%,
# which is exactly where the three-fight curve takes over.
_NEXT_FIGHT_CURVE: tuple[tuple[float, float], ...] = (
    (0.10, 0.06), (0.20, 0.24), (0.30, 0.67), (0.40, 0.86),
    (0.49, 0.91), (0.69, 0.92), (1.00, 0.92),
)


def _interpolate(curve: tuple[tuple[float, float], ...], x: float) -> float:
    x = max(0.0, min(1.0, x))
    if x <= curve[0][0]:
        return curve[0][1]
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x <= x1:
            span = x1 - x0
            return y0 + (y1 - y0) * ((x - x0) / span if span else 0.0)
    return curve[-1][1]


def survive_next_fight(hp_fraction: float) -> float:
    """P(winning the next hallway fight) at this HP fraction."""
    return _interpolate(_NEXT_FIGHT_CURVE, hp_fraction)


def survival_at(hp_fraction: float) -> float:
    """Interpolated survival probability at an HP fraction."""
    hp_fraction = max(0.0, min(1.0, hp_fraction))
    if hp_fraction <= _SURVIVAL_CURVE[0][0]:
        return _SURVIVAL_CURVE[0][1]
    for (x0, y0), (x1, y1) in zip(_SURVIVAL_CURVE, _SURVIVAL_CURVE[1:]):
        if hp_fraction <= x1:
            span = x1 - x0
            return y0 + (y1 - y0) * ((hp_fraction - x0) / span if span else 0.0)
    return _SURVIVAL_CURVE[-1][1]


def rest_heal_value(current_hp: int, max_hp: int) -> float:
    """What a rest is worth, in HP-equivalent, weighted by survival gained.

    Was `min(heal, missing_hp)` -- flat HP. That priced these identically when
    they are not remotely the same decision:

        24 -> 48 hp    0% -> 17% survival    +17 points
        40 -> 64 hp    3% -> 67% survival    +64 points
        72 -> 80 hp   73% -> 93% survival    +20 points

    Same 24 HP restored, four times the value in the middle. Flat HP also cannot
    express that healing from 24 to 48 leaves you still likely to die, which is
    the whole content of "make sure you are over half".

    Scaled back into HP units so it stays comparable with upgrade values, which
    are priced in HP saved per fight.
    """
    import math

    restored = float(min(math.floor(max_hp * REST_HEAL_FRACTION),
                         max(0, max_hp - current_hp)))
    if restored <= 0 or max_hp <= 0:
        return 0.0

    frac_before = current_hp / max_hp
    frac_after = min(1.0, (current_hp + restored) / max_hp)

    # Two horizons, and whichever is more binding wins.
    #
    # IMMEDIATE: will I survive the next fight. Dominates at low HP and is huge
    # there -- 9 -> 36 hp is 6% -> 86%. Priced against the whole remaining run,
    # because dying now costs all of it.
    immediate = max(0.0, survive_next_fight(frac_after)
                    - survive_next_fight(frac_before)) * float(max_hp)

    # HORIZON: will I survive the next few fights. Dominates once the immediate
    # question is settled -- the one-fight curve is flat above ~50% while this
    # one is still climbing steeply.
    horizon = max(0.0, survival_at(frac_after) - survival_at(frac_before)) * float(max_hp)

    # Floor at a quarter of the raw heal so a rest is never worth nothing while
    # the HP is still real.
    return max(restored * 0.25, immediate, horizon)


POOR_UPGRADE_SAMPLE = 800
"""Smith observations needed before untapped may veto an upgrade.

Below this the delta is mostly rounding: the site quantises to whole percent, so
a -1% drawn from 200 smiths is not evidence of anything.
"""

POOR_UPGRADE_MIN_TAKE_RATE = 20
"""Percent of players who take the upgrade, below which the delta is unusable.

This is the guard against reading a selection effect as a causal one, and it is
not hypothetical -- it changes the answer on a real card:

    IRON_WAVE   run -2%   act -9%   4,600 seen   upgraded  7%
    ARMAMENTS   run +1%   act  0%   7,400 seen   upgraded 66%

Iron Wave looks like a catastrophic upgrade on a healthy 4,600-smith sample. But
it is chosen 7% of the time, and the runs where someone upgrades it are
disproportionately runs where nothing better was on offer -- weak decks, already
losing. The delta measures the situation, not the upgrade.

Sample size does not fix this and more data makes it worse, not better: the bias
is in which runs enter the sample. A high take rate is the only cheap defence,
because a card upgraded by two thirds of players is being upgraded from strong
positions and weak ones alike.
"""


def _upgrading_is_measured_bad(card, floor: int) -> bool:
    """Does untapped say upgrading this card loses runs?

    The smith column is a different question from the draft column and they
    disagree in useful ways. Fiend Fire is roughly neutral to draft in act 1
    (-1% run, thin sample) and clearly bad to UPGRADE there: -6% act winrate and
    -2% run over 1,000 smiths. A single "card quality" number would have averaged
    those into a shrug.

    Held below resting rather than merely penalised, for the same reason as
    Strike and Defend: a smith spent on a card that makes the run worse is worse
    than the HP, and there is no exchange rate between HP-per-fight and run
    winrate honest enough to write down.
    """
    from sts2_env.evaluation.card_priors import card_stats

    stats = card_stats(card, decision="smith", floor=floor)
    if not stats or (stats.get("offered") or 0) < POOR_UPGRADE_SAMPLE:
        return False
    if (stats.get("taken_pct") or 0) < POOR_UPGRADE_MIN_TAKE_RATE:
        return False
    delta = stats.get("run_winrate")
    if delta is None:
        delta = stats.get("act_winrate")
    return delta is not None and delta < 0


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
    fights_remaining: int | None = None,
    seeds: Sequence[int] = (0, 1, 2, 3),
    max_upgrades_considered: int = 8,
    boss_next: bool = False,
    time_budget_s: float | None = None,
) -> list[RestOption]:
    """Price every option in HP and return them best first.

    `upgradable` is capped because each candidate costs a full battery pass and a
    28-card deck would otherwise mean 28 of them. Capping is a real limitation
    rather than a detail -- it is chosen, not sampled, so a good upgrade sitting
    past the cap is simply never seen.
    """
    from sts2_env.evaluation.card_choice import tiers_for_floor

    from sts2_env.evaluation.deck_metrics import (
        MEANINGFUL_UPGRADES_TARGET,
        upgrade_shortfall,
    )

    if fights_remaining is None:
        fights_remaining = fights_remaining_at(floor)
    shortfall = upgrade_shortfall(deck)
    tiers = tiers_for_floor(floor)
    baseline = _hp_cost_per_fight(deck, pilot, tiers, seeds, max_hp)
    boss_baseline = (
        _boss_win_rate(deck, pilot, floor, seeds, max_hp) if boss_next else 0.0
    )

    rest_value = rest_heal_value(current_hp, max_hp)
    options: list[RestOption] = [RestOption(kind="rest", hp_value=rest_value)]

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
    # `index` must stay a position in the caller's `upgradable`, NOT in this
    # deduped list. Returning the deduped position was a live bug: the bridge maps
    # a winner back with `indexes[option.index]`, so picking the third distinct
    # card upgraded the third *offered* card -- a Defend, in a starter-ordered
    # deck. It ranked right and upgraded something else.
    seen: set = set()
    deduped: list[tuple[int, object]] = []
    for position, card in enumerate(upgradable):
        key = getattr(card, "card_id", card)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((position, card))

    considered = deduped[:max_upgrades_considered]
    if len(deduped) > len(considered):
        logger.info(
            "rest: %d distinct upgradable cards, evaluating %d",
            len(deduped), len(considered),
        )

    import time as _time

    started = _time.monotonic()
    evaluated = 0
    for index, card in considered:
        # A live decision that overruns the mod's 30-second agent timeout does
        # not degrade -- the mod ends the RUN. That happened at 23:21:39 on a
        # real rest site: 33 seconds to rank 8 upgrades, then `run_complete`.
        #
        # So the budget is a hard stop that returns the best answer found so
        # far, rather than a faster-but-still-unbounded computation. Speed alone
        # is not a guarantee; a bigger deck always exists.
        if time_budget_s is not None and _time.monotonic() - started > time_budget_s:
            logger.info(
                "rest: time budget %.0fs spent after %d of %d upgrades; "
                "ranking what was measured", time_budget_s, evaluated, len(considered),
            )
            break
        if getattr(card, "upgraded", False):
            continue
        evaluated += 1
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
        # An upgrade only pays if you live to use it.
        #
        # That is the term this was missing, and it is what makes an upgrade
        # comparable to a rest without pretending they are the same kind of
        # thing. A rest is banked immediately; an upgrade is a claim on fights
        # you have not survived yet, so it is discounted by the odds of getting
        # there. At 9/91 hp those odds are 6%, which is why upgrading there is a
        # greed call however good the card is.
        #
        # Converting the upgrade's TOTAL future saving through the survival curve
        # was tried first and is wrong in a way worth recording: 3 hp/fight over
        # 19 fights became "+57 hp right now", which read as 6% -> 92% survival
        # and put the upgrade back on top at 9 hp. You do not get all 19 fights
        # of value in the next fight.
        alive = survive_next_fight(current_hp / max_hp if max_hp else 0.0)
        value = saved_per_fight * fights_remaining * alive

        # A deck below the healthy upgrade floor gets pushed toward the smith,
        # regardless of what any single upgrade measures at.
        #
        # rest_choice prices upgrades ONE AT A TIME and has no notion of the deck
        # as a whole being short of boss-ready -- which is the state the agent
        # dies in. Measured against the act 1 boss, 30 seeds:
        #
        #     4 upgrades on drafted cards   78% win
        #     4 upgrades on Strikes/Defends 19% win
        #     0 upgrades                     7% win
        #
        # Individually none of those four looks urgent, which is exactly why a
        # per-card price never finds them.
        #
        # Discounted by the same survival term, because a deck-level shortfall is
        # still a claim on fights you have not survived. At 9 hp it contributes
        # almost nothing, which is what stops this undoing the greed fix.
        # Scaled by the fights left as well as the odds of reaching them. An
        # upgrade with no fights remaining is worth nothing however short of
        # boss-ready the deck is, and without this the shortfall term paid out
        # on the last floor of the run.
        horizon = min(1.0, fights_remaining / float(MEANINGFUL_UPGRADES_TARGET))
        value += shortfall * UPGRADE_SHORTFALL_POINTS * alive * horizon

        if card.card_id in LOW_PRIORITY_UPGRADES or _upgrading_is_measured_bad(card, floor):
            # pdgeorge's rule, and it is a play-knowledge prior rather than a
            # measured one: an upgraded Defend is +3 block once a fight, an
            # upgraded Uppercut changes what the deck can beat. Measurement at
            # high HP happily ranks Defend+ above resting because the heal it is
            # compared against is nearly wasted -- which is true arithmetic and
            # the wrong move. Held strictly below rest so a starter upgrade is
            # only ever taken when there is genuinely nothing else on offer.
            value = min(value, rest_value - 1.0)

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
