"""Score a deck by playing it, instead of guessing what it is worth.

The agent has never had any signal about deck quality. Its only feedback on a
card reward is "did I eventually die, hundreds of steps later", which is the
credit-assignment problem this project keeps running into. Live, that shows up as
a policy that picked slot 2 in 19 of 28 card rewards and slot 0 in none of them.

Nothing here is learned. A deck is scored by playing it against a fixed set of
encounters and reporting what happened, which is ground truth from the simulator
rather than an opinion about card text.

THE GRID

Rows are where a run has got to, columns are what kind of fight it is:

              weak    normal    elite    boss
    act 1       .        .        .        .
    act 2       .        .        .        .
    act 3       .        .        .        .

Every cell is scored on TWO numbers, and the second is what stops the battery
saturating:

    win_rate       discriminates at the hard end
    hp_lost        discriminates at the easy end

In act 1 hallway fights a decent deck wins ~100% of the time, so win rate says
nothing -- but "wins costing 4 HP" against "wins costing 16 HP" is an enormous
difference, and it is the one that decides whether you reach the act 1 boss
alive. Reporting only a win rate throws that away.

WHAT IS HELD FIXED

Full HP, no potions, deck's own relics. That deliberately separates two things
that are otherwise impossible to tell apart: how good the deck is, and how much
trouble the run is currently in. Current HP belongs in the run-level model on
top of this, not baked into a deck score.

THE PILOT DECIDES WHAT "GOOD" MEANS

Whoever plays the test fights defines the measurement. A greedy-damage pilot
cannot pilot a block deck, so it will score block cards as worthless -- not
because they are, but because it never converts block into survival. This is the
single largest bias in the whole design and it is not fixable by adding
encounters or seeds.

It is left pluggable and named for that reason. `scripts/eval_combat_search.py`
measured greedy-damage at 73.3% +/- 8.1% on act 1, statistically level with a
trained PPO model, so greedy is a reasonable *fast* pilot for bulk work. Anything
that ranks decks for real should spot-check with a stronger pilot and be
suspicious of archetypes where the two disagree.

SEEDS ARE FIXED

Every deck faces the same encounters from the same seeds, so comparisons are
paired and differences are not seed luck. Hold a second seed set back for
validating anything the first one selects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

from sts2_env.core.constants import IRONCLAD_STARTING_HP
from sts2_env.core.rng import Rng

Pilot = Callable[[object], int]
"""Chooses an action given a CombatState. See sts2_env.evaluation.pilots."""

DEFAULT_RELICS: tuple[str, ...] = ("BURNING_BLOOD",)
"""The character's starting relic, because a deck without it is not the deck
being played.

Burning Blood heals 6 after every won combat. Omitting it is not a rounding
error: a gauntlet without it survived 4 fights where the real thing survives 5,
and every "the deck dies almost immediately" result today was measured without
it.
"""

ACT_OPENING_WEAK_FIGHTS = 3
"""How many weak encounters an act opens with before the normal pool.

Ignoring this was the larger of two measurement errors. A gauntlet run entirely
on act1_normal said a starter deck survives 2 fights; a real act 1 opens with
weak fights at a third of the cost (11.4 HP against 34.5), and the same deck
survives 5. The battery was measuring a situation that does not occur.
"""


@dataclass(frozen=True)
class Tier:
    """One cell of the grid."""

    act: int
    kind: str          # weak | normal | elite | boss

    @property
    def name(self) -> str:
        return f"act{self.act}_{self.kind}"


TIERS: tuple[Tier, ...] = tuple(
    Tier(act, kind)
    for act in (1, 2, 3)
    for kind in ("weak", "normal", "elite", "boss")
)

_ENCOUNTER_ATTR = {
    "weak": "WEAK_ENCOUNTERS",
    "normal": "NORMAL_ENCOUNTERS",
    "elite": "ELITE_ENCOUNTERS",
    "boss": "BOSS_ENCOUNTERS",
}


def encounters_for(tier: Tier) -> list:
    """The encounter setups in a cell, in a stable order."""
    import importlib

    module = importlib.import_module(f"sts2_env.encounters.act{tier.act}")
    return list(getattr(module, _ENCOUNTER_ATTR[tier.kind], []))


@dataclass(frozen=True)
class FightResult:
    won: bool
    hp_lost: int
    turns: int


@dataclass(frozen=True)
class CellScore:
    """What a deck did in one cell."""

    tier: str
    fights: int
    wins: int
    hp_lost_on_wins: float
    mean_turns: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.fights if self.fights else 0.0

    @property
    def sem(self) -> float:
        import math

        p, n = self.win_rate, self.fights
        return math.sqrt(p * (1.0 - p) / n) if n else 0.0


def play_one(
    deck: Sequence,
    encounter_setup,
    seed: int,
    pilot: Pilot,
    *,
    max_hp: int = IRONCLAD_STARTING_HP,
    starting_hp: int | None = None,
    relics: Sequence[str] | None = None,
    character_id: str = "Ironclad",
    max_steps: int = 600,
) -> FightResult:
    """Play a single fight and report what happened.

    The deck is cloned per fight, because CardInstance carries mutable per-combat
    state and a deck reused across fights would silently accumulate it.
    """
    from sts2_env.cards.base import CardInstance
    from sts2_env.core.combat import CombatState
    from sts2_env.gym_env.action_space import apply_action, get_action_mask

    fresh_deck = [
        card.clone(index) if isinstance(card, CardInstance) else card
        for index, card in enumerate(deck)
    ]

    combat = CombatState(
        player_hp=max_hp if starting_hp is None else starting_hp,
        player_max_hp=max_hp,
        deck=fresh_deck,
        relics=list(relics) if relics else list(DEFAULT_RELICS),
        rng_seed=seed,
        character_id=character_id,
    )
    encounter_setup(combat, Rng(seed))
    combat.start_combat()

    for _ in range(max_steps):
        if combat.is_over:
            break
        if not get_action_mask(combat).any():
            break
        if not apply_action(combat, pilot(combat)):
            # The mask advertised something the engine refuses. Continuing would
            # spin without advancing the turn counter.
            break

    began = max_hp if starting_hp is None else starting_hp
    return FightResult(
        won=bool(combat.player_won),
        hp_lost=max(0, began - combat.player.current_hp),
        turns=int(combat.turn_count),
    )


def score_cell(
    deck: Sequence,
    tier: Tier,
    pilot: Pilot,
    *,
    seeds: Iterable[int] = range(5),
    max_hp: int = IRONCLAD_STARTING_HP,
) -> CellScore:
    """Play every encounter in a cell, once per seed."""
    setups = encounters_for(tier)
    seeds = list(seeds)
    results: list[FightResult] = []
    for setup in setups:
        for seed in seeds:
            results.append(play_one(deck, setup, seed, pilot, max_hp=max_hp))

    wins = [r for r in results if r.won]
    return CellScore(
        tier=tier.name,
        fights=len(results),
        wins=len(wins),
        hp_lost_on_wins=float(np.mean([r.hp_lost for r in wins])) if wins else float("nan"),
        mean_turns=float(np.mean([r.turns for r in results])) if results else 0.0,
    )


def score_deck(
    deck: Sequence,
    pilot: Pilot,
    *,
    tiers: Sequence[Tier] = TIERS,
    seeds: Iterable[int] = range(5),
    max_hp: int = IRONCLAD_STARTING_HP,
) -> dict[str, CellScore]:
    """The profile: every cell, so the shape of a deck is visible.

    The shape is the answer. A deck that is cheap in act 1 and collapses in act 2
    and one that costs more early but scales look identical in a mean, and want
    opposite decisions.
    """
    seeds = list(seeds)
    return {
        tier.name: score_cell(deck, tier, pilot, seeds=seeds, max_hp=max_hp)
        for tier in tiers
    }


# --- gauntlet ---------------------------------------------------------------


def gauntlet_length(
    deck: Sequence,
    tier: Tier,
    pilot: Pilot,
    *,
    seed: int = 0,
    max_hp: int = IRONCLAD_STARTING_HP,
    limit: int = 60,
) -> int:
    """How many fights in a row this deck survives, from full HP until death.

    `score_cell` resets to full HP for every fight, which quietly assumes HP is
    free. It is not -- it is the currency a whole run is paid for in. A deck that
    wins every fight at 60 HP a time scores well there and dies on its second
    fight in an actual run.

    Carrying HP forward turns the measurement into the thing that matters: how
    deep this deck goes. Defence, damage and speed all cash out in the same
    number, and a single catastrophic fight ends the count rather than being
    averaged away.

    Encounters cycle through the tier so a deck cannot be tuned to one fight.
    """
    encounters = encounters_for(tier)
    if not encounters:
        return 0

    hp = max_hp
    survived = 0
    for index in range(limit):
        setup = encounters[index % len(encounters)]
        result = play_one(
            deck, setup, seed * 1000 + index, pilot,
            max_hp=max_hp, starting_hp=hp,
        )
        if not result.won:
            return survived
        survived += 1
        hp -= result.hp_lost
        if hp <= 0:
            return survived
    return survived


def mean_gauntlet(
    deck: Sequence,
    tier: Tier,
    pilot: Pilot,
    *,
    seeds: Iterable[int] = range(6),
    max_hp: int = IRONCLAD_STARTING_HP,
) -> float:
    """Mean fights survived across seeds -- the headline deck-strength number."""
    lengths = [
        gauntlet_length(deck, tier, pilot, seed=seed, max_hp=max_hp)
        for seed in seeds
    ]
    return float(np.mean(lengths)) if lengths else 0.0


def act_sequence(tier: Tier, length: int) -> list:
    """Encounters in the order an act actually presents them.

    Weak fights first, then the normal pool. A gauntlet that skips this measures
    a run dropped into mid-act with a starting deck, which never happens.
    """
    if tier.kind != "normal":
        pool = encounters_for(tier)
        return [pool[i % len(pool)] for i in range(length)] if pool else []

    weak = encounters_for(Tier(tier.act, "weak"))
    normal = encounters_for(tier)
    out = []
    for index in range(length):
        if index < ACT_OPENING_WEAK_FIGHTS and weak:
            out.append(weak[index % len(weak)])
        elif normal:
            out.append(normal[(index - ACT_OPENING_WEAK_FIGHTS) % len(normal)])
    return out


def gauntlet_hp(
    deck: Sequence,
    tier: Tier,
    pilot: Pilot,
    *,
    seed: int = 0,
    max_hp: int = IRONCLAD_STARTING_HP,
    fights: int = 5,
    relics: Sequence[str] | None = None,
) -> float:
    """HP left after `fights` consecutive fights, without healing. 0 if it dies.

    This is the scoring basis the decision modules should use, and the reason is
    the flaw it fixes: `score_cell` restarts every fight at full HP, so it cannot
    value HP preservation at all. Ending a fight at 5 HP and at 50 HP score the
    same there, which is why defensive cards measured as worthless -- the
    measurement handed the deck a free full heal after every fight.

    Continuous rather than a fight count, because counts land on 0, 1 or 2 for
    the decks that exist today and cannot rank anything.
    """
    sequence = act_sequence(tier, fights)
    if not sequence:
        return float(max_hp)

    hp = float(max_hp)
    for index, setup in enumerate(sequence):
        result = play_one(
            deck, setup, seed * 1000 + index, pilot,
            max_hp=max_hp, starting_hp=int(hp), relics=relics,
        )
        if not result.won:
            return 0.0
        hp -= result.hp_lost
        if hp <= 0:
            return 0.0
        # Burning Blood and friends fire after a won combat, so the heal belongs
        # between fights rather than being folded into the fight itself.
        hp = min(float(max_hp), hp + _post_combat_heal(relics))
    return hp


def _post_combat_heal(relics: Sequence[str] | None) -> float:
    """HP restored after a won fight by the relics in play."""
    names = {str(r).upper() for r in (relics if relics is not None else DEFAULT_RELICS)}
    return 6.0 if "BURNING_BLOOD" in names else 0.0


def mean_gauntlet_hp(
    deck: Sequence,
    tier: Tier,
    pilot: Pilot,
    *,
    seeds: Iterable[int] = range(6),
    max_hp: int = IRONCLAD_STARTING_HP,
    fights: int = 5,
    relics: Sequence[str] | None = None,
) -> float:
    """Mean HP surviving a gauntlet, across seeds. Higher is a better deck."""
    values = [
        gauntlet_hp(deck, tier, pilot, seed=seed, max_hp=max_hp,
                    fights=fights, relics=relics)
        for seed in seeds
    ]
    return float(np.mean(values)) if values else 0.0
