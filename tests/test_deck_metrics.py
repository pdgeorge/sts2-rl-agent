"""Tests for deck-level density metrics.

These are pure functions of a decklist, so unlike the rest of `evaluation/` the
expected values can be written down by hand and checked. That is the point of
them: no pilot, no seeds, nothing that can drift.
"""

from __future__ import annotations

from math import comb

from sts2_env.cards.factory import create_card
from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.core.enums import CardId
from sts2_env.evaluation.deck_metrics import (
    BLOCK_DENSITY_MAX,
    BLOCK_DENSITY_MIN,
    block_density,
    block_density_penalty,
    cards_drawn_by,
    cycle_time,
    p_flooded,
    p_no_block,
    upgrade_density,
)


def _mk(names):
    return [create_card(CardId[n]) for n in names]


def _zero_block_attacks(count: int):
    """`count` real cards that provide no block, discovered from the registry.

    Chosen rather than hardcoded because hardcoding is how this repo got
    `deck_features.py`, whose hand-written keyword list has 20 of 45 strings
    matching no card in this game. Two names guessed for an earlier draft of this
    very test (CLEAVE, HEAVY_BLADE) do not exist in STS2 either. Deriving the
    list from the game cannot make that mistake.
    """
    import re
    from pathlib import Path

    source = Path("sts2_env/cards/ironclad.py").read_text()
    picked = []
    for name in sorted(set(re.findall(r"@register_effect\(CardId\.(\w+)\)", source))):
        try:
            card = create_card(CardId[name])
        except Exception:  # noqa: BLE001
            continue
        if (card.base_block or 0) == 0 and (card.base_damage or 0) > 0:
            picked.append(card)
        if len(picked) == count:
            break
    assert len(picked) == count, f"only found {len(picked)} zero-block attacks"
    return picked


def test_starter_deck_densities_are_the_obvious_ones():
    deck = create_ironclad_starter_deck()
    assert len(deck) == 10
    assert block_density(deck) == 0.4       # 4 Defends of 10
    assert upgrade_density(deck) == 0.0


def test_the_same_cards_are_a_smaller_share_of_a_bigger_deck():
    """The reason this module exists. A COUNT of defensive cards is not a stable
    target -- four Defends is 40% of a starter deck and 16% of a 25-card one, so
    'two defensive cards by act 2' silently weakens as the run gets deadlier."""
    starter = create_ironclad_starter_deck()
    grown = starter + _zero_block_attacks(15)

    assert block_density(starter) == 0.4
    assert block_density(grown) < 0.20
    # Same four Defends in both.
    assert sum(1 for c in starter if (c.base_block or 0) > 0) == \
           sum(1 for c in grown if (c.base_block or 0) > 0)


def test_p_no_block_is_exact_hypergeometric_not_binomial():
    """Act 1 decks are small, which is exactly where the binomial approximation
    overstates the risk -- and where the number gets used."""
    deck = create_ironclad_starter_deck()          # 10 cards, 4 block
    expected = comb(6, 5) / comb(10, 5)            # choose 5 from the 6 non-block
    assert p_no_block(deck) == expected


def test_both_walls_of_the_band_are_real():
    """More block is not simply better: a flooded hand cannot kill anything, so
    the penalty has to be signed rather than a distance."""
    thin = create_ironclad_starter_deck() + _zero_block_attacks(12)
    thick = create_ironclad_starter_deck() + _mk(
        ["TAUNT", "TAUNT", "SHRUG_IT_OFF", "SHRUG_IT_OFF", "BLOOD_WALL", "EVIL_EYE"]
    )

    assert block_density(thin) < BLOCK_DENSITY_MIN
    assert block_density(thick) > BLOCK_DENSITY_MAX
    assert block_density_penalty(thin) < 0        # too little
    assert block_density_penalty(thick) > 0       # too much
    assert p_no_block(thin) > 0.20
    assert p_flooded(thick) > 0.20


def test_a_deck_inside_the_band_is_not_penalised():
    deck = create_ironclad_starter_deck() + _mk(["UPPERCUT", "ANGER", "IRON_WAVE"])
    assert BLOCK_DENSITY_MIN <= block_density(deck) <= BLOCK_DENSITY_MAX
    assert block_density_penalty(deck) == 0.0


def test_draw_is_read_from_the_key_the_game_actually_uses():
    """`cards`, not `draw`. The pilot's own draw term read `draw` for its whole
    life and no card in this game has ever carried that key, so it returned zero
    always. Pinned here because the same mistake is one typo away."""
    assert cards_drawn_by(_mk(["BATTLE_TRANCE"])) == 3
    assert cards_drawn_by(_mk(["POMMEL_STRIKE"])) == 1
    assert cards_drawn_by(create_ironclad_starter_deck()) == 0


def test_draw_shortens_the_cycle_and_bloat_lengthens_it():
    """The article's sharpest point, made checkable: a card that draws exactly
    one card does not beat skipping on cycle time."""
    base = create_ironclad_starter_deck()
    plus_draw = base + _mk(["BATTLE_TRANCE"])
    plus_filler = base + _mk(["RAMPAGE"])
    plus_cantrip = base + _mk(["POMMEL_STRIKE"])

    assert cycle_time(plus_draw) < cycle_time(base)
    assert cycle_time(plus_filler) > cycle_time(base)
    assert cycle_time(plus_cantrip) == cycle_time(base)


def test_empty_deck_does_not_divide_by_zero():
    assert block_density([]) == 0.0
    assert upgrade_density([]) == 0.0
    assert cycle_time([]) == 0.0
    assert p_no_block([]) == 0.0
    assert p_flooded([]) == 0.0
