"""Every difference between the simulator and the game must be declared.

The contract this file enforces:

    a card matches the decompile,  OR  MODELLING_OVERRIDES says why it does not

There is no third option. `SPITE` was the third option for months -- its damage
was correct and derived, its hand-written effect drew a card where the game hits
twice, and the missing `repeat` var was skipped in silence because
`apply_derived_values` only updated keys the factory had already declared. An
intentional difference and a forgotten one were the same thing: an absent key.
"""

from __future__ import annotations

import subprocess
import sys

from sts2_env.cards.derived_values import (
    MODELLING_OVERRIDES,
    declared_difference,
    undeclared_differences,
)
from sts2_env.core.enums import CardId


def test_no_card_differs_from_the_game_without_saying_so():
    """The load-bearing assertion. If this fails, either fix the card or declare
    the difference in MODELLING_OVERRIDES with a reason -- do not weaken this."""
    undeclared = undeclared_differences()
    assert undeclared == {}, (
        "cards carry fewer dynamic vars than the decompile with no declaration:\n"
        + "\n".join(f"  {c.name}: missing {v}" for c, v in undeclared.items())
    )


def test_every_override_says_when_it_becomes_visible():
    """An override with no observable consequence is drift wearing a disguise.

    If nobody can say when the difference shows up in play, nobody checked
    whether it matters -- and the declaration is being used to silence a test
    rather than to record a decision.
    """
    for card_id, override in MODELLING_OVERRIDES.items():
        assert override.reason.strip(), f"{card_id.name} override has no reason"
        assert len(override.observable_when.strip()) > 20, (
            f"{card_id.name} override does not say when the difference is visible"
        )


def test_an_override_must_actually_differ():
    """A declaration for a card that matches the game is stale, and stale
    declarations are how a real difference later hides behind an old excuse.

    Three kinds count, and the third exists because this test found the gap:
    DRUM_OF_BATTLE's data matches the decompile exactly and its BEHAVIOUR does
    not -- the energy is paid from a different hook. A difference that no data
    check can see is the most important kind to write down.
    """
    for card_id, override in MODELLING_OVERRIDES.items():
        assert override.fields or override.omitted_vars or override.behaviour, (
            f"{card_id.name} declares a difference but changes nothing; "
            f"delete the override, or state which fields/vars differ, or "
            f"describe the behavioural difference"
        )


def test_the_generated_doc_is_not_stale():
    """docs/MODELLING_DIFFERENCES.md is generated. A hand-maintained list of
    differences is the same artifact as CARDS_REFERENCE.md, which drifted from
    the decompile while the tests read it as an oracle and stayed green."""
    result = subprocess.run(
        [sys.executable, "scripts/generate_modelling_doc.py", "--check"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_declared_difference_is_the_single_place_to_ask():
    assert declared_difference(CardId.CONFLAGRATION) is not None
    assert declared_difference(CardId.STRIKE_IRONCLAD) is None
