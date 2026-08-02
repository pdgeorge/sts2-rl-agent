"""Tests for the frozen card-text template.

The template decides what every embedding vector means, so the properties that
matter are determinism (same code in, same bytes out) and that mechanically
distinct cards render distinctly.
"""

from __future__ import annotations

import pytest

import sts2_env.cards  # noqa: F401 -- populates the effect registry
import sts2_env.powers  # noqa: F401 -- populates the power class registry
from sts2_env.cards.factory import card_preview
from sts2_env.core.enums import CardId
from sts2_env.embedding.card_text import (
    TEMPLATE_VERSION,
    actions_taken,
    powers_applied,
    render_card_text,
    state_read,
)


def test_template_version_is_set():
    assert isinstance(TEMPLATE_VERSION, int) and TEMPLATE_VERSION >= 1


def test_rendering_is_deterministic():
    """Same code in, same bytes out.

    A template that reordered its own output between runs would silently produce
    two different vector tables from one codebase, and the difference would only
    show up as a model that mysteriously got worse.
    """
    for name in ("BASH", "BARRICADE_CARD", "BODY_SLAM", "WHIRLWIND"):
        card_id = CardId[name]
        first = render_card_text(card_id)
        second = render_card_text(card_id)
        assert first == second


def test_every_card_renders():
    rendered = 0
    for card_id in CardId:
        try:
            card_preview(card_id)
        except Exception:  # noqa: BLE001 -- not every enum member is a real card
            continue
        text = render_card_text(card_id)
        assert text.startswith(card_id.name)
        assert "\n" in text
        rendered += 1
    assert rendered > 500, f"only {rendered} cards rendered"


def test_barricade_keeps_its_identity():
    """The case that motivated code introspection.

    Barricade has no damage, no block and no effect vars. Every structured field
    it owns is shared with any other 3-cost self-target Power, so a template
    built from metadata alone would render it indistinguishably. Its identity is
    the power hook it installs.
    """
    text = render_card_text(CardId.BARRICADE_CARD)
    assert "BARRICADE" in text
    assert "should_clear_block" in text


def test_body_slam_records_that_it_reads_block():
    """Body Slam has no base damage and calls the same helpers as Whirlwind.

    Without the `reads:` line the two are near-identical, and "damage equals your
    block" -- the entire card -- is absent.
    """
    body_slam = render_card_text(CardId.BODY_SLAM)
    whirlwind = render_card_text(CardId.WHIRLWIND)

    assert "reads: block" in body_slam
    assert "reads: block" not in whirlwind
    assert body_slam != whirlwind


def test_applied_powers_are_recovered_from_the_effect():
    assert "VULNERABLE" in powers_applied(CardId.BASH)
    assert "BARRICADE" in powers_applied(CardId.BARRICADE_CARD)


def test_actions_capture_both_methods_and_module_helpers():
    """Damage and block go through module-level helpers, powers through methods.

    Capturing only attribute calls loses half the vocabulary -- which is how Body
    Slam first rendered with no `does:` line at all.
    """
    bash = actions_taken(CardId.BASH)
    assert "apply_power_to" in bash  # a method on CombatState

    body_slam = actions_taken(CardId.BODY_SLAM)
    assert any(a in body_slam for a in ("calculate_damage", "apply_damage"))


def test_plumbing_is_not_reported_as_an_action():
    """`_owner` is called by 435 of 577 effects and means nothing."""
    for card_id in (CardId.BASH, CardId.BARRICADE_CARD, CardId.BODY_SLAM):
        actions = actions_taken(card_id)
        assert "owner" not in actions
        assert "get" not in actions
        assert "range" not in actions


def test_upgrade_delta_is_derived_not_described():
    text = render_card_text(CardId.BASH)
    assert "upgrade:" in text
    assert "8->10" in text  # damage, diffed from the two constructed cards


def test_mechanically_distinct_cards_render_distinctly():
    """A template that collapses distinct cards produces collided vectors, which
    is the failure feature hashing already had."""
    names = [
        "BASH", "BARRICADE_CARD", "BODY_SLAM", "WHIRLWIND",
        "POMMEL_STRIKE", "STRIKE_IRONCLAD", "DEFEND_IRONCLAD",
    ]
    texts = {n: render_card_text(CardId[n]) for n in names}
    assert len(set(texts.values())) == len(names)


@pytest.mark.parametrize("name", ["WOUND", "DAZED"])
def test_unplayable_statuses_still_render_something_identifiable(name):
    """Thin is correct for these -- they do nothing when played -- but they must
    still be distinguishable from each other."""
    text = render_card_text(CardId[name])
    assert name in text
    assert render_card_text(CardId.WOUND) != render_card_text(CardId.DAZED)
