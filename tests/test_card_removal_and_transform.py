"""Which card the agent gives up, on the three screens it cannot tell apart.

`RlCardSelector` implements a hook the *game* calls -- its own docstring lists
"deck upgrade, deck transform, deck enchant, hand selection, and various other
card selection prompts" -- and the payload it sends carries only cards,
min_select and max_select. So the screen's purpose is not in the state and has
to come from the runner, which knows it just bought a shop removal.

Getting this wrong is not cosmetic. Before this, `remove_card` sat 4th in the
shop priority (so it did fire) and the selector sorted basics *last*, which is
right for upgrades and exactly inverted for removal: the agent kept its Strikes
and removed its Bludgeon.
"""

from __future__ import annotations

from sts2_env.bridge.agent_runner import (
    _deck_has_curse,
    _is_curse,
    _pick_card_select_indexes,
    _pick_shop_option,
)


def _card(card_id, card_type="Attack", **extra):
    return {"index": extra.pop("index", 0), "id": card_id, "type": card_type, **extra}


def _select_state(cards, min_select=1, max_select=1):
    return {
        "type": "card_select",
        "cards": [dict(c, index=i) for i, c in enumerate(cards)],
        "min_select": min_select,
        "max_select": max_select,
    }


# -- identifying a curse ----------------------------------------------------

def test_a_curse_is_recognised_from_the_type_the_mod_sends():
    assert _is_curse(_card("REGRET", "Curse"))
    assert not _is_curse(_card("BLUDGEON", "Attack"))


def test_a_curse_is_still_recognised_without_the_type_field():
    """Older payloads carry no `type`; the id still gives it away."""
    assert _is_curse({"id": "CURSE_OF_THE_BELL"})


# -- removal: curses only ---------------------------------------------------

def test_removal_takes_the_curse_and_not_the_good_card():
    state = _select_state([
        _card("BLUDGEON", "Attack"),
        _card("REGRET", "Curse"),
        _card("DEFEND_IRONCLAD", "Skill"),
    ])
    assert _pick_card_select_indexes(state, removing=True) == [1]


def test_removal_never_takes_a_strike_from_a_strike_deck():
    """The old behaviour. Strikes are the point of a strike-synergy deck."""
    state = _select_state([
        _card("STRIKE_IRONCLAD", "Attack"),
        _card("STRIKE_IRONCLAD", "Attack"),
        _card("PERFECTED_STRIKE", "Attack"),
    ])
    assert _pick_card_select_indexes(state, removing=True) == []


def test_removal_declines_rather_than_removing_a_defend():
    """A Defend may be the deck's only block, and this screen cannot say."""
    state = _select_state([
        _card("DEFEND_IRONCLAD", "Skill"),
        _card("IRON_WAVE", "Attack"),
    ])
    assert _pick_card_select_indexes(state, removing=True) == []


# -- everything else: never a curse ----------------------------------------

def test_transform_avoids_curses():
    """Transforming a curse yields another curse, sometimes a worse one."""
    state = _select_state([
        _card("REGRET", "Curse"),
        _card("IRON_WAVE", "Attack"),
    ])
    assert _pick_card_select_indexes(state) == [1]


def test_upgrade_still_prefers_a_real_card_over_a_basic():
    state = _select_state([
        _card("STRIKE_IRONCLAD", "Attack"),
        _card("BLUDGEON", "Attack"),
    ])
    assert _pick_card_select_indexes(state) == [1]


def test_a_curse_is_ranked_below_even_a_basic_when_not_removing():
    state = _select_state([
        _card("REGRET", "Curse"),
        _card("STRIKE_IRONCLAD", "Attack"),
    ])
    assert _pick_card_select_indexes(state) == [1]


# -- the shop: do not buy a removal with nothing to remove ------------------

def _shop_state(deck, options):
    return {
        "type": "shop",
        "deck": deck,
        "options": [dict(o, index=i, enabled=True) for i, o in enumerate(options)],
    }


def test_deck_has_curse_reads_the_deck():
    assert _deck_has_curse({"deck": [_card("REGRET", "Curse")]})
    assert not _deck_has_curse({"deck": [_card("BLUDGEON", "Attack")]})


def test_the_shop_skips_card_removal_when_the_deck_has_no_curse():
    state = _shop_state(
        deck=[_card("STRIKE_IRONCLAD", "Attack")],
        options=[{"action": "remove_card"}, {"action": "leave_shop"}],
    )
    assert _pick_shop_option(state) == 1, "bought a removal with no curse to remove"


def test_the_shop_buys_card_removal_when_a_curse_is_present():
    state = _shop_state(
        deck=[_card("STRIKE_IRONCLAD", "Attack"), _card("REGRET", "Curse")],
        options=[{"action": "remove_card"}, {"action": "leave_shop"}],
    )
    assert _pick_shop_option(state) == 0
