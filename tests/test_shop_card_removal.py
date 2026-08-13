"""Shop removal takes the worst card, and never the last block.

`scripts/removal_vs_relic.py` priced removal at ~3.3 points of act 1 boss win
against ~2 for a marginal relic, over 30 real live boss decks (n=360 per cell,
monotonic across 0/1/2/3 removals, 2.7 sigma end to end). That is what moved
`remove_card` to the front of `SHOP_PURCHASE_ACTION_PRIORITY` and lifted the
curses-only restriction.

The number only transfers if the SHIPPED policy is the policy that was priced,
so these pin the specific order the grid used -- curses first, then surplus
basics, Strike before Defend, unupgraded before upgraded, never below two
copies. A removal policy that is merely "sensible" but different is not covered
by that measurement.
"""

from __future__ import annotations

from sts2_env.bridge.agent_runner import (
    MIN_BASIC_COPIES_TO_KEEP,
    SHOP_PURCHASE_ACTION_PRIORITY,
    _has_removal_target,
    _pick_card_select_indexes,
    _pick_shop_option,
    _removal_order,
)


def card(card_id: str, **extra):
    return {"id": card_id, **extra}


def deck(strikes: int, defends: int, *, others: int = 0, curses: int = 0):
    out = [card("STRIKE_IRONCLAD") for _ in range(strikes)]
    out += [card("DEFEND_IRONCLAD") for _ in range(defends)]
    out += [card(f"CLEAVE_{i}") for i in range(others)]
    out += [card(f"REGRET_{i}", type="Curse") for i in range(curses)]
    return out


class TestTheFloor:
    """The objection the old curses-only rule existed to prevent."""

    def test_never_removes_the_last_defends(self):
        cards = deck(strikes=5, defends=MIN_BASIC_COPIES_TO_KEEP)
        chosen = _removal_order(cards)
        assert all(cards[i]["id"] != "DEFEND_IRONCLAD" for i in chosen), (
            "removed a Defend while the deck was down to its floor -- this is "
            "the 'that Defend was the deck's only block' failure exactly"
        )

    def test_a_deck_at_the_floor_offers_nothing(self):
        cards = deck(strikes=MIN_BASIC_COPIES_TO_KEEP,
                     defends=MIN_BASIC_COPIES_TO_KEEP, others=6)
        assert _removal_order(cards) == []
        assert not _has_removal_target({"deck": cards})

    def test_non_basics_are_never_removed(self):
        """Only basics and curses. A real card is never the target."""
        cards = deck(strikes=4, defends=4, others=5)
        for i in _removal_order(cards):
            assert cards[i]["id"].startswith(("STRIKE_", "DEFEND_")) or \
                cards[i].get("type") == "Curse"

    def test_removal_count_respects_the_floor_exactly(self):
        cards = deck(strikes=5, defends=4)
        # 5-2 surplus Strikes plus 4-2 surplus Defends.
        assert len(_removal_order(cards)) == 3 + 2


class TestTheOrder:
    """Curses, then Strike, then Defend -- the order the grid measured."""

    def test_curses_come_first(self):
        cards = deck(strikes=5, defends=5, curses=1)
        first = _removal_order(cards)[0]
        assert cards[first].get("type") == "Curse"

    def test_strike_before_defend(self):
        cards = deck(strikes=5, defends=5)
        order = _removal_order(cards)
        assert cards[order[0]]["id"] == "STRIKE_IRONCLAD"

    def test_unupgraded_copies_go_first(self):
        """Removing a Strike+ throws away a rest-site smith."""
        cards = [card("STRIKE_IRONCLAD", upgraded=True),
                 card("STRIKE_IRONCLAD"),
                 card("STRIKE_IRONCLAD", upgraded=True),
                 card("DEFEND_IRONCLAD"), card("DEFEND_IRONCLAD")]
        first = _removal_order(cards)[0]
        assert not cards[first].get("upgraded")


class TestTheScreen:
    """`_pick_card_select_indexes(removing=True)`, which spends the purchase."""

    def test_takes_the_curse_when_there_is_one(self):
        cards = deck(strikes=5, defends=5, curses=1)
        state = {"cards": cards, "min_select": 1, "max_select": 1}
        assert _pick_card_select_indexes(state, removing=True) == \
            [cards.index(next(c for c in cards if c.get("type") == "Curse"))]

    def test_takes_a_surplus_basic_with_no_curse_present(self):
        """The behaviour change: this used to decline and waste the gold."""
        cards = deck(strikes=5, defends=5)
        state = {"cards": cards, "min_select": 1, "max_select": 1}
        chosen = _pick_card_select_indexes(state, removing=True)
        assert len(chosen) == 1
        assert cards[chosen[0]]["id"] == "STRIKE_IRONCLAD"

    def test_declines_rather_than_gutting_a_thin_deck(self):
        cards = deck(strikes=MIN_BASIC_COPIES_TO_KEEP,
                     defends=MIN_BASIC_COPIES_TO_KEEP, others=4)
        state = {"cards": cards, "min_select": 1, "max_select": 1}
        assert _pick_card_select_indexes(state, removing=True) == []

    def test_upgrade_screens_are_untouched_by_this_change(self):
        """`removing=False` must still be basics-LAST, never curses."""
        cards = deck(strikes=4, defends=4, others=2, curses=1)
        state = {"cards": cards, "min_select": 1, "max_select": 1}
        chosen = _pick_card_select_indexes(state, removing=False)
        picked = cards[chosen[0]]
        assert picked["id"].startswith("CLEAVE_"), (
            "an upgrade screen picked a basic or a curse; removal ordering has "
            "leaked into the default path"
        )


class TestTheShop:
    def test_removal_outranks_relics(self):
        p = list(SHOP_PURCHASE_ACTION_PRIORITY)
        assert p.index("remove_card") < p.index("buy_relic"), (
            "removal is ~3.3 points at 75 gold against a relic's ~2 at 150-300"
        )

    def test_buys_removal_with_a_removable_deck(self):
        state = {
            "deck": deck(strikes=5, defends=5),
            "options": [
                {"index": 0, "action": "buy_relic", "enabled": True},
                {"index": 1, "action": "remove_card", "enabled": True},
            ],
        }
        assert _pick_shop_option(state) == 1

    def test_skips_removal_when_the_deck_is_already_thin(self):
        state = {
            "deck": deck(strikes=MIN_BASIC_COPIES_TO_KEEP,
                         defends=MIN_BASIC_COPIES_TO_KEEP, others=8),
            "options": [
                {"index": 0, "action": "buy_relic", "enabled": True},
                {"index": 1, "action": "remove_card", "enabled": True},
            ],
        }
        assert _pick_shop_option(state) == 0, (
            "bought a removal against a deck the screen will decline -- 75 "
            "gold for nothing"
        )
