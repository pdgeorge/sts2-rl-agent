"""The live agent and the offline agent must make the same decision.

One of these becomes the source of truth for training, so a divergence is not a
detail -- it means offline is measuring an agent that does not ship. They had
already drifted where it mattered most: both card-reward paths ranked with
`card_quality.rank_cards`, but the offline copy skipped only on
`best_score <= 0` and carried no deck-size rule, while live skips on
`best_score < SKIP_THRESHOLD or deck_size > CARD_REWARD_LARGE_DECK_SIZE`.

Deck size is what decides an act 1 boss fight -- two captured boss decks were 21
and 22 cards with nine basic Strike/Defend still in them -- so the offline agent
was measuring a deckbuilder the live agent is not.

These tests pin the shared entry points. They do not assert a particular choice;
they assert that both sides ask the same function, which is the property that
survives the thresholds being retuned.
"""

import inspect

import pytest

import sts2_env.cards  # noqa: F401
from sts2_env.bridge import agent_runner


def _code_of(fn) -> str:
    """Source with comments stripped.

    Checking raw source matched the COMMENT explaining why the old copy is not
    used, so the test failed on the very text documenting the fix.
    """
    lines = []
    for line in inspect.getsource(fn).split("\n"):
        code = line.split("#", 1)[0]
        if code.strip():
            lines.append(code)
    return "\n".join(lines)


def test_offline_routes_card_rewards_through_the_live_chooser():
    """Not a copy of it. A copy drifts the first time either side is edited."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import live_policy

    source = _code_of(live_policy.noncombat_action)
    assert "_pick_card_reward_index" in source, (
        "offline card rewards must call the live chooser"
    )
    assert "ab_archetype_picking" not in source, (
        "the second card-reward implementation is back; delete it rather than "
        "keeping it in step"
    )


def test_offline_routes_map_and_rest_through_the_live_choosers():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import live_policy

    source = _code_of(live_policy.noncombat_action)
    assert "_pick_map_node" in source
    assert "_pick_rest_option" in source


@pytest.mark.xfail(
    strict=False,
    reason=(
        "KNOWN OPEN: both skip thresholds are set where they cannot fire. "
        "SKIP_THRESHOLD=0.0 against best-on-offer scores of 1.00-5.90 over 366 "
        "real screens, and CARD_REWARD_LARGE_DECK_SIZE=30 against act 1 decks "
        "of 21-22. The values are an empirical question for the paired offline "
        "sweep, not something to pick by eye -- this test turns green when they "
        "are set somewhere they can take effect."
    ),
)
def test_the_skip_thresholds_are_reachable():
    """A threshold set where it can never fire is not a policy.

    Measured over 366 real card-reward screens from the captured protocol, the
    BEST card on offer scored between 1.00 and 5.90. SKIP_THRESHOLD was 0.0, so
    the skip never fired once -- and CARD_REWARD_LARGE_DECK_SIZE was 30 against
    act 1 decks of 21-22 cards, so the bloat rule never fired either.

    This does not assert a good value, which is an empirical question for the
    paired offline sweep. It asserts the values are inside the range the game
    actually produces, so that whatever is chosen can take effect.
    """
    from sts2_env.bridge.card_quality import SKIP_THRESHOLD

    assert 1.0 <= SKIP_THRESHOLD <= 5.9, (
        f"SKIP_THRESHOLD={SKIP_THRESHOLD} sits outside the observed range of "
        "best-on-offer scores (1.00 to 5.90), so the skip can never fire"
    )
    assert agent_runner.CARD_REWARD_LARGE_DECK_SIZE <= 25, (
        f"CARD_REWARD_LARGE_DECK_SIZE="
        f"{agent_runner.CARD_REWARD_LARGE_DECK_SIZE} is above the deck sizes act "
        "1 actually reaches (21-22), so the bloat rule can never fire"
    )
