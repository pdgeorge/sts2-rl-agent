"""The agent must be able to see what it has selected on a multi-select screen.

Found in `meta_ppo_v8_rewarded`: 61 of 2350 eval episodes (2.6%) ran to the step
cap at ~1950-1988 steps against a median of 19, and those 61 consumed roughly
**73% of all evaluation compute**.

The cause was not combat and not a step budget. A deck transform presents the
whole deck to choose three cards from. `choices_from_sim_actions` did not read
`choose` actions at all, so the entire screen encoded as zeros -- the
observation was byte-identical whether a card was selected or not. A
deterministic policy therefore picked the same option every step, toggling one
card on and off, and could never reach the three it needed to confirm.

Worse than the 2.6% suggests: transforms come from later rewards, so it struck
the *deepest* runs. The captured case was on floor 16.
"""

from __future__ import annotations

import numpy as np

from sts2_env.gym_env.choice_encoding import (
    CARD_FEATURES,
    CARD_SLOTS,
    choices_from_bridge_state,
    choices_from_sim_actions,
    encode_card_options,
)


def test_selecting_a_card_changes_the_encoding():
    """The whole bug in one assertion."""
    cards = ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"]
    unselected = encode_card_options(cards, [False, False, False])
    selected = encode_card_options(cards, [True, False, False])

    assert not np.array_equal(unselected, selected), (
        "selection is invisible in the observation -- the agent cannot see the "
        "effect of its own action and will toggle forever"
    )


def test_the_selected_flag_lands_on_the_right_slot():
    cards = ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"]
    out = encode_card_options(cards, [False, True, False])

    assert out[0 * CARD_FEATURES + 7] == 0.0
    assert out[1 * CARD_FEATURES + 7] == 1.0
    assert out[2 * CARD_FEATURES + 7] == 0.0


def test_choose_actions_reach_the_encoder():
    """`choose` was not handled, so a whole screen encoded as zeros."""
    actions = [
        {"action": "confirm_choice", "prompt": "pick 3"},
        {"action": "choose", "index": 0, "card_id": "STRIKE_IRONCLAD", "selected": False},
        {"action": "choose", "index": 1, "card_id": "BASH", "selected": True},
    ]
    extracted = choices_from_sim_actions(actions)

    assert extracted["card_names"] == ["STRIKE_IRONCLAD", "BASH"]
    assert extracted["selected"] == [False, True]


def test_a_transform_sized_screen_is_not_truncated_to_six():
    """A deck transform offers the whole deck; the observed live case had 19."""
    actions = [
        {"action": "choose", "index": i, "card_id": "STRIKE_IRONCLAD",
         "selected": i == 18}
        for i in range(19)
    ]
    extracted = choices_from_sim_actions(actions)
    assert len(extracted["card_names"]) == 19
    assert CARD_SLOTS >= 19, "19 options were observed live on floor 16"

    out = encode_card_options(extracted["card_names"], extracted["selected"])
    assert out[18 * CARD_FEATURES + 7] == 1.0, "the 19th option's state is lost"


def test_the_bridge_side_reads_selected_too():
    """Simulator and bridge must encode the same screen identically."""
    state = {"cards": [
        {"id": "STRIKE_IRONCLAD", "selected": False},
        {"id": "BASH", "selected": True},
    ]}
    extracted = choices_from_bridge_state(state)

    assert extracted["card_names"] == ["STRIKE_IRONCLAD", "BASH"]
    assert extracted["selected"] == [False, True]


def test_a_payload_without_selected_reads_as_unselected():
    """Single-pick screens carry no flag, and absent must not mean selected."""
    extracted = choices_from_bridge_state({"cards": [{"id": "BASH"}]})
    assert extracted["selected"] == [False]
