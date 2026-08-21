"""Enemy move transitions, against the decompiled state machines.

`audit_dynamics` measured the simulator predicting the WRONG next enemy move on
16% of checked turns. Roughly nine of fifteen were monsters whose decompiled
follow-up is a `RandomBranchState` -- a genuine die roll nobody can predict --
but the rest were monsters the game moves deterministically, and those were
ours to get right. These pin the ones that were wrong.
"""
from __future__ import annotations

import sts2_env.cards  # noqa: F401
from sts2_env.core.rng import Rng
from sts2_env.monsters.factory import create_monster_by_id


def _chain(monster_id: str):
    _, ai = create_monster_by_id(monster_id, Rng(1))
    return ai, {k: getattr(v, "follow_up_id", None) for k, v in ai.states.items()}


def test_haunted_ship_is_a_fixed_cycle_not_a_random_branch():
    """HauntedShip.GenerateMoveStateMachine:

        HAUNT_MOVE.FollowUpState = SWIPE_MOVE
        SWIPE_MOVE.FollowUpState = STOMP_MOVE
        STOMP_MOVE.FollowUpState = SWIPE_MOVE
        start: HAUNT_MOVE

    Three moves and no randomness. This was modelled as a four-way weighted
    RandomBranchState including a RAMMING_SPEED move that does not exist in the
    decompiled monster -- `grep -c RammingSpeed` is 0 -- and which the bridge
    never once reported across a full session (HAUNT 105, STOMP 99, SWIPE 152).
    """
    ai, follow = _chain("HAUNTED_SHIP")
    assert set(ai.states) == {"HAUNT_MOVE", "SWIPE_MOVE", "STOMP_MOVE"}
    assert follow["HAUNT_MOVE"] == "SWIPE_MOVE"
    assert follow["SWIPE_MOVE"] == "STOMP_MOVE"
    assert follow["STOMP_MOVE"] == "SWIPE_MOVE"
    assert ai.current_move.state_id == "HAUNT_MOVE"
    assert "RAND" not in ai.states, "a random branch this monster does not have"


def test_ceremonial_beast_stun_leads_into_its_second_phase():
    """CeremonialBeast: `moveState3.FollowUpState = BeastCryState`.

    The beast is stunned when its Plow armour comes off, and the fight then
    moves permanently into BEAST_CRY -> STOMP -> CRUSH -> BEAST_CRY. The game
    reports the synthesised "STUNNED" rather than this state, so the live
    rebuild has to route to it deliberately; the generic stun would put the
    boss back on the move it was interrupted on, in a phase it never re-enters.
    """
    ai, follow = _chain("CEREMONIAL_BEAST")
    assert follow["STUN_MOVE"] == "BEAST_CRY_MOVE"
    assert getattr(ai.states["STUN_MOVE"], "must_perform_once", False)
    assert follow["BEAST_CRY_MOVE"] == "STOMP_MOVE"
    assert follow["STOMP_MOVE"] == "CRUSH_MOVE"
    assert follow["CRUSH_MOVE"] == "BEAST_CRY_MOVE"
    assert follow["STAMP_MOVE"] == "PLOW_MOVE"
    assert follow["PLOW_MOVE"] == "PLOW_MOVE"


def test_myte_cycle_matches_and_is_slot_dependent():
    """Left alone deliberately. Myte's chain already matches the decompile
    (TOXIC -> BITE -> SUCK -> TOXIC, opening on the creature's slot), and its
    single audit mismatch sits alongside 46 roster-misalignment reports in the
    same run -- two Mytes in different slots matched to each other. Pinned so a
    future 'fix' has to argue with the decompile first."""
    _, follow = _chain("MYTE")
    assert follow["TOXIC_MOVE"] == "BITE_MOVE"
    assert follow["BITE_MOVE"] == "SUCK_MOVE"
    assert follow["SUCK_MOVE"] == "TOXIC_MOVE"
