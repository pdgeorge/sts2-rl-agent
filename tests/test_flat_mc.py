"""Tests for the flat Monte Carlo combat policy.

The behavioural test at the bottom is the one that matters: search has to
actually beat random, or the machinery above it is measuring nothing.
"""

from __future__ import annotations

import copy
import random

import numpy as np
import pytest

from sts2_env.core.constants import ACTION_END_TURN, MAX_HAND_SIZE
from sts2_env.gym_env.action_space import apply_action, get_action_mask
from sts2_env.gym_env.combat_env import STS2CombatEnv
from sts2_env.search.cloning import clone_combat
from sts2_env.search.flat_mc import (
    REJECTED_SCORE,
    FlatMonteCarloPolicy,
    RandomRolloutPolicy,
    reseed_combat,
    rollout,
    score_state,
)


def _fresh_combat(seed: int = 0):
    env = STS2CombatEnv()
    env.reset(seed=seed)
    assert env.combat is not None
    return env.combat


def _state_fingerprint(combat) -> tuple:
    """Enough of the combat to detect a divergence between two paths."""
    return (
        combat.player.current_hp,
        combat.player.block,
        combat.energy,
        combat.turn_count,
        len(combat.hand),
        len(combat.draw_pile),
        len(combat.discard_pile),
        tuple(e.current_hp for e in combat.enemies),
        combat.is_over,
        combat.player_won,
    )


# --- cloning: the assumption everything else rests on ------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 5, 8, 13, 21])
def test_clone_is_behaviourally_independent(seed):
    """A clone must play out identically to the original, not merely look like it.

    Plain ``deepcopy`` passes an equality check at copy time and then diverges:
    monster effects close over their creature, ``deepcopy`` copies functions by
    reference, so the clone's monsters keep acting on the original's creatures.
    The two fights corrupt each other and nothing raises. This is the test that
    catches it, so it drives both copies through the same actions and compares
    the whole way down.
    """
    env = STS2CombatEnv()
    env.reset(seed=seed)
    original = env.combat
    assert original is not None
    clone = clone_combat(original)

    rng = random.Random(seed)
    for step in range(120):
        if original.is_over or clone.is_over:
            break
        valid = np.flatnonzero(get_action_mask(original))
        if valid.size == 0:
            break
        action = int(rng.choice(list(valid)))

        accepted_original = apply_action(original, action)
        accepted_clone = apply_action(clone, action)

        assert accepted_original == accepted_clone, f"step {step}"
        assert _state_fingerprint(original) == _state_fingerprint(clone), f"step {step}"
        assert [e.block for e in original.enemies] == [e.block for e in clone.enemies], (
            f"step {step}: enemy block diverged -- a monster is acting on the "
            f"other combat's creature"
        )


def test_clone_does_not_disturb_the_original():
    """Rebinding the clone's closures must not rebind the original's.

    ``cell_contents`` is writable, so the tempting in-place fix would repoint
    the original monster's move as well and break the state being cloned from.
    """
    env = STS2CombatEnv()
    env.reset(seed=4)
    original = env.combat
    assert original is not None

    solo = clone_combat(original)
    solo_only = _state_fingerprint(solo)

    # Drive a throwaway clone hard, then confirm the original is untouched.
    before = _state_fingerprint(original)
    scratch = clone_combat(original)
    rollout(scratch, random.Random(0), max_turns=40)
    assert _state_fingerprint(original) == before
    assert solo_only == _state_fingerprint(solo)


def test_clone_of_a_clone_stays_independent():
    """Search clones from positions that are themselves clones."""
    env = STS2CombatEnv()
    env.reset(seed=6)
    assert env.combat is not None

    first = clone_combat(env.combat)
    rollout(first, random.Random(1), max_turns=3)
    second = clone_combat(first)

    rng = random.Random(6)
    for _ in range(40):
        if first.is_over or second.is_over:
            break
        valid = np.flatnonzero(get_action_mask(first))
        if valid.size == 0:
            break
        action = int(rng.choice(list(valid)))
        apply_action(first, action)
        apply_action(second, action)
        assert _state_fingerprint(first) == _state_fingerprint(second)
        assert [e.block for e in first.enemies] == [e.block for e in second.enemies]


# --- apply_action is the single definition of what an index means -------------


def test_apply_action_matches_env_step():
    """The env and the search policy must not drift apart on action meaning.

    Two paths interpreting one index differently is a documented bug class here
    and produces no error on either side, so it gets a test rather than trust.
    """
    env = STS2CombatEnv()
    env.reset(seed=7)
    # clone_combat, not deepcopy: a deep-copied shadow's monsters would act on
    # the env's creatures and the two would diverge for reasons having nothing
    # to do with action semantics, which is what this test is about.
    shadow = clone_combat(env.combat)

    rng = random.Random(7)
    for _ in range(40):
        assert env.combat is not None
        if env.combat.is_over:
            break
        valid = np.flatnonzero(get_action_mask(env.combat))
        if valid.size == 0:
            break
        action = int(rng.choice(list(valid)))

        env.step(action)
        apply_action(shadow, action)

        assert _state_fingerprint(env.combat) == _state_fingerprint(shadow)


def _empty_hand_slot_action(combat) -> int:
    """An action the engine genuinely refuses.

    Not just any masked-off index: the mask is *stricter* than the engine. Index
    1 is masked off for a card that needs a target, but ``play_card(0, None)``
    still succeeds because the engine auto-targets. Playing from a hand slot
    that holds no card is refused for real.
    """
    slot = len(combat.hand)
    assert slot < MAX_HAND_SIZE, "need a combat with a non-full hand"
    return 1 + slot


def test_apply_action_reports_rejection():
    """A refused action returns False and leaves the state untouched."""
    combat = _fresh_combat(seed=3)
    illegal = _empty_hand_slot_action(combat)
    assert get_action_mask(combat)[illegal] == 0

    before = _state_fingerprint(combat)
    assert apply_action(combat, illegal) is False
    assert _state_fingerprint(combat) == before


# --- reseeding: the bug that makes N rollouts secretly be one ----------------


def test_reseed_makes_clones_diverge():
    """Without reseeding, every rollout of an action replays one identical
    future, so averaging over rollouts averages one sample N times."""
    combat = _fresh_combat(seed=11)

    unseeded = [clone_combat(combat) for _ in range(2)]
    for c in unseeded:
        rollout(c, random.Random(0), max_turns=8)
    assert _state_fingerprint(unseeded[0]) == _state_fingerprint(unseeded[1])

    seeded = []
    for s in (101, 202, 303, 404, 505, 606):
        c = clone_combat(combat)
        reseed_combat(c, s)
        rollout(c, random.Random(s), max_turns=8)
        seeded.append(_state_fingerprint(c))

    assert len(set(seeded)) > 1, "reseeded rollouts should not all land identically"


def test_reseed_is_reproducible():
    """Same seed, same future -- otherwise nothing here is replayable."""
    combat = _fresh_combat(seed=13)
    outcomes = []
    for _ in range(2):
        c = clone_combat(combat)
        reseed_combat(c, 999)
        rollout(c, random.Random(999), max_turns=10)
        outcomes.append(_state_fingerprint(c))
    assert outcomes[0] == outcomes[1]


def test_reseed_survives_a_combat_with_no_run_attached():
    """A bare combat has no RunState to reach through; reseeding must not care.

    The same code path runs for combat-only training and for combat inside a
    full run, and only the second has run-level RNG streams.
    """
    combat = _fresh_combat(seed=5)
    reseed_combat(combat, 42)  # must not raise
    assert combat.rng.next_int(0, 100) >= 0


# --- scoring -----------------------------------------------------------------


def test_score_orders_win_above_position_above_loss():
    combat = _fresh_combat(seed=2)

    mid = score_state(combat)
    assert 0.0 <= mid <= 1.0

    won = copy.deepcopy(combat)
    won.is_over = True
    won.player_won = True
    assert score_state(won) > mid

    lost = copy.deepcopy(combat)
    lost.is_over = True
    lost.player_won = False
    assert score_state(lost) == 0.0
    assert score_state(lost) < mid


def test_score_prefers_winning_with_more_hp():
    """Ties between wins break on remaining HP, which is what makes the policy
    fight efficiently rather than merely survive."""
    healthy = _fresh_combat(seed=4)
    healthy.is_over = True
    healthy.player_won = True

    hurt = copy.deepcopy(healthy)
    hurt.player.current_hp = max(1, hurt.player.max_hp // 10)

    assert score_state(healthy) > score_state(hurt)


# --- the policy --------------------------------------------------------------


def test_policy_is_deterministic_for_a_seed():
    combat = _fresh_combat(seed=17)
    a = FlatMonteCarloPolicy(rollouts=3, seed=1).act(clone_combat(combat))
    b = FlatMonteCarloPolicy(rollouts=3, seed=1).act(clone_combat(combat))
    assert a == b


def test_policy_does_not_mutate_the_position_it_is_given():
    """Search must be side-effect free; it explores clones, never the real state."""
    combat = _fresh_combat(seed=19)
    before = _state_fingerprint(combat)
    FlatMonteCarloPolicy(rollouts=3, seed=2).act(combat)
    assert _state_fingerprint(combat) == before


def test_policy_only_returns_legal_actions():
    combat = _fresh_combat(seed=23)
    policy = FlatMonteCarloPolicy(rollouts=2, seed=3)
    for _ in range(15):
        if combat.is_over:
            break
        mask = get_action_mask(combat)
        action = policy.act(combat)
        assert mask[action] == 1
        apply_action(combat, action)


def test_rejected_actions_score_below_everything():
    """A mask bug should make search avoid the action, not hang on it."""
    combat = _fresh_combat(seed=29)
    policy = FlatMonteCarloPolicy(rollouts=2, seed=4)

    illegal = _empty_hand_slot_action(combat)
    assert policy._evaluate(combat, illegal, decision=0) == REJECTED_SCORE

    scores = policy.action_scores(combat)
    assert scores, "expected at least one legal action"
    assert min(scores.values()) > REJECTED_SCORE


def test_rollout_always_terminates():
    """Bounded by construction: an engine state that neither ends nor advances
    the turn counter must not be able to hang a rollout."""
    combat = _fresh_combat(seed=31)
    score = rollout(combat, random.Random(0), max_turns=200)
    assert 0.0 <= score <= 2.0


def test_end_turn_bias_keeps_rollouts_playing_cards():
    """A uniform rollout ends its turn about as often as it plays, which
    flattens the scores search depends on."""
    combat = _fresh_combat(seed=37)
    mask = get_action_mask(combat)
    valid = np.flatnonzero(mask)
    assert (valid != ACTION_END_TURN).sum() > 0

    policy = RandomRolloutPolicy(end_turn_prob=0.0)
    rng = random.Random(0)
    picks = [policy(combat, valid, rng) for _ in range(50)]
    assert ACTION_END_TURN not in picks

    always = RandomRolloutPolicy(end_turn_prob=1.0)
    assert always(combat, valid, rng) == ACTION_END_TURN


def test_end_turn_when_it_is_the_only_option():
    combat = _fresh_combat(seed=41)
    policy = RandomRolloutPolicy()
    only_end = np.array([ACTION_END_TURN])
    assert policy(combat, only_end, random.Random(0)) == ACTION_END_TURN


# --- the one that actually matters -------------------------------------------


@pytest.mark.slow
def test_search_beats_random_on_win_rate():
    """If search does not outplay random, it cannot serve as the pilot for deck
    evaluation, and every deck ranking built on it would be noise."""
    seeds = list(range(12))

    def play(chooser) -> int:
        wins = 0
        for s in seeds:
            env = STS2CombatEnv()
            env.reset(seed=s)
            combat = env.combat
            assert combat is not None
            for _ in range(400):
                if combat.is_over:
                    break
                mask = get_action_mask(combat)
                valid = np.flatnonzero(mask)
                if valid.size == 0:
                    break
                if not apply_action(combat, chooser(combat, valid, s)):
                    break
            wins += int(combat.player_won)
        return wins

    rng = random.Random(0)
    random_wins = play(lambda combat, valid, s: int(rng.choice(list(valid))))

    policy = FlatMonteCarloPolicy(rollouts=4, max_rollout_turns=20, seed=0)

    def search_chooser(combat, valid, s):
        return policy.act(combat)

    search_wins = play(search_chooser)

    assert search_wins > random_wins, (
        f"search won {search_wins}/{len(seeds)}, random won {random_wins}/{len(seeds)}"
    )
