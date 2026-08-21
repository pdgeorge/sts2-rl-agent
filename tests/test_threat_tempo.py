"""The threat-scaled tempo term: inert by default, capped, and board-derived."""
from __future__ import annotations

import sts2_env.cards  # noqa: F401
from sts2_env.search.evaluate import (
    DEFAULT_WEIGHTS,
    EvalWeights,
    _sustained_threat,
    evaluate_components,
)
from sts2_env.search.situation import CardRef, CombatSituation

ON = EvalWeights(threat_tempo=1.0)


def _fight(encounter: str, hp: int = 70):
    return CombatSituation(
        situation_id="t", character_id="Ironclad", current_hp=hp, max_hp=80,
        deck=tuple([CardRef("STRIKE_IRONCLAD")] * 10), encounter=encounter,
        encounter_seed=5, combat_seed=3, relics=()).to_combat()


def test_shipped_weights_leave_the_term_inert():
    """v001 must score exactly as it did before this term existed.

    Every number on the scoreboard was produced under it, and the A/B baseline
    has to be that same agent.
    """
    assert DEFAULT_WEIGHTS.threat_tempo == 0.0
    combat = _fight("setup_lagavulin_matriarch_boss")
    assert evaluate_components(combat, DEFAULT_WEIGHTS)["threat_tempo"] == 0.0


def test_threat_is_stable_across_turns():
    """The whole point of averaging the move table instead of reading this
    turn's intent. The first version oscillated: Lagavulin asleep read 0,
    Waterfall's Pressurize turn read 0 -- and a Pressurize turn is exactly when
    the searcher should hurry, because it is banking eruption damage."""
    combat = _fight("setup_lagavulin_matriarch_boss")
    seen = []
    for _ in range(4):
        seen.append(_sustained_threat(combat))
        combat.end_player_turn()
        if combat.is_over:
            break
    assert len(set(seen)) == 1, f"threat moved turn to turn: {seen}"
    assert seen[0] > 0, "a boss that attacks must not read as harmless"


def test_the_term_is_capped():
    """Same guard as `powers_cap`, which exists because an uncapped term once
    scored a sleeping elite at -1.2 and made the searcher refuse to attack."""
    for weight in (1.0, 10.0, 100.0):
        w = EvalWeights(threat_tempo=weight)
        for enc in ("setup_lagavulin_matriarch_boss", "setup_corpse_slugs_normal"):
            term = evaluate_components(_fight(enc), w)["threat_tempo"]
            assert term <= 0.0
            assert abs(term) <= w.threat_tempo_cap + 1e-9


def test_a_dead_board_costs_no_tempo():
    """With nothing left alive there is no fight to hurry out of, so the term
    must not keep charging for turns that are not going to happen."""
    combat = _fight("setup_corpse_slugs_normal")
    for enemy in combat.enemies:
        enemy.current_hp = 0
    assert evaluate_components(combat, ON)["threat_tempo"] == 0.0
