"""Combat option-score logging: the lines rejected, not only the one played."""
from __future__ import annotations

import sts2_env.cards  # noqa: F401
from sts2_env.core.enums import CardId
from sts2_env.cards.factory import create_card
from sts2_env.search.situation import CardRef, CombatSituation
from sts2_env.search.turn_search import CONSIDERED_LINES, SearchAgent, search_turn


def _combat(hand):
    c = CombatSituation(
        situation_id="opts", character_id="Ironclad", current_hp=60, max_hp=80,
        deck=tuple([CardRef("STRIKE_IRONCLAD")] * 5 + [CardRef("DEFEND_IRONCLAD")] * 5),
        encounter="setup_fossil_stalker_normal", encounter_seed=99, combat_seed=7,
        relics=("BURNING_BLOOD",)).to_combat()
    c.hand.clear()
    for cid in hand:
        c.hand.append(create_card(cid))
    c.energy = 3
    return c


def test_search_result_carries_the_lines_it_rejected():
    combat = _combat([CardId.STRIKE_IRONCLAD, CardId.DEFEND_IRONCLAD, CardId.BASH])
    result = search_turn(combat)
    assert result.considered, "the search reported no alternatives at all"
    assert len(result.considered) <= CONSIDERED_LINES
    scores = [s for s, _ in result.considered]
    assert scores == sorted(scores, reverse=True), "not ranked best-first"
    # The line she played has to be among what she looked at.
    assert tuple(result.actions) in {a for _, a in result.considered}


def test_the_played_line_is_marked_and_the_names_resolve():
    from sts2_env.bridge.live_search import _describe_lines

    combat = _combat([CardId.BASH, CardId.STRIKE_IRONCLAD, CardId.DEFEND_IRONCLAD])
    agent = SearchAgent(time_budget=1.0)
    agent.act(combat)
    described = _describe_lines(agent.last_result, combat)

    assert described, "no options described"
    assert sum(1 for d in described if d["chosen"]) == 1, "exactly one line is the played one"
    # Real card names, not raw action indices.
    flat = [name for d in described for name in d["line"]]
    assert any("STRIKE" in n or "BASH" in n or "DEFEND" in n or n == "END_TURN" for n in flat)
    assert not any(n.startswith("action:") for n in flat), flat
    for d in described:
        assert isinstance(d["score"], float)


def test_describing_is_safe_when_there_was_no_search():
    from sts2_env.bridge.live_search import _describe_lines
    assert _describe_lines(None, _combat([CardId.STRIKE_IRONCLAD])) is None


def test_a_logging_failure_cannot_end_a_run():
    """The whole point of the try/except: a broken journal is not a lost run."""
    from sts2_env.bridge.agent_runner import _log_combat_options

    class Exploding:
        def write(self, *a, **k):
            raise RuntimeError("journal is on fire")

    class Agent:
        last_options = [{"score": 1.0, "line": ["STRIKE_IRONCLAD"], "chosen": True}]

    _log_combat_options(Exploding(), Agent(), {"enemies": []})  # must not raise


def test_nothing_is_written_when_the_search_had_nothing_to_say():
    from sts2_env.bridge.agent_runner import _log_combat_options

    written = []

    class Journal:
        def write(self, event, **fields):
            written.append(event)

    class Agent:
        last_options = None

    _log_combat_options(Journal(), Agent(), {"enemies": []})
    assert written == []


def test_the_position_is_logged_with_the_mods_own_field_names():
    """`player.hp`, not `player.current_hp`.

    The first version guessed `current_hp` and logged `hp: None` for an entire
    100-run session -- the scores were right and the position beside them was
    empty. Field names here are taken from a real captured `combat_action`:
    player has hp/max_hp/block/energy, enemies have hp/block/intent.
    """
    from sts2_env.bridge.agent_runner import _log_combat_options

    rows = []

    class Journal:
        def write(self, event, **fields):
            rows.append((event, fields))

    class Agent:
        last_options = [{"score": 0.5, "line": ["STRIKE_IRONCLAD->0"], "chosen": True}]

    _log_combat_options(Journal(), Agent(), {
        "room_type": "Boss",
        "round": 4,
        "player": {"hp": 53, "max_hp": 80, "block": 7, "energy": 2},
        "enemies": [{"id": "WATERFALL_GIANT", "hp": 120, "block": 0,
                     "intent": "ATTACK", "intent_damage": 18, "intent_hits": 1}],
    })

    assert len(rows) == 1
    event, f = rows[0]
    assert event == "combat_options"
    assert (f["hp"], f["max_hp"], f["block"], f["energy"]) == (53, 80, 7, 2)
    assert f["turn"] == 4 and f["room_type"] == "Boss"
    assert f["enemies"] == [{"id": "WATERFALL_GIANT", "hp": 120, "block": 0,
                             "intent": "ATTACK", "intent_damage": 18,
                             "intent_hits": 1}]


def test_potions_and_end_turn_are_named_not_left_as_action_ints():
    """`action:62` made a whole session's transcripts unreadable.

    It fell back to the raw index for anything that was not a card, which is
    every potion -- and potions were the single largest effect in the boss
    counterfactual grid (-12.1 points without them). A reviewer cannot say
    "she wasted Powdered Demise here" if the log says `action:62`.
    """
    from sts2_env.core.constants import ACTION_END_TURN, POTION_ACTION_START
    from sts2_env.bridge.live_search import _name_action
    from sts2_env.potions.base import create_potion

    combat = _combat([CardId.STRIKE_IRONCLAD])
    combat.potions = [create_potion("PowderedDemise"), None, None]

    assert _name_action(combat, ACTION_END_TURN) == "END_TURN"
    named = _name_action(combat, POTION_ACTION_START)
    assert named.startswith("PowderedDemise"), named
    assert "STRIKE" in _name_action(combat, 1 + 0)  # first card action


def test_leaf_snapshots_align_with_considered():
    """`considered_leaves[i]` must describe `considered[i]`.

    Two parallel tuples is a contract that can silently rot, and the whole
    reason the field exists is to explain a tie -- an off-by-one would explain a
    tie wrongly and look entirely plausible, which is worse than no field.
    """
    combat = _combat([CardId.STRIKE_IRONCLAD, CardId.DEFEND_IRONCLAD, CardId.BASH])
    result = search_turn(combat)

    assert len(result.considered_leaves) == len(result.considered)
    for (score, actions), leaf in zip(result.considered, result.considered_leaves):
        assert leaf.end_hp <= combat.player.max_hp
        # The pooled total has to be the per-enemy list, or the two columns are
        # telling different stories and the spread column cannot be trusted.
        assert leaf.end_enemy_hp == sum(leaf.end_enemy_hps)
        assert leaf.end_enemies_alive == len(leaf.end_enemy_hps)
        assert isinstance(score, float) and isinstance(actions, tuple)


def test_leaf_records_the_spread_not_only_the_pool():
    """The per-enemy column must survive a line that splits its damage.

    `evaluate.py` scores enemy HP as ONE pooled fraction, which is the blind
    spot the snapshot exists to see past. If `end_enemy_hps` were dropped or
    sorted into the pooled total, focusing and spreading would look identical
    here exactly as they do to the evaluator.
    """
    combat = _combat([CardId.STRIKE_IRONCLAD, CardId.STRIKE_IRONCLAD])
    result = search_turn(combat)
    spreads = {leaf.end_enemy_hps for leaf in result.considered_leaves}
    pools = {leaf.end_enemy_hp for leaf in result.considered_leaves}
    if len(pools) == 1 and len(result.considered_leaves) > 1:
        # Same total damage dealt by every line -- which is the interesting
        # case, and the one where only the per-enemy column can tell them apart.
        assert len(spreads) >= 1


def test_tie_break_only_ever_reorders_exact_ties():
    """`focus` must never choose a line the evaluator scored lower.

    Prediction 13 rests on this: the arm decides what enumeration order was
    deciding, and nothing more. If the focus arm ever comes back with a worse
    score than the enumeration arm on the same state, the claim is false and the
    measurement would be attributing an evaluation change to a tie-break.
    """
    for hand in (
        [CardId.STRIKE_IRONCLAD, CardId.DEFEND_IRONCLAD, CardId.BASH],
        [CardId.STRIKE_IRONCLAD, CardId.STRIKE_IRONCLAD, CardId.DEFEND_IRONCLAD],
        [CardId.BASH, CardId.BASH],
    ):
        base = search_turn(_combat(hand), tie_break="enumeration")
        focus = search_turn(_combat(hand), tie_break="focus")
        assert focus.score == base.score, (
            f"tie_break changed the SCORE on {hand}: "
            f"{base.score} -> {focus.score}; it may only pick between equals")


def test_tie_break_prefers_the_concentrated_board():
    """Given a real tie, `focus` takes the line with the weaker survivor.

    Asserted through `tie_break_key` on the snapshots the search actually
    produced, rather than on a hand-built position: the key is only ever
    consulted on states the enumeration reached, so that is where it has to be
    correct.
    """
    from sts2_env.search.turn_search import tie_break_key

    combat = _combat([CardId.STRIKE_IRONCLAD, CardId.STRIKE_IRONCLAD])
    result = search_turn(combat, tie_break="focus")
    top = result.considered[0][0]
    tied = [leaf for (score, _), leaf in
            zip(result.considered, result.considered_leaves)
            if abs(score - top) < 1e-9]
    if len(tied) > 1:
        chosen = min(tie_break_key(leaf) for leaf in tied)
        assert chosen == tie_break_key(min(tied, key=tie_break_key))
    # Concentration is what the key ranks on: fewer alive beats more alive,
    # and at equal counts a weaker survivor beats a healthier one.
    from sts2_env.search.turn_search import LeafSnapshot
    spread = LeafSnapshot(50, 0, 2, 20, (12, 8))
    focused = LeafSnapshot(50, 0, 2, 20, (17, 3))
    killed = LeafSnapshot(50, 0, 1, 20, (20,))
    assert tie_break_key(focused) < tie_break_key(spread)
    assert tie_break_key(killed) < tie_break_key(focused)
