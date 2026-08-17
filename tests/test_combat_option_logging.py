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
