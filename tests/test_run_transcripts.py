"""Per-run transcripts, and the aggregate over their reviews.

The transcript is what a reviewing model reads instead of 62k tokens of JSONL,
so the two things that matter are that it stays readable (no raw action ints,
one block per turn) and that it never invents anything the journal did not say.
"""

from __future__ import annotations

import json

import pytest

from scripts.aggregate_run_reports import _load_json, _run_outcomes
from scripts.export_run_transcripts import render


def _events():
    """A minimal run: one fight over two turns, a reward, and a death."""
    return [
        {"event": "run_start", "run": 7, "character": "Ironclad", "ascension": 0,
         "policy_version": "v001", "git_sha": "abc1234"},
        {"event": "combat_start", "floor": 11, "room_type": "Elite", "hp": 62,
         "max_hp": 80, "deck_size": 19, "relics": ["BURNING_BLOOD"],
         "potions": ["PowderedDemise", None, None],
         "enemies": [{"id": "BYGONE_EFFIGY", "hp": 108, "intent": "ATTACK",
                      "intent_damage": 16, "intent_hits": 1}]},
        {"event": "combat_options", "turn": 1, "options": [
            {"score": 0.612, "line": ["BASH->0", "STRIKE_IRONCLAD->0"], "chosen": True},
            {"score": 0.608, "line": ["DEFEND_IRONCLAD", "BASH->0"], "chosen": False},
            {"score": 0.571, "line": ["action:62"], "chosen": False}]},
        {"event": "card_played", "card": "BASH", "target": 0},
        {"event": "card_played", "card": "STRIKE_IRONCLAD", "target": 0},
        {"event": "combat_options", "turn": 1, "options": [
            {"score": 0.61, "line": ["STRIKE_IRONCLAD->0"], "chosen": True}]},
        {"event": "end_turn", "round": 1},
        {"event": "turn", "round": 2, "hp": 46, "block": 0,
         "enemies": [{"id": "BYGONE_EFFIGY", "hp": 80}]},
        {"event": "card_played", "card": "DEFEND_IRONCLAD", "target": -1},
        {"event": "combat_end", "hp_before": 62, "hp_after": 0, "turns": 2,
         "cards_played": 3, "room_type": "Elite"},
        {"event": "run_end", "run": 7, "floor": 11, "act": 1, "room_type": "Elite",
         "run_hp": 0, "run_max_hp": 80, "death_enemy_id": "BYGONE_EFFIGY",
         "deck_size": 19, "relic_count": 4},
    ]


def test_a_clearly_better_alternative_is_shown_with_its_score():
    """The rejected lines are the point -- when the gap is real."""
    events = _events()
    events[2]["options"][1]["score"] = 0.50   # 0.11 behind: a real difference
    text = render(events)
    assert "passed  DEFEND_IRONCLAD, BASH->0" in text
    assert "0.612" in text


def test_near_ties_are_not_shown_at_all():
    """Asking a reviewer to ignore coin flips does not work; not showing them does.

    On the tuesday session 51% of the decisions the model flagged were inside
    0.005 of the line she played -- having been told in the prompt that those
    are coin flips. The searcher always plays its top-scored line, so a
    rejected line 0.002 behind is a position the evaluator had no opinion
    about, roughly a sixth of one HP.
    """
    events = _events()
    for alt in events[2]["options"][1:]:
        alt["score"] = 0.610          # every alternative within 0.002
    text = render(events)
    assert "passed" not in text
    assert "no clear alternative" in text
    # the turn itself still appears, so the fight still reads as a fight
    assert "played  BASH->0, STRIKE_IRONCLAD->0" in text


def test_no_raw_action_integers_reach_the_reader():
    """`action:62` is a potion, and potions were the largest effect measured."""
    text = render(_events())
    assert "action:62" not in text
    assert "potion[slot" in text


def test_a_turn_is_one_block_however_many_times_she_replanned():
    """She re-searches after every card; five rows a turn says one thing five times."""
    text = render(_events())
    assert text.count("played  BASH") == 1
    assert "(replans: 2)" in text, text


def test_the_cards_played_come_from_the_journal_not_from_the_plan():
    """Ground truth, because a plan can be abandoned and the game cannot."""
    text = render(_events())
    assert "played  BASH->0, STRIKE_IRONCLAD->0" in text
    assert "DEFEND_IRONCLAD" in text.split("T2")[1]


def test_the_run_ends_with_how_it_actually_ended():
    text = render(_events())
    assert "RUN END" in text and "floor 11" in text and "BYGONE_EFFIGY" in text


@pytest.mark.parametrize("wrapper", [
    '{}', '```json\n{}\n```', 'here you go:\n```\n{}\n```', 'sure!\n{}\nhope that helps',
])
def test_a_reply_parses_through_whatever_the_model_wrapped_it_in(wrapper):
    payload = json.dumps({"run": 1, "mistakes": []})
    assert _load_json(wrapper.replace("{}", payload)) == {"run": 1, "mistakes": []}


def test_a_refusal_is_not_silently_counted_as_a_clean_run():
    assert _load_json("I'm sorry, I cannot help with that.") is None


def test_outcomes_are_keyed_by_transcript_number_not_journal_run_index(tmp_path):
    """A session restarts on a crash and the run counter restarts with it.

    `tuesday` held 103 runs under 53 distinct indices, so matching report N
    against journal run N compared a report to a different run in another
    segment -- and the "claim on a floor the run never reached" check was
    validating against the wrong ground truth. Correctly keyed, that check went
    from 26 claims discarded to 68.
    """
    j = tmp_path / "j.jsonl"
    j.write_text("\n".join(json.dumps(r) for r in [
        {"event": "run_end", "session": "A", "run": 1, "floor": 9, "act": 1},
        {"event": "run_end", "session": "A", "run": 2, "floor": 17, "act": 2},
        # the game crashed; the counter restarted, so run 1 appears twice
        {"event": "run_end", "session": "B", "run": 1, "floor": 33, "act": 3},
    ]))
    out = _run_outcomes(j)
    assert [out[i]["floor"] for i in (1, 2, 3)] == [9, 17, 33]
    assert len(out) == 3, "three runs, not two collapsed onto one index"


# -- the review loop's contract check ---------------------------------------

@pytest.mark.parametrize("data,problem", [
    ({"mistakes": []}, None),
    ({"mistakes": [{"floor": 3, "did": "x"}]}, None),
    ({}, "missing"),
    ({"mistakes": "none"}, "must be a list"),
    ({"mistakes": [{"did": "no floor"}]}, "floor"),
])
def test_a_reply_is_only_accepted_when_it_can_be_checked(data, problem):
    """A claim without a location cannot be looked up, so it cannot be believed."""
    from scripts.review_runs import _valid
    result = _valid(data)
    if problem is None:
        assert result is None
    else:
        assert result and problem in result


def test_the_transcript_tells_the_reviewer_it_does_not_know_this_game():
    """An 8B has never seen STS2 and the names overlap StS1.

    Without the card reference it reviews a game it invented, so the primer and
    the per-run card list are load-bearing, not decoration.
    """
    from scripts.export_run_transcripts import _appendix
    cards = {"BASH": {"name": "Bash", "type": "Attack", "cost": "2",
                      "rarity": "Basic", "description": "Deal 8 damage."}}
    text = _appendix(_events(), cards)
    assert "not Slay the Spire 1" in text
    assert "**Bash** (Attack, cost 2, Basic): Deal 8 damage." in text
    assert "BYGONE_EFFIGY (108 HP)" in text
