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


def test_the_transcript_shows_what_was_rejected_with_its_score():
    """The rejected lines are the whole point.

    What she played only supports "that looks wrong". What she passed over
    supports "she had X and took Y, and they scored 0.004 apart".
    """
    text = render(_events())
    assert "passed  DEFEND_IRONCLAD, BASH->0" in text
    assert "0.608" in text and "0.612" in text


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


def test_outcomes_are_read_from_the_journal_for_checking_claims(tmp_path):
    j = tmp_path / "j.jsonl"
    j.write_text(json.dumps({"event": "run_end", "run": 3, "floor": 9, "act": 1}))
    assert _run_outcomes(j)[3]["floor"] == 9
