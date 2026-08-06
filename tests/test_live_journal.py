"""The live journal records what happened, and never costs a run.

Both halves matter. A journal that misses actions produces confident wrong
conclusions about what to fix; a journal that can raise takes down a run the
whole point of which was to be measured.
"""

from __future__ import annotations

import json

import pytest

from sts2_env.bridge.journal import RunJournal


class FakeClient:
    """Stands in for the bridge client, remembering what it was told to send."""

    def __init__(self) -> None:
        self.sent: list[tuple] = []

    def play_card(self, hand_index, target_index=None):
        self.sent.append(("play_card", hand_index, target_index))
        return True

    def end_turn(self):
        self.sent.append(("end_turn",))
        return True

    def use_potion(self, slot, target_index=None):
        self.sent.append(("use_potion", slot, target_index))
        return True

    def choose(self, index):
        self.sent.append(("choose", index))
        return True

    def choose_many(self, indexes):
        self.sent.append(("choose_many", tuple(indexes)))
        return True

    def skip(self):
        self.sent.append(("skip",))
        return True

    def ping(self):
        return True


def _combat_state(hp=70, round_number=1, **overrides):
    state = {
        "type": "combat_action",
        "player": {"hp": hp, "max_hp": 80, "block": 0, "energy": 3, "max_energy": 3},
        "hand": [
            {"id": "STRIKE_IRONCLAD", "cost": 1, "type": "Attack"},
            {"id": "BASH", "cost": 2, "type": "Attack", "upgraded": True},
        ],
        "enemies": [{"id": "JAW_WORM", "hp": 40, "max_hp": 44}],
        "round": round_number,
        "floor": 5,
        "act": 1,
        "room_type": "Monster",
        "run_hp": hp,
        "run_max_hp": 80,
        "deck_size": 14,
        "relics": ["BURNING_BLOOD"],
    }
    state.update(overrides)
    return state


def _read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def journal(tmp_path):
    j = RunJournal(tmp_path / "journal.jsonl", model="test-model")
    j.start_run(1)
    yield j
    j.close()


def _events(journal, tmp_path, name="journal.jsonl"):
    journal.close()
    return _read(tmp_path / name)


# -- the shape of a run ------------------------------------------------------

def test_the_first_state_opens_the_run(journal, tmp_path) -> None:
    journal.observe(_combat_state())
    events = _events(journal, tmp_path)
    assert events[0]["event"] == "run_start"
    assert events[0]["model"] == "test-model"


def test_entering_a_fight_records_who_and_at_what_hp(journal, tmp_path) -> None:
    journal.observe(_combat_state(hp=63))
    start = [e for e in _events(journal, tmp_path) if e["event"] == "combat_start"][0]
    assert start["hp"] == 63
    assert start["enemies"] == [{"id": "JAW_WORM", "hp": 40, "max_hp": 44}]
    assert start["room_type"] == "Monster"


def test_leaving_a_fight_records_what_it_cost(journal, tmp_path) -> None:
    journal.observe(_combat_state(hp=70))
    journal.observe({"type": "card_reward", "run_hp": 52, "floor": 5, "cards": []})

    end = [e for e in _events(journal, tmp_path) if e["event"] == "combat_end"][0]
    assert end["hp_before"] == 70
    assert end["hp_after"] == 52
    assert end["damage_taken"] == 18


def test_turns_are_recorded_as_the_round_advances(journal, tmp_path) -> None:
    journal.observe(_combat_state(round_number=1))
    journal.observe(_combat_state(round_number=2, hp=61))
    turns = [e for e in _events(journal, tmp_path) if e["event"] == "turn"]
    assert len(turns) == 1
    assert turns[0]["round"] == 2
    assert turns[0]["hp"] == 61


def test_a_new_floor_is_recorded_with_the_state_of_the_run(journal, tmp_path) -> None:
    journal.observe(_combat_state(floor=5))
    journal.observe(_combat_state(floor=6, hp=44))
    floors = [e for e in _events(journal, tmp_path) if e["event"] == "floor"]
    assert floors and floors[0]["floor"] == 6
    assert floors[0]["hp"] == 44


# -- act transitions are the run's actual progress, not a floor count --------

def test_reaching_act_2_is_recorded_as_an_act_clear(journal, tmp_path) -> None:
    """The act-1 boss was beaten when `act` increments from 1 to 2.

    Recorded as its own event so a clear can be derived from the journal
    alone -- independent of the run-end summary, which a kill -9 can lose.
    """
    journal.observe(_combat_state(act=1, floor=17))
    journal.observe({"type": "card_reward", "act": 2, "floor": 18,
                     "run_hp": 55, "cards": []})
    clears = [e for e in _events(journal, tmp_path) if e["event"] == "act_clear"]
    assert len(clears) == 1
    assert clears[0]["act_from"] == 1
    assert clears[0]["act_to"] == 2
    assert clears[0]["floor"] == 17


def test_the_first_act_seen_is_not_a_clear(journal, tmp_path) -> None:
    """Starting in act 1 is the run's opening state, not progress into it."""
    journal.observe(_combat_state(act=1, floor=1))
    clears = [e for e in _events(journal, tmp_path) if e["event"] == "act_clear"]
    assert clears == []


def test_an_act_that_does_not_change_does_not_emit(journal, tmp_path) -> None:
    journal.observe(_combat_state(act=1, floor=5))
    journal.observe(_combat_state(act=1, floor=6))
    journal.observe({"type": "card_reward", "act": 1, "floor": 6, "cards": []})
    clears = [e for e in _events(journal, tmp_path) if e["event"] == "act_clear"]
    assert clears == []


def test_act_clear_carries_the_hp_to_cross_the_boundary(journal, tmp_path) -> None:
    """Which HP the run had as it crossed into act 2 is the thing to know."""
    journal.observe(_combat_state(act=1, floor=17, hp=40))
    journal.observe({"type": "card_reward", "act": 2, "floor": 18,
                     "run_hp": 40, "run_max_hp": 80, "cards": []})
    clear = [e for e in _events(journal, tmp_path) if e["event"] == "act_clear"][0]
    assert clear["hp"] == 40
    assert clear["max_hp"] == 80


# -- the decisions -----------------------------------------------------------

def test_every_action_is_recorded_without_the_caller_doing_anything(journal, tmp_path) -> None:
    """The reason the client is wrapped rather than called at each site."""
    client = journal.wrap(FakeClient())
    journal.observe(_combat_state())

    client.play_card(1, 0)
    client.end_turn()

    events = _events(journal, tmp_path)
    played = [e for e in events if e["event"] == "card_played"]
    assert played[0]["card"] == "BASH+", "an upgraded card should say so"
    assert played[0]["target"] == 0
    assert any(e["event"] == "end_turn" for e in events)


def test_the_wrapped_client_still_sends_what_it_was_asked_to(journal) -> None:
    """Journalling must be observation, not interference."""
    fake = FakeClient()
    client = journal.wrap(fake)
    journal.observe(_combat_state())

    client.play_card(0, 0)
    client.end_turn()
    client.choose(2)
    client.skip()

    assert fake.sent == [
        ("play_card", 0, 0), ("end_turn",), ("choose", 2), ("skip",),
    ]


def test_the_wrapper_passes_through_anything_it_does_not_record(journal) -> None:
    client = journal.wrap(FakeClient())
    assert client.ping() is True


def test_a_choice_records_what_was_taken_and_what_was_offered(journal, tmp_path) -> None:
    client = journal.wrap(FakeClient())
    journal.observe({
        "type": "card_reward",
        "floor": 6,
        "run_hp": 55,
        "cards": [
            {"id": "ANGER"}, {"id": "POMMEL_STRIKE"}, {"id": "CLEAVE"},
        ],
    })
    client.choose(1)

    choice = [e for e in _events(journal, tmp_path) if e["event"] == "choice"][0]
    assert choice["chosen"] == "POMMEL_STRIKE"
    assert choice["offered"] == ["ANGER", "POMMEL_STRIKE", "CLEAVE"]
    assert choice["screen"] == "card_reward"


def test_a_skip_is_recorded_as_a_skip(journal, tmp_path) -> None:
    client = journal.wrap(FakeClient())
    journal.observe({"type": "card_reward", "cards": [{"id": "ANGER"}], "floor": 3})
    client.skip()

    choice = [e for e in _events(journal, tmp_path) if e["event"] == "choice"][0]
    assert choice["skipped"] is True
    assert choice["chosen"] is None


def test_a_potion_use_records_which_potion(journal, tmp_path) -> None:
    client = journal.wrap(FakeClient())
    journal.observe(_combat_state(potion_slots=["FirePotion", None, None]))
    client.use_potion(0)

    used = [e for e in _events(journal, tmp_path) if e["event"] == "potion_used"][0]
    assert used["potion"] == "FirePotion"


# -- runs are kept apart -----------------------------------------------------

def test_a_second_run_starts_clean(journal, tmp_path) -> None:
    journal.observe(_combat_state())
    journal.record_run_end({"floor": 9, "result": "terminated"})

    journal.start_run(2)
    journal.observe(_combat_state(hp=80))

    events = _events(journal, tmp_path)
    assert [e["event"] for e in events].count("run_start") == 2
    assert {e["run"] for e in events} == {1, 2}


# -- it can never cost a run -------------------------------------------------

def test_an_unwritable_path_degrades_to_no_journal(tmp_path) -> None:
    """A full disk or a bad path costs the record, never the run."""
    journal = RunJournal(tmp_path / "nope" / "\0bad" / "j.jsonl")
    journal.start_run(1)
    journal.observe(_combat_state())          # must not raise
    client = journal.wrap(FakeClient())
    client.end_turn()                          # must still send
    journal.close()


def test_no_journal_at_all_is_a_supported_configuration() -> None:
    journal = RunJournal(None)
    journal.start_run(1)
    journal.observe(_combat_state())
    fake = FakeClient()
    client = journal.wrap(fake)
    client.play_card(0, 0)
    assert fake.sent == [("play_card", 0, 0)]


def test_a_malformed_state_does_not_raise(journal) -> None:
    journal.observe({})
    journal.observe({"type": "combat_action"})     # no player, no hand
    journal.observe(None)                          # not a dict at all


def test_a_card_played_from_an_index_that_is_not_there_does_not_raise(journal) -> None:
    client = journal.wrap(FakeClient())
    journal.observe(_combat_state())
    client.play_card(99, 0)


# -- the summary reads it ----------------------------------------------------

def test_the_summariser_reads_a_real_journal(journal, tmp_path) -> None:
    from scripts.summarise_live_runs import load, report

    client = journal.wrap(FakeClient())
    journal.observe(_combat_state(hp=70))
    client.play_card(0, 0)
    client.end_turn()
    journal.observe({"type": "card_reward", "run_hp": 55, "floor": 5,
                     "cards": [{"id": "ANGER"}, {"id": "CLEAVE"}]})
    client.choose(0)
    journal.record_run_end({"floor": 9, "result": "terminated", "room_type": "Elite"})
    journal.close()

    text = report(load(tmp_path / "journal.jsonl"))
    assert "WHERE RUNS END" in text
    assert "ANGER" in text
    assert "STRIKE_IRONCLAD" in text


# -- naming the option, which is the whole value of the record ---------------

@pytest.mark.parametrize("option,expected", [
    # A card reward names the card in `id`.
    ({"id": "POMMEL_STRIKE", "action": "pick_card"}, "POMMEL_STRIKE"),
    ({"id": "BASH", "upgraded": True, "action": "pick_card"}, "BASH+"),
    # A rest site names the option in `id`.
    ({"id": "HEAL", "action": "rest_option", "label": "Rest"}, "HEAL"),
    # A shop puts the verb in `id` and the goods in `label`.
    ({"id": "buy_card", "action": "buy_card", "label": "CINDER"}, "CINDER"),
    ({"id": "buy_relic", "action": "buy_relic", "label": "NUNCHAKU"}, "NUNCHAKU"),
    # An event puts the verb in `id` and the choice in `label`.
    ({"id": "event_choice", "action": "event_choice", "label": "Proceed"}, "Proceed"),
    # A map node has neither, but does have a type.
    ({"row": 1, "col": 3, "type": "Elite"}, "Elite"),
])
def test_an_option_is_described_by_what_it_is(option, expected) -> None:
    from sts2_env.bridge.journal import _describe

    assert _describe(option) == expected


def test_shop_purchases_are_distinguishable_from_each_other(journal, tmp_path) -> None:
    """Every purchase logging as `buy_card` is the same as logging nothing."""
    client = journal.wrap(FakeClient())
    journal.observe({
        "type": "shop",
        "floor": 11,
        "options": [
            {"id": "leave_shop", "action": "leave_shop", "label": "Leave shop"},
            {"id": "buy_card", "action": "buy_card", "label": "CINDER"},
            {"id": "buy_relic", "action": "buy_relic", "label": "NUNCHAKU"},
        ],
    })
    client.choose(2)

    choice = [e for e in _events(journal, tmp_path) if e["event"] == "choice"][0]
    assert choice["chosen"] == "NUNCHAKU"
    assert choice["offered"] == ["Leave shop", "CINDER", "NUNCHAKU"]


def test_every_record_carries_the_session(journal, tmp_path) -> None:
    """Run numbers restart at 1 per session and the file is appended across them,
    so a file can hold runs 1,2,3,1,2,3. Without a session stamp, anything
    grouping by run number merges runs that have nothing to do with each other --
    which is what happened to the first session logged with this file."""
    journal.observe(_combat_state())
    journal.record_run_end({"floor": 9})
    events = _events(journal, tmp_path)
    assert all("session" in e for e in events)
    assert len({e["session"] for e in events}) == 1


def test_two_sessions_writing_one_file_stay_distinguishable(tmp_path) -> None:
    path = tmp_path / "shared.jsonl"
    first = RunJournal(path, model="m")
    first.start_run(1)
    first.observe(_combat_state())
    first.close()

    second = RunJournal(path, model="m")
    second.session = "different"          # a later process, counter back to 1
    second.start_run(1)
    second.observe(_combat_state())
    second.close()

    events = _read(path)
    assert len({(e["session"], e["run"]) for e in events}) == 2
