"""RNG parity tests for the game's seeded random wrapper.

These used to assert magic numbers — `Rng(seed, "up_front").seed == 1_840_945_279`
and a five-value `System.Random` sequence. Every one of them passed, and every
one of them was wrong, because they were produced by running the implementation
they were checking. The first live capture (2026-08-06) compared the simulator
to the game instead and failed 25 times out of 25; see `docs/PARITY_GAPS.md`.

So the tests below anchor to things checkable from outside this repository:

* **XxHash64 reference vectors.** Published values, independent of this project.
  `StringHelper.GetDeterministicHashCode` is `XxHash64.HashToUInt64` over UTF-8.
* **The derivation rules transcribed from the decompile**, asserted as rules
  rather than as their outputs — `Rng(seed, name)` is `Rng(seed + hash(name))`
  (`Rng.cs:53`), and a run stream is `Rng(Seed, snake_case(type))`
  (`RunRngSet.cs:139`).
* **Structural properties** — determinism, 64-bit seed preservation, counter
  behaviour — which hold regardless of the generator and still catch a
  regression without pinning an unverifiable constant.

A value here should be traceable to the game or to a published spec. If it can
only be produced by running this code, it does not belong.
"""

from sts2_env.core.rng import (
    Rng,
    ULONG_MASK,
    deterministic_hash_code,
    deterministic_hash_code_old,
    xxhash64,
)
from sts2_env.run.run_state import RunRngSet, RunState


# --- the hash -------------------------------------------------------------


def test_xxhash64_matches_published_reference_vectors():
    """External anchor: these are the reference vectors, not our output."""
    assert xxhash64(b"") == 0xEF46DB3751D8E999
    assert xxhash64(b"a") == 0xD24EC4F1A98C6E5B
    assert xxhash64(b"abc") == 0x44BC2CF5AD770999


def test_deterministic_hash_code_is_xxhash64_over_utf8():
    """`StringHelper.cs:139` -- XxHash64 of the UTF-8 bytes, seed 0."""
    for text in ("up_front", "shuffle", "niche", "SEAPUNK_WEAK", ""):
        assert deterministic_hash_code(text) == xxhash64(text.encode("utf-8"))


def test_deterministic_hash_code_returns_a_full_unsigned_64_bit_value():
    """The C# returns `ulong`. Truncating to 32 is what broke the seeds."""
    value = deterministic_hash_code("combat_card_generation")
    assert 0 <= value <= ULONG_MASK
    assert value > 0xFFFFFFFF, "a 64-bit hash that fits in 32 bits is suspicious"


def test_the_old_hash_is_kept_because_the_game_still_uses_it():
    """`RunRngSet.cs:111` uses it for an `"old"`-prefixed string seed.

    Deprecated, not dead. Pinned so the two cannot be confused again.
    """
    assert deterministic_hash_code_old("42") != deterministic_hash_code("42")
    assert -(2 ** 31) <= deterministic_hash_code_old("42") < 2 ** 31


# --- seed derivation ------------------------------------------------------


def test_a_named_stream_is_the_seed_plus_the_hash_of_its_name():
    """`Rng.cs:53`: `Rng(ulong seed, string name) : this(seed + hash(name))`.

    Asserted as the rule. The old test asserted thirteen outputs of the rule,
    which is the same claim with an unverifiable constant attached to each.
    """
    seed = 20_240_806
    for name in ("up_front", "shuffle", "niche", "monster_ai", "act_1_map"):
        expected = (seed + deterministic_hash_code(name)) & ULONG_MASK
        assert Rng(seed, name).seed == expected


def test_named_streams_are_distinct_from_each_other_and_from_the_bare_seed():
    seed = 42
    names = ("up_front", "shuffle", "combat_targets", "monster_ai", "niche")
    seeds = {Rng(seed, n).seed for n in names}
    assert len(seeds) == len(names)
    assert Rng(seed).seed not in seeds


def test_run_rng_set_exposes_the_game_named_streams():
    """Structure, not values: the streams exist and are separately seeded."""
    streams = RunRngSet(42)
    named = (
        streams.up_front, streams.shuffle, streams.combat_card_generation,
        streams.combat_potion_generation, streams.combat_card_selection,
        streams.combat_energy_costs, streams.combat_targets,
        streams.monster_ai, streams.combat_orbs, streams.treasure_room,
        streams.rewards, streams.shops, streams.transformations,
    )
    assert len({s.seed for s in named}) == len(named)


def test_event_rng_is_derived_and_distinct_per_event():
    from sts2_env.events.shared import AromaOfChaos, PunchOff

    run_state = RunState(seed=42, character_id="Ironclad")
    aroma, punch = AromaOfChaos(), PunchOff()

    assert aroma.event_entry() == "AROMA_OF_CHAOS"
    assert punch.event_entry() == "PUNCH_OFF"
    assert (aroma.create_event_rng(run_state).seed
            != punch.create_event_rng(run_state).seed)
    # Same event, same run, same stream -- the derivation is a function.
    assert (aroma.create_event_rng(run_state).seed
            == aroma.create_event_rng(run_state).seed)


# --- the generator --------------------------------------------------------


def test_a_seed_wider_than_int32_is_not_truncated():
    """The bug that made this file necessary.

    Live encounter seeds are 64-bit values like -5080831859460911205. The old
    `Rng.__init__` masked to 32 bits before the generator ever ran, so two
    seeds differing only above bit 32 produced identical streams.
    """
    low = 0x0000_0000_DEAD_BEEF
    high = 0x1234_5678_DEAD_BEEF
    assert Rng(low).seed != Rng(high).seed
    assert [Rng(low).next_int(0, 10_000) for _ in range(5)] != [
        Rng(high).next_int(0, 10_000) for _ in range(5)
    ]


def test_a_negative_seed_is_stored_as_its_unsigned_64_bit_pattern():
    """The bridge sends `(long)seed`, so large seeds arrive negative."""
    signed = -5_080_831_859_460_911_205
    assert Rng(signed).seed == signed & ULONG_MASK
    assert Rng(signed).seed == Rng(signed & ULONG_MASK).seed


def test_the_same_seed_gives_the_same_stream():
    a = [Rng(12345).next_int(0, 1_000) for _ in range(20)]
    b = [Rng(12345).next_int(0, 1_000) for _ in range(20)]
    assert a == b


def test_different_seeds_give_different_streams():
    a = [Rng(1).next_int(0, 1_000_000) for _ in range(10)]
    b = [Rng(2).next_int(0, 1_000_000) for _ in range(10)]
    assert a != b


def test_next_int_stays_in_its_inclusive_range():
    rng = Rng(7)
    for _ in range(500):
        assert 3 <= rng.next_int(3, 9) <= 9


def test_next_int_reaches_both_ends_of_a_small_range():
    """A generator that never returns the top of its range is the classic
    off-by-one in `(int)(NextDouble() * range)`; pin that it does not."""
    rng = Rng(99)
    seen = {rng.next_int(0, 2) for _ in range(300)}
    assert seen == {0, 1, 2}


def test_the_counter_advances_once_per_draw_and_can_fast_forward():
    rng = Rng(5)
    for _ in range(4):
        rng.next_int(0, 100)
    assert rng.counter == 4

    forwarded = Rng(5, counter=4)
    assert forwarded.counter == 4
    assert forwarded.next_int(0, 100) == rng.next_int(0, 100)


def test_shuffle_is_deterministic_and_a_permutation():
    values = [1, 2, 3, 4, 5]
    shuffled = list(values)
    Rng(42).shuffle(shuffled)

    assert sorted(shuffled) == sorted(values)
    again = list(values)
    Rng(42).shuffle(again)
    assert again == shuffled
