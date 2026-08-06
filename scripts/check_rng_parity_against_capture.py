"""Check the simulator's RNG against a captured live session, and show what fixes it.

Run against a `--capture-raw` file from a live session:

    python scripts/check_rng_parity_against_capture.py output/bridge_protocol_sample.jsonl

For every captured `combat_action`, this rebuilds the fight from the bridge's
own `encounter` + `encounter_seed` via `CombatSituation.from_bridge_state(...)
.to_combat()` and compares the enemy `max_hp` the simulator rolls against the
`max_hp` the live game reported in that same payload. The game told us both the
seed and the answer, so this is a closed-book check with no room to argue.

Why this exists
---------------
The first capture ever taken (2026-08-06) failed it 25 times out of 25. The
cause is not the monster HP tables -- a seed sweep shows the simulator can
produce the live value, it just does not produce it for the seed the game
used. It is the generator itself:

* The game's `Rng(ulong seed)` wraps `MegaRandom`, which is **xoshiro256\\*\\***
  with state seeded by four **Splitmix64** draws from the full 64-bit seed
  (`MegaRandom.cs:69,99,168`).
* The simulator's `Rng.__init__` masks the seed to 32 bits (`_to_uint32` then
  `_to_int32`) and feeds `_DotNetCompatRandom`, a `System.Random` clone --
  a subtractive lagged-Fibonacci generator. A different algorithm entirely.

Those agree on nothing. The 32-bit truncation is the more visible half (live
encounter seeds are 64-bit values like -5080831859460911205), but even an
in-range seed would diverge, because the streams are unrelated.

`MegaRandomShim` below is the game's generator, transcribed from the decompile
and verified against captured live data. Passing `--fixed` swaps it in and
re-runs the same comparison, which is how the 0/25 -> 18/25 result in
`docs/PARITY_GAPS.md` was produced.

**This script does not modify the simulator.** Replacing the core RNG changes
every seeded behaviour in ~50,000 LoC and is a decision to take deliberately;
see the writeup in `docs/PARITY_GAPS.md` before doing it.
"""

from __future__ import annotations

import argparse
import collections
import sys

M64 = 0xFFFFFFFFFFFFFFFF


def _rotl(x: int, k: int) -> int:
    return ((x << k) | (x >> (64 - k))) & M64


def splitmix64(x: int) -> tuple[int, int]:
    """One Splitmix64 step. Returns (advanced state, output).

    Transcribed from `MegaRandom.Splitmix64`, which takes its state by ref --
    hence returning both halves rather than mutating.
    """
    x = (x + 0x9E3779B97F4A7C15) & M64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return x, z ^ (z >> 31)


class MegaRandomShim:
    """xoshiro256** as the game implements it, with the simulator's sample() API.

    Drop-in for `_DotNetCompatRandom`: the simulator's `next_int` is already
    `int(sample() * range) + low`, which is exactly the game's
    `NextInner(maxValue) = (int)(NextDouble() * maxValue)`. Only the underlying
    stream differs, so `sample()` is the whole seam.
    """

    def __init__(self, seed: int):
        s = seed & M64
        state = []
        for _ in range(4):
            s, value = splitmix64(s)
            state.append(value)
        self._s0, self._s1, self._s2, self._s3 = state

    def next_ulong(self) -> int:
        s0, s1, s2, s3 = self._s0, self._s1, self._s2, self._s3
        result = (_rotl((s1 * 5) & M64, 7) * 9) & M64
        shifted = (s1 << 17) & M64
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= shifted
        s3 = _rotl(s3, 45)
        self._s0, self._s1, self._s2, self._s3 = s0, s1, s2, s3
        return result

    def sample(self) -> float:
        # MegaRandom.NextDouble: (NextULongInner() >> 11) * 2^-53
        return (self.next_ulong() >> 11) * 1.1102230246251565e-16

    def get_sample_for_large_range(self) -> float:
        return self.sample()


def _install_megarandom() -> None:
    """Repoint `Rng` at the game's generator, keeping the full 64-bit seed."""
    import sts2_env.core.rng as rngmod

    def patched_init(self, seed: int = 0, name: str | None = None, counter: int = 0):
        base = seed & M64
        if name is not None:
            base = (base + rngmod.deterministic_hash_code(name)) & M64
        self._base_seed = base
        self._seed = base
        self._rng = MegaRandomShim(base)
        self._counter = 0
        self.fast_forward_counter(counter)

    rngmod.Rng.__init__ = patched_init


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", help="A --capture-raw JSONL from a live session")
    parser.add_argument("--fixed", action="store_true",
                        help="Swap in MegaRandom + the full 64-bit seed first")
    args = parser.parse_args()

    if args.fixed:
        _install_megarandom()

    from sts2_env.bridge.raw_capture import load_capture
    from sts2_env.search.situation import CombatSituation

    states = load_capture(args.capture)
    combat_states = [s for s in states if s.get("type") == "combat_action"]
    if not combat_states:
        print(f"No combat_action states in {args.capture}.")
        return 1

    print(f"{len(combat_states)} combat_action states  "
          f"({'MegaRandom + 64-bit seed' if args.fixed else 'simulator as-is'})\n")

    per_encounter: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    matches = 0
    for state in combat_states:
        encounter = state.get("encounter", "?")
        live = [e.get("max_hp") for e in state.get("enemies", [])]
        try:
            combat = CombatSituation.from_bridge_state(state).to_combat()
            sim = [e.max_hp for e in combat.enemies]
        except Exception as exc:  # a raise is itself a parity answer
            per_encounter[encounter][f"RAISED {type(exc).__name__}"] += 1
            continue
        agreed = live == sim
        matches += agreed
        per_encounter[encounter][
            "match" if agreed else f"live={live} sim={sim}"
        ] += 1

    for encounter, outcomes in sorted(per_encounter.items()):
        print(f"  {encounter}")
        for outcome, n in outcomes.most_common():
            print(f"      {n:>3}x  {outcome}")

    print(f"\nmax_hp parity: {matches}/{len(combat_states)}")
    if not args.fixed and matches < len(combat_states):
        print("\nRe-run with --fixed to see the same check under the game's own "
              "generator.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
