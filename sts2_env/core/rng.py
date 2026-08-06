"""Deterministic RNG matching the game's seeded RNG wrapper.

The game's `Rng(ulong seed)` wraps `MegaRandom`: **xoshiro256\\*\\*** with state
seeded by four **Splitmix64** draws from the full 64-bit seed
(`MegaRandom.cs:69,99,168`). Named streams come from
`Rng(ulong seed, string name) : this(seed + GetDeterministicHashCode(name))`,
where that hash is **XxHash64** over the UTF-8 bytes (`StringHelper.cs:139`).

This file used to implement something else entirely: a `System.Random` clone
(subtractive lagged Fibonacci) seeded from the low 32 bits, with the game's
*deprecated* string hash. Every seeded test passed, because every seeded test
compared the simulator against itself. The first live capture (2026-08-06)
compared it against the game and failed 25 times out of 25 -- Nibbits rolling
45 where the game said 43, from the game's own seed. See
`docs/PARITY_GAPS.md`.

`_DotNetCompatRandom` and `deterministic_hash_code_old` are kept below, because
the game keeps its old hash too: `RunRngSet.cs:111` still uses it for a string
seed prefixed `"old"`, and a save migration uses it. They are legacy paths, not
the default.
"""

from __future__ import annotations

import math


INT_MAX = 2_147_483_647
INT_MIN = -2_147_483_648
INT_MAX_EXCLUSIVE = INT_MAX + 1
UINT_MASK = 0xFFFFFFFF
ULONG_MASK = 0xFFFFFFFFFFFFFFFF


def _to_int32(value: int) -> int:
    value &= UINT_MASK
    if value >= 0x80000000:
        value -= 0x100000000
    return value


def _to_uint32(value: int) -> int:
    return value & UINT_MASK


def _rotl64(value: int, bits: int) -> int:
    return ((value << bits) | (value >> (64 - bits))) & ULONG_MASK


_XXH_P1 = 0x9E3779B185EBCA87
_XXH_P2 = 0xC2B2AE3D27D4EB4F
_XXH_P3 = 0x165667B19E3779F9
_XXH_P4 = 0x85EBCA77C2B2AE63
_XXH_P5 = 0x27D4EB2F165667C5


def _xxh_round(acc: int, value: int) -> int:
    acc = (acc + value * _XXH_P2) & ULONG_MASK
    return (_rotl64(acc, 31) * _XXH_P1) & ULONG_MASK


def _xxh_merge(acc: int, value: int) -> int:
    acc ^= _xxh_round(0, value)
    return (acc * _XXH_P1 + _XXH_P4) & ULONG_MASK


def xxhash64(data: bytes, seed: int = 0) -> int:
    """XxHash64, as `System.IO.Hashing.XxHash64.HashToUInt64` computes it.

    Validated against the reference vectors for the empty string, the 1-byte
    tail path and the >=32-byte striped path.
    """
    length = len(data)
    index = 0
    if length >= 32:
        v1 = (seed + _XXH_P1 + _XXH_P2) & ULONG_MASK
        v2 = (seed + _XXH_P2) & ULONG_MASK
        v3 = seed & ULONG_MASK
        v4 = (seed - _XXH_P1) & ULONG_MASK
        while index + 32 <= length:
            v1 = _xxh_round(v1, int.from_bytes(data[index:index + 8], "little"))
            index += 8
            v2 = _xxh_round(v2, int.from_bytes(data[index:index + 8], "little"))
            index += 8
            v3 = _xxh_round(v3, int.from_bytes(data[index:index + 8], "little"))
            index += 8
            v4 = _xxh_round(v4, int.from_bytes(data[index:index + 8], "little"))
            index += 8
        acc = (
            _rotl64(v1, 1) + _rotl64(v2, 7) + _rotl64(v3, 12) + _rotl64(v4, 18)
        ) & ULONG_MASK
        for v in (v1, v2, v3, v4):
            acc = _xxh_merge(acc, v)
    else:
        acc = (seed + _XXH_P5) & ULONG_MASK

    acc = (acc + length) & ULONG_MASK
    while index + 8 <= length:
        acc ^= _xxh_round(0, int.from_bytes(data[index:index + 8], "little"))
        acc = (_rotl64(acc, 27) * _XXH_P1 + _XXH_P4) & ULONG_MASK
        index += 8
    if index + 4 <= length:
        acc ^= (int.from_bytes(data[index:index + 4], "little") * _XXH_P1) & ULONG_MASK
        acc = (_rotl64(acc, 23) * _XXH_P2 + _XXH_P3) & ULONG_MASK
        index += 4
    while index < length:
        acc ^= (data[index] * _XXH_P5) & ULONG_MASK
        acc = (_rotl64(acc, 11) * _XXH_P1) & ULONG_MASK
        index += 1

    acc ^= acc >> 33
    acc = (acc * _XXH_P2) & ULONG_MASK
    acc ^= acc >> 29
    acc = (acc * _XXH_P3) & ULONG_MASK
    acc ^= acc >> 32
    return acc


# Stream names are a small fixed vocabulary re-hashed on every RunRngSet, and a
# pure-Python XxHash64 is not free, so memoise. Bounded by the name set, which
# is enumerable from RunRngType.
_HASH_CACHE: dict[str, int] = {}


def deterministic_hash_code(text: str) -> int:
    """Match `StringHelper.GetDeterministicHashCode` -- XxHash64 over UTF-8.

    Returns the full unsigned 64-bit value, as the C# does. The game's own
    comment marks the 32-bit version this used to be as
    "should not be used for any new code".
    """
    cached = _HASH_CACHE.get(text)
    if cached is None:
        cached = xxhash64(text.encode("utf-8"))
        _HASH_CACHE[text] = cached
    return cached


def deterministic_hash_code_old(text: str) -> int:
    """Match `StringHelper.GetDeterministicHashCodeOld` -- the 32-bit predecessor.

    Still live in the game for a string seed prefixed `"old"`
    (`RunRngSet.cs:111`) and in the V18->V19 save migration. Not the default.
    """
    num = 352654597
    num2 = num
    for index in range(0, len(text), 2):
        num = _to_int32(_to_int32(num << 5) + num)
        num = _to_int32(num ^ ord(text[index]))
        if index == len(text) - 1:
            break
        num2 = _to_int32(_to_int32(num2 << 5) + num2)
        num2 = _to_int32(num2 ^ ord(text[index + 1]))
    return _to_int32(num + _to_int32(num2 * 1_566_083_941))


class _DotNetCompatRandom:
    """System.Random seeded implementation used by .NET for Random(int)."""

    def __init__(self, seed: int):
        self._seed_array = [0] * 56
        self._inext = 0
        self._inextp = 21
        self._initialize(seed)

    def _initialize(self, seed: int) -> None:
        subtraction = INT_MAX if seed == INT_MIN else abs(seed)
        mj = _to_int32(161_803_398 - subtraction)
        self._seed_array[55] = mj
        mk = 1
        ii = 0
        for i in range(1, 55):
            ii += 21
            if ii >= 55:
                ii -= 55
            self._seed_array[ii] = mk
            mk = _to_int32(mj - mk)
            if mk < 0:
                mk += INT_MAX
            mj = self._seed_array[ii]
        for _ in range(1, 5):
            for i in range(1, 56):
                n = i + 30
                if n >= 55:
                    n -= 55
                self._seed_array[i] = _to_int32(self._seed_array[i] - self._seed_array[1 + n])
                if self._seed_array[i] < 0:
                    self._seed_array[i] += INT_MAX

    def internal_sample(self) -> int:
        loc_inext = self._inext + 1
        if loc_inext >= 56:
            loc_inext = 1
        loc_inextp = self._inextp + 1
        if loc_inextp >= 56:
            loc_inextp = 1

        ret = _to_int32(self._seed_array[loc_inext] - self._seed_array[loc_inextp])
        if ret == INT_MAX:
            ret -= 1
        if ret < 0:
            ret += INT_MAX
        self._seed_array[loc_inext] = ret
        self._inext = loc_inext
        self._inextp = loc_inextp
        return ret

    def sample(self) -> float:
        return self.internal_sample() * (1.0 / INT_MAX)

    def get_sample_for_large_range(self) -> float:
        result = self.internal_sample()
        if self.internal_sample() % 2 == 0:
            result = -result
        value = result + (INT_MAX - 1)
        return value / (2 * INT_MAX - 1)


def _splitmix64(state: int) -> tuple[int, int]:
    """One Splitmix64 step: returns (advanced state, output).

    The C# takes its state by ref (`MegaRandom.Splitmix64`), hence returning
    both halves rather than mutating.
    """
    state = (state + 0x9E3779B97F4A7C15) & ULONG_MASK
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & ULONG_MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & ULONG_MASK
    return state, z ^ (z >> 31)


class MegaRandom:
    """xoshiro256\\*\\* as the game implements it (`MegaRandom.cs`).

    Exposes the same three methods `_DotNetCompatRandom` did, so `Rng` needs no
    other change. `get_sample_for_large_range` is `sample` here because the
    game's `NextInner(long)` and `NextInner(int)` both just scale `NextDouble`
    -- the split only existed to work around `System.Random`'s resolution.
    """

    __slots__ = ("_s0", "_s1", "_s2", "_s3")

    def __init__(self, seed: int):
        state = seed & ULONG_MASK
        values = []
        for _ in range(4):
            state, value = _splitmix64(state)
            values.append(value)
        self._s0, self._s1, self._s2, self._s3 = values

    def next_ulong(self) -> int:
        s0, s1, s2, s3 = self._s0, self._s1, self._s2, self._s3
        result = (_rotl64((s1 * 5) & ULONG_MASK, 7) * 9) & ULONG_MASK
        shifted = (s1 << 17) & ULONG_MASK
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= shifted
        s3 = _rotl64(s3, 45)
        self._s0, self._s1, self._s2, self._s3 = s0, s1, s2, s3
        return result

    def internal_sample(self) -> int:
        """Advance the stream once. Used only by counter fast-forward."""
        return self.next_ulong()

    def sample(self) -> float:
        """`MegaRandom.NextDouble`: the top 53 bits scaled into [0, 1)."""
        return (self.next_ulong() >> 11) * 1.1102230246251565e-16

    def get_sample_for_large_range(self) -> float:
        return self.sample()


class Rng:
    """Seeded random number generator matching the game's `Rng` class.

    `next_int(low, high)` keeps this project's inclusive upper bound, which maps
    onto the game's `Next(min, max)` exclusive form as
    `NextInner(range) = (int)(NextDouble() * range)` -- identical arithmetic, so
    only the underlying stream ever had to change.
    """

    def __init__(self, seed: int = 0, name: str | None = None, counter: int = 0):
        # Full 64 bits. The old code masked to 32 here, which silently threw
        # away half of every live encounter seed (they are values like
        # -5080831859460911205) before the generator ever ran.
        self._base_seed = seed & ULONG_MASK
        if name is None:
            self._seed = self._base_seed
        else:
            self._seed = (self._base_seed + deterministic_hash_code(name)) & ULONG_MASK
        self._rng = MegaRandom(self._seed)
        self._counter = 0
        self.fast_forward_counter(counter)

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def counter(self) -> int:
        return self._counter

    def fast_forward_counter(self, target_count: int) -> None:
        if self._counter > target_count:
            raise ValueError(
                f"Cannot fast-forward an Rng counter to a lower number "
                f"(current = {self._counter}, target = {target_count})"
            )
        while self._counter < target_count:
            self._counter += 1
            self._rng.internal_sample()

    def next_int(self, low: int, high: int) -> int:
        """Return random int in [low, high] inclusive."""
        if low > high:
            raise ValueError("low must be <= high")
        self._counter += 1
        exclusive_high = high + 1
        range_size = exclusive_high - low
        if range_size <= INT_MAX:
            return int(self._rng.sample() * range_size) + low
        return int(self._rng.get_sample_for_large_range() * range_size) + low

    def next_int_exclusive(self, low: int, high: int) -> int:
        """Return random int in [low, high), matching C# Random.Next(min, max)."""
        if low >= high:
            raise ValueError("low must be < high")
        return self.next_int(low, high - 1)

    def next_bool(self) -> bool:
        """Return random bool matching the game's Rng.NextBool."""
        return self.next_int(0, 1) == 0

    def random_int(self, low: int, high: int) -> int:
        """Backward-compatible alias for ``next_int``."""
        return self.next_int(low, high)

    def next_float(self, upper: float = 1.0) -> float:
        """Return random float in [0, upper)."""
        self._counter += 1
        return self._rng.sample() * upper

    def shuffle(self, lst: list) -> None:
        """In-place shuffle."""
        for index in range(len(lst) - 1, 0, -1):
            swap_index = self.next_int(0, index)
            lst[index], lst[swap_index] = lst[swap_index], lst[index]

    def choice(self, lst: list):
        """Pick a random element."""
        if not lst:
            raise IndexError("Cannot choose from an empty list")
        return lst[self.next_int(0, len(lst) - 1)]

    def sample(self, lst: list, k: int) -> list:
        """Pick k distinct elements."""
        if k < 0 or k > len(lst):
            raise ValueError("Sample larger than population or is negative")
        pool = list(lst)
        result = []
        for _ in range(k):
            index = self.next_int(0, len(pool) - 1)
            result.append(pool.pop(index))
        return result

    def next_float_range(self, low: float, high: float) -> float:
        """Return random float in [low, high)."""
        if low > high:
            raise ValueError("low must be <= high")
        self._counter += 1
        return low + self._rng.sample() * (high - low)

    def next_gaussian_int(self, mean: float, stddev: float, min_val: int, max_val: int) -> int:
        """Return a gaussian-distributed int in [min_val, max_val].

        Matches C# Rng.NextGaussianInt: uses rejection sampling (re-rolls
        until the result is within range) rather than clamping.
        """
        while True:
            u1 = 1.0 - self._rng.sample()
            u2 = 1.0 - self._rng.sample()
            z = math.sqrt(-2.0 * math.log(u1)) * math.sin(2.0 * math.pi * u2)
            val = round(mean + stddev * z)
            if min_val <= val <= max_val:
                return val

    def fork(self) -> Rng:
        """Create a child RNG with a derived seed."""
        return Rng(self.next_int(0, INT_MAX_EXCLUSIVE))
