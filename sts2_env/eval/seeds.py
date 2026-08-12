"""Tuning seeds and holdout seeds, split once and never by eye.

PHASE_TWO.md section 3.4. Every sweep this project has run tuned and evaluated on
the same seeds, so a threshold that happened to suit those particular runs looked
exactly like a threshold that works.

THE SPLIT IS DETERMINISTIC AND NOT CONTIGUOUS
---------------------------------------------
`seed % 4 == 3` rather than "the last quarter". Contiguous blocks would be a
trap here: run length correlates with how well a run goes, so an ordered split
can put the deep runs disproportionately on one side. Interleaving cannot.

It is also stable. Adding runs later extends both halves in the same proportion
rather than reshuffling which seed is which, so a holdout result stays comparable
across sweeps of different sizes.

HOW TO USE IT
-------------
Tune on `tuning`, then report BOTH halves. A change that appears only on the
tuning half is noise that fitted itself to those seeds -- which is the specific
failure this exists to catch, and the reason to look at holdout before deciding
anything.

    from sts2_env.eval.seeds import split

    s = split(base=60000, count=400)
    ...run both arms on s.tuning and s.holdout...
    # accept only if the effect appears on both

Do not tune against holdout. Once a decision has been made using it, it has
been spent, and the next question needs a fresh base.
"""

from __future__ import annotations

from dataclasses import dataclass

#: One seed in four is held out. A quarter is enough to catch a tuning-half
#: artefact while leaving three quarters of the runs doing the tuning work.
HOLDOUT_MODULUS = 4
HOLDOUT_REMAINDER = 3


@dataclass(frozen=True)
class SeedSplit:
    """A seed range divided into a tuning half and a holdout half."""

    base: int
    count: int
    tuning: tuple[int, ...]
    holdout: tuple[int, ...]

    @property
    def all_seeds(self) -> tuple[int, ...]:
        return tuple(sorted(self.tuning + self.holdout))

    def describe(self) -> str:
        return (f"seeds {self.base}..{self.base + self.count - 1}: "
                f"{len(self.tuning)} tuning, {len(self.holdout)} holdout "
                f"(seed % {HOLDOUT_MODULUS} == {HOLDOUT_REMAINDER})")


def is_holdout(seed: int) -> bool:
    """Is this seed reserved for confirmation rather than tuning?"""
    return seed % HOLDOUT_MODULUS == HOLDOUT_REMAINDER


def split(base: int, count: int) -> SeedSplit:
    """Split `count` seeds starting at `base` into tuning and holdout."""
    if count < 0:
        raise ValueError("count must not be negative")
    seeds = range(base, base + count)
    return SeedSplit(
        base=base,
        count=count,
        tuning=tuple(s for s in seeds if not is_holdout(s)),
        holdout=tuple(s for s in seeds if is_holdout(s)),
    )
