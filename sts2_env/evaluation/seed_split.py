"""The permanent tuning/holdout split for offline seed sweeps.

Sweeps used to tune and evaluate on the same seeds, so a weight set that fit
the noise of its own sample looked like progress; `PHASE_TWO.md` section 3.4
is the rule this module implements. The split is a function of the seed ALONE
-- stable across every script, session and policy version -- so a result can
never be re-sliced into significance after the fact, and rows written by
different sweeps on different days still partition the same way.

The convention is `seed % 4 == 3` for holdout, one quarter of every range:
large enough that a 400-seed sweep leaves 100 holdout runs, and documented in
the weekend notes as uncorrelated with the act 1 boss variant (49.6% vs 50.8%
underdocks over 4000 seeds), so the halves are exchangeable except for size.

The rule for reading it: a change must improve BOTH halves. An effect that
appears only on the tuning half is the sample fitting itself, and is reported
as noise no matter how good it looks.
"""

from __future__ import annotations

from typing import Iterable, TypeVar

HOLDOUT_MOD = 4
HOLDOUT_RESIDUE = 3

T = TypeVar("T")


def is_holdout(seed: int) -> bool:
    """True when the seed belongs to the holdout half."""
    return int(seed) % HOLDOUT_MOD == HOLDOUT_RESIDUE


def partition(rows: Iterable[T], seed_of) -> tuple[list[T], list[T]]:
    """Split rows into (tuning, holdout) by each row's seed."""
    tuning: list[T] = []
    holdout: list[T] = []
    for row in rows:
        (holdout if is_holdout(seed_of(row)) else tuning).append(row)
    return tuning, holdout
