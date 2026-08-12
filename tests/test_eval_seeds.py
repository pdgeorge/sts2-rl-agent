"""The tuning/holdout split must be deterministic, interleaved, and stable."""

from sts2_env.eval.seeds import HOLDOUT_MODULUS, is_holdout, split


def test_the_split_is_exhaustive_and_disjoint():
    s = split(base=60000, count=400)
    assert set(s.tuning) | set(s.holdout) == set(range(60000, 60400))
    assert not (set(s.tuning) & set(s.holdout))
    assert len(s.holdout) == 100


def test_the_split_is_interleaved_not_contiguous():
    """Contiguous blocks would bias the halves.

    Run length correlates with how well a run goes -- a run that clears act 1
    plays act 2 as well -- so an ordered split can put the deep runs
    disproportionately on one side. Interleaving cannot.
    """
    s = split(base=0, count=40)
    # every window of MODULUS consecutive seeds contains exactly one holdout
    for start in range(0, 40 - HOLDOUT_MODULUS + 1):
        window = range(start, start + HOLDOUT_MODULUS)
        assert sum(1 for x in window if is_holdout(x)) == 1


def test_the_split_is_stable_when_the_range_grows():
    """A seed does not change sides when the sweep gets bigger.

    Otherwise a holdout result computed at n=200 is not comparable with one at
    n=400, and the guarantee is worthless across sweeps.
    """
    small = split(base=60000, count=100)
    large = split(base=60000, count=400)
    assert set(small.holdout) <= set(large.holdout)
    assert set(small.tuning) <= set(large.tuning)


def test_an_empty_range_is_allowed():
    s = split(base=5, count=0)
    assert s.tuning == () and s.holdout == ()
