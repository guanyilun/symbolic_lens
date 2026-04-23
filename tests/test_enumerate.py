"""Smoke test for the grammar enumerator.

(a) Dedup is idempotent: running the enumerator twice produces the same
    set of canonical keys.
(b) Candidate count for a tight grammar is a reasonable number.
(c) m_triples(3) respects the sum-zero constraint and the |m| bound.
"""
import symqe as sq


def test_m_triples_sum_zero_and_bounded():
    ts = sq.m_triples(3)
    for (a, b, c) in ts:
        assert a + b + c == 0
        assert abs(a) <= 3 and abs(b) <= 3 and abs(c) <= 3
    assert len(ts) == 37        # known count at |m|<=3

    ts2 = sq.m_triples(2)
    assert len(ts2) == 19


def test_leg_filters_count():
    # max_factors=1, ladder={0}, no spectra: {1, a(l1, 0)} = 2 items.
    items = list(sq.leg_filters(sq.l1, max_factors=1,
                                ladder_spins=(0,),
                                spectra={}))
    assert len(items) == 2
    # max_factors=2, ladders {0,+2,-2}, no spectra: 4 singletons + 6 pairs = 10.
    items = list(sq.leg_filters(sq.l1, max_factors=2,
                                ladder_spins=(0, 2, -2),
                                spectra={}))
    assert len(items) == 10


def test_enumerator_dedup_idempotent():
    def enumerate_tight():
        return list(sq.enumerate_candidates(
            m_max=2,
            coeffs=(1, -1),
            parity_factors=(1, sq.P),
            max_leg_factors=1,
            ladder_spins=(0, 2, -2),
            spectra_X={"CT": sq.CT},
            spectra_Y={"CT": sq.CT},
        ))

    first  = enumerate_tight()
    second = enumerate_tight()
    keys_first  = {sq.canonical_key(expr) for _, expr in first}
    keys_second = {sq.canonical_key(expr) for _, expr in second}
    assert keys_first == keys_second
    print(f"candidates (tight grammar): {len(first)}")
    assert 10 < len(first) < 5000


if __name__ == "__main__":
    test_m_triples_sum_zero_and_bounded()
    test_leg_filters_count()
    test_enumerator_dedup_idempotent()
    print("OK")
