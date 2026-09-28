"""Oracle and fault-detection checks only; never imports native bindings."""
import numpy as np
import pytest
from check_radix_public import KS, ROWS, assert_answer, fixtures, oracle


def test_fixture_covers_rank_capacity_boundary_with_planted_ties():
    assert KS == (1024, 1025, 2000)
    for name, index, queries in fixtures():
        assert index.shape == (ROWS, 2) and ROWS > max(KS)
        assert queries.shape == (3, 2)
        distances, order = oracle(index, queries, max(KS))
        for row, query in enumerate(queries):
            # Python tuple sort independently checks NumPy's stable tie rule.
            expected = sorted(range(ROWS), key=lambda i: (sum((int(a)-int(b))**2
                              for a, b in zip(index[i], query)), i))[:max(KS)]
            assert order[row].tolist() == expected
        if name == 'all-ties':
            assert np.all(order == np.arange(max(KS)))
        assert_answer((distances.copy(), order.copy()), (distances, order), name)


def test_swapped_tied_winners_are_detected():
    _, index, queries = list(fixtures())[1]
    distances, order = oracle(index, queries, 1025)
    wrong = order.copy()
    wrong[:, [1023, 1024]] = wrong[:, [1024, 1023]]
    with pytest.raises(AssertionError, match='index order'):
        assert_answer((distances, wrong), (distances, order), 'round boundary')


def test_one_distance_bit_changed_is_detected():
    _, index, queries = next(fixtures())
    distances, order = oracle(index, queries, 1024)
    wrong = distances.copy()
    wrong.view(np.uint32)[0, 0] ^= np.uint32(1)
    with pytest.raises(AssertionError, match='distance bits'):
        assert_answer((wrong, order), (distances, order), 'one bit')
