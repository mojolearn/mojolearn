"""Filtered public search owns its candidate buffers through native compaction."""
import numpy as np
import pytest


@pytest.mark.parametrize('rows,features,queries', [(257, 7, 17), (263, 9, 65)])
def test_filtered_cagra_matches_live_candidate_oracle_and_query_batches(rows, features, queries):
    from mojolearn import CagraIndex
    rng = np.random.default_rng(719)
    data = rng.standard_normal((rows, features)).astype(np.float32)
    query = rng.standard_normal((queries, features)).astype(np.float32)
    keep = (np.arange(rows) % 3) != 0
    length, count = 23, 5
    model = CagraIndex(graph_degree=8, intermediate_graph_degree=16,
                       itopk_size=length, n_neighbors=length, n_seeds=13).fit(data)
    candidate_distances, candidate_ids = (np.asarray(value).copy() for value in model.search(query))
    expected_distances = np.full((queries, count), np.inf, dtype=np.float32)
    expected_ids = np.full((queries, count), -1, dtype=np.int32)
    for row in range(queries):
        ids = candidate_ids[row]
        valid = (ids >= 0) & (ids < rows)
        retained = np.flatnonzero(valid & keep[np.clip(ids, 0, rows - 1)])[:count]
        expected_distances[row, :len(retained)] = candidate_distances[row, retained]
        expected_ids[row, :len(retained)] = ids[retained]
    model.n_neighbors = count
    whole = tuple(np.asarray(value).copy() for value in model.search(query, filter=keep))
    assert np.array_equal(whole[0], expected_distances)
    assert np.array_equal(whole[1], expected_ids)
    for row in (0, queries // 2, queries - 1):
        alone = model.search(query[row:row + 1], filter=keep)
        assert np.array_equal(np.asarray(alone[0])[0], whole[0][row])
        assert np.array_equal(np.asarray(alone[1])[0], whole[1][row])
