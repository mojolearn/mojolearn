"""SAGE max aggregation's input gradient must equal its independent adjoint."""
import numpy as np
import pytest
from mojolearn import SAGEConv


@pytest.mark.parametrize("nodes", [37, 53])
@pytest.mark.parametrize("ties", [False, True])
def test_max_backward_matches_independent_tie_scatter(nodes, ties):
    features = 5
    x = np.arange(nodes * features, dtype=np.float32).reshape(nodes, features) * np.float32(0.125) - np.float32(1)
    if ties:
        x[1] = x[0]
    edges = np.array([[0, 1, 2, 3, 4, 1, 0, 3], [2, 2, 3, 4, 0, 3, 4, 0]], dtype=np.int32)
    layer = SAGEConv(features, features, aggr="max", root_weight=False, bias=False, random_state=61)
    layer.lin_l.weight_ = np.eye(features, dtype=np.float32)
    y = layer.forward(x, edges)
    upstream = (np.arange(nodes * features, dtype=np.float32).reshape(nodes, features) + np.float32(1)) / np.float32(8)
    actual = layer.backward(upstream)
    expected_y = np.zeros_like(x)
    expected_dx = np.zeros_like(x)
    for target in range(nodes):
        sources = edges[0, edges[1] == target]
        if not len(sources):
            continue
        maxima = np.max(x[sources], axis=0)
        expected_y[target] = maxima
        for feature in range(features):
            winners = sources[x[sources, feature] == maxima[feature]]
            share = np.float32(upstream[target, feature] / np.float32(len(winners)))
            for source in winners:
                expected_dx[source, feature] += share
    np.testing.assert_array_equal(y, expected_y)
    np.testing.assert_array_equal(actual, expected_dx)
