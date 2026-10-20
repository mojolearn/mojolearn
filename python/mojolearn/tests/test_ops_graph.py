# SPDX-License-Identifier: Apache-2.0
"""Whole-operation graph qualification; unavailable native entries fail."""
import json

import pytest

from mojolearn import _verify_graph as graph


@pytest.mark.parametrize("name", graph.EXPECTED_CASES)
def test_graph_execution_case(name):
    graph._AUDITS.clear()
    arrays = getattr(graph, name)()
    for i, value in enumerate(arrays):
        output = graph._digest(value)
        print("GRAPH_DIGEST", name + "/" + str(i), output["sha256"])
    assert graph._AUDITS, "every case must exercise audited native public calls"
    assert all(row["passed"] or row.get("expected_refusal") for row in graph._AUDITS), json.dumps(graph._AUDITS)
