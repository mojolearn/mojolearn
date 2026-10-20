# SPDX-License-Identifier: Apache-2.0
"""Verifier failure controls are runnable without building or executing Mojo."""
import copy
import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).parents[1] / "python" / "mojolearn" / "_verify_graph.py"
_SPEC = importlib.util.spec_from_file_location("graph_verifier_under_test", _PATH)
graph = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(graph)


def report():
    return {"format": graph.FORMAT, "expected_cases": list(graph.EXPECTED_CASES),
            "backend": {"numeric_mode": 1, "vendor": "cpu"},
            "cases": {name: {"status": "passed", "outputs": [
                {"shape": [2, 3], "dtype": "<f4", "sha256": "a" * 64}] * graph.OUTPUT_COUNTS[name],
                                     "native_calls": [{"passed": True}]}
                      for name in graph.EXPECTED_CASES}}


def test_comparator_accepts_complete_matching_report():
    assert graph.compare_reports(report(), report()) == []


def test_comparator_negative_controls():
    controls = graph.comparator_self_test(report())
    assert len(controls) == 5 and list(controls.values()).count("rejected") == 4


@pytest.mark.parametrize("damage", ["empty", "shape", "dtype", "digest", "missing", "skipped", "failed", "mode", "schema"])
def test_comparator_refuses_incomplete_or_different_evidence(damage):
    original, altered = report(), report()
    row = altered["cases"][graph.EXPECTED_CASES[0]]
    if damage == "empty":
        altered = {}
    elif damage in ("shape", "dtype", "digest"):
        key = "sha256" if damage == "digest" else damage
        row["outputs"][0][key] = "altered"
    elif damage == "missing":
        del altered["cases"][graph.EXPECTED_CASES[0]]
    elif damage in ("skipped", "failed"):
        row["status"] = damage
    elif damage == "mode":
        altered["backend"]["numeric_mode"] = 0
    else:
        altered["format"] = "unknown"
    assert graph.compare_reports(original, altered)
    assert graph.compare_reports(altered, original)
