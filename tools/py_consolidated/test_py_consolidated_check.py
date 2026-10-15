"""Pure saved-record regression tests; no package/native imports."""
import copy
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("pycheck", Path(__file__).with_name("check.py"))
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def record():
    return {"complete": True, "repeats": 2, "cells": {"lane/base": {
        "hashes": ["0123456789abcdef"] * 2, "verdict": "STABLE",
        "batch": ["n/a:no-batch"] * 2, "batch_verdict": "N/A",
        "timing": {"total_seconds": 1.0}}}}


def test_timing_is_not_numerical():
    a, b = record(), record()
    b["cells"]["lane/base"]["timing"]["total_seconds"] = 99
    r = check.compare_records(a, b, "lane")
    assert r["status"] == "SAME" and r["numeric"] == r["na"] == 1


def test_missing_cell_cannot_disappear_in_intersection():
    a, b = record(), record()
    b["cells"]["lane/wide"] = copy.deepcopy(b["cells"]["lane/base"])
    assert check.compare_records(a, b, "lane")["status"] != "SAME"


def test_all_repeats_and_verdicts_matter():
    a, b = record(), record()
    b["cells"]["lane/base"]["hashes"][1] = "fedcba9876543210"
    assert check.compare_records(a, b, "lane")["status"] != "SAME"
    b = record()
    b["cells"]["lane/base"]["verdict"] = "MOVED"
    assert check.compare_records(a, b, "lane")["status"] != "SAME"


def test_refusal_empty_partial_and_undeclared_fail_closed():
    for update in ({"error": "refused"}, {"hashes": ["n/a:refused"] * 2, "verdict": "N/A"},
                   {"batch": ["n/a:UNDECLARED"] * 2}):
        a = record()
        a["cells"]["lane/base"].update(update)
        assert check.compare_records(a, a, "lane")["status"] != "SAME"
    for update in ({"cells": {}}, {"complete": False}, {"partial_column": True}):
        a = record(); a.update(update)
        assert check.compare_records(a, a, "lane")["status"] != "SAME"
