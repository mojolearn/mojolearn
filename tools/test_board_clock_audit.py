# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Tests for tools/board_clock_audit.py and the board's derived clock fields.

    .pixi/envs/test/bin/python -m pytest tools/test_board_clock_audit.py
"""
import copy
import importlib.util
import json
import os
import re

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ca = _load("board_clock_audit")


def cell(arm, library, median, family="classical", device="gpu", mode=None, span=None,
         status="ok", phase=None, batch=None, comparability_upload=None, times=None):
    c = {"family": family, "lane": "kmeans", "dataset": "taxi", "arm": arm, "library": library,
         "device": device, "mode": mode or ("identical" if library == "mojolearn" else "opponent"),
         "status": status, "median_ms": median, "times_ms": times if times is not None else
         ([median] if median is not None else []), "warmup_ms": 10.0,
         "comparability": {"span": span or {}, "span_asymmetry": []},
         "ratio_ours_identical_over": None, "ratio_ours_fast_over": None}
    if phase:
        c["phase"] = phase
        c["batch"] = batch or "Xq"
    if comparability_upload is not None:
        c["comparability"]["upload_ms_untimed"] = comparability_upload
    return c


OURS_SPAN = {"input_home": "host", "pre_clock_fit": False}


# ------------------------------------------------------------------ cell_clock

def test_ours_with_separate_upload_has_both_clocks():
    k = ca.cell_clock(cell("ours", "mojolearn", 100.0, span=dict(OURS_SPAN, upload_ms_separate=30.0)))
    assert k["stored"] == "whole" and k["derivable"] == "both"
    assert k["whole_ms"] == 100.0 and k["kernel_ms"] == pytest.approx(70.0)
    assert k["copy_ms"] == 30.0 and k["copy_source"] == "upload_ms_separate"
    assert k["copy_share"] == pytest.approx(0.3)


def test_ours_without_separate_upload_is_whole_only():
    k = ca.cell_clock(cell("ours", "mojolearn", 100.0, span=OURS_SPAN))
    assert k["derivable"] == "whole" and k["kernel_ms"] is None and k["copy_ms"] is None


def test_ours_upload_not_below_median_withholds_the_kernel_clock():
    k = ca.cell_clock(cell("ours", "mojolearn", 10.0, span=dict(OURS_SPAN, upload_ms_separate=12.0)))
    assert k["derivable"] == "whole" and k["kernel_ms"] is None
    assert "withheld" in k["note"]


def test_gpu_opponent_with_pre_clock_upload_derives_whole():
    k = ca.cell_clock(cell("cuml-gpu", "cuml", 50.0, span={"input_home": "device",
                                                         "upload_ms_untimed": 20.0}))
    assert k["stored"] == "kernel" and k["derivable"] == "both"
    assert k["kernel_ms"] == 50.0 and k["whole_ms"] == pytest.approx(70.0)
    assert k["copy_source"] == "upload_ms_untimed"


def test_upload_recorded_beats_a_wrong_host_label():
    # the algos driver used to label a cuML arm with a pre-clock upload `host`
    k = ca.cell_clock(cell("cuml-gpu", "cuml", 5.0, family="algos",
                           span={"input_home": "host", "upload_ms_untimed": 2.0}))
    assert k["stored"] == "kernel" and k["whole_ms"] == pytest.approx(7.0)


def test_device_resident_torch_without_upload_time_is_kernel_only():
    k = ca.cell_clock(cell("torch-eager-fp32", "torch", 8.0, family="algos",
                           span={"input_home": "device", "upload_ms_untimed": None}))
    assert k["stored"] == "kernel" and k["derivable"] == "kernel" and k["whole_ms"] is None


def test_neural_torch_twin_includes_the_copy():
    k = ca.cell_clock(cell("torch-eager-fp32", "torch", 8.0, family="neural",
                           span={"input_home": "host"}))
    assert k["stored"] == "whole" and k["derivable"] == "whole"


def test_tree_arm_without_span_is_whole():
    k = ca.cell_clock(cell("xgboost-gpu", "xgboost", 900.0, family="trees", span={}))
    assert k["stored"] == "whole" and k["whole_ms"] == 900.0


def test_cpu_arm_has_no_device_copy():
    k = ca.cell_clock(cell("sklearn-cpu", "scikit-learn", 400.0, device="cpu",
                           span={"input_home": "host"}))
    assert k["derivable"] == "both" and k["whole_ms"] == k["kernel_ms"] == 400.0
    assert k["copy_ms"] == 0.0 and k["copy_source"] == "cpu-arm"


def test_unknown_span_on_a_gpu_opponent_is_unknown():
    k = ca.cell_clock(cell("cupy-gpu", "cupy", 3.0, family="algos", span={}))
    assert k["stored"] == "unknown" and k["derivable"] == "none" and k["median_ms"] == 3.0


def test_failed_cell_has_no_clock():
    k = ca.cell_clock(cell("ours", "mojolearn", None, span=OURS_SPAN, status="REFUSED(x)"))
    assert k["derivable"] == "none" and k["whole_ms"] is None and "REFUSED" in k["note"]


def test_infer_classical_uses_the_predict_rows_upload_not_the_fit_one():
    k = ca.cell_clock(cell("torch-gpu", "torch", 1.0, phase="infer",
                           span={"input_home": "device", "upload_ms_untimed": 90.0},
                           comparability_upload=2.0))
    assert k["copy_ms"] == 2.0 and k["whole_ms"] == pytest.approx(3.0)


def test_infer_algos_fit_upload_is_not_separable():
    k = ca.cell_clock(cell("cuml-gpu", "cuml", 1.0, family="algos", phase="infer",
                           span={"input_home": "device", "upload_ms_untimed": 90.0},
                           comparability_upload=90.0))
    assert k["derivable"] == "kernel" and k["whole_ms"] is None and "not separable" in k["note"]


def test_infer_ours_never_uses_the_fit_probe():
    k = ca.cell_clock(cell("ours", "mojolearn", 5.0, phase="infer",
                           span=dict(OURS_SPAN, upload_ms_separate=1.0)))
    assert k["derivable"] == "whole"


# ------------------------------------------------------------------ ratios

def _race(*cells):
    return ca.annotate_cells([copy.deepcopy(c) for c in cells])


def test_torch_ratio_reads_kernel_over_kernel_when_both_exist():
    cs = _race(cell("ours", "mojolearn", 100.0, span=dict(OURS_SPAN, upload_ms_separate=40.0)),
               cell("torch-gpu", "torch", 30.0, span={"input_home": "device", "upload_ms_untimed": 10.0}))
    r = cs[1]["ratio_ours_identical_clock"]
    assert r["clock"] == "kernel" and r["kind"] == "preferred" and r["label"] == "kernel/kernel"
    assert r["value"] == pytest.approx(60.0 / 30.0)


def test_torch_ratio_falls_back_to_whole_and_says_so():
    cs = _race(cell("ours", "mojolearn", 100.0, span=OURS_SPAN),
               cell("torch-gpu", "torch", 30.0, span={"input_home": "device", "upload_ms_untimed": 10.0}))
    r = cs[1]["ratio_ours_identical_clock"]
    assert r["clock"] == "whole" and r["kind"] == "fallback"
    assert r["value"] == pytest.approx(100.0 / 40.0) and "kernel not derivable" in r["label"]


def test_cuml_ratio_reads_whole_over_whole():
    cs = _race(cell("ours", "mojolearn", 100.0, span=dict(OURS_SPAN, upload_ms_separate=40.0)),
               cell("cuml-gpu", "cuml", 30.0, span={"input_home": "device", "upload_ms_untimed": 10.0}))
    r = cs[1]["ratio_ours_identical_clock"]
    assert r["label"] == "whole/whole" and r["value"] == pytest.approx(2.5)


def test_no_common_clock_is_labelled_mixed():
    cs = _race(cell("ours", "mojolearn", 100.0, family="algos", span=OURS_SPAN),
               cell("torch-eager-fp32", "torch", 25.0, family="algos", span={"input_home": "device"}))
    r = cs[1]["ratio_ours_identical_clock"]
    assert r["kind"] == "mixed" and r["label"] == "MIXED ours whole / arm kernel"
    assert r["value"] == pytest.approx(4.0)


def test_annotate_leaves_stored_fields_alone():
    base = [cell("ours", "mojolearn", 100.0, span=OURS_SPAN),
            cell("cuml-gpu", "cuml", 30.0, span={"input_home": "device", "upload_ms_untimed": 10.0})]
    base[1]["ratio_ours_identical_over"] = 100.0 / 30.0
    before = copy.deepcopy(base)
    ca.annotate_cells(base)
    for b, a in zip(before, base):
        for k, v in b.items():
            assert a[k] == v
    assert base[0]["ratio_ours_identical_clock"] is None      # ours is never in a ratio


def test_fast_and_identical_ratios_and_infer_batches_are_separate():
    cs = _race(cell("ours", "mojolearn", 100.0, span=OURS_SPAN),
               cell("ours-fast", "mojolearn", 50.0, mode="fast", span=OURS_SPAN),
               cell("sklearn-cpu", "scikit-learn", 25.0, device="cpu", span={"input_home": "host"}))
    assert cs[2]["ratio_ours_identical_clock"]["value"] == pytest.approx(4.0)
    assert cs[2]["ratio_ours_fast_clock"]["value"] == pytest.approx(2.0)
    ic = _race(cell("ours", "mojolearn", 10.0, phase="infer", batch="test", span=OURS_SPAN),
               cell("ours", "mojolearn", 20.0, phase="infer", batch="large", span=OURS_SPAN),
               cell("sklearn-cpu", "scikit-learn", 5.0, device="cpu", phase="infer", batch="large"))
    assert ic[2]["ratio_ours_identical_clock"]["value"] == pytest.approx(4.0)


# ------------------------------------------------------------------ audit

def _board():
    return {"races": {
        "classical/kmeans/taxi/rows=full": {"wall_s": 30.0, "cells": [
            cell("ours", "mojolearn", 100.0, span=OURS_SPAN),
            cell("torch-gpu", "torch", 30.0, span={"input_home": "device", "upload_ms_untimed": 10.0}),
            cell("cuml-gpu", "cuml", None, status="REFUSED(x)", span={"input_home": "device"})]},
        "algos/cnn/synthetic/rows=full": {"wall_s": 50.0, "cells": [
            cell("ours", "mojolearn", 200.0, family="algos", span=OURS_SPAN, times=[200.0]),
            cell("torch-eager-fp32", "torch", 20.0, family="algos", span={"input_home": "device"}),
            cell("cupy-gpu", "cupy", 5.0, family="algos", span={})]}}}


def test_audit_summary_counts_and_rerun_estimate():
    rows, pairs = ca.audit_board("t", _board())
    s = ca.summarize(rows, pairs)
    assert s["cells"] == 6
    assert (s["derivable_both"], s["derivable_whole"], s["derivable_kernel"], s["derivable_none"]) \
        == (1, 2, 1, 2)
    assert s["none_no_time"] == 1 and s["none_with_time"] == 1
    neither, mixed, est = ca.rerun_candidates(rows, pairs)
    assert [r["arm"] for r in neither] == ["cupy-gpu"]
    assert est["neither_arm_s"] == pytest.approx(0.015)          # 10 ms warm-up + 5 ms round
    assert est["mixed_pairs"] == 2 and est["our_cells_to_rerun"] == 1
    assert est["our_arm_s"] == pytest.approx(0.21) and est["our_race_wall_s_upper"] == 50.0
    assert est["our_lanes"] == {"algos/cnn": ["synthetic"]}


def test_main_writes_the_three_files(tmp_path):
    p = tmp_path / "board.json"
    p.write_text(json.dumps(_board()))
    out = tmp_path / "out"
    assert ca.main(["--board", "x=%s" % p, "--out", str(out)]) == 0
    doc = json.loads((out / "x.json").read_text())
    assert doc["summary"]["cells"] == 6 and len(doc["neither"]) == 1
    md = (out / "x.md").read_text()
    assert "## Rerun estimate" in md and "MIXED ours whole / arm kernel" in md
    assert json.loads((out / "summary.json").read_text())["x"]["summary"]["cells"] == 6


# ------------------------------------------------------------------ board rendering

def _bb():
    return _load("bench_board")


def _table_widths(text):
    widths, cur = [], None
    for line in text.splitlines():
        if line.startswith("| arm |"):
            cur = line.count(" | ") + 1
            widths.append([cur])
        elif cur and line.startswith("|") and not line.startswith("|---"):
            widths[-1].append(line.count(" | ") + 1)
        elif not line.startswith("|"):
            cur = None
    return widths


def test_board_renders_both_clocks_and_keeps_the_stored_ratio():
    bb = _bb()
    res = {"schema": bb.SCHEMA, "plan": [], "races": {
        "classical/kmeans/taxi/rows=full": {
            "family": "classical", "lane": "kmeans", "dataset": "taxi", "rows": None, "status": "done",
            "cells": [cell("ours", "mojolearn", 100.0, span=dict(OURS_SPAN, upload_ms_separate=40.0)),
                      dict(cell("torch-gpu", "torch", 30.0,
                                span={"input_home": "device", "upload_ms_untimed": 10.0}),
                           ratio_ours_identical_over=100.0 / 30.0)]}}}
    for c in res["races"]["classical/kmeans/taxi/rows=full"]["cells"]:
        c.update(rounds=1, min_ms=c["median_ms"], max_ms=c["median_ms"], quality={}, hash_stable=True,
                 verdict="LIKE-FOR-LIKE-SPAN", memory={}, settings={})
    stored = json.dumps(res, sort_keys=True)
    text = bb.render_board(res)
    assert json.dumps(res, sort_keys=True) == stored          # render never writes the record
    assert bb.CLOCK_HEADER in text
    row = next(l for l in text.splitlines() if l.startswith("| torch-gpu |"))
    assert "| 3.333 |" in row                                  # the stored median ratio, kept
    assert "40.0 | 30.0 | 10.00 (upload_ms_untimed) | 2.000 (kernel/kernel)" in row
    ours = next(l for l in text.splitlines() if l.startswith("| mojolearn IDENTICAL |"))
    assert "| 100.0 | 60.0 | 40.00 (upload_ms_separate) |" in ours
    for w in _table_widths(text):
        assert len(set(w)) == 1, w


def test_board_clock_cells_of_an_unannotated_cell_are_dashes():
    bb = _bb()
    assert bb.clock_cells({}) == " | ".join(["-"] * bb.CLOCK_COLUMNS)
    assert len(bb.CLOCK_HEADER.split(" | ")) == bb.CLOCK_COLUMNS
    assert not re.search(r"\b(faster|slower)\b", bb.CLOCK_HEADER, re.I)
