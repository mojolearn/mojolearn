# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The board's `ours-cpu` arm and per-arm memory, with the STUB drivers of
tools/test_bench_board.py (no mojolearn, no GPU, no dataset).

    .pixi/envs/test/bin/python -m pytest tools/test_bench_board_cpu_mem.py
"""
import importlib.util
import io
import json
import os
import sys
import types
from contextlib import redirect_stdout

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tb = _load("test_bench_board_for_cpu", os.path.join(HERE, "test_bench_board.py"))
bb = tb.bb
probe = _load("bench_board_probe_t", os.path.join(HERE, "bench_board_probe.py"))
env = tb.env          # the stub-driver fixture


def _cpu_base(env):
    base = list(env["base"])
    base.remove("--no-cpu-arm")
    return base


def _run(env, *extra, lanes="rf,kmeans"):
    return bb.main(["--vendor", "apple", "--lanes", lanes] + _cpu_base(env) + list(extra))


# --- planning -----------------------------------------------------------------

def test_plan_puts_ours_cpu_on_every_lane_with_a_cpu_path():
    for vendor in bb.VENDORS:
        races = bb.plan_races(vendor, bb.modes_for(vendor), bb.FAMILIES[:-1])
        for r in races:
            # a neural lane whose `ours` already runs on the CPU (the *-infer
            # classes, lm-host-train-step) has no separate CPU arm
            cpu_lane = r["family"] == "neural" and bb.NEURAL.DEVICE_OF[r["lane"]] == "cpu"
            assert (bb.CPU_ARM in r["arms"]) == (not cpu_lane), r["id"]
            if bb.CPU_ARM in r["arms"]:
                assert r["our_arms"][bb.CPU_ARM] == "identical"
                # right after our GPU arms, before every opponent
                assert r["arms"].index(bb.CPU_ARM) < len(r["our_arms"])
        assert bb.plan_summary(races)["cpu_cells"] == 93
        # the algos family: ours-cpu on every race, right after our GPU arms
        algos = bb.plan_races(vendor, bb.modes_for(vendor), ["algos"])
        assert all(r["arms"].index(bb.CPU_ARM) < len(r["our_arms"]) for r in algos)
    # the CPU tier is IDENTICAL only: a FAST-only Apple run has no ours-cpu arm
    fast = bb.plan_races("apple", ["fast"], ["trees", "classical", "classical2"])
    assert fast and all(bb.CPU_ARM not in r["arms"] for r in fast)
    # --no-cpu-arm
    off = bb.plan_races("nvidia", ["identical"], cpu_arm=False)
    assert all(bb.CPU_ARM not in r["arms"] for r in off)
    assert bb.arm_library(bb.CPU_ARM) == "mojolearn" and bb.arm_device(bb.CPU_ARM, "nvidia") == "cpu"
    assert bb.cpu_arm_reason("neural", "mlp-infer") and bb.cpu_arm_reason("trees", "rf") is None


@pytest.mark.parametrize("vendor,cells,off", [("apple", 455, 362), ("nvidia", 397, 304),
                                              ("amd", 385, 292)])
def test_dry_run_counts_with_and_without_the_cpu_arm(vendor, cells, off, capsys):
    fams = ["--families", "trees,classical,classical2,neural"]
    assert bb.main(["--dry-run", "--vendor", vendor] + fams) == 0
    text = capsys.readouterr().out
    assert "TOTAL races=101 cells=%d" % cells in text
    assert "ours-cpu cells=93" in text and "MOJOLEARN_VENDOR=cpu" in text
    # lanes without the arm are named, never dropped silently
    assert "ours-cpu NOT PLANNED: neural mlp-infer" in text
    assert "memory: peak_host_mb and peak_gpu_mb" in text
    assert bb.main(["--dry-run", "--vendor", vendor, "--no-cpu-arm"] + fams) == 0
    text = capsys.readouterr().out
    assert "TOTAL races=101 cells=%d" % off in text and "ours-cpu: off" in text


# --- a run with the stub drivers ------------------------------------------------

def test_run_ours_cpu_cells_bits_ratios_and_memory(env):
    assert _run(env) == 0
    res = json.loads((env["out"] / "board.json").read_text())
    assert res["config"]["cpu_arm"] is True
    rf = {c["arm"]: c for c in res["races"]["trees/rf/taxi/rows=1000"]["cells"]}
    cpu = rf["ours-cpu"]
    assert cpu["status"] == "ok" and cpu["device"] == "cpu" and cpu["mode"] == "identical"
    assert cpu["vendor_witness"] == "cpu" and "MOJOLEARN_VENDOR=cpu" in cpu["cpu_switch"]
    assert cpu["quality"]["bits_equal_vs_ours_identical"] is True
    assert cpu["bits_basis"] == "the last timed round's output hash"
    # medians: ours 102, ours-cpu 152, opponents 202; never CPU over GPU
    assert rf["sklearn-rf-cpu"]["ratio_ours_cpu_over"] == pytest.approx(152 / 202)
    assert rf["sklearn-rf-cpu"]["ratio_ours_identical_over"] == pytest.approx(102 / 202)
    assert rf["ours"]["ratio_ours_cpu_over"] is None and cpu["ratio_ours_identical_over"] is None
    # memory: the highest timed-round peak (rounds 1..3), the warm-up apart
    assert rf["ours"]["peak_host_mb"] == pytest.approx(1003.0)
    assert rf["ours"]["memory"]["warmup_host_mb"] == pytest.approx(1000.0)
    assert rf["ours"]["peak_gpu_mb"] == pytest.approx(7.0)
    assert rf["sklearn-rf-cpu"]["peak_gpu_mb"] is None
    assert rf["sklearn-rf-cpu"]["memory"]["gpu_method"] == "cpu arm"
    km = {c["arm"]: c for c in res["races"]["classical/kmeans/taxi/rows=1000"]["cells"]}
    assert km["ours-cpu"]["quality"]["bits_equal_vs_ours_identical"] is True
    assert km["ours-cpu"]["bits_basis"] == "the saved outputs, array by array"
    assert km["ours-cpu"]["device"] == "cpu" and km["ours-cpu"]["vendor_witness"] == "cpu"
    assert km["ours"]["peak_host_mb"] == pytest.approx(103.0)
    assert km["sklearn-cpu"]["ratio_ours_cpu_over"] is not None
    for rec in res["races"].values():
        for c in rec["cells"]:
            for k in ("peak_host_mb", "peak_gpu_mb", "memory", "ratio_ours_cpu_over"):
                assert k in c, (rec["id"], c["arm"], k)
    board = (env["out"] / "BOARD.md").read_text()
    assert "## Our CPU tier at a glance" in board
    assert "mojolearn CPU IDENTICAL" in board
    assert "| ours CPU / arm | peak host MB | peak GPU MB |" in board
    assert "memory, " in board and "stub host" in board
    assert "ours CPU |" in board                    # the quality glance column
    assert not tb.BANNED.search(board)


def test_ours_cpu_bits_that_differ_and_a_wrong_vendor_are_named(env, monkeypatch):
    monkeypatch.setenv("STUB_CPU_BITS", "differ")
    assert _run(env, lanes="rf") == 0
    res = json.loads((env["out"] / "board.json").read_text())
    cpu = next(c for c in res["races"]["trees/rf/taxi/rows=1000"]["cells"] if c["arm"] == "ours-cpu")
    assert cpu["quality"]["bits_equal_vs_ours_identical"] is False
    # a CPU arm that read back a GPU vendor is never timed under the CPU label
    parsed = {"arms": {"ours-cpu": {"ms": [1.0], "refused": None, "acc": {}, "hashes": ["h"],
                                    "fit": None}},
              "warmup": {}, "shape": None, "verdict": None, "verdict_line": None,
              "bindings": {"ours-cpu": {"compiled": "identical", "resolved": "identical",
                                        "vendor": "metal", "path": "/site-packages/x.so"}},
              "mem": {}}
    race = bb.plan_races("apple", ["identical"], ["trees"], ["rf"], ["taxi"], 1000)[0]
    ctx = {"vendor": "apple", "rounds": 1}
    cells = {c["arm"]: c for c in bb.tree_cells(ctx, race, parsed)}
    assert cells["ours-cpu"]["status"] == "VENDOR-MISMATCH(ours-cpu read back metal)"
    assert cells["ours-cpu"]["peak_host_mb"] is None
    assert cells["ours-cpu"]["memory"]["host_method"] == "not sampled"


def test_neural_run_with_ours_cpu(env):
    base = _cpu_base(env)
    base[base.index("--data-root") + 1] = str(env["tmp"] / "no-data-here")
    assert bb.main(["--vendor", "nvidia", "--families", "neural", "--neural-shape", "small",
                    "--lanes", "gemm,mlp-infer"] + base) == 0
    calls = tb._calls(env)
    assert any(c.startswith("neural gemm small ours,ours-cpu,torch-") for c in calls)
    assert any(c.startswith("neural mlp-infer small ours,torch-cpu-") for c in calls)
    res = json.loads((env["out"] / "board.json").read_text())
    g = {c["arm"]: c for c in res["races"]["neural/gemm/gaussian/shape=small"]["cells"]}
    assert g["ours-cpu"]["device"] == "cpu" and g["ours-cpu"]["status"] == "ok"
    assert g["ours-cpu"]["quality"]["bits_equal_vs_ours_identical"] is True
    board = (env["out"] / "BOARD.md").read_text()
    assert "Our CPU tier, no ours-cpu arm: neural" in board
    assert "no CPU opponent on this lane here" in board


# --- the probe --------------------------------------------------------------------

def test_probe_summary_bits_and_cpu_switch():
    s = probe.summarize([{"host_mb": 9.0, "gpu_mb": 1.0}, {"host_mb": 5.0, "gpu_mb": None,
                                                          "host_method": "h", "gpu_method": "g"},
                         {"host_mb": 7.0, "gpu_mb": 2.0, "host_method": "h", "gpu_method": "g"}])
    assert s["peak_host_mb"] == 7.0 and s["peak_gpu_mb"] == 2.0 and s["warmup_host_mb"] == 9.0
    assert s["rounds_sampled"] == 2 and s["host_method"] == "h"
    assert probe.summarize([])["peak_host_mb"] is None
    np = pytest.importorskip("numpy")
    a = {"y": np.arange(4, dtype=np.float32)}
    assert probe.bits_equal(a, {"y": np.arange(4, dtype=np.float32)}) is True
    assert probe.bits_equal(a, {"y": np.arange(4, dtype=np.float64)}) is False
    assert probe.bits_equal({"y": np.zeros(1, np.float32)}, {"y": -np.zeros(1, np.float32)}) is False
    assert probe.bits_equal(a, None) is None
    e = probe.ours_cpu_env({"MOJOLEARN_VENDOR_FORCE": "1"})
    assert e["MOJOLEARN_VENDOR"] == "cpu" and e["MOJOLEARN_NUMERIC_MODE"] == "identical"
    assert "MOJOLEARN_VENDOR_FORCE" not in e and e[probe.OURS_CPU_FLAG] == "1"


def test_probe_cpu_check_refuses_a_gpu_vendor(monkeypatch):
    monkeypatch.delenv(probe.OURS_CPU_FLAG, raising=False)
    ml = types.SimpleNamespace(vendor=lambda: "metal",
                               _backend=types.SimpleNamespace(vendor_how=lambda: "flat layout"))
    assert probe.ours_cpu_check(ml) == {}            # not an ours-cpu worker: nothing
    monkeypatch.setenv(probe.OURS_CPU_FLAG, "1")
    with pytest.raises(RuntimeError, match="REFUSED: ours-cpu.*'metal'"):
        probe.ours_cpu_check(ml)
    ml.vendor = lambda: "cpu"
    info = probe.ours_cpu_check(ml)
    assert info["device"] == "cpu" and info["vendor_used"] == "cpu"


def test_probe_reads_this_process():
    m = probe.MemProbe("cpu")
    m.start()
    block = bytearray(32 * 1024 * 1024)
    block[::4096] = b"x" * len(block[::4096])
    out = m.stop()
    assert isinstance(out["host_mb"], float) and out["host_mb"] >= 32.0
    assert out["host_method"] and out["gpu_mb"] is None and out["gpu_method"] == "cpu arm: no device memory"
    g = probe.MemProbe("gpu", vendor="apple").stop()
    assert g["gpu_mb"] is None and "unified memory" in g["gpu_method"]


def test_probe_reads_torch_counter_only_for_torch_arms(monkeypatch):
    """do-amd 2026-09-29: torch had initialized ROCm in the trees driver, so
    our arm read torch's allocator peak (0.0 MB). A non-torch GPU arm reads
    the driver's per-process figure; a torch arm reads torch's counter."""
    class _Cuda(object):
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def max_memory_allocated(self):
            return 0

        def reset_peak_memory_stats(self):
            pass

    fake = types.SimpleNamespace(cuda=_Cuda(), backends=types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(probe.MemProbe, "_smi", lambda self: (512.0, "driver per-process"))
    ours = probe.MemProbe("gpu", vendor="amd", shared=True, library="mojolearn")
    assert ours._read_gpu() == (512.0, "driver per-process")
    torch_arm = probe.MemProbe("gpu", vendor="amd", library="torch")
    mb, method = torch_arm._read_gpu()
    assert mb == 0.0 and method.startswith("torch.cuda.max_memory_allocated")
    assert probe.MemProbe("gpu", vendor="amd", library="gpytorch")._read_gpu()[1].startswith("torch.cuda")


# --- the trees driver's plumbing -----------------------------------------------------

def test_tree_mem_lines_parse_into_cells(tmp_path):
    fba = _load("forest_board_arms_t", os.path.join(REPO, "bench", "speed", "forest_board_arms.py"))
    mem = fba.TreeMem("rf")
    buf = io.StringIO()
    with redirect_stdout(buf):
        for r in range(3):
            for arm in ("ours", "sklearn-rf-cpu"):
                with mem.context(arm, r):
                    pass
        mem.emit("ours-cpu", 1, {"host_mb": 12.5, "host_method": "a=b c", "gpu_method": "cpu arm"})
    log = tmp_path / "rf.log"
    log.write_text(buf.getvalue())
    parsed = bb.parse_tree_log(str(log))
    assert len(parsed["mem"]["ours"]) == 3 and len(parsed["mem"]["sklearn-rf-cpu"]) == 3
    assert parsed["mem"]["ours-cpu"][0]["host_mb"] == 12.5
    assert parsed["mem"]["ours-cpu"][0]["host_method"] == "a:b c"     # no '=' in a value
    assert parsed["mem"]["sklearn-rf-cpu"][1]["gpu_method"].startswith("cpu arm")
    f = bb.memory_fields(parsed["mem"]["ours"])
    assert isinstance(f["peak_host_mb"], float) and f["memory"]["rounds_sampled"] == 2


def test_invalidate_memory_withdraws_only_torch_counter_figures(tmp_path, monkeypatch):
    """--invalidate-memory: a non-torch GPU arm whose figure came from torch's
    allocator loses the figure (times untouched); a torch arm, a CPU arm and
    a driver-counter figure keep theirs; the store gets a corrected copy; a
    second pass changes nothing."""
    torch_m = {"gpu_method": "torch.cuda.max_memory_allocated, reset before the round",
               "peak_gpu_mb": 0.0, "warmup_gpu_mb": 0.0}
    smi_m = {"gpu_method": "rocm-smi --showpids VRAM USED", "peak_gpu_mb": 900.0}

    def cell(arm, lib, dev, mem):
        return {"arm": arm, "library": lib, "device": dev, "median_ms": 10.0,
                "peak_gpu_mb": mem.get("peak_gpu_mb"), "memory": dict(mem)}
    ours = cell("ours", "mojolearn", "gpu", torch_m)
    races = {
        "trees/gbdt-symmetric/taxi/rows=full": {"status": "done", "cells": [
            ours, cell("xgboost-gpu", "xgboost", "gpu", torch_m),
            cell("catboost-cpu", "catboost", "cpu", {"gpu_method": "cpu arm: no device memory"})]},
        "neural/linear/synthetic": {"status": "done", "cells": [
            cell("torch-eager-fp32", "torch", "gpu", torch_m),
            cell("ours", "mojolearn", "gpu", smi_m)]},
        "classical/kmeans/taxi/rows=full": {"status": "done", "cells": [
            cell("ours", "mojolearn", "gpu", torch_m)]},
    }
    out = tmp_path / "run"
    out.mkdir()
    (out / "board.json").write_text(json.dumps({"races": races}))
    store = tmp_path / "opponent-store.jsonl"
    key = {"family": "trees", "lane": "gbdt-symmetric", "dataset": "taxi", "arm": "xgboost-gpu",
           "box": "b", "machine": "m", "vendor": "amd", "device": "gpu"}
    store.write_text(json.dumps({"key": key, "cell": cell("xgboost-gpu", "xgboost", "gpu", torch_m)}) + "\n")
    rendered = []
    monkeypatch.setattr(bb, "write_board", lambda o, r: rendered.append(o))   # the records here are minimal
    nb, ns = bb.invalidate_memory(str(out), "trees/,neural/", "wrong counter", "abc123", str(store))
    assert rendered == [str(out)]
    assert (nb, ns) == (2, 1)
    got = json.loads((out / "board.json").read_text())["races"]
    t = {c["arm"]: c for c in got["trees/gbdt-symmetric/taxi/rows=full"]["cells"]}
    for arm in ("ours", "xgboost-gpu"):
        assert t[arm]["peak_gpu_mb"] is None and t[arm]["median_ms"] == 10.0
        assert t[arm]["memory"]["gpu_method"] == "not measured (wrong counter; fixed at abc123)"
    assert t["catboost-cpu"]["memory"]["gpu_method"] == "cpu arm: no device memory"
    n = {c["arm"]: c for c in got["neural/linear/synthetic"]["cells"]}
    assert n["torch-eager-fp32"]["peak_gpu_mb"] == 0.0 and n["ours"]["peak_gpu_mb"] == 900.0
    # outside the prefixes: untouched
    assert got["classical/kmeans/taxi/rows=full"]["cells"][0]["peak_gpu_mb"] == 0.0
    latest = bb.STORE.load(str(store))
    assert list(latest.values())[0]["cell"]["peak_gpu_mb"] is None
    assert bb.invalidate_memory(str(out), "trees/,neural/", "wrong counter", "abc123", str(store)) == (0, 0)


def test_invalidate_arm_refuses_its_cells_and_recomputes_ratios(tmp_path, monkeypatch):
    """--invalidate-arm: every finished cell of the arm (fit and inference, and
    its store records of this vendor) becomes REFUSED(reason) with its times
    withdrawn; the race's other cells and ratios stay right; a second pass
    changes nothing."""
    def cell(arm, lib, ms, **kw):
        return dict({"arm": arm, "library": lib, "mode": "identical" if lib == "mojolearn" else "opponent",
                     "status": "ok", "median_ms": ms, "min_ms": ms, "max_ms": ms, "times_ms": [ms],
                     "rounds": 1, "quality": {"auc": 0.6}}, **kw)
    cells = [cell("ours", "mojolearn", 50.0), cell("xgboost-gpu", "xgboost", 100.0),
             cell("xgboost-cpu", "xgboost", 200.0)]
    infer = [cell("ours", "mojolearn", 5.0, batch="test"), cell("xgboost-gpu", "xgboost", 10.0, batch="test")]
    bb.add_ratios(cells)
    races = {"trees/gbdt-depthwise/taxi/rows=full": {"status": "done", "cells": cells, "infer_cells": infer},
             "trees/gbdt-symmetric/taxi/rows=full": {"status": "done", "cells": [cell("ours", "mojolearn", 1.0)]}}
    out = tmp_path / "run"
    out.mkdir()
    (out / "board.json").write_text(json.dumps({"races": races}))
    store = tmp_path / "opponent-store.jsonl"
    rows = [{"key": {"arm": "xgboost-gpu", "vendor": "amd", "lane": "gbdt-depthwise", "box": "a"},
             "cell": cell("xgboost-gpu", "xgboost", 100.0)},
            {"key": {"arm": "xgboost-gpu", "vendor": "nvidia", "lane": "gbdt-depthwise", "box": "n"},
             "cell": cell("xgboost-gpu", "xgboost", 7.0)}]
    store.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(bb, "write_board", lambda o, r: None)
    why = "PyPI xgboost is a CUDA build; on AMD it trained on the CPU"
    assert bb.invalidate_arm(str(out), "xgboost-gpu", why, str(store), vendor="amd") == (2, 1)
    r = json.loads((out / "board.json").read_text())["races"]["trees/gbdt-depthwise/taxi/rows=full"]
    c = {x["arm"]: x for x in r["cells"]}
    g = c["xgboost-gpu"]
    assert g["status"] == "REFUSED(%s)" % why and g["median_ms"] is None and g["times_ms"] == []
    assert g["withdrawn"]["median_ms"] == 100.0 and g["ratio_ours_identical_over"] is None
    assert c["xgboost-cpu"]["ratio_ours_identical_over"] == 0.25 and c["xgboost-cpu"]["status"] == "ok"
    assert c["ours"]["median_ms"] == 50.0
    i = {x["arm"]: x for x in r["infer_cells"]}
    assert i["xgboost-gpu"]["status"].startswith("REFUSED(") and i["ours"]["median_ms"] == 5.0
    latest = {v["key"]["vendor"]: v for v in bb.STORE.load(str(store)).values()}
    assert latest["amd"]["cell"]["status"].startswith("REFUSED(") and latest["nvidia"]["cell"]["median_ms"] == 7.0
    assert bb.invalidate_arm(str(out), "xgboost-gpu", why, str(store), vendor="amd") == (0, 0)


def test_invalidate_arm_scoped_to_races_rewords_a_library_refusal(tmp_path, monkeypatch):
    """--invalidate-arm-races: only the named races; a cell the library
    already refused there gets the board's reason, the library's words kept."""
    def cell(arm, status="ok", ms=5.0):
        return {"arm": arm, "library": "x", "mode": "opponent", "status": status, "median_ms": ms,
                "times_ms": [ms] if ms else [], "rounds": 1}
    races = {"classical2/gmm/taxi/rows=full": {"status": "done", "cells": [
                 cell("ours"), cell("sklearn-cpu", "REFUSED(error: ValueError ill-defined)", None)]},
             "classical2/gmm/istella/rows=full": {"status": "done", "cells": [cell("sklearn-cpu")]}}
    out = tmp_path / "run"
    out.mkdir()
    (out / "board.json").write_text(json.dumps({"races": races}))
    monkeypatch.setattr(bb, "write_board", lambda o, r: None)
    why = "scikit-learn on OpenBLAS collapses a component at reg_covar 1e-6 on this box"
    assert bb.invalidate_arm(str(out), "sklearn-cpu", why, None, races="classical2/gmm/taxi/") == (1, 0)
    got = json.loads((out / "board.json").read_text())["races"]
    st = got["classical2/gmm/taxi/rows=full"]["cells"][1]["status"]
    assert st == "REFUSED(%s; the library said: error: ValueError ill-defined)" % why
    assert got["classical2/gmm/istella/rows=full"]["cells"][0]["status"] == "ok"
    assert bb.invalidate_arm(str(out), "sklearn-cpu", why, None, races="classical2/gmm/taxi/") == (0, 0)
