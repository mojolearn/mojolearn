# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Our CPU never races on the board (Andrew, Oct 2 2026), per-arm memory, with the STUB drivers of
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


def _run(env, *extra, lanes="rf,kmeans"):
    return bb.main(["--vendor", "apple", "--lanes", lanes] + env["base"] + list(extra))


def _no_cpu_flag(env):
    base = list(env["base"])
    base.remove("--no-cpu-arm")
    return base


# --- planning: our CPU is never raced (Andrew, Oct 2 2026) -----------------------

def test_plan_never_puts_ours_cpu_on_any_race():
    for vendor in bb.VENDORS:
        races = bb.plan_races(vendor, bb.modes_for(vendor))
        assert races
        for r in races:
            assert bb.CPU_ARM not in r["arms"], r["id"]
            assert not any(probe.is_our_cpu_arm(a) for a in r["arms"]), r["id"]
            # no neural lane whose ours runs the CPU binding is planned at all
            assert not (r["family"] == "neural" and bb.NEURAL.DEVICE_OF[r["lane"]] == "cpu")
        assert "cpu_cells" not in bb.plan_summary(races)
    with pytest.raises(TypeError):                # the switch itself is gone
        bb.plan_races("apple", ["identical"], cpu_arm=True)
    # the lanes that left the board are exactly the CPU-binding ones
    planned = {r["lane"] for r in bb.plan_races("nvidia", ["identical"], ["neural"])}
    gone = {l for l in bb.NEURAL_LANES if bb.NEURAL.DEVICE_OF.get(l) == "cpu"}
    assert gone and not planned & gone
    assert planned | gone == set(bb.NEURAL_LANES)


@pytest.mark.parametrize("vendor", ["apple", "nvidia", "amd"])
def test_dry_run_same_plan_with_or_without_no_cpu_arm(vendor, capsys):
    fams = ["--families", "trees,classical,classical2,neural"]
    # 93 races: the counts are pinned per vendor in test_bench_board.py
    races = bb.plan_races(vendor, bb.modes_for(vendor), bb.FAMILIES[:-1])
    total = "TOTAL races=%d cells=%d" % (len(races), sum(len(r["arms"]) for r in races))
    assert len(races) == 93
    texts = []
    for flag in ([], ["--no-cpu-arm"]):
        assert bb.main(["--dry-run", "--vendor", vendor] + flag + fams) == 0
        texts.append(capsys.readouterr().out)
    for text in texts:
        assert total in text
        assert "our CPU: never raced (the board races only our GPU)" in text
        assert "ours-cpu cells=" not in text and "ours-cpu NOT PLANNED" not in text
        assert "memory: peak_host_mb and peak_gpu_mb" in text


# --- a run with the stub drivers ------------------------------------------------

def test_run_memory_and_no_ours_cpu_cells(env):
    # without --no-cpu-arm: the default never races our CPU either
    assert bb.main(["--vendor", "apple", "--lanes", "rf,kmeans"] + _no_cpu_flag(env)) == 0
    res = json.loads((env["out"] / "board.json").read_text())
    assert "cpu_arm" not in res["config"]
    assert not any("--ours-cpu" in rec["command"] for rec in res["races"].values())
    rf = {c["arm"]: c for c in res["races"]["trees/rf/taxi/rows=1000"]["cells"]}
    assert set(rf) == {"ours", "ours-ab", "sklearn-rf-cpu", "lightgbm-cpu"}
    assert all("ratio_ours_cpu_over" not in c for c in rf.values())
    # memory: the highest timed-round peak (rounds 1..3), the warm-up apart
    assert rf["ours"]["peak_host_mb"] == pytest.approx(1003.0)
    assert rf["ours"]["memory"]["warmup_host_mb"] == pytest.approx(1000.0)
    assert rf["ours"]["peak_gpu_mb"] == pytest.approx(7.0)
    assert rf["sklearn-rf-cpu"]["peak_gpu_mb"] is None
    assert rf["sklearn-rf-cpu"]["memory"]["gpu_method"] == "cpu arm"
    km = {c["arm"]: c for c in res["races"]["classical/kmeans/taxi/rows=1000"]["cells"]}
    assert bb.CPU_ARM not in km and "sklearn-cpu" not in km      # torch-gpu races, so no CPU one
    assert km["ours"]["peak_host_mb"] == pytest.approx(103.0)
    for rec in res["races"].values():
        for c in rec["cells"]:
            assert c["arm"] != bb.CPU_ARM
            for k in ("peak_host_mb", "peak_gpu_mb", "memory"):
                assert k in c, (rec["id"], c["arm"], k)
    board = (env["out"] / "BOARD.md").read_text()
    assert "## Our CPU tier at a glance" not in board
    assert "Our CPU: never raced" in board and "Our CPU is never raced" in board
    assert "races `torch-cpu-*`" not in board
    assert "ours CPU" not in board and "mojolearn CPU" not in board
    assert "memory, " in board and "stub host" in board
    assert not tb.BANNED.search(board)


def test_neural_run_skips_the_cpu_binding_lanes(env):
    base = list(env["base"])
    base[base.index("--data-root") + 1] = str(env["tmp"] / "no-data-here")
    assert bb.main(["--vendor", "nvidia", "--families", "neural", "--neural-shape", "small",
                    "--lanes", "gemm,mlp-infer"] + base) == 0
    calls = tb._calls(env)
    assert len(calls) == 1 and calls[0].startswith("neural gemm small ours,torch-")
    assert "ours-cpu" not in calls[0] and "torch-cpu-" not in calls[0]
    res = json.loads((env["out"] / "board.json").read_text())
    assert set(res["races"]) == {"neural/gemm/gaussian/shape=small"}
    board = (env["out"] / "BOARD.md").read_text()
    assert "Neural, not planned: " in board and "mlp-infer" in board
    assert "Our CPU tier, no ours-cpu arm" not in board


def test_an_ours_cpu_cell_is_refused():
    race = bb.plan_races("apple", ["identical"], ["trees"], ["rf"], ["taxi"], 1000)[0]
    with pytest.raises(SystemExit, match="our CPU never races"):
        bb.base_cell({"vendor": "apple", "rounds": 1}, race, bb.CPU_ARM, "identical")


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
    assert not hasattr(probe, "ours_cpu_env") and not hasattr(probe, "ours_cpu_check")


def test_probe_refuses_any_arm_of_ours_on_the_cpu():
    for arm in ("ours-cpu", "ours-host", "ours-fast-cpu", "ours-host-train"):
        assert probe.is_our_cpu_arm(arm), arm
        with pytest.raises(SystemExit, match="our CPU is never raced or timed"):
            probe.refuse_our_cpu_arms(["ours", arm], "t")
    for arm in ("ours", "ours-fast", "ours-ab", "sklearn-cpu", "torch-cpu-eager-fp32"):
        assert not probe.is_our_cpu_arm(arm), arm
    assert probe.refuse_our_cpu_arms(["ours", "sklearn-cpu"]) == ["ours", "sklearn-cpu"]


def test_old_record_drops_every_cell_of_ours_on_the_cpu():
    """A board.json recorded before Oct 2 2026 may hold `ours-cpu` cells, a
    neural *-infer race whose ours ran on the host, and ratio_ours_cpu_over:
    none of it survives a load or reaches BOARD.md, and no pick uses it."""
    def c(arm, lib, dev, mode, ms, **kw):
        d = {"arm": arm, "library": lib, "device": dev, "mode": mode, "median_ms": ms,
             "min_ms": ms, "max_ms": ms, "rounds": 1, "status": "ok", "quality": {}}
        d.update(kw)
        return d
    rf = [c("ours-cpu", "mojolearn", "cpu", "identical", 1.0),
          c("ours", "mojolearn", "gpu", "identical", 50.0),
          c("sklearn-rf-cpu", "scikit-learn", "cpu", None, 100.0, ratio_ours_cpu_over=0.01)]
    assert bb.ours_of(rf, "identical")["arm"] == "ours"
    with pytest.raises(ValueError):
        bb.ours_of(rf, "cpu")
    bb.add_ratios(rf)
    assert rf[2]["ratio_ours_identical_over"] == pytest.approx(0.5)
    assert "ratio_ours_cpu_over" not in rf[2]
    res = {"races": {
        "trees/rf/taxi/rows=full": {"family": "trees", "lane": "rf", "dataset": "taxi",
                                    "rows": None, "status": "done", "cells": rf,
                                    "infer_cells": [dict(rf[0], batch="test")]},
        "neural/mlp-infer/x/shape=full": {"family": "neural", "lane": "mlp-infer", "dataset": "x",
                                          "status": "done", "cells": [
                                              c("ours", "mojolearn", "cpu", "identical", 2.0),
                                              c("torch-cpu-eager-fp32", "torch", "cpu", None, 3.0)]}}}
    board = bb.render_board(json.loads(json.dumps(res)))
    assert "mlp-infer" not in board.split("## Not covered")[0]
    assert "mojolearn CPU" not in board and "ours CPU" not in board
    out = bb.strip_our_cpu(res)
    assert set(out["races"]) == {"trees/rf/taxi/rows=full"}
    assert [x["arm"] for x in out["races"]["trees/rf/taxi/rows=full"]["cells"]] == \
        ["ours", "sklearn-rf-cpu"]
    assert out["races"]["trees/rf/taxi/rows=full"]["infer_cells"] == []


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
        mem.emit("ours-ab", 1, {"host_mb": 12.5, "host_method": "a=b c", "gpu_method": "cpu arm"})
    log = tmp_path / "rf.log"
    log.write_text(buf.getvalue())
    parsed = bb.parse_tree_log(str(log))
    assert len(parsed["mem"]["ours"]) == 3 and len(parsed["mem"]["sklearn-rf-cpu"]) == 3
    assert parsed["mem"]["ours-ab"][0]["host_mb"] == 12.5
    assert parsed["mem"]["ours-ab"][0]["host_method"] == "a:b c"     # no '=' in a value
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
