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
