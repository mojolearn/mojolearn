# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""tools/bench_board.py with STUB drivers: no mojolearn, no GPU, no dataset.

    .pixi/envs/test/bin/python -m pytest tools/test_bench_board.py

The stubs speak the real drivers' contracts (FSPEED lines for the trees
driver, the race JSON for the classical racer), so the planning, the parse,
the resume, the schema and the rendering are exercised end to end in seconds.
"""
import importlib.util
import json
import os
import re
import sys
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("bench_board", os.path.join(HERE, "bench_board.py"))
bb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bb)

BANNED = re.compile(r"faster|slower", re.I)

STUB_TREES = textwrap.dedent(r'''
    import argparse, os, sys
    p = argparse.ArgumentParser()
    p.add_argument("--lane"); p.add_argument("--dataset"); p.add_argument("--rows")
    p.add_argument("--devices"); p.add_argument("--arms"); p.add_argument("--ours-ab")
    p.add_argument("--ours-only", action="store_true")
    p.add_argument("--infer", action="store_true")    # tools/test_bench_board_infer.py
    p.add_argument("--ours-cpu", action="store_true")
    p.add_argument("--mem", action="store_true")
    a = p.parse_args()
    with open(os.environ["STUB_CALLS"], "a") as fh:
        fh.write("trees %s %s\n" % (a.lane, a.dataset))
    if os.environ.get("STUB_FAIL") == a.lane:
        sys.exit(3)
    n = int(os.environ["MOJOLEARN_SPEED_ROUNDS"])
    mode = os.environ["MOJOLEARN_NUMERIC_MODE"]
    shape = "%s-%sx16" % (a.dataset, a.rows or 4110786)
    ours = [("ours", mode, 100.0)]
    if a.ours_ab:
        ours.append(("ours-ab", a.ours_ab.split("=")[1].strip("'"), 80.0))
    if a.ours_cpu:
        ours.append(("ours-cpu", "identical", 150.0))
    opps = [(o, None, 200.0) for o in (a.arms.split(",") if a.arms else [])]
    for name, m, _ in ours:
        vendor = "cpu" if name == "ours-cpu" else os.environ["MOJOLEARN_SPEED_EXPECTED_VENDOR"]
        print("BENCH_BINDING arm=%s requested=%s resolved=%s compiled=%s vendor=%s "
              "path=/v/lib/python3.12/site-packages/mojolearn/_x.so"
              % (name, m, m, m, vendor))
    arms = ours + opps
    refuse = os.environ.get("STUB_REFUSE")
    for name, _, base in arms:
        if name == refuse:
            print("FSPEED-REFUSED lane=%s arm=%s reason=ImportError: no module, it is faster elsewhere"
                  % (a.lane, name))
            continue
        print("FSPEED-HEADER family=forest lane=%s arm=%s mode=%s device=x rounds=%d size=shipped"
              % (a.lane, name, mode.upper(), n))
        print("FSPEED-WARMUP lane=%s arm=%s shape=%s ms=%.3f" % (a.lane, name, shape, base * 2))
    for r in range(0 if a.mem else 1, n + 1):
        for name, _, base in arms:
            if name == refuse:
                continue
            if a.mem:
                print("FSPEED-MEM lane=%s arm=%s round=%d host_mb=%.1f gpu_mb=%s children_mb=0.0 "
                      "host_method=stub host peak gpu_method=%s"
                      % (a.lane, name, r, base * 10 + r, "-" if name.endswith("-cpu") else "7.0",
                         "cpu arm" if name.endswith("-cpu") else "stub gpu"))
            if r == 0:
                continue
            # ours-cpu predicts the same bits as ours (the promise); STUB_CPU_BITS=differ breaks it
            h = "ours" if name == "ours-cpu" and os.environ.get("STUB_CPU_BITS") != "differ" else name
            print("FSPEED lane=%s arm=%s shape=%s round=%d ms=%.3f hash=h%s"
                  % (a.lane, name, shape, r, base + r, h))
    for name, _, base in arms:
        if name == refuse:
            continue
        print("FSPEED-ACC lane=%s arm=%s metric=logloss value=%.6f" % (a.lane, name, base / 1000))
    print("FSPEED-NOTE lane=%s arms=ours metric=x delta=0 reason=this note says slower" % a.lane)
    print("FSPEED-FIT-VERDICT lane=%s arms=ours,x leaves=ours:10,x:10 spread=0.0000 verdict=COMPARABLE"
          % a.lane)
''')

STUB_CLASSICAL = textwrap.dedent(r'''
    import argparse, json, os, sys
    BLOCK = {"kmeans": "big", "pca": "big", "ols": "big", "knn": "knn", "kde": "kde",
             "svc": "svc", "dbscan": "dbscan"}
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    for f in ("--data", "--lanes", "--datasets", "--max-rows", "--lane", "--dataset", "--out",
              "--work", "--root", "--arms", "--rounds", "--round-seconds", "--warmup-seconds",
              "--ours-python", "--theirs-python"):
        p.add_argument(f)
    p.add_argument("--infer", action="store_true")    # tools/test_bench_board_infer.py
    a = p.parse_args()
    with open(os.environ["STUB_CALLS"], "a") as fh:
        fh.write("classical %s %s %s\n" % (a.cmd, a.lane or a.lanes, a.dataset or a.datasets))
    if a.cmd == "prep":
        os.makedirs(a.data, exist_ok=True)
        for lane in a.lanes.split(","):
            for ds in a.datasets.split(","):
                with open(os.path.join(a.data, "%s-%s.json" % (BLOCK[lane], ds)), "w") as fh:
                    json.dump({"arrays": {"X": {"shape": [int(a.max_rows or 4000000), 11]}}}, fh)
        sys.exit(0)
    assert a.ours_python and a.theirs_python
    n = int(a.rounds)
    arms = {}
    for i, arm in enumerate(a.arms.split(",")):
        ms = [10.0 * (i + 1) + r for r in range(n)]
        info = {"device": "cpu" if arm.endswith("-cpu") else "gpu", "version": "1.0",
                "library": "mojolearn" if arm.startswith("ours") else arm.split("-")[0]}
        if arm.startswith("ours"):
            info["numeric_mode_used"] = "fast" if arm == "ours-fast" else "identical"
            info["module_path"] = "/v/lib/python3.12/site-packages/mojolearn/__init__.py"
        if arm == "ours-cpu":
            info.update(device="cpu", vendor_used="cpu")
        mem = [{"host_mb": 100.0 * (i + 1) + r, "gpu_mb": None if info["device"] == "cpu" else 5.0,
                "host_method": "stub host", "gpu_method": "stub gpu"} for r in range(n + 1)]
        arms[arm] = {"ms": ms, "warmup_ms": 99.0, "digests": ["d"] * (n + 1), "status": "ok",
                     "info": info, "digest_stable": True, "mem": mem,
                     "span": {"input_home": "device" if arm.startswith("torch") else "host",
                              "pre_clock_fit": False}}
    out = {"lane": a.lane, "dataset": a.dataset,
           "block": {"arrays": {"X": {"shape": [4000, 11]}}}, "arms": arms,
           "quality": {arm: {"inertia": 1.5 + i, "n_iter": 20, "reference": "x"}
                       for i, arm in enumerate(arms)}}
    if "ours-cpu" in arms:
        out["quality"]["ours-cpu"]["bits_equal_vs_ours_identical"] = True
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "%s-%s.json" % (a.lane, a.dataset)), "w") as fh:
        json.dump(out, fh)
''')


STUB_NEURAL = textwrap.dedent(r'''
    import argparse, json, os, sys
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    for f in ("--lane", "--shape", "--arms", "--rounds", "--out", "--work", "--ours-python",
              "--theirs-python", "--ready-seconds", "--warmup-seconds", "--round-seconds",
              # FAIRNESS FIX (2026-10-13 audit): a compile arm's own warm-up deadline
              "--compile-warmup-seconds"):
        p.add_argument(f)
    p.add_argument("--keep-outputs", action="store_true")
    a = p.parse_args()
    assert a.cmd == "race" and a.ours_python and a.theirs_python and a.keep_outputs
    with open(os.environ["STUB_CALLS"], "a") as fh:
        fh.write("neural %s %s %s\n" % (a.lane, a.shape, a.arms))
    if os.environ.get("STUB_FAIL") == a.lane:
        sys.exit(3)
    n = int(a.rounds)
    data = "bytes" if a.lane.split("-")[0] in ("lm", "samba") else "gaussian"
    if a.lane.endswith("-fit") or a.lane == "tape-user-model":
        data = "higgs"                    # the fit lanes (bench_board_neural.FIT_LANES)
    if a.lane in ("sequential-fit", "sequential-predict"):
        data = "synthetic"                # bench_board_neural.SYNTH_FIT_LANES
    if a.lane.startswith("causal-lm-generate"):     # bench_board_neural.GEN_LANES: the checkpoint is the dataset
        data = "smollm2-135m" if a.lane.endswith("-smollm2") else "mojolearn-162m"
    arms, qual = {}, {}
    for i, arm in enumerate(a.arms.split(",")):
        ours = arm in ("ours", "ours-cpu")
        cpu = "-cpu-" in arm or arm == "ours-cpu" or (ours and a.lane.endswith("-infer"))
        info = {"device": "cpu" if cpu else "gpu", "version": "1.0",
                "library": "mojolearn" if ours else "torch",
                "device_name": "stub gpu"}
        if ours:
            info["numeric_mode_used"] = "identical"
            info["module_path"] = "/v/lib/python3.12/site-packages/mojolearn/_x.so"
        if arm == "ours-cpu":
            info["vendor_used"] = "cpu"
        if os.environ.get("STUB_REFUSE") == arm:
            arms[arm] = {"ms": [], "status": "not_ready", "error": {"error": "no MPS, it is faster"},
                         "info": info}
            continue
        base = 80.0 if ours else 40.0 + 10.0 * (i - 1)
        arms[arm] = {"ms": [base + r for r in range(n)], "warmup_ms": 500.0,
                     "digests": [None] * (n + 1), "status": "ok", "info": info,
                     "digest_stable": None,
                     "span": {"input_home": "host", "pre_clock_fit": False}}
        qual[arm] = {"loss_last_step": 5.5 + i * 1e-4, "steps": n + 1}
        if not ours:
            qual[arm]["loss_last_abs_diff_vs_ours"] = 1e-4
        if arm == "ours-cpu":
            qual[arm]["bits_equal_vs_ours_identical"] = True
    out = {"lane": a.lane, "dataset": data, "shape": "stub-%s" % a.shape, "arms": arms,
           "quality": qual, "inputs": {"seed": 7}}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "%s-%s.json" % (a.lane, data)), "w") as fh:
        json.dump(out, fh)
''')


STUB_MORE = textwrap.dedent(r"""
    import argparse, json, os, sys
    import importlib.util
    spec = importlib.util.spec_from_file_location("bbm", os.path.join(os.environ["STUB_TOOLS"],
                                                                       "bench_board_more.py"))
    bbm = importlib.util.module_from_spec(spec); spec.loader.exec_module(bbm)
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    for f in ("--data", "--lanes", "--datasets", "--max-rows", "--lane", "--dataset", "--out",
              "--work", "--arms", "--rounds", "--ours-python", "--theirs-python",
              "--ready-seconds", "--warmup-seconds", "--round-seconds"):
        p.add_argument(f)
    a = p.parse_args()
    with open(os.environ["STUB_CALLS"], "a") as fh:
        fh.write("classical2 %s %s %s\n" % (a.cmd, a.lane or a.lanes, a.dataset or a.datasets))
    if a.cmd == "prep":
        os.makedirs(a.data, exist_ok=True)
        for lane in a.lanes.split(","):
            for ds in bbm.datasets_of(lane):
                if ds in ("taxi", "istella") and ds not in a.datasets.split(","):
                    continue
                with open(os.path.join(a.data, "%s-%s.json" % (bbm.block_of(lane), ds)), "w") as fh:
                    json.dump({"block": bbm.block_of(lane)}, fh)
        sys.exit(0)
    assert os.path.exists(os.path.join(a.data, "%s-%s.json" % (bbm.block_of(a.lane), a.dataset)))
    if os.environ.get("STUB_FAIL") == a.lane:
        sys.exit(3)
    n = int(a.rounds)
    arms, qual = {}, {}
    for i, arm in enumerate(a.arms.split(",")):
        ours = arm.startswith("ours")
        lib = "mojolearn" if ours else {"umap": "umap-learn"}.get(arm.split("-")[0], arm.split("-")[0])
        info = {"device": "cpu" if "-cpu" in arm else "gpu", "version": "1.0", "library": lib}
        if ours:
            info["numeric_mode_used"] = "fast" if arm == "ours-fast" else "identical"
            info["module_path"] = "/v/lib/python3.12/site-packages/mojolearn/__init__.py"
        if os.environ.get("STUB_REFUSE") == arm:
            arms[arm] = {"ms": [], "status": "not_ready", "error": {"error": "no module, faster"},
                         "info": info}
            continue
        arms[arm] = {"ms": [30.0 * (i + 1) + r for r in range(n)], "warmup_ms": 90.0,
                     "digests": ["d"] * (n + 1), "status": "ok", "info": info,
                     "digest_stable": True,
                     "span": {"input_home": "device" if arm.startswith("cu") else "host",
                              "pre_clock_fit": False}}
        qual[arm] = {"trustworthiness_k15": 0.9 - 0.01 * i}
    out = {"lane": a.lane, "dataset": a.dataset, "shape": "X 2000x11", "arms": arms,
           "quality": qual, "lane_config": bbm.LANE_CONFIG[a.lane]}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "%s-%s.json" % (a.lane, a.dataset)), "w") as fh:
        json.dump(out, fh)
""")


@pytest.fixture
def env(tmp_path, monkeypatch):
    # These fake-driver schema tests cannot produce real worker receipts or
    # upload evidence. Evidence/preflight integration has separate CPU tests.
    run_race = bb.run_race
    monkeypatch.setattr(bb, "run_race", lambda ctx, race: run_race(dict(ctx, capture_measurements=False), race))
    monkeypatch.setattr(bb, "audit_plan", lambda ctx, races: dict(verdict="MATCHED", races={}))
    # The stub drivers stand in for our arm, so the GPU-set load probe stands in too: with the source tree's
    # mojolearn on PYTHONPATH and no built bindings (a CPU test pod) the real probe refuses every NVIDIA/AMD run.
    # The probe itself is tested in test_gpu_set_refusal_names_the_wheels_reason.
    monkeypatch.setattr(bb, "gpu_set_refusal", lambda python, vendor: None)
    data = tmp_path / "data"
    for rel in bb.DATA_FILES.values():
        f = data / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x" * 16)
    trees = tmp_path / "stub_trees.py"
    trees.write_text(STUB_TREES)
    classical = tmp_path / "stub_classical.py"
    classical.write_text(STUB_CLASSICAL)
    neural = tmp_path / "stub_neural.py"
    neural.write_text(STUB_NEURAL)
    more = tmp_path / "stub_more.py"
    more.write_text(STUB_MORE)
    monkeypatch.setenv("STUB_TOOLS", HERE)
    calls = tmp_path / "calls.txt"
    calls.write_text("")
    monkeypatch.setenv("STUB_CALLS", str(calls))
    monkeypatch.delenv("STUB_FAIL", raising=False)
    monkeypatch.delenv("STUB_REFUSE", raising=False)
    out = tmp_path / "out"
    base = ["--out", str(out), "--python-env", sys.executable, "--skip-install",
            "--with-opponents",  # These stub schema tests explicitly exercise opponent cells.
            "--data-root", str(data), "--tree-driver", str(trees),
            "--classical-driver", str(classical), "--neural-driver", str(neural),
            "--more-driver", str(more),
            "--rows", "1000", "--rounds", "3", "--no-cpu-arm"]
    return {"out": out, "base": base, "calls": calls, "tmp": tmp_path}


def _calls(env):
    return [l for l in env["calls"].read_text().splitlines() if l]


# --- vendor and modes -------------------------------------------------------

def test_detect_vendor():
    none = lambda name: None                                   # noqa: E731
    assert bb.detect_vendor("Darwin", none, lambda p: False) == "apple"
    assert bb.detect_vendor("Linux", lambda n: "/x" if n == "nvidia-smi" else None,
                            lambda p: False) == "nvidia"
    assert bb.detect_vendor("Linux", lambda n: "/x" if n == "rocm-smi" else None,
                            lambda p: False) == "amd"
    assert bb.detect_vendor("Linux", none, lambda p: p == "/dev/kfd") == "amd"
    # both tools present: still NVIDIA
    assert bb.detect_vendor("Linux", lambda n: "/x", lambda p: True) == "nvidia"
    assert bb.detect_vendor("Linux", none, lambda p: False) is None


def test_full_plan_refusal_prevents_every_timed_race(env, monkeypatch):
    monkeypatch.setattr(bb, "audit_plan", lambda ctx, races: dict(verdict="REFUSED", races={}))
    timed = []
    monkeypatch.setattr(bb, "run_race", lambda *a: timed.append(a))
    rc = bb.main(env["base"] + ["--vendor", "amd", "--families", "trees", "--lanes", "rf",
                              "--datasets", "taxi", "--no-smoke-gate"])
    assert rc == 1
    assert not timed
    assert json.loads((env["out"] / "board.json").read_text())["preflight"]["verdict"] == "REFUSED"


@pytest.mark.parametrize("vendor", ["nvidia", "amd"])
def test_explicit_gpu_only_excludes_cpu_fallbacks(vendor):
    races = bb.gpu_opponents_only(bb.plan_races(vendor, ["identical"]), vendor)
    assert all(bb.arm_device(a, vendor) == "gpu" for r in races for a in r["arms"])


def test_modes_apple_both_others_identical_only():
    assert bb.modes_for("apple") == ["fast", "identical"]
    assert bb.modes_for("nvidia") == ["identical"]
    assert bb.modes_for("amd") == ["identical"]
    assert bb.modes_for("apple", "identical") == ["identical"]
    assert bb.modes_for("apple", "identical,fast") == ["fast", "identical"]
    for v in ("nvidia", "amd"):
        with pytest.raises(SystemExit):
            bb.modes_for(v, "fast")
        with pytest.raises(SystemExit):
            bb.modes_for(v, "fast,identical")


# --- planning ---------------------------------------------------------------

def test_plan_apple_carries_fast_and_identical_arms():
    races = bb.plan_races("apple", bb.modes_for("apple"))
    more = sum(len([d for d in bb.MORE.datasets_of(l) if d in bb.DATASETS]) or 1
               for l in bb.MORE_LANES)
    tasks = sum(len(bb.tree_task_datasets(l, bb.DATASETS)) for l in bb.TREE_TASK_LANES)
    algos = sum(len(bb.ALGOS.datasets_of(l)) for l in bb.ALGOS_LANES)
    assert len(races) == ((len(bb.TREE_LANES) + len(bb.CLASSICAL_LANES)) * len(bb.DATASETS)
                          + len([l for l in bb.NEURAL_LANES if bb.NEURAL.DEVICE_OF.get(l) != "cpu"]) + more + tasks + algos)
    for r in races:
        if r["family"] == "neural":
            want = ({"ours-fast": "fast"} if r["lane"] in bb.NEURAL.FIT_LANES
                    else {"ours": "identical", "ours-fast": "fast"})
            assert r["our_arms"] == want, r["id"]
            continue
        modes = sorted(set(r["our_arms"].values()))
        assert modes == ["fast", "identical"], r["id"]
        if r["family"] == "trees":
            # smoke-fixes r13: gbdt-lossguide races its second own configuration (tools/board_pairing.py)
            extra = {a: "identical" for a in bb.PAIR.extra_own_arms(r["lane"])}
            if r["lane"] == "gbdt-categorical":
                extra = {}  # Apple has the CPU CatBoost pairing only.
            assert r["our_arms"] == dict({"ours": "identical", "ours-ab": "fast"}, **extra)
            assert all(o.endswith("-cpu") for o in r["opponents"])
        else:
            assert r["our_arms"] == {"ours": "identical", "ours-fast": "fast"}
            assert "cuml-gpu" not in r["opponents"]


@pytest.mark.parametrize("vendor", ["nvidia", "amd"])
def test_plan_gpu_boxes_identical_only(vendor):
    races = bb.plan_races(vendor, bb.modes_for(vendor))
    for r in races:
        # smoke-fixes r13: gbdt-lossguide races its second own configuration (tools/board_pairing.py)
        extra = {a: "identical" for a in bb.PAIR.extra_own_arms(r["lane"])} if r["family"] == "trees" else {}
        expected = dict({"ours": "identical"}, **extra)
        if r["family"] == "trees" and r["lane"] == "gbdt-categorical":
            expected = {"ours-catboost-gpu" if vendor == "nvidia" else "ours": "identical"}
        assert r["our_arms"] == expected
        assert "ours-ab" not in r["arms"] and "ours-fast" not in r["arms"]
    nv = {r["lane"]: r["opponents"] for r in races if r["family"] == "trees"}
    if vendor == "nvidia":
        assert nv["gbdt-symmetric"] == ["catboost-gpu"]
        assert nv["rf"] == ["cuml-rf-gpu"]
    else:
        assert "cuml-rf-gpu" not in nv["rf"]


@pytest.mark.parametrize("vendor", ["apple", "nvidia", "amd"])
def test_plan_races_only_our_gpu_and_never_mixes_cpu_and_gpu_opponents(vendor):
    """Andrew, Oct 2 2026: the board races only our GPU; our GPU races GPU
    opponents only, and a race keeps CPU opponents only when it has no GPU one."""
    races = bb.plan_races(vendor, bb.modes_for(vendor))
    assert races
    for r in races:
        assert bb.CPU_ARM not in r["arms"], r["id"]
        assert all(bb.arm_device(a, vendor) == "gpu" for a in r["our_arms"]), r["id"]
        devices = {bb.arm_device(a, vendor) for a in r["opponents"]}
        assert len(devices) <= 1, (r["id"], r["opponents"])
    assert bb.gpu_opponents_first(["sklearn-cpu", "cuml-gpu", "lightgbm-cpu"]) == ["cuml-gpu"]
    assert bb.gpu_opponents_first(["sklearn-cpu", "lightgbm-cpu"]) == ["sklearn-cpu", "lightgbm-cpu"]


def test_plan_filters_and_counts():
    races = bb.plan_races("apple", ["fast", "identical"], ["trees"], ["rf"], ["taxi"], 5000)
    assert [r["id"] for r in races] == ["trees/rf/taxi/rows=5000"]
    s = bb.plan_summary(races)
    assert s == {"races": 1, "cells": 4, "unsupported_races": 0,
                 "by_family": {"trees": {"races": 1, "cells": 4}}}


def test_tree_command_apple_interleaves_fast():
    race = bb.plan_races("apple", ["fast", "identical"], ["trees"], ["iforest"], ["taxi"], None)[0]
    ctx = {"python": "py", "tree_driver": "drv", "vendor": "apple", "rounds": 5,
           "arm_budget_s": 1, "race_deadline_s": 2}
    cmd, env = bb.tree_cmd(ctx, race)
    assert "--ours-ab" in cmd and cmd[cmd.index("--ours-ab") + 1] == "numeric_mode='fast'"
    assert "--ours-cpu" not in cmd and "--mem" in cmd
    off = bb.plan_races("apple", ["fast", "identical"], ["trees"], ["iforest"], ["taxi"], None)[0]
    assert "--ours-cpu" not in bb.tree_cmd(ctx, off)[0] and "--mem" in bb.tree_cmd(ctx, off)[0]
    assert "--rows" not in cmd          # full size: no cap
    assert env["MOJOLEARN_NUMERIC_MODE"] == "identical"
    assert env["MOJOLEARN_SPEED_EXPECTED_VENDOR"] == "metal"
    assert env["MOJOLEARN_SPEED_ROUNDS"] == "5"
    ctx["vendor"] = "nvidia"
    race = bb.plan_races("nvidia", ["identical"], ["trees"], ["gbdt-lossguide"], ["istella"], None)[0]
    cmd, env = bb.tree_cmd(ctx, race)
    assert "--ours-ab" not in cmd
    assert cmd[cmd.index("--devices") + 1] == "gpu"
    assert env["MOJOLEARN_SPEED_EXPECTED_VENDOR"] == "cuda"


def test_opponent_pins_come_from_opponent_wheels_sh():
    pins = bb.opponent_pins()
    trees = next(v for k, v in pins.items() if k.startswith("trees-"))
    assert any(r.startswith("catboost==") for r in trees[1])
    assert all("==" in r for r in trees[1])
    groups = bb.opponent_requirements("nvidia", pins)
    assert len(groups) == 2 and groups[1][0]            # rapids has its extra index
    assert len(bb.opponent_requirements("apple", pins)) == 1


def test_fairness_fix_20261013_amd_installs_rocm_xgboost(monkeypatch):
    """FAIRNESS FIX (2026-10-13 audit): on AMD the board installs AMD's ROCm build of the same
    XGBoost release (amd_xgboost 3.2.0) from the index matching the box's ROCm, instead of PyPI's
    CUDA wheel, and falls back to PyPI's xgboost when no index fits. Results raced before this
    change were wrong for the AMD tree lanes; apply extra scrutiny."""
    import tempfile
    pins = bb.opponent_pins()
    rocm = next(v for k, v in pins.items() if k.startswith("rocm-xgboost-"))
    assert rocm[1] == ["amd_xgboost==3.2.0"] and rocm[0].startswith("https://pypi.amd.com/")
    trees = next(v for k, v in pins.items() if k.startswith("trees-"))
    assert "xgboost==3.2.0" in trees[1]           # the same release on every vendor
    monkeypatch.delenv("MOJOLEARN_AMD_XGB_INDEX", raising=False)
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "version")
        for box, want in (("7.1.1-38", "rocm-7.1.1"), ("7.2.3", "rocm-7.2.3"),
                          ("7.0.2-56", "rocm-7.0.2"), ("7.1.0", "rocm-7.0.2")):
            with open(f, "w") as fh:
                fh.write(box)
            assert want in bb.amd_xgboost_index("DEFAULT", f), box
        with open(f, "w") as fh:
            fh.write("6.4.4")
        assert bb.amd_xgboost_index("DEFAULT", f) is None
        assert bb.amd_xgboost_index("DEFAULT", os.path.join(d, "missing")) == "DEFAULT"
    monkeypatch.setattr(bb, "amd_xgboost_index", lambda default, *a: default)
    groups = bb.opponent_requirements("amd", pins)
    assert not any(r.startswith("xgboost==") for _, reqs in groups for r in reqs)
    assert groups[-1] == (rocm[0], ["amd_xgboost==3.2.0"])
    monkeypatch.setattr(bb, "amd_xgboost_index", lambda default, *a: None)
    assert bb.opponent_requirements("amd", pins) == [trees]
    assert bb.amd_xgboost_fallback(pins) == ["xgboost==3.2.0"]
    for v in ("apple", "nvidia"):
        assert bb.opponent_requirements(v, pins)[0] == trees


def test_dry_run_prints_plan_and_touches_nothing(env, capsys):
    rc = bb.main(["--dry-run", "--vendor", "apple"] + env["base"])
    assert rc == 0
    text = capsys.readouterr().out
    # the races outside the algos family: 93 since Oct 2 2026, when the eight neural lanes
    # that ran OUR CPU binding left the board and CPU opponents left races that have a GPU one;
    # 99 since Oct 7 2026 (347f4b2d9): the six GPU inference lanes transformer-decode,
    # mamba1-decode, mamba2-decode, mamba3-decode, samba-decode and mlp-predict joined
    algos = bb.plan_races("apple", bb.modes_for("apple"), ["algos"], rows=1000)
    before = bb.plan_races("apple", bb.modes_for("apple"), bb.FAMILIES[:-1], rows=1000)
    # 328 since Oct 3 2026: the Apple FAST neural tier adds `ours-fast` to the neural races;
    # 346 since Oct 7 2026: +18 cells from the six inference lanes (transformer-decode
    # ours, ours-fast, torch-eager-fp32, torch-eager-bf16 = 4; the four Mamba/Samba decode
    # lanes race ours and ours-fast alone = 8; mlp-predict ours, ours-fast + 4 torch = 6)
    # FAIRNESS FIX (2026-10-13 audit): 342 since the classical2 linearsvc / linearsvr races on
    # Apple lost their scikit-learn arm (liblinear with a penalized intercept is a different
    # algorithm and model from ours' L-BFGS primal; bench_board_more.REFUSED): 2 lanes x 2 datasets
    # FAIRNESS FIX (2026-10-13 audit, decomposition group): 340 since the classical2 tsvd race on
    # Apple lost its scikit-learn arm (ARPACK, a different algorithm from ours' covariance_eigh;
    # bench_board_more.REFUSED): 1 lane x 2 datasets
    # smoke-fixes (2026-10-14): 331 since gbdt-multiclass, gbdt-categorical, gbdt-rank-pairlogit and gbdt-rank-yetirank
    # race CatBoost only (the tree driver refuses XGBoost/LightGBM there by name on every device): -9 trees cells
    # smoke-fixes r13 (2026-10-14): 333, gbdt-lossguide races `ours-gain0` beside `ours` (+1 cell x 2 datasets);
    # 335, gbdt-depthwise races `ours-prebinned` beside `ours` (+1 cell x 2 datasets)
    # 2026-10-19 (lane ag-bench): 104 races, 365 cells with the five neural fit lanes, each on Apple
    # ours-fast + 4 torch GPU settings + mlx-fp32 (+5 races, +30 cells)
    # 2026-10-20 (lane ag-inference-board): 109 races with sequential-fit, ops-session-step, sequential-predict and the two
    # causal-lm-generate lanes; the cell total is derived from the plan so a new arm does not stale this line.
    cells = sum(len(r["arms"]) for r in before)
    assert len(before) == 109
    assert "TOTAL races=%d cells=%d" % (109 + len(algos), cells + sum(len(r["arms"]) for r in algos)) in text
    assert "family algos" in text
    # every algos race names whether its class is in the source tree (once every
    # lane has merged its classes, no race reads "not built yet")
    algo_lines = [ln for ln in text.splitlines() if ln.startswith("RACE algos/")]
    assert algo_lines and all("[in source]" in ln or "not built yet: SKIPPED" in ln
                              for ln in algo_lines)
    assert "our CPU: never raced (the board races only our GPU)" in text
    # neural on Apple: the 18 GPU lanes (the 8 that ran our CPU binding left the board; the
    # six GPU inference lanes *-decode and mlp-predict joined Oct 7 2026), each `ours` plus
    # its torch GPU arms: 64 cells identical only, 82 with the Apple FAST arm `ours-fast`
    # (see test_plan_neural_fast_on_apple_only)
    # (2026-10-19) the five fit lanes race ours-fast only on Apple (identical is never timed there), so an
    # identical-only Apple plan has none of them and the full plan adds 5 races of 6 cells each
    neural = bb.plan_races("apple", ["identical"], ["neural"])
    assert len(neural) == 21  # 2026-10-20: + sequential-predict and the two causal-lm-generate lanes (cells vary by vendor)
    both = bb.plan_races("apple", bb.modes_for("apple"), ["neural"])
    assert len(both) == 28  # 2026-10-20: + sequential-predict and the two causal-lm-generate lanes at both shapes
    assert "family neural     races=%d cells=%d" % (len(both), sum(len(r["arms"]) for r in both)) in text
    assert "ours-ab[fast]" in text and "ours-fast[fast]" in text
    assert not env["out"].exists()
    assert _calls(env) == []
    rc = bb.main(["--dry-run", "--vendor", "nvidia", "--rows", "full"])
    text = capsys.readouterr().out
    assert "modes=identical" in text and "[fast]" not in text


# --- a run with stub drivers ------------------------------------------------

def _run(env, *extra):
    return bb.main(["--vendor", "apple", "--lanes", "rf,kmeans"] + env["base"] + list(extra))


def test_run_schema_ratios_quality_and_board(env):
    assert _run(env) == 0
    res = json.loads((env["out"] / "board.json").read_text())
    assert res["schema"] == bb.SCHEMA
    for k in ("host", "os", "gpu", "python", "packages", "mojolearn", "repo"):
        assert k in res["box"], k
    assert res["box"]["repo"]["script_sha256"]
    assert res["config"]["modes"] == ["fast", "identical"]
    assert res["config"]["smoke"] is True
    assert set(res["races"]) == set(res["plan"]) == {
        "trees/rf/taxi/rows=1000", "trees/rf/istella/rows=1000",
        "classical/kmeans/taxi/rows=1000", "classical/kmeans/istella/rows=1000"}
    required = ("family", "lane", "dataset", "rows", "mode", "arm", "library", "settings",
                "times_ms", "median_ms", "quality", "verdict", "status",
                "ratio_ours_identical_over", "ratio_ours_fast_over")
    for rec in res["races"].values():
        assert rec["status"] == "done"
        for c in rec["cells"]:
            for k in required:
                assert k in c, (rec["id"], k)
            assert c["settings"]["seed"] == 7
            assert c["status"] == "ok", (rec["id"], c["arm"], c["status"])
            assert len(c["times_ms"]) == 3
            assert c["quality"]
    rf = {c["arm"]: c for c in res["races"]["trees/rf/taxi/rows=1000"]["cells"]}
    assert rf["ours"]["mode"] == "identical" and rf["ours-ab"]["mode"] == "fast"
    assert rf["ours"]["installed_wheel"] == "wheel"
    assert rf["ours"]["verdict"] == "COMPARABLE"
    # median ours 102, ours-ab 82, opponents 202
    assert rf["sklearn-rf-cpu"]["ratio_ours_identical_over"] == pytest.approx(102 / 202)
    assert rf["sklearn-rf-cpu"]["ratio_ours_fast_over"] == pytest.approx(82 / 202)
    assert rf["ours"]["ratio_ours_identical_over"] is None
    # never FAST over IDENTICAL: that is the cost of identity, not a board number
    assert rf["ours-ab"]["ratio_ours_identical_over"] is None
    assert rf["ours"]["ratio_ours_fast_over"] is None
    km = {c["arm"]: c for c in res["races"]["classical/kmeans/taxi/rows=1000"]["cells"]}
    assert km["ours-fast"]["mode"] == "fast" and km["ours-fast"]["mode_witness"] == "fast"
    # FAIRNESS FIX (2026-10-13 audit): torch-gpu (the harness's own k-means) is no
    # opponent; Apple has no library GPU k-means, so its CPU opponent races. Results
    # raced before this change were wrong for this lane; apply extra scrutiny.
    assert "torch-gpu" not in km
    assert km["sklearn-cpu"]["verdict"] == "LIKE-FOR-LIKE-SPAN"
    assert "reference" not in km["ours"]["quality"]
    # classical blocks live in the cache, not beside board.json
    assert (env["out"] / "cache" / "ctd-data" / "rows-1000" / "big-taxi.json").exists()
    board = (env["out"] / "BOARD.md").read_text()
    assert "Quality at a glance" in board and "SMOKE RUN" in board
    assert "mojolearn FAST" in board and "mojolearn IDENTICAL" in board
    assert not BANNED.search(board)


def test_resume_skips_finished_and_retries_failed(env, monkeypatch):
    monkeypatch.setenv("STUB_FAIL", "rf")
    assert _run(env) == 1
    first = _calls(env)
    assert sum(1 for c in first if c.startswith("trees")) == 2
    res = json.loads((env["out"] / "board.json").read_text())
    assert res["races"]["trees/rf/taxi/rows=1000"]["status"] == "failed"
    assert res["races"]["classical/kmeans/taxi/rows=1000"]["status"] == "done"
    # a rerun retries ONLY the failed races; prep is not repeated
    monkeypatch.delenv("STUB_FAIL")
    env["calls"].write_text("")
    assert _run(env) == 0
    again = _calls(env)
    assert sorted(again) == ["trees rf istella", "trees rf taxi"]
    # and a third run does nothing at all
    env["calls"].write_text("")
    assert _run(env) == 0
    assert _calls(env) == []


def test_skip_failed_flag(env, monkeypatch):
    monkeypatch.setenv("STUB_FAIL", "rf")
    _run(env)
    env["calls"].write_text("")
    assert _run(env, "--skip-failed") == 1
    assert _calls(env) == []


def test_resume_refuses_a_different_box(env):
    assert _run(env) == 0
    p = env["out"] / "board.json"
    res = json.loads(p.read_text())
    res["box"]["host"]["hostname"] = "some-other-box"
    p.write_text(json.dumps(res))
    with pytest.raises(SystemExit, match="different box"):
        _run(env)


def _git(repo, *a):
    import subprocess
    return subprocess.run(["git", "-C", str(repo)] + list(a), check=True, capture_output=True,
                          text=True).stdout.strip()


def test_repo_commit_names_the_patch_synced_commit_not_the_pods_base(tmp_path):
    repo = tmp_path / "tree"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("a")
    _git(repo, "add", "a.txt")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "a.txt").write_text("patched")        # the applied patch
    # no sync record: the tree's own HEAD, dirty
    assert bb.repo_commit(str(repo)) == base + "-dirty"
    assert bb.repo_sync(str(repo)) is None
    synced = "c" * 40
    (repo / ".git" / "devpod_synced").write_text(
        "commit=%s\nworktree_dirty=0\nbase=%s\npatch_sha256=%s\n" % (synced, base, "d" * 64))
    assert bb.repo_commit(str(repo)) == synced
    assert bb.repo_sync(str(repo))["patch_sha256"] == "d" * 64
    (repo / ".git" / "devpod_synced").write_text(
        "commit=%s\nworktree_dirty=1\nbase=%s\npatch_sha256=%s\n" % (synced, base, "d" * 64))
    assert bb.repo_commit(str(repo)) == synced + "-dirty"
    # a record whose base is not HEAD (the tree moved since) is ignored
    (repo / ".git" / "devpod_synced").write_text("commit=%s\nbase=%s\n" % (synced, "e" * 40))
    assert bb.repo_commit(str(repo)) == base + "-dirty"


def test_gpu_set_refusal_names_the_wheels_reason(tmp_path):
    fake = tmp_path / "py"
    fake.write_text("#!/bin/sh\necho 'ImportError: this device is sm_86 ... Carried: sm_89'; exit 3\n")
    fake.chmod(0o755)
    assert "sm_86" in bb.gpu_set_refusal(str(fake), "nvidia")
    assert bb.gpu_set_refusal(str(fake), "apple") is None
    ok = tmp_path / "ok"
    ok.write_text("#!/bin/sh\necho OK\n")
    ok.chmod(0o755)
    assert bb.gpu_set_refusal(str(ok), "amd") is None


def _nv_box(host="pod-a", gpu="NVIDIA A40", driver="580.159.04", cuml="26.8.0"):
    return {"host": {"hostname": host}, "gpu": {"vendor": "nvidia", "name": gpu, "driver": driver},
            "mojolearn": {"version": "0.8.25", "wheel": {"sha256": "ab"}},
            "packages": {"cuml-cu12": cuml, "xgboost": "3.2.0", "torch": "2.4.1+cu124",
                         "unrelated": "1"}}


def test_nvidia_resume_key_ignores_the_pod_but_not_the_gpu_driver_or_pins():
    k = bb.box_key(_nv_box())
    assert "hostname" not in k
    assert k["driver_major"] == "580"
    assert k["pinned"] == {"cuml-cu12": "26.8.0", "xgboost": "3.2.0", "torch": "2.4.1+cu124"}
    # a new pod with the same GPU model, driver major and pins resumes
    assert bb.box_key(_nv_box(host="pod-b", driver="580.95.05")) == k
    # a different GPU model, driver major or pinned version does not
    assert bb.box_key(_nv_box(gpu="NVIDIA GeForce RTX 4090")) != k
    assert bb.box_key(_nv_box(driver="570.1")) != k
    assert bb.box_key(_nv_box(cuml="26.6.0")) != k
    # every other vendor keeps the hostname
    apple = {"host": {"hostname": "m3ultra-b"}, "gpu": {"vendor": "apple", "name": "M3 Ultra"}}
    assert bb.box_key(apple)["hostname"] == "m3ultra-b"


def test_races_record_the_host_they_ran_on(env, monkeypatch):
    monkeypatch.setenv("RUNPOD_POD_ID", "podxyz")
    assert _run(env) == 0
    res = json.loads((env["out"] / "board.json").read_text())
    for rec in res["races"].values():
        assert rec["host"]["pod_id"] == "podxyz" and rec["host"]["hostname"]
    assert "(pod podxyz)" in (env["out"] / "BOARD.md").read_text()


def test_refusal_and_driver_text_never_put_direction_words_on_board(env, monkeypatch):
    monkeypatch.setenv("STUB_REFUSE", "lightgbm-cpu")
    assert _run(env) == 0
    res = json.loads((env["out"] / "board.json").read_text())
    lg = next(c for c in res["races"]["trees/rf/taxi/rows=1000"]["cells"] if c["arm"] == "lightgbm-cpu")
    assert lg["status"].startswith("REFUSED(")
    assert lg["ratio_ours_identical_over"] is None
    board = (env["out"] / "BOARD.md").read_text()
    assert "REFUSED" in board
    assert not BANNED.search(board)
    assert not BANNED.search(bb.render_board(res))


def test_missing_arm_is_unknown_not_blank(env, monkeypatch):
    monkeypatch.setenv("STUB_REFUSE", "sklearn-rf-cpu")
    _run(env)
    res = json.loads((env["out"] / "board.json").read_text())
    arms = {c["arm"]: c["status"] for c in res["races"]["trees/rf/istella/rows=1000"]["cells"]}
    # the stub prints no line at all for a refused arm except FSPEED-REFUSED
    assert arms["sklearn-rf-cpu"].startswith("REFUSED")
    assert set(arms) >= {"ours", "ours-ab", "sklearn-rf-cpu", "lightgbm-cpu"}


def test_missing_dataset_refuses_without_download(env):
    os.remove(os.path.join(str(env["tmp"]), "data", bb.DATA_FILES["istella"]))
    with pytest.raises(SystemExit, match="never downloads"):
        _run(env)
    assert _calls(env) == []


def test_render_only(env):
    _run(env)
    (env["out"] / "BOARD.md").unlink()
    assert bb.main(["--render-only", "--out", str(env["out"])]) == 0
    assert (env["out"] / "BOARD.md").exists()


def test_render_board_on_an_empty_result_has_no_direction_words():
    text = bb.render_board({"schema": bb.SCHEMA, "races": {}, "plan": []})
    assert "mojolearn benchmark board" in text
    assert not BANNED.search(text)
    assert bb.clean("ours is Faster and theirs slower") == \
        "ours is [direction word removed] and theirs [direction word removed]"


# --- the neural family ------------------------------------------------------

NEURAL_IDS = [
    "neural/lm-train-step/bytes/shape=full", "neural/lm-forward/bytes/shape=full",
    "neural/gemm/gaussian/shape=full",
    "neural/transformer-forward/gaussian/shape=full",
    "neural/mamba1-forward/gaussian/shape=full",
    "neural/mamba2-forward/gaussian/shape=full",
    "neural/mamba3-forward/gaussian/shape=full",
    "neural/samba-train-step/bytes/shape=full", "neural/samba-forward/bytes/shape=full",
    "neural/mlp-train-step/gaussian/shape=full",
    # the GPU inference lanes (Oct 7 2026, 347f4b2d9)
    "neural/transformer-decode/gaussian/shape=full",
    "neural/mamba1-decode/gaussian/shape=full",
    "neural/mamba2-decode/gaussian/shape=full",
    "neural/mamba3-decode/gaussian/shape=full",
    "neural/samba-decode/bytes/shape=full",
    "neural/mlp-predict/gaussian/shape=full",
    "neural/gemm-bf16/gaussian/shape=full", "neural/gemm-int8/gaussian/shape=full",
    # the whole-fit lanes of the tape-trained models (2026-10-19, lane ag-bench)
    "neural/mlp-fit/higgs/shape=full", "neural/lstm-fit/higgs/shape=full",
    "neural/gru-fit/higgs/shape=full", "neural/rnn-fit/higgs/shape=full",
    "neural/tape-user-model/higgs/shape=full",
    # the public tape API lanes (2026-10-19, lane ag-board-twins)
    "neural/sequential-fit/synthetic/shape=full", "neural/ops-session-step/gaussian/shape=full",
    # the captured-program lane (2026-10-19, lane ag-program)
    "neural/program-run/gaussian/shape=full"]
    "neural/sequential-fit/synthetic/shape=full",
    # the inference lanes (2026-10-19, lane ag-inference-board): sequential-predict is a fit lane
    "neural/sequential-predict/synthetic/shape=full",
    "neural/ops-session-step/gaussian/shape=full",
    "neural/causal-lm-generate/mojolearn-162m/shape=full",
    "neural/causal-lm-generate-smollm2/smollm2-135m/shape=full"]
#: the fit lanes race ours-fast only on Apple (identical is never timed there), and MLX beside torch
FIT_IDS = {i for i in NEURAL_IDS if "/higgs/" in i or "/sequential-fit/" in i or "/sequential-predict/" in i}
#: fit lanes with no MLX twin (bench_board_neural.PREDICT_LANES)
NO_MLX_IDS = {i for i in NEURAL_IDS if "/sequential-predict/" in i}
#: decode lanes with no torch carried-state twin race ours alone (bench_board_neural.DECODE_LANES
#: minus DECODE_TORCH_TWIN); transformer-decode's twin is a per-token loop, never compiled
NO_TWIN_DECODE = ("mamba1-decode", "mamba2-decode", "mamba3-decode", "samba-decode")
#: the *-infer lanes and lm-host-train-step run OUR CPU binding: never raced (Oct 2 2026)
GPU_ARMS = {
    "nvidia": ["torch-eager-fp32", "torch-eager-tf32", "torch-compile-fp32", "torch-compile-tf32",
               "torch-eager-bf16", "torch-compile-bf16"],
    "amd": ["torch-eager-fp32", "torch-compile-fp32", "torch-eager-bf16", "torch-compile-bf16"],
    "apple": ["torch-eager-fp32", "torch-compile-fp32", "torch-eager-bf16", "torch-compile-bf16"],
}
CPU_ARMS = ["torch-cpu-eager-fp32", "torch-cpu-compile-fp32", "torch-cpu-eager-bf16",
            "torch-cpu-compile-bf16"]


@pytest.mark.parametrize("vendor", ["apple", "nvidia", "amd"])
def test_plan_neural_fast_on_apple_only(vendor):
    races = bb.plan_races(vendor, bb.modes_for(vendor), ["neural"])
    assert [r["id"] for r in races] == NEURAL_IDS
    # the Apple FAST neural tier (Oct 3 2026): `ours-fast` beside `ours` on Apple only
    ours = ["ours", "ours-fast"] if vendor == "apple" else ["ours"]
    for r in races:
        fit = r["id"] in FIT_IDS
        if fit and vendor == "apple":
            assert r["our_arms"] == {"ours-fast": "fast"} and r["modes"] == ["fast"], r["id"]
            mlx = [] if r["id"] in NO_MLX_IDS else ["mlx-fp32"]
            assert r["opponents"] == GPU_ARMS[vendor] + mlx, r["id"]
            assert r["arms"] == ["ours-fast"] + GPU_ARMS[vendor] + mlx
            continue
        assert r["our_arms"] == ({"ours": "identical", "ours-fast": "fast"} if vendor == "apple"
                                 else {"ours": "identical"})
        assert r["modes"] == (["fast", "identical"] if vendor == "apple" else ["identical"])
        assert "ours-ab" not in r["arms"]
        cpu_lane = r["lane"].endswith("-infer") or r["lane"] == "lm-host-train-step"
        want = CPU_ARMS if cpu_lane else GPU_ARMS[vendor]
        if r["lane"] == "gemm-bf16":
            want = [a for a in want if a.endswith("bf16")]
        if r["lane"] == "gemm-int8":
            want = ["torch-eager-int8", "torch-compile-int8"] if vendor == "nvidia" else []
        if (r["lane"].startswith("mamba1-") or r["lane"] == "transformer-decode"
                or r["lane"].startswith("causal-lm-generate")):
            # the per-token reference scan / decode loop is not a compile target (NOT_PLANNED)
            want = [a for a in want if "-compile-" not in a]
        if r["lane"] in NO_TWIN_DECODE:
            want = []
        if r["lane"] in ("mamba1-forward", "mamba2-forward", "mamba3-forward"):
            # FAIRNESS FIX (2026-10-13 audit): mamba_ssm's fused kernels, the real Mamba opponent
            # (the torch references are planned and refused by name at race time)
            want = want + {"nvidia": ["mamba-ssm-fp32", "mamba-ssm-tf32"],
                           "amd": ["mamba-ssm-fp32"]}.get(vendor, [])
        assert r["opponents"] == want, r["id"]
        assert r["arms"] == ours + want
        # TF32 exists on NVIDIA CUDA only; it is never planned elsewhere
        assert any("tf32" in a for a in r["arms"]) == (vendor == "nvidia" and not cpu_lane
                                                      and not r["lane"].startswith("gemm-")
                                                      and r["lane"] not in NO_TWIN_DECODE)
    for arm in GPU_ARMS["nvidia"]:
        assert bb.arm_library(arm) == "torch" and bb.arm_device(arm, vendor) == "gpu"
    for arm in CPU_ARMS:
        assert bb.arm_library(arm) == "torch" and bb.arm_device(arm, vendor) == "cpu"
    small = bb.plan_races(vendor, ["identical"], ["neural"], ["gemm"], neural_shape="small")
    assert [r["id"] for r in small] == ["neural/gemm/gaussian/shape=small"]
    cells = sum(len(r["arms"]) for r in races)
    # +6 inference lanes (Oct 7 2026): apple 64 -> 82, amd 52 -> 64, nvidia 73 -> 88
    # FAIRNESS FIX (2026-10-13 audit): + the mamba-ssm arms on the three Mamba forward lanes
    # (NVIDIA fp32 and tf32: +6; AMD fp32: +3)
    # + the 5 fit lanes (2026-10-19): apple ours-fast + 4 torch + mlx (+30), amd ours + 4 torch (+25),
    # nvidia ours + 6 torch (+35)
    # + sequential-fit (a fit lane: apple +6, amd +5, nvidia +7) and ops-session-step (apple ours + ours-fast
    # + 4 torch +6, amd +5, nvidia +7), 2026-10-19 lane ag-board-twins
    # + program-run (the same arms as ops-session-step: apple +6, amd +5, nvidia +7), lane ag-program
    assert cells == {"apple": 130, "amd": 107, "nvidia": 150}[vendor]
    assert bb.plan_summary(races)["by_family"] == {"neural": {"races": 26, "cells": cells}}
    # + sequential-predict (a fit lane without MLX: apple ours-fast + 4 torch +5, amd +5, nvidia +7) and the two
    # causal-lm-generate lanes (eager torch only: apple ours + ours-fast + 2 +4 each, amd ours + 2 +3 each,
    # nvidia ours + 3 +4 each), 2026-10-19 lane ag-inference-board
    assert cells == {"apple": 137, "amd": 113, "nvidia": 158}[vendor]
    assert bb.plan_summary(races)["by_family"] == {"neural": {"races": 28, "cells": cells}}


def test_fast_for_neural_is_the_apple_tier(env):
    # Apple (Oct 3 2026): a FAST-only neural plan races `ours-fast` alone, GPU lanes only
    races = bb.plan_races("apple", ["fast"], ["neural"])
    assert [r["id"] for r in races] == NEURAL_IDS
    assert all(r["our_arms"] == {"ours-fast": "fast"} and r["arms"][0] == "ours-fast"
               and r["modes"] == ["fast"] for r in races)
    assert bb.main(["--dry-run", "--vendor", "apple", "--modes", "fast", "--families", "neural"]) == 0
    # FAST is still the Apple tier for trees and classical
    races = bb.plan_races("apple", ["fast"], ["trees", "classical"])
    assert races and all(set(r["our_arms"].values()) == {"fast"} for r in races)
    # NVIDIA and AMD: the neural family is IDENTICAL only, refused by name without identical
    for v in ("nvidia", "amd"):
        with pytest.raises(SystemExit, match="neural family"):
            bb.plan_races(v, ["fast"], ["neural"])
        with pytest.raises(SystemExit, match="neural family"):
            bb.check_neural_modes(["neural"], ["fast"], v)
        with pytest.raises(SystemExit):
            bb.main(["--dry-run", "--vendor", v, "--modes", "fast", "--families", "neural"])
    assert _calls(env) == []


# Oct 7 2026: the six GPU inference lanes (transformer-decode, mamba1/2/3-decode, samba-decode,
# mlp-predict) add 6 races and 18 Apple / 15 NVIDIA / 12 AMD neural cells
# FAIRNESS FIX (2026-10-13 audit): the harness-written torch arms are no opponents. NVIDIA
# kmeans and knn lose torch-gpu (-4), ols torch-gpu and torch-gpu-eigh (-4) and pca torch-gpu
# (-2): 275 -> 265; AMD ols keeps one CPU cell where two torch cells raced (-2): 234 -> 232;
# Apple keeps its count (scikit-learn races where torch-gpu did, no GPU opponent left), and so
# do AMD kmeans, knn and pca.
# FAIRNESS FIX (2026-10-13 audit, linear group): ols torch-gpu is torch.linalg.lstsq(driver='gels'),
# torch's own routine, and races again on NVIDIA (+2: 265 -> 267; on AMD it replaces the CPU cell);
# the classical2 linearsvc / linearsvr scikit-learn arms (liblinear, a different algorithm and
# model) leave Apple and AMD (-4 each: Apple 346 -> 342 and classical2 134 -> 130, AMD 232 -> 228
# and classical2 90 -> 86).
# FAIRNESS FIX (2026-10-13 audit, decomposition group): the classical2 tsvd scikit-learn arm
# (ARPACK, a different algorithm from ours' covariance_eigh) leaves Apple and AMD (-2 each:
# Apple 342 -> 340 and classical2 130 -> 128, AMD 228 -> 226 and classical2 86 -> 84).
# FAIRNESS FIX (2026-10-13 audit, neural group): the mamba-ssm arms add 6 NVIDIA and 3 AMD
# neural cells (NVIDIA 267 -> 273 and neural 88 -> 94, AMD 226 -> 229 and neural 64 -> 67).
# smoke-fixes (2026-10-14): gbdt-multiclass, -categorical, -rank-pairlogit, -rank-yetirank race CatBoost only (XGBoost and
# LightGBM are refused there by name on every device): Apple 340 -> 331, NVIDIA 273 -> 264; AMD unchanged at 229.
# smoke-fixes r13 (2026-10-14): gbdt-lossguide races `ours-gain0` beside `ours` (+2 cells per vendor), then
# gbdt-depthwise `ours-prebinned` beside `ours` (+2 more): Apple 335, NVIDIA 268, AMD 233.
# 2026-10-19 (lane ag-bench): + the five neural fit lanes: apple 5 x (ours-fast + 4 torch + mlx-fp32) = +30,
# nvidia 5 x (ours + 6 torch) = +35, amd 5 x (ours + 4 torch) = +25; +5 races everywhere
# 2026-10-19 (lanes ag-board-twins + ag-inference-board): + sequential-fit and ops-session-step (apple +12, nvidia
# +14, amd +10) and sequential-predict + the two causal-lm-generate lanes (apple +13, nvidia +15, amd +11); +5 races
@pytest.mark.parametrize("vendor,cells,more,neural", [("apple", 390, 128, 137),
                                                      ("nvidia", 332, 88, 158),
                                                      ("amd", 279, 84, 113)])
def test_dry_run_counts_per_vendor(vendor, cells, more, neural, capsys):
    assert bb.main(["--dry-run", "--vendor", vendor, "--no-cpu-arm",
                    "--families", "trees,classical,classical2,neural"]) == 0
    text = capsys.readouterr().out
    assert "TOTAL races=109 cells=%d" % cells in text
    assert "family classical2 races=44 cells=%d" % more in text
    assert "family neural     races=28 cells=%d" % neural in text
    assert "neural: ours IDENTICAL" in text
    assert ("ours-fast, the Apple tier" in text) == (vendor == "apple")
    # what is left off the plan is printed by name, never dropped silently
    assert "neural not planned: torch-compile-* on mamba1-forward" in text
    assert ("neural not planned: torch-eager-tf32" in text) == (vendor != "nvidia")


def test_neural_command_and_settings():
    race = bb.plan_races("amd", ["identical"], ["neural"], ["lm-train-step"], neural_shape="small")[0]
    ctx = {"python": "py", "neural_driver": "drv", "vendor": "amd", "rounds": 4, "out": "/o",
           "round_seconds": 0}
    cmd, env, ceiling = bb.neural_cmd(ctx, race)
    assert cmd[:4] == ["py", "-u", "drv", "race"]
    assert cmd[cmd.index("--lane") + 1] == "lm-train-step"
    assert cmd[cmd.index("--shape") + 1] == "small"
    assert cmd[cmd.index("--arms") + 1] == ",".join(["ours"] + GPU_ARMS["amd"])
    assert cmd[cmd.index("--rounds") + 1] == "4"
    assert env == {} and ceiling > 0
    s = bb.race_settings(ctx, race)
    assert s["seed"] == 7 and s["driver"] == "tools/bench_board_neural.py"
    assert "compile" in s["opponent_mode"] and "bf16" in s["opponent_mode"]
    assert s["torch_settings"]["torch-eager-fp32"] == "float32, TF32 off"
    assert "autocast" in s["torch_settings"]["torch-compile-bf16"]
    assert "identical" in s["numeric_mode"]
    assert s["optimizer"].startswith("AdamW lr 1e-3")
    assert s["shape_dims"] == "B2 L64 DM64 H4 KV2 HD16 FF128 layers2 V256 (smoke)"
    # a lane whose `ours` runs our CPU binding is never planned (Oct 2 2026)
    assert bb.plan_races("amd", ["identical"], ["neural"], ["mamba2-infer"]) == []


def _run_neural(env, *extra):
    base = list(env["base"])
    # a neural-only run needs no taxi/Istella: point --data-root at nothing
    base[base.index("--data-root") + 1] = str(env["tmp"] / "no-data-here")
    return bb.main(["--vendor", "nvidia", "--families", "neural", "--neural-shape", "small"]
                   + base + list(extra))


def test_nvidia_board_from_the_old_key_resumes_on_the_same_and_a_new_pod(env, monkeypatch):
    """A board.json written before the NVIDIA key change (its box carries the
    hostname, its races no `host`) resumes under the new key: on the same pod,
    and on a new pod with the same GPU model, driver major and pins; a
    different GPU model refuses."""
    gpu = {"vendor": "nvidia", "api": "cuda", "name": "NVIDIA A40", "driver": "580.159.04"}
    monkeypatch.setattr(bb, "gpu_info", lambda vendor: dict(gpu))
    assert _run_neural(env) == 0
    p = env["out"] / "board.json"
    res = json.loads(p.read_text())
    host = res["box"]["host"]["hostname"]
    for rec in res["races"].values():
        rec.pop("host", None)                 # what the old code wrote
    p.write_text(json.dumps(res))
    env["calls"].write_text("")
    assert _run_neural(env) == 0              # same pod: resumes, runs nothing
    assert _calls(env) == []
    assert host in (env["out"] / "BOARD.md").read_text()
    monkeypatch.setattr(bb.platform, "node", lambda: "a-new-pod")
    assert _run_neural(env) == 0              # a new pod, same GPU/driver/pins: resumes
    assert _calls(env) == []
    gpu["name"] = "NVIDIA GeForce RTX 4090"
    with pytest.raises(SystemExit, match="different box"):
        _run_neural(env)


def test_neural_run_schema_quality_and_board(env):
    assert _run_neural(env) == 0
    res = json.loads((env["out"] / "board.json").read_text())
    assert res["config"]["neural_shape"] == "small"
    assert res["config"]["smoke"] is True
    assert set(res["races"]) == {i.replace("shape=full", "shape=small") for i in NEURAL_IDS}
    assert sorted(_calls(env)) == sorted(
        "neural %s small %s" % (l, ",".join(["ours"] + bb.gpu_opponents_first(
            bb.NEURAL_OPPONENTS["nvidia"][l])))
        for l in bb.NEURAL_LANES if bb.NEURAL.DEVICE_OF.get(l) != "cpu")
    rec = res["races"]["neural/lm-train-step/bytes/shape=small"]
    assert rec["status"] == "done" and rec["shape"] == "small"
    cells = {c["arm"]: c for c in rec["cells"]}
    ours, torch = cells["ours"], cells["torch-eager-fp32"]
    assert ours["mode"] == "identical" and ours["mode_witness"] == "identical"
    assert ours["installed_wheel"] == "wheel"
    assert torch["mode"] == "opponent" and torch["library"] == "torch"
    assert ours["status"] == torch["status"] == "ok"
    assert ours["settings"]["seed"] == 7 and ours["neural_shape"] == "small"
    # stub: ours 81, torch 41 (medians of 3 rounds)
    assert torch["ratio_ours_identical_over"] == pytest.approx(81 / 41)
    assert torch["ratio_ours_fast_over"] is None
    assert torch["quality"]["loss_last_abs_diff_vs_ours"] == pytest.approx(1e-4)
    assert torch["verdict"] == "LIKE-FOR-LIKE-SPAN"
    assert ours["shape"] == "stub-small"
    board = (env["out"] / "BOARD.md").read_text()
    assert "## Neural" in board and "neural shape small" in board
    assert "--neural-shape small" in board and "SMOKE RUN" in board
    assert "loss_last_step" in board and "torch-eager-fp32" in board
    for arm in GPU_ARMS["nvidia"]:
        assert "| %s | torch |" % arm in board, arm
    # our CPU is never raced: no *-infer race, and no torch CPU arm, on the board
    assert not any(l.endswith("-infer") for l in (r["lane"] for r in res["races"].values()))
    for arm in CPU_ARMS:
        assert "| %s | torch |" % arm not in board, arm
    assert "## Trees" not in board
    # the neural Not covered lines come from the driver's tables
    assert "not mamba-ssm's fused CUDA/Triton kernels" in board
    assert not BANNED.search(board)
    # resume: nothing left to run
    env["calls"].write_text("")
    assert _run_neural(env) == 0
    assert _calls(env) == []


def test_neural_refused_arm_and_failed_race(env, monkeypatch):
    monkeypatch.setenv("STUB_REFUSE", "torch-eager-fp32")
    monkeypatch.setenv("STUB_FAIL", "gemm")
    assert _run_neural(env) == 1
    res = json.loads((env["out"] / "board.json").read_text())
    assert res["races"]["neural/gemm/gaussian/shape=small"]["status"] == "failed"
    gemm = res["races"]["neural/gemm/gaussian/shape=small"]["cells"]
    assert all(c["status"].startswith("UNKNOWN(no race json") for c in gemm)
    fwd = {c["arm"]: c for c in res["races"]["neural/lm-forward/bytes/shape=small"]["cells"]}
    assert fwd["torch-eager-fp32"]["status"].startswith("REFUSED(")
    assert fwd["torch-eager-fp32"]["ratio_ours_identical_over"] is None
    board = (env["out"] / "BOARD.md").read_text()
    assert "REFUSED" in board and not BANNED.search(board)
    # the retry reruns only the failed lane
    monkeypatch.delenv("STUB_FAIL")
    env["calls"].write_text("")
    assert _run_neural(env) == 0
    assert _calls(env) == ["neural gemm small %s" % ",".join(["ours"] + GPU_ARMS["nvidia"])]


def test_neural_shape_change_is_a_new_race_not_a_resume(env):
    assert _run_neural(env, "--lanes", "gemm") == 0
    env["calls"].write_text("")
    base = list(env["base"])
    base[base.index("--data-root") + 1] = str(env["tmp"] / "no-data-here")
    assert bb.main(["--vendor", "nvidia", "--families", "neural", "--lanes", "gemm"] + base) == 0
    assert _calls(env) == ["neural gemm full %s" % ",".join(["ours"] + GPU_ARMS["nvidia"])]
    res = json.loads((env["out"] / "board.json").read_text())
    assert "neural/gemm/gaussian/shape=full" in res["races"]
    assert res["config"]["smoke"] is False


_nspec = importlib.util.spec_from_file_location("bench_board_neural_t",
                                                os.path.join(HERE, "bench_board_neural.py"))
bbn = importlib.util.module_from_spec(_nspec)
_nspec.loader.exec_module(bbn)


def test_neural_driver_tables_and_arm_refusals(tmp_path):
    assert set(bbn.LANES) == set(bbn.MODEL_OF) == set(bbn.LANE_TEXT) == set(bbn.DATA_OF)
    assert bbn.arm_setting("torch-cpu-eager-bf16") == ("cpu", "eager-bf16")
    assert bbn.arm_setting("torch-compile-tf32") == ("gpu", "compile-tf32")
    for arm in [a for a in bbn.ARMS if a.startswith("torch-")]:
        assert bbn.precision_text(bbn.arm_setting(arm)[1])
    with pytest.raises(ValueError):
        bbn.arm_setting("ours")
    # a CPU arm on a GPU lane (and the reverse) is refused by name before torch loads
    np = pytest.importorskip("numpy")
    for lane, arm in (("gemm", "torch-cpu-eager-fp32"), ("mlp-infer", "torch-eager-fp32")):
        with pytest.raises(RuntimeError, match="REFUSED: arm %s" % arm):
            bbn.build_runner(lane, arm, "small", {})
    # an arm the driver does not know is refused by the race before any worker starts
    with pytest.raises(SystemExit, match="no arm 'torch-eager-fp64'"):
        bbn.main(["race", "--lane", "gemm", "--arms", "ours,torch-eager-fp64", "--out",
                  str(tmp_path / "o"), "--work", str(tmp_path / "w"), "--rounds", "1"])


def test_neural_inference_lanes_tables_and_quality():
    # 2026-10-19 (lane ag-inference-board): the generate lanes' checkpoint is the dataset, eager torch only;
    # sequential-predict is a fit lane whose clock is the prediction alone; the generate quality names the
    # first new-token position where an arm leaves ours
    np = pytest.importorskip("numpy")
    assert bbn.DATA_OF["causal-lm-generate"] == "mojolearn-162m"
    assert bbn.DATA_OF["causal-lm-generate-smollm2"] == "smollm2-135m"
    assert bbn.DATA_OF["sequential-predict"] == "synthetic" and "sequential-predict" in bbn.FIT_LANES
    assert bbn.fit_span("sequential-predict")["pre_clock_fit"] is True
    assert bbn.opponents("nvidia", "causal-lm-generate") == ("torch-eager-fp32", "torch-eager-tf32",
                                                             "torch-eager-bf16")
    assert bbn.opponents("apple", "sequential-predict") == tuple("torch-" + s for s in bbn.GPU_SETTINGS["apple"])
    assert bbn.gen_params("causal-lm-generate")["new_tokens"] == bbn.GEN_SHAPES["full"]["new_tokens"]
    prompt = np.arange(6, dtype=np.int32).reshape(2, 3)
    ours = {"ids": np.array([[0, 1, 2, 9, 9, 9], [3, 4, 5, 7, 7, 7]], np.int32),
            "first": np.array([[0, 1, 2, 9], [3, 4, 5, 7]], np.int32)}
    theirs = {"ids": np.array([[0, 1, 2, 9, 9, 8], [3, 4, 5, 7, 6, 7]], np.int32), "first": ours["first"].copy()}
    q = bbn.gen_quality({"prompt_ids": prompt}, {"ours": ours, "torch-eager-fp32": theirs})
    row = q["torch-eager-fp32"]
    assert row["tokens_equal_ours"] is False and row["first_divergence_index"] == 1
    assert row["matching_prefix_tokens"] == 1 and row["prefill_token_equal_ours"] is True
    assert q["ours"]["new_tokens"] == 3 and q["ours"]["prompt_echoed"] is True
    same = bbn.gen_quality({"prompt_ids": prompt}, {"ours": ours, "torch-eager-bf16": ours})
    assert same["torch-eager-bf16"]["tokens_equal_ours"] is True
    assert same["torch-eager-bf16"]["first_divergence_index"] is None
    a = {"round_extra": [{"prefill_ms": 9.0, "generate_ms": 99.0, "warmup": True},
                         {"prefill_ms": 2.0, "generate_ms": 20.0, "decode_ms_per_token": 1.5, "new_tokens": 13,
                          "warmup": False}],
         "device_memory": [{"peak_bytes": {"process": 10, "device": None}, "resident_bytes": {"process": 6},
                            "process_peak_bytes": 12, "method": "m"}]}
    got = bbn.inference_summary(a)
    assert got["prefill_ms"] == 2.0 and got["generate_ms"] == 20.0 and got["decode_ms_per_token"] == 1.5
    assert got["peak_bytes_process"] == 10 and got["resident_bytes_process"] == 6 and got["process_peak_bytes"] == 12


def test_neural_tape_api_lanes_tables_fixtures_and_quality(tmp_path):
    # 2026-10-19 (lane ag-board-twins): sequential-fit is a fit lane on its own synthetic data, its
    # predict_proba inside the clock; ops-session-step is one recorded step with its own span
    np = pytest.importorskip("numpy")
    assert "sequential-fit" in bbn.FIT_LANES and bbn.DATA_OF["sequential-fit"] == "synthetic"
    assert bbn.DATA_OF["ops-session-step"] == "gaussian" and "ops-session-step" in bbn.OPS_LANES
    assert bbn.fit_span("sequential-fit")["inside_clock"].startswith("one whole fit and the held-out")
    assert bbn.fit_params("sequential-fit")["layers"] == [
        "linear 32 128 bias", "relu", "linear 128 64 bias", "relu", "linear 64 1 bias"]
    assert bbn.ops_params("small")["layers"][0] == "linear 16 32 bias"
    for lane in ("sequential-fit", "ops-session-step"):
        rec = bbn.make_inputs(lane, "small", 2, str(tmp_path / ("%s.npz" % lane)))
        assert rec["lane"] == lane
    with np.load(str(tmp_path / "sequential-fit.npz")) as z:
        assert z["X_train"].shape == (4096, 32) and z["X_test"].shape == (1024, 32)
        assert 0.2 < float(z["y_train"].mean()) < 0.8
    with np.load(str(tmp_path / "ops-session-step.npz")) as z:
        data = {k: z[k] for k in z.files}
    assert data["X"].shape == (64, 16) and data["y_codes"].shape == (64, 1)
    assert (data["y_codes"][:, 0].astype(np.int64) == data["y"]).all()
    # the quality: every arm's gradient digests, every other arm's differences from ours
    grads = [np.ones_like(data["w:" + n]) for n in bbn.OPS_PARAMS]
    ours = bbn._ops_outputs(np, 1.5, grads, [data["w:" + n] for n in bbn.OPS_PARAMS])
    theirs = bbn._ops_outputs(np, 1.25, [g * 2 for g in grads], [data["w:" + n] for n in bbn.OPS_PARAMS])
    q = bbn.quality("ops-session-step", data, {"ours": ours, "torch-eager-fp32": theirs})
    assert sorted(q["ours"]["grad_digests"]) == sorted(bbn.OPS_PARAMS) and q["ours"]["loss"] == 1.5
    t = q["torch-eager-fp32"]
    assert t["loss_abs_diff_vs_ours"] == 0.25 and t["grad_max_abs_diff_vs_ours_w1"] == 1.0
    assert t["param_max_abs_diff_vs_ours_b2"] == 0.0 and t["grad_digests_equal_ours"] is False


def test_neural_program_run_lane_tables_fixtures_and_quality(tmp_path):
    # 2026-10-19 (lane ag-program): program-run is one inference forward of a captured Program, with
    # ops-session-step's weights and batch and its own span
    np = pytest.importorskip("numpy")
    assert "program-run" in bbn.PROGRAM_LANES and bbn.DATA_OF["program-run"] == "gaussian"
    assert bbn.DEVICE_OF["program-run"] == "gpu" and bbn.MODEL_OF["program-run"] == "program"
    assert bbn.program_params("small")["layers"] == ["linear 16 32 bias", "relu", "linear 32 4 bias", "softmax"]
    assert bbn.lane_settings("program-run")["clock"] == bbn.PROGRAM_SPAN["inside_clock"]
    rec = bbn.make_inputs("program-run", "small", 2, str(tmp_path / "program-run.npz"))
    assert rec["lane"] == "program-run"
    with np.load(str(tmp_path / "program-run.npz")) as z:
        data = {k: z[k] for k in z.files}
    assert data["X"].shape == (64, 16) and data["w:w2"].shape == (32, 4)
    x = data["X"].astype(np.float64)
    z = np.maximum(x @ data["w:w1"] + data["w:b1"], 0) @ data["w:w2"] + data["w:b2"]
    p = np.exp(z - z.max(axis=1, keepdims=True))
    p = (p / p.sum(axis=1, keepdims=True)).astype(np.float32)
    q = bbn.quality("program-run", data, {"ours": {"y": p}, "torch-eager-fp32": {"y": p + np.float32(0.5)}})
    assert q["ours"]["max_abs_err_vs_fp64"] < 1e-6 and q["ours"]["finite"]
    t = q["torch-eager-fp32"]
    assert abs(t["max_abs_diff_vs_ours"] - 0.5) < 1e-6 and t["digest_equal_ours"] is False


def test_neural_driver_inputs_and_quality(tmp_path, monkeypatch):
    # This unit checks inter-arm quality math; actual saved-output host checks
    # have independent pass/failure coverage in test_neural_forward_quality.py.
    monkeypatch.setattr(bbn, "forward_host_quality", lambda *a: {"ours": {"host_reference_passed": True}})
    np = pytest.importorskip("numpy")
    path = str(tmp_path / "in.npz")
    rec = bbn.make_inputs("mlp-train-step", "small", 3, path)
    with np.load(path) as z:
        assert z["X"].shape == (3, 32, 8) and z["y"].shape == (3, 32)
        assert z["w:weight1"].shape == (16, 8) and z["X"].dtype == np.float32
    assert rec["seed"] == 7 and rec["shape"]["label"] == "rows32 8-16-3"
    rec2 = bbn.make_inputs("mlp-train-step", "small", 3, str(tmp_path / "again.npz"))
    assert rec2["sha256"] == rec["sha256"]                 # same seed, same bytes
    rec = bbn.make_inputs("transformer-infer", "small", 2, path)
    with np.load(path) as z:
        assert z["x"].shape == (2, 64, 64)
        assert set(k[2:] for k in z.files if k.startswith("w:")) == set(bbn.TRANSFORMER_NAMES)
    reg = bbn.samba_registry(bbn.SAMBA_SHAPES["small"])
    assert reg[0] == ("embed.weight", (256, 64)) and reg[-1] == ("norm_f.weight", (64,))
    assert [n for n, _ in reg[1:10]] == ["layers.0." + n for n in bbn.MAMBA_NAMES["mamba3"]]
    # train lanes: losses side by side, the difference against ours
    q = bbn.quality("samba-train-step", {}, {
        "ours": {"losses": np.array([5.5, 5.4])},
        "torch-eager-bf16": {"losses": np.array([5.501, 5.398])}})
    assert q["torch-eager-bf16"]["loss_first_abs_diff_vs_ours"] == pytest.approx(1e-3)
    assert q["torch-eager-bf16"]["loss_last_abs_diff_vs_ours"] == pytest.approx(2e-3)
    assert "loss_first_abs_diff_vs_ours" not in q["ours"]
    # forward lanes: max abs and relative difference against ours
    y = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    q = bbn.quality("mamba2-forward", {}, {"ours": {"y": y}, "torch-eager-fp32": {"y": y + 0.5},
                                          "torch-compile-bf16": {"y": y[:1]}})
    assert q["torch-eager-fp32"] == {"max_abs_diff_vs_ours": 0.5, "max_rel_diff_vs_ours": 0.5 / 11}
    assert "shape_mismatch_vs_ours" in q["torch-compile-bf16"]
    # every arm is also measured against the eager FP32 torch output (scorer gaps, 2026-10-15)
    assert q["ours"]["host_reference_passed"] is True
    assert q["ours"]["rel_fro_vs_torch_eager_fp32"] == pytest.approx(
        np.linalg.norm(np.full(12, .5)) / np.linalg.norm(np.arange(12) + .5))
    assert "shape_mismatch_vs_torch_eager_fp32" in q["torch-compile-bf16"]


# --- the classical2 family ---------------------------------------------------

def test_plan_classical2_per_vendor():
    ap = {r["id"]: r for r in bb.plan_races("apple", bb.modes_for("apple"), ["classical2"])}
    umap = ap["classical2/umap/taxi/rows=full"]
    assert umap["our_arms"] == {"ours": "identical", "ours-fast": "fast"}
    assert umap["opponents"] == ["umap-learn-cpu", "umap-learn-cpu-unseeded"]
    assert ap["classical2/ivf/istella/rows=full"]["opponents"] == ["faiss-cpu"]
    # the time series lanes run once, on their own synthetic data
    assert [i for i in ap if "/arima/" in i or "/ets/" in i] == [
        "classical2/arima/synthetic/rows=full", "classical2/ets/synthetic/rows=full"]
    assert ap["classical2/arima/synthetic/rows=full"]["opponents"] == ["statsmodels-cpu"]
    nv = {r["id"]: r for r in bb.plan_races("nvidia", ["identical"], ["classical2"])}
    assert nv["classical2/umap/taxi/rows=full"]["arms"] == ["ours", "cuml-gpu"]
    assert nv["classical2/ivf/taxi/rows=full"]["opponents"] == ["cuvs-gpu"]
    assert nv["classical2/gmm/taxi/rows=full"]["opponents"] == ["sklearn-cpu"]
    assert nv["classical2/ets/synthetic/rows=full"]["opponents"] == ["cuml-gpu"]
    for r in nv.values():
        assert r["our_arms"] == {"ours": "identical"}
    amd = bb.plan_races("amd", ["identical"], ["classical2"])
    assert all("cuml-gpu" not in r["arms"] and "cuvs-gpu" not in r["arms"] for r in amd)
    # every vendor names what it does not plan
    for v in bb.VENDORS:
        assert bb.MORE.NOT_PLANNED[v]
        assert set(bb.MORE.OPPONENTS[v]) == set(bb.MORE_LANES)
    only_taxi = bb.plan_races("apple", ["fast", "identical"], ["classical2"], ["gmm", "ets"], ["taxi"])
    assert [r["id"] for r in only_taxi] == ["classical2/gmm/taxi/rows=full",
                                            "classical2/ets/synthetic/rows=full"]
    fast_only = bb.plan_races("apple", ["fast"], ["classical2"], ["gmm"], ["taxi"])
    assert fast_only[0]["our_arms"] == {"ours-fast": "fast"}
    assert bb.arm_library("umap-learn-cpu-unseeded") == "umap-learn"
    assert bb.arm_device("umap-learn-cpu-unseeded", "apple") == "cpu"
    assert bb.arm_library("cuvs-gpu") == "cuvs" and bb.arm_device("cuvs-gpu", "nvidia") == "gpu"


def test_classical2_identical_only_lane_has_no_fast_arm(monkeypatch):
    monkeypatch.setattr(bb.MORE, "has_fast", lambda lane: lane != "gmm")
    r = bb.plan_races("apple", ["fast", "identical"], ["classical2"], ["gmm"], ["taxi"])[0]
    assert r["our_arms"] == {"ours": "identical"} and "ours-fast" not in r["arms"]
    assert bb.plan_races("apple", ["fast"], ["classical2"], ["gmm"], ["taxi"]) == []


def test_classical2_command_and_settings():
    race = bb.plan_races("amd", ["identical"], ["classical2"], ["umap"], ["taxi"], 2000)[0]
    ctx = {"python": "py", "more_driver": "drv", "vendor": "amd", "rounds": 2, "out": "/o",
           "round_seconds": 0, "more_data": "/c/more"}
    cmd, env, ceiling = bb.more_cmd(ctx, race)
    assert cmd[:4] == ["py", "-u", "drv", "race"]
    assert cmd[cmd.index("--data") + 1] == "/c/more/rows-2000"
    assert cmd[cmd.index("--arms") + 1] == "ours,umap-learn-cpu,umap-learn-cpu-unseeded"
    assert env == {} and ceiling > 0
    s = bb.race_settings(ctx, race)
    assert s["seed"] == 7 and s["driver"] == "tools/bench_board_more.py"
    assert "random_state=7" in s["lane_config"]["params"]
    assert any("one thread" in m for m in s["lane_config"]["mismatches"])


def test_noncompiled_classical_cell_keeps_actual_fit_receipts(monkeypatch):
    monkeypatch.setattr(bb, "base_cell", lambda *a: {"device":"gpu", "arm":"cuml-gpu"})
    iterations=[dict(status="recorded",values=[17])]
    fitted=[dict(params={"max_iter":1000})]
    race=dict(arms=["cuml-gpu"],our_arms={})
    raw=dict(arms={"cuml-gpu":dict(ms=[2],status="ok",iteration_receipts=iterations,
                                 fitted_parameters=fitted)})
    cell=bb.classical_cells(dict(rounds=1),race,raw)[0]
    assert cell["iteration_receipts"] == iterations
    assert cell["fitted_parameters"] == fitted


def test_global_quality_failure_survives_cell_extraction(monkeypatch):
    monkeypatch.setattr(bb, "base_cell", lambda *a: {"device":"gpu", "arm":"cuml-gpu"})
    race=dict(arms=["cuml-gpu"],our_arms={})
    for quality in ({"error":"scorer failed"}, {"status":"pending_reference"}):
        raw=dict(arms={"cuml-gpu":dict(ms=[2],status="ok")},quality=quality)
        cell=bb.classical_cells(dict(rounds=1),race,raw)[0]
        assert cell["quality"]["error"]


def _run_more(env, *extra):
    return bb.main(["--vendor", "apple", "--families", "classical2", "--lanes", "umap,arima",
                    "--datasets", "taxi"] + env["base"] + list(extra))


def test_classical2_run_schema_quality_and_board(env):
    assert _run_more(env) == 0
    res = json.loads((env["out"] / "board.json").read_text())
    assert set(res["races"]) == {"classical2/umap/taxi/rows=1000",
                                 "classical2/arima/synthetic/rows=1000"}
    calls = _calls(env)
    assert calls[0].startswith("classical2 prep arima,umap taxi")
    assert sum(1 for c in calls if " prep " in c) == 1
    rec = res["races"]["classical2/umap/taxi/rows=1000"]
    assert rec["status"] == "done" and rec["lane_config"]["timed"]
    cells = {c["arm"]: c for c in rec["cells"]}
    assert set(cells) == {"ours", "ours-fast", "umap-learn-cpu", "umap-learn-cpu-unseeded"}
    assert cells["ours-fast"]["mode"] == "fast" and cells["ours-fast"]["mode_witness"] == "fast"
    assert cells["ours"]["installed_wheel"] == "wheel"
    assert cells["umap-learn-cpu"]["library"] == "umap-learn"
    # medians of 3 rounds: ours 31, ours-fast 61, umap-learn-cpu 91
    assert cells["umap-learn-cpu"]["ratio_ours_identical_over"] == pytest.approx(31 / 91)
    assert cells["umap-learn-cpu"]["ratio_ours_fast_over"] == pytest.approx(61 / 91)
    assert cells["ours-fast"]["ratio_ours_identical_over"] is None
    assert cells["ours"]["settings"]["lane_config"]["quality"].startswith("trustworthiness")
    assert (env["out"] / "cache" / "more-data" / "rows-1000" / "arma-synthetic.json").exists()
    board = (env["out"] / "BOARD.md").read_text()
    assert "## Classical, wave 2" in board and "trustworthiness_k15" in board
    assert "mismatch: umap-learn-cpu: random_state=7" in board
    assert "not planned on this vendor: cuML and cuVS" in board
    assert "SMOKE RUN" in board and not BANNED.search(board)
    env["calls"].write_text("")
    assert _run_more(env) == 0
    assert _calls(env) == []


def test_classical2_refused_arm_and_failed_race(env, monkeypatch):
    monkeypatch.setenv("STUB_REFUSE", "umap-learn-cpu-unseeded")
    monkeypatch.setenv("STUB_FAIL", "arima")
    assert _run_more(env) == 1
    res = json.loads((env["out"] / "board.json").read_text())
    assert res["races"]["classical2/arima/synthetic/rows=1000"]["status"] == "failed"
    um = {c["arm"]: c for c in res["races"]["classical2/umap/taxi/rows=1000"]["cells"]}
    assert um["umap-learn-cpu-unseeded"]["status"].startswith("REFUSED(")
    board = (env["out"] / "BOARD.md").read_text()
    assert "REFUSED" in board and not BANNED.search(board)
    monkeypatch.delenv("STUB_FAIL")
    env["calls"].write_text("")
    assert _run_more(env) == 0
    assert _calls(env) == ["classical2 race arima synthetic"]


def test_classical2_pins_installed_per_vendor(tmp_path, monkeypatch):
    cmds = []
    monkeypatch.setattr(bb, "run_logged", lambda cmd, *a, **k: cmds.append(list(cmd)) or 0)
    wheel = tmp_path / "mojolearn-0.8.22-py3-none-any.whl"
    wheel.write_bytes(b"w")
    for vendor, fams in (("apple", "classical2"), ("nvidia", "classical2"), ("amd", "trees")):
        cmds.clear()
        args = bb.build_parser().parse_args(["--python-env", "py", "--mojolearn-wheel", str(wheel),
                                             "--families", fams, "--torch-spec", ""])
        bb.setup_python(args, vendor, str(tmp_path), str(tmp_path / "log"))
        flat = [" ".join(c) for c in cmds]
        if fams == "classical2":
            assert any(" ".join(bb.MORE_PINS[vendor]) in c for c in flat), (vendor, flat)
        else:
            assert not any("statsmodels" in c for c in flat)
    assert "umap-learn==0.5.12" in bb.MORE_PINS["apple"] and "faiss-cpu==1.15.1" in bb.MORE_PINS["amd"]
    assert all("==" in p for v in bb.MORE_PINS.values() for p in v)


def test_nvidia_installs_the_cu129_torch_first_from_its_extra_index(tmp_path, monkeypatch):
    cmds = []
    monkeypatch.setattr(bb, "run_logged", lambda cmd, *a, **k: cmds.append(list(cmd)) or 0)
    wheel = tmp_path / "mojolearn-0.8.25-py3-none-any.whl"
    wheel.write_bytes(b"w")
    args = bb.build_parser().parse_args(["--python-env", "py", "--mojolearn-wheel", str(wheel),
                                         "--families", "trees,algos,neural"])
    bb.setup_python(args, "nvidia", str(tmp_path), str(tmp_path / "log"))
    flat = [" ".join(c) for c in cmds]
    t = next(i for i, c in enumerate(flat) if "torch==2.13.0+cu129" in c)
    assert "--extra-index-url https://download.pytorch.org/whl/cu129" in flat[t]
    assert "--index-url" not in flat[t].replace("--extra-index-url", "")
    # right after our wheel, before every opponent set (gpytorch, torch-geometric, ...)
    assert str(wheel) in flat[t - 1]
    assert all("torch==2.13.0+cu129" not in c for c in flat[t + 1:])
    assert any("gpytorch" in c for c in flat[t + 1:])


def test_skip_install_reuses_only_a_prepared_arm_venv(tmp_path, monkeypatch):
    # smoke-fixes (2026-10-14; algos/als/taxi-zones): the campaign passes --python-env and --skip-install
    cmds = []
    monkeypatch.setattr(bb, "run_logged", lambda cmd, *a, **k: cmds.append(list(cmd)) or 0)
    cache = tmp_path / "c"
    args = bb.build_parser().parse_args(["--python-env", "py", "--skip-install", "--families", "algos",
                                         "--cache", str(cache)])
    assert bb.setup_arm_venvs(args, "nvidia", str(tmp_path), str(tmp_path / "log")) == {}
    assert cmds == []
    got, problems = bb.prepare_arm_venvs(str(cache), "py", "nvidia", {"implicit-gpu"}, str(tmp_path / "log"))
    assert problems == {} and set(got) == {"implicit-gpu"}
    py = cache / "venv-implicit-gpu" / "bin" / "python"
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("")
    n = len(cmds)
    assert bb.setup_arm_venvs(args, "nvidia", str(tmp_path), str(tmp_path / "log")) == {"implicit-gpu": str(py)}
    assert bb.prepare_arm_venvs(str(cache), "py", "nvidia", {"implicit-gpu"}, str(tmp_path / "log"))[0] \
        == {"implicit-gpu": str(py)}
    assert len(cmds) == n          # prepared once: nothing reinstalled
    (cache / "venv-implicit-gpu" / bb.ARM_VENV_MARKER).write_text("implicit==0.7.2\n")
    assert bb.setup_arm_venvs(args, "nvidia", str(tmp_path), str(tmp_path / "log")) == {}


def test_implicit_gpu_gets_its_own_venv_and_the_algos_race_passes_it(tmp_path, monkeypatch):
    cmds = []
    monkeypatch.setattr(bb, "run_logged", lambda cmd, *a, **k: cmds.append(list(cmd)) or 0)
    args = bb.build_parser().parse_args(["--families", "algos", "--cache", str(tmp_path / "c")])
    got = bb.setup_arm_venvs(args, "nvidia", str(tmp_path), str(tmp_path / "log"))
    assert set(got) == {"implicit-gpu"} and got["implicit-gpu"].endswith("venv-implicit-gpu/bin/python")
    flat = [" ".join(c) for c in cmds]
    assert any("-m venv" in c and "venv-implicit-gpu" in c for c in flat)
    assert any("implicit==0.7.3" in c and "rmm-cu13==26.4.0" in c and "cuda-toolkit" in c for c in flat)
    assert bb.setup_arm_venvs(args, "apple", str(tmp_path), str(tmp_path / "log")) == {}
    ctx = {"round_seconds": 60, "python": "py", "algos_driver": "d", "rounds": 1, "out": str(tmp_path),
           "algos_data": str(tmp_path), "arm_python": got}
    race = {"lane": "als", "dataset": "taxi-zones", "rows": None,
            "arms": ["ours", "implicit-cpu", "implicit-gpu"]}
    cmd, _, _ = bb.algos_cmd(ctx, race)
    i = cmd.index("--arm-python")
    assert cmd[i + 1] == "implicit-gpu=" + got["implicit-gpu"]
    race["arms"] = ["ours", "implicit-cpu"]
    assert "--arm-python" not in bb.algos_cmd(ctx, race)[0]


def test_nvidia_refuses_system_site_packages():
    with pytest.raises(SystemExit, match="system-site-packages"):
        bb.main(["--vendor", "nvidia", "--system-site-packages", "--dry-run"])


# --- the opponent store (tools/bench_board_store.py) -------------------------

READBACK = {"max_depth": 8, "n_estimators": 500, "seed": 7}


def _store_ctx(tmp_path):
    data_root = tmp_path / "data"
    (data_root / "taxi").mkdir(parents=True)
    (data_root / bb.DATA_FILES["taxi"]).write_bytes(b"taxi bytes")
    box = {"host": {"hostname": "m3ultra-b", "cpu_model": "Apple M3 Ultra", "machine": "arm64"},
           "os": {"platform": "macOS-26-arm64"},
           "gpu": {"vendor": "apple", "name": "Apple M3 Ultra"},
           "mojolearn": {"version": "0.8.22", "wheel": {"sha256": "ab" * 32}}}
    ctx = {"vendor": "apple", "rounds": 1, "data_root": str(data_root), "box": box,
           "store_path": str(tmp_path / "opponent-store.jsonl"), "retime": False,
           "python": "py", "tree_driver": "drv", "arm_budget_s": 60, "race_deadline_s": 600,
           "infer": False, "commit": "c0ffee", "out": str(tmp_path / "out"), "nice": 0}
    race = bb.plan_races("apple", ["identical"], ["trees"], ["gbdt-symmetric"], ["taxi"],
                         None)[0]
    return ctx, race


def _key(ctx, race, arm, params=READBACK, version="1.2.8", device_name=None, over=None):
    key = bb.opponent_key(ctx["box"], race, arm, bb.race_settings(ctx, race),
                          bb.race_data_sha(ctx, race, ctx["box"]), ctx["rounds"],
                          params=params, version=version, device_name=device_name)
    key.update(over or {})
    return key


def _stored_record(ctx, race, arm, status="ok", over=None):
    cell = dict(bb.base_cell(ctx, race, arm, None), status=status, median_ms=1000.0,
                times_ms=[1000.0], rounds=1)
    return bb.STORE.record(_key(ctx, race, arm, over=over), cell,
                           measured_at="2026-09-29T12:00:00Z", commit="c0ffee")


def _probe_returns(monkeypatch, result):
    calls = []

    def fake(ctx, race, arms):
        calls.append(list(arms))
        return {a: dict(result) for a in arms}
    monkeypatch.setattr(bb, "params_probe", fake)
    return calls


def test_store_hit_constructs_then_skips_the_opponent_and_runs_ours_only(tmp_path, monkeypatch):
    ctx, race = _store_ctx(tmp_path)
    assert race["opponents"] == ["catboost-cpu"]
    bb.STORE.append(ctx["store_path"], _stored_record(ctx, race, "catboost-cpu"))
    probes = _probe_returns(monkeypatch, {"params": READBACK, "version": "1.2.8"})
    seen = {}

    def fake(ctx_, race_):
        seen["race"] = race_
        cells = [dict(bb.base_cell(ctx_, race_, "ours", "identical"), status="ok",
                      median_ms=500.0, times_ms=[500.0], rounds=1)]
        return {"cells": cells, "status": "done", "rc": 0, "finished": "now"}
    monkeypatch.setattr(bb, "_run_race", fake)
    rec = bb.run_race(ctx, race)
    assert probes == [["catboost-cpu"]]          # constructed and read back first
    assert seen["race"]["opponents"] == [] and "catboost-cpu" not in seen["race"]["arms"]
    cmd, _env = bb.tree_cmd(ctx, seen["race"])
    assert "--ours-only" in cmd and "--arms" not in cmd
    cat = [c for c in rec["cells"] if c["arm"] == "catboost-cpu"][0]
    assert cat["source"].startswith("stored (measured 2026-09-29T12:00:00Z on m3ultra-b")
    assert cat["ratio_ours_identical_over"] == 0.5
    assert rec["stored_arms"] == ["catboost-cpu"] and rec["stored_now"] == 0


def test_default_skips_unstored_opponents_with_opponents_races_them(tmp_path, monkeypatch):
    ctx, race = _store_ctx(tmp_path)
    _probe_returns(monkeypatch, {"params": READBACK, "version": "1.2.8"})
    seen = []

    def fake(ctx_, race_):
        seen.append(race_)
        cells = [dict(bb.base_cell(ctx_, race_, a, race_["our_arms"].get(a)), status="ok",
                      median_ms=500.0, times_ms=[500.0], rounds=1) for a in race_["arms"]]
        return {"cells": cells, "status": "done", "rc": 0, "finished": "now"}
    monkeypatch.setattr(bb, "_run_race", fake)
    rec = bb.run_race(ctx, race)                     # empty store, no flag: ours only
    assert seen[-1]["opponents"] == [] and "catboost-cpu" not in seen[-1]["arms"]
    assert rec["skipped_opponents"] == ["catboost-cpu"] and "catboost-cpu" not in rec["arms"]
    rec = bb.run_race(dict(ctx, with_opponents=True), race)
    assert seen[-1]["opponents"] == ["catboost-cpu"] and rec["skipped_opponents"] == []


def test_store_reuses_only_the_same_read_back_version_and_device(tmp_path, monkeypatch):
    ctx, race = _store_ctx(tmp_path)
    bb.STORE.append(ctx["store_path"], _stored_record(ctx, race, "catboost-cpu"))
    _probe_returns(monkeypatch, {"params": dict(READBACK, max_depth=6), "version": "1.2.8"})
    assert bb.stored_opponents(ctx, race) == {}          # a parameter changed
    _probe_returns(monkeypatch, {"params": READBACK, "version": "1.2.9"})
    assert bb.stored_opponents(ctx, race) == {}          # a new library version
    _probe_returns(monkeypatch, {"params": READBACK, "version": "1.2.8"})
    assert set(bb.stored_opponents(ctx, race)) == {"catboost-cpu"}
    # an arm that did not construct runs normally
    monkeypatch.setattr(bb, "params_probe", lambda c, r, a: {})
    assert bb.stored_opponents(ctx, race) == {}
    # nothing stored that could match: nothing is constructed
    calls = _probe_returns(monkeypatch, {"params": READBACK, "version": "1.2.8"})
    assert bb.stored_opponents(dict(ctx, store_path=str(tmp_path / "empty.jsonl")), race) == {}
    assert calls == []


def test_store_miss_when_any_key_field_differs(tmp_path):
    ctx, race = _store_ctx(tmp_path)
    base = _key(ctx, race, "catboost-cpu")
    for field in bb.STORE.KEY_FIELDS:
        rec = _stored_record(ctx, race, "catboost-cpu", over={field: "something else"})
        store = {bb.STORE.key_id(rec["key"]): rec}
        assert bb.STORE.lookup(store, base) is None, field
    store = {bb.STORE.key_id(base): _stored_record(ctx, race, "catboost-cpu")}
    assert bb.STORE.lookup(store, base) is not None
    # a key with a missing field neither hits nor is stored
    assert bb.STORE.lookup(store, dict(base, library_version=None)) is None
    assert "params_sha256" in bb.STORE.missing(_key(ctx, race, "catboost-cpu", params=None))


def test_store_retime_partial_and_measured_cells(tmp_path, monkeypatch):
    ctx, race = _store_ctx(tmp_path)
    bb.STORE.append(ctx["store_path"], _stored_record(ctx, race, "catboost-cpu"))
    _probe_returns(monkeypatch, {"params": READBACK, "version": "1.2.8"})
    assert bb.stored_opponents(dict(ctx, retime=True), race) == {}
    part = tmp_path / "partial.jsonl"
    bb.STORE.append(str(part), _stored_record(ctx, race, "catboost-cpu", status="PARTIAL(0/1 rounds)"))
    assert bb.stored_opponents(dict(ctx, store_path=str(part)), race) == {}

    def fake(ctx_, race_):
        cells = [dict(bb.base_cell(ctx_, race_, a, race_["our_arms"].get(a)), status="ok",
                      median_ms=700.0, times_ms=[700.0], rounds=1,
                      library_version=None if a == "ours" else "1.2.8") for a in race_["arms"]]
        return {"cells": cells, "status": "done", "rc": 0, "finished": "2026-09-29T13:00:00Z",
                "params": {"arms": {"catboost-cpu": {"params": READBACK}}}}
    monkeypatch.setattr(bb, "_run_race", fake)
    rec = bb.run_race(dict(ctx, retime=True), race)
    cat = [c for c in rec["cells"] if c["arm"] == "catboost-cpu"][0]
    assert cat["source"] == "measured this run" and rec["stored_now"] == 1
    latest = bb.STORE.lookup(bb.STORE.load(ctx["store_path"]), _key(ctx, race, "catboost-cpu"))
    assert latest["measured_at"] == "2026-09-29T13:00:00Z"
    assert latest["params"] == {"params": READBACK}


def test_backfill_store_imports_and_counts_skips_by_reason(tmp_path, monkeypatch):
    ctx, race = _store_ctx(tmp_path)
    cell = dict(bb.base_cell(ctx, race, "catboost-cpu", None), status="ok", median_ms=900.0,
                library_version="1.2.8")
    ours = dict(bb.base_cell(ctx, race, "ours", "identical"), status="ok", median_ms=450.0)
    sha = bb.race_data_sha(ctx, race, ctx["box"])
    params = {"arms": {"catboost-cpu": {"params": READBACK}}}
    board = {"box": ctx["box"], "created": "2026-09-29T10:00:00Z",
             "config": {"rounds": 1, "data_root": ctx["data_root"],
                        "data": {"taxi": {"size_ok": True, "pinned_sha256": sha}}},
             "races": {race["id"]: {"family": "trees", "lane": race["lane"], "dataset": "taxi",
                                    "rows": None, "finished": "2026-09-29T11:00:00Z",
                                    "params": params, "cells": [ours, cell]},
                       "trees/gbdt-symmetric/istella/rows=full": {
                           "family": "trees", "lane": "gbdt-symmetric", "dataset": "istella",
                           "rows": None, "params": params, "cells": [dict(cell, dataset="istella")]},
                       "trees/gbdt-depthwise/taxi/rows=full": {
                           "family": "trees", "lane": "gbdt-depthwise", "dataset": "taxi",
                           "rows": None, "params": params,
                           "cells": [dict(cell, library_version=None, lane="gbdt-depthwise")]}}}
    path = tmp_path / "board.json"
    path.write_text(json.dumps(board))
    store = tmp_path / "store.jsonl"
    imported, skipped = bb.backfill_store(str(path), str(store))
    assert imported == 1 and sum(skipped.values()) == 2
    assert any(w.startswith("no sha256 for istella") for w in skipped)
    assert any(w.startswith("library version: no recorded venv") for w in skipped)
    ctx["store_path"] = str(store)
    _probe_returns(monkeypatch, {"params": READBACK, "version": "1.2.8"})
    assert set(bb.stored_opponents(ctx, race)) == {"catboost-cpu"}


def test_backfill_corpus_pin_and_venv_version(tmp_path, monkeypatch):
    # a corpus key takes its manifest pin when the staged file has the pinned size
    corpus = tmp_path / "input.txt"
    corpus.write_bytes(b"x" * 11)
    monkeypatch.setattr(bb, "_manifest_pins", lambda: {"corpus/enwik8/input.txt": (11, "cd" * 32)})
    monkeypatch.setattr(bb.ALGOS, "corpus_path", lambda key: str(corpus))
    race = {"family": "algos", "lane": "bpe-encode", "dataset": "enwik8", "rows": None}
    ctx = {"data_root": str(tmp_path), "data_sha": {}}
    files = bb.race_data_files(ctx, race)
    assert files == [("corpus/enwik8/input.txt", str(corpus))]
    # the venv version: only a dist-info written before the board started
    site = tmp_path / "venv" / "lib" / "python3.11" / "site-packages"
    dist = site / "catboost-1.2.8.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: catboost\nVersion: 1.2.8\n")
    py = str(tmp_path / "venv" / "bin" / "python")
    os.utime(str(dist), (1_700_000_000, 1_700_000_000))
    assert bb._venv_version(py, "catboost", "2026-09-29T10:00:00Z") == ("1.2.8", None)
    v, why = bb._venv_version(py, "catboost", "2023-01-01T00:00:00Z")
    assert v is None and "changed in the venv after the board started" in why
    v, why = bb._venv_version(py, "xgboost", "2026-09-29T10:00:00Z")
    assert v is None and "not in the recorded venv" in why


# --- the smoke check and its gate --------------------------------------------

def _smoke_cell(arm, status="ok", median=10.0, quality=None):
    return {"arm": arm, "status": status, "median_ms": median,
            "quality": {"r2": 0.5} if quality is None else quality}


def _smoke_race(arms=("ours", "sklearn-cpu"), lane="kmeans"):
    return {"id": "classical/%s/taxi/rows=2000" % lane, "family": "classical", "lane": lane,
            "dataset": "taxi", "arms": list(arms)}


def test_smoke_verdict_pass_and_each_failure():
    race = _smoke_race()
    ok = {"status": "done", "params_check": "MATCHED",
          "cells": [_smoke_cell("ours"), _smoke_cell("sklearn-cpu")]}
    assert bb.smoke_verdict(race, ok, "apple") == []
    assert bb.smoke_verdict(race, None, "apple") == [("-", "not run")]
    assert ("-", "BOARD-PARAMS NOT CHECKED") in bb.smoke_verdict(
        race, dict(ok, params_check="NOT CHECKED"), "apple")
    assert bb.smoke_verdict(race, dict(ok, status="failed"), "apple")[0][1].startswith("status failed")
    bad = dict(ok, cells=[_smoke_cell("ours"), _smoke_cell("sklearn-cpu", median=None)])
    assert bb.smoke_verdict(race, bad, "apple") == [("sklearn-cpu", "no time")]
    bad = dict(ok, cells=[_smoke_cell("ours", quality={}), _smoke_cell("sklearn-cpu")])
    assert bb.smoke_verdict(race, bad, "apple") == [("ours", "no quality value")]
    bad = dict(ok, cells=[_smoke_cell("ours")])
    assert bb.smoke_verdict(race, bad, "apple") == [("sklearn-cpu", "no cell")]
    bad = dict(ok, cells=[_smoke_cell("ours"), _smoke_cell("sklearn-cpu", "REFUSED(x)")])
    assert bb.smoke_verdict(race, bad, "apple") == [("sklearn-cpu", "status REFUSED(x)")]


def test_smoke_planned_refusal_passes_only_as_a_named_refusal():
    race = {"id": "trees/gbdt-lossguide/taxi/rows=2000", "family": "trees",
            "lane": "gbdt-lossguide", "dataset": "taxi", "arms": ["ours", "lightgbm-cuda"]}
    rec = {"status": "done", "params_check": "MATCHED",
           "cells": [_smoke_cell("ours"), _smoke_cell("lightgbm-cuda", "REFUSED(no CUDA build)")]}
    assert bb.smoke_verdict(race, rec, "nvidia") == []
    # on another vendor the same refusal is a failure: the plan does not declare it there
    assert bb.smoke_verdict(race, rec, "apple")
    rec["cells"][1] = _smoke_cell("lightgbm-cuda", "UNKNOWN(not in log)")
    assert bb.smoke_verdict(race, rec, "nvidia")


def test_shards_split_the_plan_round_robin_by_id():
    races = bb.plan_races("nvidia", ["identical"], ["classical", "classical2"])
    a, b = bb.shard_races(races, "1/2"), bb.shard_races(races, "2/2")
    ids = sorted(r["id"] for r in races)
    assert sorted(r["id"] for r in a + b) == ids and not {r["id"] for r in a} & {r["id"] for r in b}
    assert [r["id"] for r in a] == ids[0::2] and [r["id"] for r in b] == ids[1::2]
    assert bb.shard_races(races, "1/2") == a
    for bad in ("0/2", "3/2", "x"):
        with pytest.raises(SystemExit):
            bb.shard_races(races, bad)


def _write_smoke(path, races, vendor="apple", files=None, fail=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": bb.SMOKE_SCHEMA, "vendor": vendor,
        "files_sha256": files or bb.smoke_files_sha256(),
        "races": {bb.smoke_key(r): {"id": r["id"], "pass": bb.smoke_key(r) not in fail,
                                    "failures": []} for r in races}}))


def test_smoke_gate_needs_every_planned_race_in_the_union_of_shards(tmp_path):
    races = bb.plan_races("apple", ["identical"], ["classical"])
    out = tmp_path / "board"
    assert "no SMOKE PASS for" in bb.smoke_gate(str(out), "apple", races)
    one, two = bb.shard_races(races, "1/2"), bb.shard_races(races, "2/2")
    _write_smoke(tmp_path / "board-smoke" / "smoke.json", one)
    assert bb.smoke_gate(str(out), "apple", races)                  # half the plan
    _write_smoke(tmp_path / "board-smoke-2" / "smoke.json", two)
    assert bb.smoke_gate(str(out), "apple", races) is None          # the union covers it
    assert bb.smoke_gate(str(out), "nvidia", races)                 # another vendor
    _write_smoke(tmp_path / "board-smoke-2" / "smoke.json", two, fail={bb.smoke_key(two[0])})
    assert bb.smoke_key(two[0]) in bb.smoke_gate(str(out), "apple", races)
    _write_smoke(tmp_path / "board-smoke-2" / "smoke.json", two, files="0" * 64)
    assert bb.smoke_gate(str(out), "apple", races)                  # other board files


def test_smoke_run_writes_its_own_dir_and_fails_by_name(env, capsys):
    assert _run(env, "--smoke") == 1        # the stubs print no BOARD-PARAMS line
    text = capsys.readouterr().out
    assert re.search(r"SMOKE FAIL \d+/\d+", text) and "BOARD-PARAMS" in text
    smoke = env["tmp"] / "out-smoke"
    doc = json.loads((smoke / "smoke.json").read_text())
    assert doc["schema"] == bb.SMOKE_SCHEMA and doc["total"] == len(doc["races"]) > 0
    assert not env["out"].exists()          # never the board's own directory
    res = json.loads((smoke / "board.json").read_text())
    assert res["config"]["opponent_store"] is None and res["config"]["rounds"] == 1


def test_full_board_refuses_without_smoke_and_records_an_override(env):
    with pytest.raises(SystemExit, match="REFUSING to start a full board"):
        _run(env, "--rows", "full")
    assert _run(env, "--rows", "full", "--no-smoke-gate") == 0
    res = json.loads((env["out"] / "board.json").read_text())
    assert res["config"]["smoke_gate"] == "overridden (--no-smoke-gate)"


def test_full_board_starts_after_a_smoke_pass(env):
    races = bb.plan_races("apple", bb.modes_for("apple"), bb.FAMILIES, ["rf", "kmeans"],
                          bb.DATASETS, None, "full")
    _write_smoke(env["tmp"] / "out-smoke" / "smoke.json", races)
    assert _run(env, "--rows", "full") == 0
    res = json.loads((env["out"] / "board.json").read_text())
    assert res["config"]["smoke_gate"].startswith("passed")


# --- GPU-only board (Andrew, Oct 2 2026): locked in code, no switch -----------

@pytest.mark.parametrize("vendor", ["apple", "nvidia", "amd"])
def test_board_races_only_our_gpu(vendor):
    races = bb.plan_races(vendor, bb.modes_for(vendor))
    for r in races:
        assert not any(bb._ours_runs_on_cpu(r["family"], r["lane"], a) for a in r["our_arms"]), r["id"]
        opp = r["opponents"]
        assert not (any(bb._is_cpu_arm(a) for a in opp) and any(not bb._is_cpu_arm(a) for a in opp)), r["id"]


def test_gpu_only_guard_refuses_cpu_races():
    with pytest.raises(SystemExit, match="our arm ours-cpu runs on the CPU"):
        bb.enforce_gpu_only([{"id": "x", "family": "algos", "lane": "pca",
                              "our_arms": {"ours": "identical", "ours-cpu": "identical"},
                              "opponents": ["cuml-gpu"]}])
    with pytest.raises(SystemExit, match="race beside GPU ones"):
        bb.enforce_gpu_only([{"id": "x", "family": "algos", "lane": "pca",
                              "our_arms": {"ours": "identical"},
                              "opponents": ["cuml-gpu", "sklearn-cpu"]}])
    with pytest.raises(SystemExit, match="runs on the CPU"):
        bb.enforce_gpu_only([{"id": "x", "family": "neural", "lane": "mamba2-infer",
                              "our_arms": {"ours": "identical"}, "opponents": []}])
    with pytest.raises(SystemExit, match="our CPU never races"):
        bb.base_cell({"vendor": "nvidia"}, {"id": "x", "family": "algos", "lane": "pca"},
                     "ours-cpu", "identical")


@pytest.mark.parametrize('quality', [{'error': 'host output empty'}, {'error': None},
                                     {'error': '', 'relative_error_vs_own_host': 0.}])
def test_smoke_explicit_quality_error_never_counts_as_evidence(quality):
    race = next(r for r in bb.plan_races('amd', ['identical'], rows=2000)
                if r['lane'] == 'gcn')
    race = dict(race, arms=['ours'], opponents=[])
    rec = {'status': 'done', 'params_check': 'MATCHED',
           'cells': [{'arm': 'ours', 'status': 'ok', 'median_ms': 1., 'quality': quality}]}
    assert any(reason.startswith('quality error:') for _, reason in bb.smoke_verdict(race, rec, 'amd'))


def test_hdbscan_shape_uses_consumed_recipe_slice():
    raw = dict(lane='hdbscan', block={'arrays': {'X': {'shape': [1000, 11]}}, 'hdbscan': {'rows': 137}})
    assert bb.classical_measured_shape(raw) == '137x11'
    assert bb.classical_measured_shape(raw, 'dbscan') == '1000x11'
    raw['block']['hdbscan']['rows'] = 1001
    assert bb.classical_measured_shape(raw) is None


def test_classical_outputs_survive_until_evidence_seal(tmp_path, monkeypatch):
    saved = tmp_path / 'work/hdbscan-fixture-ours.npz'
    raw = tmp_path / 'raw.json'
    def execute(*args, **kwargs):
        saved.parent.mkdir(exist_ok=True)
        saved.write_bytes(b'scored output bytes')
        raw.write_text('{}')
        return 0
    monkeypatch.setattr(bb, 'classical_cmd', lambda *a: (['fixture'], {}, 30))
    monkeypatch.setattr(bb, 'classical_json_path', lambda *a: str(raw))
    monkeypatch.setattr(bb, 'child_env', lambda *a: {})
    monkeypatch.setattr(bb, 'run_logged', execute)
    monkeypatch.setattr(bb, 'classical_cells', lambda *a: [])
    monkeypatch.setattr(bb, 'add_ratios', lambda cells: cells)
    monkeypatch.setattr(bb, 'attach_params', lambda *a: None)
    race = dict(id='fixture', family='classical', lane='hdbscan', dataset='fixture', rows=None,
                arms=['ours'], our_arms={'ours': 'identical'})
    bb._run_race(dict(out=str(tmp_path), nice=0), race)
    assert saved.read_bytes() == b'scored output bytes'


def test_paired_lossguide_plans_two_own_configurations_and_pairs_each_ratio():
    # smoke-fixes r13 (Andrew 2026-10-14, tools/board_pairing.py): gbdt-lossguide races `ours`
    # (min_split_gain None) against catboost-gpu only and `ours-gain0` (0.0) against xgboost-* only
    race = bb.plan_races("nvidia", ["identical"], ["trees"], ["gbdt-lossguide"], ["taxi"], None)[0]
    assert race["our_arms"] == {"ours": "identical", "ours-gain0": "identical"}
    assert race["our_pairing"] == {"ours": ["catboost-gpu"], "ours-gain0": ["xgboost-*"]}
    assert bb.arm_library("ours-gain0") == "mojolearn" and bb.arm_device("ours-gain0", "nvidia") == "gpu"
    ctx = {"vendor": "nvidia", "rounds": 1}
    ms = {"ours": 100.0, "ours-gain0": 60.0, "catboost-gpu": 50.0, "xgboost-gpu": 30.0, "lightgbm-cuda": 10.0}
    cells = [dict(bb.base_cell(ctx, race, a, race["our_arms"].get(a)), status="ok", median_ms=m,
                  times_ms=[m], rounds=1) for a, m in ms.items()]
    by = {c["arm"]: c for c in bb.add_ratios(cells)}
    assert by["ours"]["pairs_with"] == ["catboost-gpu"] and by["ours-gain0"]["pairs_with"] == ["xgboost-*"]
    assert by["catboost-gpu"]["ratio_ours_identical_over"] == pytest.approx(2.0)
    assert by["xgboost-gpu"]["ratio_ours_identical_over"] == pytest.approx(2.0)
    assert by["xgboost-gpu"]["ratio_ours_arm"] == {"identical": "ours-gain0"}
    assert by["lightgbm-cuda"]["ratio_ours_identical_over"] is None     # no own configuration lists it
    assert "ours-gain0, vs xgboost-*" in bb._arm_label(by["ours-gain0"])
    # an unpaired lane keeps the old rule and carries no pairing field
    sym = bb.plan_races("nvidia", ["identical"], ["trees"], ["gbdt-symmetric"], ["taxi"], None)[0]
    assert "our_pairing" not in sym and "pairs_with" not in bb.base_cell(ctx, sym, "ours", "identical")
    # Apple: ours-ab (FAST) is `ours` with one key changed, so it pairs as `ours`
    apple = bb.plan_races("apple", ["fast", "identical"], ["trees"], ["gbdt-lossguide"], ["taxi"], None)[0]
    assert bb.base_cell({"vendor": "apple", "rounds": 1}, apple, "ours-ab", "fast")["pairs_with"] == ["catboost-gpu"]
    # decision 1: gbdt-depthwise pairs CatBoost with the raw-input `ours`, XGBoost with `ours-prebinned`
    dw = bb.plan_races("nvidia", ["identical"], ["trees"], ["gbdt-depthwise"], ["taxi"], None)[0]
    assert dw["our_arms"] == {"ours": "identical", "ours-prebinned": "identical"}
    assert dw["our_pairing"] == {"ours": ["catboost-*"], "ours-prebinned": ["xgboost-*"]}
    assert bb.PAIR.prebinned_arm("gbdt-depthwise", "xgboost-gpu") and bb.PAIR.prebinned_arm("gbdt-lossguide", "ours-gain0")
    assert not bb.PAIR.prebinned_arm("gbdt-depthwise", "ours") and not bb.PAIR.prebinned_arm("gbdt-lossguide", "catboost-gpu")
    assert not bb.PAIR.prebinned_arm("gbdt-symmetric", "xgboost-gpu")


def test_samba_torch_arms_that_cannot_build_are_excluded_by_name_on_every_vendor():
    # smoke-fixes 2026-10-14: a named exclusion, never a missing probe; applied only to an arm the probe did not build
    for vendor in ("apple", "amd", "nvidia"):
        for lane in ("samba-forward", "samba-train-step"):
            why = bb._planned_refusal(vendor, "neural/%s" % lane, "torch-eager-fp32")
            assert why and "no library implements Samba" in why
    assert bb._planned_refusal("amd", "neural/mamba1-forward", "torch-eager-fp32") is None
