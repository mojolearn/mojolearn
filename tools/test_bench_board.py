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
    opps = [(o, None, 200.0) for o in (a.arms.split(",") if a.arms else [])]
    for name, m, _ in ours:
        print("BENCH_BINDING arm=%s requested=%s resolved=%s compiled=%s vendor=%s "
              "path=/v/lib/python3.12/site-packages/mojolearn/_x.so"
              % (name, m, m, m, os.environ["MOJOLEARN_SPEED_EXPECTED_VENDOR"]))
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
    for r in range(1, n + 1):
        for name, _, base in arms:
            if name == refuse:
                continue
            print("FSPEED lane=%s arm=%s shape=%s round=%d ms=%.3f hash=h%s"
                  % (a.lane, name, shape, r, base + r, name))
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
        arms[arm] = {"ms": ms, "warmup_ms": 99.0, "digests": ["d"] * (n + 1), "status": "ok",
                     "info": info, "digest_stable": True,
                     "span": {"input_home": "device" if arm.startswith("torch") else "host",
                              "pre_clock_fit": False}}
    out = {"lane": a.lane, "dataset": a.dataset,
           "block": {"arrays": {"X": {"shape": [4000, 11]}}}, "arms": arms,
           "quality": {arm: {"inertia": 1.5 + i, "n_iter": 20, "reference": "x"}
                       for i, arm in enumerate(arms)}}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "%s-%s.json" % (a.lane, a.dataset)), "w") as fh:
        json.dump(out, fh)
''')


STUB_NEURAL = textwrap.dedent(r'''
    import argparse, json, os, sys
    p = argparse.ArgumentParser()
    p.add_argument("cmd")
    for f in ("--lane", "--shape", "--arms", "--rounds", "--out", "--work", "--ours-python",
              "--theirs-python", "--ready-seconds", "--warmup-seconds", "--round-seconds"):
        p.add_argument(f)
    a = p.parse_args()
    assert a.cmd == "race" and a.ours_python and a.theirs_python
    with open(os.environ["STUB_CALLS"], "a") as fh:
        fh.write("neural %s %s %s\n" % (a.lane, a.shape, a.arms))
    if os.environ.get("STUB_FAIL") == a.lane:
        sys.exit(3)
    n = int(a.rounds)
    data = "bytes" if a.lane.split("-")[0] in ("lm", "samba") else "gaussian"
    arms, qual = {}, {}
    for i, arm in enumerate(a.arms.split(",")):
        ours = arm == "ours"
        cpu = "-cpu-" in arm or (ours and a.lane.endswith("-infer"))
        info = {"device": "cpu" if cpu else "gpu", "version": "1.0",
                "library": "mojolearn" if ours else "torch",
                "device_name": "stub gpu"}
        if ours:
            info["numeric_mode_used"] = "identical"
            info["module_path"] = "/v/lib/python3.12/site-packages/mojolearn/_x.so"
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
            "--data-root", str(data), "--tree-driver", str(trees),
            "--classical-driver", str(classical), "--neural-driver", str(neural),
            "--more-driver", str(more),
            "--rows", "1000", "--rounds", "3"]
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
    assert len(races) == ((len(bb.TREE_LANES) + len(bb.CLASSICAL_LANES)) * len(bb.DATASETS)
                          + len(bb.NEURAL_LANES) + more)
    for r in races:
        if r["family"] == "neural":
            assert r["our_arms"] == {"ours": "identical"}, r["id"]
            continue
        modes = sorted(r["our_arms"].values())
        assert modes == ["fast", "identical"], r["id"]
        if r["family"] == "trees":
            assert r["our_arms"] == {"ours": "identical", "ours-ab": "fast"}
            assert all(o.endswith("-cpu") for o in r["opponents"])
        else:
            assert r["our_arms"] == {"ours": "identical", "ours-fast": "fast"}
            assert "cuml-gpu" not in r["opponents"]


@pytest.mark.parametrize("vendor", ["nvidia", "amd"])
def test_plan_gpu_boxes_identical_only(vendor):
    races = bb.plan_races(vendor, bb.modes_for(vendor))
    for r in races:
        assert r["our_arms"] == {"ours": "identical"}
        assert "ours-ab" not in r["arms"] and "ours-fast" not in r["arms"]
    nv = {r["lane"]: r["opponents"] for r in races if r["family"] == "trees"}
    if vendor == "nvidia":
        assert nv["gbdt-symmetric"] == ["catboost-gpu"]
        assert nv["rf"] == ["cuml-rf-gpu"]
    else:
        assert "cuml-rf-gpu" not in nv["rf"]


def test_plan_filters_and_counts():
    races = bb.plan_races("apple", ["fast", "identical"], ["trees"], ["rf"], ["taxi"], 5000)
    assert [r["id"] for r in races] == ["trees/rf/taxi/rows=5000"]
    s = bb.plan_summary(races)
    assert s == {"races": 1, "cells": 4, "by_family": {"trees": {"races": 1, "cells": 4}}}


def test_tree_command_apple_interleaves_fast():
    race = bb.plan_races("apple", ["fast", "identical"], ["trees"], ["iforest"], ["taxi"], None)[0]
    ctx = {"python": "py", "tree_driver": "drv", "vendor": "apple", "rounds": 5,
           "arm_budget_s": 1, "race_deadline_s": 2}
    cmd, env = bb.tree_cmd(ctx, race)
    assert "--ours-ab" in cmd and cmd[cmd.index("--ours-ab") + 1] == "numeric_mode='fast'"
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


def test_dry_run_prints_plan_and_touches_nothing(env, capsys):
    rc = bb.main(["--dry-run", "--vendor", "apple"] + env["base"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "TOTAL races=88 cells=312" in text
    assert "family neural     races=16 cells=76" in text
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
    assert km["torch-gpu"]["verdict"].startswith("SPAN-ASYMMETRIC")
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
    "neural/transformer-forward/gaussian/shape=full", "neural/transformer-infer/gaussian/shape=full",
    "neural/mamba1-forward/gaussian/shape=full", "neural/mamba1-infer/gaussian/shape=full",
    "neural/mamba2-forward/gaussian/shape=full", "neural/mamba2-infer/gaussian/shape=full",
    "neural/mamba3-forward/gaussian/shape=full", "neural/mamba3-infer/gaussian/shape=full",
    "neural/samba-train-step/bytes/shape=full", "neural/samba-forward/bytes/shape=full",
    "neural/samba-infer/bytes/shape=full",
    "neural/mlp-train-step/gaussian/shape=full", "neural/mlp-infer/gaussian/shape=full"]
GPU_ARMS = {
    "nvidia": ["torch-eager-fp32", "torch-eager-tf32", "torch-compile-fp32", "torch-compile-tf32",
               "torch-eager-bf16", "torch-compile-bf16"],
    "amd": ["torch-eager-fp32", "torch-compile-fp32", "torch-eager-bf16", "torch-compile-bf16"],
    "apple": ["torch-eager-fp32", "torch-compile-fp32", "torch-eager-bf16", "torch-compile-bf16"],
}
CPU_ARMS = ["torch-cpu-eager-fp32", "torch-cpu-compile-fp32", "torch-cpu-eager-bf16",
            "torch-cpu-compile-bf16"]


@pytest.mark.parametrize("vendor", ["apple", "nvidia", "amd"])
def test_plan_neural_identical_only_on_every_vendor(vendor):
    races = bb.plan_races(vendor, bb.modes_for(vendor), ["neural"])
    assert [r["id"] for r in races] == NEURAL_IDS
    for r in races:
        assert r["our_arms"] == {"ours": "identical"}
        assert "ours-ab" not in r["arms"] and "ours-fast" not in r["arms"]
        want = CPU_ARMS if r["lane"].endswith("-infer") else GPU_ARMS[vendor]
        if r["lane"].startswith("mamba1-"):
            # the per-token reference scan is not a compile target (named in NOT_PLANNED)
            want = [a for a in want if "-compile-" not in a]
        assert r["opponents"] == want, r["id"]
        assert r["arms"] == ["ours"] + want
        # TF32 exists on NVIDIA CUDA only; it is never planned elsewhere
        assert any("tf32" in a for a in r["arms"]) == (vendor == "nvidia"
                                                      and not r["lane"].endswith("-infer"))
    for arm in GPU_ARMS["nvidia"]:
        assert bb.arm_library(arm) == "torch" and bb.arm_device(arm, vendor) == "gpu"
    for arm in CPU_ARMS:
        assert bb.arm_library(arm) == "torch" and bb.arm_device(arm, vendor) == "cpu"
    small = bb.plan_races(vendor, ["identical"], ["neural"], ["gemm"], neural_shape="small")
    assert [r["id"] for r in small] == ["neural/gemm/gaussian/shape=small"]
    cells = sum(len(r["arms"]) for r in races)
    assert cells == {"apple": 76, "amd": 76, "nvidia": 95}[vendor]
    assert bb.plan_summary(races)["by_family"] == {"neural": {"races": 16, "cells": cells}}


def test_fast_refused_for_neural_by_name(env):
    with pytest.raises(SystemExit, match="neural family"):
        bb.plan_races("apple", ["fast"], ["neural"])
    with pytest.raises(SystemExit, match="neural family"):
        bb.plan_races("apple", ["fast"])                  # neural is in the default families
    with pytest.raises(SystemExit, match="neural family"):
        bb.main(["--dry-run", "--vendor", "apple", "--modes", "fast", "--families", "neural"])
    # FAST without neural is still the Apple tier for trees and classical
    races = bb.plan_races("apple", ["fast"], ["trees", "classical"])
    assert races and all(set(r["our_arms"].values()) == {"fast"} for r in races)
    for v in ("nvidia", "amd"):
        with pytest.raises(SystemExit):
            bb.main(["--dry-run", "--vendor", v, "--modes", "fast", "--families", "neural"])
    assert _calls(env) == []


@pytest.mark.parametrize("vendor,cells,more,neural", [("apple", 312, 134, 76),
                                                      ("nvidia", 261, 94, 95),
                                                      ("amd", 246, 90, 76)])
def test_dry_run_counts_per_vendor(vendor, cells, more, neural, capsys):
    assert bb.main(["--dry-run", "--vendor", vendor]) == 0
    text = capsys.readouterr().out
    assert "TOTAL races=88 cells=%d" % cells in text
    assert "family classical2 races=44 cells=%d" % more in text
    assert "family neural     races=16 cells=%d" % neural in text
    assert "neural: IDENTICAL only" in text
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
    infer = bb.plan_races("amd", ["identical"], ["neural"], ["mamba2-infer"])[0]
    si = bb.race_settings(ctx, infer)
    assert si["ours_device"] == "cpu" and "Mamba2BlockInference" in si["ours_call"]
    assert si["shape_dims"] == "B1 L512 DM384"             # the CPU lanes' length cap


def _run_neural(env, *extra):
    base = list(env["base"])
    # a neural-only run needs no taxi/Istella: point --data-root at nothing
    base[base.index("--data-root") + 1] = str(env["tmp"] / "no-data-here")
    return bb.main(["--vendor", "nvidia", "--families", "neural", "--neural-shape", "small"]
                   + base + list(extra))


def test_neural_run_schema_quality_and_board(env):
    assert _run_neural(env) == 0
    res = json.loads((env["out"] / "board.json").read_text())
    assert res["config"]["neural_shape"] == "small"
    assert res["config"]["smoke"] is True
    assert set(res["races"]) == {i.replace("shape=full", "shape=small") for i in NEURAL_IDS}
    assert sorted(_calls(env)) == sorted(
        "neural %s small %s" % (l, ",".join(("ours",) + bb.NEURAL_OPPONENTS["nvidia"][l]))
        for l in bb.NEURAL_LANES)
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
    for arm in GPU_ARMS["nvidia"] + CPU_ARMS:
        assert "| %s | torch |" % arm in board, arm
    cpu = {c["arm"]: c for c in res["races"]["neural/mlp-infer/gaussian/shape=small"]["cells"]}
    assert cpu["torch-cpu-eager-bf16"]["device"] == "cpu"
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
    for arm in bbn.ARMS[1:]:
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


def test_neural_driver_inputs_and_quality(tmp_path):
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
    assert q["ours"] == {}


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
    assert nv["classical2/ets/synthetic/rows=full"]["opponents"] == ["cuml-gpu", "statsmodels-cpu"]
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
