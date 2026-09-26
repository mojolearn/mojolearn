# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The board's INFERENCE cells: planning, the drivers' --infer phases (default
off), parsing, cells, ratios and rendering. NumPy only; no mojolearn, no GPU,
no dataset, no opponent library.

    .pixi/envs/test/bin/python -m pytest tools/test_bench_board_infer.py
"""
import importlib.util
import io
import json
import os
import sys
import types
from contextlib import redirect_stdout

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bb = _load("bench_board_for_infer", os.path.join(HERE, "bench_board.py"))
bbt = _load("test_bench_board_for_infer", os.path.join(HERE, "test_bench_board.py"))
inf = bb.INFER
ctd = _load("ctd_for_infer", os.path.join(HERE, "classical_two_datasets.py"))
sys.path.insert(0, HERE)
fsa = _load("forest_speed_arm_for_infer", os.path.join(REPO, "bench", "speed", "forest_speed_arm.py"))

env = bbt.env            # the shared fixture (stub drivers, fake data)

# The trees stub, extended: with --infer it emits the FSPEED-INFER protocol.
STUB_TREES_INFER = bbt.STUB_TREES + r'''
if a.infer:
    for name, _, base in arms:
        if name == refuse:
            continue
        print("FSPEED-INFER-PATH lane=%s arm=%s call=stub %s predict_proba(X), column 1"
              % (a.lane, name, name))
    for batch, rows, scale in (("test", 500, 1.0), ("large", 1000, 2.0)):
        for name, _, base in arms:
            if name == refuse:
                continue
            print("FSPEED-INFER-WARMUP lane=%s arm=%s batch=%s rows=%d ms=%.3f"
                  % (a.lane, name, batch, rows, base))
        for r in range(1, n + 1):
            for name, _, base in arms:
                if name == refuse:
                    continue
                print("FSPEED-INFER lane=%s arm=%s batch=%s rows=%d round=%d ms=%.3f hash=i%s"
                      % (a.lane, name, batch, rows, r, scale * base / 10 + r, name))
        if batch == "test":
            for name, _, base in arms:
                if name == refuse:
                    continue
                print("FSPEED-INFER-ACC lane=%s arm=%s batch=test metric=logloss value=%.6f"
                      % (a.lane, name, base / 1000))
        if a.ours_ab:
            print("FSPEED-INFER-AGREE lane=%s batch=%s arms=ours,ours-ab rows=%d bits_equal=yes "
                  "max_abs_diff=0" % (a.lane, batch, rows))
'''

# The classical stub, extended: with --infer the race JSON carries `infer`.
STUB_CLASSICAL_INFER = bbt.STUB_CLASSICAL + r'''
if a.infer:
    path = os.path.join(a.out, "%s-%s.json" % (a.lane, a.dataset))
    with open(path) as fh:
        out = json.load(fh)
    iarms, iq = {}, {}
    for i, arm in enumerate(a.arms.split(",")):
        iarms[arm] = {"ms": [5.0 * (i + 1) + r for r in range(n)], "warmup_ms": 9.0,
                      "digests": ["e"] * (n + 1), "status": "ok", "digest_stable": True,
                      "call_text": "stub %s predict(Xq)" % arm,
                      "info": {"upload_ms_untimed": 1.0 if arm.startswith("torch") else None}}
        iq[arm] = {"eval_inertia": 2.0 + i}
        if arm != "ours":
            iq[arm]["bits_equal_vs_ours"] = arm == "ours-fast"
    out["infer"] = {"call": "predict", "batch": "Xq", "rows": 500, "arms": iarms, "quality": iq}
    with open(path, "w") as fh:
        json.dump(out, fh)
'''


@pytest.fixture
def ienv(env):
    t = env["tmp"] / "stub_trees_infer.py"
    t.write_text(STUB_TREES_INFER)
    c = env["tmp"] / "stub_classical_infer.py"
    c.write_text(STUB_CLASSICAL_INFER)
    base = list(env["base"])
    base[base.index("--tree-driver") + 1] = str(t)
    base[base.index("--classical-driver") + 1] = str(c)
    env["base"] = base
    return env


# --- planning -----------------------------------------------------------------

def test_plan_cells_trees_two_batches_classical_four_lanes():
    races = bb.plan_races("apple", ["fast", "identical"], ["trees", "classical"], None, ["taxi"], None)
    by = {r["lane"]: inf.plan_cells(r) for r in races}
    assert len(by["rf"]) == 2 * len(bb.TREE_OPPONENTS["apple"]["rf"] + ("ours", "ours-ab"))
    assert {b for _, b in by["rf"]} == {"test", "large"}
    for lane in ("kmeans", "pca", "ols", "svc"):
        assert by[lane] and all(b == "Xq" for _, b in by[lane])
    for lane in ("knn", "kde", "dbscan", "hdbscan"):
        assert by[lane] == []


def test_driver_args_follow_the_flag():
    race = bb.plan_races("nvidia", ["identical"], ["trees"], ["rf"], ["taxi"], None)[0]
    ctx = {"python": "py", "tree_driver": "drv", "vendor": "nvidia", "rounds": 5,
           "arm_budget_s": 1, "race_deadline_s": 2}
    cmd, _ = bb.tree_cmd(ctx, race)
    assert "--infer" not in cmd                  # no ctx["infer"]: the driver runs as before
    ctx["infer"] = True
    cmd, _ = bb.tree_cmd(ctx, race)
    assert "--infer" in cmd
    cctx = dict(ctx, classical_driver="c", out="/o", ctd_data="/d", round_seconds=0)
    for lane, want in (("kmeans", True), ("svc", True), ("knn", False), ("dbscan", False)):
        race = bb.plan_races("nvidia", ["identical"], ["classical"], [lane], ["taxi"], None)[0]
        cmd, _, _ = bb.classical_cmd(cctx, race)
        assert ("--infer" in cmd) is want, lane


def test_dry_run_counts_inference_cells(env, capsys):
    assert bb.main(["--dry-run", "--vendor", "apple", "--families", "trees,classical"]
                   + env["base"]) == 0
    text = capsys.readouterr().out
    # trees: 12 races, arms (ours, ours-ab + opponents) x 2 batches; classical:
    # kmeans/pca/ols/svc x 2 datasets, every arm once.
    trees = sum(2 * (2 + len(v)) for v in bb.TREE_OPPONENTS["apple"].values()) * 2
    classical = sum(2 + len(bb.CLASSICAL_OPPONENTS["apple"][l]) for l in inf.CLASSICAL_INFER_LANES) * 2
    assert "INFER cells=%d (classical %d, trees %d;" % (trees + classical, classical, trees) in text
    assert bb.main(["--dry-run", "--vendor", "apple", "--no-infer"] + env["base"]) == 0
    assert "INFER cells" not in capsys.readouterr().out


# --- an end-to-end run with stub drivers ---------------------------------------

def test_run_infer_cells_ratios_and_board(ienv):
    assert bb.main(["--vendor", "apple", "--lanes", "rf,kmeans"] + ienv["base"]) == 0
    res = json.loads((ienv["out"] / "board.json").read_text())
    assert res["config"]["infer"] is True
    rf = res["races"]["trees/rf/taxi/rows=1000"]
    # the fit cells are exactly what they were
    assert {c["arm"] for c in rf["cells"]} == {"ours", "ours-ab", "sklearn-rf-cpu", "lightgbm-cpu"}
    assert all("phase" not in c for c in rf["cells"])
    ic = {(c["arm"], c["batch"]): c for c in rf["infer_cells"]}
    assert len(ic) == 8
    for c in ic.values():
        assert c["phase"] == "infer" and c["status"] == "ok" and len(c["times_ms"]) == 3
        assert c["hash_stable"] is True and c["call"].startswith("stub ")
    # median ours = 100/10+2 = 12, ours-ab 10, opponents 22 on `test`; x2 on `large`
    assert ic[("sklearn-rf-cpu", "test")]["ratio_ours_identical_over"] == pytest.approx(12 / 22)
    assert ic[("sklearn-rf-cpu", "test")]["ratio_ours_fast_over"] == pytest.approx(10 / 22)
    assert ic[("sklearn-rf-cpu", "large")]["ratio_ours_identical_over"] == pytest.approx(22 / 42)
    assert ic[("ours-ab", "test")]["ratio_ours_identical_over"] is None   # never FAST / IDENTICAL
    assert ic[("ours-ab", "test")]["quality"]["bits_equal_vs_ours_identical"] is True
    assert ic[("ours", "test")]["quality"]["logloss_matches_fit"] is True
    assert ic[("ours", "test")]["batch_rows"] == 500 and ic[("ours", "large")]["batch_rows"] == 1000
    km = res["races"]["classical/kmeans/taxi/rows=1000"]
    kc = {c["arm"]: c for c in km["infer_cells"]}
    assert set(kc) == {"ours", "ours-fast", "sklearn-cpu", "torch-gpu"}
    assert kc["sklearn-cpu"]["ratio_ours_identical_over"] == pytest.approx(6 / 16)
    assert kc["torch-gpu"]["verdict"].startswith("SPAN-ASYMMETRIC")
    assert kc["ours-fast"]["quality"]["bits_equal_vs_ours"] is True
    board = (ienv["out"] / "BOARD.md").read_text()
    assert "## Inference at a glance" in board and "Inference cells: 24 (ok 24)." in board
    assert "inference call, sklearn-rf-cpu: stub sklearn-rf-cpu predict_proba(X), column 1" in board
    assert not bbt.BANNED.search(board)


def test_no_infer_run_has_no_infer_cells(ienv):
    assert bb.main(["--vendor", "apple", "--lanes", "rf", "--no-infer"] + ienv["base"]) == 0
    res = json.loads((ienv["out"] / "board.json").read_text())
    for rec in res["races"].values():
        assert "infer_cells" not in rec
        assert "--infer" not in rec["command"]
    assert "Inference at a glance" not in (ienv["out"] / "BOARD.md").read_text()


def test_driver_without_infer_lines_is_unknown_not_blank(env):
    # the plain stub accepts --infer and prints nothing for it
    assert bb.main(["--vendor", "apple", "--lanes", "rf", "--datasets", "taxi"] + env["base"]) == 0
    res = json.loads((env["out"] / "board.json").read_text())
    st = {c["status"] for c in res["races"]["trees/rf/taxi/rows=1000"]["infer_cells"]}
    assert st == {"UNKNOWN(no inference lines)"}


# --- the trees driver's phase (forest_speed_arm.py --infer) ----------------------

def test_forest_driver_infer_is_off_by_default():
    a = fsa.build_parser().parse_args(["--lane", "rf"])
    assert a.infer is False and a.infer_large_rows == 1_000_000


class _Proba:
    def __init__(self, shift=0.0):
        self.shift = shift

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-(X[:, 0].astype(np.float64) + self.shift)))
        return np.column_stack((1.0 - p, p))


def _data(n_train=64, n_test=32):
    rng = np.random.default_rng(7)
    xt = rng.normal(size=(n_train, 3)).astype(np.float32)
    xs = rng.normal(size=(n_test, 3)).astype(np.float32)
    d = fsa.spec.Data("stub", xt, xs, (xt[:, 0] > 0).astype(np.float32),
                      (xs[:, 0] > 0).astype(np.float32), "binary", 2)
    fsa.prepare_our_inputs(d)
    return d


def _infer_lines(arms, models, data, rounds=2, large=40):
    buf = io.StringIO()
    with redirect_stdout(buf):
        fsa.run_inference("rf", arms, models, data, rounds, large, float("inf"))
    return buf.getvalue()


def test_forest_infer_phase_lines_parse_and_agree(tmp_path):
    data = _data()
    arms = [types.SimpleNamespace(name=n, library=l) for n, l in
            (("ours", "mojolearn"), ("ours-ab", "mojolearn"), ("sklearn-rf-cpu", "sklearn"),
             ("mystery-arm", "mystery"))]
    models = {"ours": _Proba(), "ours-ab": _Proba(), "sklearn-rf-cpu": _Proba(0.5),
              "mystery-arm": _Proba()}
    text = _infer_lines(arms, models, data)
    log = tmp_path / "rf.log"
    log.write_text(text)
    p = inf.parse_tree_infer(str(log))
    assert p["rounds"][("ours", "test")]["rows"] == 32
    assert p["rounds"][("ours", "large")]["rows"] == 40          # capped by --infer-large-rows
    assert len(p["rounds"][("sklearn-rf-cpu", "large")]["ms"]) == 2
    assert p["agree"]["test"]["bits_equal"] == "yes" and p["agree"]["large"]["bits_equal"] == "yes"
    assert "no inference path wired" in p["refused"][("mystery-arm", "all")]
    assert "predict_proba" in p["paths"]["ours"]
    assert set(p["acc"][("ours", "test")]) == {"logloss", "auc"}
    # the fit parser never reads an inference line
    summ = _load("summ_for_infer", os.path.join(HERE, "bench_all_summarize.py"))
    assert summ.parse_tree_log(str(log))[0] == {}


def test_forest_infer_fast_identical_disagreement_is_reported():
    data = _data()
    arms = [types.SimpleNamespace(name=n, library="mojolearn") for n in ("ours", "ours-ab")]
    text = _infer_lines(arms, {"ours": _Proba(), "ours-ab": _Proba(1e-3)}, data)
    agree = [l for l in text.splitlines() if l.startswith("FSPEED-INFER-AGREE")]
    assert len(agree) == 2 and all("bits_equal=no" in l for l in agree)


def test_forest_infer_spec_paths_per_library():
    class Booster:
        def inplace_predict(self, X):
            return np.zeros(len(X))

        def predict(self, X):
            return np.zeros(len(X))

    class XGB:
        def get_booster(self):
            return Booster()

    class LGB:
        booster_ = Booster()

    _, _, t = fsa.infer_spec("xgboost-cpu", "gbdt-depthwise", "binary", XGB())
    assert "inplace_predict" in t and "=" not in t
    _, _, t = fsa.infer_spec("lightgbm-cpu", "rf", "binary", LGB())
    assert "Booster.predict" in t
    _, _, t = fsa.infer_spec("ours", "iforest", "binary", types.SimpleNamespace(score_samples=None))
    assert "rebuilt" in t and "score_samples" in t


# --- the classical racer's phase (race --infer) ---------------------------------

def test_classical_infer_runner_dispatch_and_quality():
    rng = np.random.default_rng(7)
    xq = rng.normal(size=(20, 3)).astype(np.float32)
    C = rng.normal(size=(4, 3)).astype(np.float32)

    class Est:
        def predict(self, X):
            d = ((X[:, None, :] - C[None]) ** 2).sum(axis=2)
            return d.argmin(axis=1)

    runner = ctd.SkKMeans.__new__(ctd.SkKMeans)
    runner.est = Est()
    ir = ctd.infer_runner("kmeans", runner, {"Xq": xq})
    ir.call()
    ir.sync()
    out = ir.outputs()
    assert out["pred"].dtype == np.int32 and "sklearn" in ir.desc
    fit = {"ours": {"centers": C}, "sklearn-cpu": {"centers": C}}
    bad = dict(out)
    bad = {"pred": (out["pred"] + 1) % 4}
    q = ctd.infer_quality("kmeans", {"Xq": xq}, {"ours": out, "sklearn-cpu": bad}, fit)
    assert q["ours"]["label_agreement_own_centers"] == 1.0
    assert q["sklearn-cpu"]["label_agreement_own_centers"] < 1.0
    assert q["sklearn-cpu"]["bits_equal_vs_ours"] is False
    with pytest.raises(RuntimeError):
        ctd.infer_runner("knn", runner, {"Xq": xq})


def test_classical_ols_and_pca_quality_against_own_model():
    rng = np.random.default_rng(7)
    xq = rng.normal(size=(30, 4)).astype(np.float32)
    coef = rng.normal(size=4)
    yq = (xq.astype(np.float64) @ coef + 0.5).astype(np.float32)
    pred = (xq.astype(np.float64) @ coef + 0.5).astype(np.float32)
    q = ctd.infer_quality("ols", {"Xq": xq, "yq": yq}, {"ours": {"pred": pred}},
                          {"ours": {"coef": coef, "intercept": np.array([0.5])}})
    assert q["ours"]["r2_eval"] == pytest.approx(1.0)
    assert q["ours"]["predict_max_rel_err_own_fp64"] < 1e-6
    comp = np.linalg.qr(rng.normal(size=(4, 2)))[0].T
    mean = xq.mean(axis=0)
    z = ((xq - mean) @ comp.T).astype(np.float32)
    q = ctd.infer_quality("pca", {"Xq": xq}, {"ours": {"pred": z}, "ours-fast": {"pred": z}},
                          {"ours": {"components": comp, "mean": mean}})
    assert q["ours"]["transform_max_rel_err_own_fp64"] < 1e-6
    assert q["ours-fast"]["bits_equal_vs_ours"] is True
    assert q["ours-fast"]["max_abs_diff_vs_ours"] == 0.0


class _FakeWorker:
    def __init__(self, ms, fail_round=None):
        self.alive, self.ms, self.fail, self.sent = True, ms, fail_round, []

    def send(self, text):
        self.sent.append(text)

    def read(self, seconds):
        r = int(self.sent[-1].split()[1])
        if r == self.fail:
            return {"event": "infer_error", "error": "boom"}
        return {"event": "infer", "round": r, "ms": self.ms + r, "digest": "d",
                "call": "stub", "info": {}}

    def kill(self, status, error):
        self.alive = False


def test_classical_infer_phase_rounds_and_refusal(capsys):
    args = types.SimpleNamespace(rounds=3, warmup_seconds=1, round_seconds=1, pause=0.0)
    workers = {"ours": _FakeWorker(10.0), "sklearn-cpu": _FakeWorker(20.0, fail_round=2),
               "torch-gpu": _FakeWorker(5.0)}
    result = {"block": {"arrays": {"Xq": {"shape": [500, 11]}}},
              "arms": {"ours": {"ms": [1, 2, 3]}, "sklearn-cpu": {"ms": [1, 2, 3]},
                       "torch-gpu": {"ms": [1]}}}          # torch-gpu did not finish its fit
    rec = ctd.infer_phase(args, "kmeans", "taxi", list(workers), workers, result)
    assert rec["rows"] == 500 and rec["call"] == "predict"
    assert rec["arms"]["ours"]["ms"] == [11.0, 12.0, 13.0] and rec["arms"]["ours"]["warmup_ms"] == 10.0
    assert rec["arms"]["sklearn-cpu"]["status"] == "error"
    assert workers["sklearn-cpu"].alive                   # its fit outputs can still be saved
    assert rec["arms"]["torch-gpu"]["status"] == "no_fit" and workers["torch-gpu"].sent == []
    ctd.infer_summary("kmeans", "taxi", rec, 3)
    out = capsys.readouterr().out
    assert "CTD-INFER lane=kmeans dataset=taxi arm=ours call=predict rows=500 status=ok" in out
    assert "CTD-INFER-REFUSED lane=kmeans dataset=taxi arm=sklearn-cpu stage=infer2" in out
