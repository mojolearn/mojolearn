# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`python -m mojolearn verify --all`: the reference table, the cell states,
the verdict, the flags, and the one thing the command must never do, hash a
cell differently from tools/identity_break.py.

    cd python && python3 -m mojolearn.tests.test_verify_all      (or pytest)

The table, judge, verdict and flag tests need no GPU and no fit. The run
tests (a corrupted reference reading DIVERGENT with exit 1, and the drift
test) need an importable identical build and skip without one; the drift
test runs the public CPU reference lanes on the base fixture by default and
takes MOJOLEARN_VERIFY_ALL_DRIFT_LANES=a,b (or `all`) for more.
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from mojolearn import __main__ as cli
from mojolearn import _verify_all as va
from mojolearn import _verify_reference as vref

PKG = Path(va.__file__).resolve().parent
ROOT = PKG.parents[1] if (PKG.parents[1] / "tools" / "identity_break.py").is_file() else None


# ---------------------------------------------------------------- the table

def test_shipped_table_loads_and_is_small():
    path = vref.table_path()
    table = vref.load_table(path)
    assert table["format"] == vref.FORMAT
    assert os.path.getsize(path) < 1_000_000, "the wheel's reference table must stay under 1 MB"
    assert table["cells"], "the shipped table carries no cell"
    for rec in table["records"]:
        assert set(("dir", "file", "vendor", "class", "commit")) <= set(rec)
        assert rec["class"] in vref.CLASSES
    for key, cell in table["cells"].items():
        assert "/" in key
        for part, ent in cell.items():
            assert part in vref.PARTS
            assert ent.get("conflict") or isinstance(ent["ref"], str), (key, part)
            for cls, ref in ent["cols"].items():
                idx = ref if isinstance(ref, int) else ref[0]
                assert table["records"][idx]["class"] == cls


@pytest.mark.skipif(ROOT is None, reason="needs the checkout's committed records")
def test_every_reference_is_in_the_record_it_names():
    """No hash in the table is typed: each agreeing column's record file
    carries exactly that value for that cell part."""
    table = vref.load_table()
    base = ROOT / "bench" / "results" / "identity_break"
    cache = {}
    checked = 0
    for key, cell in table["cells"].items():
        for part, ent in cell.items():
            for cls, ref in ent["cols"].items():
                idx, value = (ref, ent["ref"]) if isinstance(ref, int) else ref
                rec = table["records"][idx]
                p = base / rec["dir"] / rec["file"]
                if p not in cache:
                    cache[p] = json.loads(p.read_text(encoding="utf-8"))
                assert vref._part_value(cache[p]["cells"][key], part) == value, (key, part, cls, str(p))
                checked += 1
    assert checked > 1000


def test_load_table_refuses_a_missing_or_foreign_file(tmp_path):
    with pytest.raises(vref.TableError):
        vref.load_table(str(tmp_path / "absent.json"))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(dict(format="something-else", cells={})))
    with pytest.raises(vref.TableError):
        vref.load_table(str(bad))


def _column(vendor, commit, cells, fixtures):
    return dict(mode="identical", commit=commit, vendor=vendor, repeats=2, heldout_seed=1,
                fixtures=fixtures, heldout={}, cells=cells, package=dict(par_devices="0"))


def test_build_table_newest_wins_and_same_commit_disagreement_is_a_conflict(tmp_path):
    class H:
        LANES = {"lane-a": None, "lane-b": None}
        FIXTURES = ["base"]
        __file__ = __file__

        @staticmethod
        def _h(x):
            return "fx"

        @staticmethod
        def fixture(f):
            return (0, 0, 0)

        @staticmethod
        def heldout(f):
            return 0

    fx = {"base": dict(X="fx", y_clf="fx", y_reg="fx")}
    stable = lambda h: dict(verdict="STABLE", hashes=[h, h], infer=["n/a:function"] * 2, infer_verdict="N/A")
    d = tmp_path / "bench" / "results" / "identity_break" / "r"
    d.mkdir(parents=True)
    files = {
        "apple-m4.json": _column("apple-m4", "a" * 40, {"lane-a/base": stable("1111"), "lane-b/base": stable("5555")}, fx),
        "nvidia-h100.json": _column("nvidia-h100", "a" * 40, {"lane-a/base": stable("1111"), "lane-b/base": stable("6666")}, fx),
        "amd-mi300x.sabotage.json": _column("amd-mi300x", "a" * 40, {"lane-a/base": stable("ffff")}, fx),
    }
    for name, body in files.items():
        (d / name).write_text(json.dumps(body))
    table = vref.build_table([str(d / n) for n in files], H, str(tmp_path))
    a = table["cells"]["lane-a/base"]["train"]
    assert a["ref"] == "1111" and set(a["cols"]) == {"apple", "nvidia"}, "a sabotage file must never be admitted"
    b = table["cells"]["lane-b/base"]["train"]
    assert b.get("conflict") and b["ref"] is None
    assert table["cells"]["lane-a/base"]["infer"]["ref"] == "n/a:function"


# ---------------------------------------------------------------- judging

ENT = dict(ref="0123456789abcdef", cols={"apple": 0})


@pytest.mark.parametrize("value, ent, state", [
    ("0123456789abcdef", ENT, vref.IDENTICAL),
    ("fedcba9876543210", ENT, vref.DIVERGENT),
    ("MOVED", ENT, vref.DIVERGENT),
    ("BATCH_MOVED:predict:row 3:...", ENT, vref.DIVERGENT),
    ("RELOAD-MOVED", ENT, vref.DIVERGENT),
    ("0123456789abcdef", None, vref.OWED),
    ("0123456789abcdef", dict(ref=None, conflict=True, cols={}), vref.OWED),
    (None, ENT, vref.REFUSED),
    ("n/a:transductive", dict(ref="n/a:transductive", cols={}), vref.NA),
    ("n/a:no-save", None, vref.NA),
    ("0123456789abcdef", dict(ref="n/a:function", cols={}), vref.OWED),
    ("n/a:function", ENT, vref.OWED),
])
def test_judge_states(value, ent, state):
    got, _ = vref.judge(value, ent, error="NotImplementedError: no CPU implementation of x.y yet")
    assert got == state


def test_refused_carries_the_sentence():
    state, detail = vref.judge(None, ENT, error="NotImplementedError: no CPU implementation of x.y yet")
    assert state == vref.REFUSED and "no CPU implementation of x.y yet" in detail


def _counts(**kw):
    c = {s: 0 for s in vref.STATES}
    c.update({k.replace("NA", "N/A"): v for k, v in kw.items()})
    return c


def test_verdict_exit_codes():
    # a refused part did not run, so it costs the run its pass
    # (lane/expose-inference-surface, 2026-09-16; it used to read VERIFIED)
    assert va.verdict(_counts(IDENTICAL=5, OWED=3, REFUSED=1))[0] == va.EXIT_CANNOT_RUN
    assert va.verdict(_counts(IDENTICAL=5, OWED=3))[0] == va.EXIT_VERIFIED
    assert va.verdict(_counts(IDENTICAL=5, DIVERGENT=1))[0] == va.EXIT_MISMATCH
    assert va.verdict(_counts(REFUSED=2, OWED=1))[0] == va.EXIT_CANNOT_RUN
    assert va.verdict(_counts(OWED=4, NA=1))[0] == va.EXIT_NO_REFERENCE
    from mojolearn import _verify
    assert (va.EXIT_VERIFIED, va.EXIT_MISMATCH, va.EXIT_USAGE, va.EXIT_REFUSED_FAST, va.EXIT_CANNOT_RUN,
            va.EXIT_NO_REFERENCE) == (_verify.EXIT_VERIFIED, _verify.EXIT_MISMATCH, _verify.EXIT_USAGE,
                                      _verify.EXIT_REFUSED_FAST, _verify.EXIT_CANNOT_RUN, _verify.EXIT_NO_REFERENCE)


def test_a_run_that_refused_is_not_reported_as_verified():
    """THE VERIFICATION THAT COULD NOT FAIL (lane/expose-inference-surface,
    2026-09-16). Measured on an Apple M4 CPU-only install whose host bindings
    were stale: 44 IDENTICAL parts, 288 REFUSED, and the public command
    printed `RESULT: VERIFIED ... exit 0`. A user ran our verification, saw
    VERIFIED, and had checked 13 percent of what they believed they checked.

    A REFUSED part is a part that DID NOT RUN. It can never be evidence of
    success, and no number of parts that did run makes up for it, so there is
    no threshold below which refusals are tolerable. The run is INCOMPLETE and
    exits non-zero; only a run with nothing refused may print VERIFIED."""
    code, headline = va.verdict(_counts(IDENTICAL=44, REFUSED=288))
    assert headline != "VERIFIED", "a run with refused parts must not print VERIFIED"
    assert code != va.EXIT_VERIFIED, "a run with refused parts must not exit 0"
    assert (code, headline) == (va.EXIT_CANNOT_RUN, "INCOMPLETE")

    # one refused part is enough
    assert va.verdict(_counts(IDENTICAL=5, OWED=3, REFUSED=1)) == (va.EXIT_CANNOT_RUN, "INCOMPLETE")
    # a run with nothing refused is still VERIFIED, and OWED and N/A do not spoil it
    assert va.verdict(_counts(IDENTICAL=5, OWED=3, NA=2)) == (va.EXIT_VERIFIED, "VERIFIED")
    # a wrong answer still outranks an absent one
    assert va.verdict(_counts(IDENTICAL=5, DIVERGENT=1, REFUSED=9))[0] == va.EXIT_MISMATCH


def test_the_summary_says_how_much_of_the_run_was_actually_checked():
    """`verified 44 of 332 parts` cannot be misread as `verified`."""
    text = va.detail_line(_counts(IDENTICAL=44, REFUSED=288))
    assert "44 of 332" in text, text
    assert "288 refused" in text, text
    clean = va.detail_line(_counts(IDENTICAL=332))
    assert "332 of 332" in clean, clean


def test_the_self_test_cannot_pass_with_a_broken_comparator(monkeypatch):
    """`verify --self-test` exists so a user can WATCH the comparison fail, and
    it is worth nothing unless it would notice a comparator that cannot fail.

    It is two-sided on purpose: the untouched arm must read IDENTICAL and the
    perturbed arm DIVERGENT. A comparator stuck on IDENTICAL passes the first
    and fails the second; one stuck on DIVERGENT does the reverse. Both stubs
    are exercised here, because a self-test that passes with a broken
    comparator is the same defect one level up (lane/expose-inference-surface,
    2026-09-16).

    This runs no lane: the two arms are fed to the same judging code the real
    self-test uses, which is where the property lives.
    """
    table = vref.load_table()
    ent = vref.entry(table, va.SELF_TEST_LANE, va.SELF_TEST_FIXTURE, "train")
    assert ent and isinstance(ent.get("ref"), str) and not ent["ref"].startswith("n/a"), (
        "the shipped table must carry a real train reference for the self-test lane, or the "
        "self-test has nothing to disagree with")
    ref = ent["ref"]
    wrong = ("0" if ref[0] != "0" else "1") + ref[1:]

    def arms(clean_value, dirty_value):
        rows = [dict(lane=va.SELF_TEST_LANE, fixture=va.SELF_TEST_FIXTURE, part="train",
                     value=v, error=None) for v in (clean_value, dirty_value)]
        judged = va.judge_rows(rows, table)
        return judged[0]["state"], judged[1]["state"]

    # honest comparator: the real arms behave as the self-test demands
    assert arms(ref, wrong) == (vref.IDENTICAL, vref.DIVERGENT)

    # stuck on IDENTICAL: the perturbed arm no longer diverges, so the
    # self-test's second condition fails
    monkeypatch.setattr(vref, "judge", lambda value, ent, error=None: (vref.IDENTICAL, ""))
    clean, dirty = arms(ref, wrong)
    assert dirty != vref.DIVERGENT, "the stub did not take effect"
    assert not (clean == vref.IDENTICAL and dirty == vref.DIVERGENT), (
        "a comparator stuck on IDENTICAL would satisfy the self-test; it must not")

    # stuck on DIVERGENT: the untouched arm no longer matches, so the first
    # condition fails
    monkeypatch.setattr(vref, "judge", lambda value, ent, error=None: (vref.DIVERGENT, "stub"))
    clean, dirty = arms(ref, wrong)
    assert clean != vref.IDENTICAL
    assert not (clean == vref.IDENTICAL and dirty == vref.DIVERGENT), (
        "a comparator stuck on DIVERGENT would satisfy the self-test; it must not")


def test_the_self_test_perturbation_is_not_inert():
    """The perturbation must actually change the answer, or the perturbed arm
    reads IDENTICAL and the self-test proves nothing. The FIRST version of it
    moved a single value, `X[0, 0]`, by one ULP, and that was measured INERT
    for this lane: `ols` fits 20,000 x 16 and one last-bit change in one of
    320,000 inputs never reached the rounded coefficients. This holds the
    replacement to being column-wide, so nobody shrinks it back by accident.
    """
    src = (Path(va.__file__).read_text(encoding="utf-8"))
    assert "Xp[:, 0] = np.nextafter" in src, (
        "the self-test perturbation is no longer column-wide; a single-value one-ULP change was "
        "measured inert for this lane and would make the perturbed arm pass by accident")
    assert "values_changed" in src, "the report must say how many values were perturbed"


def _cross(pairs, vendor="metal"):
    """The cross-check's own accounting over (lane, part, gpu, cpu) tuples."""
    rows, agree, differ = [], 0, 0
    for lane, part, g, c in pairs:
        na = isinstance(g, str) and g.startswith("n/a")
        same = (g == c)
        if not na:
            agree += same
            differ += (not same)
        rows.append(dict(lane=lane, fixture="base", part=part, gpu=g, cpu=c,
                         agree=None if na else same, na=na, seconds=0.1))
    return dict(ran=True, vendor=vendor, device_class="apple",
                lanes=sorted({r["lane"] for r in rows}), fixtures=["base"],
                compared=len(rows), agree=agree, differ=differ, skipped={},
                cells=rows, passed=(differ == 0) if rows else None, elapsed_s=0.2)


def test_cross_check_reports_a_mismatch_and_names_both_hashes():
    """The cross-check is worth nothing if it can only ever agree. A perturbed
    side must read DIFFER and BOTH hashes must appear, so a reader can see
    which two values disagreed rather than taking `MISMATCH` on trust."""
    good = _cross([("ols", "infer", "2546a13c03838433", "2546a13c03838433"),
                   ("ols", "batch", "aaaa1111bbbb2222", "aaaa1111bbbb2222")])
    assert good["passed"] is True and good["differ"] == 0

    bad = _cross([("ols", "infer", "2546a13c03838433", "2546a13c03838433"),
                  ("ols", "batch", "aaaa1111bbbb2222", "ffff9999eeee8888")])
    assert bad["passed"] is False and bad["differ"] == 1
    text = va.format_cross_check(bad)
    assert "DIFFER" in text
    assert "aaaa1111bbbb2222" in text and "ffff9999eeee8888" in text, (
        "both differing hashes must be printed, or the mismatch cannot be inspected")


def test_cross_check_nothing_compared_is_not_a_mismatch():
    """`compared == 0` once printed `MISMATCH. 0 of 0 cells differ`, which is
    self-contradictory: nothing was compared, so nothing differed and nothing
    agreed. Reporting a verdict about hashes never computed is the same defect
    as VERIFIED over a refused run (lane/verify-cross-check, 2026-09-16)."""
    empty = _cross([])
    assert empty["passed"] is None, "an empty run must not read as a failure"
    text = va.format_cross_check(empty)
    assert "NOTHING COMPARED" in text
    assert "MISMATCH" not in text, "an empty run must never report a mismatch"


def test_cross_check_on_a_cpu_only_install_says_so_and_does_not_pass():
    """No GPU means no second piece of hardware, so the check did not run.
    That is neither a pass nor a failure, and must never be a silent skip."""
    r = dict(ran=False, vendor="cpu", lanes=[],
             reason="no GPU in this installation, so there is no second piece of hardware to "
                    "compare against. This is not a pass and not a failure: the cross-check "
                    "did not run.")
    text = va.format_cross_check(r)
    assert "NOT RUN" in text
    assert "not a pass" in text
    assert "AGREE" not in text, "a CPU-only install must not print an agreement"


def test_cross_check_batch_na_is_respected_not_invented():
    """A lane declaring `n/a` for batch keeps it. Inventing a comparison there
    would be a check that cannot fail."""
    r = _cross([("umap", "infer", "1111222233334444", "1111222233334444"),
                ("umap", "batch", "n/a:batch-dependent-by-contract",
                 "n/a:batch-dependent-by-contract")])
    assert r["compared"] == 2 and r["agree"] == 1, "the n/a part must not be counted as agreement"
    assert r["passed"] is True
    assert "n/a" in va.format_cross_check(r)


def test_cross_check_scope_tiers_respect_the_apple_lane_cap():
    """The default must not exceed what one Apple Metal process may run:
    identity_break refuses a full column outside a release, in code."""
    assert va.APPLE_LANE_CAP == 24
    harness = va.load_harness()
    quick, every, per_family = va.cross_check_lanes(harness, "quick")
    default, _, _ = va.cross_check_lanes(harness, "default")
    every_lanes, _, _ = va.cross_check_lanes(harness, "all")
    assert set(quick) == set(per_family.values()), "quick is one lane per family"
    assert len(default) <= va.APPLE_LANE_CAP, "the default would be refused on Apple"
    assert len(quick) <= len(default) <= len(every_lanes)
    assert set(default) <= set(every), "the default must stay inside the intersection"


def test_a_corrupted_reference_hash_reads_divergent_and_exit_1():
    """The table's own references, judged as if this box produced them,
    verify; flip one character of one shipped hash and the same rows read
    DIVERGENT and exit 1 (the check that can fail, seen failing)."""
    table = vref.load_table()
    key, cell = next((k, c) for k, c in table["cells"].items()
                     if isinstance(c.get("train", {}).get("ref"), str) and not c["train"]["ref"].startswith("n/a"))
    lane, fixture = key.split("/")
    rows = [dict(lane=lane, fixture=fixture, part=p, value=e["ref"], error=None)
            for p, e in cell.items() if isinstance(e.get("ref"), str)]
    good = va.judge_rows(rows, table)
    counts = {s: sum(1 for r in good if r["state"] == s) for s in vref.STATES}
    assert va.verdict(counts)[0] == va.EXIT_VERIFIED
    bad_table = copy.deepcopy(table)
    ref = bad_table["cells"][key]["train"]["ref"]
    bad_table["cells"][key]["train"]["ref"] = ("0" if ref[0] != "0" else "1") + ref[1:]
    bad = va.judge_rows(rows, bad_table)
    counts = {s: sum(1 for r in bad if r["state"] == s) for s in vref.STATES}
    assert counts[vref.DIVERGENT] == 1
    assert va.verdict(counts)[0] == va.EXIT_MISMATCH


# ---------------------------------------------------------------- lanes and flags

class _FakeHarness:
    LANES = {"rf-clf": None, "rf-reg": None, "ols": None, "ridge": None, "par-forest": None, "no-ref": None}
    FIXTURES = ["base", "ties"]


def _fake_table():
    ent = dict(ref="0123456789abcdef", cols={"apple": 0})
    cells = {f"{l}/{f}": {"train": ent} for l in ("rf-clf", "rf-reg", "ols", "ridge", "par-forest") for f in ("base", "ties")}
    return dict(format=vref.FORMAT, records=[dict(dir="r", file="a.json", vendor="apple-m4", commit="a" * 40)],
                cells=cells, fixtures={}, heldout={})


def test_quick_is_one_lane_per_family_on_base():
    lanes, fixtures = va.select_lanes(_FakeHarness, _fake_table(), "apple", "quick", [])
    fam = va.family_map(lanes)
    assert fixtures == ["base"]
    assert len(set(fam.values())) == len(lanes)
    assert "rf-clf" in lanes and "rf-reg" not in lanes and "ols" in lanes and "ridge" not in lanes
    assert "no-ref" not in lanes, "quick picks a lane the table can judge"


def test_full_and_cpu_lane_sets():
    lanes, fixtures = va.select_lanes(_FakeHarness, _fake_table(), "nvidia", "full", [])
    assert lanes == list(_FakeHarness.LANES) and fixtures == _FakeHarness.FIXTURES
    cpu, _ = va.select_lanes(_FakeHarness, _fake_table(), "cpu", "full", [])
    assert set(cpu) <= set(va.host_surface().public_reference_lanes())
    with pytest.raises(ValueError):
        va.select_lanes(_FakeHarness, _fake_table(), "apple", "full", ["nope"])
    with pytest.raises(ValueError):
        va.select_lanes(_FakeHarness, _fake_table(), "cpu", "full", ["rf-clf"])


def test_flags_route_to_the_suite_or_the_card():
    p = cli.build_parser()
    for argv in (["verify", "--all"], ["verify", "--quick"], ["verify", "--full"], ["verify", "--lanes", "ols"],
                 ["verify", "--all", "--json", "--repeats", "2", "--fixtures", "base"]):
        args = p.parse_args(argv)
        assert cli._wants_suite(args), argv
    for argv in (["verify"], ["verify", "--all-stages"], ["verify", "--json"]):
        assert not cli._wants_suite(p.parse_args(argv)), argv
    args = p.parse_args(["verify", "--quick", "--full"])
    assert va.cmd_verify_all(args) == va.EXIT_USAGE


def test_harness_overrides_are_refused(monkeypatch):
    monkeypatch.setenv("MOJOLEARN_IDENTITY_N", "40000")
    with pytest.raises(va.CannotRun):
        va.load_harness("/nonexistent/identity_break.py")


# ---------------------------------------------------------------- portable models

def test_models_manifest_points_at_small_files():
    base = PKG / vref.TABLE_DIR / va.MODELS_DIR
    manifest = base / va.MODELS_MANIFEST
    if not manifest.is_file():
        pytest.skip("no portable models shipped")
    m = json.loads(manifest.read_text(encoding="utf-8"))
    assert m["format"] == va.MODELS_FORMAT
    table = vref.load_table()
    for model in m["models"]:
        f = base / model["file"]
        assert f.is_file() and f.stat().st_size == model["bytes"] and model["bytes"] < 300_000
        assert vref.entry(table, model["lane"], model["fixture"], "model")["ref"] == model["model_hash"]
        assert vref.entry(table, model["lane"], model["fixture"], "batch")["ref"] == model["batch_hash"]


# ---------------------------------------------------------------- runs (need a build)

def _identical_build():
    code = "import mojolearn, mojolearn._backend as b; print(b.numeric_mode(), b.vendor())"
    env = dict(os.environ, MOJOLEARN_NUMERIC_MODE="identical")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(PKG.parent))
    return r.stdout.split() if r.returncode == 0 else None


def _run_cli(argv, env_extra=None):
    env = dict(os.environ, MOJOLEARN_NUMERIC_MODE="identical", **(env_extra or {}))
    return subprocess.run([sys.executable, "-m", "mojolearn"] + argv, capture_output=True, text=True,
                          env=env, cwd=str(PKG.parent))


def test_cli_corrupted_table_exits_1():
    build = _identical_build()
    if not build:
        pytest.skip("no importable identical build")
    table = vref.load_table()
    ref = vref.entry(table, "gemm-pinned", "base", "train")["ref"]
    ok = _run_cli(["verify", "--lanes", "gemm-pinned", "--fixtures", "base", "--no-models", "--json"])
    assert ok.returncode == 0, ok.stdout[-2000:] + ok.stderr[-2000:]
    with tempfile.TemporaryDirectory() as tmp:
        bad = copy.deepcopy(table)
        bad["cells"]["gemm-pinned/base"]["train"]["ref"] = ("0" if ref[0] != "0" else "1") + ref[1:]
        path = os.path.join(tmp, "table.json")
        vref.write_table(bad, path)
        r = _run_cli(["verify", "--lanes", "gemm-pinned", "--fixtures", "base", "--no-models", "--json",
                      "--reference-table", path])
    assert r.returncode == 1, r.stdout[-2000:] + r.stderr[-2000:]
    report = json.loads(r.stdout)
    assert report["counts"]["DIVERGENT"] == 1 and report["verdict"] == "MISMATCH"


@pytest.mark.skipif(ROOT is None, reason="needs tools/identity_break.py beside the package")
def test_shipped_verifier_hashes_like_the_harness():
    """Same machine, same lanes: the column tools/identity_break.py writes and
    the values the verifier produces are equal part for part."""
    build = _identical_build()
    if not build:
        pytest.skip("no importable identical build")
    _, vendor = build
    lanes = os.environ.get("MOJOLEARN_VERIFY_ALL_DRIFT_LANES", "").strip()
    if not lanes:
        selected = list(va.host_surface().public_reference_lanes())
        # CAP IT ON AN APPLE GPU. This asked for every public reference lane in
        # one process, which was 9 and is 39 since the 2026-09-16 promotion, and
        # `identity_break.refuse_routine_apple_column` refuses more than 24 in
        # one Metal process because a full Apple column is a per-release
        # artifact. The parity this test checks is per lane, so a subset proves
        # exactly the same thing; asking for a column here only made the test
        # unrunnable on any Mac with a GPU build (it still passes on a CPU-only
        # install, which is why this went unseen until a Metal tree ran it).
        if sys.platform == "darwin" and vendor != "cpu":
            selected = selected[:va.APPLE_LANE_CAP]
        lanes = ",".join(selected)
    with tempfile.TemporaryDirectory() as tmp:
        column = os.path.join(tmp, "column.json")
        argv = [str(ROOT / "tools" / "identity_break.py"), "--json", column, "--fixtures", "base",
                "--repeats", "1", "--no-rlpair"]
        if lanes != "all":
            argv += ["--lanes", lanes]
        env = dict(os.environ, MOJOLEARN_NUMERIC_MODE="identical", PYTHONPATH=str(PKG.parent))
        h = subprocess.run([sys.executable] + argv, capture_output=True, text=True, env=env, cwd=str(ROOT))
        assert os.path.isfile(column), h.stdout[-2000:] + h.stderr[-2000:]
        harness_cells = json.loads(Path(column).read_text())["cells"]
        verify_argv = ["verify", "--fixtures", "base", "--no-models", "--json"]
        if lanes != "all":
            verify_argv += ["--lanes", lanes]
        else:
            verify_argv += ["--full"]
        r = _run_cli(verify_argv, dict(MOJOLEARN_IDENTITY_BREAK=str(ROOT / "tools" / "identity_break.py")))
        report = json.loads(r.stdout)
    compared = 0
    for row in report["cells"]:
        if row["lane"].startswith("portable:"):
            continue
        cell = harness_cells[f"{row['lane']}/{row['fixture']}"]
        if cell.get("verdict") == "REFUSED":
            assert row["value"] is None, row
            continue
        if row["part"] == "train":
            want = cell["hashes"][0]
        else:
            want = cell[row["part"]][0]
            if row["part"] == "model" and cell.get("model_verdict") == "RELOAD-MOVED":
                want = "RELOAD-MOVED"
        assert row["value"] == want, (row["lane"], row["part"], row["value"], want)
        compared += 1
    assert compared >= 4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
