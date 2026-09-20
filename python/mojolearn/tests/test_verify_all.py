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

THE THIRD PREREQUISITE IS NUMPY. tools/identity_break.py imports it at
module scope, so the five tests that need the real harness cannot run
without it, and `verify` itself answers exit 4 CANNOT RUN there rather than
failing. Those five state that prerequisite the same way they state the
other two, and skip. NOTHING ELSE IS SKIPPED: every other reason the
harness refuses to load (a MOJOLEARN_IDENTITY_* override set, non-zero
MOJOLEARN_PAR_DEVICES, a harness that raises) still fails, loudly.
"""
import copy
import json
import os
import subprocess
import sys
import traceback
import tempfile
from pathlib import Path

import pytest

from mojolearn import __main__ as cli
from mojolearn import _verify_all as va
from mojolearn import _verify_reference as vref

PKG = Path(va.__file__).resolve().parent
ROOT = PKG.parents[1] if (PKG.parents[1] / "tools" / "identity_break.py").is_file() else None

try:
    import numpy as _numpy  # noqa: F401  (probed, never used: see _need_numpy)
except ImportError as _numpy_exc:  # pragma: no cover - depends on the interpreter
    _NO_NUMPY = f"verification needs NumPy and this interpreter has none: {_numpy_exc}"
else:
    _NO_NUMPY = ""


def _need_numpy():
    """Skip ONLY for an absent NumPy, and only where the real harness is
    needed. The probe is this interpreter's own import, and the subprocess
    tests launch sys.executable, so it answers for them too. It is
    deliberately NOT a catch of CannotRun: an override that changes what the
    harness hashes must still read as a failure, never as a skip."""
    if _NO_NUMPY:
        pytest.skip(_NO_NUMPY)


# ---------------------------------------------------------------- the table

def test_shipped_table_loads_and_is_small():
    path = vref.table_path()
    table = vref.load_table(path)
    assert table["format"] == vref.FORMAT
    # THE BOUND IS ON WHAT THE WHEEL CARRIES, and it moved once, with the
    # measurement (lane/reference-regen, 2026-09-17). The regeneration that
    # closed the optional properties took the table from 6,680 cell parts to
    # 15,426: `stepfull`, `batchgrad`, `batchscale`, `ragged` and `rlpair` are
    # emitted now, where before only the four default parts were, and every
    # registered lane has a cell on every fixture (2,052 = 228 x 9). 1.21 MB
    # for 2.3x the content is proportionate, and the old 1 MB would be met
    # only by dropping parts a user can check.
    #
    # 192 KB of that, 16 percent of the file, is THIRTY distinct `n/a:` reason
    # strings repeated 7,402 times, one per cell part. Storing each once and
    # referring to it would take the file to about 1.02 MB and is the obvious
    # saving; it is a format change to `ref`, which every reader of the table
    # destructures, so it is not made here in passing.
    assert os.path.getsize(path) < 2_000_000, "the wheel's reference table must stay under 2 MB"
    assert table["cells"], "the shipped table carries no cell"
    for rec in table["records"]:
        assert set(("dir", "file", "vendor", "class", "commit")) <= set(rec)
        assert rec["class"] in vref.CLASSES
    for key, cell in table["cells"].items():
        assert "/" in key
        for part, ent in cell.items():
            # The optional properties are in the shipped table since
            # lane/reference-regen: they are emitted by
            # `--emit-reference --batch-checks` and read by
            # `verify --all --batch-checks`, so a table that carries them is
            # the point rather than a surprise.
            assert part in vref.PARTS + vref.OPTIONAL_PARTS
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
                fixtures=fixtures, heldout={f: dict(X=v["X"]) for f,v in fixtures.items()}, cells=cells, package=dict(par_devices="0"))


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
        "apple-m4.json": _column("apple-m4", "a" * 40, {"lane-a/base": stable("1111111111111111"), "lane-b/base": stable("5555555555555555")}, fx),
        "nvidia-h100.json": _column("nvidia-h100", "a" * 40, {"lane-a/base": stable("1111111111111111"), "lane-b/base": stable("6666666666666666")}, fx),
        "amd-mi300x.sabotage.json": _column("amd-mi300x", "a" * 40, {"lane-a/base": stable("ffffffffffffffff")}, fx),
    }
    for name, body in files.items():
        (d / name).write_text(json.dumps(body))
    table = vref.build_table([str(d / n) for n in files], H, str(tmp_path))
    a = table["cells"]["lane-a/base"]["train"]
    assert a["ref"] == "1111111111111111" and set(a["cols"]) == {"apple", "nvidia"}, "a sabotage file must never be admitted"
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
    got, _ = vref.judge(value, ent, error="NotImplementedError: no CPU implementation of x.y yet" if value is None else None)
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
    assert va.verdict(_counts(IDENTICAL=5, OWED=3))[0] == va.EXIT_NO_REFERENCE
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
    # Missing references cannot be offset by successful comparisons; N/A is different.
    assert va.verdict(_counts(IDENTICAL=5, OWED=3, NA=2)) == (va.EXIT_NO_REFERENCE, "INCOMPLETE")
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


def _doc(vendor, device, device_class, cells, commit="a809d92f2", verdict="VERIFIED"):
    """A minimal evidence document, the shape `verify --all --json` writes.

    A cell is `(lane, fixture, part, value)` or `(lane, fixture, part, value,
    state)`; `state` is the verdict THAT document reached about that cell
    against its own reference table, which `--compare` carries up rather than
    absorbing into its agreement count."""
    rows = []
    for c in cells:
        l, f, pt, v = c[:4]
        rows.append(dict(lane=l, fixture=f, part=pt, value=v,
                         state=(c[4] if len(c) > 4 else "IDENTICAL")))
    return dict(format="mojolearn.verify-all-report.v1", verdict=verdict,
                detail="verified %d of %d cell parts" % (len(rows), len(rows)),
                device=dict(mojolearn_version="0.8.5", commit=commit, vendor=vendor,
                            device_class=device_class, device=device, cpu_model="cpu-x",
                            platform="p", python="3.14.6"),
                bindings=[dict(module="m", sha256="d" * 64, size=1)],
                verification_contract=dict(harness_sha256='e' * 64,
                    fixtures={r['fixture']: {'X': 'input'} for r in rows},
                    heldout={r['fixture']: {'X': 'held'} for r in rows},
                    protocols={r['part']: {'version': 1} for r in rows}),
                cells=rows)


_BASE_CELLS = [("ols", "base", "train", "3d1d7c30b12d9872"),
               ("ols", "base", "infer", "2546a13c03838433"),
               ("umap", "base", "batch", "n/a:batch-dependent-by-contract")]


def test_compare_finds_the_one_differing_cell():
    """`--compare` exists for ADVERSARIAL use: two strangers diff their own
    documents with us out of the loop. A comparer that reports agreement on
    mismatched inputs would be the worst instance of the defect this lane has
    been removing, so it is held to naming the exact cell."""
    a = _doc("metal", "Apple M2", "apple", _BASE_CELLS)
    bad = list(_BASE_CELLS)
    bad[1] = ("ols", "base", "infer", "ffff0000ffff0000")
    b = _doc("cuda", "RTX 4090", "nvidia", bad)

    r = va.compare_documents(a, b, "a.json", "b.json")
    assert r["verdict"] == "MISMATCH" and r["exit"] == va.EXIT_MISMATCH
    assert r["differ"] == 1 and r["agree"] == 1
    assert [(d["lane"], d["part"]) for d in r["differing"]] == [("ols", "infer")]
    text = va.format_compare(r)
    assert "2546a13c03838433" in text and "ffff0000ffff0000" in text, (
        "both values must be printed, or the mismatch cannot be inspected")


def test_compare_agrees_only_when_the_hashes_match():
    a = _doc("metal", "Apple M2", "apple", _BASE_CELLS)
    b = _doc("cuda", "RTX 4090", "nvidia", _BASE_CELLS)
    r = va.compare_documents(a, b)
    assert r["verdict"] == "AGREE" and r["exit"] == va.EXIT_VERIFIED
    assert r["agree"] == 2 and r["differ"] == 0
    # the n/a part is an absence both sides agreed on, not an agreement
    assert r["n_a"] == 1
    assert r["provenance"]["independent"] is True


def test_compare_absence_is_never_agreement():
    """A cell in only one document is INCOMPARABLE. Counting it as a match is
    how a comparer becomes unable to fail."""
    a = _doc("metal", "Apple M2", "apple", _BASE_CELLS)
    b = _doc("cuda", "RTX 4090", "nvidia", _BASE_CELLS + [("kde", "base", "train", "aaaa")])
    r = va.compare_documents(a, b)
    assert r["differ"] == 0, "no cell actually differs"
    assert r["verdict"] == "INCOMPLETE" and r["exit"] == va.EXIT_CANNOT_RUN
    assert r["only_in_b"] == [["kde", "base", "train"]]
    assert "Absence is not agreement" in va.format_compare(r)


def test_compare_warns_when_both_documents_are_the_same_device():
    """Two documents from one machine show repeatability, not cross-hardware
    identity. The output must say so rather than let a reader assume more.

    The two documents differ (different commits), so this is a real pair of
    runs, not one file twice; that case has its own verdict below."""
    a = _doc("metal", "Apple M2", "apple", _BASE_CELLS, commit="a809d92f2")
    b = _doc("metal", "Apple M2", "apple", _BASE_CELLS, commit="bb579ecb8")
    r = va.compare_documents(a, b)
    assert r["verdict"] == "AGREE", "two runs of one machine do agree"
    assert r["provenance"]["same_device"] is True
    assert r["provenance"]["independent"] is False
    text = va.format_compare(r)
    assert "SAME device" in text and "proves much less" in text


def test_compare_refuses_one_document_handed_over_twice():
    """THE CHEAPEST FORGERY: run `--all` once, copy the file, compare it with
    itself, publish `AGREE  exit 0`. A byte-identical pair is one document,
    and one document cannot corroborate itself."""
    a = _doc("metal", "Apple M2", "apple", _BASE_CELLS)
    b = _doc("metal", "Apple M2", "apple", _BASE_CELLS)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    r = va.compare_documents(a, b)
    assert r["verdict"] == "SAME DOCUMENT" and r["exit"] == va.EXIT_CANNOT_RUN
    assert r["provenance"]["same_document"] is True
    assert r["provenance"]["independent"] is False
    text = va.format_compare(r)
    assert "byte-identical" in text and "cannot corroborate itself" in text
    assert "RESULT: AGREE" not in text


def test_compare_never_agrees_on_a_cell_neither_side_computed():
    """`value` is null where the lane or the probe RAISED. Two nulls are equal
    as Python values, and the first comparer counted that as agreement: two
    machines that both failed to run a lane, reported as having matched on it.
    An absence on both sides is still an absence."""
    cells = list(_BASE_CELLS)
    cells[1] = ("ols", "base", "infer", None)
    a = _doc("metal", "Apple M2", "apple", cells)
    b = _doc("cuda", "RTX 4090", "nvidia", cells)
    r = va.compare_documents(a, b)
    assert r["agree"] == 1, "only the train hash is a real agreement"
    assert r["uncomputed"] == 1
    assert r["verdict"] == "INCOMPLETE" and r["exit"] == va.EXIT_CANNOT_RUN
    text = va.format_compare(r)
    assert "ols/base infer" in text and "two absences, not a match" in text


def test_compare_never_agrees_on_two_boxes_that_each_contradicted_themselves():
    """`MOVED` means THAT BOX gave two different hashes for one fit; a
    `BATCH_MOVED:` or `RELOAD-MOVED` says the same for the other parts. Two
    documents both carrying the string agree only that the central claim is
    false, so string equality must not read it as a pass."""
    for bad in ("MOVED", "BATCH_MOVED:whole!=split", "RELOAD-MOVED"):
        cells = list(_BASE_CELLS)
        cells[1] = ("ols", "base", "infer", bad)
        a = _doc("metal", "Apple M2", "apple", cells)
        b = _doc("cuda", "RTX 4090", "nvidia", cells)
        r = va.compare_documents(a, b)
        assert r["verdict"] == "SELF-CONTRADICTED", bad
        assert r["exit"] == va.EXIT_MISMATCH, bad
        assert r["moved"] == 1 and r["agree"] == 1, bad
        text = va.format_compare(r)
        assert "disagreed with ITSELF" in text and bad in text, bad
        assert "RESULT: AGREE" not in text, bad


def test_compare_refuses_a_document_with_a_cell_row_twice():
    """THE ATTACK A LAST-WINS DICT INVITES: append a second row for the cell
    you lost on, copied from the other party, and the honest answer is
    overwritten before anything is compared. Both rows are named."""
    cells = list(_BASE_CELLS) + [("ols", "base", "infer", "2546a13c03838433")]
    cells[1] = ("ols", "base", "infer", "ffff0000ffff0000")
    a = _doc("metal", "Apple M2", "apple", cells)
    b = _doc("cuda", "RTX 4090", "nvidia", _BASE_CELLS)
    r = va.compare_documents(a, b, "a.json", "b.json")
    assert r["verdict"] == "MALFORMED" and r["exit"] == va.EXIT_USAGE
    assert r["differ"] == 0 and r["agree"] == 0, "a malformed document is not compared at all"
    text = va.format_compare(r)
    assert "appears twice" in text and "ffff0000ffff0000" in text
    assert "RESULT: MALFORMED" in text and "NOT a pass" in text


def test_compare_refuses_json_that_is_not_an_evidence_document():
    """Any JSON with a `cells` key would otherwise be compared on a guess, and
    two files with no cells "do not disagree"."""
    for junk, why in ((dict(cells=[]), "format"), ([1, 2, 3], "top level"),
                      (dict(format=va.COMPARE_INPUT_FORMAT), "no `cells`")):
        r = va.compare_documents(junk, _doc("cuda", "RTX 4090", "nvidia", _BASE_CELLS), "x", "b")
        assert r["verdict"] == "MALFORMED" and r["exit"] == va.EXIT_USAGE, junk
        assert any(why in m for m in r["problems"]), (junk, r["problems"])


def test_compare_does_not_fold_two_different_n_a_reasons_together():
    """`n/a:batch-dependent-by-contract` against `n/a:no-batch-probe` is two
    builds disagreeing about what the part IS. It is not a bit mismatch, and
    it is certainly not an agreement."""
    other = list(_BASE_CELLS)
    other[2] = ("umap", "base", "batch", "n/a:no-batch-probe")
    r = va.compare_documents(_doc("metal", "Apple M2", "apple", _BASE_CELLS),
                             _doc("cuda", "RTX 4090", "nvidia", other))
    assert r["n_a"] == 0 and r["n_a_differing"] == 1 and r["differ"] == 0
    assert r["verdict"] == "INCOMPLETE" and r["exit"] == va.EXIT_CANNOT_RUN
    assert "different n/a reasons" in va.format_compare(r).lower()


def test_compare_does_not_absorb_a_divergence_into_its_agreement_count():
    """THE SAME DEFECT `verdict()` WAS FIXED FOR, ONE LEVEL UP. Until
    2026-09-16 `verify --all` returned VERIFIED as soon as one part read
    IDENTICAL, before it looked at REFUSED, so a CPU-only install printed
    `VERIFIED, exit 0` over 44 identical and 288 refused parts. A comparer
    reaches the same place through agreement: two parties can hold the same
    hash for a cell that one of them already judged DIVERGENT against its own
    table. They agree, and they agree on an answer recorded as wrong."""
    cells = [("ols", "base", "train", "3d1d7c30b12d9872", "IDENTICAL"),
             ("ols", "base", "infer", "2546a13c03838433", "DIVERGENT")]
    a = _doc("metal", "Apple M2", "apple", cells, verdict="MISMATCH")
    b = _doc("cuda", "RTX 4090", "nvidia", cells, verdict="MISMATCH")
    r = va.compare_documents(a, b, "a.json", "b.json")
    assert r["differ"] == 0, "the two documents really do hold the same bits"
    assert r["agree"] == 1 and r["agreed_divergent"] == 1, (
        "the divergent cell must not be counted as a plain agreement")
    assert r["verdict"] == "AGREED ON A DIVERGENT ANSWER"
    assert r["exit"] == va.EXIT_MISMATCH, "a wrong answer outranks an absent one, as in verdict()"
    text = va.format_compare(r)
    assert "ols/base infer" in text and "judged DIVERGENT" in text
    assert "RESULT: AGREE." not in text
    # and each document's own verdict is shown next to the agreement
    assert text.count("MISMATCH") >= 2, "both documents' own verdicts must be printed"


def test_compare_shows_each_document_s_own_verdict_about_its_own_run():
    """Two parties can agree while one of them checked a fraction of what a
    reader assumes. The only honest place for that is beside the agreement."""
    a = _doc("metal", "Apple M2", "apple", _BASE_CELLS, commit="aaa", verdict="INCOMPLETE")
    b = _doc("cuda", "RTX 4090", "nvidia", _BASE_CELLS, commit="bbb", verdict="VERIFIED")
    r = va.compare_documents(a, b, "a.json", "b.json")
    assert r["verdict"] == "AGREE"
    assert r["provenance"]["own_verdict_a"] == "INCOMPLETE"
    assert r["provenance"]["own_verdict_b"] == "VERIFIED"
    text = va.format_compare(r)
    assert "its own verdict" in text and "INCOMPLETE" in text


def test_compare_prints_every_differing_cell_or_says_how_many_it_hid():
    """PRINT THE MATCHES, NOT THE COUNT. Where there are more than the listing
    shows, the output must say so rather than quietly stop."""
    many = [("lane%03d" % i, "base", "infer", "%016x" % i) for i in range(60)]
    flipped = [(l, f, p, "f" + v[1:]) for l, f, p, v in many]
    r = va.compare_documents(_doc("metal", "Apple M2", "apple", many),
                             _doc("cuda", "RTX 4090", "nvidia", flipped))
    assert r["differ"] == 60
    text = va.format_compare(r)
    assert "... and 20 more" in text, "a truncated list must announce its truncation"
    assert "lane000/base infer" in text


def test_compare_with_no_shared_cell_is_not_an_agreement():
    a = _doc("metal", "Apple M2", "apple", [("ols", "base", "train", "1111")])
    b = _doc("cuda", "RTX 4090", "nvidia", [("kde", "base", "train", "2222")])
    r = va.compare_documents(a, b)
    assert r["agree"] == 0 and r["differ"] == 0
    assert r["verdict"] in ("INCOMPLETE", "NOTHING COMPARED")
    assert r["exit"] == va.EXIT_CANNOT_RUN


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
    _need_numpy()
    harness = va.load_harness()
    quick, every, per_family = va.cross_check_lanes(harness, "quick")
    default, _, _ = va.cross_check_lanes(harness, "default")
    every_lanes, _, _ = va.cross_check_lanes(harness, "all")
    assert set(quick) == set(per_family.values()), "quick is one lane per family"
    assert len(default) <= va.APPLE_LANE_CAP, "the default would be refused on Apple"
    assert len(quick) <= len(default) <= len(every_lanes)
    assert set(default) <= set(every), "the default must stay inside the intersection"


# --------------------------------------------- every lane accounted for

def _exposure(lanes, status="NOT APPLICABLE", reason="claim requires two devices"):
    return {l: dict(status=status, reason=reason, exposed=status == "EXPOSED") for l in lanes}


def test_the_accounting_denominator_is_the_whole_harness():
    """THE NUMBER A USER READS IS 256, NOT 186 (lane/verifier-full-exposure,
    2026-09-20). Every lane the harness defines gets exactly one state, and
    the states sum to the lane list. A lane that fell out of the accounting
    would be exactly the silent absence this block exists to remove, so the
    sum is asserted rather than assumed."""
    harness = va.load_harness()
    lanes = list(harness.LANES)
    exposure = va.host_surface().lane_exposure(lanes)
    acc = va.lane_accounting(lanes, exposure, [], [])
    assert acc["total"] == len(lanes) == 256
    assert sum(acc["counts"].values()) == len(lanes)
    assert set(acc["lanes"]) == set(lanes)
    # and the 70 that a CPU-only install does not run are each a named state
    assert acc["counts"][va.LANE_NOT_APPLICABLE] == 55, "the par-* drivers"
    assert acc["counts"][va.LANE_UNDECLARED] == 0
    assert all(e["reason"] for e in acc["lanes"].values() if e["state"] != va.LANE_VERIFIED)


@pytest.mark.parametrize("state", [s for s in va.LANE_STATES
                                   if s not in (va.LANE_VERIFIED,) + va.LANE_STATES_THAT_DO_NOT_GATE])
def test_no_lane_state_but_verified_can_contribute_to_a_pass(state):
    """Every state that leaves something UNKNOWN, held to the one rule that
    matters. NOT APPLICABLE is excluded by construction and has its own two
    tests below; the parametrization reads the exemption list rather than
    naming it, so adding a second non-gating state cannot quietly widen this
    test's blind spot."""
    acc = dict(total=1, counts={state: 1},
               lanes={"x": dict(state=state, reason="because", exposed=False, ran=False, ran_state=None)})
    assert va.lane_verdict_gaps(acc, ["x"]) != {}, f"{state} was read as coverage"
    assert va.verdict(_counts(IDENTICAL=99), va.lane_verdict_gaps(acc, ["x"]))[0] != va.EXIT_VERIFIED


def test_part_level_na_and_lane_level_not_applicable_agree():
    """ONE RUN, ONE RULE, WHICHEVER LAYER SAYS `INAPPLICABLE`
    (lane/verifier-full-exposure, 2026-09-20, correcting the same lane).

    `verdict()` consults DIVERGENT, REFUSED, OWED and IDENTICAL, and `vref.NA`
    appears in it ZERO times: a part declared `n/a:no-backward` has never made
    a run INCOMPLETE. The first version of the lane accounting gated at the
    LANE level on exactly that concept, so the same run was judged by two
    different rules depending on which layer the inapplicability happened to
    be expressed at.

    This pins them together. The left arm expresses inapplicability in parts,
    the right arm in a lane state, everything else is identical, and the two
    verdicts must be the same. If either layer starts gating, this fails."""
    part_level = _counts(IDENTICAL=8, NA=99)
    assert va.verdict(part_level)[0] == va.EXIT_VERIFIED, (
        "part-level n/a has never gated; if that changed, change the lane level with it")

    lanes = ["ols", "par-forest"]
    acc = va.lane_accounting(
        lanes,
        dict(ols=dict(status="EXPOSED", reason=None, exposed=True),
             **{"par-forest": dict(status="NOT APPLICABLE", reason="claim requires two devices",
                                   exposed=False)}),
        lanes,
        [dict(lane=l, fixture="base", part="train", state=vref.IDENTICAL) for l in lanes])
    lane_level = va.verdict(_counts(IDENTICAL=8, NA=99), va.lane_verdict_gaps(acc, lanes),
                            lanes_checked=va.lane_scope(acc, lanes)["checked"])
    assert lane_level == va.verdict(part_level), (
        "the same run is judged differently depending on which layer says `inapplicable`")


def test_not_applicable_has_exactly_one_derivation():
    """THE ESCAPE HATCH, NAILED SHUT. NOT APPLICABLE is the one state that
    does not gate, so if a lane could be moved into it by editing a reason
    string, any hold could be converted into a pass. It cannot: the ONLY way
    to reach it is `host_surface.PUBLIC_EXCLUDED_PREFIXES`, which is a prefix
    rule over lane names, requires a written sentence that
    `tools/lane_accounting.py` refuses to let be missing, and cannot be
    reached from `PUBLIC_PENDING_LANES` for any reason string at all."""
    surface = va.host_surface()
    lanes = list(va.load_harness().LANES)
    exposure = surface.lane_exposure(lanes)
    inapplicable = [l for l, r in exposure.items() if r["status"] == surface.LANE_NOT_APPLICABLE]
    assert inapplicable, "the state exists, so something must reach it"
    assert all(l.startswith(surface.PUBLIC_EXCLUDED_PREFIXES) for l in inapplicable)
    assert not (set(inapplicable) & set(surface.PUBLIC_PENDING_LANES)), (
        "a pending lane reached the non-gating state")
    # and no reason string can move a pending lane into it
    for why in ("no reference", "one column", "unwatched", "owed artifact nvidia x", "anything"):
        probe = dict(surface.PUBLIC_PENDING_LANES, ols=why)
        saved, surface.PUBLIC_PENDING_LANES = surface.PUBLIC_PENDING_LANES, probe
        try:
            assert surface.lane_exposure(["ols"])["ols"]["status"] != surface.LANE_NOT_APPLICABLE
        finally:
            surface.PUBLIC_PENDING_LANES = saved


def test_a_clean_run_with_inapplicable_lanes_beside_it_is_still_verified():
    """A VERDICT THAT CAN NEVER BE POSITIVE CARRIES NO INFORMATION
    (lane/verifier-full-exposure, 2026-09-20, correcting itself).

    The first version of this block made every non-VERIFIED lane a gap, which
    made VERIFIED unreachable -- not merely hard on a laptop, but unreachable
    on ANY hardware, because `load_harness()` refuses a set
    MOJOLEARN_PAR_DEVICES, so `verify` is a one-device run on an eight-GPU box
    too and the 55 `par-*` lanes are inapplicable there as well. A user who
    can never earn a pass stops reading the verdict, which is the 0.8.6 defect
    wearing the opposite sign.

    So the pass has to stay reachable, and what it must not do is OVERSTATE
    itself. The scope rides with it, as it already does in
    `format_cross_check`."""
    lanes = ["ols", "ridge", "par-forest", "par-mlp"]
    exposure = {l: (dict(status="EXPOSED", reason=None, exposed=True) if not l.startswith("par-")
                    else dict(status="NOT APPLICABLE", reason="claim requires two devices",
                              exposed=False))
                for l in lanes}
    rows = [dict(lane=l, fixture="base", part=p, state=vref.IDENTICAL)
            for l in lanes for p in ("train", "infer")]
    acc = va.lane_accounting(lanes, exposure, lanes, rows)
    assert acc["counts"][va.LANE_VERIFIED] == 2 and acc["counts"][va.LANE_NOT_APPLICABLE] == 2
    scope = va.lane_scope(acc, lanes)
    assert scope == dict(scope=4, checked=2, not_applicable=2, gaps={})
    assert va.verdict(_counts(IDENTICAL=8), scope["gaps"], lanes_checked=scope["checked"]) == (
        va.EXIT_VERIFIED, "VERIFIED")
    # and it says so rather than implying it covered everything
    clause = va.lane_scope_clause(acc, lanes)
    assert clause == "2 of 4 lanes verified; 2 not applicable to any run of this command"


def test_an_all_inapplicable_run_is_cannot_run_because_nothing_was_checked():
    """NOTHING CHECKED IS NOT A PASS, WHICH IS NOT THE SAME AS A GAP.

    A run whose entire scope is NOT APPLICABLE leaves nothing unknown, so it
    has no gaps -- and it also established nothing. Every cell part it
    produces can read IDENTICAL, because a one-device `par-*` column is
    compared against itself and passes whatever the code does. The cell-part
    counts alone say VERIFIED, and the assertion below shows them saying it.

    `format_cross_check` already draws this line, printing NOTHING COMPARED
    rather than a pass when no lane produced both answers. MEASURED end to end
    on an Apple M4 at this commit:

        python -m mojolearn verify --all --lanes par-forest,par-mlp \\
            --fixtures base --no-models
        ... verified 8 of 10 cell parts (0 divergent, 0 owed, 0 refused, 2 n/a)
        RESULT: CANNOT RUN ... exit 4
    """
    lanes = ["par-forest", "par-mlp"]
    rows = [dict(lane=l, fixture="base", part=p, state=vref.IDENTICAL)
            for l in lanes for p in ("train", "infer", "batch", "stepfull")]
    counts = _counts(IDENTICAL=len(rows), NA=2)
    assert va.verdict(counts) == (va.EXIT_VERIFIED, "VERIFIED"), (
        "the cell-part counts alone are a pass; that is the reading that must not survive")

    acc = va.lane_accounting(lanes, _exposure(lanes), lanes, rows)
    assert acc["counts"][va.LANE_NOT_APPLICABLE] == 2
    assert acc["counts"][va.LANE_VERIFIED] == 0, "a degenerate cell is not a verified lane"
    assert all(e["ran_state"] == va.LANE_VERIFIED for e in acc["lanes"].values()), (
        "the run's own reading is kept as evidence, it is just not the verdict")
    scope = va.lane_scope(acc, lanes)
    assert scope["gaps"] == {}, "nothing is unknown, so nothing is a gap"
    assert scope["checked"] == 0, "and nothing was checked either"
    assert va.verdict(counts, scope["gaps"], lanes_checked=scope["checked"]) == (
        va.EXIT_CANNOT_RUN, "CANNOT RUN")


def test_the_accounting_is_printed_even_when_nothing_is_wrong():
    """`186 lanes` used to be printed by a passing run and said nothing about
    the 70. The block is unconditional, so the passing run carries the
    denominator too."""
    acc = va.lane_accounting(["a", "b"], _exposure(["a", "b"], "EXPOSED", None), ["a", "b"],
                             [dict(lane=l, fixture="base", part="train", state=vref.IDENTICAL)
                              for l in ("a", "b")])
    text = "\n".join(va.format_lane_accounting(acc))
    assert "2 of 2 harness lanes accounted for" in text and "2 VERIFIED" in text


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


# ---------------------------------------------------------------- the decode part

#: the eight lanes tools/identity_break.py declares a real stepfull part for,
#: and the hash all four recorded columns landed on
#: (docs/lanes/LANE_STATUS_lane-decode-columns.md, 2026-09-16)
#: mamba2-dtlimit was re-recorded at its corrected clamp on 2026-09-17;
#: see identity_break/2026-09-17_cpu-mamba2-completion/cpu-mac-clean.json.
#:
#: FOUR OF THESE MOVED ON 2026-09-17 (lane/reference-regen) and the reason is
#: the fixture, not the arithmetic: `transformer`, `transformer-window` and
#: `mamba3` are at `norms-near-one-1` (their two same-shape RMSNorm weights
#: stopped being one tensor) and `samba-untied-dropout-accum` is at
#: `steps-3-1`. Those are exactly the lanes `identity_break.LANE_REVISIONS`
#: moved, and exactly the lanes whose old cells the regeneration dropped; the
#: four whose revision did NOT move carry the same hash as before.
#: `mamba2-dtlimit` is the cross-check: another session recorded it
#: independently at the corrected clamp and this lane's CPU column reproduced
#: `6cefffbc50e10b84` bit for bit.
#: Source: bench/results/identity_break/2026-09-17_reference-regen/.
STEPFULL_LANES = {
    "transformer": "5b8c0b041a329128",
    "transformer-window": "787dac866cd00329",
    "mamba1": "f582474b00117f8e",
    "mamba2": "bfd516aa93fe1b12",
    "mamba2-dtlimit": "6cefffbc50e10b84",
    "mamba3": "c1eb221a7da85e5d",
    "samba": "e9c89afd1eb7f273",
    "samba-untied-dropout-accum": "dd836479c7307500",
}


def test_stepfull_is_a_compared_part_and_the_shipped_table_carries_it():
    """THE USER-FACING DOOR TO THE DECODE PROPERTY (lane/expose-stepfull,
    2026-09-16). `PARTS` was the four, so `verify --all` could not compare
    `stepfull` at all and a user could not check that step-by-step decoding
    with a carried state gives the bits of one fresh-state forward pass,
    although four columns had proved it. Adding the part without regenerating
    the table would have been worse than leaving it out, because no row of the
    old table carried a value for it, so this test holds the two together."""
    assert "stepfull" in vref.PARTS
    table = vref.load_table()
    for lane, want in STEPFULL_LANES.items():
        ent = vref.entry(table, lane, "base", "stepfull")
        assert ent is not None, f"{lane}: the shipped table carries no stepfull reference"
        assert ent.get("ref") == want, (lane, ent.get("ref"), want)
        state, _ = vref.judge(want, ent)
        assert state == vref.IDENTICAL


def test_a_lane_with_no_decode_state_reads_na_and_one_with_it_can_read_divergent():
    """A PART NO LANE CAN FAIL IS WORSE THAN NO PART AT ALL. Both halves:
    a lane with no carried state declares `n/a:no-decode-state` and reads
    N/A rather than a spurious OWED or a silent pass, and a lane that has the
    part reads DIVERGENT when its bits differ, with BOTH values printed."""
    assert vref.judge("n/a:no-decode-state", None)[0] == vref.NA
    na_ent = dict(ref="n/a:no-decode-state", cols={"cpu": 0})
    assert vref.judge("n/a:no-decode-state", na_ent)[0] == vref.NA

    table = vref.load_table()
    ent = vref.entry(table, "mamba1", "base", "stepfull")
    state, detail = vref.judge("0000000000000000", ent)
    assert state == vref.DIVERGENT
    assert "0000000000000000" in detail and ent["ref"] in detail, detail
    # the harness's own failure verdict, not a hash, is still DIVERGENT and
    # still names the position rather than a count
    moved = ("BATCH_MOVED:forward(x) vs allocate_state + step, L=16: FIRST DIFFERING POSITION 6 "
             "of 16: element 0: full 0x3eb0f4dc vs 0x3eb0f4dd")
    state, detail = vref.judge(moved, ent)
    assert state == vref.DIVERGENT and "FIRST DIFFERING POSITION 6" in detail, detail


def test_a_corrupted_stepfull_reference_costs_the_run_its_pass():
    """The same shape as the train-part test above, on the new part: the
    table's own stepfull references judge as IDENTICAL, and flipping one of
    them turns the run into a MISMATCH with exit 1."""
    table = vref.load_table()
    rows = [dict(lane=lane, fixture="base", part="stepfull",
                 value=vref.entry(table, lane, "base", "stepfull")["ref"], error=None)
            for lane in STEPFULL_LANES]
    counts = {s: sum(1 for r in va.judge_rows(rows, table) if r["state"] == s) for s in vref.STATES}
    assert counts[vref.IDENTICAL] == len(STEPFULL_LANES)
    assert va.verdict(counts)[0] == va.EXIT_VERIFIED
    bad = copy.deepcopy(table)
    ref = bad["cells"]["mamba1/base"]["stepfull"]["ref"]
    bad["cells"]["mamba1/base"]["stepfull"]["ref"] = ("0" if ref[0] != "0" else "1") + ref[1:]
    judged = va.judge_rows(rows, bad)
    counts = {s: sum(1 for r in judged if r["state"] == s) for s in vref.STATES}
    assert counts[vref.DIVERGENT] == 1
    assert va.verdict(counts)[0] == va.EXIT_MISMATCH
    line = next(r["detail"] for r in judged if r["state"] == vref.DIVERGENT)
    assert ref in line and bad["cells"]["mamba1/base"]["stepfull"]["ref"] in line, line


def test_a_harness_with_no_stepfull_part_refuses_rather_than_passing():
    """AN ABSENT PART IS NOT AN `n/a`. Returning a declaration when the
    harness cannot run the part would make stepfull read N/A on every lane on
    every install, which is a check that cannot fail. It REFUSES, which costs
    the run its VERIFIED, and the sentence says where the part lives."""
    class _NoPart:
        __file__ = "/somewhere/old_identity_break.py"
        BATCH_ALONE = 1
    value, error = va._probe_stepfull(_NoPart, object(), "mamba1", None, {})
    assert value is None and "defines no stepfull part" in error, error
    assert vref.judge(value, vref.entry(vref.load_table(), "mamba1", "base", "stepfull"), error)[0] == vref.REFUSED
    assert va.verdict(_counts(IDENTICAL=100, REFUSED=1))[0] != va.EXIT_VERIFIED


# ------------------------------------------- what a refused part carries


def _harness_for_error_text():
    if ROOT is None:
        pytest.skip("no checkout: tools/identity_break.py is not beside this package")
    _need_numpy()
    return va.load_harness(str(ROOT / "tools" / "identity_break.py"))


def _nested_failure():
    """A raise a few frames deep whose message carries a SECOND traceback,
    the shape `_parallel_pool._call` raises: 'GPU worker failed:\n<the
    worker's own traceback>'."""
    def innermost():
        raise ValueError("MOJOLEARN_TEST_INNERMOST_CAUSE")

    def middle():
        innermost()

    try:
        middle()
    except ValueError:
        worker = traceback.format_exc()
    try:
        raise RuntimeError("GPU worker failed:\n" + worker)
    except RuntimeError as exc:
        return exc


def test_a_refused_part_carries_the_type_message_and_traceback():
    """A REFUSAL THAT NAMES NO CAUSE IS NOT A REFUSAL (2026-09-19). Nine
    par-queries-nn batch cells from a two-device MI300X run carried 300
    characters of the WORKER'S OWN dispatch frame and stopped mid-word: the
    cut kept the outermost frames. The text must carry the exception type,
    its message and the frames."""
    harness = _harness_for_error_text()
    text = harness._exc_text("batch", _nested_failure())
    assert text.startswith("batch: RuntimeError: GPU worker failed:")
    assert "ValueError: MOJOLEARN_TEST_INNERMOST_CAUSE" in text
    assert "in innermost" in text and "in middle" in text


def test_the_cap_keeps_the_innermost_frames_and_the_full_text_survives(tmp_path):
    """THE FAILING SIDE FIRST: the old `[:300]` head cut is run here and must
    LOSE the cause, or this test could pass against the bug it names."""
    harness = _harness_for_error_text()
    exc = _nested_failure()
    old = f"batch: {type(exc).__name__}: {exc}"[:300]
    assert "MOJOLEARN_TEST_INNERMOST_CAUSE" not in old, (
        "the old head cut kept the cause, so this test cannot tell the fix from the bug")

    text = harness._exc_text("batch", exc)
    key = "par-demo/base batch"
    clipped = harness._clip_error(text, key=key, limit=400)
    assert len(clipped) <= 400
    assert clipped.startswith("batch: RuntimeError: GPU worker failed:")
    assert "MOJOLEARN_TEST_INNERMOST_CAUSE" in clipped, clipped
    assert "elided" in clipped

    path = harness.write_error_sidecar(str(tmp_path / "record.json"))
    assert path and path.endswith(".errors.txt")
    body = Path(path).read_text()
    assert key in body and text in body, "the sidecar must carry the UNTRUNCATED text"
    harness._FULL_ERRORS.clear()


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
    # A lane a CPU-only install does not run is refused by name rather than
    # silently dropped. The example is `par-forest` rather than a lane that
    # merely happens to be off the list: `par-*` is excluded by RULE
    # (host_surface.PUBLIC_EXCLUDED_PREFIXES), so this stays a real test of
    # the refusal as the public set grows. It used `rf-clf` until
    # lane/ship-cpu-host-families shipped the rf binding and made that lane
    # public, at which point the assertion had nothing left to catch.
    with pytest.raises(ValueError):
        va.select_lanes(_FakeHarness, _fake_table(), "cpu", "full", ["par-forest"])


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


def test_explicit_training_and_inference_scopes(monkeypatch):
    parser = cli.build_parser()
    observed = []
    monkeypatch.setattr(va, 'cmd_verify_all', lambda args: observed.append(args) or 0)
    training = parser.parse_args(['verify', '--training', '--lanes', 'ols', '--fixtures', 'base'])
    assert cli._verify_dispatch(training) == 0
    assert observed[-1].no_models and not observed[-1].models_only
    assert observed[-1].lanes == 'ols' and observed[-1].fixtures == 'base'
    for flag in ('--inference', '--models-only'):
        inference = parser.parse_args(['verify', flag])
        assert cli._verify_dispatch(inference) == 0
        assert observed[-1].models_only and not observed[-1].no_models
    with pytest.raises(SystemExit):
        parser.parse_args(['verify', '--training', '--inference'])


def test_harness_overrides_are_refused(monkeypatch):
    monkeypatch.setenv("MOJOLEARN_IDENTITY_N", "40000")
    with pytest.raises(va.CannotRun):
        va.load_harness("/nonexistent/identity_break.py")


def test_missing_numpy_has_an_actionable_verification_refusal(monkeypatch):
    def missing(*args):
        raise ModuleNotFoundError("No module named 'numpy'", name="numpy")

    monkeypatch.setattr(va, "_load_by_path", missing)
    with pytest.raises(va.CannotRun, match="python -m pip install numpy"):
        va.load_harness("identity_break.py")


def test_other_harness_import_errors_are_not_reported_as_missing_numpy(monkeypatch):
    def broken(*args):
        raise ModuleNotFoundError("No module named 'internal_missing'", name="internal_missing")

    monkeypatch.setattr(va, "_load_by_path", broken)
    with pytest.raises(ModuleNotFoundError) as error:
        va.load_harness("identity_break.py")
    assert error.value.name == "internal_missing"


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
    # env_extra OVERRIDES rather than collides: a caller pinning the tier
    # itself (`--compare` must work under `fast`) is exactly what it is for
    env = dict(os.environ, MOJOLEARN_NUMERIC_MODE="identical")
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-m", "mojolearn"] + argv, capture_output=True, text=True,
                          env=env, cwd=str(PKG.parent))


def test_cli_compare_exit_codes_are_what_a_stranger_scripts_against(tmp_path):
    """THE EXIT CODE IS THE INTERFACE. Two parties compare in a script, and
    what that script branches on is `$?`. Held end to end through the real
    CLI, because the function returning the right number proves nothing about
    what the process exits with, and a crash exits 1, which is MISMATCH.

    It also runs under `MOJOLEARN_NUMERIC_MODE=fast`, which makes every other
    check REFUSE with exit 3. A third party has none of our bindings and no
    tier set, so `--compare` must dispatch before that gate, not behind it.
    """
    def write(name, cells, vendor="metal", device="Apple M2", cls="apple", commit="a"):
        path = tmp_path / name
        path.write_text(json.dumps(_doc(vendor, device, cls, cells, commit=commit)),
                        encoding="utf-8")
        return str(path)

    a = write("a.json", _BASE_CELLS, commit="aaa")
    same = write("same.json", _BASE_CELLS, commit="bbb", vendor="cuda",
                 device="RTX 4090", cls="nvidia")
    bad = list(_BASE_CELLS)
    bad[1] = ("ols", "base", "infer", "ffff0000ffff0000")
    diff = write("diff.json", bad, commit="bbb", vendor="cuda", device="RTX 4090", cls="nvidia")
    null = list(_BASE_CELLS)
    null[1] = ("ols", "base", "infer", None)
    nulls = write("nulls.json", null, commit="bbb", vendor="cuda", device="RTX 4090", cls="nvidia")
    copy_of_a = write("copy.json", _BASE_CELLS, commit="aaa")
    junk = tmp_path / "junk.json"
    junk.write_text("this is not json", encoding="utf-8")

    cases = [
        ((a, same), va.EXIT_VERIFIED, "RESULT: AGREE"),
        ((a, diff), va.EXIT_MISMATCH, "ffff0000ffff0000"),
        ((a, nulls), va.EXIT_CANNOT_RUN, "Absence is not agreement"),
        ((a, a), va.EXIT_USAGE, "RESULT: SAME FILE"),
        ((a, copy_of_a), va.EXIT_CANNOT_RUN, "RESULT: SAME DOCUMENT"),
        ((a, str(junk)), va.EXIT_USAGE, "RESULT: CANNOT READ"),
    ]
    for (x, y), want, token in cases:
        for tier in ("identical", "fast"):
            r = _run_cli(["verify", "--compare", x, y], dict(MOJOLEARN_NUMERIC_MODE=tier))
            assert r.returncode == want, (x, y, tier, r.returncode, r.stdout[-800:])
            assert token in r.stdout, (x, y, tier, r.stdout[-800:])
            # no path out may be silent, and only a real pass may say AGREE
            assert "RESULT:" in r.stdout, (x, y, tier)
            if want != va.EXIT_VERIFIED:
                assert "RESULT: AGREE" not in r.stdout, (x, y, tier, r.stdout[-800:])


def test_cli_corrupted_table_exits_1():
    build = _identical_build()
    if not build:
        pytest.skip("no importable identical build")
    _need_numpy()
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


def test_the_printed_verdict_carries_its_own_scope_end_to_end():
    """THE SKIPPED COUNT RIDES WITH THE VERDICT, seen on the real RESULT line.

    Caught by its own absence (2026-09-20). A sabotage that dropped the scope
    clause from `cmd_verify_all` was NOT REFUSED by the first version of these
    tests: the clause was asserted only against `lane_scope_clause()` in
    isolation, so nothing checked that the sentence a USER reads carries it.
    A check on a helper nobody is required to call is not a check on the
    output. This runs the command and reads the printed line.

    Both halves matter. A pass must say how much of the harness it covered,
    and a run that covered nothing must not say VERIFIED at all."""
    build = _identical_build()
    if not build:
        pytest.skip("no importable identical build")
    ok = _run_cli(["verify", "--lanes", "ols,ridge,par-forest", "--fixtures", "base", "--no-models"])
    assert ok.returncode == 0, ok.stdout[-2000:] + ok.stderr[-2000:]
    result = [l for l in ok.stdout.splitlines() if l.startswith("RESULT:")][-1]
    assert "VERIFIED" in result, result
    assert "2 of 3 lanes verified" in result, (
        "a pass that does not say its own scope overstates itself: " + result)
    assert "1 not applicable" in result, result

    nothing = _run_cli(["verify", "--lanes", "par-forest,par-mlp", "--fixtures", "base", "--no-models"])
    result = [l for l in nothing.stdout.splitlines() if l.startswith("RESULT:")][-1]
    assert nothing.returncode == va.EXIT_CANNOT_RUN, result
    assert "CANNOT RUN" in result and "0 of 2 lanes verified" in result, result


@pytest.mark.skipif(ROOT is None, reason="needs tools/identity_break.py beside the package")
def test_shipped_verifier_hashes_like_the_harness():
    """Same machine, same lanes: the column tools/identity_break.py writes and
    the values the verifier produces are equal part for part."""
    build = _identical_build()
    if not build:
        pytest.skip("no importable identical build")
    _need_numpy()
    _, vendor = build
    lanes = os.environ.get("MOJOLEARN_VERIFY_ALL_DRIFT_LANES", "").strip()
    if not lanes:
        selected = list(va.host_surface().public_reference_lanes())
        # A routine Metal diagnostic is one lane. The installed-wheel
        # release gates cover broader device surfaces separately; the CPU
        # parity check still exercises every public reference lane.
        if sys.platform == "darwin" and vendor != "cpu":
            selected = selected[:1]  # current routine Metal diagnostic policy
        lanes = ",".join(selected)
    with tempfile.TemporaryDirectory() as tmp:
        column = os.path.join(tmp, "column.json")
        # --step-full because `verify --all` runs the stepfull part on every
        # cell (lane/expose-stepfull, 2026-09-16) while the harness runs it
        # only when asked. Without it this test reads a part the column does
        # not carry, and the parity it exists to check would not cover the
        # part most recently added, which is exactly where drift starts.
        argv = [str(ROOT / "tools" / "identity_break.py"), "--json", column, "--fixtures", "base",
                "--repeats", "1", "--no-rlpair", "--step-full"]
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
    parts_seen = set()
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
        parts_seen.add(row["part"])
    assert compared >= 4
    assert "stepfull" in parts_seen, (
        "no stepfull part was compared, so this test would not notice the verifier and the "
        "harness drifting apart on the decode property")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
