#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CTD-SPAN facts and the span_asymmetry field, against fixtures.

    python3 tools/test_ctd_span.py

Main lane alone executes this file. No GPU, no worker processes, no dataset:
the ready records below are the dicts `_cuml_setup`, `_torch_setup`, `SkKDE`
and `_ours_info` already build, so the logic that turns them into a CTD-SPAN
line is exercised on a laptop.

WHY IT EXISTS. The span reporting reads terms that were ALREADY being
collected and were never surfaced. An emission path that only ever runs
inside a full rented race is a path nobody can prove ran, which is the
failure this lane keeps finding; so the part with the judgement in it --
which asymmetries count, and against whom -- is a function a check can call.

WHAT IT DOES NOT COVER: the `race` loop itself, which needs live workers.
This proves the facts and the verdict, not the plumbing around them.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import classical_two_datasets as ctd           # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print("PASS %s" % name)
    else:
        print("FAIL %s %s" % (name, detail))
        FAILURES.append(name)


# The ready records, exactly as the setup helpers build them -- including the
# `pre_clock_fit` flag every builder now sets, and `scikit-learn` as the
# library string `_sklearn_info` really reports (it is not "sklearn"; a
# fixture that disagrees with the real record is how a reader passes a test
# and fails a run).
OURS = {"library": "mojolearn", "device": "gpu", "numeric_mode_used": "identical",
        "pre_clock_fit": False}
CUML = {"library": "cuml", "device": "gpu", "upload_ms_untimed": 19.7,
        "pre_clock_fit": True,
        "config": "cuml.neighbors.KernelDensity(...); fit before the clock, "
                  "score_samples timed"}
CUML_FIT_TIMED = {"library": "cuml", "device": "gpu", "upload_ms_untimed": 4.2,
                  "pre_clock_fit": False,
                  "config": "cuml.decomposition.PCA(...); fit timed"}
TORCH = {"library": "torch", "device": "gpu", "upload_ms_untimed": 12.0,
         "pre_clock_fit": False,
         "config": "torch kmeans; fit timed"}
SKL = {"library": "scikit-learn", "device": "cpu", "pre_clock_fit": True,
       "config": "KernelDensity(...); score_samples timed", "tree_fit_ms_untimed": 880.0}


def test_facts():
    ours = ctd._span_facts("ours", OURS)
    check("ours input is on the host", ours["input_home"] == "host", ours)
    check("ours uploads inside its clock",
          "host_to_device_upload" in ours["inside_clock"], ours)
    check("ours validates inside its clock",
          "host_validation" in ours["inside_clock"], ours)
    check("ours has no untimed upload", ours["upload_ms_untimed"] is None, ours)
    check("ours does not fit before the clock", ours["pre_clock_fit"] is False, ours)

    cuml = ctd._span_facts("cuml-gpu", CUML)
    check("cuml input is device-resident", cuml["input_home"] == "device", cuml)
    check("cuml upload is outside its clock",
          cuml["upload_ms_untimed"] == 19.7, cuml)
    check("cuml kde fits before its clock", cuml["pre_clock_fit"] is True, cuml)

    pca = ctd._span_facts("cuml-gpu", CUML_FIT_TIMED)
    check("cuml pca does NOT fit before its clock",
          pca["pre_clock_fit"] is False, pca)

    skl = ctd._span_facts("sklearn-cpu", SKL)
    check("sklearn input is on the host", skl["input_home"] == "host", skl)
    check("sklearn kde tree fit is outside its clock",
          skl["fit_ms_untimed"] == 880.0 and skl["pre_clock_fit"] is True, skl)

    # An arm that never sent a ready record must not crash the report.
    empty = ctd._span_facts("ghost", None)
    check("a missing ready record degrades quietly",
          empty["upload_ms_untimed"] is None, empty)


def test_flag_is_declared_not_inferred():
    """Every record this file builds carries the flag, so nothing is guessed."""
    for name, rec in (("ours", OURS), ("cuml-gpu", CUML),
                      ("cuml-pca", CUML_FIT_TIMED), ("torch-gpu", TORCH),
                      ("sklearn-cpu", SKL)):
        s = ctd._span_facts(name, rec)
        check("%s pre_clock_fit is declared" % name,
              s["pre_clock_fit_source"] == "declared", s)
        check("%s carries no guess warning" % name,
              "pre_clock_fit_warning" not in s, s)


def test_structured_flag_survives_a_prose_rewrite():
    """THE SABOTAGE, and the reason the flag exists.

    Reword the config string -- a copy-edit, a rename, a translation -- and
    the DECLARED fact must not move. The same record with the flag removed is
    the old behaviour, and it gets the answer WRONG while still succeeding,
    which is why the fallback has to announce itself."""
    reworded = dict(CUML)
    reworded["config"] = ("cuml.neighbors.KernelDensity(...); index built "
                          "during setup, scoring timed")
    s = ctd._span_facts("cuml-gpu", reworded)
    check("declared flag survives a prose rewrite", s["pre_clock_fit"] is True, s)
    check("declared flag is marked declared",
          s["pre_clock_fit_source"] == "declared", s)

    stripped = {k: v for k, v in reworded.items() if k != "pre_clock_fit"}
    s = ctd._span_facts("cuml-gpu", stripped)
    check("without the flag the true fact is LOST", s["pre_clock_fit"] is False, s)
    check("but the guess is marked as a guess",
          s["pre_clock_fit_source"] == "INFERRED-FROM-PROSE", s)
    check("and the guess warns", "GUESSED" in s.get("pre_clock_fit_warning", ""), s)


def test_missing_flag_drops_the_asymmetry_but_not_silently():
    """Show the check FAILING when the flag is missing.

    With the prose reworded and no flag, cuML's pre-clock fit disappears from
    `span_asymmetry` -- the exact silent zero the old code produced. What is
    no longer silent is the arm: it is marked INFERRED-FROM-PROSE and carries
    a warning, so the omission is visible in the log rather than absent from
    it. Restoring the flag restores the asymmetry."""
    no_flag = {k: v for k, v in CUML.items() if k != "pre_clock_fit"}
    no_flag["config"] = "index built during setup, scoring timed"
    spans = {"ours": ctd._span_facts("ours", OURS),
             "cuml-gpu": ctd._span_facts("cuml-gpu", no_flag)}
    text = ",".join(ctd._span_warnings(spans, ["cuml-gpu"]))
    check("the pre-clock fit vanishes from the asymmetry (the silent zero)",
          "fit_before_its_clock" not in text, text)
    check("the upload asymmetry still stands", "upload_outside_its_clock" in text,
          text)
    check("the arm is marked as a guess, so the omission is visible",
          spans["cuml-gpu"]["pre_clock_fit_source"] == "INFERRED-FROM-PROSE",
          spans["cuml-gpu"])

    with_flag = dict(no_flag)
    with_flag["pre_clock_fit"] = True
    spans["cuml-gpu"] = ctd._span_facts("cuml-gpu", with_flag)
    text = ",".join(ctd._span_warnings(spans, ["cuml-gpu"]))
    check("with the flag the asymmetry is reported again",
          "fit_before_its_clock" in text, text)


def test_declared_false_beats_misleading_prose():
    """A stale comment must not make a fact true. The flag is the fact."""
    lying = dict(CUML_FIT_TIMED)
    lying["config"] = ("PCA(...); fit before the clock -- STALE COMMENT, the "
                       "fit is timed")
    s = ctd._span_facts("cuml-gpu", lying)
    check("declared False beats prose claiming otherwise",
          s["pre_clock_fit"] is False, s)


def test_source_declares_the_flag():
    """THE DECLARATION MUST EXIST AT THE SOURCE, not only in these fixtures.

    Every other check here builds its ready record by hand, so not one of them
    would notice if someone deleted `pre_clock_fit` from CumlKDE -- the
    fixtures would still carry it and every assertion would still pass while
    the real run fell back to guessing from prose. That is the same
    reached-but-inert shape this whole lane keeps finding, one level up.

    So this reads the file: the three classes that fit before their clock must
    SET the flag, the four ready-record builders must DEFAULT it, and SkQuota
    must propagate it. Delete any one of those lines and this fails BY NAME.
    """
    with open(ctd.__file__.replace(".pyc", ".py")) as fh:
        src = fh.read()
    for cls in ("CumlKNN", "CumlKDE", "SkKDE"):
        body = src.split("class %s" % cls, 1)[1].split("\nclass ", 1)[0]
        check("%s sets pre_clock_fit = True at the source" % cls,
              'self.info["pre_clock_fit"] = True' in body, cls)
    for builder in ("_ours_info", "_sklearn_info", "_cuml_setup", "_torch_setup"):
        body = src.split("def %s" % builder, 1)[1].split("\ndef ", 1)[0]
        check("%s declares a pre_clock_fit default" % builder,
              "pre_clock_fit" in body, builder)
    quota = src.split("class SkQuota", 1)[1].split("\nclass ", 1)[0]
    check("SkQuota propagates pre_clock_fit to the capped arm",
          '"pre_clock_fit"' in quota, "SkQuota")


def test_warnings():
    spans = {k: ctd._span_facts(k, v) for k, v in
             (("ours", OURS), ("cuml-gpu", CUML), ("sklearn-cpu", SKL),
              ("torch-gpu", TORCH))}
    warn = ctd._span_warnings(spans, ["cuml-gpu", "sklearn-cpu", "torch-gpu"])
    text = ",".join(warn)
    check("cuml asymmetry is named", "cuml-gpu:" in text, text)
    check("cuml upload is quantified", "19.700" in text, text)
    check("cuml pre-clock fit is named", "fit_before_its_clock" in text, text)
    # cuML's KDE demonstrably fits before its clock and nothing in its ready
    # record TIMES that fit. The fact must be reported and the magnitude must
    # say `unmeasured` -- never a dash, which reads as a formatting fault, and
    # never a zero, which would be a number we did not take.
    check("an unmeasured magnitude says so",
          "fit_before_its_clock(unmeasured)" in text, text)
    check("no dash-ms artefact in the field", "-_ms" not in text, text)
    check("torch upload is named", "torch-gpu:upload_outside_its_clock" in text, text)
    # sklearn is on the host like ours, so its only asymmetry is the pre-clock
    # tree fit -- and NOT an upload it never performs.
    skl = [w for w in warn if w.startswith("sklearn-cpu:")]
    check("sklearn is not accused of an upload it never did",
          skl and "upload_outside" not in skl[0], warn)
    check("sklearn pre-clock fit is named", skl and "880.000" in skl[0], warn)


def test_no_false_asymmetry():
    """Every arm host-resident and fitting inside its clock: nothing to warn
    about, and the field must say `none` rather than going quiet."""
    ours = {"library": "mojolearn", "device": "gpu"}
    opp = {"library": "sklearn", "device": "cpu", "config": "fit timed"}
    spans = {"ours": ctd._span_facts("ours", ours),
             "sklearn-cpu": ctd._span_facts("sklearn-cpu", opp)}
    warn = ctd._span_warnings(spans, ["sklearn-cpu"])
    check("no asymmetry means no warning", warn == [], warn)


def test_direction_is_against_us_only():
    """The field reports clocks that cover LESS work than ours. An opponent
    that pays MORE inside its clock than we do is not flagged, because that
    asymmetry does not make our ratio look better than it is."""
    ours_preclock = {"library": "mojolearn", "device": "gpu"}
    spans = {"ours": ctd._span_facts("ours", ours_preclock),
             "cuml-gpu": ctd._span_facts("cuml-gpu", CUML_FIT_TIMED)}
    warn = ctd._span_warnings(spans, ["cuml-gpu"])
    check("device-resident opponent is still flagged on upload",
          any("upload_outside_its_clock" in w for w in warn), warn)


def main():
    test_facts()
    test_flag_is_declared_not_inferred()
    test_structured_flag_survives_a_prose_rewrite()
    test_missing_flag_drops_the_asymmetry_but_not_silently()
    test_declared_false_beats_misleading_prose()
    test_source_declares_the_flag()
    test_warnings()
    test_no_false_asymmetry()
    test_direction_is_against_us_only()
    if FAILURES:
        print("\nFAILED %d: %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
