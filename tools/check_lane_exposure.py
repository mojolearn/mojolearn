#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""NO LANE IS SILENTLY ABSENT, AND NO PUBLIC ALGORITHM IS SILENTLY LANELESS.

THE GAP THIS CLOSES (lane/verifier-full-exposure, 2026-09-20). The identity
harness defines 256 lanes. The verifier that ships in the wheel --
`python -m mojolearn verify --all` -- exposed 186 of them, and the other 70
were not reported as anything at all. Fifty-five went out by prefix
(`host_surface.PUBLIC_EXCLUDED_PREFIXES = ("par-",)`) and fifteen sat in
`PUBLIC_PENDING_LANES`. Neither route printed a word. A user read `186 lanes`
and had no way to tell that number from `all of them`, which is the same
shape as the defect that cost 0.8.6 its release: something that did not run
reading exactly like something that passed.

An absence has no failure mode of its own. It cannot be caught by looking at
the thing that is absent, only by counting -- so this file counts. Two
countings, in the two directions the gap can open:

  DOWNWARD, lane by lane. Every lane `tools/identity_break.py` registers must
  come back from `host_surface.lane_exposure()` as EXPOSED, or as NOT
  APPLICABLE / OWED / HELD with a written reason. `UNDECLARED` -- the status
  a lane gets when nothing in the manifest mentions it -- is refused here.
  That is the point of having it: the default for a lane nobody thought about
  is a red check, not a quiet omission.

  UPWARD, algorithm by algorithm. A public class or function with no lane at
  all cannot be missing a cell, so a lane census cannot see it: it is invisible
  to every count this tree keeps. `tools/verification_matrix.py` already
  derives that set; this file holds it to a DECLARATION, both ways. A seventh
  laneless algorithm fails. So does one of the six declared here gaining a
  lane and keeping its excuse, which is what makes the declaration shrink as
  the lanes land instead of sitting there.

WHAT IS NOT CHECKED HERE. Whether a reason is TRUE.
`test_host_surface.test_public_reference_lanes_are_derived_and_every_pending_reason_is_true`
holds each `PUBLIC_PENDING_LANES` reason against the shipped table and the
harness revisions; this file only insists that a reason EXISTS and names a
real lane. The two are deliberately separate: one asks "is this claim true",
this one asks "is anything claimed at all".

    python3 tools/check_lane_exposure.py              # the check
    python3 tools/check_lane_exposure.py --list       # the accounting
    python3 tools/check_lane_exposure.py --self-test  # watch it refuse

A CHECK THAT HAS NEVER BEEN SEEN TO FAIL IS NOT A CHECK, so `--self-test`
perturbs the real manifest seven ways -- a prefix with no sentence, a lane no
table mentions, a blank reason, a declaration for a lane that does not exist,
a public lane the harness does not define, an undeclared laneless algorithm
and a declared one that has gained a lane -- and requires each to be REFUSED
by name. It runs in the test suite as
`test_host_surface.test_the_lane_exposure_check_refuses_every_way_a_lane_can_vanish`.
"""
import argparse
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SURFACE = os.path.join(ROOT, "python/mojolearn/host_surface.py")
HARNESS = os.path.join(ROOT, "tools/identity_break.py")

#: PUBLIC ALGORITHMS WITH NO IDENTITY LANE, each with what is owed.
#:
#: Read `tools/verification_matrix.py --json` for how the set is derived: a
#: public export that no lane body reaches, after re-export collapsing and
#: after `NOT_ALGORITHMS` removes the names that are state containers,
#: constants and mode switches rather than algorithms.
#:
#: THIS DICT IS CHECKED BOTH WAYS. A laneless algorithm missing from it fails,
#: and an entry here that HAS gained a lane fails too, so the list shrinks as
#: the lanes land rather than outliving them. All six below are the queue
#: lane/unlaned-public-algorithms is working; they are declared rather than
#: silently tolerated so that a SEVENTH one cannot arrive unnoticed.
DECLARED_LANELESS = {
    "mamba.Mamba1DecodeSession":
        "owed: an identity_break lane that steps the Mamba-1 decode session "
        "token by token against a fresh-state forward pass (the stepfull part "
        "shape). lane/unlaned-public-algorithms",
    "transformer.TransformerDecodeSession":
        "owed: an identity_break lane that steps the transformer decode "
        "session against a whole-sequence forward pass. "
        "lane/unlaned-public-algorithms",
    "models.ParallelCausalLM":
        "owed: an identity_break lane fitting and decoding the parallel "
        "causal LM. lane/unlaned-public-algorithms",
    "parallel_gaussian_process.fit_gaussian_process_classifier":
        "owed: an identity_break lane fitting the parallel GPC driver. "
        "lane/unlaned-public-algorithms",
    "parallel_gaussian_process.predict_gaussian_process_classifier":
        "owed: an identity_break lane predicting through the parallel GPC "
        "driver. lane/unlaned-public-algorithms",
    "parallel_model_selection.cross_val_score":
        "owed: an identity_break lane scoring through the parallel "
        "cross-validation driver. lane/unlaned-public-algorithms",
}

#: A reason has to say something. The shortest real one in the tree is
#: `no reference` at 13 characters, and a reason below this length is a
#: placeholder rather than an explanation.
MIN_REASON = 8


def load(path, name):
    """A private by-path load, so `--self-test` can perturb the module's
    globals without touching the copy anything else imported."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def harness_lanes(path=HARNESS):
    """Every lane name the harness registers, in its own order."""
    return list(load(path, "_cle_identity_break").LANES)


# ----------------------------------------------------------------- downward

def exposure_problems(surface, lanes):
    """Every way a lane can be absent from `verify --all` without saying so."""
    bad = []
    prefixes = tuple(surface.PUBLIC_EXCLUDED_PREFIXES)
    reasons = dict(surface.PUBLIC_EXCLUDED_PREFIX_REASONS)
    for prefix in prefixes:
        why = reasons.get(prefix)
        if not why or len(why) < MIN_REASON:
            covered = sorted(l for l in lanes if l.startswith(prefix))
            bad.append(
                f"PUBLIC_EXCLUDED_PREFIXES carries {prefix!r} and "
                f"PUBLIC_EXCLUDED_PREFIX_REASONS gives no reason for it. "
                f"{len(covered)} lane(s) would leave `verify --all` with nothing said "
                f"about them ({', '.join(covered[:4])}{'...' if len(covered) > 4 else ''})")
    for prefix in reasons:
        if prefix not in prefixes:
            bad.append(f"PUBLIC_EXCLUDED_PREFIX_REASONS explains {prefix!r}, which "
                       "PUBLIC_EXCLUDED_PREFIXES does not exclude: a dead reason reads "
                       "like a live one")

    known = set(lanes)
    for lane in surface.PUBLIC_PENDING_LANES:
        if lane not in known:
            bad.append(f"PUBLIC_PENDING_LANES holds {lane!r}, which the harness does not "
                       "define. A declaration for a lane that does not exist accounts for "
                       "nothing and hides the lane it was meant to be")
    for lane in surface.public_reference_lanes():
        if lane not in known and lane not in surface.PUBLIC_HOST_ONLY_LANES:
            bad.append(f"public_reference_lanes() exposes {lane!r}, which the harness does "
                       "not define; `verify --all` would count a lane it cannot run")

    try:
        exposure = surface.lane_exposure(lanes)
    except RuntimeError as exc:                      # a prefix with no sentence
        bad.append(str(exc))
        return bad
    for lane in lanes:
        row = exposure[lane]
        status, why = row["status"], row["reason"]
        if status == surface.LANE_UNDECLARED:
            bad.append(f"{lane}: UNDECLARED. Nothing in host_surface.py exposes it, excludes "
                       "it by prefix or holds it in PUBLIC_PENDING_LANES, so `verify --all` "
                       "would neither run it nor mention it. Either declare a CPU route for "
                       "it or hold it with a written reason")
        elif status == surface.LANE_EXPOSED:
            if why:
                bad.append(f"{lane}: EXPOSED but carries a hold reason {why!r}")
        elif not why or len(str(why).strip()) < MIN_REASON:
            bad.append(f"{lane}: held back as {status} with no reason a reader can act on "
                       f"({why!r}). An absence with no sentence is the defect this check exists "
                       "to refuse")
    if len(exposure) != len(known):
        bad.append(f"lane_exposure() accounted for {len(exposure)} of {len(known)} lanes")
    return bad


# ------------------------------------------------------------------- upward

def laneless_algorithms(harness_path=HARNESS, declared=None):
    """`{public name: reason}` for every public algorithm no lane reaches.

    Derived by `tools/verification_matrix.py`, not restated here: one
    derivation, so the matrix document and this gate cannot disagree about
    what is laneless.
    """
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    try:
        import verification_matrix as vm
    finally:
        sys.path.pop(0)
    harness = load(harness_path, "_cle_vm_harness")
    surface_mod = load(SURFACE, "_cle_vm_surface")
    surface, _modules = vm.public_surface()
    # `algorithm_rows` reads only these four keys off a lane row, and the
    # column corpus that fills them costs seconds of IO to answer a question
    # this file does not ask. A gate must be cheap enough to always run.
    stub = {name: dict(gpu=[], cpu=None, sabotage="", batch="") for name in harness.LANES}
    algos = vm.algorithm_rows(harness, stub, surface, vm.harness_references(harness),
                              vm.host_family_classes(surface_mod))
    declared = DECLARED_LANELESS if declared is None else declared
    return {a["name"]: declared.get(a["name"]) for a in algos.values()
            if not a["lanes"] and not a["routed"]}, algos


def laneless_problems(harness_path=HARNESS, declared=None):
    declared = DECLARED_LANELESS if declared is None else declared
    found, algos = laneless_algorithms(harness_path, declared)
    bad = []
    for name in sorted(found):
        why = declared.get(name)
        if not why or len(str(why).strip()) < MIN_REASON:
            bad.append(f"{name}: a public algorithm with NO identity lane and no declaration. "
                       "An algorithm with no lane cannot be missing a cell, so no lane census "
                       "can see it. Write a lane, or declare here what is owed")
    with_lanes = {a["name"] for a in algos.values() if a["lanes"] or a["routed"]}
    for name in sorted(declared):
        if name in with_lanes:
            bad.append(f"{name}: declared laneless, but a lane reaches it now. Delete the "
                       "declaration; an excuse that outlives its debt makes the list a memo")
        elif name not in found:
            bad.append(f"{name}: declared laneless, but it is not a public algorithm at all "
                       "(renamed, removed, or in NOT_ALGORITHMS). Delete the declaration")
    return bad


# ------------------------------------------------------------------- driving

def check(surface=None, lanes=None, harness_path=HARNESS, declared=None, upward=True):
    surface = surface or load(SURFACE, "_cle_host_surface")
    lanes = harness_lanes(harness_path) if lanes is None else lanes
    bad = exposure_problems(surface, lanes)
    if upward:
        bad += laneless_problems(harness_path, declared)
    return bad


def accounting(surface=None, lanes=None, harness_path=HARNESS):
    surface = surface or load(SURFACE, "_cle_host_surface_list")
    lanes = harness_lanes(harness_path) if lanes is None else lanes
    return surface.lane_exposure(lanes), surface.lane_exposure_counts(lanes)


# ------------------------------------------------------------------ self test

def _fresh(name):
    return load(SURFACE, name)


def _mutations():
    """(label, fragment the refusal must contain, a function returning the
    perturbed (surface, lanes, declared)). Each one is a way a lane or an
    algorithm has actually gone missing, or could."""

    def no_sentence_for_a_prefix():
        s = _fresh("_cle_mut1")
        s.PUBLIC_EXCLUDED_PREFIX_REASONS = {}
        return s, None, None

    def a_lane_nothing_declares():
        s = _fresh("_cle_mut2")
        return s, harness_lanes() + ["brand-new-lane"], None

    def a_blank_reason():
        s = _fresh("_cle_mut3")
        s.PUBLIC_PENDING_LANES = dict(s.PUBLIC_PENDING_LANES, mamba3="")
        return s, None, None

    def a_declaration_for_no_lane():
        s = _fresh("_cle_mut4")
        s.PUBLIC_PENDING_LANES = dict(s.PUBLIC_PENDING_LANES)
        s.PUBLIC_PENDING_LANES["lane-that-never-was"] = "no reference"
        return s, None, None

    def a_public_lane_the_harness_lost():
        s = _fresh("_cle_mut5")
        return s, [l for l in harness_lanes() if l != "ols"], None

    def an_undeclared_laneless_algorithm():
        s = _fresh("_cle_mut6")
        return s, None, {k: v for k, v in DECLARED_LANELESS.items()
                         if k != "models.ParallelCausalLM"}

    def an_excuse_that_outlived_its_lane():
        s = _fresh("_cle_mut7")
        return s, None, dict(DECLARED_LANELESS, **{"KMeans": "owed: nothing, this has lanes"})

    return [
        ("a prefix excludes 55 lanes and says why nowhere", "PUBLIC_EXCLUDED_PREFIX_REASONS gives no reason", no_sentence_for_a_prefix),
        ("a lane the manifest never heard of", "brand-new-lane: UNDECLARED", a_lane_nothing_declares),
        ("a hold with a blank reason", "mamba3: held back as HELD with no reason", a_blank_reason),
        ("a declaration for a lane that does not exist", "lane-that-never-was", a_declaration_for_no_lane),
        ("a public lane the harness does not define", "public_reference_lanes() exposes 'ols'", a_public_lane_the_harness_lost),
        ("a seventh laneless public algorithm", "models.ParallelCausalLM: a public algorithm with NO identity lane", an_undeclared_laneless_algorithm),
        ("an excuse that outlived its debt", "KMeans: declared laneless, but a lane reaches it now", an_excuse_that_outlived_its_lane),
    ]


def self_test():
    ok = True
    clean = check()
    print(f"# unmutated tree: {len(clean)} problem(s)")
    for line in clean:
        print("    " + line)
    ok &= not clean
    for label, fragment, build in _mutations():
        surface, lanes, declared = build()
        got = check(surface=surface, lanes=lanes, declared=declared)
        hit = [g for g in got if fragment in g]
        print(f"\n# MUTATION: {label}")
        if hit:
            for line in hit:
                print("    REFUSED " + line)
        else:
            print(f"    NOT REFUSED. This check cannot see {label!r}, so every other verdict "
                  f"it gives is worth less. All problems seen: {got}")
            ok = False
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--list", action="store_true", help="print the lane accounting")
    p.add_argument("--self-test", action="store_true",
                   help="perturb the manifest and require each perturbation to be refused")
    p.add_argument("--no-upward", action="store_true",
                   help="skip the public-algorithm half (it imports the harness)")
    p.add_argument("--harness", default=HARNESS)
    a = p.parse_args(argv)
    if a.self_test:
        good = self_test()
        print("\nok: every way a lane can vanish above was refused" if good else "\nFAILED: see above")
        return 0 if good else 1
    if a.list:
        exposure, counts = accounting(harness_path=a.harness)
        for lane, row in exposure.items():
            print(f"{lane:34s} {row['status']:<15} {row['reason'] or ''}"[:160])
        print("")
        for status, n in counts.items():
            print(f"# {status:<15} {n}")
        print(f"# {'TOTAL':<15} {sum(counts.values())}")
        return 0
    bad = check(harness_path=a.harness, upward=not a.no_upward)
    for line in bad:
        print("LANE EXPOSURE: " + line)
    lanes = harness_lanes(a.harness)
    print(f"{'REFUSED' if bad else 'OK'}: {len(lanes)} harness lanes, "
          f"{len(DECLARED_LANELESS)} declared laneless algorithm(s), {len(bad)} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
