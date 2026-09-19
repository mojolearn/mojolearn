# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`python -m mojolearn verify --all`: check mojolearn's identity claims on
this machine, cell by cell, against the reference table shipped in the wheel.

WHAT RUNS. The identity_break lanes, through the harness itself: the wheel
carries a byte copy of `tools/identity_break.py` as
`mojolearn/_identity_break.py`, and this module imports it and calls its
fixtures, its lanes, `_train_hash`, `_probe_fit` and `_probe_batch`. No lane
is defined here, so a hash this command prints is the hash the harness would
write for the same cell on the same machine
(python/mojolearn/tests/test_verify_all.py holds the two to that).

WHAT IS COMPARED. Each cell part this box produces (train: the fit read back;
infer: the model on held-out rows; model: the saved file's bytes; batch: the
held-out rows whole, alone, split and by prefix; stepfull: a sequence decoded
one token at a time with a carried state against one fresh-state forward pass
over the whole of it, bitwise per position) against
`mojolearn/verify_reference/table.json` (`_verify_reference.py` builds it
from the committed records). Each part reads IDENTICAL, DIVERGENT, OWED (no
record carries it yet), REFUSED (the lane or probe raised; the sentence is
printed) or N/A (the estimator has no such output).

ON A GPU INSTALL every lane the harness defines runs. ON A CPU-ONLY INSTALL
the lanes the manifest lists as public reference checks run
(`host_surface.public_reference_lanes()`), and the portable models run on
every install: small models trained on a GPU and saved, whose file bytes
and whose batch answers the table carries for the GPU columns. Loading one
on a CPU and getting the same answers is the train on GPU, infer anywhere
claim, checked where the user is.

WHAT A PASS DOES NOT SAY. It is one machine. It does not re-measure other
vendors (the committed records did), cover inputs other than the fixtures,
or say anything about speed. An OWED part is not a pass.
"""
import hashlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback

from . import _verify_reference as vref

EXIT_VERIFIED = 0
EXIT_MISMATCH = 1
EXIT_USAGE = 2
EXIT_REFUSED_FAST = 3
EXIT_CANNOT_RUN = 4
EXIT_NO_REFERENCE = 5

QUICK_FIXTURES = ("base",)
MODELS_DIR = "models"
MODELS_MANIFEST = "models.json"
MODELS_FORMAT = "mojolearn.verify-models.v1"
#: the lanes whose base-fixture model `--emit-models` saves: one forest, one
#: boosting model, one linear model and one decomposition
DEFAULT_MODEL_LANES = ("rf-clf", "gbdt-symmetric", "ols", "pca")
#: the environment switches that make the harness hash different inputs
#: than the record; a run with any of them set is refused before any fit
_HARNESS_OVERRIDES = ("MOJOLEARN_IDENTITY_N", "MOJOLEARN_IDENTITY_WIDE",
                      "MOJOLEARN_IDENTITY_BATCH_SABOTAGE")
_REEXEC_ENV = "MOJOLEARN_VERIFY_ALL_REEXEC"

#: a lane with no manifest family is grouped by its leading name
_ROOT_FAMILIES = (("par-", "parallel drivers"), ("byte-lm", "byte_lm"), ("samba", "samba"),
                  ("tokenizer", "tokenizer"), ("ivf", "ivf"), ("embedding", "embedding"))


class CannotRun(Exception):
    """Maps to exit 4."""


def _pkg_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _emit(text, stream=None):
    stream = stream or sys.stdout
    stream.write(text if text.endswith("\n") else text + "\n")
    stream.flush()


def _load_by_path(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def host_surface():
    """The manifest, loaded by path so this works before any binding import."""
    return _load_by_path("mojolearn_verify_all_host_surface", os.path.join(_pkg_dir(), "host_surface.py"))


# --------------------------------------------------------------------------
# the harness
# --------------------------------------------------------------------------

def harness_path():
    """(path, how): MOJOLEARN_IDENTITY_BREAK, the wheel copy, or the checkout."""
    from . import _identity
    return _identity.harness_path()


def load_harness(path=None):
    for var in _HARNESS_OVERRIDES:
        if os.environ.get(var, "").strip() not in ("", "0"):
            raise CannotRun(f"{var} is set; it changes what the harness hashes, so no cell would "
                            "be comparable with the record. Unset it.")
    if os.environ.get("MOJOLEARN_PAR_DEVICES", "0").strip() not in ("", "0"):
        raise CannotRun("MOJOLEARN_PAR_DEVICES is set to more than device 0; the reference is the "
                        "one-device record. Unset it.")
    if path is None:
        path, _ = harness_path()
    try:
        return _load_by_path("mojolearn_verify_all_harness", path)
    except ModuleNotFoundError as exc:
        if exc.name != "numpy":
            raise
        raise CannotRun("Verification requires NumPy. Install it with: "
                        "python -m pip install numpy") from exc


def family_map(lanes):
    """{lane: family}. The manifest's families first (every CPU training lane
    is declared there), then the lane's leading name."""
    fam = {}
    for f in host_surface().FAMILIES:
        for lane in f["training_lanes"]:
            fam.setdefault(lane, f["family"])
    out = {}
    for lane in lanes:
        if lane in fam:
            out[lane] = fam[lane]
            continue
        out[lane] = next((label for prefix, label in _ROOT_FAMILIES if lane.startswith(prefix)), "other")
    return out


def select_lanes(harness, table, vendor_class, depth, asked, include_pending=False):
    """(lanes, fixtures) for this run, or raises ValueError naming the problem."""
    all_lanes = list(harness.LANES)
    if vendor_class == "cpu":
        surface = host_surface()
        public = surface.public_reference_lanes()
        # A user may explicitly exercise a declared CPU route before its
        # reference is ready. Its results read OWED, never VERIFIED.
        eligible = (set(public) | set(surface.covered_lanes())) if asked or include_pending else set(public)
        allowed = [l for l in harness.LANES if l in eligible
                   and (include_pending or not l.startswith(surface.PUBLIC_EXCLUDED_PREFIXES))]
    else:
        allowed = all_lanes
    if asked:
        unknown = [l for l in asked if l not in harness.LANES]
        if unknown:
            raise ValueError(f"--lanes names lanes the harness does not define: {unknown}")
        outside = [l for l in asked if l not in allowed]
        if outside:
            raise ValueError(f"--lanes names lanes this CPU-only install does not run: {outside}; "
                             f"it runs {allowed}")
        lanes = [l for l in allowed if l in asked]
    else:
        lanes = list(allowed)
    fixtures = list(harness.FIXTURES)
    if depth == "quick":
        fixtures = [f for f in fixtures if f in QUICK_FIXTURES]
        if not asked:
            fam = family_map(lanes)
            chosen, seen = [], set()
            for lane in lanes:
                if fam[lane] in seen:
                    continue
                # the family's first lane that the table has a train reference for
                ent = vref.entry(table, lane, fixtures[0], "train")
                if ent is None or ent.get("ref") is None:
                    continue
                chosen.append(lane)
                seen.add(fam[lane])
            lanes = chosen
    return lanes, fixtures


# --------------------------------------------------------------------------
# running cells
# --------------------------------------------------------------------------

def _collapse(values, errors):
    """One value over the repeats: None (raised), the hash, MOVED, or the
    first BATCH_MOVED / RELOAD-MOVED verdict."""
    if not values or any(v is None for v in values):
        return None, (errors[0] if errors else "raised")
    for v in values:
        if isinstance(v, str) and (v.startswith("BATCH_MOVED") or v.startswith("RELOAD-MOVED") or v.startswith("RLPAIR_MOVED")):
            return v, None
    if errors:
        return None, errors[0]
    return (values[0], None) if len(set(values)) == 1 else ("MOVED", None)


#: THE DECODE PART, by the name the harness gives it. It is one of
#: `identity_break.EXTRA_PARTS`, which the harness runs only behind
#: `--step-full` because a column is a maintainer artifact; here it always
#: runs, because a property a user cannot check is not a property they have.
STEPFULL = "stepfull"


def _capped(harness, text):
    """`text` through the harness's cap, which keeps the INNERMOST frames.
    A harness without one (an override pointed at an older file, a stub in a
    test) gets the text whole: too long beats a cause thrown away."""
    cap = getattr(harness, "_clip_error", None)
    return cap(text) if callable(cap) else text


def _error_text(harness, stage, exc):
    """The text a refused part carries: the stage, the exception TYPE, its
    message, and the traceback of the raise.

    NEVER A HEAD CUT (2026-09-19). This used to be
    `f"{type(exc).__name__}: {exc}"[:400]`, and on a traceback a head cut
    keeps the outermost frames -- the harness calling in, which the cell key
    already says -- and drops the frame that raised. A two-device MI300X run
    refused nine `par-queries-nn` batch cells and every one of them carried
    the same 300 characters of the worker's dispatch frame."""
    fmt = getattr(harness, "_exc_text", None)
    if callable(fmt):
        return _capped(harness, fmt(stage, exc))
    message = str(exc)
    head = f"{type(exc).__name__}: {message.splitlines()[0] if message else ''}"
    tail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip("\n")
    return _capped(harness, f"{stage + ': ' if stage else ''}{head}\n{tail}")


def _probe_stepfull(harness, fit, lane, ml, held):
    """(value, error) for the stepfull part of one fit.

    THE DECODE IS THE PREFILL. `step(x_t, state)` walked over a sequence must
    answer, at every position, the bits `forward(x)` answers over the whole of
    it from a zero state. The harness's own evaluator reports the FIRST
    differing position with both values, so a failure names a place rather
    than a count; the value here is that evaluator's, unchanged.

    A HARNESS WITHOUT THE PART IS AN ERROR, NEVER AN ABSENCE. Returning an
    `n/a` when `_probe_part` is missing would make this part read N/A on every
    lane on every install, which is a check that cannot fail. It refuses
    instead, which costs the run its VERIFIED and says why."""
    probe = getattr(harness, "_probe_part", None)
    if probe is None or STEPFULL not in (getattr(harness, "EXTRA_PARTS", None) or {}):
        return None, (f"stepfull: this harness ({getattr(harness, '__file__', 'unknown')}) defines no "
                      "stepfull part, so the decode property was not checked on this box. It is "
                      "in tools/identity_break.py since 2026-09-16; unset MOJOLEARN_IDENTITY_BREAK "
                      "or point it at a current harness.")
    value, err, _notes = probe(STEPFULL, fit, lane, ml, held.copy(), harness.BATCH_ALONE, "")
    return value, err


def run_cell(harness, ml, lane, fixture, data, held, repeats, extra_parts=()):
    """{part: (value, error)} for one cell, through the harness's own calls,
    in the harness's order (train, then infer and model, then batch, then
    stepfull). The order is load bearing: every part after the first runs on
    its OWN copy of the held-out rows and after the parts it must not move, so
    no hash a record already carries can change because a part was added."""
    X, yc, yr = data
    parts = tuple(vref.PARTS) + tuple(extra_parts)
    vals = {p: [] for p in parts}
    errs = {p: [] for p in parts}
    for repeat in range(repeats):
        harness._DUMP_TAG = f"{lane}/{fixture}/{repeat}"
        try:
            fit = harness.LANES[lane](ml, X, yc, yr, held.copy())
        except Exception as exc:
            # KEEP THE INNERMOST FRAMES, not the outermost (harness
            # `_exc_text`/`_clip_error`, 2026-09-19): a `[:400]` head cut
            # threw away the cause of nine refused par-queries-nn cells.
            text = _error_text(harness, None, exc)
            return {p: (None, text) for p in parts}
        vals["train"].append(harness._train_hash(fit))
        infer, model, reload, err = harness._probe_fit(fit, lane)
        if err:
            stage = err.split(":", 1)[0]
            err = _capped(harness, err)
            errs["infer" if stage == "infer" else "model"].append(err)
            if stage == "infer":
                errs["model"].append(err)
        if reload is not None and infer is not None and reload != infer:
            model = "RELOAD-MOVED"
        vals["infer"].append(infer)
        vals["model"].append(model)
        batch, berr = harness._probe_batch(fit, lane, ml, held.copy(), harness.BATCH_ALONE, False)
        if berr:
            errs["batch"].append(_capped(harness, berr))
        vals["batch"].append(batch)
        # the decode part last, for the same reason the batch part is not first
        step, serr = _probe_stepfull(harness, fit, lane, ml, held)
        if serr:
            errs[STEPFULL].append(_capped(harness, serr))
        vals[STEPFULL].append(step)
        for part in extra_parts:
            try:
                if part == "rlpair":
                    if not hasattr(harness, "RLPAIR") or not hasattr(harness, "_probe_rlpair"):
                        raise RuntimeError("this harness has no sampler/replay probe")
                    if lane not in harness.RLPAIR:
                        value, error = "n/a:no-sampler-trainer-pair", None
                    else:
                        value, error, _notes = harness._probe_rlpair(fit, lane, ml, held.copy(), False)
                else:
                    value, error, _notes = harness._probe_part(
                        part, fit, lane, ml, held.copy(), harness.BATCH_ALONE, "")
            except Exception as exc:
                value, error = None, _error_text(harness, part, exc)
            vals[part].append(value)
            if error:
                errs[part].append(error)
    return {p: _collapse(vals[p], errs[p]) for p in parts}


def run_models(harness, ml, table, pkg_dir=None, log=None, repeats=1, host_only=False):
    """The portable models: for each saved model the file's hash against the
    table's model reference, then the harness's batch part of the LOADED
    model against the table's batch reference. Returns result rows."""
    log = log or (lambda s: None)
    base = os.path.join(pkg_dir or _pkg_dir(), vref.TABLE_DIR, MODELS_DIR)
    manifest_path = os.path.join(base, MODELS_MANIFEST)
    def manifest_refusal(detail):
        return [dict(lane="portable:manifest", fixture="manifest", part="model", value=None, error=detail)]
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, ValueError) as exc:
        return manifest_refusal(f"portable model manifest unavailable: {exc}")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("models"), list) or not manifest["models"]:
        return manifest_refusal("portable model manifest has no model entries")
    rows = []
    held_cache = {}
    for m in manifest.get("models", []):
        lane, fixture = m["lane"], m["fixture"]
        path = os.path.join(base, m["file"])
        t0 = time.time()
        key = f"portable:{lane}"
        try:
            with open(path, "rb") as fh:
                file_hash = hashlib.sha256(fh.read()).hexdigest()[:16]
        except OSError as exc:
            rows.append(dict(lane=key, fixture=fixture, part="file", value=None, error=str(exc)))
            continue
        rows.append(dict(lane=key, fixture=fixture, part="model", value=file_hash, error=None,
                         reference_part=("model", lane)))
        values, errors = [], []
        for _ in range(max(1, repeats)):
            try:
                # --models-only explicitly exercises the CPU saved-model door,
                # even when the process also has GPU bindings.
                if host_only or ml.vendor() == "cpu":
                    est = ml.host_model(path)
                else:
                    est = getattr(getattr(ml, m["class"]), m.get("load", "load"))(path)
                if fixture not in held_cache:
                    held_cache[fixture] = harness.heldout(fixture)
                fit = harness.Fit({})
                fit.est = est
                batch, berr = harness._probe_batch(fit, lane, ml, held_cache[fixture].copy(), harness.BATCH_ALONE, False)
            except Exception as exc:
                batch, berr = None, _error_text(harness, None, exc)
            values.append(batch)
            if berr:
                errors.append(berr)
        batch, berr = _collapse(values, errors)
        rows.append(dict(lane=key, fixture=fixture, part="batch", value=batch, error=berr,
                         reference_part=("batch", lane)))
        log(f"  {key:<34} {fixture:<8} {time.time() - t0:6.1f}s")
    return rows


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------

def _device_block(ml, harness):
    from . import _verify
    from . import _backend
    from . import _identity
    commit, commit_source = _identity.commit_witness()
    if not commit:
        commit = _verify._git_commit(_pkg_dir())
        commit_source = "git checkout beside the package" if commit else "none (no witness in this install)"
    vendor = _backend.vendor()
    return dict(
        mojolearn_version=getattr(ml, "__version__", "unknown"),
        commit=commit, commit_source=commit_source,
        numeric_mode=_backend.numeric_mode(),
        vendor=vendor, device_class=vref.VENDOR_CLASS.get(vendor),
        device=_verify.describe_device(), cpu_model=harness.cpu_model(),
        requested_parallel_devices=list(harness._par_devices()),
        verifier_cpu_threads=os.environ.get('MOJOLEARN_VERIFY_CPU_THREADS'),
        platform=platform.platform(), python=platform.python_version(),
        numpy=__import__("numpy").__version__,
    )


def summarize(rows, families):
    """Per family counts over judged rows, in first-seen order."""
    order, table = [], {}
    for r in rows:
        fam = families.get(r["lane"], "portable models" if r["lane"].startswith("portable:") else "other")
        if fam not in table:
            order.append(fam)
            table[fam] = {s: 0 for s in vref.STATES}
            table[fam]["lanes"] = set()
        table[fam][r["state"]] += 1
        table[fam]["lanes"].add(r["lane"])
    return [(fam, table[fam]) for fam in order]


def verdict(counts, scope_gaps=()):
    """(exit code, headline) from the state counts of every judged part.

    A REFUSED PART DID NOT RUN, so it is never evidence of success, and the
    parts that did run do not make up for it. Until 2026-09-16 one IDENTICAL
    part outranked any number of REFUSED ones, so a CPU-only install with
    stale bindings printed `VERIFIED, exit 0` on 44 IDENTICAL and 288 REFUSED
    parts: the user had checked 13 percent of what they believed they checked
    (lane/expose-inference-surface). There is no threshold below which a part
    that did not run counts as checked, so ANY refusal makes the run
    INCOMPLETE and exits non-zero, and only a run with nothing refused may
    print VERIFIED. A wrong answer still outranks an absent one, so DIVERGENT
    is still read first.
    """
    if counts[vref.DIVERGENT]:
        return EXIT_MISMATCH, "MISMATCH"
    if counts[vref.REFUSED]:
        return EXIT_CANNOT_RUN, "INCOMPLETE"
    if counts[vref.OWED] or scope_gaps:
        return EXIT_NO_REFERENCE, "INCOMPLETE" if counts[vref.IDENTICAL] else "NO REFERENCE"
    if counts[vref.IDENTICAL]:
        return EXIT_VERIFIED, "VERIFIED"
    return EXIT_NO_REFERENCE, "NO REFERENCE"


#: `verify --self-test`: the lane it perturbs, and the fixture it uses. `ols`
#: is chosen because it is cheap (well under a second), it is a public
#: reference lane on every install, and the shipped table carries a real train
#: reference for it on `base`.
SELF_TEST_LANE = "ols"
SELF_TEST_FIXTURE = "base"


def self_test(harness, ml, table, log=None):
    """CAN THIS INSTALLATION'S VERIFIER ACTUALLY FAIL? (lane/expose-inference-surface,
    2026-09-16.)

    A user who runs `verify` and reads VERIFIED is trusting two things they
    cannot see: that we wrote an honest table, and that the comparison is real.
    A check that can only pass is the defect we kept finding in our own code,
    and it had no business sitting in the public command. This lets the user
    watch the comparison catch a wrong answer, here, on their machine.

    HOW. One lane is run TWICE through the ordinary path, `run_cell` then
    `judge_rows` then `_verify_reference.judge`, the same code every real lane
    goes through. Nothing about the verdict is simulated:

      clean      the fixture untouched                  -> must read IDENTICAL
      perturbed  column 0 moved up by ONE ULP per value -> must read DIVERGENT

    The perturbation is real arithmetic at run time on the input array, so it
    needs no sabotage build and no second binding: the lane computes correctly
    over an input that differs in its last bits, which is genuinely different
    bytes out, and the ordinary comparison is what notices.

    THE SIZE OF THE PERTURBATION IS MEASURED, NOT ASSUMED. The first version of
    this perturbed a SINGLE value, `X[0, 0]`, by one ULP, and the lane's hash
    did not move: `ols` fits 20,000 x 16 and one last-bit change in one of
    320,000 inputs does not reach the rounded coefficients. That self-test
    could not have failed, which is the exact defect it exists to catch, and
    the two arms caught it on the first run rather than passing quietly.
    Measured on the base fixture: `X[0,0]` one ULP is INERT, `y[0]` one ULP is
    INERT, and `X[:, 0]` one ULP moves the hash (3d1d7c30b12d9872 ->
    dd42c9b607526efe). The smallest perturbation shown to move it is the one
    used; if a future change makes it inert again, the perturbed arm reads
    IDENTICAL and this self-test fails rather than lying.

    WHY BOTH ARMS. One arm alone could pass while broken. A comparator stuck on
    IDENTICAL passes the clean arm and fails the perturbed one; a comparator
    stuck on DIVERGENT does the reverse; a table of nonsense fails the clean
    arm. Only a comparison that actually discriminates passes both, which is
    the property being demonstrated.

    WHAT IT DOES NOT SAY. It exercises one lane on one fixture. It does not
    re-measure any vendor, and it is not evidence about the other lanes'
    arithmetic; it is evidence about the MACHINERY that judges them.
    """
    import numpy as np
    log = log or (lambda s: None)
    lane, fixture = SELF_TEST_LANE, SELF_TEST_FIXTURE
    if lane not in harness.LANES:
        raise CannotRun(f"self-test: this harness does not define the {lane!r} lane")
    ent = vref.entry(table, lane, fixture, "train")
    if ent is None or not isinstance(ent.get("ref"), str) or ent["ref"].startswith("n/a"):
        raise CannotRun(f"self-test: the shipped table has no train reference for {lane}/{fixture}, "
                        "so there is nothing for the comparison to disagree with")

    X, yc, yr = harness.fixture(fixture)
    held = harness.heldout(fixture)

    # The perturbation, reported so a reader can repeat it: every value of
    # column 0 moved up by one ULP. Column-wide because a single value was
    # MEASURED to be inert for this lane (see the docstring).
    Xp = np.array(X, copy=True)
    before = float(Xp[0, 0])
    Xp[:, 0] = np.nextafter(Xp[:, 0], np.asarray(np.inf, dtype=Xp.dtype))
    after = float(Xp[0, 0])
    if after == before or not (Xp != X).any():
        raise CannotRun("self-test: the one-ULP perturbation did not change the input at all, so "
                        "the perturbed arm would be identical to the clean one and prove nothing")

    from ._cpu_reference import reference_training
    rows = []
    with reference_training():
        for arm, data in (("clean", (X, yc, yr)), ("perturbed", (Xp, yc, yr))):
            parts = run_cell(harness, ml, lane, fixture, data, held, 1)
            value, error = parts["train"]
            rows.append(dict(lane=lane, fixture=fixture, part="train", value=value, error=error, arm=arm))
            log(f"  {arm:<10} {lane}/{fixture} train -> {value}")

    judged = judge_rows([{k: v for k, v in r.items() if k != "arm"} for r in rows], table)
    for r, j in zip(rows, judged):
        j["arm"] = r["arm"]
    clean = next(j for j in judged if j["arm"] == "clean")
    dirty = next(j for j in judged if j["arm"] == "perturbed")

    problems = []
    if clean["state"] != vref.IDENTICAL:
        problems.append(f"the UNTOUCHED input read {clean['state']}, not IDENTICAL "
                        f"({clean['detail'] or 'no detail'}); this installation does not reproduce "
                        "the reference at all, so the perturbed arm proves nothing")
    if dirty["state"] != vref.DIVERGENT:
        problems.append(f"the PERTURBED input read {dirty['state']}, not DIVERGENT; the comparison "
                        "did not notice a wrong answer, which means a VERIFIED from this "
                        "installation is worth nothing")
    return dict(lane=lane, fixture=fixture,
                perturbation=dict(cell="X[:, 0]", before=before, after=after,
                                  kind="one ULP up, every value of column 0",
                                  values_changed=int((Xp != X).sum())),
                clean=dict(state=clean["state"], value=clean["value"], reference=clean["reference"]),
                perturbed=dict(state=dirty["state"], value=dirty["value"], reference=dirty["reference"]),
                passed=not problems, problems=problems)


def format_self_test(r):
    lines = ["# python -m mojolearn verify --self-test", ""]
    lines.append(f"Ran the {r['lane']}/{r['fixture']} lane twice through the ordinary comparison, "
                 "once untouched and")
    p = r["perturbation"]
    lines.append(f"once with {p['cell']} moved {p['kind']}: {p.get('values_changed', '?')} values, "
                 f"the first {p['before']!r} -> {p['after']!r}.")
    lines.append("")
    lines.append(f"  untouched   {r['clean']['value']}   vs reference {r['clean']['reference']}   "
                 f"-> {r['clean']['state']}")
    lines.append(f"  perturbed   {r['perturbed']['value']}   vs reference {r['perturbed']['reference']}   "
                 f"-> {r['perturbed']['state']}")
    lines.append("")
    if r["passed"]:
        lines.append("RESULT: THE VERIFIER WORKS ON THIS MACHINE. The comparison reproduced the")
        lines.append("recorded bytes on the untouched input and DETECTED A WRONG ANSWER on an input")
        lines.append("differing by one bit, in this installation, using the same code path that")
        lines.append("judges every other lane. A VERIFIED from this install is therefore a result")
        lines.append("that could have failed.")
        lines.append("")
        lines.append("It does not re-measure any vendor, and it speaks for the machinery, not for")
        lines.append("the arithmetic of the other lanes.")
    else:
        lines.append("RESULT: THE VERIFIER IS NOT TRUSTWORTHY ON THIS MACHINE.")
        for p in r["problems"]:
            lines.append(f"  - {p}")
    return "\n".join(lines)


def _cmd_self_test(args, ml):
    json_out = getattr(args, "json", False)
    log = (lambda s: _emit(s, sys.stderr)) if json_out else _emit
    try:
        table = vref.load_table(getattr(args, "reference_table", None) or vref.table_path())
    except vref.TableError as exc:
        return _finish(args, EXIT_NO_REFERENCE, "NO REFERENCE", str(exc))
    try:
        harness = load_harness()
        result = self_test(harness, ml, table, log=log)
    except (FileNotFoundError, CannotRun) as exc:
        return _finish(args, EXIT_CANNOT_RUN, "CANNOT RUN", str(exc))
    if json_out:
        _emit(json.dumps(dict(format="mojolearn.verify-self-test.v1", **result), indent=1, sort_keys=True))
    else:
        _emit(format_self_test(result))
    return EXIT_VERIFIED if result["passed"] else EXIT_MISMATCH


#: One Apple Metal process may run at most this many lanes before
#: `identity_break.refuse_routine_apple_column` refuses it: a full column is a
#: per-release artifact, not something a routine command takes. The cross-check
#: respects it rather than tripping over it, so the default is capped here.
APPLE_LANE_CAP = 24


def cross_check_lanes(harness, scope="default"):
    """The lanes a GPU-against-CPU cross-check runs, and what it leaves out.

    The intersection is every registered lane with both a GPU path and a
    host inference route reachable from the bindings shipped in the wheel.
    Derive this set from the manifest so newly exposed variants are included.

    Separate scopes keep routine verification bounded:

      quick    one lane per family, base fixture. Seconds, for someone in a
               hurry or wiring this into CI.
      default  up to APPLE_LANE_CAP lanes, base fixture. Minutes. Capped
               because one Apple Metal process may not run more than that
               without naming a release, which is enforced in identity_break
               rather than advisory.
      all      the whole intersection. On Apple this is REFUSED by that same
               rule, deliberately; on NVIDIA and AMD it runs.
    """
    hs = host_surface()
    shipped = set(hs.wheel_bindings())
    routes = hs.inference_routes()
    per_family, every = {}, []
    for f in hs.FAMILIES:
        reachable = f["ships_in_wheel"] or routes.get(f["routes"]) in shipped
        for lane in f["inference_lanes"]:
            if not reachable or lane not in harness.LANES:
                continue
            every.append(lane)
            per_family.setdefault(f["family"], lane)
    every = sorted(set(every))
    representative = sorted(set(per_family.values()))
    if scope == "quick":
        chosen = representative
    elif scope == "all":
        chosen = every
    else:
        # the representatives first, so every family is covered even if the cap
        # bites, then fill up to the cap in lane order
        chosen = list(representative)
        for lane in every:
            if len(chosen) >= APPLE_LANE_CAP:
                break
            if lane not in chosen:
                chosen.append(lane)
        chosen = sorted(chosen[:APPLE_LANE_CAP])
    return chosen, every, dict(per_family)


def cross_check(harness, ml, lanes, fixtures, log=None):
    """THEIR GPU AGAINST THEIR CPU, on the user's own machine.

    The strongest of the three checks, because it requires trusting NOBODY.
    Comparing against our shipped table asks the user to believe we recorded
    honestly; this asks them to believe nothing. They generate both sides
    themselves, on two genuinely different pieces of hardware in their own
    box, and what it demonstrates is exactly the claim: the same computation
    yields the same bits on different hardware. It also works on any GPU we
    support, not only the three vendors we happened to record, which answers
    "I do not own an H100".

    ONE DIGEST, SO THE TWO SIDES CANNOT DRIFT. The lane is fitted ONCE on the
    GPU. `fit.probe(gpu_estimator)` gives one answer and
    `fit.probe(host_model(saved))` the other, both hashed by the harness's own
    `_h`. There is no second comparison path and no second digest to disagree
    with the first; the only difference between the arms is which binding
    answers.

    AND THE BINDING THAT ANSWERED IS THE ONE NAMED. `_probe_fit_host` learned
    this the hard way: a forest sabotage column once read IDENTICAL on a
    RunPod pod because the binding actually loaded was not the one asked for,
    so the check was comparing a thing against itself. The same guard is kept
    here.
    """
    import tempfile
    from . import _backend
    log = log or (lambda s: None)
    vendor = _backend.vendor()
    if vendor == "cpu":
        return dict(ran=False, vendor=vendor, lanes=[],
                    reason=("no GPU in this installation, so there is no second piece of hardware "
                            "to compare against. This is not a pass and not a failure: the "
                            "cross-check did not run. `verify --all` still checks this machine "
                            "against the recorded columns, and `--self-test` still shows the "
                            "comparison can fail."))
    from ._forest_host import host_model, binary_path
    rows, agree, differ, skipped = [], 0, 0, {}
    for lane in lanes:
        for fx in fixtures:
            t0 = time.time()
            X, yc, yr = harness.fixture(fx)
            held = harness.heldout(fx)
            try:
                fit = harness.LANES[lane](ml, X, yc, yr, held.copy())
            except Exception as exc:
                skipped[lane] = f"the lane raised: {type(exc).__name__}: {exc}"[:200]
                continue
            if not callable(fit.probe):
                skipped[lane] = f"no out-of-sample probe ({fit.probe})"
                continue
            sl = harness._save_load(fit.est)
            if sl is None:
                skipped[lane] = "the estimator has no save/load, so the CPU side has nothing to load"
                continue
            save, _load, suffix = sl
            try:
                gpu_est = harness._public_est(lane, fit.est)
                pair = {"infer": [harness._h(*fit.probe(gpu_est)), None]}
                # THE BATCH PART TOO, where the lane has one. Batch invariance is
                # a DIFFERENT AXIS from cross-vendor identity: cross-vendor asks
                # "same input, different hardware, same bits", batch invariance
                # asks "same row, different batch neighbours, same bits". A
                # serving system batches dynamically, so a user whose prediction
                # changes with traffic has a real problem, and this is the one
                # place they can test both at once on their own machine. A lane
                # that declares `n/a` for batch keeps its n/a; inventing a
                # comparison there would be a check that cannot fail.
                gb, gerr = harness._probe_batch(fit, lane, ml, held.copy(), harness.BATCH_ALONE, False)
                if gerr is None and isinstance(gb, str) and not gb.startswith("n/a"):
                    pair["batch"] = [gb, None]
                elif isinstance(gb, str) and gb.startswith("n/a"):
                    pair["batch"] = [gb, gb]        # declared absent on both sides
                with tempfile.TemporaryDirectory(prefix="mojolearn_cross_") as tmp:
                    path = os.path.join(tmp, f"{lane}{suffix}")
                    getattr(fit.est, save)(path)
                    host = host_model(path)
                    bound = getattr(getattr(host, "_binding", None), "__file__", None)
                    if bound is not None and type(host).__name__ in ("HostGBDT", "HostForest") and (
                            os.path.realpath(bound) != os.path.realpath(binary_path())):
                        raise RuntimeError(f"the host binding that answered is {bound}, not "
                                           f"{binary_path()}; this would compare a thing to itself")
                    pair["infer"][1] = harness._h(*fit.probe(host))
                    if "batch" in pair and pair["batch"][1] is None:
                        hfit = harness.Fit({})
                        hfit.est = host
                        hfit.probe = fit.probe
                        hb, herr = harness._probe_batch(hfit, lane, ml, held.copy(),
                                                        harness.BATCH_ALONE, False)
                        pair["batch"][1] = hb if herr is None else f"raised: {herr}"[:120]
            except Exception as exc:
                skipped[lane] = f"{type(exc).__name__}: {exc}"[:200]
                continue
            secs = round(time.time() - t0, 3)
            verdicts = []
            for part, (g, c) in sorted(pair.items()):
                na = isinstance(g, str) and g.startswith("n/a")
                same = (g == c)
                if not na:
                    agree += same
                    differ += (not same)
                rows.append(dict(lane=lane, fixture=fx, part=part, gpu=g, cpu=c,
                                 agree=None if na else same, na=na, seconds=secs))
                verdicts.append(f"{part}={'n/a' if na else ('agree' if same else 'DIFFER')}")
            # streamed per lane: it shows the run is alive, and it is itself
            # evidence that work happened. A verification that returns instantly
            # invites the suspicion this whole command exists to remove.
            log(f"  {lane:<26} {fx:<8} {' '.join(verdicts):<28} {secs:6.2f}s")
    # NOTHING COMPARED IS NOT A MISMATCH. The first run of this printed
    # "MISMATCH. 0 of 0 cells differ", which is self-contradictory: no cell was
    # compared, so nothing differed and nothing agreed. Reporting a result
    # about hashes that were never computed is the same defect as VERIFIED over
    # a refused run (lane/verify-cross-check, 2026-09-16).
    return dict(ran=True, vendor=vendor, device_class=vref.VENDOR_CLASS.get(vendor),
                lanes=sorted({r['lane'] for r in rows}), fixtures=list(fixtures),
                compared=len(rows), agree=agree, differ=differ,
                skipped=skipped, cells=rows,
                passed=(differ == 0) if rows else None)


def format_cross_check(r):
    lines = ["# python -m mojolearn verify --cross-check", ""]
    if not r["ran"]:
        lines.append("NOT RUN: " + r["reason"])
        return "\n".join(lines)
    lines.append(f"Fitted each lane ONCE on this machine's GPU ({r['vendor']}), then asked the same")
    lines.append("fitted model for the same held-out answer twice: from the GPU estimator, and from")
    lines.append("the saved model reloaded through the CPU host binding. Same digest both times.")
    lines.append("")
    lines.append("`infer` is cross-vendor identity: same input, different hardware, same bits.")
    lines.append("`batch` is batch invariance: same row, different batch neighbours, same bits --")
    lines.append("a different axis, and the one that bites a serving system batching dynamically.")
    lines.append("")
    for c in r["cells"]:
        if c.get("na"):
            lines.append(f"  {c['lane']:<26} {c['fixture']:<8} {c['part']:<6} n/a  {c['gpu']}")
            continue
        lines.append(f"  {c['lane']:<26} {c['fixture']:<8} {c['part']:<6} "
                     f"gpu={c['gpu']}  cpu={c['cpu']}  {'agree' if c['agree'] else 'DIFFER'}")
    lines.append("")
    if r["skipped"]:
        lines.append(f"Not compared ({len(r['skipped'])}):")
        for lane, why in sorted(r["skipped"].items()):
            lines.append(f"  {lane}: {why}")
        lines.append("")
    if r["passed"] is None:
        lines.append("RESULT: NOTHING COMPARED. No lane produced a GPU answer and a CPU answer, so")
        lines.append("there is nothing to agree or disagree about. This is not a pass and not a")
        lines.append("failure; the reasons are listed above.")
    elif r["passed"]:
        asked = len(r["lanes"]) + len(r["skipped"])
        lines.append(f"RESULT: YOUR GPU AND YOUR CPU AGREE on {r['agree']} of {r['agree']} compared "
                     f"cell parts,")
        # THE SKIPPED COUNT RIDES WITH THE VERDICT. A confident headline over a
        # run that compared 5 of 24 lanes is a pass whose scope is much smaller
        # than it looks, which is the failure this command exists to remove.
        lines.append(f"across {len(r['lanes'])} of {asked} lanes"
                     + (f" ({len(r['skipped'])} SKIPPED, listed above)" if r["skipped"] else "")
                     + f" and {len(r['fixtures'])} fixture(s), in {r.get('elapsed_s', 0):.1f}s.")
        if r["skipped"]:
            lines.append("A skipped lane was NOT checked. This agreement covers only the lanes")
            lines.append("named above, and says nothing about the ones that did not run.")
        lines.append("You generated both sides on two different pieces of hardware in this machine,")
        lines.append("so this result depends on trusting nobody: not our recorded columns, not us.")
    else:
        lines.append(f"RESULT: MISMATCH. {r['differ']} of {r['compared']} cells differ between this")
        lines.append("machine's GPU and its CPU. The differing hashes are above and in --json.")
    return "\n".join(lines)


def _cmd_cross_check(args, ml):
    """`verify --cross-check [fast|all]`: this machine's GPU against its CPU."""
    json_out = getattr(args, "json", False)
    log = (lambda s: _emit(s, sys.stderr)) if json_out else _emit
    scope = getattr(args, "cross_check", None) or "default"
    started = time.time()
    try:
        harness = load_harness()
    except (FileNotFoundError, CannotRun) as exc:
        return _finish(args, EXIT_CANNOT_RUN, "CANNOT RUN", str(exc))

    chosen, every, per_family = cross_check_lanes(harness, scope)
    asked = [x for x in (getattr(args, "lanes", "") or "").split(",") if x]
    if asked:
        unknown = [l for l in asked if l not in every]
        if unknown:
            _emit(f"USAGE: --lanes names lanes outside the cross-check intersection: {unknown}; "
                  f"the intersection is {len(every)} lanes", sys.stderr)
            return EXIT_USAGE
        chosen = [l for l in every if l in asked]
    fixtures = [x for x in (getattr(args, "fixtures", "") or "").split(",") if x] or ["base"]
    bad = [f for f in fixtures if f not in harness.FIXTURES]
    if bad:
        _emit(f"USAGE: --fixtures names fixtures the harness does not define: {bad}", sys.stderr)
        return EXIT_USAGE

    log(f"# cross-check ({scope}): {len(chosen)} of {len(every)} intersection lanes x "
        f"{len(fixtures)} fixture(s)")
    result = cross_check(harness, ml, chosen, fixtures, log=log)
    result.update(scope=scope, intersection=len(every), intersection_lanes=every,
                  representative_of=per_family, elapsed_s=round(time.time() - started, 2),
                  apple_lane_cap=APPLE_LANE_CAP)
    if json_out:
        _emit(json.dumps(dict(format="mojolearn.verify-cross-check.v1", **result),
                         indent=1, sort_keys=True))
    else:
        _emit(format_cross_check(result))
    if not result["ran"] or result["passed"] is None:
        # no GPU, or no lane produced both answers: nothing was compared, which
        # is CANNOT RUN rather than a verdict about hashes never computed
        return EXIT_CANNOT_RUN
    return EXIT_VERIFIED if result["passed"] else EXIT_MISMATCH


#: The evidence document `--compare` reads. A file that does not announce
#: itself as one is REFUSED BY NAME rather than compared on a guess. This
#: command exists to be used adversarially, and quietly accepting any JSON
#: that happens to carry a `cells` key is the first step toward a comparer
#: that always agrees: two files with no cells at all would "not disagree".
COMPARE_INPUT_FORMAT = "mojolearn.verify-all-report.v1"

#: What a real cell value looks like: a hash this box actually computed. The
#: harness writes truncated lowercase sha256, so anything outside this shape
#: is NOT a bit pattern and must never be compared as though it were one.
_HASH_RE = re.compile(r"\A[0-9a-f]{8,64}\Z")

#: Values that mean THE BOX DISAGREED WITH ITSELF. `vref.judge` reads all
#: three as DIVERGENT. Two documents both carrying `MOVED` for a cell hold
#: the same STRING, and equality on strings counted that as agreement: two
#: machines "agreeing" that neither of them is deterministic, reported as a
#: pass. That is the precise opposite of the claim being checked, so it gets
#: its own verdict and its own non-zero exit.
_SELF_CONTRADICTED = ("MOVED", "BATCH_MOVED", "RELOAD-MOVED", "RLPAIR_MOVED")


def _value_kind(v):
    """Classify one cell value. The whole point is that only `hash` is a bit
    pattern two machines can agree ON; the rest are agreements about nothing."""
    if not isinstance(v, str) or not v:
        return "missing"                       # None where the probe raised
    if v == "MOVED" or v.startswith("BATCH_MOVED") or v.startswith("RELOAD-MOVED") or v.startswith("RLPAIR_MOVED"):
        return "moved"
    if v.startswith("n/a"):
        return "n/a"
    if _HASH_RE.match(v):
        return "hash"
    return "missing"                           # unrecognized: never a match


def _read_cells(doc, label):
    """`{(lane, fixture, part): value}` plus every structural complaint.

    DUPLICATE KEYS ARE A COMPLAINT, NOT A LAST-WINS. The obvious attack on a
    comparer is to append a second row for a cell, copied from the other
    party's document, so the dict build overwrites the honest answer and the
    mismatch disappears. A document with two rows for one cell part is not a
    document this command will read.
    """
    problems, out, states, seen = [], {}, {}, {}
    if not isinstance(doc, dict):
        return out, states, [f"{label}: not an evidence document (top level is "
                             f"{type(doc).__name__}, expected an object)"]
    fmt = doc.get("format")
    if fmt != COMPARE_INPUT_FORMAT:
        problems.append(f"{label}: format is {fmt!r}, expected {COMPARE_INPUT_FORMAT!r}. "
                        "Write it with `verify --all --json-out <path>`")
    rows = doc.get("cells")
    if rows is None:
        problems.append(f"{label}: has no `cells`, so there is nothing to compare")
        rows = []
    if not isinstance(rows, list):
        return out, states, problems + [f"{label}: `cells` is {type(rows).__name__}, expected a list"]
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            problems.append(f"{label}: cells[{i}] is {type(r).__name__}, expected an object")
            continue
        key = (r.get("lane"), r.get("fixture"), r.get("part"))
        if key in seen:
            problems.append(f"{label}: {key[0]}/{key[1]} {key[2]} appears twice "
                            f"(rows {seen[key]} and {i}, values {out[key]!r} and {r.get('value')!r})")
            continue
        seen[key] = i
        out[key] = r.get("value")
        states[key] = r.get("state")
    return out, states, problems


def comparison_context_problems(a, b, keys):
    """Hash agreement is meaningful only under the same recorded input contract."""
    if not keys:
        return []
    ac, bc = a.get('verification_contract'), b.get('verification_contract')
    if not isinstance(ac, dict) or not isinstance(bc, dict):
        return ['missing verification_contract; rerun verification with a current wheel']
    problems = []
    if not ac.get('harness_sha256') or ac.get('harness_sha256') != bc.get('harness_sha256'):
        problems.append('different or missing harness digest')
    for lane, fixture, part in keys:
        for field in ('fixtures', 'heldout'):
            av, bv = ac.get(field, {}).get(fixture), bc.get(field, {}).get(fixture)
            if not av or av != bv:
                problems.append(f'{fixture}: different or missing {field} fingerprints')
        if part not in ('train', 'infer', 'model', 'file'):
            av, bv = ac.get('protocols', {}).get(part), bc.get('protocols', {}).get(part)
            if not av or av != bv:
                problems.append(f'{part}: different or missing protocol')
    return sorted(set(problems))


def verification_contract(harness, harness_file, data, held, extra_parts):
    protocols = dict(batch=dict(alone=harness.BATCH_ALONE,
        split=list(harness.BATCH_SPLIT) + ['n'], prefix='1,7,full-1', enabled=True))
    for part in ('stepfull', *extra_parts):
        protocols[part] = (harness._rlpair_protocol() if part == 'rlpair'
                           else harness._part_protocol(part, harness.BATCH_ALONE))
    return dict(harness_sha256=vref.sha256_file(harness_file),
                fixtures={f: dict(zip(('X', 'y_clf', 'y_reg'), map(harness._h, values)))
                          for f, values in data.items()},
                heldout={f: dict(X=harness._h(x)) for f, x in held.items()},
                protocols=protocols)


# ----------------------------------------------------------- commit-reveal
#
# THE HOLE EVERY STRUCTURAL DEFENSE ABOVE LEAVES OPEN. `_read_cells`,
# `_value_kind` and `compare_documents` all police documents that are
# malformed or that contradict themselves. None of them says anything about a
# perfectly well formed document whose numbers were not computed by the
# machine it names. Whichever party receives the other's file FIRST can paste
# its cell values into a document carrying their own provenance, and the
# comparer will read a clean AGREE across two "independent" vendors. That is
# the exact scenario this feature exists for -- us wanting to show outside
# corroboration -- and the exact question a reader is right to ask: how do
# they know we did not manufacture both sides?
#
# Commit-reveal closes it and costs nothing. Before either party sees the
# other's file, each publishes a hash of their own document under a random
# nonce they keep back. Only then do they exchange documents, nonces included.
# A party who has published a commitment cannot copy, because what they are
# bound to was fixed before there was anything to copy from.

COMMITMENT_FORMAT = "mojolearn.verify-commitment.v1"

#: Domain separation. A commitment digest must never be confusable with any
#: other sha256 this project prints -- a cell hash, a binding digest, the
#: harness digest, a table digest. Prefixing the preimage with a string that
#: appears nowhere else means a value lifted out of one context cannot be
#: replayed as the other.
_COMMITMENT_DOMAIN = b"mojolearn.verify-commitment.v1\n"

#: Where a sealed document carries its nonce and its own copy of the
#: commitment. EXCLUDED from the preimage, necessarily: a hash cannot cover
#: itself, and the nonce is the one field that must differ between two honest
#: documents.
REVEAL_KEY = "commitment_reveal"

#: 128 bits. The nonce is not a key and guards no secret; it exists so that
#: publishing the commitment does not publish the document. Without it the
#: preimage space is guessable -- our shipped reference table pins every
#: expected cell hash, and a device block is a handful of short strings -- so
#: a bare hash of the document would let the party who receives it first
#: brute-force its content before revealing their own. That is the very
#: asymmetry commit-reveal removes, so removing it again for tidiness would
#: be self-defeating.
NONCE_BYTES = 16

#: What the commitment covers, in the order `commitment_preimage` builds it.
#: Named here so a test can hold this list against the fields
#: `compare_documents` actually reads, rather than a reader holding them
#: against each other by eye.
COMMITMENT_COVERS = ("format", "cells", "device", "verification_contract",
                     "bindings", "verdict", "detail")

#: Fields deliberately left OUT, with the reason, because "why is this not
#: covered" is the question a later lane will ask.
COMMITMENT_EXCLUDES = {
    "elapsed_s": "wall clock; nobody compares it and it would bind a party to a stopwatch",
    "lane_seconds": "wall clock, per lane; same reason",
    REVEAL_KEY: "the nonce and the commitment itself; a hash cannot cover itself",
}


def commitment_preimage(doc):
    """The canonical bytes a commitment is taken over.

    WHAT IT COVERS IS THE WHOLE DESIGN, so the reason lives here, in the code
    a change has to walk past, and not in a document the next lane overrides.

    NOT THE CELLS ALONE. A commitment over cell values only would let a party
    commit to numbers and then swap the `device` block, claiming an M2
    produced what a 4090 did. The provenance IS the claim; leaving it out
    commits a forger to precisely the half they never needed to change.

    NOT THE FILE BYTES. Hashing the file defeats itself. Re-indenting,
    reordering keys, a different JSON writer, a round trip through any tool
    would all break an honest document's commitment, and a check that fires
    on innocent handling is a check people learn to click past. This hashes
    PARSED values re-serialized canonically, so formatting is invisible and
    only content moves the digest.

    NOT THE WHOLE PARSED DOCUMENT EITHER. `elapsed_s` and `lane_seconds` are
    wall clock. Covering them would bind a party to timings no comparison
    reads, which is the same false-alarm failure one step in.

    THE RULE, and it is checkable rather than tasteful: cover exactly the
    fields `compare_documents` reads.
      * a field the comparer reads that the commitment omits is a field a
        party may still change after seeing the other document -- the hole
        left open;
      * a field the commitment covers that the comparer never reads is a
        false alarm waiting to be normalized.
    `python/mojolearn/tests/test_verify_compare_commitment.py` mutates each
    covered field in turn and requires the digest to move, and mutates each
    excluded field and requires it not to, so a later lane that teaches
    `compare_documents` to read a new field and forgets this function is
    caught by a test rather than by an adversary.
    """
    if not isinstance(doc, dict):
        raise ValueError(f"not an evidence document (top level is {type(doc).__name__})")
    cells = []
    for r in (doc.get("cells") or []):
        cells.append([r.get("lane"), r.get("fixture"), r.get("part"), r.get("value"),
                      r.get("state")] if isinstance(r, dict) else r)
    # Sorted so that reordering rows is not a change, and DUPLICATES KEPT:
    # `_read_cells` refuses a document that names one cell part twice, and the
    # commitment must bind a party to the document that refusal describes, not
    # to a tidied-up one they never handed over.
    cells.sort(key=lambda c: json.dumps(c, sort_keys=True, default=str))
    covered = dict(
        format=doc.get("format"),
        cells=cells,
        # the WHOLE device block, not the eight keys printed side by side.
        # `numeric_mode` is in there: a party who ran under `fast` and edited
        # that one word afterwards would otherwise be committing to nothing
        # about the tier their numbers came from.
        device=doc.get("device"),
        # hash agreement is only ever READ under an equal contract
        # (`comparison_context_problems`), so a party who could swap the
        # contract after the exchange could turn an INCOMPARABLE into an
        # AGREE. It carries no wall clock, so covering it whole is safe.
        verification_contract=doc.get("verification_contract"),
        # THE BINDING DIGESTS, deliberately. `binding_artifacts()` enumerates
        # loaded modules from sys.modules and then re-reads the file from disk
        # to hash it, so there is a window between load and hash in which the
        # file could be swapped. That window is a limit on what the provenance
        # block MEASURES, and no commitment can upgrade a self-reported field
        # into a measurement. It is not a reason to leave the list out: the
        # comparer reads and prints it, and covering it stops a party editing
        # their claimed binary after seeing which binary the other party used.
        bindings=[b.get("sha256") if isinstance(b, dict) else b
                  for b in (doc.get("bindings") or [])],
        # each document's own verdict and detail are printed beside the
        # comparison precisely so a reader can see that one side checked a
        # fraction of what they assumed. Uncovered, that line is editable.
        verdict=doc.get("verdict"),
        detail=doc.get("detail"),
    )
    assert tuple(covered) == COMMITMENT_COVERS, "COMMITMENT_COVERS no longer names what is covered"
    return json.dumps(covered, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, default=str).encode("utf-8")


def commitment_digest(doc, nonce):
    """sha256 over domain, nonce and preimage, in that order.

    The nonce goes BEFORE the document bytes. Nothing here is vulnerable to a
    length extension, but a prefix construction cannot become vulnerable to
    one later either, and the ordering costs nothing to get right now."""
    if not isinstance(nonce, str) or not re.match(r"\A[0-9a-f]{16,128}\Z", nonce):
        raise ValueError("a nonce is 16 to 128 lowercase hex characters")
    h = hashlib.sha256()
    h.update(_COMMITMENT_DOMAIN)
    h.update(nonce.encode("ascii"))
    h.update(b"\n")
    h.update(commitment_preimage(doc))
    return h.hexdigest()


def seal_document(doc, nonce=None):
    """Attach a fresh nonce and its commitment to `doc`, in place, and return
    the commitment. THE COMMITMENT IS WHAT YOU PUBLISH; the nonce stays in the
    document and is published only with it, at the reveal."""
    import secrets
    nonce = nonce or secrets.token_hex(NONCE_BYTES)
    commitment = commitment_digest(doc, nonce)
    doc[REVEAL_KEY] = dict(format=COMMITMENT_FORMAT, nonce=nonce, commitment=commitment,
                           covers=list(COMMITMENT_COVERS))
    return commitment


def _without_reveal(doc):
    """The document minus its reveal block.

    SEALING MUST NOT DISARM THE COPY DEFENCE. `same_document` refuses two
    byte-identical files, which is the cheapest forgery there is: run `--all`
    once, copy the file, compare it with itself. A nonce is random per seal,
    so once sealing exists a copied document stops being byte-identical and
    that refusal would quietly stop firing -- a new check silently removing an
    old one, which is the failure this file keeps a list of. The nonce is the
    one field that MUST differ between two honest documents, so it is removed
    before that comparison rather than compared."""
    if not isinstance(doc, dict) or REVEAL_KEY not in doc:
        return doc
    return {k: v for k, v in doc.items() if k != REVEAL_KEY}


_COMMITMENT_HEX = re.compile(r"\A[0-9a-f]{64}\Z")


def read_published_commitment(value):
    """`(commitment_hex, problem)` from whatever a party actually published.

    A commitment is 64 characters. It gets pasted into a message, a mailing
    list post or a tweet far more often than it gets sent as a file, and a
    tool that only accepts a file pushes people into writing the file
    themselves, badly. So: a bare hex string is taken as itself, anything else
    is opened as a path and read as either the sidecar JSON or a text file
    holding the line."""
    if not isinstance(value, str) or not value.strip():
        return None, "empty commitment"
    text = value.strip()
    if _COMMITMENT_HEX.match(text):
        return text, None
    try:
        with open(text, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except OSError as exc:
        return None, (f"{text!r} is neither 64 hex characters nor a readable file: {exc}")
    stripped = raw.strip()
    if _COMMITMENT_HEX.match(stripped):
        return stripped, None
    try:
        obj = json.loads(raw)
    except ValueError:
        return None, f"{text}: not a commitment file and not a 64-character commitment"
    if not isinstance(obj, dict):
        return None, f"{text}: commitment file is {type(obj).__name__}, expected an object"
    if obj.get("format") != COMMITMENT_FORMAT:
        return None, (f"{text}: format is {obj.get('format')!r}, expected "
                      f"{COMMITMENT_FORMAT!r}")
    got = obj.get("commitment")
    if not isinstance(got, str) or not _COMMITMENT_HEX.match(got):
        return None, f"{text}: `commitment` is {got!r}, expected 64 hex characters"
    return got, None


#: Commitment states, most to least informative. Only `verified` binds a
#: party. `absent` and `self-declared` are WEAKER RESULTS, not failures; the
#: rest are failures and carry the comparison to COMMITMENT BROKEN.
_COMMITMENT_OK = ("verified", "absent", "self-declared")


def commitment_state(doc, published, label):
    """What one document's commitment does and does not prove.

    A COMMITMENT THAT TRAVELS WITH ITS NONCE PROVES NOTHING. The reveal block
    inside a document carries both halves, so anyone holding the document can
    recompute it; it is a convenience for the party, not evidence. Only a
    commitment the other side held BEFORE the exchange binds anything, which
    is why the published value is a separate argument and why a document that
    merely carries a reveal block is reported as `self-declared` rather than
    as a check that passed.
    """
    out = dict(label=label, state="absent", published=None, recomputed=None,
               nonce=None, problem=None)
    reveal = doc.get(REVEAL_KEY) if isinstance(doc, dict) else None
    pub, pub_problem = (None, None) if published is None else read_published_commitment(published)
    out["published"] = pub
    if published is not None and pub is None:
        out.update(state="UNREADABLE", problem=f"{label}: {pub_problem}")
        return out
    if not isinstance(reveal, dict) or not isinstance(reveal.get("nonce"), str):
        if pub is None:
            return out                                   # neither side of it exists: absent
        out.update(state="UNSEALED", problem=(
            f"{label}: {pub} was published for this document, but the document carries no "
            f"`{REVEAL_KEY}` nonce, so nothing can be checked against it. An unverifiable "
            f"commitment is not a verified one. Either the wrong file was handed over, or the "
            f"reveal was stripped out of it"))
        return out
    out["nonce"] = reveal["nonce"]
    try:
        recomputed = commitment_digest(doc, reveal["nonce"])
    except (ValueError, TypeError) as exc:
        out.update(state="UNREADABLE", problem=f"{label}: cannot recompute its commitment: {exc}")
        return out
    out["recomputed"] = recomputed
    stored = reveal.get("commitment")
    if isinstance(stored, str) and stored != recomputed:
        # The document was edited after it was sealed. A forger who reseals
        # defeats this, which is exactly why the PUBLISHED commitment is the
        # mechanism and this is only a free extra catch -- but a document that
        # disagrees with its own commitment is never reported as anything else.
        out.update(state="SELF-INCONSISTENT", problem=(
            f"{label}: the document does not match the commitment it carries itself "
            f"(carries {stored}, recomputes to {recomputed}); it was edited after sealing"))
        return out
    if pub is None:
        out.update(state="self-declared")
        return out
    if pub != recomputed:
        out.update(state="MISMATCH", problem=(
            f"{label}: does not match the commitment published for it "
            f"(published {pub}, this document commits to {recomputed})"))
        return out
    out.update(state="verified")
    return out


def commitment_report(a, b, label_a, label_b, commitment_a, commitment_b):
    """The commitment block of a comparison: both states and every cross-check
    that only makes sense with the pair in hand."""
    sa = commitment_state(a, commitment_a, label_a)
    sb = commitment_state(b, commitment_b, label_b)
    problems = [s["problem"] for s in (sa, sb) if s["problem"]]
    # ONE COMMITMENT PRESENTED TWICE. A party who copies the other's document
    # AND their published commitment would otherwise have both sides read
    # `verified`. Two honest documents cannot collide here: the nonces are 128
    # random bits, so equal commitments mean literally the same sealed file.
    if sa["published"] and sa["published"] == sb["published"]:
        problems.append(
            f"{label_a} and {label_b} were checked against the SAME published commitment "
            f"({sa['published']}). That is one commitment handed over twice, not two parties "
            f"each binding themselves before the exchange")
    if sa["nonce"] and sa["nonce"] == sb["nonce"]:
        problems.append(
            f"{label_a} and {label_b} carry the same nonce ({sa['nonce']}). A nonce is "
            f"{NONCE_BYTES * 8} random bits; two parties cannot draw the same one, so one "
            f"document's reveal block was copied from the other")
    verified = sa["state"] == "verified" and sb["state"] == "verified"
    published = [s["label"] for s in (sa, sb) if s["published"]]
    return dict(a=sa, b=sb, problems=problems, both_verified=verified and not problems,
                published_by=published, broken=bool(problems))


def _commitment_lines(c, la, lb):
    """The paragraph a reader sees about commitments. A comparison run without
    them is NOT a failure and must not be printed as one -- it is a weaker
    result, labelled the same way `same_device` is labelled weaker than
    `independent`, with the thing to do about it spelled out."""
    lines = []
    if c["broken"]:
        lines.append("COMMITMENTS: BROKEN. A document here is not the document that was")
        lines.append("committed to, so nothing below can be read as corroboration.")
        for m in c["problems"]:
            lines.append(f"  {m}")
        return lines
    if c["both_verified"]:
        lines.append("COMMITMENTS: both documents match a commitment published BEFORE the")
        lines.append("exchange, so neither party could have copied the other's numbers: each was")
        lines.append("bound to its own document before it could see the other's.")
        lines.append(f"  {la}: {c['a']['recomputed']}")
        lines.append(f"  {lb}: {c['b']['recomputed']}")
        return lines
    if c["published_by"]:
        bound = c["published_by"][0]
        free = lb if bound == la else la
        lines.append(f"COMMITMENTS: only {bound} is bound. A commitment was published for it")
        lines.append(f"before the exchange, and none was published for {free}, so {free} could")
        lines.append("have been written after seeing the other file and copying its cell values")
        lines.append("under its own provenance. A one-sided commit-reveal is stronger than none")
        lines.append("and much weaker than two.")
        return lines
    selfdec = [s["label"] for s in (c["a"], c["b"]) if s["state"] == "self-declared"]
    if selfdec:
        lines.append("COMMITMENTS: none were exchanged. "
                     f"{' and '.join(selfdec)} carr{'y' if len(selfdec) > 1 else 'ies'} one")
        lines.append("INSIDE the document, but a commitment that travels with its own nonce")
        lines.append("proves nothing: anyone holding the document can recompute it. Only a")
        lines.append("commitment the other party held BEFORE the exchange binds anything, and")
        lines.append("none was given to this comparison, so neither document is shown to have")
        lines.append("been computed rather than copied.")
    else:
        lines.append("COMMITMENTS: none were exchanged. Nothing here shows that either document's")
        lines.append("numbers were COMPUTED by the machine it names. Whichever party received the")
        lines.append("other's file first could have pasted its cell values into a document")
        lines.append("carrying their own provenance, and this command cannot tell that apart from")
        lines.append("two honest runs. This result is WEAKER for the same reason two documents")
        lines.append("from one device are weaker than two vendors.")
    lines.append("To close it, before either party sees the other's file:")
    lines.append("  python -m mojolearn verify --commitment mine.json     # publish the line it prints")
    lines.append("then exchange documents and add")
    lines.append(f"  --commitment-a <{la}'s published line> --commitment-b <{lb}'s>")
    return lines


def compare_documents(a, b, label_a="A", label_b="B", commitment_a=None, commitment_b=None):
    """Diff two evidence documents: where two machines agree, and where they do not.

    THE POINT IS THAT WE ARE NOT IN THE LOOP. Everything else this command
    offers still rests on our recorded table being honest. Two strangers, one
    with a 4090 and one with an M2, can each run `verify --all --json-out
    mine.json`, swap files, and run this. If the hashes match they have
    demonstrated the central claim TO EACH OTHER with us entirely absent,
    which is stronger evidence than anything we can publish about ourselves.

    IT NEEDS NO LANE SET OF ITS OWN, and that is deliberate. The lanes come
    from the two documents; this function never enumerates, greps or imports
    a list of lanes. A second idea of what the lane set is, in a second code
    path, is how one afternoon produced four different lane totals. Where a
    lane list IS needed it is read from the registry by import, the same way
    `tools/lane_select.py` and `tools/verification_matrix.py` read it.

    A COMPARER'S ONE FAILURE MODE IS AGREEING TOO EASILY, so every way of
    "matching" without two machines having computed the same bits is broken
    out and given a non-agreeing outcome:

    * a cell in only ONE document is INCOMPARABLE, never a match;
    * a cell NEITHER side computed (`value` null, where the probe raised) is
      two absences, not an agreement, however equal the nulls are;
    * a cell either side recorded as MOVED, BATCH_MOVED or RELOAD-MOVED says
      that box contradicted ITSELF, and two such documents agree only on the
      claim being false;
    * two DIFFERENT `n/a` reasons are a disagreement about what the part even
      is, so they are not folded in with an agreed `n/a`;
    * a duplicated cell row makes the whole document unreadable, because
      last-wins would let one party paste the other's answer over their own;
    * two documents that are byte-identical are ONE document passed twice,
      which compares nothing;
    * a cell both sides carry the SAME hash for, which either side judged
      DIVERGENT against its reference table, is an agreement on an answer one
      of them already recorded as wrong.

    NONE OF THAT TOUCHES THE ONE HOLE A STRUCTURAL CHECK CANNOT REACH: a well
    formed document whose numbers were never computed by the machine it names,
    because the party wrote them down after reading the other party's file.
    `commitment_a` and `commitment_b` are the commitments each party published
    BEFORE the exchange, and checking a document against one is the only thing
    here that distinguishes a run from a transcription. Passing neither is not
    an error and never will be -- a stranger with two files and no prior
    arrangement still gets a full comparison -- it is a WEAKER result, and
    `_commitment_lines` says so in the output in the same place and the same
    voice `same_device` is called weaker than `independent`.

    AGREE IS THE LAST OUTCOME TRIED, and that ordering is the point. On
    2026-09-16 `verdict()` was fixed for the same defect one level down: it
    returned VERIFIED as soon as ONE part read IDENTICAL, before it looked at
    REFUSED, so a CPU-only install printed `VERIFIED, exit 0` over 44
    identical and 288 refused parts. There is no number of agreements that
    makes up for one problem, here either. A wrong answer outranks an absent
    one, so the exit-1 outcomes are read before the exit-4 ones, exactly as
    `verdict()` reads DIVERGENT before REFUSED.
    """
    ca, sa, pa_ = _read_cells(a, label_a)
    cb, sb, pb_ = _read_cells(b, label_b)
    problems = pa_ + pb_
    # COMPUTED EVEN FOR A MALFORMED DOCUMENT, and reported even when the cells
    # are never compared. The preimage is built from the raw rows, so a
    # document with a duplicated cell part still has one; "this file is not the
    # file you were promised" is a thing a reader must be told whether or not
    # the file also failed to parse as evidence.
    commitment = commitment_report(a, b, label_a, label_b, commitment_a, commitment_b)

    def _sortkey(k):
        return tuple("" if x is None else str(x) for x in k)

    # A MALFORMED DOCUMENT IS NOT COMPARED AT ALL. Comparing the readable part
    # of an unreadable file produces a cell count, and a cell count next to a
    # complaint is exactly the shape a reader skims as a result.
    shared = [] if problems else sorted(set(ca) & set(cb), key=_sortkey)
    context_problems = comparison_context_problems(a, b, shared) if not problems else []
    if context_problems:
        shared = []
    agree, differ, moved, uncomputed, na, na_differ, agreed_div = [], [], [], [], [], [], []
    for key in shared:
        va_, vb_ = ca[key], cb[key]
        ka, kb = _value_kind(va_), _value_kind(vb_)
        row = dict(lane=key[0], fixture=key[1], part=key[2], a=va_, b=vb_,
                   kind_a=ka, kind_b=kb, state_a=sa.get(key), state_b=sb.get(key))
        if "moved" in (ka, kb):
            moved.append(row)
        elif "missing" in (ka, kb):
            uncomputed.append(row)
        elif ka == "n/a" and kb == "n/a":
            (na if va_ == vb_ else na_differ).append(row)
        elif va_ != vb_:
            differ.append(row)
        elif vref.DIVERGENT in (sa.get(key), sb.get(key)):
            # THE TWO PARTIES AGREE AND AT LEAST ONE OF THEM JUDGED THESE VERY
            # BITS WRONG. `verify --all` was fixed on 2026-09-16 to stop
            # letting parts that did run outrank parts that did not; the same
            # defect reaches this command through agreement, so a divergence
            # either side recorded is carried up rather than absorbed into the
            # agreement count.
            agreed_div.append(row)
        else:
            agree.append(row)
    only_a = [] if problems else sorted(set(ca) - set(cb), key=_sortkey)
    only_b = [] if problems else sorted(set(cb) - set(ca), key=_sortkey)

    da, db = (a.get("device") or {}) if isinstance(a, dict) else {}, \
             (b.get("device") or {}) if isinstance(b, dict) else {}
    same_class = da.get("device_class") == db.get("device_class")
    same_device = (da.get("device"), da.get("cpu_model")) == (db.get("device"), db.get("cpu_model"))
    try:
        same_document = (json.dumps(_without_reveal(a), sort_keys=True)
                         == json.dumps(_without_reveal(b), sort_keys=True))
    except (TypeError, ValueError):
        same_document = False
    prov = dict(
        a={k: da.get(k) for k in ("mojolearn_version", "commit", "vendor", "device_class",
                                  "device", "cpu_model", "platform", "python")},
        b={k: db.get(k) for k in ("mojolearn_version", "commit", "vendor", "device_class",
                                  "device", "cpu_model", "platform", "python")},
        same_device_class=same_class, same_device=same_device, same_document=same_document,
        bindings_a=[x.get("sha256") for x in ((a.get("bindings") or []) if isinstance(a, dict) else [])],
        bindings_b=[x.get("sha256") for x in ((b.get("bindings") or []) if isinstance(b, dict) else [])],
        independent=(not same_device) and (not same_class) and (not same_document),
        # each document's OWN verdict about its own run, carried through
        # unchanged. Two parties can agree with each other while one of them
        # checked a fraction of what the reader assumes, and the only honest
        # place to see that is next to the agreement.
        own_verdict_a=(a.get("verdict") if isinstance(a, dict) else None),
        own_verdict_b=(b.get("verdict") if isinstance(b, dict) else None),
        own_detail_a=(a.get("detail") if isinstance(a, dict) else None),
        own_detail_b=(b.get("detail") if isinstance(b, dict) else None),
        commitments_verified=commitment["both_verified"],
    )

    if problems:
        verdict_, code = "MALFORMED", EXIT_USAGE
    elif commitment["broken"]:
        # ABOVE EVERY CELL OUTCOME, INCLUDING MISMATCH. If a document is not
        # the one its party committed to, the reader does not yet know that
        # its cells are the cells that were computed, so no headline about
        # those cells is honest. Only MALFORMED outranks it, because a file
        # that will not parse was never a document at all.
        verdict_, code = "COMMITMENT BROKEN", EXIT_MISMATCH
    elif same_document:
        verdict_, code = "SAME DOCUMENT", EXIT_CANNOT_RUN
    elif context_problems:
        verdict_, code = "INCOMPARABLE", EXIT_CANNOT_RUN
    elif differ:
        verdict_, code = "MISMATCH", EXIT_MISMATCH
    elif moved:
        verdict_, code = "SELF-CONTRADICTED", EXIT_MISMATCH
    elif agreed_div:
        verdict_, code = "AGREED ON A DIVERGENT ANSWER", EXIT_MISMATCH
    elif only_a or only_b or uncomputed or na_differ:
        verdict_, code = "INCOMPLETE", EXIT_CANNOT_RUN
    elif agree:
        verdict_, code = "AGREE", EXIT_VERIFIED
    else:
        verdict_, code = "NOTHING COMPARED", EXIT_CANNOT_RUN
    return dict(format="mojolearn.verify-compare.v1", verdict=verdict_, exit=code,
                labels=dict(a=label_a, b=label_b), provenance=prov, problems=problems,
                context_problems=context_problems, commitment=commitment,
                agree=len(agree), differ=len(differ), n_a=len(na), moved=len(moved),
                uncomputed=len(uncomputed), n_a_differing=len(na_differ),
                agreed_divergent=len(agreed_div),
                only_in_a=[list(k) for k in only_a], only_in_b=[list(k) for k in only_b],
                differing=differ, agreeing=agree, self_contradicted=moved,
                not_computed=uncomputed, n_a_differing_cells=na_differ,
                agreed_divergent_cells=agreed_div)


def _cell_lines(title, rows, la, lb, limit=40):
    """Name the cells, and say how many were not named. PRINT THE MATCHES, NOT
    THE COUNT: a bare number is a thing a reader cannot check."""
    out = ["", f"{title} ({len(rows)}):"]
    for d in rows[:limit]:
        out.append(f"  {d['lane']}/{d['fixture']} {d['part']}: {la}={d['a']}  {lb}={d['b']}")
    if len(rows) > limit:
        out.append(f"  ... and {len(rows) - limit} more, not shown; run with --json for all of them")
    return out


def format_compare(r):
    p, la, lb = r["provenance"], r["labels"]["a"], r["labels"]["b"]
    commitment = r.get("commitment") or dict(broken=False, both_verified=False,
                                             published_by=[], problems=[],
                                             a=dict(state="absent"), b=dict(state="absent"))
    lines = ["# python -m mojolearn verify --compare", ""]
    if r["problems"]:
        lines.append("THESE FILES ARE NOT BOTH READABLE EVIDENCE DOCUMENTS:")
        for m in r["problems"][:20]:
            lines.append(f"  {m}")
        if len(r["problems"]) > 20:
            lines.append(f"  ... and {len(r['problems']) - 20} more")
        lines.append("")
        # the commitment paragraph prints here too. A malformed file that is
        # ALSO not the file its party committed to says more about why it is
        # malformed than the parse errors do.
        lines += _commitment_lines(commitment, la, lb)
        lines.append("")
        lines.append("RESULT: MALFORMED. Nothing was compared, and this is NOT a pass.")
        return "\n".join(lines)
    lines.append(f"  {'':<22} {la:<34} {lb}")
    for key, name in (("mojolearn_version", "version"), ("commit", "commit"),
                      ("vendor", "vendor"), ("device_class", "class"),
                      ("device", "device"), ("cpu_model", "cpu"), ("python", "python")):
        av, bv = str(p["a"].get(key)), str(p["b"].get(key))
        mark = "" if av == bv else "   <- differs"
        lines.append(f"  {name:<22} {av[:34]:<34} {bv[:34]}{mark}")
    lines.append(f"  {'bindings hashed':<22} {len(p['bindings_a']):<34} {len(p['bindings_b'])}")
    lines.append(f"  {'its own verdict':<22} {str(p['own_verdict_a'])[:34]:<34} "
                 f"{str(p['own_verdict_b'])[:34]}")
    for lbl, key in ((la, "own_detail_a"), (lb, "own_detail_b")):
        if p.get(key):
            lines.append(f"    {lbl}: {p[key]}")
    lines.append("")
    if commitment["broken"]:
        # THE INDEPENDENCE SENTENCE IS EXACTLY WHAT A FORGER WANTS A SKIMMER TO
        # READ, and it is read out of the provenance block, which is the part a
        # broken commitment says cannot be taken at face value. Printing "two
        # independent machines reaching the same bits" above a broken
        # commitment would hand the forgery the strongest line this command has.
        lines.append("THE PROVENANCE ABOVE CANNOT BE TAKEN AT FACE VALUE. A document here does not")
        lines.append("match the commitment its party published before the exchange, so the hardware")
        lines.append("it names is not something this comparison can stand behind. Nothing is said")
        lines.append("here about how independent the two machines were.")
    elif p["same_document"]:
        lines.append("THESE TWO FILES ARE BYTE-IDENTICAL. That is one document handed over twice,")
        lines.append("not two parties comparing, and it can only ever agree with itself.")
    elif p["independent"]:
        lines.append("These documents come from DIFFERENT hardware classes, which is what makes")
        lines.append("this worth doing: agreement here is two independent machines reaching the")
        lines.append("same bits.")
    elif p["same_device"]:
        lines.append("WARNING: both documents describe the SAME device. Agreement then shows")
        lines.append("repeatability, not cross-hardware identity, and proves much less.")
    else:
        lines.append("NOTE: these documents share a device class. Agreement is weaker evidence")
        lines.append("than two genuinely different vendors would give.")
    lines.append("")
    lines += _commitment_lines(commitment, la, lb)
    lines.append("")
    lines.append(f"  agree {r['agree']}   differ {r['differ']}   self-contradicted {r['moved']}   "
                 f"neither computed {r['uncomputed']}   agreed on a divergence "
                 f"{r['agreed_divergent']}")
    lines.append(f"  n/a agreed {r['n_a']}   n/a differing {r['n_a_differing']}   "
                 f"only in {la}: {len(r['only_in_a'])}   only in {lb}: {len(r['only_in_b'])}")
    if r["differing"]:
        lines += _cell_lines("DIFFERING CELLS", r["differing"], la, lb)
    if r["agreed_divergent_cells"]:
        lines += _cell_lines("AGREED, BUT ONE SIDE JUDGED THESE VERY BITS DIVERGENT",
                             r["agreed_divergent_cells"], la, lb)
    if r["self_contradicted"]:
        lines += _cell_lines("SELF-CONTRADICTED CELLS, a box that disagreed with ITSELF",
                             r["self_contradicted"], la, lb)
    if r["not_computed"]:
        lines += _cell_lines("NEITHER SIDE COMPUTED THESE, so they are two absences, not a match",
                             r["not_computed"], la, lb)
    if r["n_a_differing_cells"]:
        lines += _cell_lines("DIFFERENT n/a REASONS, a disagreement about what the part is",
                             r["n_a_differing_cells"], la, lb)
    for label, keys in ((la, r["only_in_a"]), (lb, r["only_in_b"])):
        if keys:
            lines.append("")
            lines.append(f"ONLY IN {label} ({len(keys)}), not comparable:")
            for k in keys[:20]:
                lines.append(f"  {k[0]}/{k[1]} {k[2]}")
            if len(keys) > 20:
                lines.append(f"  ... and {len(keys) - 20} more")
    lines.append("")
    if r["verdict"] == "COMMITMENT BROKEN":
        lines.append("RESULT: COMMITMENT BROKEN. A document here is not the document its party")
        lines.append("committed to before the exchange, so whatever its cells say, they cannot be")
        lines.append("read as an independent run. This is NOT a pass.")
        lines.extend("  " + problem for problem in commitment["problems"])
    elif r["verdict"] == "INCOMPARABLE":
        lines.append("RESULT: INCOMPARABLE. Input or protocol provenance is missing or differs.")
        lines.extend("  " + problem for problem in r['context_problems'])
    elif r["verdict"] == "AGREE":
        lines.append(f"RESULT: AGREE. {r['agree']} cell parts match across both documents, none")
        lines.append("differ, and none is present in only one. Neither machine trusted the other,")
        lines.append("and neither had to trust us.")
        if commitment["both_verified"]:
            lines.append("Both documents were committed to before either party saw the other's, so")
            lines.append("neither set of numbers could have been copied from the other.")
        else:
            # THE QUALIFIER RIDES ON THE RESULT LINE, not only in a paragraph
            # above it. A reader who greps for `RESULT:` -- and a script that
            # prints the last few lines -- must not get the strong sentence
            # without the reason it is weaker.
            lines.append("WEAKER THAN IT LOOKS: no commitment was exchanged, so neither document is")
            lines.append("shown to have been computed rather than copied. See COMMITMENTS above.")
    elif r["verdict"] == "MISMATCH":
        lines.append(f"RESULT: MISMATCH. {r['differ']} cell parts differ; they are named above.")
    elif r["verdict"] == "SELF-CONTRADICTED":
        lines.append(f"RESULT: SELF-CONTRADICTED. No cell differs BETWEEN the documents, but "
                     f"{r['moved']} cell")
        lines.append("part(s) record a box that gave two different answers for the same fit. Two")
        lines.append("documents agreeing on that agree the claim is false, which is not a pass.")
    elif r["verdict"] == "AGREED ON A DIVERGENT ANSWER":
        lines.append(f"RESULT: AGREED ON A DIVERGENT ANSWER. The two documents match on every")
        lines.append(f"shared cell, but {r['agreed_divergent']} of them carry a hash that one side "
                     "judged DIVERGENT")
        lines.append("against its own reference table. Two machines reaching the same wrong answer")
        lines.append("is a finding, not a pass.")
    elif r["verdict"] == "SAME DOCUMENT":
        lines.append("RESULT: SAME DOCUMENT. The two files are byte-identical, so nothing was")
        lines.append("compared. A document cannot corroborate itself.")
    elif r["verdict"] == "NOTHING COMPARED":
        lines.append("RESULT: NOTHING COMPARED. The two documents share no cell, so there is")
        lines.append("nothing to agree or disagree about.")
    else:
        missing = len(r["only_in_a"]) + len(r["only_in_b"])
        lines.append("RESULT: INCOMPLETE. Absence is not agreement.")
        lines.append(f"No cell differs, but {missing} cell part(s) appear in only one document, "
                     f"{r['uncomputed']}")
        lines.append(f"were computed by neither side, and {r['n_a_differing']} carry different n/a "
                     "reasons, so the two")
        lines.append("runs did not cover the same ground.")
    return "\n".join(lines)


def _cmd_compare(args):
    """`verify --compare A B`: diff two evidence documents. No GPU, no bindings.

    EVERY PATH OUT OF HERE PRINTS ONE `RESULT:` LINE and returns a documented
    exit code. A command that dies with a traceback exits 1, which is the code
    for MISMATCH, and an empty stdout is indistinguishable from a run that was
    never made; neither may be mistaken for the pass this is used to claim.
    """
    pa, pb = args.compare
    try:
        if os.path.realpath(pa) == os.path.realpath(pb):
            return _compare_refusal(args, EXIT_USAGE, "SAME FILE",
                                    f"both arguments name the same file ({os.path.realpath(pa)}). "
                                    "Two parties compare TWO documents; one file passed twice "
                                    "compares nothing and could only ever agree.")
        docs = []
        for p in (pa, pb):
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    docs.append(json.load(fh))
            except (OSError, ValueError) as exc:
                return _compare_refusal(args, EXIT_USAGE, "CANNOT READ", f"cannot read {p}: {exc}")
        r = compare_documents(docs[0], docs[1], os.path.basename(pa), os.path.basename(pb),
                              commitment_a=getattr(args, "commitment_a", None),
                              commitment_b=getattr(args, "commitment_b", None))
    except Exception as exc:                      # never let a crash exit 1 and read as MISMATCH
        return _compare_refusal(args, EXIT_CANNOT_RUN, "CANNOT RUN",
                                f"comparing raised {type(exc).__name__}: {exc}")
    if getattr(args, "json", False):
        _emit(json.dumps(r, indent=1, sort_keys=True))
    else:
        _emit(format_compare(r))
    return r["exit"]


def _cmd_commitment(args):
    """`verify --commitment DOC`: seal an evidence document and print the line
    to publish. No GPU, no bindings, no network, no repo.

    IT IS A SEPARATE COMMAND ON PURPOSE. Folding the seal into
    `--all --json-out` would put a random nonce into every evidence document
    anyone ever writes, including the ones we publish, and would make the
    binding step look like something the harness does rather than something a
    party does. It does not: the binding property comes entirely from
    PUBLISHING the commitment before you have seen the other document, and no
    amount of code here can supply that. A separate command puts the one act
    that matters in the party's own hands, where it belongs.

    Sealing is idempotent. A document that already carries a nonce is
    re-checked and its commitment reprinted rather than re-nonced, so running
    this twice cannot invalidate a line you already published.
    """
    path = args.commitment
    if getattr(args, "compare", None):
        _emit("USAGE: --commitment seals ONE document; --compare checks two against "
              "commitments already published. Use --commitment-a/--commitment-b with "
              "--compare.", sys.stderr)
        return EXIT_USAGE
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:
        _emit(f"# python -m mojolearn verify --commitment\n\n"
              f"RESULT: CANNOT READ. cannot read {path}: {exc}")
        return EXIT_USAGE
    if not isinstance(doc, dict) or doc.get("format") != COMPARE_INPUT_FORMAT:
        fmt = doc.get("format") if isinstance(doc, dict) else type(doc).__name__
        _emit(f"# python -m mojolearn verify --commitment\n\n"
              f"RESULT: NOT AN EVIDENCE DOCUMENT. {path} announces {fmt!r}, expected "
              f"{COMPARE_INPUT_FORMAT!r}. Write it with `verify --all --json-out {path}`.")
        return EXIT_USAGE
    reveal = doc.get(REVEAL_KEY)
    resealed = False
    try:
        if isinstance(reveal, dict) and isinstance(reveal.get("nonce"), str):
            commitment = commitment_digest(doc, reveal["nonce"])
            if reveal.get("commitment") != commitment:
                _emit(f"# python -m mojolearn verify --commitment\n\n"
                      f"RESULT: BROKEN. {path} was edited after it was sealed: it carries "
                      f"{reveal.get('commitment')} and now commits to {commitment}. If you have "
                      f"already published the old line, this document is not the one you "
                      f"published it for.")
                return EXIT_MISMATCH
        else:
            commitment = seal_document(doc)
            resealed = True
    except (ValueError, TypeError) as exc:
        _emit(f"# python -m mojolearn verify --commitment\n\n"
              f"RESULT: CANNOT RUN. sealing raised {type(exc).__name__}: {exc}")
        return EXIT_CANNOT_RUN
    sidecar = path + ".commitment"
    if resealed:
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(doc, fh, indent=1, sort_keys=True)
                fh.write("\n")
        except OSError as exc:
            _emit(f"# python -m mojolearn verify --commitment\n\n"
                  f"RESULT: CANNOT RUN. cannot write the nonce back into {path}: {exc}")
            return EXIT_CANNOT_RUN
    try:
        with open(sidecar, "w", encoding="utf-8") as fh:
            json.dump(dict(format=COMMITMENT_FORMAT, commitment=commitment,
                           document=os.path.basename(path), covers=list(COMMITMENT_COVERS),
                           note=("Publish this, or just the commitment string, BEFORE you see the "
                                 "other party's document. It does not contain the nonce; the nonce "
                                 "travels inside the document and is revealed with it.")),
                      fh, indent=1, sort_keys=True)
            fh.write("\n")
    except OSError as exc:
        _emit(f"# python -m mojolearn verify --commitment\n\n"
              f"RESULT: CANNOT RUN. cannot write {sidecar}: {exc}")
        return EXIT_CANNOT_RUN
    if getattr(args, "json", False):
        _emit(json.dumps(dict(format=COMMITMENT_FORMAT, commitment=commitment,
                              document=path, sidecar=sidecar, sealed_now=resealed,
                              covers=list(COMMITMENT_COVERS),
                              excludes=COMMITMENT_EXCLUDES), indent=1, sort_keys=True))
        return EXIT_VERIFIED
    lines = ["# python -m mojolearn verify --commitment", ""]
    lines.append(f"  document   {path}")
    lines.append(f"  nonce      {'written into the document just now' if resealed else 'already in the document'}"
                 "; do not publish the document yet")
    lines.append(f"  covers     {', '.join(COMMITMENT_COVERS)}")
    lines.append(f"  excludes   {', '.join(sorted(COMMITMENT_EXCLUDES))}")
    lines.append("")
    lines.append("PUBLISH THIS LINE NOW, BEFORE YOU SEE THE OTHER PARTY'S DOCUMENT:")
    lines.append("")
    lines.append(f"  {commitment}")
    lines.append("")
    lines.append(f"(the same value is in {sidecar}; either form is accepted)")
    lines.append("")
    lines.append("Then exchange documents -- they carry the nonces -- and either party runs:")
    lines.append("")
    lines.append("  python -m mojolearn verify --compare mine.json theirs.json \\")
    lines.append("      --commitment-a <the line you published> \\")
    lines.append("      --commitment-b <the line they published>")
    lines.append("")
    lines.append("RESULT: SEALED. This binds THIS document. It proves nothing on its own: what")
    lines.append("makes it evidence is that the other party held the line above before they sent")
    lines.append("you anything, so you could not have copied their answers into it.")
    _emit("\n".join(lines))
    return EXIT_VERIFIED


def _compare_refusal(args, code, headline, detail):
    """A refusal shaped like a compare result, so a reader parsing `format` is
    not handed a `verify-all-report` that never ran."""
    if getattr(args, "json", False):
        _emit(json.dumps(dict(format="mojolearn.verify-compare.v1", verdict=headline,
                              exit=code, detail=detail), indent=1, sort_keys=True))
    else:
        _emit("# python -m mojolearn verify --compare\n\n"
              f"RESULT: {headline}. {detail}")
    return code


def detail_line(counts):
    """The sentence under the verdict. It leads with how much of the run was
    actually checked, so `verified 44 of 332 cell parts` cannot be misread as
    `verified`."""
    total = sum(counts[s] for s in vref.STATES)
    return (f"verified {counts[vref.IDENTICAL]} of {total} cell parts "
            f"({counts[vref.DIVERGENT]} divergent, {counts[vref.OWED]} owed, "
            f"{counts[vref.REFUSED]} refused, {counts[vref.NA]} n/a)")


def lanes_line(report):
    """What was checked and what was not, in lanes rather than cell parts, so
    the two counts cannot be confused with each other."""
    cells = report["cells"]
    lanes = {r["lane"] for r in cells}
    clean = {l for l in lanes if all(r["state"] in (vref.IDENTICAL, vref.NA)
                                     for r in cells if r["lane"] == l)}
    judged = {l for l in clean if any(r["state"] == vref.IDENTICAL for r in cells if r["lane"] == l)}
    return (f"checked {len(judged)} of {len(lanes)} lanes end to end "
            f"({len(lanes) - len(clean)} with a divergent, refused or unreferenced part)")



def judge_rows(raw, table, families=None, unreferenced_lanes=(), device_class=None):
    """Attach state, detail and reference columns to raw result rows."""
    out = []
    for r in raw:
        part, lane = r["part"], r["lane"]
        ref_part, ref_lane = r.get("reference_part") or (part, lane)
        ent = (None if ref_lane in unreferenced_lanes else
               vref.entry(table, ref_lane, r["fixture"], ref_part, device_class=device_class))
        state, detail = vref.judge(r["value"], ent, r.get("error"))
        out.append(dict(lane=lane, fixture=r["fixture"], part=part, value=r["value"], state=state,
                        detail=detail, reference=(ent or {}).get("ref"),
                        columns=vref.columns_of(table, ent),
                        # carried through from the run: a cell reported at zero
                        # seconds did not happen, so the document keeps it
                        local_check=("not_run" if r["value"] is None or r.get("error")
                                     else "not_applicable" if isinstance(r["value"], str) and r["value"].startswith("n/a")
                                     else "passed" if isinstance(r["value"], str) and _HASH_RE.fullmatch(r["value"]) and not r.get("error")
                                     else "failed") if part in ("batch", "stepfull", "batchgrad", "batchscale", "ragged", "rlpair") else None,
                        seconds=r.get("seconds"),
                        family=(families or {}).get(lane, "portable models" if lane.startswith("portable:") else "other")))
    return out


def format_human(report):
    lines = []
    d = report["device"]
    lines.append("# python -m mojolearn verify --all")
    lines.append(f"# mojolearn {d['mojolearn_version']}  commit {d['commit'] or 'unknown'} ({d['commit_source']})")
    lines.append(f"# mode {d['numeric_mode']}  vendor {d['vendor']} (class {d['device_class']})  device {d['device']}")
    lines.append(f"# cpu {d['cpu_model']}  {d['platform']}  python {d['python']}  numpy {d['numpy']}")
    t = report["table"]
    lines.append(f"# reference table {t['path']} (sha256 {t['sha256'][:16]}, {t['records']} records)")
    lines.append(f"# depth {report['depth']}: {len(report['lanes'])} lanes x {len(report['fixtures'])} fixtures, "
                 f"{report['repeats']} repeat(s), {report['models_checked']} portable model(s)")
    if not report["harness"]["matches_table"]:
        lines.append("# note: this harness is not the one the table was generated with; lanes added or "
                     "changed since then read OWED or DIVERGENT until the table is regenerated")
    lines.append("")
    w = max([len("family")] + [len(f) for f, _ in report["families"]])
    head = f"| {'family':<{w}} | lanes | IDENTICAL | DIVERGENT | OWED | REFUSED | N/A |"
    lines.append(head)
    lines.append("|" + "-" * (w + 2) + "|-------|-----------|-----------|------|---------|-----|")
    for fam, c in report["families"]:
        lines.append(f"| {fam:<{w}} | {c['lanes']:>5} | {c['IDENTICAL']:>9} | {c['DIVERGENT']:>9} | "
                     f"{c['OWED']:>4} | {c['REFUSED']:>7} | {c['N/A']:>3} |")
    c = report["counts"]
    lines.append(f"| {'all':<{w}} | {len(report['lanes']) + report.get('model_lanes_checked', report['models_checked']):>5} | {c['IDENTICAL']:>9} | "
                 f"{c['DIVERGENT']:>9} | {c['OWED']:>4} | {c['REFUSED']:>7} | {c['N/A']:>3} |")
    def _detail_lines(prefix, detail):
        """`prefix: detail` with a multi-line detail's continuation indented.
        A refusal now carries the traceback of the raise (2026-09-19); it
        stays WHOLE here and is merely indented, never re-cut."""
        first, _, rest = ("" if detail is None else str(detail)).partition("\n")
        return [f"{prefix}: {first}"] + ["      " + line for line in rest.splitlines()]

    bad = [r for r in report["cells"] if r["state"] == vref.DIVERGENT]
    refused = [r for r in report["cells"] if r["state"] == vref.REFUSED]
    if bad:
        lines.append("")
        lines.append(f"DIVERGENT ({len(bad)}):")
        for r in bad[:40]:
            lines.extend(_detail_lines(f"  {r['lane']}/{r['fixture']} {r['part']}", r["detail"]))
        if len(bad) > 40:
            lines.append(f"  ... {len(bad) - 40} more in --json")
    if refused:
        lines.append("")
        lines.append(f"REFUSED ({len(refused)}):")
        seen = set()
        for r in refused:
            key = (r["lane"], r["detail"])
            if key in seen:
                continue
            seen.add(key)
            lines.extend(_detail_lines(f"  {r['lane']} {r['part']}", r["detail"]))
            if len(seen) >= 20:
                break
    lines.append("")
    if report.get("scope_gaps"):
        lines.append(f"UNVERIFIED LANES ({len(report['scope_gaps'])}):")
        lines.extend(f"  {name}: {reason}" for name, reason in report["scope_gaps"].items())
    if "properties" in report:
        batch = report["properties"].get("batch", {})
        lines.append("Batch invariance: " + ", ".join(f"{state}={n}" for state, n in batch.items()))
        for part in ("batchgrad", "batchscale", "ragged", "rlpair"):
            if part in report["properties"]:
                lines.append(part + ": " + ", ".join(f"{state}={n}" for state, n in report["properties"][part].items()))
    lines.append("Scope: selected fixtures and properties only; use verify --coverage for omissions.")
    lines.append(f"RESULT: {report['verdict']} ({report['detail']}). {report['elapsed_s']:.1f}s. exit {report['exit']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the command
# --------------------------------------------------------------------------

def _depth(args):
    if getattr(args, "quick", False) and getattr(args, "full", False):
        raise ValueError("--quick and --full are exclusive")
    if getattr(args, "models_only", False) and any(getattr(args, flag, False)
            for flag in ("no_models", "lanes", "fixtures", "include_pending", "batch_checks", "quick")):
        raise ValueError("--models-only cannot be combined with lane, fixture, pending, batch, quick or no-models selection")
    return "quick" if getattr(args, "quick", False) else "full"


def _reexec_identical(argv):
    """`python -m mojolearn verify --all` with no mode chosen: run it again
    in a child that selects the identical tier at import, which is the only
    tier the claim is about. A mode the user chose explicitly is never
    overridden."""
    env = dict(os.environ)
    env["MOJOLEARN_NUMERIC_MODE"] = "identical"
    env[_REEXEC_ENV] = "1"
    return subprocess.run([sys.executable, "-m", "mojolearn"] + list(argv), env=env).returncode


def cmd_verify_all(args):
    started = time.time()
    json_out = getattr(args, "json", False)
    log = (lambda s: _emit(s, sys.stderr)) if json_out else _emit
    # BEFORE ANY IMPORT OR TIER CHECK: comparing two evidence documents is a
    # pure function over two JSON files. It needs no GPU, no bindings and no
    # numeric mode, so it must not be gated behind them -- the whole point is
    # that a third party can run it on a machine that has none of ours.
    if getattr(args, "compare", None):
        return _cmd_compare(args)
    # Sealing a document is the same kind of thing: a pure function over one
    # JSON file, run by a party who may have none of our bindings.
    if getattr(args, "commitment", None):
        return _cmd_commitment(args)
    for flag in ("commitment_a", "commitment_b"):
        if getattr(args, flag, None):
            _emit(f"USAGE: --{flag.replace('_', '-')} is only meaningful with --compare; it is "
                  "the commitment that party published before the two documents were exchanged.",
                  sys.stderr)
            return EXIT_USAGE
    try:
        depth = _depth(args)
    except ValueError as exc:
        _emit(f"USAGE: {exc}", sys.stderr)
        return EXIT_USAGE
    if getattr(args, "emit_reference", None):
        return _cmd_emit_reference(args)

    requested = os.environ.get("MOJOLEARN_NUMERIC_MODE", "").strip().lower()
    try:
        import mojolearn as ml
        from . import _backend
        loaded = _backend.numeric_mode()
    except Exception as exc:
        return _finish(args, EXIT_CANNOT_RUN, "CANNOT RUN", f"importing mojolearn raised {type(exc).__name__}: {exc}")
    if loaded != "identical":
        if not requested and not os.environ.get(_REEXEC_ENV) and getattr(args, "argv", None) is not None:
            log("# no MOJOLEARN_NUMERIC_MODE was set; running again under the identical tier")
            return _reexec_identical(args.argv)
        return _finish(args, EXIT_REFUSED_FAST, "REFUSED",
                       f"this process loaded the {loaded!r} tier, which makes no bitwise promise; "
                       "run with MOJOLEARN_NUMERIC_MODE=identical (or leave it unset)")
    if getattr(args, "emit_models", None):
        return _cmd_emit_models(args, ml)
    if getattr(args, "self_test", False):
        return _cmd_self_test(args, ml)
    if getattr(args, "cross_check", None):
        return _cmd_cross_check(args, ml)

    try:
        table_file = getattr(args, "reference_table", None) or vref.table_path()
        table = vref.load_table(table_file)
    except vref.TableError as exc:
        return _finish(args, EXIT_NO_REFERENCE, "NO REFERENCE", str(exc))
    try:
        harness_file, harness_how = harness_path()
        harness = load_harness(harness_file)
    except FileNotFoundError as exc:
        return _finish(args, EXIT_NO_REFERENCE, "NO REFERENCE", str(exc))
    except CannotRun as exc:
        return _finish(args, EXIT_CANNOT_RUN, "CANNOT RUN", str(exc))

    vendor = _backend.vendor()
    vclass = vref.VENDOR_CLASS.get(vendor)
    if vclass is None:
        return _finish(args, EXIT_CANNOT_RUN, "CANNOT RUN", f"unknown vendor read-back {vendor!r}")
    from . import _verification_coverage as coverage
    coverage_report = coverage.inventory(harness, table, vclass)
    if getattr(args, "coverage", False):
        _emit(json.dumps(coverage_report, indent=1, sort_keys=True) if json_out
              else coverage.format_human(coverage_report))
        out_path = getattr(args, "json_out", None)
        if out_path:
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(coverage_report, fh, indent=1, sort_keys=True)
                fh.write("\n")
        return EXIT_VERIFIED  # successful inspection, explicitly not execution

    asked = [x for x in (getattr(args, "lanes", "") or "").split(",") if x]
    include_pending = getattr(args, "include_pending", False)
    models_only = getattr(args, "models_only", False)
    try:
        lanes, fixtures = select_lanes(harness, table, vclass, depth, asked, include_pending)
        if models_only:
            lanes, fixtures = [], []
    except ValueError as exc:
        _emit(f"USAGE: {exc}", sys.stderr)
        return EXIT_USAGE
    if getattr(args, "fixtures", ""):
        want = [x for x in args.fixtures.split(",") if x]
        unknown = [f for f in want if f not in harness.FIXTURES]
        if unknown:
            _emit(f"USAGE: --fixtures names fixtures the harness does not define: {unknown}", sys.stderr)
            return EXIT_USAGE
        fixtures = [f for f in harness.FIXTURES if f in want]
    repeats = max(1, int(getattr(args, "repeats", 1) or 1))

    device = _device_block(ml, harness)
    log(f"# verify --all: {vendor} ({vclass}), {len(lanes)} lanes x {len(fixtures)} fixtures, "
        f"harness {harness_how}, table {table_file}")

    # the fixture bytes first: a box whose numpy draws different fixtures
    # would read every cell DIVERGENT for a reason that is not arithmetic
    data, held, bad_fix = {}, {}, []
    for f in fixtures:
        data[f] = harness.fixture(f)
        held[f] = harness.heldout(f)
        X, yc, yr = data[f]
        got = dict(X=harness._h(X), y_clf=harness._h(yc), y_reg=harness._h(yr))
        if table["fixtures"].get(f) not in (None, got) or table["heldout"].get(f) not in (None, dict(X=harness._h(held[f]))):
            bad_fix.append(f)
    if bad_fix:
        return _finish(args, EXIT_CANNOT_RUN, "CANNOT RUN",
                       f"this machine generates different fixture bytes than the record for {bad_fix} "
                       f"(numpy {device['numpy']}); no cell would be comparable")

    # A LANE WHOSE FIXTURE MOVED PAST ITS REFERENCE IS NOT COMPARABLE
    # (2026-09-16, lane/identity-fixtures-light). A reference hash describes
    # one exact input. When a lane's fixture changes, the harness bumps its
    # LANE_REVISIONS entry and every hash taken at the old input stops
    # describing anything this harness can produce. Comparing anyway would
    # report DIVERGENT for a reason that has nothing to do with the user's
    # machine, which is the worst failure this tool has: it looks exactly like
    # the identity claim being false. These lanes are dropped from the
    # comparison and named, so the run is short of coverage (which the
    # verdict reflects) rather than quietly wrong.
    stale = [l for l in vref.stale_reference_lanes(table, harness) if l in lanes]
    if stale and not include_pending:
        log(f"# STALE REFERENCE, not compared ({len(stale)}): {', '.join(stale)}")
        log("#   their fixture moved (identity_break LANE_REVISIONS) past the reference this table "
            "carries; the next release record regenerates it")
        lanes = [l for l in lanes if l not in stale]
        if not lanes:
            return _finish(args, EXIT_CANNOT_RUN, "CANNOT RUN",
                           f"every selected lane has a stale reference ({', '.join(stale)}): each one's "
                           "fixture moved past the hash this table carries, so no cell would be "
                           "comparable; regenerate with --emit-reference from a current record")

    families = family_map(lanes)
    raw = []
    from ._cpu_reference import reference_training
    # Wall time per lane and per cell. Weak evidence alone, but cheap, and a
    # fit reported at zero milliseconds did not happen, so a fabricated run is
    # obvious in the document (lane/expose-inference-surface, 2026-09-16).
    extra_parts = vref.OPTIONAL_PARTS if getattr(args, "batch_checks", False) else ()
    contract_data, contract_held = dict(data), dict(held)
    if not getattr(args, "no_models", False) and 'base' not in contract_data:
        contract_data['base'], contract_held['base'] = harness.fixture('base'), harness.heldout('base')
    contract = verification_contract(harness, harness_file, contract_data, contract_held, extra_parts)
    lane_seconds, cell_seconds = {}, {}
    with reference_training():
        for lane in lanes:
            t0 = time.time()
            for f in fixtures:
                c0 = time.time()
                parts = run_cell(harness, ml, lane, f, data[f], held[f], repeats, extra_parts=extra_parts)
                cell_seconds[(lane, f)] = round(time.time() - c0, 4)
                for part, (value, error) in parts.items():
                    raw.append(dict(lane=lane, fixture=f, part=part, value=value, error=error,
                                    seconds=cell_seconds[(lane, f)]))
            lane_seconds[lane] = round(time.time() - t0, 3)
            log(f"  {lane:<34} {families[lane]:<16} {lane_seconds[lane]:6.1f}s")
    model_rows = [] if getattr(args, "no_models", False) else run_models(harness, ml, table, log=log, repeats=repeats, host_only=models_only)
    rows = judge_rows(raw + model_rows, table, families, unreferenced_lanes=stale, device_class=vclass)
    counts = {s: sum(1 for r in rows if r["state"] == s) for s in vref.STATES}
    code, headline = verdict(counts)
    fams = []
    for fam, c in summarize(rows, families):
        c = dict(c)
        c["lanes"] = len(c["lanes"])
        fams.append((fam, c))
    # An explicitly selected subset can pass its own scope. A full request
    # cannot hide withheld CPU routes or lanes dropped for stale references.
    withheld = {name: row["reason"] for name, row in coverage_report["lanes"].items()
                if row["status"] == "withheld"}
    scope_gaps = dict(withheld) if not asked and depth == "full" and not models_only else {}
    if include_pending:
        scope_gaps.update({name: withheld[name] for name in lanes if name in withheld})
        if vclass == "cpu":
            # A CPU checks the driver's logical shards, never physical GPU communication.
            scope_gaps.update({name: "CPU logical-shard replay; physical multi-GPU qualification pending"
                               for name in lanes if name.startswith("par-")})
            if not asked and depth == "full":
                scope_gaps.update({name: row["reason"] for name, row in coverage_report["lanes"].items()
                                   if name not in lanes and row["status"] in ("excluded", "unavailable")})
    scope_gaps.update({name: "stale reference" for name in stale})
    code, headline = verdict(counts, scope_gaps)
    detail = detail_line(counts)
    harness_sha = vref.sha256_file(harness_file)
    report = dict(
        format="mojolearn.verify-all-report.v1", verdict=headline, exit=code, detail=detail,
        depth=depth, lanes=lanes, fixtures=fixtures, repeats=repeats,
        models_checked=len({(r["lane"], r["fixture"]) for r in model_rows
                            if r["lane"] != "portable:manifest"}),
        model_lanes_checked=len({r["lane"] for r in model_rows
                                if r["lane"] != "portable:manifest"}),
        elapsed_s=round(time.time() - started, 2), device=device,
        harness=dict(path=harness_file, how=harness_how, sha256=harness_sha,
                     matches_table=harness_sha == table.get("harness_sha256")),
        table=dict(path=table_file, sha256=vref.sha256_file(table_file), format=table["format"],
                   records=len(table["records"]), harness_sha256=table.get("harness_sha256")),
        counts=counts, families=fams, cells=rows,
        lane_seconds=lane_seconds,
        coverage=coverage_report, scope_gaps=scope_gaps, verification_contract=contract,
        selection=dict(include_pending=include_pending, models_only=models_only,
                       cpu_logical_shard_lanes=[name for name in lanes if vclass == "cpu" and name.startswith("par-")]),
        properties={part: {state: sum(r["part"] == part and r["state"] == state for r in rows)
                           for state in vref.STATES} for part in tuple(vref.PARTS) + extra_parts},
    )
    try:
        from . import _verify
        report["bindings"] = [dict(module=b["module"], sha256=b["sha256"], size=b["size"])
                              for b in _verify.binding_artifacts()]
    except Exception as exc:                                  # provenance never costs the report
        report["bindings"] = []
        report["bindings_error"] = f"{type(exc).__name__}: {exc}"[:200]
    # THE HOST BINDINGS TOO (lane/expose-inference-surface, 2026-09-16).
    # `binding_artifacts()` scans sys.modules for `mojolearn._mojolearn*`, but
    # host bindings load under `mojolearn._host.*`, so on a CPU-ONLY install --
    # the install this command exists for -- the provenance block came back
    # EMPTY and answered nothing about which binary produced the numbers.
    try:
        report["bindings"] = report.get("bindings", []) + host_binding_artifacts()
    except Exception as exc:
        report["host_bindings_error"] = f"{type(exc).__name__}: {exc}"[:200]

    # EVIDENCE, NOT A VERDICT (lane/expose-inference-surface, 2026-09-16). A
    # reader who did not run this cannot audit the word VERIFIED, so the
    # document carries what they would need to check it themselves: every
    # reference resolved to the committed column it came from, lane-level
    # counts that cannot be confused with cell-part counts, and the self-test.
    # It is built from the SAME dict `format_human` renders, so the two cannot
    # drift into describing different runs.
    report["lanes_summary"] = lane_counts(rows, lanes, stale)
    report["reference_evidence"] = reference_evidence(rows)
    try:
        report["self_test"] = (dict(passed=None, ran=False,
            reason="models-only does not train; run verify --self-test separately")
            if models_only else self_test(harness, ml, table))
    except Exception as exc:
        report["self_test"] = dict(passed=None, error=f"{type(exc).__name__}: {exc}"[:200])

    if report["self_test"].get("passed") is False:
        code, headline = EXIT_MISMATCH, "MISMATCH"
        report.update(exit=code, verdict=headline,
                      detail=report["detail"] + "; comparator self-test failed")

    # THE THIRD CHECK, in the same artifact (lane/verify-cross-check,
    # 2026-09-16). A reader's agent should see all three at once, because they
    # answer different questions and only together mean much:
    #   1 this machine's GPU against its own CPU  -- trusts nobody
    #   2 this machine against our recorded columns -- trusts the table, which
    #     is auditable because the raw columns are committed
    #   3 the self-test -- shows the comparison can fail at all
    # On a CPU-only install (1) records that it did not run and why, which is
    # not a pass; it is never silently omitted.
    # IT IS NOT RUN IMPLICITLY, and that is a deliberate reversal. Folding a
    # cross-check into every --all seemed right (one artifact, all three
    # checks) until the shape of it was clear: on a GPU box it makes a
    # documented command ACQUIRE THE GPU as a side effect. Two runs at once, or
    # a run beside a gate, would then contend for the single Metal device --
    # the concurrency that previously returned NaN, constant and zero outputs
    # in two lanes. A command that quietly grabs a scarce device is the hidden
    # coupling this lane exists to remove, so --all records that the
    # cross-check was not run AND HOW TO RUN IT, which is not a pass, and
    # `verify --cross-check` stays the explicit door.
    report["cross_check"] = dict(
        ran=False, passed=None, scope=None,
        reason=("not run: --all does not take the GPU implicitly, because that would make this "
                "command contend for the single GPU with any other run. Use "
                "`python -m mojolearn verify --cross-check` to compare this machine's GPU "
                "against its CPU; on a CPU-only install that will say so rather than skip."))

    if json_out:
        _emit(json.dumps(report, indent=1, sort_keys=True))
    else:
        _emit(format_human(report))
    out_path = getattr(args, "json_out", None)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, sort_keys=True)
            fh.write("\n")
        _emit(f"# evidence written to {out_path}", sys.stderr if json_out else sys.stdout)
    return code


#: Where the committed columns a reference came from live, so `record` in the
#: evidence block is a path a reader can open rather than a bare name.
RECORD_ROOT = "bench/results/identity_break"


def host_binding_artifacts():
    """The CPU host bindings this process actually loaded, hashed.

    A reader asking "could this install have printed VERIFIED without the real
    code running" needs the sha256 of the binary that produced the numbers. On
    a CPU-only install that binary is a host binding under `mojolearn/host/`,
    which `_verify.binding_artifacts()` does not see because it scans for
    `mojolearn._mojolearn*` while these load under `mojolearn._host.*`."""
    from . import _backend
    out = []
    for basename in _backend.host_families_built():
        path = _backend.host_module_path(basename)
        try:
            with open(path, "rb") as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()
            out.append(dict(module=f"mojolearn._host.{basename}", sha256=digest,
                            size=os.path.getsize(path), path=path, kind="host"))
        except OSError:
            continue
    return out


def lane_counts(rows, lanes, stale=()):
    """Lanes checked, lanes skipped and why. Kept apart from the cell-part
    counts so `39 of 39 lanes` and `1065 of 1412 parts` cannot be confused.

    `stale` are lanes whose fixture moved past the reference this table
    carries. They are dropped by default or executed without references
    under --include-pending. They are
    logged by `cmd_verify_all`, but a lane that silently vanished from the
    comparison is precisely what this block exists to surface, so they are
    named here too.

    PORTABLE MODELS ARE COUNTED SEPARATELY. They are not harness lanes, and
    folding them in produced `6 checked of 2 requested`, a count that is
    self-evidently wrong and exactly the sort this block exists to prevent
    (caught on its first run, lane/expose-inference-surface, 2026-09-16).
    """
    asked = set(lanes)
    by_lane = {}
    for r in rows:
        by_lane.setdefault(r["lane"], []).append(r["state"])

    def classify(names):
        checked, skipped = [], {}
        for lane in sorted(names):
            states = by_lane[lane]
            if any(s == vref.IDENTICAL for s in states) and all(s in (vref.IDENTICAL, vref.NA) for s in states):
                checked.append(lane)
            elif any(s in (vref.REFUSED, vref.DIVERGENT, vref.OWED) for s in states):
                skipped[lane] = "one or more parts are divergent, refused or lack a reference"
            elif all(s in (vref.OWED, vref.NA) for s in states):
                skipped[lane] = "no committed record carries a comparable part yet"
            else:
                skipped[lane] = "no part reached IDENTICAL"
        return checked, skipped

    lane_names = {n for n in by_lane if not n.startswith("portable:")}
    model_names = {n for n in by_lane if n.startswith("portable:")}
    checked, skipped = classify(lane_names)
    m_checked, m_skipped = classify(model_names)
    stale_note = "fixture moved past the reference this table carries; not comparable"
    skipped.update({l: stale_note for l in stale})
    return dict(requested=len(asked | set(stale)), checked=len(checked), checked_lanes=checked,
                attempted=len(lane_names), attempted_lanes=sorted(lane_names),
                skipped=len(skipped), skipped_lanes=skipped,
                stale_references=sorted(stale),
                not_run=sorted(asked - lane_names),
                portable_models=dict(checked=len(m_checked), checked_models=m_checked,
                                     skipped=len(m_skipped), skipped_models=m_skipped),
                divergent_lanes=sorted({r["lane"] for r in rows if r["state"] == vref.DIVERGENT}),
                refused_lanes=sorted({r["lane"] for r in rows if r["state"] == vref.REFUSED}))


def reference_evidence(rows):
    """Every distinct committed column a reference in this run came from, as a
    path under the repository. This is what turns the table from an assertion
    into a pointer at evidence a reader can open and re-run."""
    seen = {}
    for r in rows:
        for cls, col in (r.get("columns") or {}).items():
            rec = col.get("record")
            if not rec or rec in seen:
                continue
            seen[rec] = dict(path=f"{RECORD_ROOT}/{rec}", device_class=cls,
                             vendor=col.get("vendor"), recorded_at_commit=col.get("commit"))
    return dict(root=RECORD_ROOT, columns=[seen[k] for k in sorted(seen)])


def _finish(args, code, headline, detail):
    if getattr(args, "json", False):
        _emit(json.dumps(dict(format="mojolearn.verify-all-report.v1", verdict=headline, exit=code,
                              detail=detail), indent=1, sort_keys=True))
    else:
        _emit(f"RESULT: {headline}: {detail}. exit {code}")
    return code


# --------------------------------------------------------------------------
# maintainer paths
# --------------------------------------------------------------------------

def _checkout_root():
    from . import _identity
    return _identity._checkout_root()


def _cmd_emit_reference(args):
    """Regenerate the table from committed records. Needs no binding."""
    root = _checkout_root()
    records = list(getattr(args, "records", None) or [])
    if not records:
        if not root:
            _emit("USAGE: --emit-reference needs --records DIR outside a checkout", sys.stderr)
            return EXIT_USAGE
        records = [os.path.join(root, "bench", "results", "identity_break")]
    paths = []
    for r in records:
        if os.path.isfile(r):
            paths.append(r)
            continue
        for dirpath, _, names in os.walk(r):
            paths.extend(os.path.join(dirpath, n) for n in names if n.endswith(".json"))
    try:
        harness_file = os.path.join(root, "tools", "identity_break.py") if root else harness_path()[0]
        harness = load_harness(harness_file)
    except (FileNotFoundError, CannotRun) as exc:
        _emit(f"CANNOT RUN: {exc}", sys.stderr)
        return EXIT_CANNOT_RUN
    logs = []
    lanes = [lane.strip() for lane in getattr(args, "lanes", "").split(",") if lane.strip()]
    unknown = set(lanes) - set(harness.LANES)
    if unknown:
        _emit(f"USAGE: unknown reference lanes: {sorted(unknown)}", sys.stderr)
        return EXIT_USAGE
    table = vref.build_table(paths, harness, root or os.getcwd(), log=logs.append, lanes=lanes or None,
                             parts=vref.PARTS + (vref.OPTIONAL_PARTS if getattr(args, "batch_checks", False) else ()))
    if getattr(args, "reference_table", None):
        try:
            table = vref.merge_reference_lanes(vref.load_table(args.reference_table), table, lanes)
        except vref.TableError as exc:
            _emit(f"REFUSED: {exc}", sys.stderr)
            return EXIT_USAGE
    vref.write_table(table, args.emit_reference)
    for line in logs:
        _emit(line, sys.stderr)
    s = vref.summary(table)
    s["bytes"] = os.path.getsize(args.emit_reference)
    _emit(json.dumps(s, sort_keys=True))
    return EXIT_VERIFIED


def _cmd_emit_models(args, ml):
    """Save the DEFAULT_MODEL_LANES base-fixture models on THIS box (a GPU
    install), and write the manifest only for models whose file bytes equal
    the table's model reference, so a shipped file is the file every
    recorded vendor wrote."""
    from . import _backend
    out_dir = args.emit_models
    vendor = _backend.vendor()
    if vref.VENDOR_CLASS.get(vendor) in (None, "cpu"):
        _emit(f"CANNOT RUN: --emit-models saves GPU-trained models; this install's vendor is {vendor!r}", sys.stderr)
        return EXIT_CANNOT_RUN
    table = vref.load_table(getattr(args, "reference_table", None) or vref.table_path())
    harness = load_harness()
    os.makedirs(out_dir, exist_ok=True)
    fixture = "base"
    X, yc, yr = harness.fixture(fixture)
    Xh = harness.heldout(fixture)
    models, problems = [], []
    for lane in [x for x in (getattr(args, "lanes", "") or "").split(",") if x] or DEFAULT_MODEL_LANES:
        fit = harness.LANES[lane](ml, X, yc, yr, Xh.copy())
        sl = harness._save_load(fit.est)
        if sl is None:
            problems.append(f"{lane}: the estimator has no save/load")
            continue
        save, load, suffix = sl
        name = f"{lane}.{fixture}{suffix}"
        path = os.path.join(out_dir, name)
        getattr(fit.est, save)(path)
        h = harness._hfile(path)
        ent = vref.entry(table, lane, fixture, "model")
        bent = vref.entry(table, lane, fixture, "batch")
        if ent is None or ent.get("ref") != h:
            problems.append(f"{lane}: file hash {h}, table model reference {(ent or {}).get('ref')}")
            continue
        if bent is None or not isinstance(bent.get("ref"), str) or bent["ref"].startswith("n/a"):
            problems.append(f"{lane}: the table has no batch reference")
            continue
        models.append(dict(lane=lane, fixture=fixture, file=name, bytes=os.path.getsize(path),
                           model_hash=h, batch_hash=bent["ref"], load=load,
                           **{"class": type(fit.est).__name__},
                           trained_on=dict(vendor=vendor, classes_agreeing=sorted(
                               c for c, v in ent["cols"].items() if isinstance(v, int)))))
    with open(os.path.join(out_dir, MODELS_MANIFEST), "w", encoding="utf-8") as fh:
        json.dump(dict(format=MODELS_FORMAT, models=models), fh, indent=1, sort_keys=True)
        fh.write("\n")
    _emit(json.dumps(dict(models=models, problems=problems), indent=1, sort_keys=True))
    return EXIT_VERIFIED if not problems else EXIT_MISMATCH
