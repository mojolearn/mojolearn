# SPDX-License-Identifier: Apache-2.0
"""The single-device verify commands that `verify --all` runs beside its identity cells.

`python -m mojolearn verify --all` runs, in the same process and the same report, each of these with its own
profile name, counts and verdict (Andrew, 2026-11-26):

    verify-distributed-program   captured-program distribution: several logical shards on device 0
    verify-partitioned-program   model/tensor/pipeline execution: logical stages and tensor shards on device 0
    verify-graph                 captured graph execution
    verify-generation            native causal generation, sampling, streaming and resume (CPU, and the GPU on a
                                 GPU install, compared with each other in the run)
    verify-causal-lm             whole loaded-model capture (CPU, and the GPU on a GPU install, compared in the run)
    verify-recipe                one bundled composed-model recipe (paper 5.4): a Sequential fitted from scratch
                                 on every device of the install, its fitted parameters and predictions compared
    verify-model                 one composed program saved with its report in one call (`Program.save`), then
                                 `verify-model` runs it in a fresh process on every device of the install and
                                 checks each output against that report (paper 5.4)

Each standalone subcommand keeps its own module and is the one body: this file calls those bodies (a capture in
a worker process, a comparison in the controller) and judges. Everything here runs on ONE device (Andrew,
2026-11-26): no case needs a second GPU and no case reads PENDING.

VERDICTS. VERIFIED: compared with something independent (the other device of this install, or a reference
report shipped in the wheel under `verify_reference/extended/<profile>.json`) and every compared value matched.
DIVERGENT: a compared value differed. FAILED: the command's own checks failed (a wrong analytic answer, a resume
that did not reproduce, a refused call). OWED: every own check passed but there was nothing to compare with (a
CPU-only install and no shipped reference, or a shipped reference of another version or profile): a capture
with nothing to compare against reads OWED, never VERIFIED. DIVERGENT and FAILED fail the run; OWED does not.

PARALLEL PLAN. Every capture is an independent unit (`units`): it runs in its own worker process
(`_verify_worker` action "extended") inside the same thread pool as the identity cell batches, so the
`--cell-workers` allocation covers both. The CPU and GPU captures of one command are separate units; only the
saved-program check waits for its save. This is verification tooling (AGENTS.md "The verify command is
protected"): it calls the public API around product calls, never inside one.
"""
import json
import os

FORMAT = "mojolearn.verify-extended.v1"

VERIFIED, DIVERGENT, FAILED, OWED = "VERIFIED", "DIVERGENT", "FAILED", "OWED"
VERDICTS = (VERIFIED, DIVERGENT, FAILED, OWED)
#: verdicts that fail `verify --all`
GATING = (DIVERGENT, FAILED)

PROFILES = ("verify-distributed-program", "verify-partitioned-program", "verify-graph", "verify-generation",
            "verify-causal-lm", "verify-recipe", "verify-model")
#: the bundled recipe `verify --all` fits (a shipped example, its fitted parameters compared too)
RECIPE_CASE = "seq-transformer"
#: weight formats of the loaded-model capture: float32 only, the command's own default (eight tiny architectures);
#: `verify-causal-lm --formats float32 bfloat16 int8` runs the other two (fixture budget, Andrew 2026-11-26: each
#: added case light, the whole addition under a minute)
CAUSAL_LM_FORMATS = ("float32",)
#: one unit's worker limit: far above the few seconds a unit takes, so only a hang reaches it
UNIT_TIMEOUT_S = 600.0
REFERENCE_DIR = "extended"


def devices(vclass):
    """The devices of this install: the CPU, and the GPU when the install has one."""
    return ("cpu",) if vclass == "cpu" else ("cpu", "gpu")


def units(vclass):
    """`[(profile, steps)]`: the independent units, longest first. A unit's steps run one after the other (only
    verify-model has several: the save, then a check per device, each in a fresh process, each needing the save);
    every unit is independent of every other, so they all join the cell workers' pool at once."""
    both = devices(vclass)
    return ([("verify-causal-lm", (d,)) for d in both]
            + [("verify-generation", (d,)) for d in both]
            + [("verify-graph", ("run",)), ("verify-distributed-program", ("run",)),
               ("verify-partitioned-program", ("run",)), ("verify-recipe", ("run",)),
               ("verify-model", ("save",) + tuple("check-" + d for d in both))])


# --------------------------------------------------------------------------
# in the worker: one capture, through the standalone command's own body
# --------------------------------------------------------------------------

def _program_fixture(ops, np):
    """A small composed program: dense, tanh, dense, with fixed weights."""
    rng = np.random.default_rng(54)
    s = ops.Session(numeric_mode="identical")
    x = s.tensor(np.zeros((4, 3), np.float32))
    w1 = s.tensor(rng.normal(0, 0.5, size=(3, 5)).astype(np.float32))
    b1 = s.tensor(rng.normal(0, 0.1, size=(1, 5)).astype(np.float32))
    w2 = s.tensor(rng.normal(0, 0.5, size=(5, 2)).astype(np.float32))
    return s, s.capture([x], [s.matmul(s.tanh(s.linear(x, w1, b1)), w2)])


def run_unit(profile, step, workdir, vendor):
    """One unit's capture, JSON-ready. Never raises for a product failure: the failure is in the value."""
    import traceback
    try:
        return dict(ok=True, value=_capture(profile, step, workdir, vendor))
    except Exception as exc:
        return dict(ok=False, error=f"{type(exc).__name__}: {exc}"[:2000], traceback=traceback.format_exc()[-4000:])


def _capture(profile, step, workdir, vendor):
    if profile == "verify-distributed-program":
        from . import _verify_distributed_program as m
        return m.verify((0,))
    if profile == "verify-partitioned-program":
        from . import _verify_partitioned_program as m
        return m.verify((0,))
    if profile == "verify-graph":
        from . import _verify_graph as m
        return m.run_checks(vendor)
    if profile == "verify-generation":
        from . import _verify_generation as m
        return m.capture(step)
    if profile == "verify-causal-lm":
        from . import _verify_causal_lm as m
        return m.capture(step, CAUSAL_LM_FORMATS)
    if profile == "verify-recipe":
        import mojolearn as ml
        from . import _verify_recipe as r
        recipe = r.load_recipe("example:" + RECIPE_CASE)[0]
        recipe["outputs"]["parameters"] = True
        r.check_recipe(recipe)
        results, arrays = r.fit_on_devices(ml, recipe)
        verdict = r.compare_results([(d, res, arrays.get(d)) for d, res in results.items()])
        record = r.make_record(recipe, results, verdict, getattr(ml, "__version__", None), ml.vendor())
        if all(res.get("ran") for res in results.values()):
            controls = r.self_test(ml, recipe, None, results, arrays, record["version"], record["vendor"])
            record["tamper_test"] = [dict(control=n, passed=ok, words=w) for n, ok, w in controls]
        return record
    if profile == "verify-model":
        from . import _verify_model as vm
        path = os.path.join(workdir, "composed.program")
        if step == "save":
            import numpy as np
            from . import ops
            s, program = _program_fixture(ops, np)
            try:
                program.save(path)          # one call: the program file and its report beside it
            finally:
                s.close()
            with open(vm.sidecar_path(path), encoding="utf-8") as fh:
                return dict(path=path, report=json.load(fh))
        device = step.split("-", 1)[1]
        with open(vm.sidecar_path(path), encoding="utf-8") as fh:
            expect = json.load(fh)
        report = vm.verify(path, expect=expect, device=device)
        verdict, rows, notes = vm.compare(report, expect)
        return dict(device=device, report=report, verdict=verdict, outputs=rows, notes=notes)
    raise ValueError(f"unknown extended profile {profile!r}")


# --------------------------------------------------------------------------
# in the controller: judge one profile from its units' values
# --------------------------------------------------------------------------

#: the shipped reference file's schema (tools/admit_extended_references.py writes it; the source tree carries an
#: `admitted: false` placeholder, so a release's data-only repack replaces a member and never adds one)
REFERENCE_SCHEMA = "mojolearn.verify-extended-reference.v1"
#: the profiles the identity record captures and admits a reference report for (the others compare the CPU and
#: GPU of the install, or a saved report, in the run)
REFERENCED = ("verify-graph", "verify-distributed-program", "verify-partitioned-program")


def reference(profile, pkg_dir):
    """The admitted reference report shipped for `profile` (`verify_reference/extended/<profile>.json`), or None
    (no file, or the placeholder before the first admission)."""
    path = os.path.join(pkg_dir, "verify_reference", REFERENCE_DIR, profile + ".json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if doc.get("schema") != REFERENCE_SCHEMA or doc.get("profile") != profile or not doc.get("admitted"):
        return None
    return doc.get("report")


def _block(profile, verdict, detail, counts, values, compared_with=None):
    return dict(profile=profile, verdict=verdict, detail=detail, counts=counts, compared_with=compared_with,
                reports=values)


def judge(profile, values, ref=None):
    """`values` is {step: unit result} (`run_unit`'s dict, or dict(ok=False, error) for a worker that did not
    answer). Returns the profile's block: verdict, detail, counts, compared_with and the whole reports."""
    failed = {step: v.get("error") for step, v in values.items() if not v.get("ok")}
    if failed:
        return _block(profile, FAILED, "; ".join(f"{s}: {e}" for s, e in failed.items()), dict(units=len(values),
                      failed=len(failed)), values)
    got = {step: v["value"] for step, v in values.items()}
    return globals()["_judge_" + profile.replace("verify-", "").replace("-", "_")](profile, got, ref)


def _cases_counts(report):
    states = [c.get("status") for c in report.get("cases", [])]
    return dict(cases=len(states), passed=states.count("PASS"), failed=len(states) - states.count("PASS"))


def _judge_program(profile, got, ref):
    report = got["run"]
    counts = _cases_counts(report)
    if report.get("status") == "FAIL" or counts["failed"]:
        why = (report.get("failure") or {}).get("message") or "a case did not pass"
        return _block(profile, FAILED, why, counts, got)
    if ref is None:
        return _block(profile, OWED, f"{counts['passed']} case(s) passed their own oracles; no reference report "
                      f"admitted yet (the release's identity record admits one)", counts, got)
    from . import _verify_distributed_program as d, _verify_partitioned_program as p
    module = d if profile == "verify-distributed-program" else p
    try:
        rows = module.compare(report, ref)
    except ValueError as exc:
        return _block(profile, OWED, f"the shipped reference is not comparable ({exc})", counts, got)
    differ = [r["name"] for r in rows if r["status"] != "MATCH"]
    counts.update(compared=len(rows), differ=len(differ))
    where = _where(ref)
    if differ:
        return _block(profile, DIVERGENT, "digest differs from the reference: " + ", ".join(differ), counts, got,
                      where)
    return _block(profile, VERIFIED, f"{len(rows)} case digest(s) equal to the reference", counts, got,
                  where)


_judge_distributed_program = _judge_program
_judge_partitioned_program = _judge_program


def _where(ref):
    install = (ref or {}).get("install") or (ref or {}).get("backend") or {}
    return dict(source="shipped reference", vendor=install.get("vendor"),
                version=install.get("mojolearn_version") or (ref or {}).get("version"))


def _judge_graph(profile, got, ref):
    from . import _verify_graph as g
    report = got["run"]
    rows = report.get("cases", {})
    counts = dict(cases=len(rows), passed=sum(r.get("status") == "passed" for r in rows.values()))
    counts["failed"] = counts["cases"] - counts["passed"]
    if counts["failed"]:
        bad = [f"{n}: {r.get('error')}" for n, r in rows.items() if r.get("status") != "passed"][:5]
        return _block(profile, FAILED, "; ".join(bad), counts, got)
    if ref is None:
        return _block(profile, OWED, f"{counts['passed']} case(s) passed; no reference report admitted "
                      "yet (the release's identity record admits one)", counts, got)
    errors = g.compare_reports(report, ref)
    unusable = [e for e in errors if e.startswith("expected:")]
    if unusable:
        return _block(profile, OWED, "the shipped reference is not comparable: " + "; ".join(unusable[:3]), counts,
                      got)
    counts.update(compared=len(rows), differ=len(errors))
    where = dict(source="shipped reference", vendor=(ref.get("backend") or {}).get("vendor"))
    if errors:
        return _block(profile, DIVERGENT, "; ".join(errors[:5]), counts, got, where)
    return _block(profile, VERIFIED, f"{len(rows)} case(s), every output word equal to the reference", counts,
                  got, where)


def _two_sided(profile, got, ref, own_check, compare, name):
    """CPU and GPU captures compared in the run; on a CPU-only install, the CPU capture against a shipped
    reference, else OWED."""
    counts = dict(captures=len(got))
    for step, report in got.items():
        problem = own_check(report)
        if problem:
            return _block(profile, FAILED, f"{step}: {problem}", counts, got)
    if "gpu" in got:
        notes = []
        try:
            equal = compare(got["cpu"], got["gpu"], notes)
        except ValueError as exc:
            return _block(profile, FAILED, f"the CPU and GPU captures are not comparable: {exc}", counts, got)
        counts.update(compared=1, differ=0 if equal else 1)
        where = dict(source="this install", sides=["cpu", "gpu"], notes=notes)
        if not equal:
            return _block(profile, DIVERGENT, f"the GPU {name} differ from the CPU's", counts, got, where)
        return _block(profile, VERIFIED, f"the CPU and GPU {name} are equal", counts, got, where)
    if ref is None:
        return _block(profile, OWED, f"the CPU {name} passed their own checks; no GPU on this install and no "
                      "reference report to compare with", counts, got)
    try:
        equal = compare(ref, got["cpu"], [])
    except ValueError as exc:
        return _block(profile, OWED, f"the shipped reference is not comparable ({exc})", counts, got)
    counts.update(compared=1, differ=0 if equal else 1)
    where = _where(ref)
    if not equal:
        return _block(profile, DIVERGENT, f"the {name} differ from the reference", counts, got, where)
    return _block(profile, VERIFIED, f"the {name} equal the reference", counts, got, where)


def _judge_generation(profile, got, ref):
    from . import _verify_generation as m

    def own(report):
        try:
            m._cells(report)
        except ValueError as exc:
            return str(exc)
        return None
    return _two_sided(profile, got, ref, own, lambda a, b, _notes: m.compare(a, b), "generation outputs")


def _judge_causal_lm(profile, got, ref):
    from . import _verify_causal_lm as m

    def own(report):
        if report.get("status") != "CAPTURED_UNQUALIFIED":
            bad = [f"{c['architecture']}/{c['weight_format']}: " + ",".join(k for k, v in c["checks"].items() if not v)
                   for c in report.get("cases", []) if not all(c["checks"].values())]
            return f"status {report.get('status')}: " + "; ".join(bad[:5])
        return None
    return _two_sided(profile, got, ref, own, m.compare, "loaded-model captures")


def _judge_recipe(profile, got, ref):
    from . import _verify_recipe as r
    record = got["run"]
    verdict = record.get("verdict") or {}
    ran = [d for d, res in (record.get("results") or {}).items() if res.get("ran")]
    counts = dict(devices=len(record.get("results") or {}), ran=len(ran),
                  outputs=verdict.get("outputs"), self_test_failed=sum(not c["passed"] for c in
                                                                       record.get("tamper_test", [])))
    if verdict.get("verdict") == "NOT RUN" or len(ran) != counts["devices"]:
        return _block(profile, FAILED, r.verdict_line(verdict) if verdict else "not run", counts, got)
    if counts["self_test_failed"]:
        return _block(profile, FAILED, "the recipe tamper test failed a control", counts, got)
    if verdict.get("verdict") == "DIVERGENT":
        return _block(profile, DIVERGENT, r.verdict_line(verdict), counts, got,
                      dict(source="this install", sides=ran))
    if verdict.get("verdict") != "IDENTICAL":
        return _block(profile, FAILED, r.verdict_line(verdict), counts, got)
    if len(ran) > 1:
        return _block(profile, VERIFIED, r.verdict_line(verdict), counts, got, dict(source="this install", sides=ran))
    if ref is None:
        return _block(profile, OWED, "fitted on the one device of this install; no reference record to compare "
                      "with", counts, got)
    try:
        r.check_against(record, ref)
    except r.Refused as exc:
        return _block(profile, OWED, f"the shipped reference record is not comparable ({exc})", counts, got)
    cross = r.compare_results(r.against_entries(record, {}, ref, "reference"))
    where = dict(source="shipped reference", vendor=ref.get("vendor"), version=ref.get("version"))
    if not cross["identical"]:
        return _block(profile, DIVERGENT if cross["verdict"] == "DIVERGENT" else FAILED, r.verdict_line(cross),
                      counts, got, where)
    return _block(profile, VERIFIED, r.verdict_line(cross), counts, got, where)


def _judge_model(profile, got, ref):
    saved = got.get("save") or {}
    checks = {step: v for step, v in got.items() if step.startswith("check-")}
    counts = dict(checks=len(checks), match=sum(v["verdict"] == "MATCH" for v in checks.values()))
    counts["differ"] = counts["checks"] - counts["match"]
    saved_on = ((saved.get("report") or {}).get("install") or {}).get("vendor")
    if counts["differ"]:
        bad = [f"{v['device']}: " + ", ".join(o["name"] for o in v["outputs"] if o["verdict"] != "MATCH")
               + ("; " + "; ".join(v["notes"]) if v["notes"] else "") for v in checks.values()
               if v["verdict"] != "MATCH"]
        return _block(profile, DIVERGENT, "verify-model against the saved report: " + " | ".join(bad), counts, got,
                      dict(source="the report saved with the program", saved_on=saved_on))
    devices_checked = sorted(v["device"] for v in checks.values())
    where = dict(source="the report saved with the program", saved_on=saved_on, checked_on=devices_checked)
    if saved_on and saved_on != "cpu" and "cpu" in devices_checked:
        return _block(profile, VERIFIED, f"saved on {saved_on} with its report in one call; verify-model MATCH on "
                      + ", ".join(devices_checked), counts, got, where)
    return _block(profile, OWED, "saved with its report and verify-model MATCH in a fresh process on the one "
                  "device of this install; another device is owed", counts, got, where)


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------

def summary(blocks):
    """{verdict: n} over the profiles."""
    return {v: sum(b["verdict"] == v for b in blocks.values()) for v in VERDICTS}


def gate(blocks):
    """The profiles whose verdict fails the run."""
    return [name for name, b in blocks.items() if b["verdict"] in GATING]


def human_lines(blocks):
    lines = []
    for name in PROFILES:
        b = blocks.get(name)
        if b is None:
            continue
        counts = ", ".join(f"{k} {v}" for k, v in (b.get("counts") or {}).items() if v is not None)
        lines.append(f"  {name:<28} {b['verdict']:<9} {b['detail']}" + (f" [{counts}]" if counts else ""))
    return lines
