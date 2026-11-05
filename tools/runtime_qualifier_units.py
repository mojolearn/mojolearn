# SPDX-License-Identifier: Apache-2.0
"""Per-unit receipts for the installed-runtime qualifier: bounded, resumable, reusable.

Test/tooling controller only: never compiles, never imports the product, never runs a test.
docs/release/RUNTIME_QUALIFICATION.md is the contract; in short:

* A UNIT is one registered public test file (qualify_native_runtime_installed.PUBLIC_TESTS).
  Every finished unit writes one receipt, ``qualify/units/<file>.json``, with its raw junit beside
  it, and the same pair into the unit store (``--unit-store``) at once, so an interrupted run loses
  only the unit that was running.
* A unit is REUSED (not run) when the newest passing receipt for it in the store or a prior
  directory was taken on the same device class and either
  - ``wheel-bytes``: the same candidate wheel bytes and the same test closure, or
  - ``closure``: the same unit source closure (test file + its static test helpers + the package
    Python modules they import, closed over the package's imports by tools/lane_select.py) or a
    closure whose only differences the shared selector rules call inert (the release
    ``__version__`` literal, docstring/comment-only edits), and the native modules the unit loaded
    carry the same bytes in the new wheels, and the same interpreter.
  ``--redo`` runs every unit.
* A unit's budget is three times its last measured seconds, at least 120 s (DEFAULT_BUDGET_S with
  no history); the whole-run budget stays the outer cap. A unit over its budget, or never started
  because the outer cap ran out, is FAILED by name.
* Tiers: ``smoke`` is the release gate's own installed smoke (the eleven release-light-v1
  correctness records on NVIDIA/AMD, the macOS wheel smoke on Apple), which every gate runs and
  never reuses; ``qualify`` is every unit here, run only when its reuse rule refuses.
"""
import ast
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess

UNIT_SCHEMA = "mojolearn.runtime-qualifier-unit.v1"
PLAN_SCHEMA = "mojolearn.runtime-qualifier-plan.v1"
CLOSURE_DEFINITION = "runtime-unit-closure.v1"
UNITS_DIR = "units"
PLAN = "units.json"
RUNS_DIR = "runs"
#: budget = max(MIN_BUDGET_S, BUDGET_FACTOR x last measured seconds)
BUDGET_FACTOR = 3
MIN_BUDGET_S = 120
#: a unit with no measured history (73 files took ~125 min on an L40S at 0.8.65, about 100 s each;
#: the slowest files several times that)
DEFAULT_BUDGET_S = 1800
PASSING = ("passed", "not-applicable")
TEST_PREFIX = "python/mojolearn/tests/"
PKG_PREFIX = "python/mojolearn/"
#: the gate's own installed smoke per vendor: always run by the gate step, never reused, never a unit here
SMOKE = {
    "nvidia": "tools/gate_smoke_box.py release-light-v1 correctness (the eleven installed records)",
    "amd": "tools/gate_smoke_box.py release-light-v1 correctness (the eleven installed records)",
    "apple": "tools/release.py macos-smoke (tools/qualify_verifier_wheel.py --scope expanded)",
    "cpu": "tools/gate_smoke_box.py release-light-v1 correctness on the NVIDIA box (loaded-lm-cpu and the installed checks)",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def stem(unit):
    return unit[:-3] if unit.endswith(".py") else unit


def tiers(units):
    """{tier: [unit]}: the gate smoke (named, run by the gate step) and the qualifier units."""
    return dict(smoke=["gate-installed-smoke"], qualify=list(units))


def budget(prior_seconds, default=DEFAULT_BUDGET_S):
    """The unit's run budget in seconds from its last measured seconds (None: no history)."""
    if prior_seconds is None:
        return int(default)
    return int(max(MIN_BUDGET_S, BUDGET_FACTOR * float(prior_seconds) + 0.999))


def device_class(vendor, target=None, machine=None):
    """A unit's device class: the requirement vendor plus the GPU target (or machine for CPU/Apple)."""
    detail = target or machine
    return vendor + ("-" + detail if detail else "")


# ---------------------------------------------------------------- closures
def test_closure(frozen_tests, unit):
    """(files {tests-relative path: sha256}, unresolved [reason]): the unit's test file, its ancestor
    conftest/__init__ files and the test helpers it statically imports, by the same resolver the
    runtime composition proves unchanged test inputs with (tools/runtime_test_tree_reuse.py prove).

    prove() only walks the helper closure when the trees differ, so it is handed the frozen hashes
    plus one absent sentinel path: nothing is compared but the walk, and every refusal it reports
    (dynamic imports, unresolved helpers) is kept: an unresolved closure is never reused by closure."""
    import runtime_test_tree_reuse as reuse
    root = Path(frozen_tests).resolve()
    hashes = {p.relative_to(root).as_posix(): sha256_file(p)
              for p in sorted(root.rglob("*")) if p.is_file() and not p.is_symlink()}
    probe = "__runtime_unit_closure_probe__.py"
    require(probe not in hashes, "closure probe collides with a test file")
    proof = reuse.prove(dict(hashes, **{probe: "0" * 64}), root, [unit], hashes)
    unresolved = [row.get("reason", "") + (" (" + row["file"] + ")" if row.get("file") else "")
                  for row in proof["unresolved"]]
    return dict(sorted(proof["closure"].items())), unresolved


def _package_imports(path):
    """The `mojolearn` module names one test-side Python file imports (tests modules excluded)."""
    try:
        tree = ast.parse(Path(path).read_text())
    except (OSError, SyntaxError, UnicodeError):
        return None
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            names.add(node.module)
            names.update(node.module + "." + alias.name for alias in node.names)
    return sorted(n for n in names if (n == "mojolearn" or n.startswith("mojolearn."))
                  and n != "mojolearn.tests" and not n.startswith("mojolearn.tests."))


def package_seeds(names, listing):
    """Package files for imported module names: the module (or package __init__) and every parent
    package's __init__ (importing mojolearn.a runs mojolearn/__init__.py first)."""
    files = set(listing)
    seeds = {PKG_PREFIX + "__init__.py"} & files
    for name in names:
        parts = name.split(".")[1:]
        for depth in range(len(parts) + 1):
            base = PKG_PREFIX + "/".join(parts[:depth])
            for candidate in ((base.rstrip("/") + "/__init__.py"), base + ".py") if depth else ():
                if candidate in files:
                    seeds.add(candidate)
    return sorted(seeds)


def product_closure(seeds, lane_select=None):
    """The package's own whole-file import closure of SEEDS (tools/lane_select.py _python_closure: the
    same walk, with its backwards subclass/patch edge, every identity lane's Python door uses)."""
    if lane_select is None:
        import lane_select
    return sorted(lane_select._python_closure(seeds))


def _blob_ids(git_root, commit, prefix):
    raw = subprocess.check_output(["git", "-C", str(git_root), "ls-tree", "-r", "-z", commit, "--", prefix])
    out = {}
    for row in raw.split(b"\0"):
        if row:
            metadata, name = row.split(b"\t", 1)
            out[name.decode()] = metadata.split()[2].decode()
    return out


def git_blob_id(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def unit_closure(source_root, git_root, source_commit, frozen_tests, unit, *, lane_select=None):
    """The unit's source closure at SOURCE_COMMIT: {definition, available, why, tests, product,
    tests_sha256, product_sha256, sha256}. Unavailable (never reused by closure) when the test walk
    is unresolved, the selector is not this tree's, or a product file's working bytes differ from the
    commit."""
    tests, unresolved = test_closure(frozen_tests, unit)
    result = dict(definition=CLOSURE_DEFINITION, source_commit=source_commit, available=True, why=[],
                  tests=tests, tests_sha256=digest(tests))
    if unresolved:
        result.update(available=False, why=["test closure unresolved: " + r for r in unresolved[:5]])
    if lane_select is None:
        import lane_select
    if Path(lane_select.ROOT).resolve() != Path(source_root).resolve():
        result.update(available=False, why=result["why"] + ["tools/lane_select.py is not the source tree's"])
        result.update(product={}, product_sha256=None, sha256=None)
        return result
    names = set()
    for rel in tests:
        if rel.endswith(".py"):
            found = _package_imports(Path(frozen_tests) / rel)
            if found is None:
                result.update(available=False, why=result["why"] + ["cannot parse " + rel])
                continue
            names.update(found)
    listing = lane_select._python_files()
    files = product_closure(package_seeds(sorted(names), listing), lane_select)
    blobs = _blob_ids(git_root, source_commit, PKG_PREFIX)
    product = {}
    for rel in files:
        data = (Path(source_root) / rel).read_bytes()
        if blobs.get(rel) != git_blob_id(data):
            result.update(available=False, why=result["why"] + ["working bytes differ from the commit: " + rel])
        product[rel] = sha256_bytes(data)
    result.update(product=product, product_sha256=digest(product),
                  sha256=digest(dict(tests=result["tests_sha256"], product=digest(product))))
    return result


def natives_of(installed):
    """{wheel member: sha256} of the native modules a unit's worker loaded (installed-modules.json)."""
    out = {}
    for member, row in (installed.get("modules") or {}).items():
        if row.get("native"):
            out[member] = row.get("sha256")
    return out


def inert_changes(changed, old_commit, lane_select=None):
    """(inert {path: reason}, live [path]) for product files whose bytes differ from OLD_COMMIT, judged
    by the selector's shared rules (release_label_only: the release __version__ literal;
    docstring_only: comments and docstrings). Anything those rules cannot prove stays live."""
    if lane_select is None:
        import lane_select
    inert, live = {}, []
    for path in changed:
        reason = None
        try:
            reason = lane_select.release_label_only(old_commit, path)
            if not reason and lane_select.docstring_only(old_commit, path):
                reason = "docstring/comment-only edit (lane_select.docstring_only)"
        except (OSError, ValueError, SyntaxError, subprocess.CalledProcessError):
            reason = None
        if reason:
            inert[path] = reason
        else:
            live.append(path)
    return inert, live


# ---------------------------------------------------------------- receipts and the store
def receipt_path(units_dir, unit):
    return Path(units_dir) / (unit + ".json")


def junit_name(unit):
    return stem(unit) + ".tests.xml"


def write_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def put_receipt(units_dir, receipt, junit_bytes):
    """Write RECEIPT and its junit into UNITS_DIR (qualify/units/<file>.json + <stem>.tests.xml)."""
    receipt = dict(receipt, junit=junit_name(receipt["unit"]), junit_sha256=sha256_bytes(junit_bytes))
    write_atomic(Path(units_dir) / receipt["junit"], junit_bytes)
    write_atomic(receipt_path(units_dir, receipt["unit"]), (json.dumps(receipt, indent=2) + "\n").encode())
    return receipt


def store_receipt(store, receipt, junit_bytes):
    """Append RECEIPT to the unit store: <store>/<stem>/<finished_at>-<sha12>.json and its junit.
    History is kept: nothing in the store is overwritten or deleted."""
    body = (json.dumps(receipt, indent=2) + "\n").encode()
    name = receipt["finished_at"].replace(":", "") + "-" + sha256_bytes(body)[:12]
    directory = Path(store) / stem(receipt["unit"])
    write_atomic(directory / (name + ".tests.xml"), junit_bytes)
    stored = dict(receipt, junit=name + ".tests.xml", junit_sha256=sha256_bytes(junit_bytes))
    write_atomic(directory / (name + ".json"), (json.dumps(stored, indent=2) + "\n").encode())
    return directory / (name + ".json")


def load_receipt(path):
    """(receipt, junit bytes) of one stored or retained unit receipt, or None when it is not a whole,
    self-consistent receipt (wrong schema, missing or altered junit)."""
    path = Path(path)
    try:
        receipt = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(receipt, dict) or receipt.get("schema") != UNIT_SCHEMA or not receipt.get("unit"):
        return None
    junit = path.parent / str(receipt.get("junit", ""))
    if not receipt.get("junit") or "/" in receipt["junit"] or not junit.is_file():
        return None
    data = junit.read_bytes()
    if sha256_bytes(data) != receipt.get("junit_sha256"):
        return None
    return dict(receipt, origin=str(path.resolve())), data


def collect(directories):
    """{unit: [(receipt, junit bytes)] newest first} from every unit receipt under DIRECTORIES."""
    found = {}
    for directory in directories:
        directory = Path(directory)
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.json")):
            loaded = load_receipt(path)
            if loaded:
                found.setdefault(loaded[0]["unit"], []).append(loaded)
    for rows in found.values():
        rows.sort(key=lambda row: str(row[0].get("finished_at", "")), reverse=True)
    return found


def stage_prior(sources, destination, vendors):
    """Copy the newest PASSING receipt per (device class, unit) of VENDORS found under SOURCES into
    DESTINATION/<device class>/<stem>/ (receipt + junit), plus the newest receipt of any status when it is
    newer (its seconds set the unit's budget). Returns the number of receipts staged. The originals stay
    where they are; this is the reuse input a gate job takes to the box."""
    staged = 0
    chosen = {}
    for unit, rows in collect(sources).items():
        for receipt, junit in rows:
            if receipt.get("vendor") not in vendors or not receipt.get("device_class"):
                continue
            key = (receipt["device_class"], unit)
            picks = chosen.setdefault(key, [])
            if not picks or (receipt.get("status") in PASSING
                             and not any(r.get("status") in PASSING for r, _ in picks)):
                picks.append((receipt, junit))
    for (device, unit), picks in sorted(chosen.items()):
        if "/" in device or device.startswith("."):
            continue
        for receipt, junit in picks:
            body = {k: v for k, v in receipt.items() if k != "origin"}
            store_receipt(Path(destination) / device, body, junit)
            staged += 1
    return staged


def last_seconds(rows):
    """The newest measured seconds of a unit that ran to an end (passed or failed tests, not stopped)."""
    for receipt, _ in rows:
        if receipt.get("seconds") is not None and not receipt.get("stopped"):
            return receipt["seconds"]
    return None


def decide(unit, rows, *, wheels, device, closure, payload=None, interpreter=None, redo=False,
           inert=None):
    """(decision, detail): ("reuse", {rule, receipt, junit, ...}) or ("run", {reason}).

    ROWS are the unit's prior receipts, newest first. WHEELS {name: sha256} are this run's candidate
    bytes, DEVICE its device class, CLOSURE its unit_closure, PAYLOAD the new wheels' member
    inventory (qualify_native_runtime_installed.wheel_inventory), INTERPRETER the worker's
    cache tag. INERT(changed product paths, old commit) -> (inert, live) judges closure differences."""
    if redo:
        return "run", dict(reason="--redo")
    refusals = []
    for receipt, junit in rows:
        where = receipt.get("finished_at", "?")
        if receipt.get("status") not in PASSING:
            refusals.append(f"{where}: status {receipt.get('status')}")
            continue
        if receipt.get("device_class") != device:
            refusals.append(f"{where}: device class {receipt.get('device_class')}")
            continue
        old = receipt.get("closure") or {}
        same_tests = old.get("tests_sha256") is not None and old.get("tests_sha256") == closure.get("tests_sha256")
        if receipt.get("wheels") == wheels and same_tests:
            return "reuse", dict(rule="wheel-bytes", receipt=receipt, junit=junit)
        if not same_tests:
            changed = sorted(k for k in set(old.get("tests") or {}) | set(closure.get("tests") or {})
                             if (old.get("tests") or {}).get(k) != (closure.get("tests") or {}).get(k))
            refusals.append(f"{where}: test closure changed: " + ", ".join(changed[:5]))
            continue
        if not closure.get("available") or not old.get("available"):
            refusals.append(f"{where}: closure unavailable: " + "; ".join((closure.get("why") or old.get("why") or [])[:3]))
            continue
        if interpreter is not None and receipt.get("interpreter_cache_tag") != interpreter:
            refusals.append(f"{where}: interpreter {receipt.get('interpreter_cache_tag')} != {interpreter}")
            continue
        natives = receipt.get("natives") or {}
        if payload is None or not natives or any((payload.get(m) or {}).get("sha256") != s for m, s in natives.items()):
            refusals.append(f"{where}: the native modules the unit loaded differ in these wheels")
            continue
        ours, theirs = closure.get("product") or {}, old.get("product") or {}
        changed = sorted(k for k in set(ours) | set(theirs) if ours.get(k) != theirs.get(k))
        if not changed:
            return "reuse", dict(rule="closure", receipt=receipt, junit=junit, inert={})
        if inert is None or set(changed) - set(ours) or set(changed) - set(theirs):
            refusals.append(f"{where}: closure changed: " + ", ".join(changed[:5]))
            continue
        quiet, live = inert(changed, old.get("source_commit") or receipt.get("source_commit"))
        if live:
            refusals.append(f"{where}: closure changed: " + ", ".join(live[:5]))
            continue
        return "reuse", dict(rule="closure", receipt=receipt, junit=junit, inert=quiet)
    return "run", dict(reason="; ".join(refusals[:3]) if refusals else "no receipt for this unit")


def unit_receipt(*, unit, vendor, backend, device, status, seconds, budget_s, wheels, source_commit,
                 test_commit, closure, scope, cases, interpreter=None, natives=None, run=None, reason="",
                 stopped=None, reused=None, finished_at=None):
    """One unit receipt (UNIT_SCHEMA); put_receipt adds the junit name and digest."""
    interpreter = interpreter or {}
    return dict(schema=UNIT_SCHEMA, unit=unit, tier="qualify", status=status, reason=reason,
                seconds=seconds, budget_s=budget_s, stopped=stopped, vendor=vendor, backend=backend,
                device_class=device, wheels=dict(wheels), source_commit=source_commit,
                test_commit=test_commit, closure=closure, scope=scope, cases=cases,
                interpreter=interpreter, interpreter_cache_tag=interpreter.get("cache_tag"),
                natives=dict(natives or {}), run=run, reused=reused, finished_at=finished_at or now())


def reused_receipt(receipt, rule, *, wheels, closure, inert=None):
    """The receipt a reused unit carries into a new run: the original receipt's observation, unchanged,
    with what it was reused under (rule, the current wheels and closure, the original's path)."""
    original = {k: v for k, v in receipt.items() if k != "origin"}
    return dict(original, reused=dict(rule=rule, origin=receipt.get("origin"), original_wheels=original["wheels"],
                                      original_closure_sha256=(original.get("closure") or {}).get("sha256"),
                                      wheels=dict(wheels), closure_sha256=closure.get("sha256"),
                                      tests_sha256=closure.get("tests_sha256"), inert=dict(inert or {})))


def counts_line(units):
    """RUNTIME units reused=n run=m failed=k (UNITS: plan/report unit rows with decision and status)."""
    reused = sum(1 for u in units if u.get("decision") == "reuse")
    ran = sum(1 for u in units if u.get("decision") == "run")
    failed = sum(1 for u in units if u.get("status") not in PASSING)
    return f"RUNTIME units reused={reused} run={ran} failed={failed}"


# ---------------------------------------------------------------- re-derivation (summarize)
def summarize(out, runtime, q):
    """The runtime-<vendor> report of a per-unit run, re-derived from the retained files only:
    coverage.json, request.json, qualify/units.json (the plan) and every unit receipt with its junit.
    Mixed reused and new units are one report: a run unit names this run's wheels, a reused unit
    the rule (wheel-bytes or closure) the plan admitted it under."""
    out = Path(out).resolve()
    qualify = out / runtime.QUALIFY_DIR
    mapping = runtime.coverage_map(json.loads((out / "coverage.json").read_text()))
    request = json.loads((out / "request.json").read_text())
    plan = json.loads((qualify / PLAN).read_text())
    vendor = request.get("vendor")
    require(vendor in runtime.BACKEND and plan.get("schema") == PLAN_SCHEMA and plan.get("vendor") == vendor,
            "unit plan names another vendor")
    backend = runtime.BACKEND[vendor]
    wheels = request.get("wheels") or {}
    require(plan.get("wheels") == wheels and wheels, "unit plan names other wheels")
    selected = [name for names in mapping.values() for name in names]
    planned = {row["unit"]: row for row in plan.get("units", [])}
    require(list(planned) == selected, "unit plan differs from the runtime coverage map")
    units_dir = qualify / UNITS_DIR
    rows, scopes, evidence_paths, cases, failed_units = [], {}, [out / "coverage.json", out / "request.json",
                                                                qualify / PLAN], 0, []
    for unit in selected:
        entry = planned[unit]
        path = receipt_path(units_dir, unit)
        loaded = load_receipt(path) if path.is_file() else None
        status, why = "failed", "no unit receipt"
        if loaded:
            receipt, junit = loaded
            evidence_paths += [path, units_dir / receipt["junit"]]
            error = _unit_errors(receipt, junit, entry, plan, wheels, backend, unit, q)
            status = "failed" if error else receipt["status"]
            why = error or receipt.get("reason", "")
            if status in PASSING:
                scopes[unit] = receipt["scope"]
                cases += receipt.get("cases") or 0
        if status not in PASSING:
            failed_units.append(dict(unit=unit, reason=why))
        rows.append(dict(unit=unit, decision=entry.get("decision"), rule=entry.get("rule"), status=status))
    coverage, missing = [], []
    for name, files in mapping.items():
        present = [scopes[f] for f in files if f in scopes]
        not_applicable = sum(len(s["not_applicable"]) for s in present)
        expected = sum(s["total"] for s in present) - not_applicable
        completed = sum(s["passed"] for s in present)
        failed = sum(s["failures"] for s in present)
        skipped = sum(len(s["unexpected_skips"]) for s in present)
        ok = (len(present) == len(files) and expected > 0 and completed == expected and not failed and not skipped
              and all(s["status"] != "failed-or-incomplete" for s in present))
        if expected == 0:
            missing.append(name)
        coverage.append(dict(id=name, status="passed" if ok else "failed", exit_code=0 if ok else 1,
                             expected=expected, completed=completed, failed=failed, skipped=skipped,
                             not_applicable=not_applicable, files=list(files)))
    passed = not failed_units and all(row["status"] == "passed" for row in coverage)
    evidence = [dict(path=p.relative_to(out).as_posix(), sha256=sha256_file(p)) for p in evidence_paths]
    return dict(schema=runtime.REPORT_SCHEMA, tool="tools/release_runtime_installed.py",
                origin="per-unit-receipts", vendor=vendor, backend=backend, numeric_mode="identical",
                source_revision=plan.get("source_commit"), test_revision=plan.get("test_commit"),
                device_class=plan.get("device_class"), wheels=dict(wheels),
                units=dict(reused=sum(1 for r in rows if r["decision"] == "reuse"),
                           run=sum(1 for r in rows if r["decision"] == "run"), failed=len(failed_units)),
                unit_rows=rows, failed_units=failed_units, coverage=coverage, coverage_without_tests=missing,
                cases=cases, status="passed" if passed else "failed", exit_code=0 if passed else 1,
                evidence=evidence)


def _unit_errors(receipt, junit, entry, plan, wheels, backend, unit, q):
    """'' when one unit receipt stands for this run, else why not."""
    if receipt.get("unit") != unit or receipt.get("backend") != backend:
        return "unit receipt names another unit or backend"
    if receipt.get("device_class") != plan.get("device_class"):
        return "unit receipt is for another device class"
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".xml") as temporary:
        temporary.write(junit)
        temporary.flush()
        tests = q.junit_summary(temporary.name)
    scope = q.test_scope_summary(tests, [unit], backend, interpreter=receipt.get("interpreter") or None)
    if receipt.get("status") in PASSING and (scope["scopes"].get(unit) != receipt.get("scope")
                                             or tests["total"] != receipt.get("cases")):
        return "unit junit differs from its receipt"
    if receipt.get("status") in PASSING and not scope["complete"]:
        return "unit junit is incomplete"
    if receipt.get("status") in PASSING and scope["scopes"][unit]["status"] != receipt["status"]:
        return "unit status differs from its junit"
    closure = entry.get("closure") or {}
    reused = receipt.get("reused")
    if entry.get("decision") == "run":
        if reused is not None or receipt.get("wheels") != wheels:
            return "a unit run here names other wheels"
        if (receipt.get("closure") or {}).get("sha256") != closure.get("sha256") \
                or (receipt.get("closure") or {}).get("tests_sha256") != closure.get("tests_sha256"):
            return "a unit run here names another closure"
        return ""
    if entry.get("decision") != "reuse" or not reused or reused.get("rule") != entry.get("rule"):
        return "unit reuse is not the plan's"
    if reused.get("wheels") != wheels or reused.get("tests_sha256") != closure.get("tests_sha256") \
            or reused.get("closure_sha256") != closure.get("sha256"):
        return "unit reuse names another run's wheels or closure"
    if (receipt.get("closure") or {}).get("tests_sha256") != closure.get("tests_sha256"):
        return "a reused unit's test closure differs"
    if entry["rule"] == "wheel-bytes" and receipt.get("wheels") != wheels:
        return "a unit reused by wheel bytes ran other wheels"
    if entry["rule"] == "closure":
        old = (receipt.get("closure") or {})
        changed = sorted(k for k in set(old.get("product") or {}) | set(closure.get("product") or {})
                         if (old.get("product") or {}).get(k) != (closure.get("product") or {}).get(k))
        if set(changed) != set(entry.get("inert") or {}) or not entry.get("natives_match"):
            return "a unit reused by closure has an unexplained closure or native difference"
    return ""
