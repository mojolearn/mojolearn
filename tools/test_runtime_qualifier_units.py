# SPDX-License-Identifier: Apache-2.0
"""Per-unit installed-runtime qualification (tools/runtime_qualifier_units.py): fake receipts only.

Nothing here installs a wheel, imports the product or runs a device: the qualifier is replaced by a
fake that writes the receipts the real per-unit mode writes."""
import base64
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qualify_native_runtime_installed as q  # noqa: E402
import release  # noqa: E402
import release_ledger_admission as admission  # noqa: E402
import release_runtime_installed as runtime  # noqa: E402
import runtime_qualifier_units as units  # noqa: E402
from test_release_installed_evidence import (admit, configuration_for, git, ledger_for, put,  # noqa: E402
                                             repo, sha)  # noqa: F401  (repo is a fixture)

RG = release.release_gate_module()
SELECTED = [name for names in json.loads(runtime.COVERAGE_PATH.read_text())["coverage"].values() for name in names]
NATIVE = "mojolearn/_mojolearn.so"
INTERPRETER = {"implementation": "cpython", "version": "3.11.9 (main)", "cache_tag": "cpython-311"}


def make_wheel(path, init=b"api = 1\n", native=b"\x7fELF native"):
    """A wheel q.wheel_inventory accepts (RECORD covers every member)."""
    import zipfile
    path.parent.mkdir(parents=True, exist_ok=True)
    members = {"mojolearn/__init__.py": init, NATIVE: native,
               "mojolearn/identity_columns/COMMIT": b"fixture\n"}
    record = "mojolearn_core-0.9.1.dist-info/RECORD"
    rows = []
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
            encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
            rows.append(f"{name},sha256={encoded},{len(data)}")
        rows.append(f"{record},,")
        archive.writestr(record, "\n".join(rows) + "\n")
    return path


def with_tests(root):
    """The fixture repo plus one committed test file per registered unit."""
    for name in SELECTED:
        put(root / "python/mojolearn/tests" / name, f"def test_case():\n    assert '{name}'\n")
    put(root / "python/mojolearn/tests/conftest.py", "")
    git(root, "add", ".")
    git(root, "commit", "-qm", "registered runtime tests")
    return git(root, "rev-parse", "HEAD")


def fake_closure(changed=(), version="1"):
    """closure_of: each unit's own test bytes, a product file that is VERSION for units in CHANGED."""
    def closure_of(source, git_root, commit, frozen, unit):
        tests = {unit: units.sha256_file(Path(frozen) / unit)}
        product = {"python/mojolearn/__init__.py": "v" + (version if unit in changed else "1")}
        return dict(definition=units.CLOSURE_DEFINITION, source_commit=commit, available=True, why=[],
                    tests=tests, tests_sha256=units.digest(tests), product=product,
                    product_sha256=units.digest(product),
                    sha256=units.digest(dict(tests=units.digest(tests), product=units.digest(product))))
    return closure_of


def junit_for(name, failing=False):
    body = '<failure message="boom">boom</failure>' if failing else ""
    return (f'<testsuites><testsuite><testcase classname="mojolearn.tests.{units.stem(name)}" name="test_case">'
            f'{body}</testcase></testsuite></testsuites>').encode()


def scope_of(junit, name, backend, tmp):
    path = tmp / ("scope-" + units.junit_name(name))
    path.write_bytes(junit)
    tests = q.junit_summary(path)
    return tests, q.test_scope_summary(tests, [name], backend, interpreter=INTERPRETER)


def fake_qualifier(tmp, crash_after=None, failing=()):
    """The per-unit qualifier: one receipt per --test-file into the plan's units dir and store."""
    calls = []

    def launch(command):
        plan = json.loads(Path(command[command.index("--unit-plan") + 1]).read_text())
        files = [command[i + 1] for i, arg in enumerate(command) if arg == "--test-file"]
        calls.append(files)
        planned = {row["unit"]: row for row in plan["units"]}
        payload, _ = q.wheel_inventory([command[i + 1] for i, a in enumerate(command) if a == "--wheel"])
        for count, name in enumerate(files):
            if crash_after is not None and count >= crash_after:
                return 137  # the box went away mid-run
            junit = junit_for(name, name in failing)
            tests, scope = scope_of(junit, name, plan["backend"], tmp)
            receipt = units.unit_receipt(
                unit=name, vendor=plan["vendor"], backend=plan["backend"], device=plan["device_class"],
                status="failed" if name in failing else "passed", seconds=4.0, budget_s=planned[name]["budget_s"],
                wheels=plan["wheels"], source_commit=plan["source_commit"], test_commit=plan["test_commit"],
                closure=planned[name]["closure"], scope=scope["scopes"][name], cases=tests["total"],
                interpreter=INTERPRETER, natives={NATIVE: payload[NATIVE]["sha256"]},
                run="qualify/run/runs/" + units.stem(name), reason="boom" if name in failing else "")
            receipt = units.put_receipt(plan["units_dir"], receipt, junit)
            if plan["store"]:
                units.store_receipt(plan["store"], receipt, junit)
        return 1 if failing else 0
    launch.calls = calls
    return launch


def run_args(root, commit, wheel, out, store, **extra):
    return SimpleNamespace(vendor="nvidia", source_root=root, source_commit=commit, test_source_commit=None,
                           git_root=None, wheel=[str(wheel)], python="python3", timeout=3600, test_timeout=900,
                           out=out, device_class="nvidia-sm_89", unit_store=store, prior_units=[], redo=False,
                           interpreter="cpython-311", **extra)


def report_of(out):
    return json.loads((out / runtime.REPORT).read_text())


# ---------------------------------------------------------------- the decision
def prior(tmp, *, closure, wheels=None, device="nvidia-sm_89", status="passed", natives=None):
    receipt = units.unit_receipt(unit="f.py", vendor="nvidia", backend="cuda", device=device, status=status,
                                 seconds=10.0, budget_s=120, wheels=wheels or {"core.whl": "a" * 64},
                                 source_commit="c" * 40, test_commit="c" * 40, closure=closure, scope={},
                                 cases=1, interpreter=INTERPRETER, natives=natives or {NATIVE: "n" * 64})
    return receipt, b"<testsuite/>"


@pytest.fixture
def closure(tmp_path):
    (tmp_path / "f.py").write_text("def test_x():\n    pass\n")
    return fake_closure()(None, None, "c" * 40, tmp_path, "f.py")


def test_reuse_by_wheel_bytes(tmp_path, closure):
    row = prior(tmp_path, closure=closure)
    decision, detail = units.decide("f.py", [row], wheels={"core.whl": "a" * 64}, device="nvidia-sm_89",
                                    closure=closure, payload={})
    assert decision == "reuse" and detail["rule"] == "wheel-bytes"


def test_reuse_by_closure_needs_the_same_native_bytes_and_interpreter(tmp_path, closure):
    row = prior(tmp_path, closure=closure)
    new = {"core.whl": "b" * 64}
    same = {NATIVE: {"sha256": "n" * 64}}
    decision, detail = units.decide("f.py", [row], wheels=new, device="nvidia-sm_89", closure=closure,
                                    payload=same, interpreter="cpython-311")
    assert decision == "reuse" and detail["rule"] == "closure" and detail["inert"] == {}
    decision, detail = units.decide("f.py", [row], wheels=new, device="nvidia-sm_89", closure=closure,
                                    payload={NATIVE: {"sha256": "m" * 64}}, interpreter="cpython-311")
    assert decision == "run" and "native modules" in detail["reason"]
    decision, detail = units.decide("f.py", [row], wheels=new, device="nvidia-sm_89", closure=closure,
                                    payload=same, interpreter="cpython-312")
    assert decision == "run" and "interpreter" in detail["reason"]


def test_invalidation_by_closure_test_bytes_device_status_and_redo(tmp_path, closure):
    new, same = {"core.whl": "b" * 64}, {NATIVE: {"sha256": "n" * 64}}
    changed = fake_closure(changed=("f.py",), version="2")(None, None, "c" * 40, tmp_path, "f.py")
    decision, detail = units.decide("f.py", [prior(tmp_path, closure=closure)], wheels=new, device="nvidia-sm_89",
                                    closure=changed, payload=same)
    assert decision == "run" and "closure changed: python/mojolearn/__init__.py" in detail["reason"]
    # the selector's inert rules (release label, docstring-only) may explain a closure difference
    decision, detail = units.decide("f.py", [prior(tmp_path, closure=closure)], wheels=new, device="nvidia-sm_89",
                                    closure=changed, payload=same,
                                    inert=lambda paths, old: ({p: "release label" for p in paths}, []))
    assert decision == "reuse" and detail["inert"] == {"python/mojolearn/__init__.py": "release label"}
    decision, _ = units.decide("f.py", [prior(tmp_path, closure=closure)], wheels=new, device="nvidia-sm_89",
                               closure=changed, payload=same, inert=lambda paths, old: ({}, list(paths)))
    assert decision == "run"
    (tmp_path / "f.py").write_text("def test_x():\n    assert 0\n")
    edited = fake_closure()(None, None, "c" * 40, tmp_path, "f.py")
    decision, detail = units.decide("f.py", [prior(tmp_path, closure=closure)], wheels={"core.whl": "a" * 64},
                                    device="nvidia-sm_89", closure=edited, payload=same)
    assert decision == "run" and "test closure changed: f.py" in detail["reason"]  # same wheels, new test
    decision, detail = units.decide("f.py", [prior(tmp_path, closure=closure, device="amd-gfx942")],
                                    wheels={"core.whl": "a" * 64}, device="nvidia-sm_89", closure=closure)
    assert decision == "run" and "device class" in detail["reason"]
    decision, detail = units.decide("f.py", [prior(tmp_path, closure=closure, status="failed")],
                                    wheels={"core.whl": "a" * 64}, device="nvidia-sm_89", closure=closure)
    assert decision == "run" and "status failed" in detail["reason"]
    decision, detail = units.decide("f.py", [prior(tmp_path, closure=closure)], wheels={"core.whl": "a" * 64},
                                    device="nvidia-sm_89", closure=closure, redo=True)
    assert (decision, detail["reason"]) == ("run", "--redo")
    assert units.decide("f.py", [], wheels={}, device="x", closure=closure)[1]["reason"] == "no receipt for this unit"


def test_budget_is_three_times_the_last_seconds_with_a_floor():
    assert units.budget(None) == units.DEFAULT_BUDGET_S
    assert units.budget(10) == 120 and units.budget(100) == 300 and units.budget(100.2) == 301


def test_the_test_closure_walks_static_helpers_and_ancestors(tmp_path):
    put(tmp_path / "conftest.py", "")
    put(tmp_path / "_helpers.py", "VALUE = 1\n")
    put(tmp_path / "other.py", "")
    put(tmp_path / "test_a.py", "from mojolearn.tests import _helpers\nimport mojolearn\n")
    files, unresolved = units.test_closure(tmp_path, "test_a.py")
    assert set(files) == {"test_a.py", "_helpers.py", "conftest.py"} and not unresolved
    assert units._package_imports(tmp_path / "test_a.py") == ["mojolearn"]
    listing = ["python/mojolearn/__init__.py", "python/mojolearn/a/__init__.py", "python/mojolearn/a/b.py"]
    assert units.package_seeds(["mojolearn.a.b", "mojolearn.Klass"], listing) == listing


# ---------------------------------------------------------------- budgets in the qualifier
def test_a_unit_over_its_budget_and_one_never_started_are_failed_by_name(tmp_path):
    first, second = SELECTED[0], SELECTED[1]
    plan = dict(schema=units.PLAN_SCHEMA, vendor="nvidia", backend="cuda", device_class="nvidia-sm_89",
                wheels={"core.whl": "a" * 64}, source_commit="c" * 40, test_commit="c" * 40,
                units_dir=str(tmp_path / "qualify/units"), store=str(tmp_path / "store"),
                units=[dict(unit=first, decision="run", budget_s=150, closure={}),
                       dict(unit=second, decision="run", budget_s=150, closure={})])
    put(tmp_path / "plan.json", plan)
    args = SimpleNamespace(timeout=1000, test_timeout=900, vendor="cuda", numeric_mode="identical",
                           unit_plan=tmp_path / "plan.json", jobs=1)
    times = iter([0, 0, 1000, 1000])
    seen = []

    def bounded(command, **kw):
        seen.append(kw["run_timeout"])
        return -9, dict(kind="run", budget_s=kw["run_timeout"], elapsed_s=150.4), 150.4
    out = tmp_path / "qualify/run"
    out.mkdir(parents=True)
    results = q.run_units(args, [first, second], out=out, tests=tmp_path / "frozen", python="python3",
                          env={"MOJOLEARN_GENERATION_DEVICE": "gpu"}, payload={}, test_commit="c" * 40,
                          clock=lambda: next(times), bounded=bounded, available=lambda: 64 * 2**30, cores=8)
    assert seen == [150]
    assert [r["status"] for r in results] == ["failed", "failed"]
    assert results[0]["reason"].startswith(f"unit {first} over its budget of 150 s")
    assert results[1]["reason"].startswith(f"unit {second} not started")
    for name in (first, second):
        receipt, _ = units.load_receipt(units.receipt_path(tmp_path / "qualify/units", name))
        assert receipt["status"] == "failed" and name in receipt["reason"]
    assert len(list((tmp_path / "store").rglob("*.json"))) == 2  # both persisted for the next run's budget


# ---------------------------------------------------------------- parallel units
def unit_plan(tmp_path, budgets):
    names = SELECTED[:len(budgets)]
    plan = dict(schema=units.PLAN_SCHEMA, vendor="nvidia", backend="cuda", device_class="nvidia-sm_89",
                wheels={"core.whl": "a" * 64}, source_commit="c" * 40, test_commit="c" * 40,
                units_dir=str(tmp_path / "qualify/units"), store=str(tmp_path / "store"),
                units=[dict(unit=name, decision="run", budget_s=b, closure={}) for name, b in zip(names, budgets)])
    put(tmp_path / "plan.json", plan)
    out = tmp_path / "qualify/run"
    out.mkdir(parents=True)
    return names, out


class TimedUnits:
    """A fake bounded worker: holds each unit for SECONDS and records when it started and ended."""
    def __init__(self, seconds=0.4, rss=None):
        import threading
        self.seconds, self.rss, self.lock, self.spans, self.budgets = seconds, rss, threading.Lock(), {}, {}

    def __call__(self, command, **kw):
        import time
        name = Path(kw["cwd"]).parent.name
        begin = time.monotonic()
        time.sleep(self.seconds)
        with self.lock:
            self.spans[name] = (begin, time.monotonic())
            self.budgets[name] = (kw["run_timeout"], kw["test_timeout"])
        if self.rss is not None and kw.get("usage") is not None:
            kw["usage"]["peak_rss_bytes"] = self.rss
        return 0, None, round(self.seconds, 1)


def run_parallel(tmp_path, budgets, jobs, fake, *, available=lambda: 64 * 2**30, test_timeout=200):
    names, out = unit_plan(tmp_path, budgets)
    args = SimpleNamespace(timeout=10000, test_timeout=test_timeout, vendor="cuda", numeric_mode="identical",
                           unit_plan=tmp_path / "plan.json", jobs=jobs)
    results = q.run_units(args, list(names), out=out, tests=tmp_path / "frozen", python="python3",
                          env={"MOJOLEARN_GENERATION_DEVICE": "gpu"}, payload={}, test_commit="c" * 40,
                          bounded=fake, available=available, cores=8, admit_poll=0.01)
    return names, results


def overlapping(spans):
    spans = sorted(spans.values())
    return sum(1 for (_, end), (begin, _) in zip(spans, spans[1:]) if begin < end)


def test_units_run_side_by_side_with_their_own_budgets_and_receipts(tmp_path, capsys):
    budgets = [150, 600, 300, 120]
    fake = TimedUnits()
    names, results = run_parallel(tmp_path, budgets, 4, fake)
    printed = capsys.readouterr().out
    assert "QUALIFY units=4 workers=4 of 8 core(s) (--jobs 4)" in printed
    assert overlapping(fake.spans) == 3                       # all four were running at once
    for name, budget in zip(names, budgets):                  # each unit under its own budget and stall bound
        assert fake.budgets[units.stem(name)] == (budget, min(200, budget))
    assert [r["unit"] for r in results] == names              # results in selection order, as before
    assert sum(line.startswith("UNIT ") for line in printed.splitlines()) == 4
    # the receipts have exactly the shape a one-at-a-time run writes
    serial_root = tmp_path / "serial"
    serial_root.mkdir()
    run_parallel(serial_root, budgets, 1, TimedUnits(seconds=0.01))
    for name in names:
        parallel, _ = units.load_receipt(units.receipt_path(tmp_path / "qualify/units", name))
        serial, _ = units.load_receipt(units.receipt_path(serial_root / "qualify/units", name))
        assert sorted(parallel) == sorted(serial) and parallel["budget_s"] == serial["budget_s"]
    assert len(list((tmp_path / "store").rglob("*.json"))) == 4


def test_one_job_runs_one_at_a_time_heaviest_first(tmp_path):
    fake = TimedUnits(seconds=0.05)
    names, _ = run_parallel(tmp_path, [120, 900, 300], 1, fake)
    assert overlapping(fake.spans) == 0
    order = [stem for stem, _ in sorted(fake.spans.items(), key=lambda row: row[1][0])]
    assert order == [units.stem(names[1]), units.stem(names[2]), units.stem(names[0])]


def test_a_unit_starts_only_while_free_memory_holds_one_more(tmp_path, capsys):
    # three workers asked for, but the free memory holds no second unit: they run one after another
    fake = TimedUnits(seconds=0.1, rss=2 * 2**30)
    run_parallel(tmp_path, [150, 150, 150], 3, fake, available=lambda: 2**30)
    assert overlapping(fake.spans) == 0
    assert "QUALIFY unit-memory" in capsys.readouterr().out


def test_worker_count_is_cores_capped_by_memory_and_units(monkeypatch):
    monkeypatch.delenv(q.QUALIFY_JOBS_ENV, raising=False)
    assert q.unit_workers(73, cores=10, free=None)[0] == 10
    assert q.unit_workers(3, cores=10, free=None)[0] == 3
    workers, why = q.unit_workers(73, cores=10, free=6 * 2**30)
    assert workers == 4 and "1.5 GiB per unit" in why
    assert q.unit_workers(73, cores=10, free=2**20)[0] == 1
    assert q.unit_workers(73, cores=10, free=None, env={q.QUALIFY_JOBS_ENV: "6"}) == (6, "MOJOLEARN_QUALIFY_JOBS=6")
    assert q.unit_workers(73, jobs=2, cores=10, free=None, env={q.QUALIFY_JOBS_ENV: "6"})[0] == 2
    with pytest.raises(ValueError):
        q.unit_workers(73, cores=10, env={q.QUALIFY_JOBS_ENV: "0"})


# ---------------------------------------------------------------- units per GPU
class DeviceUnits(TimedUnits):
    """TimedUnits that also records each unit's visible-devices variable and its live bounds at the end."""
    def __init__(self, seconds=0.3):
        super().__init__(seconds=seconds)
        self.devices, self.final = {}, {}

    def __call__(self, command, **kw):
        result = super().__call__(command, **kw)
        name = Path(kw["cwd"]).parent.name
        with self.lock:
            self.devices[name] = (kw["env"] or {}).get("CUDA_VISIBLE_DEVICES")
            self.final[name] = kw["limits"]()
        return result


def most_at_once(spans):
    events = sorted([(begin, 1) for begin, _ in spans.values()] + [(end, -1) for _, end in spans.values()],
                    key=lambda row: (row[0], row[1]))
    now = most = 0
    for _, step in events:
        now += step
        most = max(most, now)
    return most


def run_on_gpus(tmp_path, monkeypatch, gpus, rows, fake, *, per_gpu=None, test_timeout=200):
    monkeypatch.delenv(q.QUALIFY_JOBS_ENV, raising=False)
    if per_gpu is None:
        monkeypatch.delenv(q.UNITS_PER_GPU_ENV, raising=False)
    else:
        monkeypatch.setenv(q.UNITS_PER_GPU_ENV, str(per_gpu))
    names = SELECTED[:len(rows)]
    plan = dict(schema=units.PLAN_SCHEMA, vendor="nvidia", backend="cuda", device_class="nvidia-sm_89",
                wheels={"core.whl": "a" * 64}, source_commit="c" * 40, test_commit="c" * 40,
                units_dir=str(tmp_path / "qualify/units"), store=str(tmp_path / "store"),
                units=[dict(unit=name, decision="run", budget_s=budget, prior_s=prior, closure={})
                       for name, (budget, prior) in zip(names, rows)])
    put(tmp_path / "plan.json", plan)
    out = tmp_path / "qualify/run"
    out.mkdir(parents=True)
    args = SimpleNamespace(timeout=10000, test_timeout=test_timeout, vendor="cuda", numeric_mode="identical",
                           unit_plan=tmp_path / "plan.json", jobs=None)
    results = q.run_units(args, list(names), out=out, tests=tmp_path / "frozen", python="python3",
                          env={"MOJOLEARN_GENERATION_DEVICE": "gpu"}, payload={}, test_commit="c" * 40,
                          bounded=fake, available=lambda: 64 * 2**30, cores=48, admit_poll=0.01, gpus=gpus)
    return names, results


def test_a_one_gpu_host_runs_at_most_two_units_and_scales_their_budgets(tmp_path, monkeypatch, capsys):
    # 0.8.66 run t: 48 cores, one RTX 4090; five units at once blew their serial budgets
    fake = DeviceUnits()
    names, results = run_on_gpus(tmp_path, monkeypatch, ["0"], [(120, 40.0)] * 6, fake)
    printed = capsys.readouterr().out
    assert "QUALIFY units=6 workers=2 of 48 core(s) (48 core(s), " in printed
    assert ", capped by 1 GPU(s) x 2 unit(s) per GPU)" in printed
    assert "gpus=1 (given) units-per-gpu=2" in printed
    assert most_at_once(fake.spans) == 2
    assert all(device is None for device in fake.devices.values())  # one GPU: the environment is left alone
    for name in names:
        receipt, _ = units.load_receipt(units.receipt_path(tmp_path / "qualify/units", name))
        assert receipt["budget_s"] == 120                          # the serial bound, unchanged
        assert receipt["concurrency"] == 2 and receipt["scaled_budget_s"] == 240
        assert fake.final[units.stem(name)] == (240, 240)          # run bound and one-test bound, both x2
    assert len(results) == 6 and "shared-gpu=2" in printed


def test_a_four_gpu_host_runs_eight_units_two_to_each_gpu(tmp_path, monkeypatch, capsys):
    fake = DeviceUnits(seconds=0.4)
    names, _ = run_on_gpus(tmp_path, monkeypatch, ["0", "1", "2", "3"], [(120, 30.0)] * 12, fake)
    assert "workers=8 of 48 core(s)" in capsys.readouterr().out
    assert most_at_once(fake.spans) == 8
    assert set(fake.devices.values()) == {"0", "1", "2", "3"}      # each unit sees only its own GPU
    for gpu in "0123":
        assert most_at_once({k: v for k, v in fake.spans.items() if fake.devices[k] == gpu}) <= 2


def test_a_heavy_unit_runs_with_at_most_one_sibling(tmp_path, monkeypatch):
    # four units per GPU allowed, but a unit of 60 s or more alone (or no history) keeps its GPU to two
    fake = DeviceUnits(seconds=0.2)
    rows = [(600, 200.0), (300, None)] + [(120, 10.0)] * 6
    names, _ = run_on_gpus(tmp_path, monkeypatch, ["0"], rows, fake, per_gpu=4)
    spans = fake.spans
    for heavy in (units.stem(names[0]), units.stem(names[1])):
        begin, end = spans[heavy]
        beside = [k for k, (b, e) in spans.items() if k != heavy and b < end and begin < e]
        alive = most_at_once({k: (max(b, begin), min(e, end)) for k, (b, e) in spans.items()
                              if k == heavy or k in beside})
        assert alive <= 2
    assert most_at_once({k: v for k, v in spans.items() if k not in (units.stem(n) for n in names[:2])}) == 4


def test_an_explicit_job_count_lifts_the_gpu_bound_but_still_scales(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(q.QUALIFY_JOBS_ENV, "4")
    monkeypatch.delenv(q.UNITS_PER_GPU_ENV, raising=False)
    fake = DeviceUnits()
    names, out = unit_plan(tmp_path, [150] * 4)
    args = SimpleNamespace(timeout=10000, test_timeout=200, vendor="cuda", numeric_mode="identical",
                           unit_plan=tmp_path / "plan.json", jobs=None)
    q.run_units(args, list(names), out=out, tests=tmp_path / "frozen", python="python3",
                env={"MOJOLEARN_GENERATION_DEVICE": "gpu"}, payload={}, test_commit="c" * 40,
                bounded=fake, available=lambda: 64 * 2**30, cores=48, admit_poll=0.01, gpus=["0"])
    assert "units-per-gpu=off (explicit job count)" in capsys.readouterr().out
    assert most_at_once(fake.spans) == 4
    receipt, _ = units.load_receipt(units.receipt_path(tmp_path / "qualify/units", names[0]))
    assert receipt["concurrency"] == 4 and receipt["scaled_budget_s"] == 600


def test_the_cpu_backend_shares_no_gpu(tmp_path, monkeypatch):
    monkeypatch.delenv(q.QUALIFY_JOBS_ENV, raising=False)
    fake = DeviceUnits(seconds=0.1)
    names, out = unit_plan(tmp_path, [150] * 3)
    args = SimpleNamespace(timeout=10000, test_timeout=200, vendor="cpu", numeric_mode="identical",
                           unit_plan=tmp_path / "plan.json", jobs=None)
    q.run_units(args, list(names), out=out, tests=tmp_path / "frozen", python="python3",
                env={"MOJOLEARN_GENERATION_DEVICE": "cpu"}, payload={}, test_commit="c" * 40,
                bounded=fake, available=lambda: 64 * 2**30, cores=8, admit_poll=0.01, gpus=[])
    assert most_at_once(fake.spans) == 3
    receipt, _ = units.load_receipt(units.receipt_path(tmp_path / "qualify/units", names[0]))
    assert receipt["concurrency"] == 1 and receipt["scaled_budget_s"] == 150


def test_an_over_budget_reason_names_the_scaled_bound(tmp_path, monkeypatch):
    def bounded(command, **kw):
        import time
        time.sleep(0.2)
        run_limit, _ = kw["limits"]()
        return -9, dict(kind="run", budget_s=run_limit, elapsed_s=241.0), 241.0
    names, results = run_on_gpus(tmp_path, monkeypatch, ["0"], [(120, 40.0)] * 2, bounded)
    assert results[0]["reason"].startswith(
        f"unit {names[0]} over its budget of 240 s (120 s serial x 2 units sharing its GPU) (stopped after 241.0 s)")


def test_worker_count_is_capped_by_gpus_times_units_per_gpu(monkeypatch):
    monkeypatch.delenv(q.QUALIFY_JOBS_ENV, raising=False)
    assert q.unit_workers(73, cores=48, free=None, gpus=1, per_gpu=2)[0] == 2
    assert q.unit_workers(73, cores=48, free=None, gpus=4, per_gpu=2)[0] == 8
    assert q.unit_workers(73, cores=6, free=None, gpus=4, per_gpu=2)[0] == 6
    assert q.unit_workers(73, cores=48, free=None, gpus=1, per_gpu=2, env={q.QUALIFY_JOBS_ENV: "5"})[0] == 5
    assert q.units_per_gpu({}) == 2 and q.units_per_gpu({q.UNITS_PER_GPU_ENV: "6"}) == 6
    with pytest.raises(ValueError):
        q.units_per_gpu({q.UNITS_PER_GPU_ENV: "0"})


def test_host_gpus_counts_the_visible_devices():
    def smi(listing):
        return lambda command, **kw: SimpleNamespace(stdout=listing.get(command[0], ""))

    def missing(command, **kw):
        raise FileNotFoundError(command[0])
    nvidia = "GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-x)\nGPU 1: NVIDIA GeForce RTX 4090 (UUID: GPU-y)\n"
    assert q.host_gpus("cuda", env={}, run=smi({"nvidia-smi": nvidia})) == (["0", "1"], "nvidia-smi")
    assert q.host_gpus("cuda", env={"CUDA_VISIBLE_DEVICES": "2,3"}, run=missing) == (["2", "3"],
                                                                                    "CUDA_VISIBLE_DEVICES")
    rocm = "device,GPU ID\ncard0,0x74a1\ncard1,0x74a1\ncard2,0x74a1\n"
    assert q.host_gpus("hip", env={}, run=smi({"rocm-smi": rocm})) == (["0", "1", "2"], "rocm-smi")
    assert q.host_gpus("hip", env={}, run=smi({"amd-smi": "GPU: 0\n  BDF: x\nGPU: 1\n"}))[0] == ["0", "1"]
    assert q.host_gpus("hip", env={"ROCR_VISIBLE_DEVICES": "5"}, run=missing) == (["0"], "ROCR_VISIBLE_DEVICES")
    assert q.host_gpus("cuda", env={}, run=missing) == (["0"], "GPU count unreadable, one assumed")
    assert q.host_gpus("metal", env={}, run=missing) == (["0"], "Apple GPU")
    assert q.host_gpus("cpu", env={}, run=missing) == (None, "cpu backend")


def test_last_seconds_of_a_shared_gpu_run_is_a_serial_bound():
    shared = dict(seconds=90.0, stopped=None, concurrency=3)
    assert units.last_seconds([(shared, None)]) == 30.0
    assert units.last_seconds([(dict(seconds=90.0, stopped=None), None)]) == 90.0  # older receipts: serial


def test_free_memory_is_the_cgroup_headroom_when_lower(monkeypatch):
    import build_sizing
    import mac_slot
    monkeypatch.setattr(mac_slot, "available_bytes", lambda: 64 * 2**30)
    monkeypatch.setattr(build_sizing, "cgroup_memory_headroom", lambda: 8 * 2**30)
    assert q.free_memory_bytes() == 8 * 2**30
    monkeypatch.setattr(build_sizing, "cgroup_memory_headroom", lambda: None)
    assert q.free_memory_bytes() == 64 * 2**30


def test_busy_cores_of_total_are_printed_every_period():
    lines = []
    sampler = q.BusySampler(10, lambda: 7, every=0.02, sample=lambda: 9.5,
                            say=lambda text, flush=False: lines.append(text))
    sampler.start()
    import time
    time.sleep(0.15)
    sampler.stop()
    assert lines and lines[0].startswith("QUALIFY busy cores=9.5 of 10 running=7 at ")
    assert sampler.samples[0]["of"] == 10


def test_run_bounded_measures_the_worker_peak_rss(tmp_path):
    usage = {}
    with (tmp_path / "log").open("wb") as log:
        code, stop, _ = q.run_bounded([sys.executable, "-c", "x = bytearray(64 * 2**20)"], cwd=tmp_path, env=None,
                                      log=log, progress=tmp_path / "progress", run_timeout=60, test_timeout=60,
                                      poll=0.05, usage=usage)
    assert (code, stop) == (0, None) and usage["peak_rss_bytes"] >= 64 * 2**20


# ---------------------------------------------------------------- whole runs
def test_resume_after_interruption_runs_only_the_missing_units(tmp_path, repo):
    commit = with_tests(repo)
    wheel = make_wheel(tmp_path / "w" / "mojolearn_core-0.9.1-py3-none-manylinux_2_28_x86_64.whl")
    store = tmp_path / "store"
    crash = fake_qualifier(tmp_path, crash_after=10)
    assert runtime.run(run_args(repo, commit, wheel, tmp_path / "r1", store), closure_of=fake_closure(),
                       launch=crash) == 1
    first = report_of(tmp_path / "r1")
    assert first["status"] == "failed" and first["units"] == dict(reused=0, run=len(SELECTED),
                                                                  failed=len(SELECTED) - 10)
    assert all(row["reason"] == "no unit receipt" for row in first["failed_units"])
    again = fake_qualifier(tmp_path)
    assert runtime.run(run_args(repo, commit, wheel, tmp_path / "r2", store), closure_of=fake_closure(),
                       launch=again) == 0
    assert again.calls == [SELECTED[10:]]
    second = report_of(tmp_path / "r2")
    assert second["status"] == "passed" and second["units"] == dict(reused=10, run=len(SELECTED) - 10, failed=0)
    rules = {row["rule"] for row in second["unit_rows"] if row["decision"] == "reuse"}
    assert rules == {"wheel-bytes"}
    # a third run with nothing changed runs nothing
    idle = fake_qualifier(tmp_path)
    assert runtime.run(run_args(repo, commit, wheel, tmp_path / "r3", store), closure_of=fake_closure(),
                       launch=idle) == 0
    assert idle.calls == [] and report_of(tmp_path / "r3")["units"]["reused"] == len(SELECTED)
    # --redo runs everything again
    redo = fake_qualifier(tmp_path)
    args = run_args(repo, commit, wheel, tmp_path / "r4", store)
    args.redo = True
    assert runtime.run(args, closure_of=fake_closure(), launch=redo) == 0 and redo.calls == [SELECTED]


def test_mixed_reused_and_new_units_are_one_report_the_ledger_admits(tmp_path, repo, capsys):
    commit = with_tests(repo)
    old = make_wheel(tmp_path / "old" / "mojolearn_core-0.9.1-py3-none-manylinux_2_28_x86_64.whl")
    store = tmp_path / "store"
    assert runtime.run(run_args(repo, commit, old, tmp_path / "r1", store), closure_of=fake_closure(),
                       launch=fake_qualifier(tmp_path)) == 0
    # new wheel bytes (another __init__), the same native module; six units' closures changed
    new = make_wheel(tmp_path / "new" / old.name, init=b"api = 2\n")
    changed = SELECTED[:6]
    launch = fake_qualifier(tmp_path)
    capsys.readouterr()
    assert runtime.run(run_args(repo, commit, new, tmp_path / "r2", store),
                       closure_of=fake_closure(changed=changed, version="2"), launch=launch) == 0
    assert launch.calls == [changed]
    printed = capsys.readouterr().out
    assert f"RUNTIME units reused={len(SELECTED) - 6} run=6 failed=0" in printed
    out = tmp_path / "r2"
    report = report_of(out)
    assert report["origin"] == "per-unit-receipts" and report["status"] == "passed"
    assert {row["rule"] for row in report["unit_rows"] if row["decision"] == "reuse"} == {"closure"}
    assert report["qualifier_exit"] == 0 and report["wheels"] == {new.name: sha(new)}
    ledger, rows = ledger_for(tmp_path, repo, commit, {"runtime-nvidia"}, configuration_for({new.name: sha(new)}))
    review, result = admit(tmp_path, repo, ledger, "installed-runtime", out / runtime.REPORT)
    assert review["admitted"] == ["runtime-nvidia"]
    assert ledger.gate(stage="pre") == (True, [])
    # a reused unit's retained junit altered after the fact: the report no longer re-derives
    receipt = json.loads(units.receipt_path(out / "qualify/units", SELECTED[-1]).read_text())
    junit = out / "qualify/units" / receipt["junit"]
    junit.write_bytes(junit.read_bytes().replace(b"test_case", b"test_kase"))
    assert runtime.summarize(out)["status"] == "failed"
    ledger, _ = ledger_for(tmp_path / "b", repo, commit, {"runtime-nvidia"}, configuration_for({new.name: sha(new)}))
    with pytest.raises(ValueError, match="differs"):
        admission.prepare("installed-runtime", repo, ledger.path, tmp_path / "b/review", dict(report=out / runtime.REPORT))


def test_a_failed_unit_fails_the_report_and_is_named(tmp_path, repo):
    commit = with_tests(repo)
    wheel = make_wheel(tmp_path / "w" / "mojolearn_core-0.9.1-py3-none-manylinux_2_28_x86_64.whl")
    assert runtime.run(run_args(repo, commit, wheel, tmp_path / "r1", tmp_path / "store"),
                       closure_of=fake_closure(), launch=fake_qualifier(tmp_path, failing=(SELECTED[3],))) == 1
    report = report_of(tmp_path / "r1")
    assert report["status"] == "failed" and [r["unit"] for r in report["failed_units"]] == [SELECTED[3]]


# ---------------------------------------------------------------- smoke and qualify
def test_smoke_and_qualify_are_separate_tiers(tmp_path, repo):
    commit = with_tests(repo)
    wheel = make_wheel(tmp_path / "w" / "mojolearn_core-0.9.1-py3-none-manylinux_2_28_x86_64.whl")
    runtime.run(run_args(repo, commit, wheel, tmp_path / "r1", None), closure_of=fake_closure(),
                launch=fake_qualifier(tmp_path))
    plan = json.loads((tmp_path / "r1/qualify/units.json").read_text())
    assert plan["tiers"] == {"smoke": ["gate-installed-smoke"], "qualify": SELECTED}
    assert "eleven" in plan["smoke"] and set(units.SMOKE) == set(runtime.BACKEND)
    assert all(row["tier"] == "qualify" for row in plan["units"])
    import release_gate_profiles
    assert len(release_gate_profiles.CORRECTNESS_JOBS) == 11  # the smoke every gate runs, never reused


# ---------------------------------------------------------------- producers on the gate path
def test_the_gate_job_produces_runtime_vendor_and_cpu_with_unit_receipts():
    text = (RG.ROOT / RG.JOB_SCRIPT).read_text()
    assert 'run_runtime "$GATE_VENDOR" "$GATE_VENDOR-$GATE_TARGET"' in text
    assert 'run_runtime cpu "cpu-linux-$(uname -m)"' in text and 'GATE_RUNTIME_CPU' in text
    assert '--unit-store "$RES/out/runtime-units-$class"' in text and "--prior-units" in text
    assert "grep -q -- '--unit-store' tools/release_runtime_installed.py" in text  # an older source still runs


def test_gate_env_runs_cpu_on_nvidia_and_ships_prior_receipts(tmp_path):
    from test_release_gate_jobs import args
    nvidia = RG.Runner(args(tmp_path, "nvidia")).make_plan()[0]
    amd = RG.Runner(args(tmp_path, "amd")).make_plan()[0]
    assert "GATE_RUNTIME_CPU=1" in nvidia.env_text() and "GATE_RUNTIME_CPU=0" in amd.env_text()
    assert not any(r.endswith(RG.RUNTIME_PRIOR_TGZ) for _, r, _ in nvidia.uploads())
    staged = tmp_path / "prior"
    receipt, junit = prior(tmp_path, closure={"tests_sha256": "t"})
    units.store_receipt(staged / "nvidia-sm_89", receipt, junit)
    a = args(tmp_path, "nvidia")
    a.runtime_prior = str(staged)
    plan = RG.Runner(a).make_plan()[0]
    assert any(r.endswith("/in/" + RG.RUNTIME_PRIOR_TGZ) for _, r, _ in plan.uploads())
    one = RG.write_runtime_prior(staged, tmp_path / "s1")[1]
    assert one == RG.write_runtime_prior(staged, tmp_path / "s2")[1]  # deterministic bytes for gate.env


def test_stage_prior_keeps_the_newest_passing_receipt_per_class(tmp_path):
    for status, when in (("passed", "2026-11-01"), ("failed", "2026-11-03"), ("passed", "2026-10-01")):
        receipt, junit = prior(tmp_path, closure={"tests_sha256": "t"}, status=status)
        receipt["finished_at"] = when
        units.store_receipt(tmp_path / "store", receipt, junit)
    apple, junit = prior(tmp_path, closure={"tests_sha256": "t"})
    apple.update(vendor="apple", device_class="apple-m2-pro")
    units.store_receipt(tmp_path / "store", apple, junit)
    assert units.stage_prior([tmp_path / "store"], tmp_path / "staged", ("nvidia", "cpu")) == 2
    staged = units.collect([tmp_path / "staged"])["f.py"]
    assert [(r["status"], r["finished_at"]) for r, _ in staged] == [("failed", "2026-11-03"),
                                                                    ("passed", "2026-11-01")]
    assert not (tmp_path / "staged" / "apple-m2-pro").exists()


class FakeRelease(release.Release):
    rel = commit = version = dry = base = None


def apple_release(tmp_path):
    r = object.__new__(FakeRelease)
    r.dry, r.version, r.commit = False, "0.9.1", "c" * 40
    r.base = tmp_path / "release" / "0.9.1"
    r.rel = r.base / ("c" * 12)
    r.args = SimpleNamespace()
    wheel = tmp_path / "mojolearn-0.9.1-py3-none-macosx_11_0_arm64.whl"
    wheel.write_bytes(b"wheel")
    r.macos_wheel = lambda: wheel
    r.smoke_python = lambda: "/usr/bin/python3.12"
    r.source_checkout = lambda create=True: tmp_path / "source"
    r.apple_device_class = lambda: "apple-m2-pro"
    r.say = lambda line: None
    r.calls = []

    def must(cmd, **kw):
        cmd = [str(c) for c in cmd]
        r.calls.append(cmd)
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True)
        put(out / "runtime-report.json", dict(status="passed", exit_code=0, cases=7300, vendor="apple",
                                             wheels={wheel.name: sha(wheel)}, units=dict(reused=70, run=3)))
    r.must = must
    return r, wheel


def test_gate_apple_produces_runtime_apple_from_unit_receipts(tmp_path):
    r, wheel = apple_release(tmp_path)
    older = tmp_path / "release" / "0.8.65" / ("b" * 12) / "gate-apple" / "runtime-units-apple"
    older.mkdir(parents=True)
    why = r.apple_runtime_gate()
    assert why.startswith("runtime-apple PASSED: 7300 installed-runtime cases (units reused=70 run=3)")
    cmd = r.calls[0]
    assert "metal" in cmd and cmd[cmd.index("--vendor") + 1] == "apple"
    assert cmd[cmd.index("--device-class") + 1] == "apple-m2-pro"
    assert cmd[cmd.index("--unit-store") + 1] == str(r.apple_runtime_store())
    assert str(older) in cmd  # earlier freezes' Apple receipts are reuse inputs
    assert r.apple_runtime_gate().endswith("(receipt exists)") and len(r.calls) == 1
    wheel.write_bytes(b"another wheel")
    assert not r.apple_runtime_ok()[0]


def test_the_ledger_producers_name_runtime_cpu_and_runtime_apple(tmp_path):
    r, _ = apple_release(tmp_path)
    r.macos_wheel = lambda: None
    r.macos_bincache = lambda: tmp_path / "no-cache"
    r.retained_leg_dirs = lambda: []
    for report in (r.rel / "gate-nvidia/stamp/runtime-cpu/runtime-report.json",
                   r.rel / "gate-nvidia/stamp/runtime-nvidia/runtime-report.json",
                   r.apple_runtime_out() / "runtime-report.json"):
        put(report, {})
    produced = [p["report"] for kind, p in r.retained_producers() if kind == "installed-runtime"]
    assert sorted(Path(p).parent.name for p in produced) == ["runtime-apple", "runtime-cpu", "runtime-nvidia"]
