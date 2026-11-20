# SPDX-License-Identifier: Apache-2.0
"""The "what ran" block of the shared GPU/CPU check and its fit-on-both-sides form, at the Python boundary.

No binding and no GPU. Two stub installs:

- in process: `mojolearn` is a namespace whose `activity_door()` hands a stub of the runtime's activity record
  (core/activity_record.mojo, read through `activity_begin` / `activity_end` of the core host binding). The stub
  "GPU" estimator counts device launches and copies into it, the stub "CPU" model counts a host routine and a
  thread split, so the two sides of one check carry records that are non-empty and different.
- on disk: a stub `mojolearn` package (with a copy of the real `_check_both.py`) that a real child process imports
  under `MOJOLEARN_VENDOR=cpu`, for the fit-on-both-sides path.

What is held: the record is read once per side and only when asked for, the printer's bounds, the lines of each
side, a call's behaviour without the record, the child-process path (match, mismatch, the product's own refusal as
NOT RUN, the row cap and the warning), and the few source facts the cost rule rests on.
"""
import importlib.util
import json
import shutil
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "python/mojolearn"

_spec = importlib.util.spec_from_file_location("check_both_trace_under_test", PKG / "_check_both.py")
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)

LABELS = dict(describe_device=lambda: "NVIDIA Test GPU, 550.1", cpu_model=lambda: "Test CPU 16-Core")


# ---------------------------------------------------------------- the in-process stub install

class Runtime:
    """The stub of the runtime's record: counts only between begin and end."""

    def __init__(self):
        self.counting = False
        self.begins = self.ends = 0
        self._clear()

    def _clear(self):
        self.c = dict(launches=0, upload_calls=0, upload_bytes=0, download_calls=0, download_bytes=0,
                      host_splits=0, host_tasks=0, host_tasks_widest=0, host_entries=0, names_dropped=0)
        self.launch_names, self.host_names = {}, {}

    def activity_begin(self):
        self.begins += 1
        self._clear()
        self.counting = True
        return True

    def activity_end(self):
        self.ends += 1
        self.counting = False
        record = dict(self.c, format=cb.ACTIVITY_FORMAT,
                      launch_names=[dict(name=n, count=k) for n, k in self.launch_names.items()],
                      host_names=[dict(name=n, count=k) for n, k in self.host_names.items()])
        self._clear()
        return record

    def launch(self, name, times=1):
        if self.counting:
            self.c["launches"] += times
            self.launch_names[name] = self.launch_names.get(name, 0) + times

    def copy(self, up=0, down=0):
        if self.counting:
            self.c["upload_calls"] += bool(up)
            self.c["upload_bytes"] += up
            self.c["download_calls"] += bool(down)
            self.c["download_bytes"] += down

    def host(self, name, tasks):
        if self.counting:
            self.c["host_entries"] += 1
            self.host_names[name] = self.host_names.get(name, 0) + 1
            self.c["host_splits"] += 1
            self.c["host_tasks"] += tasks
            self.c["host_tasks_widest"] = max(self.c["host_tasks_widest"], tasks)


class GpuModel:
    numeric_mode = "identical"

    def __init__(self, runtime, coef, kernels=("lin_rows", "lin_fold")):
        self.runtime, self.coef, self.kernels = runtime, coef, kernels

    def predict(self, X):
        self.runtime.copy(up=X.nbytes)
        for k, name in enumerate(self.kernels):
            self.runtime.launch(name, times=k + 1)
        self.runtime.copy(down=4 * len(X))
        return (X @ self.coef).astype(np.float32)

    def save(self, path):
        np.savez(path, coef=self.coef)


class CpuModel:
    def __init__(self, runtime, coef):
        self.runtime, self.coef = runtime, coef

    def predict(self, X):
        self.runtime.host("lin_host_predict", tasks=16)
        return (X @ self.coef).astype(np.float32)


def install(tmp_path, door=True):
    runtime = Runtime()
    pkg = str(tmp_path / "site/mojolearn")
    report = dict(version="9.9.9", numeric_mode="identical", vendor="cuda", gpu_arch="sm_89", code_kind="native",
                  package_dir=pkg, bindings={
                      "mojolearn._mojolearn": dict(path=f"{pkg}/cuda/sm_89/identical/_mojolearn.so", loaded=True),
                      "mojolearn._host._mojolearn_estimators_host": dict(
                          path=f"{pkg}/host/_mojolearn_estimators_host.so", loaded=True)})
    ml = SimpleNamespace(vendor=lambda: "cuda", numeric_mode=lambda: "identical", runtime_report=lambda: report,
                         host_model=lambda path: CpuModel(runtime, np.load(path)["coef"]))
    if door:
        ml.activity_door = lambda: runtime
    return ml, runtime


def rows():
    return np.random.default_rng(11).random((64, 3), dtype=np.float32)


COEF = np.array([0.5, -1.25, 2.0], dtype=np.float32)


# ---------------------------------------------------------------- the record of each side

def test_each_side_carries_what_the_runtime_counted_and_the_two_records_differ(tmp_path):
    ml, runtime = install(tmp_path)
    X = rows()
    result = cb.check_estimator(ml, GpuModel(runtime, COEF), X, **LABELS)
    assert result["ran"] and result["verdict"]["match"]
    gpu, cpu = result["sides"]["gpu"]["activity"], result["sides"]["cpu"]["activity"]
    assert gpu != cpu
    assert gpu["launches"] == 3 and gpu["launch_names"] == [["lin_fold", 2], ["lin_rows", 1]]
    assert gpu["upload_bytes"] == X.nbytes and gpu["download_bytes"] == 4 * len(X) and gpu["host_entries"] == 0
    assert cpu["launches"] == 0 and cpu["upload_bytes"] == 0 and cpu["download_bytes"] == 0
    assert cpu["host_names"] == [["lin_host_predict", 1]] and cpu["host_tasks_widest"] == 16
    assert (runtime.begins, runtime.ends) == (2, 2), "the record is read once per side"
    json.dumps(result)


def test_the_report_shows_different_work_and_the_same_bytes(tmp_path):
    ml, runtime = install(tmp_path)
    text = cb.format_check(cb.check_estimator(ml, GpuModel(runtime, COEF), rows(), **LABELS))
    gpu_block, cpu_block = text.split("side      cpu")[0], text.split("side      cpu")[1].split("first values")[0]
    assert "  launches  3 device launches at named sites: lin_fold x2, lin_rows x1" in gpu_block
    assert "  copies    768 B to the device in 1, 256 B back in 1 (shared transfer entries)" in gpu_block
    assert "host" not in gpu_block.split("  launches  ")[1]
    assert "  launches  0 device launches, 0 bytes to the device, 0 bytes back" in cpu_block
    assert "  host      routines: lin_host_predict x1" in cpu_block
    assert "  threads   1 host thread splits, 16 tasks, widest 16" in cpu_block
    side_by_side = text.split("first values, side by side\n")[1].split("\n")
    assert side_by_side[0].startswith("  output 0  gpu  ") and side_by_side[1].startswith("            cpu  ")
    assert side_by_side[0].split("gpu  ")[1] == side_by_side[1].split("cpu  ")[1]
    assert text.rstrip().split("\n")[-1].startswith("MATCH: 1 output(s), 64 element(s)")


def test_the_block_is_bounded(tmp_path):
    ml, runtime = install(tmp_path)
    many = tuple(f"kernel_{k:02d}" for k in range(40))
    result = cb.check_estimator(ml, GpuModel(runtime, COEF, kernels=many), rows(), **LABELS)
    text = cb.format_check(result)
    for side in cb.SIDES:
        record = result["sides"][side]
        assert len(cb.activity_lines(side, record["activity"])) <= 4
        block = cb.environment_lines(record) + cb.activity_lines(side, record["activity"]) \
            + cb.output_lines(record, values=False)
        assert len(block) <= 10, block
    launches = next(line for line in text.split("\n") if line.startswith("  launches  820 "))
    assert launches.count(" x") == cb.TOP_NAMES and launches.endswith("(+36 more)")
    assert "kernel_39 x40, kernel_38 x39" in launches, "the names with the most launches come first"


def test_a_gpu_path_with_no_counted_door_says_so_and_prints_no_zero(tmp_path):
    lines = cb.activity_lines("gpu", dict(launches=0, upload_calls=0, upload_bytes=0, download_calls=0,
                                          download_bytes=0, host_splits=2, host_tasks=8, host_tasks_widest=4,
                                          host_entries=0, names_dropped=0, launch_names=[], host_names=[]))
    assert lines == ["  launches  not counted on this path (its device work passes no counted door)",
                     "  threads   2 host thread splits, 8 tasks, widest 4"]
    assert cb.activity_lines("gpu", None) == [] and cb.activity_line("gpu", None) == ""


def test_records_sum_per_lane_and_the_widest_split_is_the_maximum():
    a = dict(launches=3, upload_calls=1, upload_bytes=10, download_calls=1, download_bytes=4, host_splits=1,
             host_tasks=4, host_tasks_widest=4, host_entries=0, names_dropped=0,
             launch_names=[["a", 2], ["b", 1]], host_names=[])
    b = dict(a, launches=5, host_tasks_widest=9, launch_names=[["b", 4], ["c", 1]])
    total = cb.activity_merge(cb.activity_merge(None, a), b)
    assert total["launches"] == 8 and total["upload_bytes"] == 20 and total["host_tasks_widest"] == 9
    assert total["launch_names"] == [["b", 5], ["a", 2], ["c", 1]]
    assert a["launch_names"] == [["a", 2], ["b", 1]], "a merge does not write into its arguments"
    assert cb.activity_merge(total, None) is total
    assert cb.activity_line("gpu", total).startswith("8 device launches at named sites: b x5, a x2, c x1; ")


# ---------------------------------------------------------------- off by default

def test_a_call_without_the_package_never_touches_the_record(tmp_path):
    ml, runtime = install(tmp_path)
    X = rows()
    record, arrays = cb.run_side("gpu", lambda: GpuModel(runtime, COEF).predict(X))
    assert "activity" not in record and (runtime.begins, runtime.ends) == (0, 0)
    counted, counted_arrays = cb.run_side("gpu", lambda: GpuModel(runtime, COEF).predict(X), ml=ml)
    assert counted["digest"] == record["digest"] and counted["outputs"] == record["outputs"]
    assert np.array_equal(arrays[0], counted_arrays[0]), "counting changes no output"


def test_an_install_without_the_record_checks_as_before_and_prints_no_activity_line(tmp_path):
    ml, runtime = install(tmp_path, door=False)
    result = cb.check_estimator(ml, GpuModel(runtime, COEF), rows(), **LABELS)
    assert result["ran"] and result["verdict"]["match"]
    assert result["sides"]["gpu"]["activity"] is None and result["sides"]["cpu"]["activity"] is None
    text = cb.format_check(result)
    assert "launch" not in text and "copies" not in text and "threads" not in text
    assert (runtime.begins, runtime.ends) == (0, 0)


def test_a_raising_call_still_stops_the_record(tmp_path):
    ml, runtime = install(tmp_path)

    def call():
        raise RuntimeError("the call raised")

    with pytest.raises(RuntimeError):
        cb.run_side("gpu", call, ml=ml)
    assert (runtime.begins, runtime.ends) == (1, 1) and not runtime.counting


def test_a_record_in_another_format_is_not_printed(tmp_path):
    ml, runtime = install(tmp_path)
    runtime.activity_end = lambda: dict(format="something-else/9", launches=7)
    record, _ = cb.run_side("gpu", lambda: np.zeros(2), ml=ml)
    assert record["activity"] is None


# ---------------------------------------------------------------- fit on both sides: a real child process

STUB_PACKAGE = '''
    import os
    import numpy as np

    def vendor():
        return os.environ.get("MOJOLEARN_VENDOR", "cuda")

    def numeric_mode():
        return "identical"

    def runtime_report():
        pkg = os.path.dirname(__file__)
        where = "host" if vendor() == "cpu" else "cuda/sm_89/identical"
        name = "mojolearn._host._mojolearn_linear_host" if vendor() == "cpu" else "mojolearn._mojolearn_linear"
        return dict(version="9.9.9", numeric_mode="identical", vendor=vendor(), gpu_arch="sm_89",
                    code_kind="native", package_dir=pkg,
                    bindings={name: dict(path=os.path.join(pkg, where, "lin.so"), loaded=True)})

    class _Runtime:
        def __init__(self):
            self.rows = dict(launches=0, host_splits=0, host_tasks=0, host_tasks_widest=0)

        def activity_begin(self):
            self.__init__()
            return True

        def activity_end(self):
            record = dict(format="mojolearn-activity-record/1", upload_calls=0, upload_bytes=0, download_calls=0,
                          download_bytes=0, host_entries=0, names_dropped=0, host_names=[],
                          launch_names=[dict(name="lin_fit", count=self.rows["launches"])] if self.rows["launches"]
                          else [], **self.rows)
            self.__init__()
            return record

    _RUNTIME = _Runtime()

    def activity_door():
        return _RUNTIME

    class Lin:
        """Least squares; on the stub CPU it can be made to answer one ULP off, or to refuse."""
        numeric_mode = "identical"

        def __init__(self, scale=1.0, cpu="same"):
            self.scale, self.cpu = scale, cpu

        def get_params(self, deep=True):
            return dict(scale=self.scale, cpu=self.cpu)

        def fit(self, X, y):
            if vendor() == "cpu":
                if self.cpu == "refuse":
                    raise RuntimeError("Lin(cpu='refuse') has no CPU path; it runs on a GPU installation only")
                _RUNTIME.rows.update(host_splits=1, host_tasks=4, host_tasks_widest=4)
            else:
                _RUNTIME.rows["launches"] += 5
            self.rows_fitted = len(X)
            self.coef = (np.linalg.lstsq(X.astype(np.float64), y.astype(np.float64), rcond=None)[0]
                         * self.scale).astype(np.float32)
            return self

        def predict(self, X):
            y = (X @ self.coef).astype(np.float32)
            if vendor() == "cpu" and self.cpu == "one-ulp":
                y[3] = np.nextafter(y[3], np.float32(np.inf))
            return y
'''


@pytest.fixture
def stub(tmp_path, monkeypatch):
    site = tmp_path / "site"
    (site / "mojolearn").mkdir(parents=True)
    (site / "mojolearn/__init__.py").write_text(textwrap.dedent(STUB_PACKAGE))
    shutil.copy(PKG / "_check_both.py", site / "mojolearn/_check_both.py")
    monkeypatch.setenv("PYTHONPATH", str(site))
    monkeypatch.delenv("MOJOLEARN_VENDOR", raising=False)
    spec = importlib.util.spec_from_file_location("stub_mojolearn_parent", site / "mojolearn/__init__.py")
    ml = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ml)
    return ml


def training():
    rng = np.random.default_rng(5)
    X = rng.random((40, 3), dtype=np.float32)
    return X, (X @ COEF).astype(np.float32)


def test_fit_on_both_sides_runs_the_cpu_fit_in_a_child_process_and_matches(stub):
    Xt, yt = training()
    est = stub.Lin(scale=2.0)
    result = cb.check_estimator(stub, est, rows(), fit=(Xt, yt), fit_both=True, **LABELS)
    assert result["ran"] and result["form"] == "fit-both" and result["verdict"]["match"], result
    gpu, cpu = result["sides"]["gpu"], result["sides"]["cpu"]
    assert gpu["digest"] == cpu["digest"] and gpu["outputs"] == cpu["outputs"]
    assert gpu["binaries"]["directories"] == ["cuda/sm_89/identical"] and cpu["binaries"]["directories"] == ["host"]
    assert gpu["activity"]["launches"] == 5 and gpu["activity"]["host_splits"] == 0
    assert cpu["activity"]["launches"] == 0 and cpu["activity"]["host_splits"] == 1
    assert cpu["fit_wall_s"] >= 0 and cpu["wall_s"] >= 0 and cpu["process_wall_s"] > 0
    assert est.rows_fitted == 40 and result["fit_rows"] == result["fit_rows_given"] == 40 and "warning" not in result
    text = cb.format_check(result)
    for needle in ("Fitted on each side: on the GPU in this process, on the CPU in a child process",
                   "  launches  5 device launches at named sites: lin_fit x5",
                   "  threads   1 host thread splits, 4 tasks, widest 4", "  process   ", "MATCH:"):
        assert needle in text, needle
    assert text.count("  fit       ") == 2 and text.count("  wall      ") == 2
    json.dumps(result)


def test_a_cpu_fit_that_answers_other_bytes_is_a_mismatch_at_a_named_element(stub):
    Xt, yt = training()
    result = cb.check_estimator(stub, stub.Lin(cpu="one-ulp"), rows(), fit=(Xt, yt), fit_both=True, **LABELS)
    assert result["ran"] and not result["verdict"]["match"]
    assert result["verdict"]["first"]["flat"] == 3 and result["verdict"]["first"]["why"] == "value"
    assert "MISMATCH: output 0, first differing element at index [3]" in cb.format_check(result)


def test_a_class_the_cpu_does_not_run_is_not_run_with_the_products_own_message(stub):
    Xt, yt = training()
    result = cb.check_estimator(stub, stub.Lin(cpu="refuse"), rows(), fit=(Xt, yt), fit_both=True, **LABELS)
    assert result["ran"] is False and "verdict" not in result
    assert "has no CPU path; it runs on a GPU installation only" in result["reason"]
    text = cb.format_check(result)
    assert text.split("\n")[-1].startswith("NOT RUN: the CPU side did not run: RuntimeError: Lin(cpu='refuse')")
    assert "MATCH" not in text


def test_the_row_cap_fits_both_sides_on_the_first_rows_and_a_large_fit_warns(stub, monkeypatch):
    Xt, yt = training()
    est = stub.Lin()
    result = cb.check_estimator(stub, est, rows(), fit=(Xt, yt), fit_both=True, max_rows=16, **LABELS)
    assert result["verdict"]["match"] and est.rows_fitted == 16
    assert (result["fit_rows"], result["fit_rows_given"]) == (16, 40)
    assert "on the first 16 of 40 rows" in cb.format_check(result)
    monkeypatch.setattr(cb, "FIT_BOTH_WARN_ROWS", 10)
    warned = cb.check_estimator(stub, stub.Lin(), rows(), fit=(Xt, yt), fit_both=True, **LABELS)
    assert warned["verdict"]["match"] and "fitting 40 rows on the CPU as well" in warned["warning"]
    assert "WARNING: fitting 40 rows" in cb.format_check(warned)


def test_fit_on_both_sides_is_opt_in_and_says_what_it_needs(stub):
    Xt, yt = training()
    assert "needs the fit arguments" in cb.check_estimator(stub, stub.Lin(), rows(), fit_both=True)["reason"]
    spawned = []
    fitted = stub.Lin().fit(Xt, yt)
    fitted.save = lambda path: np.savez(path, coef=fitted.coef)
    stub.host_model = lambda path: (_ for _ in ()).throw(ValueError("no CPU model for this file"))
    result = cb.check_estimator(stub, fitted, rows(), spawn=lambda d: spawned.append(d), **LABELS)
    assert not spawned, "no child process unless fit_both is asked for"
    assert result["ran"] is False and result["form"] == "fitted-model"
    assert "fit_both=True (--fit-both) fits on each side instead" in result["reason"]

    class Odd(stub.Lin):
        def get_params(self, deep=True):
            return dict(scale=object())

    refused = cb.check_estimator(stub, Odd(), rows(), fit=(Xt, yt), fit_both=True, **LABELS)
    assert refused["ran"] is False and "cannot be sent to the CPU process" in refused["reason"]


def test_a_child_that_ends_without_a_result_is_not_run(stub):
    Xt, yt = training()
    result = cb.check_estimator(stub, stub.Lin(), rows(), fit=(Xt, yt), fit_both=True,
                                spawn=lambda directory: (9, "the process was killed"), **LABELS)
    assert result["ran"] is False and "ended (exit 9) without a result: the process was killed" in result["reason"]


# ---------------------------------------------------------------- the source facts the cost rule rests on

def _text(path):
    return (ROOT / path).read_text()


def test_a_named_launch_site_still_tests_one_flag_when_nothing_is_on():
    log = _text("core/launch_log.mojo")
    body = log.split("def log_launch(name: StringSlice) raises:")[1].split("\ndef ")[0]
    code = [line.strip() for line in body.split('"""')[2].split("\n") if line.strip()]
    assert code[:3] == ["var st = _LAUNCH_LOG.get_or_create_ptr()", "if not st[].watch:", "return"]
    clock = _text("core/launch_clock.mojo").split("def log_launch_ctx(")[1]
    assert clock.index("if not st[].watch:") < clock.index("if st[].counting:") < clock.index("if not st[].enabled:")
    assert 'SharedSlot[_LaunchLogState, "MojolearnRFLaunchLogV2", 2]' in log, "the layout moved with the name"
    assert "self.counting = False" in log and "self.watch = self.enabled" in log, "off unless a verification asks"


def test_the_counted_doors_are_the_central_ones_and_each_reads_the_switch_once():
    split = _text("core/host_parallel.mojo")
    assert split.count("activity_host_split(n)") == 2, "the two host entries, once each"
    task = split.split("def task(i: Int)")[1].split("sync_parallelize(task, n)")[0]
    assert "activity" not in task, "nothing is counted inside a task"
    transfer = _text("core/transfer.mojo")
    assert transfer.count("activity_uploaded(") == 3 and transfer.count("activity_downloaded(") == 3
    for entry in ("activity_uploaded", "activity_downloaded", "activity_host_split", "activity_host"):
        body = _text("core/launch_log.mojo").split(f"def {entry}(")[1].split("\ndef ")[0]
        assert body.count("_LAUNCH_LOG.get_or_create_ptr()") == 1 and "if not st[].counting:" in body, entry
    for path, name in (("core/forest_host_predict.mojo", "rf_host_predict"),
                       ("core/forest_host_predict.mojo", "et_host_predict"),
                       ("core/gbdt_host_predict.mojo", "gbdt_host_predict"),
                       ("core/knn_host_predict.mojo", "host_knn_search")):
        assert _text(path).count(f'activity_host("{name}")') == 1, name


def test_the_door_is_registered_exported_and_called_by_verifys_cell_worker():
    binding = _text("bindings/_mojolearn_core_host.mojo")
    assert 'module.def_function[activity_begin_binding]("activity_begin")' in binding
    assert 'module.def_function[activity_end_binding]("activity_end")' in binding
    assert f'"{cb.ACTIVITY_FORMAT}"' in binding
    assert '"activity_begin", "activity_end",' in _text("python/mojolearn/host_surface.py")
    worker = _text("python/mojolearn/_verify_worker.py")
    assert "check_both.activity_begin(ml)" in worker and "check_both.activity_end(ml) if counting else None" in worker
    verify = _text("python/mojolearn/_verify_all.py")
    assert "cb.activity_line(what['side'], what['lanes'][lane])" in verify and '"what_ran":' in verify
    assert verify.count(", ml=ml)") >= 4, "each cross-check answer is counted"
