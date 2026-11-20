# SPDX-License-Identifier: Apache-2.0
"""The shared GPU/CPU check (python/mojolearn/_check_both.py) and its two callers, at the Python boundary.

No binding and no GPU: the module is loaded by path, `mojolearn` is a stub whose "GPU" estimator and "CPU" model are
two Python objects that share saved state, and verify's `cross_check` / `format_cross_check` and the `verify --check`
command are the real functions, taken from their source files. What is held here is the boundary: a side record
carries what it says, MATCH and MISMATCH are decided by one compare, a MISMATCH names the first differing element,
fast mode and a CPU-only install are refused before anything runs, and verify has no compare or print of its own.
"""
import ast
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "python/mojolearn"

_spec = importlib.util.spec_from_file_location("check_both_under_test", PKG / "_check_both.py")
cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cb)


def source_functions(path, names, namespace):
    tree = ast.parse((PKG / path).read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(PKG / path), "exec"), namespace)
    return namespace


# ---------------------------------------------------------------- the stub install

class GpuModel:
    """Answers X @ coef; `save` writes the coefficients."""
    numeric_mode = "identical"

    def __init__(self, coef=None):
        self.coef = coef
        self.fits = 0

    def fit(self, X, y):
        self.fits += 1
        self.coef = np.linalg.lstsq(X.astype(np.float64), y.astype(np.float64), rcond=None)[0].astype(np.float32)
        return self

    def predict(self, X):
        return (X @ self.coef).astype(np.float32)

    def save(self, path):
        np.savez(path, coef=self.coef)


class CpuModel:
    def __init__(self, coef, nudge=None):
        self.coef, self.nudge = coef, nudge

    def predict(self, X):
        y = (X @ self.coef).astype(np.float32)
        if self.nudge is not None:
            y[self.nudge] = np.nextafter(y[self.nudge], np.float32(np.inf))   # one ULP, one element
        return y


def install(tmp_path, vendor="cuda", mode="identical", nudge=None, host_loaded=True, host_error=None):
    pkg = str(tmp_path / "site/mojolearn")
    bindings = {
        "mojolearn._mojolearn": dict(path=f"{pkg}/cuda/sm_89/identical/_mojolearn.so", loaded=True),
        "mojolearn._mojolearn_gbdt": dict(path=f"{pkg}/cuda/sm_89/identical/_mojolearn_gbdt.so", loaded=True),
        "mojolearn._mojolearn_unused": dict(path=f"{pkg}/cuda/sm_89/identical/_mojolearn_unused.so", loaded=False),
        "mojolearn._host._mojolearn_estimators_host": dict(path=f"{pkg}/host/_mojolearn_estimators_host.so",
                                                           loaded=host_loaded),
    }
    report = dict(version="9.9.9", numeric_mode=mode, vendor=vendor, gpu_arch="sm_89", code_kind="native",
                  package_dir=pkg, bindings=bindings)

    def host_model(path):
        if host_error:
            raise ValueError(host_error)
        return CpuModel(np.load(path)["coef"], nudge)

    return SimpleNamespace(vendor=lambda: vendor, numeric_mode=lambda: mode, runtime_report=lambda: report,
                           host_model=host_model, GpuModel=SimpleNamespace(
                               load=lambda path: GpuModel(np.load(path)["coef"])))


LABELS = dict(describe_device=lambda: "NVIDIA Test GPU, 550.1", cpu_model=lambda: "Test CPU 16-Core")


def data():
    rng = np.random.default_rng(7)
    X = rng.random((32, 3), dtype=np.float32)
    return X, GpuModel(np.array([0.5, -1.25, 2.0], dtype=np.float32))


# ---------------------------------------------------------------- one side, two sides

def test_a_side_record_carries_wall_time_and_each_outputs_shape_dtype_hash_and_first_values():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    record, arrays = cb.run_side("gpu", lambda: (a, a[:, 0].astype(np.int64)))
    assert record["side"] == "gpu" and record["wall_s"] >= 0 and len(arrays) == 2
    first, second = record["outputs"]
    assert first == dict(shape=[3, 4], dtype="float32", sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                         head=["0.0", "1.0", "2.0", "3.0"])
    assert second["shape"] == [3] and second["dtype"] == "int64" and second["head"] == ["0", "4", "8"]
    assert record["digest"] == cb.outputs_digest(arrays)
    assert set(record) == {"side", "wall_s", "digest", "outputs"}, "nothing per iteration or per kernel"
    with pytest.raises(ValueError):
        cb.run_side("metal", lambda: a)


def test_a_reduced_value_is_a_record_without_arrays_and_a_caller_digest_is_kept():
    record, arrays = cb.run_side("cpu", lambda: "aaaa1111bbbb2222")
    assert arrays is None and record["digest"] == "aaaa1111bbbb2222" and record["outputs"] == []
    record, _ = cb.run_side("cpu", lambda: np.zeros(3), digest=lambda *arrays: "caller-digest")
    assert record["digest"] == "caller-digest"


def pair(g, c):
    gpu, ga = cb.run_side("gpu", lambda: g)
    cpu, ca = cb.run_side("cpu", lambda: c)
    return cb.compare(gpu, cpu, ga, ca)


def test_the_same_bytes_match_and_one_ulp_is_a_mismatch_at_a_named_element():
    a = np.linspace(0, 1, 20, dtype=np.float32).reshape(4, 5)
    same = pair(a, a.copy())
    assert same["match"] and same["verdict"] == "MATCH" and same["elements"] == 20 and same["first"] is None
    assert cb.verdict_line(same).startswith("MATCH: 1 output(s), 20 element(s)")
    b = a.copy()
    b[2, 3] = np.nextafter(b[2, 3], np.float32(2))
    b[3, 4] = 9
    moved = pair(a, b)
    assert not moved["match"] and moved["verdict"] == "MISMATCH"
    assert moved["first"] == dict(why="value", output=0, flat=13, index=[2, 3],
                                  gpu=repr(a[2, 3].item()), cpu=repr(b[2, 3].item()))
    line = cb.verdict_line(moved)
    assert "index [2, 3]" in line and repr(a[2, 3].item()) in line and repr(b[2, 3].item()) in line


def test_bytes_decide_not_values():
    assert not pair(np.array([0.0], dtype=np.float32), np.array([-0.0], dtype=np.float32))["match"]
    assert pair(np.array([np.nan], dtype=np.float32), np.array([np.nan], dtype=np.float32))["match"]


@pytest.mark.parametrize("g,c,why", [
    (np.zeros(3, np.float32), np.zeros(3, np.float64), "dtype"),
    (np.zeros((3, 1), np.float32), np.zeros(3, np.float32), "shape"),
    ((np.zeros(3), np.zeros(3)), np.zeros(3), "count"),
])
def test_a_different_dtype_shape_or_output_count_is_a_mismatch_that_says_which(g, c, why):
    verdict = pair(g, c)
    assert not verdict["match"] and verdict["first"]["why"] == why
    assert "MISMATCH" in cb.verdict_line(verdict)


def test_compare_takes_the_gpu_record_then_the_cpu_record():
    gpu, _ = cb.run_side("gpu", lambda: np.zeros(1))
    with pytest.raises(ValueError):
        cb.compare(gpu, gpu)


# ---------------------------------------------------------------- a user's estimator

def test_a_users_estimator_matches_and_the_report_shows_two_ways_to_one_output(tmp_path):
    X, est = data()
    result = cb.check_estimator(install(tmp_path), est, X, **LABELS)
    assert result["ran"] and result["verdict"]["match"] and result["method"] == "predict"
    assert est.fits == 0, "fit is never run unless asked for"
    gpu, cpu = result["sides"]["gpu"], result["sides"]["cpu"]
    assert gpu["outputs"] == cpu["outputs"] and gpu["digest"] == cpu["digest"]
    assert gpu["binaries"] == dict(directories=["cuda/sm_89/identical"], loaded=2)
    assert cpu["binaries"] == dict(directories=["host"], loaded=1)
    json.dumps(result)   # the result is a plain document
    text = cb.format_check(result)
    for needle in ("side      gpu", "side      cpu", "NVIDIA Test GPU, 550.1 (cuda/sm_89/native)", "Test CPU 16-Core",
                   "mojolearn 9.9.9, identical mode", "cuda/sm_89/identical (2 loaded)", "host (1 loaded)",
                   "shape 32  dtype float32  sha256 " + gpu["outputs"][0]["sha256"], "first values", "MATCH:"):
        assert needle in text, needle
    assert text.count("  wall      ") == 2 and "upload" not in text and "launch" not in text


def test_a_wrong_cpu_answer_is_a_mismatch_with_the_index_and_both_values(tmp_path):
    X, est = data()
    result = cb.check_estimator(install(tmp_path, nudge=5), est, X, **LABELS)
    assert result["ran"] and not result["verdict"]["match"]
    first = result["verdict"]["first"]
    assert first["index"] == [5] and first["gpu"] != first["cpu"]
    text = cb.format_check(result)
    assert "MISMATCH: output 0, first differing element at index [5]" in text
    assert first["gpu"] in text and first["cpu"] in text and "MATCH:" not in text.replace("MISMATCH:", "")


def test_fit_is_opt_in_and_the_fitted_model_is_what_both_sides_answer_from(tmp_path):
    X, _ = data()
    est = GpuModel()
    y = X @ np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = cb.check_estimator(install(tmp_path), est, X, fit=(X, y), **LABELS)
    assert est.fits == 1 and result["ran"] and result["fitted_here"] and result["verdict"]["match"]
    assert "fit_wall_s" in result["sides"]["gpu"] and "fit_wall_s" not in result["sides"]["cpu"]
    assert "Fitted on the GPU" in cb.format_check(result)


def test_fast_is_refused_before_anything_runs(tmp_path):
    X, est = data()
    est.predict = lambda X: pytest.fail("a refused check must not call the estimator")
    result = cb.check_estimator(install(tmp_path, mode="fast"), est, X, **LABELS)
    assert not result["ran"] and result["refused"] == "fast" and "identical mode only" in result["reason"]
    assert cb.format_check(result).splitlines()[-1].startswith("REFUSED: ")
    est.numeric_mode = "fast"
    result = cb.check_estimator(install(tmp_path), est, X, **LABELS)
    assert not result["ran"] and result["refused"] == "fast" and "numeric_mode='fast'" in result["reason"]


def test_a_cpu_only_install_is_not_run_and_never_a_match(tmp_path):
    X, est = data()
    result = cb.check_estimator(install(tmp_path, vendor="cpu"), est, X, **LABELS)
    assert not result["ran"] and result["reason"] == cb.NO_GPU and "verdict" not in result
    text = cb.format_check(result)
    assert "NOT RUN: " in text and "MATCH" not in text


@pytest.mark.parametrize("case,needle", [
    ("no-save", "has no save"),
    ("host-refuses", "the CPU side cannot load this model: ValueError: format refused"),
    ("no-method", "has no method 'transform'"),
    ("host-lacks-method", "has no predict_proba"),
    ("no-cpu-binary", "compare a thing to itself"),
])
def test_a_model_that_cannot_cross_is_not_run_with_the_reason(tmp_path, case, needle):
    X, est = data()
    ml = install(tmp_path, host_error="format refused" if case == "host-refuses" else None,
                 host_loaded=case != "no-cpu-binary")
    method = None
    if case == "no-save":
        est.save = None
    elif case == "no-method":
        method = "transform"
    elif case == "host-lacks-method":
        est.predict_proba = est.predict
        method = "predict_proba"
    result = cb.check_estimator(ml, est, X, method=method, **LABELS)
    assert not result["ran"] and needle in result["reason"] and "refused" not in result
    assert "MATCH" not in cb.format_check(result)


# ---------------------------------------------------------------- verify runs the same module

class Fit(dict):
    est = None
    probe = None


def harness(batch="n/a:no row axis"):
    X = np.random.default_rng(3).random((16, 3), dtype=np.float32)
    held = np.random.default_rng(4).random((8, 3), dtype=np.float32)

    def lane(ml, X, yc, yr, held):
        fit = Fit()
        fit.est = GpuModel(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        fit.probe = lambda est: (est.predict(held),)
        return fit

    def digest(*arrays):
        return hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes() for a in arrays)).hexdigest()[:16]

    def probe_batch(fit, name, ml, Xh, alone, sabotage):
        return (batch, None) if batch.startswith("n/a") else (digest(fit.est.predict(Xh)), None)

    return SimpleNamespace(LANES=dict(lin=lane), fixture=lambda fx: (X, None, None), heldout=lambda fx: held,
                           _save_load=lambda est: ("save", "load", ".npz"), _public_est=lambda lane, est: est,
                           _h=digest, _probe_batch=probe_batch, BATCH_ALONE=4, Fit=Fit,
                           cpu_model=lambda: "Test CPU 16-Core")


def verify_functions(monkeypatch):
    monkeypatch.setattr(cb, "side_environment",
                        lambda ml, side, describe_device=None, cpu_model=None, _real=cb.side_environment:
                        _real(ml, side, describe_device=LABELS["describe_device"], cpu_model=LABELS["cpu_model"]))
    ns = dict(cb=cb, time=time, os=os, vref=SimpleNamespace(VENDOR_CLASS=dict(cuda="nvidia")),
              _require_forest_host_binding=lambda report: None)
    return source_functions("_verify_all.py", ("cross_check", "_cross_check_result", "format_cross_check"), ns)


@pytest.mark.parametrize("batch", ["n/a:no row axis", "rows"])
def test_verifys_cross_check_agrees_through_the_shared_module_and_prints_both_sides(tmp_path, monkeypatch, batch):
    va = verify_functions(monkeypatch)
    result = va["cross_check"](harness(batch), install(tmp_path), ["lin"], ["base"])
    assert result["passed"] is True and result["complete"] and result["differ"] == 0
    infer = next(r for r in result["cells"] if r["part"] == "infer")
    assert infer["agree"] is True and infer["gpu"] == infer["cpu"] and len(infer["gpu"]) == 16
    assert infer["sides"]["gpu"]["outputs"] == infer["sides"]["cpu"]["outputs"]
    batch_row = next(r for r in result["cells"] if r["part"] == "batch")
    assert (batch_row["na"], batch_row["agree"]) == ((True, None) if batch.startswith("n/a") else (False, True))
    assert result["sides"]["gpu"]["binaries"]["directories"] == ["cuda/sm_89/identical"]
    assert result["sides"]["cpu"]["binaries"]["directories"] == ["host"]
    json.dumps(result)
    text = va["format_cross_check"](dict(result, elapsed_s=0.1))
    assert "side      gpu" in text and "side      cpu" in text and "host (1 loaded)" in text
    assert f"gpu={infer['gpu']}  cpu={infer['cpu']}  agree" in text and "YOUR GPU AND YOUR CPU AGREE" in text


def test_verifys_cross_check_can_fail_and_names_the_first_differing_element(tmp_path, monkeypatch):
    va = verify_functions(monkeypatch)
    result = va["cross_check"](harness(), install(tmp_path, nudge=2), ["lin"], ["base"])
    assert result["passed"] is False and result["differ"] == 1
    infer = next(r for r in result["cells"] if r["part"] == "infer")
    assert infer["agree"] is False and infer["gpu"] != infer["cpu"] and infer["first"]["index"] == [2]
    text = va["format_cross_check"](dict(result, elapsed_s=0.1))
    assert "DIFFER" in text and infer["gpu"] in text and infer["cpu"] in text
    assert "first differing element at index [2]" in text and "RESULT: MISMATCH" in text


def test_verifys_cross_check_does_not_run_without_a_gpu(tmp_path, monkeypatch):
    va = verify_functions(monkeypatch)
    result = va["cross_check"](harness(), install(tmp_path, vendor="cpu"), ["lin"], ["base"])
    assert result["ran"] is False and "NOT RUN" in va["format_cross_check"](result)


def test_verify_has_no_compare_or_verdict_words_of_its_own():
    tree = ast.parse((PKG / "_verify_all.py").read_text())
    source = {n.name: ast.unparse(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    run, text = source["cross_check"], source["format_cross_check"]
    assert run.count("cb.run_side(") == 4 and "cb.compare(" in run and "cb.to_cpu(" in run
    assert "harness._h(*" not in run, "a digest computed beside the shared module is a second compare path"
    assert "cb.pair_line(" in text and "cb.sides_lines(" in text and "'DIFFER'" not in text
    check = (PKG / "_verify_check.py").read_text()
    assert "cb.check_estimator(" in check and "cb.format_check" in check and "sha256" not in check.split('"""')[-1]


# ---------------------------------------------------------------- python -m mojolearn verify --check

def command(monkeypatch, ml):
    monkeypatch.setitem(sys.modules, "mojolearn", ml)
    monkeypatch.setattr(cb, "side_environment",
                        lambda ml, side, describe_device=None, cpu_model=None, _real=cb.side_environment:
                        _real(ml, side, describe_device=LABELS["describe_device"], cpu_model=LABELS["cpu_model"]))
    ns = dict(cb=cb, json=json, sys=sys, EXIT_MATCH=0, EXIT_MISMATCH=1, EXIT_USAGE=2, EXIT_REFUSED_FAST=3,
              EXIT_CANNOT_RUN=4)
    return source_functions("_verify_check.py", ("cmd_check", "_load_inputs"), ns)["cmd_check"]


def saved(tmp_path):
    X, est = data()
    model, rows = str(tmp_path / "model.npz"), str(tmp_path / "rows.npy")
    est.save(model)
    np.save(rows, X)
    return model, rows


@pytest.mark.parametrize("kind,code,needle", [
    ("match", 0, "MATCH: 1 output(s), 32 element(s)"),
    ("mismatch", 1, "MISMATCH: output 0, first differing element at index [9]"),
    ("fast", 3, "REFUSED: "),
    ("cpu-only", 4, "NOT RUN: "),
    ("host-refuses", 4, "NOT RUN: the CPU side cannot load this model"),
])
def test_the_command_exits_by_verdict(tmp_path, monkeypatch, capsys, kind, code, needle):
    model, rows = saved(tmp_path)
    ml = install(tmp_path, nudge=9 if kind == "mismatch" else None, mode="fast" if kind == "fast" else "identical",
                 vendor="cpu" if kind == "cpu-only" else "cuda",
                 host_error="format refused" if kind == "host-refuses" else None)
    args = SimpleNamespace(check=model, estimator="GpuModel", inputs=rows, method=None, json=False)
    assert command(monkeypatch, ml)(args) == code
    out = capsys.readouterr().out
    assert needle in out
    if code:
        assert "MATCH:" not in out.replace("MISMATCH:", "")


def test_the_command_refuses_a_missing_or_unknown_estimator_as_usage(tmp_path, monkeypatch, capsys):
    model, rows = saved(tmp_path)
    run = command(monkeypatch, install(tmp_path))
    assert run(SimpleNamespace(check=model, estimator=None, inputs=rows, method=None, json=False)) == 2
    assert run(SimpleNamespace(check=model, estimator="Nothing", inputs=rows, method=None, json=False)) == 2
    assert "USAGE" in capsys.readouterr().err


def test_the_command_json_is_the_same_document(tmp_path, monkeypatch, capsys):
    model, rows = saved(tmp_path)
    args = SimpleNamespace(check=model, estimator="GpuModel", inputs=rows, method="predict", json=True)
    assert command(monkeypatch, install(tmp_path))(args) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["format"] == cb.FORMAT and doc["verdict"]["verdict"] == "MATCH" and set(doc["sides"]) == {"gpu", "cpu"}
