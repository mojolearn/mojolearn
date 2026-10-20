# SPDX-License-Identifier: Apache-2.0
"""Test-only installed-package qualification for captured graph execution.

Python makes fixtures, independent oracles and reports. Product graph, control
flow, input conversion and profiling execute through native binding calls.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
import sys
import tempfile

FORMAT = "mojolearn-graph-verification/1"
EXPECTED_CASES = (
    "native_inputs", "dynamic_shapes_and_recovery", "lazy_conditional",
    "bounded_while", "shape_growing_while", "exact_duplicate_elimination",
    "training_after_optimization", "node_diagnostics_and_profile",
    "saved_component_roundtrip", "native_session_boundary", "native_checkpoint_boundary",
)
OUTPUT_COUNTS = dict(zip(EXPECTED_CASES, (4, 1, 1, 4, 1, 4, 4, 1, 1, 3, 2)))


_AUDITS = []


def _invoke(operation, *args, **kwargs):
    from ._verify_native_runtime import audit_native_call
    audit = audit_native_call(operation, *args, **kwargs)
    _AUDITS.append(audit.report())
    if audit.exception is not None:
        if audit.frames or audit.forbidden_native or audit.unobserved_threads or audit.wrapper_work:
            audit.assert_native()
        raise audit.exception
    return audit.assert_native()


def _equal(actual, expected):
    import numpy as np
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert a.dtype == b.dtype, (a.dtype, b.dtype)
    assert a.tobytes() == b.tobytes(), "output words differ"


def _refuses(call, text, error_type=Exception):
    try:
        call()
    except Exception as error:
        assert isinstance(error, error_type), (error_type.__name__, type(error).__name__)
        assert text in str(error), (text, str(error))
        if _AUDITS and _AUDITS[-1]["exception"] is not None:
            _AUDITS[-1]["expected_refusal"] = True
        return
    raise AssertionError("invalid operation unexpectedly succeeded: " + text)


def native_inputs():
    import numpy as np
    from . import ops
    x = np.arange(12, dtype=np.float32).reshape(4, 3)
    with ops.Session() as s:
        h = s.tensor(x)
        p = s.capture(h, s.scale(h, 2.0))
        outputs = []
        for a in (x, x.astype(np.float64), x[:, ::-1], x.tolist()):
            out = _invoke(p.run, a)
            _equal(out, np.asarray(a, dtype=np.float32) * np.float32(2))
            outputs.append(out)
        _refuses(lambda: _invoke(p.run, x, x), "input arrays")
        _refuses(lambda: _invoke(p.run, [[1.0], [2.0, 3.0]]), "rectangular")
        _refuses(lambda: _invoke(p.run, x.astype(">f4")), "native byte order")
        integer = s.ids(np.array([[1, 2]], dtype=np.int64))
        id_program = s.capture(integer, s.embedding(s.tensor(np.ones((4, 2), np.float32)), integer))
        _refuses(lambda: _invoke(id_program.run, np.ones((1, 2), np.float32)), "integer", TypeError)
        _equal(_invoke(p.run, x), outputs[0])
        return outputs


def dynamic_shapes_and_recovery():
    import numpy as np
    from . import ops
    w = np.array([[1, 0], [0, 1], [1, -1]], dtype=np.float32)
    with ops.Session() as s:
        x = s.tensor(np.ones((2, 4, 3), dtype=np.float32))
        p = s.capture(x, s.matmul(x, s.tensor(w)))
        a = np.arange(18, dtype=np.float32).reshape(3, 2, 3)
        out = _invoke(p.run, a, dynamic_shapes=True)
        _equal(out, a @ w)
        _refuses(lambda: _invoke(p.run, a), "shape", ValueError)
        _refuses(lambda: _invoke(p.run, np.ones((2, 3), np.float32), dynamic_shapes=True), "rank")
        _refuses(lambda: _invoke(p.run, np.ones((3, 2, 5), np.float32), dynamic_shapes=True), "program node 0")
        _equal(_invoke(p.run, a, dynamic_shapes=True), out)
        return [out]


def lazy_conditional():
    import numpy as np
    from . import ops
    a = np.ones((2, 3), np.float32)
    with ops.Session() as s:
        x = s.tensor(a)
        condition = s.capture(x, s.greater(s.sum(x), s.full((1, 1), 0)))
        body = s.capture(x, s.scale(x, 2))
        weight = s.tensor(np.ones((3, 2), np.float32))
        other = s.capture(x, s.matmul(x, weight))
        s.free(weight)
        out = _invoke(body.run_if, condition, other, a)
        _equal(out, a * np.float32(2))
        # The freed parameter proves the unselected branch was not replayed
        # or eagerly validated. Selecting it must fail, then recovery works.
        _refuses(lambda: _invoke(body.run_if, condition, other, -a), "not a live")
        _equal(_invoke(body.run_if, condition, other, a), out)
        return [out]


def bounded_while():
    import numpy as np
    from . import ops
    a = np.array([[0]], np.float32)
    with ops.Session() as s:
        x = s.tensor(a)
        condition = s.capture(x, s.less(x, s.full((1, 1), 3)))
        body = s.capture(x, s.add(x, s.full((1, 1), 1)))
        done = _invoke(body.while_loop, a, condition=condition, max_iterations=10)
        assert done["iterations"] == 3 and done["stopped"] is True
        _equal(done["outputs"], np.array([[3]], np.float32))
        bound = _invoke(body.while_loop, a, condition=condition, max_iterations=2)
        assert bound["iterations"] == 2 and bound["stopped"] is False
        _equal(bound["outputs"], np.array([[2]], np.float32))
        zero = _invoke(body.while_loop, a, condition=condition, max_iterations=0)
        assert zero["iterations"] == 0 and zero["stopped"] is False
        _equal(zero["outputs"], a)
        stopped = _invoke(body.while_loop, np.array([[5]], np.float32), condition=condition, max_iterations=0)
        assert stopped["iterations"] == 0 and stopped["stopped"] is True
        nonfinite = s.capture(x, x)
        _refuses(lambda: _invoke(body.while_loop, np.array([[np.nan]], np.float32), condition=nonfinite,
                                        max_iterations=3), "finite")
        _refuses(lambda: _invoke(body.while_loop, a, condition=condition, max_iterations=-1), "must be >= 0")
        _refuses(lambda: _invoke(body.while_loop, a, condition=condition, max_iterations=1.5), "integer")
        _equal(_invoke(body.while_loop, a, condition=condition, max_iterations=3)["outputs"], done["outputs"])
        return [done["outputs"], bound["outputs"], zero["outputs"], stopped["outputs"]]


def shape_growing_while():
    import numpy as np
    from . import ops
    a = np.ones((1, 1), np.float32)
    with ops.Session() as s:
        x = s.tensor(a)
        condition = s.capture(x, s.less(s.sum(x), s.full((1, 1), 4)))
        body = s.capture(x, s.concat([x, x], axis=0))
        out = _invoke(body.while_loop, a, condition=condition, max_iterations=5)
        assert out["iterations"] == 2 and out["stopped"] is True
        _equal(out["outputs"], np.ones((4, 1), np.float32))
        invalid = s.capture(x, s.reshape(x, (1, 1, 1)))
        _refuses(lambda: _invoke(invalid.while_loop, a, condition=condition, max_iterations=1), "preserve")
        _equal(_invoke(body.while_loop, a, condition=condition, max_iterations=5)["outputs"], out["outputs"])
        return [out["outputs"]]


def exact_duplicate_elimination():
    import numpy as np
    from . import ops
    a = np.array([[-0.0, 1, -2], [3, -4, 0.0]], np.float32)
    with ops.Session() as s:
        x = s.tensor(a)
        u, v = s.relu(x), s.relu(x)
        p = s.capture(x, s.add(u, v))
        before = _invoke(p.inspect, )
        candidate = _invoke(p.inspect, optimize=True)
        assert len(before["nodes"]) == 3 and candidate["nodes_removed"] == 1
        eager, reduced = _invoke(p.run, a), _invoke(p.run, a, optimize=True)
        _equal(eager, reduced)
        assert _invoke(p.inspect, ) == before, "optimization must not mutate the captured program"
        plus, minus = s.scale(x, 0.0), s.scale(x, -0.0)
        zeros = s.capture(x, [plus, minus])
        assert _invoke(zeros.inspect, optimize=True)["nodes_removed"] == 0
        zo = _invoke(zeros.run, a, optimize=True)
        _equal(zo[0], a * np.float32(0))
        _equal(zo[1], a * np.float32(-0.0))
        d = s.capture(x, [s.dropout(x, 0.25, seed=3), s.dropout(x, 0.25, seed=3)])
        assert _invoke(d.inspect, optimize=True)["nodes_removed"] == 0
        return [eager, reduced, *zo]


def training_after_optimization():
    import numpy as np
    from . import ops
    a = np.arange(12, dtype=np.float32).reshape(4, 3) / np.float32(8)
    target = np.ones((4, 1), np.float32)
    states = []
    for optimize_first in (False, True):
        with ops.Session() as s:
            x = s.tensor(a)
            w = s.tensor(np.zeros((3, 1), np.float32))
            p = s.capture(x, s.add(s.matmul(x, w), s.matmul(x, w)))
            _invoke(p.run, a, optimize=optimize_first)
            opt = s.optimizer("adam", lr=0.01)
            _invoke(p.fit, a, targets=target, optimizer=opt, epochs=3, batch_size=2, shuffle=False)
            state = s.optimizer_state(opt, w)
            assert state["step"] == 6
            states.append([s.numpy(w), np.asarray(p.loss_curve_, np.float32),
                           s.numpy(state["exp_avg"]), s.numpy(state["exp_avg_sq"])])
    for a, b in zip(*states):
        _equal(a, b)
    return states[0]


def node_diagnostics_and_profile():
    import numpy as np
    from . import ops
    a = np.ones((2, 3), np.float32)
    with ops.Session() as s:
        x = s.tensor(a)
        w = s.tensor(np.ones((3, 2), np.float32))
        p = s.capture(x, s.relu(s.matmul(x, w)))
        report = _invoke(p.profile, a)
        out, profile = report["outputs"], report["profile"]
        _equal(out, a @ np.ones((3, 2), np.float32))
        assert profile["executed_nodes"] == len(profile["nodes"]) == 2
        assert profile["input_logical_bytes"] == a.nbytes
        assert profile["output_bytes"] == out.nbytes
        assert profile["parameter_bytes"] == 24
        assert profile["node_sync_count"] == 4
        assert profile["peak_tape_reserved_bytes"] >= out.nbytes
        for row in profile["nodes"]:
            assert row["enqueue_ns"] >= 0 and row["wait_ns"] >= 0
        _refuses(lambda: _invoke(p.run, np.ones((2, 5), np.float32), dynamic_shapes=True),
                 "program node 0 (matmul) inputs=[(2, 5), (3, 2)]")
        _equal(_invoke(p.run, a), out)
        return [out]


def saved_component_roundtrip():
    import numpy as np
    from . import ops
    a = np.array([[0]], np.float32)
    with tempfile.TemporaryDirectory(prefix="mojolearn-graph-") as d:
        with ops.Session() as s:
            x = s.tensor(a)
            condition = s.capture(x, s.less(x, s.full((1, 1), 3)))
            body = s.capture(x, s.add(x, s.full((1, 1), 1)))
            _invoke(condition.save, Path(d) / "condition.mjl")
            _invoke(body.save, Path(d) / "body.mjl")
            for name in ("condition", "body"):
                path = Path(d) / (name + ".mjl")
                report = json.loads(Path(str(path) + ".verify.json").read_text())
                assert report["format"] == "mojolearn-model-verify/1"
                assert report["model"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
                assert report["model"]["bytes"] == path.stat().st_size
                assert report["inputs"][0]["shape"] == [1, 1]
                assert report["outputs"][0]["shape"] == [1, 1]
                from ._verify_model import verify, compare
                assert compare(verify(str(path), expect=report), report)[0] == "MATCH"
        with ops.Session() as s:
            condition = _invoke(s.load_program, Path(d) / "condition.mjl")
            body = _invoke(s.load_program, Path(d) / "body.mjl")
            out = _invoke(body.while_loop, a, condition=condition, max_iterations=4)
            assert out["iterations"] == 3 and out["stopped"] is True
            _equal(out["outputs"], np.array([[3]], np.float32))
            return [out["outputs"]]


def native_session_boundary():
    import numpy as np
    from . import ops
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    with ops.Session() as s:
        x = _invoke(s.tensor, a.tolist())
        y = _invoke(s.scale, x, 2.0)
        pieces = _invoke(s.split, y, [1, 2], axis=1)
        joined = _invoke(s.concat, pieces, axis=1)
        out = _invoke(s.numpy, joined)
        _equal(out, a * np.float32(2))
        assert _invoke(s.shape, joined) == (2, 3)
        p = _invoke(s.capture, x, joined)
        _equal(_invoke(p.run, a), out)
        _refuses(lambda: _invoke(s.reshape, x, (True, 3)), "integer", TypeError)
        _refuses(lambda: _invoke(s.ids, a), "integer", TypeError)
        _refuses(lambda: _invoke(s.gelu, x, approximate="bad"), "approximate", ValueError)
    with ops.Session(record=True) as s:
        x = _invoke(s.tensor, [[2.0]], requires_grad=True)
        loss = _invoke(s.mul, x, x)
        _invoke(s.backward, loss, create_graph=True)
        first = _invoke(s.grad, x)
        _equal(first, np.array([[4]], np.float32))
        gradient = _invoke(s.grad_tensor, x)
        _invoke(s.backward, gradient)
        second = _invoke(s.grad, x)
        _equal(second, np.array([[2]], np.float32))
        opt = _invoke(s.optimizer, "sgd", lr=0.25)
        _invoke(s.step, opt, x)
        _equal(_invoke(s.numpy, x), np.array([[1.5]], np.float32))
        state = _invoke(s.optimizer_state, opt, x)
        assert state["step"] == 1
        _invoke(s.load_optimizer_state, opt, x, state)
        _invoke(s.zero_grad)
        _invoke(s.reset)
        _invoke(s.free, x)
    return [out, first, second]


def native_checkpoint_boundary():
    import numpy as np
    from . import ops
    with tempfile.TemporaryDirectory(prefix="mojolearn-state-") as d:
        path = Path(d) / "state.ckpt"
        with ops.Session(record=True) as s:
            x = s.tensor([[2.0]], requires_grad=True)
            opt = s.optimizer("sgd", lr=0.25, momentum=0.5)
            s.backward(s.mul(x, x))
            s.step(opt, x)
            saved = s.numpy(x)
            _equal(saved, np.array([[1]], np.float32))
            _invoke(s.save_state, path, [x], optimizer=opt, seed=2**63 + 1, epoch=3, step=1)
            s.reset()
            s.backward(s.mul(x, x))
            s.step(opt, x)
            expected = s.numpy(x)
        with ops.Session(record=True) as s:
            opt = s.optimizer("sgd", lr=0.25, momentum=0.5)
            state = _invoke(s.load_state, path, optimizer=opt)
            assert state["seed"] == 2**63 + 1 and state["epoch"] == 3 and state["step"] == 1
            x, = state["params"]
            _equal(s.numpy(x), saved)
            s.backward(s.mul(x, x))
            s.step(opt, x)
            restored = s.numpy(x)
            _equal(restored, expected)
            _equal(restored, np.array([[0]], np.float32))
    return [saved, restored]


def _digest(a):
    import numpy as np
    x = np.ascontiguousarray(a)
    return {"shape": list(x.shape), "dtype": x.dtype.str,
            "sha256": hashlib.sha256(x.tobytes()).hexdigest()}


def compare_reports(actual, expected):
    errors = []
    for label, report in (("actual", actual), ("expected", expected)):
        if report.get("format") != FORMAT or report.get("expected_cases") != list(EXPECTED_CASES):
            errors.append(label + ": schema or coverage differs")
        if report.get("backend", {}).get("numeric_mode") != 1:
            errors.append(label + ": native identical mode is unproven")
        rows = report.get("cases", {})
        if set(rows) != set(EXPECTED_CASES):
            errors.append(label + ": missing or unexpected cases")
        for name in EXPECTED_CASES:
            row = rows.get(name, {})
            if row.get("status") != "passed" or len(row.get("outputs", [])) != OUTPUT_COUNTS[name]:
                errors.append(label + ": incomplete " + name)
            for output in row.get("outputs", []):
                if (not re.fullmatch(r"[0-9a-f]{64}", str(output.get("sha256", "")))
                        or output.get("dtype") != "<f4"
                        or not isinstance(output.get("shape"), list)
                        or not output["shape"]
                        or not all(isinstance(d, int) and d > 0 for d in output["shape"])):
                    errors.append(label + ": malformed output evidence " + name)
            audits = row.get("native_calls", [])
            if not audits or any(a.get("python_frames") or a.get("forbidden_native_calls")
                                 or a.get("wrapper_work") or a.get("unobserved_existing_threads")
                                 or not (a.get("passed") or a.get("expected_refusal")) for a in audits):
                errors.append(label + ": native execution unproven " + name)
    if errors:
        return errors
    for name in EXPECTED_CASES:
        if actual["cases"][name]["outputs"] != expected["cases"][name]["outputs"]:
            errors.append("output words differ: " + name)
    return errors


def comparator_self_test(report):
    assert not compare_reports(report, report)
    corrupt = copy.deepcopy(report)
    first = EXPECTED_CASES[0]
    corrupt["cases"][first]["outputs"][0]["sha256"] = "0" * 64
    assert compare_reports(report, corrupt), "one changed digest must fail"
    missing = copy.deepcopy(report)
    del missing["cases"][first]
    assert compare_reports(report, missing), "missing evidence must fail"
    failed = copy.deepcopy(report)
    failed["cases"][first]["status"] = "failed"
    assert compare_reports(report, failed), "failed execution must fail comparison"
    fast = copy.deepcopy(report)
    fast["backend"]["numeric_mode"] = 0
    assert compare_reports(report, fast), "wrong numeric mode must fail comparison"
    return {"same_report": "passed", "changed_digest": "rejected", "missing_case": "rejected",
            "failed_case": "rejected", "wrong_mode": "rejected"}


def run_checks(vendor, require_installed=False):
    from . import _backend
    import mojolearn
    report = {"format": FORMAT, "expected_cases": list(EXPECTED_CASES), "cases": {}, "backend": {}}
    try:
        b = _backend.binding("_mojolearn_x_sequence", "identical")
        report["backend"] = {"vendor": str(b.x_sequence_vendor()), "numeric_mode": int(b.x_sequence_numeric_mode()),
                             "binding_file": str(getattr(b, "__file__", "")), "binding_type": type(b).__name__,
                             "package_file": str(Path(mojolearn.__file__).resolve()), "python": sys.executable}
        assert report["backend"]["vendor"] == vendor, report["backend"]
        assert report["backend"]["numeric_mode"] == 1, report["backend"]
        for entry in ("ops_program_execute", "ops_program_control", "ops_program_inspect"):
            assert hasattr(b, entry), "missing native entry " + entry
        if require_installed:
            package_path = Path(mojolearn.__file__).resolve()
            assert "site-packages" in package_path.parts or "dist-packages" in package_path.parts, package_path
            assert package_path.is_relative_to(Path(sys.prefix).resolve()), (package_path, sys.prefix)
    except Exception as error:
        for name in EXPECTED_CASES:
            report["cases"][name] = {"status": "cannot_run", "error": str(error), "outputs": []}
        return report
    for name in EXPECTED_CASES:
        _AUDITS.clear()
        try:
            values = globals()[name]()
            report["cases"][name] = {"status": "passed", "outputs": [_digest(a) for a in values],
                                      "native_calls": list(_AUDITS)}
        except Exception as error:
            report["cases"][name] = {"status": "failed", "error": str(error), "outputs": [],
                                      "native_calls": list(_AUDITS)}
    if all(row["status"] == "passed" for row in report["cases"].values()):
        report["comparator_controls"] = comparator_self_test(report)
    return report


def cmd_verify_graph(args):
    report = run_checks(args.vendor, args.require_installed)
    rc = 0 if all(row["status"] == "passed" for row in report["cases"].values()) else 4
    if args.expect:
        try:
            expected = json.loads(Path(args.expect).read_text())
            errors = compare_reports(report, expected)
        except Exception as error:
            errors = ["reference could not be read: " + str(error)]
        report["comparison"] = {"expected": args.expect, "errors": errors, "matched": not errors}
        if errors and rc == 0:
            rc = 1
    report["exit_code"] = rc
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        counts = {status: sum(row["status"] == status for row in report["cases"].values())
                  for status in ("passed", "failed", "cannot_run")}
        print("GRAPH", report["backend"].get("vendor", "unknown"), counts, "exit", rc)
    return rc


def add_parser(sub):
    p = sub.add_parser("verify-graph", help="qualify this install's native captured graph execution")
    p.add_argument("--vendor", required=True, choices=("cpu", "metal", "cuda", "hip"))
    p.add_argument("--require-installed", action="store_true")
    p.add_argument("--expect", help="compare all output bits with another vendor's report")
    p.add_argument("--out", help="retain the complete report at this path")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_verify_graph)
