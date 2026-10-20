# SPDX-License-Identifier: Apache-2.0
"""Installed-package verification of native captured-program distribution.

This is a testing controller, never a product execution path or timer. It
retains failures, independent analytic accuracy, complete-state digests,
physical-device scope and whole-operation Python traces in a JSON report.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import tempfile
import traceback

FORMAT = "mojolearn-distributed-program-verify/1"


def _digest(state):
    import numpy as np
    h = hashlib.sha256()
    for name in ("parameters", "first_moments", "second_moments"):
        for a in state[name]:
            h.update(np.asarray(a, dtype=np.float32).tobytes(order="C"))
    h.update(json.dumps([state[k] for k in ("epoch", "step", "seed", "logical_shards")]).encode())
    return h.hexdigest()


def verify(devices=(0,), expected=None):
    import numpy as np
    from . import ops, _backend
    from ._verify_model import _install
    from ._verify_native_runtime import audit_native_call

    report = {"format": FORMAT, "install": _install("identical"),
              "devices_requested": list(devices), "cases": [], "runtime_audits": [],
              "physical_multi_gpu": len(devices) > 1, "status": "FAIL"}

    def native(name, operation, *args, **kwargs):
        audit = audit_native_call(operation, *args, **kwargs)
        entry = audit.report()
        entry["operation"] = name
        report["runtime_audits"].append(entry)
        return audit.assert_native()

    try:
        b = _backend.binding("_mojolearn_x_sequence", "identical")
        names = ("open", "load", "fit", "run", "state", "residency", "save", "close")
        missing = ["ops_distributed_" + n for n in names if not hasattr(b, "ops_distributed_" + n)]
        if missing:
            raise RuntimeError("installed binding lacks entries: " + ", ".join(missing))
        with tempfile.TemporaryDirectory(prefix="mojolearn-distributed-verify-") as directory:
            with ops.Session(numeric_mode="identical") as s:
                x = s.tensor(np.ones((3, 1), np.float32))
                w = s.tensor(np.zeros((1, 1), np.float32))
                p = s.capture([x], [s.matmul(x, w)])
                opt = s.optimizer("sgd", lr=0.125)
                with native("sgd.open", p.distribute, devices, logical_shards=max(3, len(devices)), optimizer=opt) as d:
                    X, y = np.ones((5, 1), np.float32), np.full((5, 1), 2, np.float32)
                    losses = native("sgd.fit", d.fit, X, targets=y, epochs=1, batch_size=5, shuffle=False)
                    prediction = native("sgd.run", d.run, X)
                    state = native("sgd.state", d.state)
                    np.testing.assert_array_equal(prediction, np.full((5, 1), 0.5, np.float32))
                    np.testing.assert_array_equal(state["parameters"][0], [[0.5]])
                    np.testing.assert_array_equal(losses, [4])
                    report["cases"].append({"name": "analytic_sgd", "status": "PASS", "digest": _digest(state),
                        "oracle": "Independent analytic gradient: d mean((w*1-2)^2)/dw=-4 at w=0; lr=1/8 => w=1/2."})

            rng = np.random.default_rng(104)
            X = rng.normal(size=(19, 3)).astype(np.float32)
            y = rng.normal(size=(19, 2)).astype(np.float32)
            with ops.Session(numeric_mode="identical") as s:
                x = s.tensor(np.zeros((3, 3), np.float32))
                w1 = s.tensor(rng.normal(0, 0.1, size=(3, 5)).astype(np.float32))
                bias = s.tensor(np.zeros((1, 5), np.float32))
                w2 = s.tensor(rng.normal(0, 0.1, size=(5, 2)).astype(np.float32))
                h = s.dropout(s.tanh(s.linear(x, w1, bias)), p=0.25, seed=17)
                p = s.capture([x], [s.matmul(h, w2)])
                opt = s.optimizer("adamw", lr=0.01, weight_decay=0.02,
                                  schedule="step", step_size=2, gamma=0.5)
                logical = max(4, len(devices))
                path = str(pathlib.Path(directory) / "resume.ckpt")
                with native("adam.open", p.distribute, devices, logical_shards=logical, optimizer=opt, seed=91) as d:
                    native("adam.fit.first", d.fit, X, targets=y, epochs=2, batch_size=7)
                    native("adam.save", d.save, path)
                    before = native("adam.state.first", d.state)
                    residency = native("adam.residency", d.residency)
                    assert tuple(residency["devices"]) == tuple(devices)
                    assert sum(residency["optimizer_elements_per_rank"]) == 30
                    report["residency"] = residency
                    native("adam.fit.continue", d.fit, X, targets=y, epochs=3, batch_size=7)
                    final = native("adam.state.final", d.state)
                with native("adam.load", s.load_distributed, path, devices=(0,)) as restored:
                    assert _digest(native("adam.state.restored", restored.state)) == _digest(before)
                    native("adam.fit.resume", restored.fit, X, targets=y, epochs=3, batch_size=7)
                    assert _digest(native("adam.state.resumed", restored.state)) == _digest(final)
                report["cases"].append({"name": "checkpoint_resume", "status": "PASS", "digest": _digest(final),
                    "scope": "15 optimizer steps, uneven batches, dropout, row shuffle, AdamW moments and step schedule."})
                with native("topology.open.single", p.distribute, (0,), logical_shards=logical, optimizer=opt, seed=91) as single:
                    native("topology.fit.single", single.fit, X, targets=y, epochs=5, batch_size=7)
                    assert _digest(native("topology.state.single", single.state)) == _digest(final)
                report["cases"].append({"name": "physical_topology", "status": "PASS" if len(devices) > 1 else "PENDING",
                    "digest": _digest(final), "scope": "Two or more real GPU IDs required" if len(devices) == 1 else "one versus requested physical GPUs"})
        if expected is not None:
            if expected.get("format") != FORMAT:
                raise ValueError("reference report has the wrong format")
            wanted = {c["name"]: c.get("digest") for c in expected["cases"] if c["status"] == "PASS"}
            comparisons = []
            for case in report["cases"]:
                if case["name"] in wanted:
                    equal = case.get("digest") == wanted[case["name"]]
                    comparisons.append({"name": case["name"], "status": "MATCH" if equal else "DIFFER"})
                    if not equal:
                        raise AssertionError("cross-install identity mismatch: " + case["name"])
            if not comparisons:
                raise ValueError("reference contains no comparable passing cases")
            report["cross_install"] = comparisons
        report["status"] = "PASS" if len(devices) > 1 else "PASS_SINGLE_DEVICE_PENDING_PHYSICAL"
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    return report


def command(args):
    try:
        devices = tuple(int(s) for s in args.devices.split(","))
        expected = None if args.compare is None else json.loads(pathlib.Path(args.compare).read_text())
        report = verify(devices, expected)
    except Exception as exc:
        report = {"format": FORMAT, "status": "FAIL", "failure": {"type": type(exc).__name__, "message": str(exc)}}
    if args.output:
        pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "cases": [{"name": c["name"], "status": c["status"]}
                     for c in report.get("cases", [])], "evidence": args.output,
                     "failure": report.get("failure", {}).get("message")}))
    return 1 if report["status"] == "FAIL" else 0


def add_parser(sub):
    parser = sub.add_parser("verify-distributed-program", help="verify native captured-program training and checkpoints")
    parser.add_argument("--devices", default="0", help="comma-separated physical GPU IDs; 0 alone is single-device coverage")
    parser.add_argument("--compare", help="reference report from another installed vendor/CPU build")
    parser.add_argument("--output", required=True, help="retain the complete JSON report, including runtime traces and failures")
    parser.set_defaults(func=command)
