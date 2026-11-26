# SPDX-License-Identifier: Apache-2.0
"""Installed-package verification of native captured-program distribution, on ONE device.

The logical shards (three and four of them) all run on device 0: the single-GPU
form of multi-GPU training. Verify is one GPU only (Andrew, 2026-11-26): no
physical multi-GPU case, option or PENDING state ships here. This is a testing
controller, never a product execution path or timer. It retains failures,
independent analytic accuracy, complete-state digests and whole-operation
Python traces in a JSON report.
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


#: the one device every case runs on
DEVICES = (0,)


def verify(expected=None):
    import numpy as np
    from functools import partial
    from . import ops
    from ._verify_model import _install
    from ._verify_native_runtime import audited, audited_context

    devices = DEVICES
    report = {"format": FORMAT, "install": _install("identical"), "device": devices[0],
              "cases": [], "runtime_audits": [], "status": "FAIL"}
    native = partial(audited, report["runtime_audits"])
    opened = partial(audited_context, report["runtime_audits"])

    # No entry probe: the public `Program.distribute` call below refuses by
    # name when the installed binding lacks an entry, and that refusal fails
    # this verifier the same way.
    try:
        with tempfile.TemporaryDirectory(prefix="mojolearn-distributed-verify-") as directory:
            with ops.Session(numeric_mode="identical") as s:
                x = s.tensor(np.ones((3, 1), np.float32))
                w = s.tensor(np.zeros((1, 1), np.float32))
                p = s.capture([x], [s.matmul(x, w)])
                opt = s.optimizer("sgd", lr=0.125)
                with opened("sgd.open", p.distribute, devices, logical_shards=3, optimizer=opt) as d:
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
                logical = 4
                path = pathlib.Path(directory) / "resume.ckpt"
                with opened("adam.open", p.distribute, devices, logical_shards=logical, optimizer=opt, seed=91) as d:
                    native("adam.fit.first", d.fit, X, targets=y, epochs=2, batch_size=7)
                    native("adam.save", d.save, path)
                    before = native("adam.state.first", d.state)
                    residency = native("adam.residency", d.residency)
                    assert tuple(residency["devices"]) == tuple(devices)
                    assert sum(residency["optimizer_elements_per_rank"]) == 30
                    report["residency"] = residency
                    native("adam.fit.continue", d.fit, X, targets=y, epochs=3, batch_size=7)
                    final = native("adam.state.final", d.state)
                with opened("adam.load", s.load_distributed, path, devices=(0,)) as restored:
                    assert _digest(native("adam.state.restored", restored.state)) == _digest(before)
                    native("adam.fit.resume", restored.fit, X, targets=y, epochs=3, batch_size=7)
                    assert _digest(native("adam.state.resumed", restored.state)) == _digest(final)
                report["cases"].append({"name": "checkpoint_resume", "status": "PASS", "digest": _digest(final),
                    "scope": "15 optimizer steps, uneven batches, dropout, row shuffle, AdamW moments and step schedule."})
                with opened("replay.open", p.distribute, devices, logical_shards=logical, optimizer=opt, seed=91) as again:
                    native("replay.fit", again.fit, X, targets=y, epochs=5, batch_size=7)
                    assert _digest(native("replay.state", again.state)) == _digest(final)
                report["cases"].append({"name": "uninterrupted_replay", "status": "PASS", "digest": _digest(final),
                    "scope": "Five epochs in one fit equal two, a checkpoint, then three; four logical shards on device 0."})
        if expected is not None:
            report["cross_install"] = comparisons = compare(report, expected)
            for row in comparisons:
                if row["status"] != "MATCH":
                    raise AssertionError("cross-install identity mismatch: " + row["name"])
        report["status"] = "PASS"
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    return report


def compare(report, expected):
    """Each passing case's digest against the same case of `expected` (a report
    from another install): `[{name, status: MATCH|DIFFER}]`. ValueError when
    `expected` is not a comparable reference (wrong format, no passing case in
    common): nothing was compared. One body for `verify-distributed-program
    --compare` and `verify --all` (_verify_extended.py)."""
    if expected.get("format") != FORMAT:
        raise ValueError("reference report has the wrong format")
    wanted = {c["name"]: c.get("digest") for c in expected.get("cases", []) if c.get("status") == "PASS"}
    comparisons = [{"name": case["name"], "status": "MATCH" if case.get("digest") == wanted[case["name"]] else "DIFFER"}
                   for case in report.get("cases", []) if case.get("status") == "PASS" and case["name"] in wanted]
    if not comparisons:
        raise ValueError("reference contains no comparable passing cases")
    return comparisons


def command(args):
    try:
        expected = None if args.compare is None else json.loads(pathlib.Path(args.compare).read_text())
        report = verify(expected)
    except Exception as exc:
        report = {"format": FORMAT, "status": "FAIL", "failure": {"type": type(exc).__name__, "message": str(exc)}}
    if args.output:
        pathlib.Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "cases": [{"name": c["name"], "status": c["status"]}
                     for c in report.get("cases", [])], "evidence": args.output,
                     "failure": report.get("failure", {}).get("message")}))
    return 1 if report["status"] == "FAIL" else 0


def add_parser(sub):
    parser = sub.add_parser("verify-distributed-program",
                            help="verify native captured-program training and checkpoints (logical shards on one device)")
    parser.add_argument("--compare", help="reference report from another installed vendor/CPU build")
    parser.add_argument("--output", required=True, help="retain the complete JSON report, including runtime traces and failures")
    parser.set_defaults(func=command)
