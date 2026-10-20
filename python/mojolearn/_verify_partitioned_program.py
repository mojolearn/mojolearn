# SPDX-License-Identifier: Apache-2.0
"""Installed test controller for physical model/tensor/pipeline execution.

No product path imports this verifier. Oracles and data preparation are test
code, outside every audited native public operation.
"""
from __future__ import annotations
import hashlib
import json
import pathlib
import tempfile
import traceback

FORMAT = "mojolearn-partitioned-program-verify/1"


def _digest(state):
    h = hashlib.sha256()
    for key in ("parameters", "first_moments", "second_moments"):
        for a in state[key]:
            h.update(a.tobytes(order="C"))
    h.update(json.dumps([state[k] for k in ("epoch", "step", "seed", "logical_stages",
                                           "tensor_shards", "microbatches")]).encode())
    return h.hexdigest()


def verify(devices=(0,), expected=None):
    import numpy as np
    from . import ops, _backend
    from ._verify_model import _install
    from ._verify_native_runtime import audit_native_call
    report = {"format": FORMAT, "install": _install("identical"), "status": "FAIL",
              "devices_requested": list(devices), "cases": [], "runtime_audits": [],
              "physical_multi_gpu": len(devices) > 1}

    def native(name, fn, *args, **kwargs):
        audited = audit_native_call(fn, *args, **kwargs)
        item = audited.report()
        item["operation"] = name
        report["runtime_audits"].append(item)
        return audited.assert_native()

    try:
        b = _backend.binding("_mojolearn_x_sequence", "identical")
        names = ("open", "load", "fit", "run", "state", "residency", "save", "close", "profile")
        missing = ["ops_partitioned_"+name for name in names if not hasattr(b, "ops_partitioned_"+name)]
        if missing:
            raise RuntimeError("installed binding lacks entries: " + ", ".join(missing))
        with ops.Session(numeric_mode="identical") as s:
            a = s.tensor(np.zeros((3, 1), np.float32))
            w = s.tensor(np.zeros((1, 4), np.float32))
            p = s.capture([a], [s.matmul(a, w)])
            opt = s.optimizer("sgd", lr=.125)
            with native("analytic.open", p.partition, devices, stages=max(2, len(devices)),
                        tensor_shards=2, microbatches=3, optimizer=opt) as d:
                x = np.ones((5, 1), np.float32)
                y = np.full((5, 4), 2, np.float32)
                losses = native("analytic.fit", d.fit, x, targets=y, epochs=1, batch_size=5, shuffle=False)
                state = native("analytic.state", d.state)
                prediction = native("analytic.run", d.run, x)
                for a in state["parameters"]:
                    np.testing.assert_allclose(a, .125, rtol=0, atol=2e-8)
                np.testing.assert_allclose(prediction, .125, rtol=0, atol=2e-8)
                np.testing.assert_array_equal(losses, [4])
                r = native("analytic.residency", d.residency)
                assert r["partitioned_products"] == 1
                assert sum(r["parameter_elements_per_rank"]) == 4
                if len(devices) > 1:
                    assert max(r["parameter_elements_per_rank"]) < 4
                    assert len(set(r["parameter_owner_devices"])) == 2
                report["tensor_residency"] = r
                report["cases"].append({"name": "tensor_analytic_sgd", "status": "PASS", "digest": _digest(state),
                    "oracle": "Four-column MSE gradient is -1 per weight at X=1,w=0,y=2; SGD lr1/8 gives 1/8."})
        rng = np.random.default_rng(34)
        x = rng.normal(size=(11, 2)).astype(np.float32)
        y = rng.normal(size=(11, 2)).astype(np.float32)
        with tempfile.TemporaryDirectory(prefix="mojolearn-partition-verify-") as temp:
            with ops.Session(numeric_mode="identical") as s:
                a = s.tensor(np.zeros((3, 2), np.float32))
                w0 = np.arange(8, dtype=np.float32).reshape(2, 4)/32
                b0 = np.full((1, 4), .125, np.float32)
                v0 = np.arange(8, dtype=np.float32).reshape(4, 2)/64
                w, bias, v = s.tensor(w0), s.tensor(b0), s.tensor(v0)
                hidden = s.tanh(s.linear(a, w, bias))
                plain = s.capture([a], [s.matmul(hidden, v)])
                opt = s.optimizer("adamw", lr=.01, weight_decay=.02,
                                  schedule="step", step_size=2, gamma=.5)
                stages = max(3, len(devices))
                with native("pipeline.open", plain.partition, devices, stages=stages, tensor_shards=1,
                            microbatches=3, optimizer=opt) as p:
                    prediction = native("pipeline.run", p.run, x)
                    np.testing.assert_allclose(prediction, np.tanh(x@w0+b0)@v0, rtol=2e-6, atol=2e-7)
                    report["cases"].append({"name": "pipeline_numpy_forward", "status": "PASS",
                        "digest": hashlib.sha256(prediction.tobytes()).hexdigest(), "oracle": "Independent NumPy dense/tanh/dense."})
                dropped = s.dropout(hidden, p=.25, seed=17)
                graph = s.capture([a], [s.matmul(dropped, v)])
                path = str(pathlib.Path(temp)/"partition.ckpt")
                with native("resume.open", graph.partition, devices, stages=stages, tensor_shards=2,
                            microbatches=3, optimizer=opt, seed=41) as p:
                    native("resume.fit.first", p.fit, x, targets=y, epochs=2, batch_size=4)
                    before = native("resume.state.first", p.state)
                    native("resume.save", p.save, path)
                    native("resume.fit.continue", p.fit, x, targets=y, epochs=3, batch_size=4)
                    final = native("resume.state.final", p.state)
                    report["partition_residency"] = native("resume.residency", p.residency)
                    report["operation_intervals"] = native("resume.profile", p.profile)
                with native("resume.load", s.load_partitioned, path, devices=(0,)) as restored:
                    assert _digest(native("resume.state.loaded", restored.state)) == _digest(before)
                    assert restored.residency()["partitioned_products"] == 2
                    native("resume.fit.restored", restored.fit, x, targets=y, epochs=3, batch_size=4)
                    assert _digest(native("resume.state.restored", restored.state)) == _digest(final)
                with native("topology.open", graph.partition, (0,), stages=stages, tensor_shards=2,
                            microbatches=3, optimizer=opt, seed=41) as single:
                    native("topology.fit", single.fit, x, targets=y, epochs=5, batch_size=4)
                    assert _digest(native("topology.state", single.state)) == _digest(final)
                report["cases"].append({"name": "partition_checkpoint_resume", "status": "PASS", "digest": _digest(final),
                    "scope": "15 AdamW updates, uneven batches, dropout, shuffle, schedule, tensor slices and changed physical topology."})
                events = report["operation_intervals"]
                overlap = [(a["node"], a["microbatch"], a["device"], b["node"], b["microbatch"], b["device"])
                    for a in events for b in events if a["device"] != b["device"]
                    and a["microbatch"] != b["microbatch"]
                    and max(a["start_ns"],b["start_ns"]) < min(a["end_ns"],b["end_ns"])]
                report["physical_pipeline"] = {"overlapping_operation_pairs": len(overlap),
                    "devices_observed": sorted(set(e["device"] for e in events)),
                    "scope": "Native operation intervals include synchronized GPU work; this is not a throughput measurement."}
                if len(devices) > 1:
                    assert overlap, "No observed pipeline overlap across physical GPUs"
                    assert set(e["device"] for e in events) == set(devices)
                report["cases"].append({"name": "physical_topology_and_pipeline", "status": "PASS" if len(devices)>1 else "PENDING",
                                        "digest": _digest(final)})
        if expected is not None:
            if expected.get("format") != FORMAT:
                raise ValueError("reference report has the wrong format")
            wanted = {c["name"]: c["digest"] for c in expected["cases"] if c["status"] == "PASS"}
            comparisons = []
            for c in report["cases"]:
                if c["name"] in wanted:
                    equal = c["digest"] == wanted[c["name"]]
                    comparisons.append({"name": c["name"], "status": "MATCH" if equal else "DIFFER"})
                    assert equal, "cross-install identity mismatch: " + c["name"]
            if not comparisons:
                raise ValueError("reference has no comparable passing cases")
            report["cross_install"] = comparisons
        report["status"] = "PASS" if len(devices)>1 else "PASS_SINGLE_DEVICE_PENDING_PHYSICAL"
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    return report


def command(args):
    try:
        devices = tuple(map(int, args.devices.split(",")))
        expected = None if args.compare is None else json.loads(pathlib.Path(args.compare).read_text())
        report = verify(devices, expected)
    except Exception as exc:
        report = {"format": FORMAT, "status": "FAIL", "failure": {"type": type(exc).__name__, "message": str(exc)}}
    pathlib.Path(args.output).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"status": report["status"], "cases": [{"name":c["name"],"status":c["status"]}
                     for c in report.get("cases",[])], "evidence": args.output,
                     "failure":report.get("failure",{}).get("message")}))
    return 1 if report["status"] == "FAIL" else 0


def add_parser(sub):
    parser = sub.add_parser("verify-partitioned-program", help="verify native model/tensor/pipeline execution")
    parser.add_argument("--devices", default="0", help="comma-separated physical GPU IDs")
    parser.add_argument("--compare", help="retained report from another installed vendor/CPU build")
    parser.add_argument("--output", required=True, help="retain complete JSON including native traces and failures")
    parser.set_defaults(func=command)
