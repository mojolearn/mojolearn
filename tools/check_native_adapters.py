#!/usr/bin/env python3
"""External qualification controller, never imported by product execution.

Build/load the optional native PyTorch registration and check its real public
dispatcher. Keep logs and result JSON under the caller's retained run directory.
No performance or cross-vendor qualification is inferred from this one machine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import traceback


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"complete": False, "scope": "adapter functionality on one machine", "checks": [], "failures": []}
    try:
        import torch
        from torch.utils.cpp_extension import load
        root = Path(__file__).resolve().parents[1]
        report["torch"] = torch.__version__
        report["hip"] = torch.version.hip
        report["device"] = args.device
        report["native_sha256"] = hashlib.sha256((out / "libmojolearn_adapter.so").read_bytes()).hexdigest()
        load(name="mojolearn_torch", sources=[str(root / "adapters/torch/registration.cpp")],
             build_directory=str(out), is_python_module=False, verbose=True,
             extra_cflags=["-O2"], extra_ldflags=[f"-L{out}", "-lmojolearn_adapter", f"-Wl,-rpath,{out}"])
        operation = torch.ops.mojolearn.operation
        device = torch.device(args.device)
        if device.type == "cuda":
            assert torch.cuda.is_available(), "GPU qualification cannot fall back to CPU"
            report["hardware"] = torch.cuda.get_device_name(device)

        def checked(name, fn):
            try:
                fn()
                report["checks"].append({"name": name, "passed": True})
            except Exception:
                report["failures"].append({"name": name, "traceback": traceback.format_exc()})

        def equal(actual, expected):
            assert actual.device == expected.device
            assert actual.dtype == expected.dtype
            assert tuple(actual.shape) == tuple(expected.shape)
            assert torch.equal(actual.view(torch.int32), expected.view(torch.int32)), "output words differ"

        x = torch.tensor([[1., -2., 0.5], [-0.25, 2., -1.]], device=device)
        y = torch.tensor([[0.5, 1., 2.], [1., 0.5, -0.25]], device=device)
        checked("add", lambda: equal(operation([x, y], 2, [2, 3]), x + y))
        checked("subtract", lambda: equal(operation([x, y], 3, [2, 3]), x - y))
        checked("multiply", lambda: equal(operation([x, y], 4, [2, 3]), x * y))
        checked("relu", lambda: equal(operation([x], 6, [2, 3]), torch.relu(x)))
        checked("copy", lambda: equal(operation([x], 31, [2, 3]), x))
        w = torch.tensor([[0.5, 1.], [-1., 2.], [0.25, -0.5]], device=device)
        checked("matmul", lambda: equal(operation([x, w], 1, [2, 2]), x @ w))

        def refuse(name, fn, message):
            def call():
                try:
                    fn()
                except (RuntimeError, ValueError) as error:
                    assert message in str(error), str(error)
                else:
                    raise AssertionError("unsupported call was accepted")
            checked(name, call)

        refuse("shape_mismatch", lambda: operation([x], 31, [3, 2]), "output shape")
        refuse("strided_input", lambda: operation([x.t()], 31, [3, 2]), "contiguous")
        refuse("unsupported_dtype", lambda: operation([x.double()], 31, [2, 3]), "float32 or int32")
        refuse("unsupported_backward", lambda: operation([x.clone().requires_grad_()], 6, [2, 3]), "no registered backward")
        refuse("pointer_index_form", lambda: operation([x], 32, [2, 3], [0]*8), "tensor indices")
        refuse("unknown_operation", lambda: operation([x], 47, [2, 3]), "unknown operation")

        def repeated():
            for _ in range(20):
                equal(operation([x, y], 2, [2, 3]), x + y)
            # A failed call must leave the next successful execution usable.
            equal(operation([x, w], 1, [2, 2]), x @ w)
        checked("repeated_and_after_error", repeated)

        def capture():
            def model(a, b):
                return operation([a, b], 2, [2, 3])
            compiled = torch.compile(model, backend="eager", fullgraph=True)
            equal(compiled(x, y), model(x, y))
        checked("torch_fullgraph_capture", capture)

        def negative_control():
            actual = operation([x, y], 2, [2, 3])
            changed = actual.clone()
            changed.view(torch.int32).flatten()[0] ^= 1
            try:
                equal(changed, actual)
            except AssertionError:
                return
            raise AssertionError("word comparator failed to detect injected output corruption")
        checked("output_word_negative_control", negative_control)

        def native_boundary():
            # OpOverload itself is framework Python binding code, so qualify
            # the registered C++ implementation separately via dispatch dump.
            # Do not call a Python frame trace a whole-native-runtime pass.
            table = torch._C._dispatch_dump_table("mojolearn::operation")
            (out / "dispatch-table.txt").write_text(table)
            assert "registration.cpp" in table
        checked("native_registration_present", native_boundary)
        report["complete"] = not report["failures"]
    except Exception:
        report["failures"].append({"name": "setup", "traceback": traceback.format_exc()})
    (out / "adapter-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": report["complete"], "passed": len(report["checks"]),
                      "failures": [f["name"] for f in report["failures"]]}))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
