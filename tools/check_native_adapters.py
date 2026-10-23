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
import os
import subprocess
import traceback


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fault", choices=["none", "wrong-operation", "stall"], default="none")
    p.add_argument("--stage", choices=["both", "build", "test"], default="both")
    args = p.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"complete": False, "scope": "adapter functionality on one machine", "checks": [], "failures": []}
    try:
        import torch
        from torch.utils.cpp_extension import include_paths, library_paths
        root = Path(__file__).resolve().parents[1]
        report["torch"] = torch.__version__
        report["hip"] = torch.version.hip
        report["device"] = args.device
        report["native_sha256"] = hashlib.sha256((out / "libmojolearn_adapter.so").read_bytes()).hexdigest()
        report["fault"] = args.fault
        if args.stage != "test":
            source = (root / "adapters/torch/registration.cpp").read_text()
            if args.fault == "wrong-operation":
                original = "device.is_cpu() ? -1 : 0, code,"
                assert source.count(original) == 1
                source = source.replace(original, "device.is_cpu() ? -1 : 0, code == 2 ? 3 : code,")
            elif args.fault == "stall":
                source = "#include <chrono>\n#include <thread>\n#include <cstdio>\n" + source
                original = "const auto status = mojolearn_adapter_operation("
                assert source.count(original) == 1
                source = source.replace(original,
                    'std::fputs("ADAPTER_FAULT_STALL_ENTERED\\n", stderr); std::fflush(stderr);\n'
                    "  for (;;) std::this_thread::sleep_for(std::chrono::seconds(1));\n  " + original)
            # Retain the actual source bytes compiled, including fault controls.
            cpp = out / "registration.cpp"
            cpp.write_text(source)
            # Native dispatcher registration requires libtorch, not its Python
            # binding library. cpp_extension.load also links libtorch_python;
            # build this standalone module using the public Torch path/ABI API.
            torch_libraries = library_paths()
            command = [os.environ.get("CXX", "c++"), "-std=c++20", "-O2", "-fPIC", "-shared",
                       f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
                       str(cpp), "-o", str(out / "mojolearn_torch.so")]
            command += [f"-I{path}" for path in include_paths()]
            command += [f"-L{path}" for path in torch_libraries] + [f"-L{out}"]
            command += ["-lmojolearn_adapter", "-ltorch", "-ltorch_cpu", "-lc10", "-Wl,-z,defs",
                        f"-Wl,-rpath,{out}"] + [f"-Wl,-rpath,{path}" for path in torch_libraries]
            (out / "torch-build-command.json").write_text(json.dumps(command, indent=2) + "\n")
            subprocess.run(command, check=True)
            symbols = subprocess.check_output(["nm", "-D", str(out / "mojolearn_torch.so")], text=True)
            dependencies = subprocess.check_output(["readelf", "-d", str(out / "mojolearn_torch.so")], text=True)
            (out / "torch-symbols.txt").write_text(symbols)
            (out / "torch-dynamic.txt").write_text(dependencies)
            assert "libtorch_python" not in dependencies and "libpython" not in dependencies
            assert not any(line.split()[-1].startswith(("Py", "_Py")) for line in symbols.splitlines() if line.split())
            torch.ops.load_library(str(out / "mojolearn_torch.so"))
            report["compiled"] = True
            if args.stage == "build":
                (out / "build-results.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({"compiled": True, "fault": args.fault}))
                return 0
        else:
            torch.ops.load_library(str(out / "mojolearn_torch.so"))
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
        checked("transpose_materialization", lambda: equal(operation([x], 13, [3, 2]), x.t().contiguous()))
        checked("reshape", lambda: equal(operation([x], 14, [3, 2], [3, 2]), x.reshape(3, 2)))
        checked("broadcast", lambda: equal(operation([x, y[:1].contiguous()], 2, [2, 3]), x + y[:1]))
        ids = torch.tensor([1, 0], dtype=torch.int32, device=device)
        checked("int32_copy", lambda: equal(operation([ids], 31, [2], integer_output=True), ids))
        special = torch.tensor([-2147483648, 1, 2143289410, 2139095040, -8388608], dtype=torch.int32, device=device).view(torch.float32)
        checked("copy_preserves_special_float_words", lambda: equal(operation([special], 31, [5]), special))
        checked("embedding_indices", lambda: equal(operation([x, ids], 32, [2, 3], [-1]), x[[1, 0]]))
        checked("take_indices", lambda: equal(operation([x, ids], 41, [2, 2], [1]), x[:, [1, 0]]))
        checked("int32_argmax", lambda: equal(operation([x], 72, [2], [1, 0, 0], integer_output=True), x.argmax(1).int()))
        checked("concat", lambda: equal(operation([x, y], 39, [4, 3], [0]), torch.cat([x, y])))
        checked("scale", lambda: equal(operation([x], 12, [2, 3], scalars=[.5]), x * .5))
        checked("full_without_inputs", lambda: equal(operation([], 30, [2, 3], [2, 3], [.5], device=device), torch.full_like(x, .5)))
        checked("clip", lambda: equal(operation([x], 56, [2, 3], scalars=[-1., 1.]), x.clamp(-1., 1.)))

        def unchanged_and_lifetime():
            before = x.clone()
            result = operation([x], 31, [2, 3])
            assert result.data_ptr() != x.data_ptr()
            equal(x, before)
            temporary = x.clone()
            copied = operation([temporary], 31, [2, 3])
            del temporary
            equal(copied, before)
        checked("ownership_and_input_unchanged", unchanged_and_lifetime)

        def alternate_stream():
            if device.type != "cuda":
                return
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                produced = x * 2
                result = operation([produced, y], 2, [2, 3])
            torch.cuda.current_stream(device).wait_stream(stream)
            equal(result, x * 2 + y)
        checked("producer_on_nondefault_stream", alternate_stream)

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
        refuse("integer_arithmetic", lambda: operation([ids, ids], 2, [2]), "requires float32")
        refuse("float_indices", lambda: operation([x, ids.float()], 32, [2, 3], [-1]), "indices must be int32")
        refuse("wrong_output_dtype", lambda: operation([x], 31, [2, 3], integer_output=True), "output dtype")
        bad_ids = torch.tensor([99], dtype=torch.int32, device=device)
        refuse("out_of_bounds_indices", lambda: operation([x, bad_ids], 32, [1, 3], [-1]), "embedding")
        refuse("session_only_gather", lambda: operation([x, ids], 22, [2, 3], [0, 2]), "permutation provenance")

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

        def inductor():
            def model(a, b):
                return operation([a, b], 2, [2, 3])
            compiled = torch.compile(model, backend="inductor", fullgraph=True)
            equal(compiled(x, y), x + y)
        checked("torch_inductor_custom_op", inductor)

        def scripted_composition():
            def model(a, b):
                product = operation([a, b], 1, [2, 2])
                return operation([product], 6, [2, 2])
            compiled = torch.jit.trace(model, (x, w))
            compiled.save(str(out / "torch-composition.pt"))
            for _ in range(4):
                equal(compiled(x, w), torch.relu(x @ w))
        checked("native_torchscript_composition", scripted_composition)

        def exported_composition():
            class Model(torch.nn.Module):
                def forward(self, a, b):
                    product = operation([a, b], 1, [2, 2])
                    return operation([product], 6, [2, 2])
            exported = torch.export.export(Model(), (x, w), strict=True)
            torch.export.save(exported, out / "torch-composition.pt2")
            equal(exported.module()(x, w), torch.relu(x @ w))
        checked("torch_export_custom_ops", exported_composition)

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
