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
import platform
import subprocess
import traceback


def device_parts(device):
    """(type, index) of a torch.device, a device string ("mps", "cuda:0") or anything with .type/.index."""
    if isinstance(device, str):
        kind, _, index = device.partition(":")
        return kind, (int(index) if index else None)
    return str(device.type), device.index


def same_device(actual, requested):
    """Same device type, and the same index when both name one. torch reports an MPS tensor on
    torch.device('mps', 0) while the requested device is 'mps' (index None), so plain == refuses it."""
    a_type, a_index = device_parts(actual)
    r_type, r_index = device_parts(requested)
    return a_type == r_type and (a_index is None or r_index is None or a_index == r_index)


def word_diff(got, want):
    """None when the two int32 word arrays are equal, else the byte diff: sizes, how many words differ
    and the first few (index, got, want) in hex. Test oracle/inspection only."""
    import numpy as np
    got, want = np.asarray(got).reshape(-1), np.asarray(want).reshape(-1)
    if got.size != want.size:
        return "%d words != %d words" % (got.size, want.size)
    bad = np.flatnonzero(got != want)
    if bad.size == 0:
        return None
    first = ", ".join("[%d] %08x != %08x" % (i, int(got[i]) & 0xFFFFFFFF, int(want[i]) & 0xFFFFFFFF) for i in bad[:4])
    return "%d of %d words differ: %s" % (bad.size, got.size, first)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fault", choices=["none", "wrong-operation", "stall", "training-wrong-lr", "training-stall"], default="none")
    p.add_argument("--stage", choices=["both", "build", "test"], default="both")
    p.add_argument("--host-staging", action="store_true",
                   help="test artifact only: force the MPS transport path on a CUDA/HIP GPU")
    p.add_argument("--installed", action="store_true",
                   help="check the adapters an installed mojolearn wheel ships (mojolearn.adapters.torch), "
                        "nothing built; --out receives the results only")
    args = p.parse_args()
    if args.installed and (args.fault != "none" or args.host_staging or args.stage != "both"):
        p.error("--installed checks the shipped registration as it is: no --fault, --host-staging or --stage")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"complete": False, "scope": "adapter functionality on one machine", "checks": [], "failures": [],
              "not_applicable": [], "test_host_staging": args.host_staging}
    try:
        import torch
        from torch.utils.cpp_extension import include_paths, library_paths
        root = Path(__file__).resolve().parents[1]
        report["torch"] = torch.__version__
        report["hip"] = torch.version.hip
        report["device"] = args.device
        report["fault"] = args.fault
        if args.installed:
            # THE INSTALLED WHEEL'S ADAPTERS: the shipped manifest, the shipped
            # library's digest, and the registration loaded the way a user
            # loads it (importing mojolearn.adapters.torch), never this tree's.
            from mojolearn.adapters import _shipped
            native = Path(_shipped.path("native"))
            report["installed"] = {k: _shipped.manifest().get(k) for k in ("vendor", "arch", "torch", "max", "files")}
            report["native_sha256"] = hashlib.sha256(native.read_bytes()).hexdigest()
            assert report["native_sha256"] == report["installed"]["files"]["native"]["sha256"], \
                "the installed adapter library differs from its adapters.json"
            from mojolearn.adapters import torch as installed_adapter
            report["registration"] = installed_adapter.library_path()
            report["compiled"] = False
        elif args.stage != "test":
            report["native_sha256"] = hashlib.sha256((out / "libmojolearn_adapter.so").read_bytes()).hexdigest()
            source = (root / "adapters/torch/registration.cpp").read_text()
            if args.fault == "wrong-operation":
                original = "device.is_cpu() ? -1 : 0, code,"
                assert source.count(original) == 1
                source = source.replace(original, "device.is_cpu() ? -1 : 0, code == 2 ? 3 : code,")
            elif args.fault == "stall":
                source = "#include <chrono>\n#include <thread>\n#include <cstdio>\n" + source
                original = "const auto status = entry("
                assert source.count(original) == 1
                source = source.replace(original,
                    'std::fputs("ADAPTER_FAULT_STALL_ENTERED\\n", stderr); std::fflush(stderr);\n'
                    "  for (;;) std::this_thread::sleep_for(std::chrono::seconds(1));\n  " + original)
            elif args.fault.startswith("training-"):
                original = "const auto train_status = mojolearn_adapter_train("
                assert source.count(original) == 1
                if args.fault == "training-wrong-lr":
                    injection = ("std::vector<double> injected_scalars(scalars.begin(), scalars.end());\n"
                                 "  injected_scalars[0] *= 2; scalars = at::ArrayRef<double>(injected_scalars);\n  ")
                else:
                    source = "#include <chrono>\n#include <thread>\n#include <cstdio>\n" + source
                    injection = ('std::fputs("TRAINING_FAULT_STALL_ENTERED\\n", stderr); std::fflush(stderr);\n'
                                 "  for (;;) std::this_thread::sleep_for(std::chrono::seconds(1));\n  ")
                source = source.replace(original, injection + original)
            # Retain the actual source bytes compiled, including fault controls.
            cpp = out / "registration.cpp"
            cpp.write_text(source)
            (out / "abi.h").write_bytes((root / "adapters/native/abi.h").read_bytes())
            # Native dispatcher registration requires libtorch, not its Python
            # binding library. cpp_extension.load also links libtorch_python;
            # build this standalone module using the public Torch path/ABI API.
            torch_libraries = library_paths()
            command = [os.environ.get("CXX", "c++"), "-std=c++20", "-O2", "-fPIC", "-shared",
                       f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
                       str(cpp), "-o", str(out / "mojolearn_torch.so")]
            if args.host_staging:
                command += ["-DMOJOLEARN_ADAPTER_TEST_HOST_STAGING=1"]
            command += [f"-I{path}" for path in include_paths()]
            command += [f"-L{path}" for path in torch_libraries] + [f"-L{out}"]
            link_check = "-Wl,-undefined,error" if platform.system() == "Darwin" else "-Wl,-z,defs"
            command += ["-lmojolearn_adapter", "-ltorch", "-ltorch_cpu", "-lc10", link_check,
                        f"-Wl,-rpath,{out}"] + [f"-Wl,-rpath,{path}" for path in torch_libraries]
            (out / "torch-build-command.json").write_text(json.dumps(command, indent=2) + "\n")
            subprocess.run(command, check=True)
            symbol_command = ["nm", "-g"] if platform.system() == "Darwin" else ["nm", "-D"]
            dependency_command = ["otool", "-L"] if platform.system() == "Darwin" else ["readelf", "-d"]
            symbols = subprocess.check_output(symbol_command + [str(out / "mojolearn_torch.so")], text=True)
            dependencies = subprocess.check_output(dependency_command + [str(out / "mojolearn_torch.so")], text=True)
            (out / "torch-symbols.txt").write_text(symbols)
            (out / "torch-dynamic.txt").write_text(dependencies)
            assert "libtorch_python" not in dependencies and "libpython" not in dependencies
            assert not any(line.split()[-1].lstrip("_").startswith("Py") for line in symbols.splitlines() if line.split())
            torch.ops.load_library(str(out / "mojolearn_torch.so"))
            report["compiled"] = True
            if args.stage == "build":
                (out / "build-results.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({"compiled": True, "fault": args.fault}))
                return 0
        else:
            report["native_sha256"] = hashlib.sha256((out / "libmojolearn_adapter.so").read_bytes()).hexdigest()
            build_command = json.loads((out / "torch-build-command.json").read_text())
            assert ("-DMOJOLEARN_ADAPTER_TEST_HOST_STAGING=1" in build_command) == args.host_staging
            torch.ops.load_library(str(out / "mojolearn_torch.so"))
        operation = torch.ops.mojolearn.operation
        device = torch.device(args.device)
        if device.type == "cuda":
            assert torch.cuda.is_available(), "GPU qualification cannot fall back to CPU"
            report["hardware"] = torch.cuda.get_device_name(device)
        elif device.type == "mps":
            assert torch.backends.mps.is_available(), "MPS qualification cannot fall back to CPU"
            assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0", "disable MPS CPU fallback for qualification"
            assert not args.host_staging, "MPS already uses staging; do not qualify a forced-route artifact"
            report["hardware"] = subprocess.check_output(["system_profiler", "SPDisplaysDataType"], text=True)
        if args.host_staging:
            assert device.type == "cuda", "forced staging checks require a real CUDA/HIP GPU"
        report["transport"] = "host_buffers_gpu_math" if device.type == "mps" or args.host_staging else "native_device_buffers"

        def checked(name, fn):
            try:
                fn()
                report["checks"].append({"name": name, "passed": True})
            except Exception:
                report["failures"].append({"name": name, "traceback": traceback.format_exc()})

        def equal(actual, expected):
            assert same_device(actual.device, expected.device), (str(actual.device), str(expected.device))
            assert actual.dtype == expected.dtype
            assert tuple(actual.shape) == tuple(expected.shape)
            # Test oracle/inspection only: compare raw words after byte copies.
            assert torch.equal(actual.cpu().contiguous().view(torch.int32),
                               expected.cpu().contiguous().view(torch.int32)), "output words differ"

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

        # The adapter is the Python API in another format: the same operation through mojolearn.ops.Session
        # (the plain Python API, on this install's GPU) must return the same words. Required for --installed;
        # a source-tree run checks it when mojolearn is importable.
        try:
            import numpy as np
            from mojolearn import ops as ml_ops
        except ImportError:
            ml_ops = None
            if args.installed:
                raise
        if ml_ops is None:
            report["not_applicable"].append("python_api: mojolearn is not importable in this environment")
        else:
            session = ml_ops.Session(numeric_mode="identical")
            host = lambda t: t.detach().cpu().contiguous().numpy()  # glue: framework tensor to host bytes
            X, Y, W = session.tensor(host(x)), session.tensor(host(y)), session.tensor(host(w))
            ids_host = host(ids)

            def api_equal(actual, handle, integer=False):
                # Test oracle/inspection only: compare raw words; the Session keeps matrices 2-D, so a 1-D
                # adapter result is compared as the same words in row-major order.
                expected = np.ascontiguousarray(session.numpy(handle))
                assert same_device(actual.device, device), \
                    "adapter output on %s, requested %s" % (actual.device, device)
                words = actual.detach().cpu().contiguous().view(torch.int32).numpy().reshape(-1)
                assert words.size == expected.size, (tuple(actual.shape), expected.shape)
                diff = word_diff(words, expected.view(np.int32).reshape(-1))
                assert diff is None, "adapter and Python API words differ: " + diff

            api = [
                ("add", lambda: operation([x, y], 2, [2, 3]), lambda: session.add(X, Y)),
                ("subtract", lambda: operation([x, y], 3, [2, 3]), lambda: session.sub(X, Y)),
                ("multiply", lambda: operation([x, y], 4, [2, 3]), lambda: session.mul(X, Y)),
                ("relu", lambda: operation([x], 6, [2, 3]), lambda: session.relu(X)),
                ("copy", lambda: operation([x], 31, [2, 3]), lambda: session.copy(X)),
                ("matmul", lambda: operation([x, w], 1, [2, 2]), lambda: session.matmul(X, W)),
                ("transpose", lambda: operation([x], 13, [3, 2]), lambda: session.transpose(X)),
                ("reshape", lambda: operation([x], 14, [3, 2], [3, 2]), lambda: session.reshape(X, (3, 2))),
                ("broadcast", lambda: operation([x, y[:1].contiguous()], 2, [2, 3]),
                 lambda: session.add(X, session.tensor(host(y[:1])))),
                ("embedding", lambda: operation([x, ids], 32, [2, 3], [-1]), lambda: session.embedding(X, ids_host)),
                ("take", lambda: operation([x, ids], 41, [2, 2], [1]), lambda: session.take(X, ids_host, axis=1)),
                ("int32_argmax", lambda: operation([x], 72, [2], [1, 0, 0], integer_output=True),
                 lambda: session.argmax(X, axis=1, int32=True)),
                ("concat", lambda: operation([x, y], 39, [4, 3], [0]), lambda: session.concat([X, Y], axis=0)),
                ("scale", lambda: operation([x], 12, [2, 3], scalars=[.5]), lambda: session.scale(X, .5)),
                ("full", lambda: operation([], 30, [2, 3], [2, 3], [.5], device=device),
                 lambda: session.full((2, 3), .5)),
                ("clip", lambda: operation([x], 56, [2, 3], scalars=[-1., 1.]), lambda: session.clip(X, -1., 1.)),
                ("linear_relu_composition", lambda: operation([operation([x, w], 1, [2, 2])], 6, [2, 2]),
                 lambda: session.relu(session.matmul(X, W))),
            ]
            for name, through_adapter, through_api in api:
                checked("python_api:" + name, lambda a=through_adapter, b=through_api: api_equal(a(), b()))

            def api_negative_control():
                changed = operation([x, y], 2, [2, 3]).cpu().clone()
                changed.view(torch.int32).flatten()[0] ^= 1
                try:
                    api_equal(changed.to(device), session.add(X, Y))
                except AssertionError:
                    return
                raise AssertionError("Python API comparator failed to detect injected output corruption")
            checked("python_api:negative_control", api_negative_control)
            report["python_api"] = "mojolearn.ops.Session(numeric_mode='identical')"

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
        if device.type == "cuda":
            checked("producer_on_nondefault_stream", alternate_stream)
        else:
            report["not_applicable"].append("producer_on_nondefault_stream: CUDA/HIP-specific API")

        def producer_consumer():
            produced = x * 2
            result = operation([produced, y], 2, [2, 3])
            equal(result * 2, (x * 2 + y) * 2)
        checked("queued_producer_and_consumer", producer_consumer)

        def offset_storage():
            base = torch.arange(30, dtype=torch.float32, device=device).reshape(10, 3)
            part = base[3:5]
            assert part.is_contiguous() and part.storage_offset() != 0
            before = base.clone()
            equal(operation([part], 31, [2, 3]), part)
            equal(base, before)
        checked("nonzero_storage_offset", offset_storage)

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
        refuse("unsupported_dtype", lambda: operation([x.half()], 31, [2, 3]), "float32 or int32")
        refuse("unsupported_backward", lambda: operation([x.clone().requires_grad_()], 6, [2, 3]), "no registered backward")
        refuse("pointer_index_form", lambda: operation([x], 32, [2, 3], [0]*8), "tensor indices")
        refuse("unknown_operation", lambda: operation([x], 47, [2, 3]), "unknown operation")
        refuse("integer_arithmetic", lambda: operation([ids, ids], 2, [2]), "requires float32")
        refuse("integer_value_table", lambda: operation([x.int(), ids], 32, [2, 3], [-1], integer_output=True), "requires float32")
        refuse("integer_concatenation", lambda: operation([ids, ids], 39, [4], [0], integer_output=True), "requires float32")
        refuse("float_indices", lambda: operation([x, ids.float()], 32, [2, 3], [-1]), "indices must be int32")
        refuse("wrong_output_dtype", lambda: operation([x], 31, [2, 3], integer_output=True), "output dtype")
        bad_ids = torch.tensor([99], dtype=torch.int32, device=device)
        refuse("out_of_bounds_indices", lambda: operation([x, bad_ids], 32, [1, 3], [-1]), "embedding")
        refuse("session_only_gather", lambda: operation([x, ids], 22, [2, 3], [0, 2]), "permutation provenance")
        if device.type != "cpu":
            refuse("mixed_devices", lambda: operation([x, y.cpu()], 2, [2, 3]), "share the output device")
        if device.type == "mps":
            checked("mps_implicit_index_no_inputs", lambda: equal(
                operation([], 30, [2, 3], [2, 3], [.5], device=torch.device("mps")), torch.full_like(x, .5)))

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
            changed = actual.cpu().clone()
            changed.view(torch.int32).flatten()[0] ^= 1
            try:
                equal(changed.to(device), actual)
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
