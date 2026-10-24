#!/usr/bin/env python3
"""External qualification controller, never imported by product execution.

The torch-compile phase of the adapter smoke. Builds (or, with --installed,
loads the shipped) PyTorch registration and checks the WHOLE-MODEL PyTorch
surface on one device:

  * torch.compile(model, backend="mojolearn") on several models: the output is
    close to eager PyTorch (torch sums in another order) and BIT FOR BIT the
    output of mojolearn.import_model(model, example_inputs=x).run(x), on the
    input's device, the same bits on every repeated call, re-imported after an
    in-place weight change;
  * a model with an operation the import does not support is refused, nothing
    runs eagerly;
  * the registration defines torch.ops.mojolearn.train and nothing else: the
    per-operation torch.ops.mojolearn.operation was removed on 2026-10-24
    (docs/TOMBSTONES.md) and its return is a failure here.

--stage build compiles the registration only (the training checks and their
fault controls load it). Keep logs and result JSON under the caller's retained
run directory. No performance or cross-vendor qualification is inferred here.
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fault", choices=["none", "training-wrong-lr", "training-stall"], default="none")
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
            if args.fault.startswith("training-"):
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
        report["surface"] = "whole-model: torch.compile backend, import_model, torch.ops.mojolearn.train"

        def checked(name, fn):
            try:
                fn()
                report["checks"].append({"name": name, "passed": True})
            except Exception:
                report["failures"].append({"name": name, "traceback": traceback.format_exc()})

        def registration():
            # The registration is whole-model only: train, never a per-operation entry.
            assert hasattr(torch.ops.mojolearn, "train"), "torch.ops.mojolearn.train is not registered"
            table = torch._C._dispatch_dump_table("mojolearn::train")
            (out / "dispatch-table.txt").write_text(table)
            assert "registration.cpp" in table
            try:
                torch.ops.mojolearn.operation
            except (AttributeError, RuntimeError):
                return
            raise AssertionError("torch.ops.mojolearn.operation is registered again: per-op surfaces were removed "
                                 "on 2026-10-24 and must not come back")
        checked("registration_is_whole_model_only", registration)

        import numpy as np
        if not args.installed:  # --installed: importing mojolearn.adapters.torch registered the backend
            from mojolearn.adapters import _torch_compile
            _torch_compile.register()
        from mojolearn import import_model

        def words(t):
            # Test oracle/inspection only: the raw 32-bit words of a result.
            a = t.detach().cpu().contiguous().numpy() if hasattr(t, "detach") else np.ascontiguousarray(t)
            return a.view(np.uint32 if a.dtype.itemsize == 4 else np.uint8)

        def same_bits(actual, expected, what):
            a, e = words(actual), words(expected)
            assert a.shape == e.shape and np.array_equal(a, e), what + ": words differ"

        class TwoHeads(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.body = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh())
                self.a, self.b = torch.nn.Linear(8, 3), torch.nn.Linear(8, 2)

            def forward(self, x):
                h = self.body(x)
                return self.a(h), torch.sigmoid(self.b(h))

        rng = np.random.default_rng(0)
        torch.manual_seed(0)
        models = {
            "mlp": (torch.nn.Sequential(torch.nn.Linear(3, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2)), (32, 3)),
            "conv": (torch.nn.Sequential(torch.nn.Conv2d(1, 4, 3, padding=1), torch.nn.ReLU(), torch.nn.Flatten(),
                                         torch.nn.Linear(4 * 6 * 6, 5)), (8, 1, 6, 6)),
            "layernorm_softmax": (torch.nn.Sequential(torch.nn.Linear(5, 12), torch.nn.LayerNorm(12), torch.nn.GELU(),
                                                      torch.nn.Linear(12, 4), torch.nn.Softmax(dim=-1)), (16, 5)),
            "two_outputs": (TwoHeads(), (10, 6)),
        }

        def whole_model(name, net, shape):
            net = net.eval().to(device)
            x = torch.from_numpy(rng.standard_normal(shape, dtype=np.float32)).to(device)
            torch._dynamo.reset()
            compiled = torch.compile(net, backend="mojolearn", fullgraph=True, dynamic=False)
            with torch.no_grad():
                got = compiled(x)
                eager = net(x)
                again = compiled(x)
            got = got if isinstance(got, tuple) else (got,)
            eager = eager if isinstance(eager, tuple) else (eager,)
            again = again if isinstance(again, tuple) else (again,)
            direct = import_model(net, example_inputs=x).run(x.detach().cpu().numpy())
            direct = direct if isinstance(direct, tuple) else (direct,)
            assert len(got) == len(eager) == len(direct), (len(got), len(eager), len(direct))
            for k, (g, e, d, r) in enumerate(zip(got, eager, direct, again)):
                assert g.device == x.device, f"output {k} on {g.device}, input on {x.device}"
                assert tuple(g.shape) == tuple(e.shape) and g.dtype == e.dtype
                # torch sums in another order: close to eager, never claimed to be its bits
                assert torch.allclose(g.cpu(), e.cpu(), rtol=1e-4, atol=1e-5), f"{name} output {k} far from eager"
                same_bits(g, d, f"{name} output {k}: compiled vs import_model")
                same_bits(r, g, f"{name} output {k}: repeated call")
            np.savez(out / f"torch-compile-{name}.npz", x=x.cpu().numpy(),
                     **{f"compiled_{k}": g.cpu().numpy() for k, g in enumerate(got)},
                     **{f"import_model_{k}": np.asarray(d) for k, d in enumerate(direct)})
            # an in-place weight change re-imports: the compiled output follows import_model on the new weights
            with torch.no_grad():
                first = next(net.parameters())
                first.mul_(0.5)
                changed = compiled(x)
            changed = changed if isinstance(changed, tuple) else (changed,)
            fresh = import_model(net, example_inputs=x).run(x.detach().cpu().numpy())
            fresh = fresh if isinstance(fresh, tuple) else (fresh,)
            for k, (c, f) in enumerate(zip(changed, fresh)):
                same_bits(c, f, f"{name} output {k}: after an in-place weight change")
        for name, (net, shape) in models.items():
            checked("torch_compile:" + name, lambda n=name, m=net, s=shape: whole_model(n, m, s))

        def refuses_unsupported():
            class TopK(torch.nn.Module):
                def forward(self, x):
                    return torch.topk(x, 2, dim=1).values
            torch._dynamo.reset()
            compiled = torch.compile(TopK(), backend="mojolearn", fullgraph=True, dynamic=False)
            try:
                with torch.no_grad():
                    compiled(torch.ones(4, 5, device=device))
            except Exception as error:  # torch wraps the backend's UnsupportedModel
                text = "%s: %s" % (type(error).__name__, error)
                assert "UnsupportedModel" in text or "TopK" in text or "topk" in text, text
                return
            raise AssertionError("an unsupported model compiled: something ran outside mojolearn")
        checked("torch_compile:refuses_unsupported_model", refuses_unsupported)

        def negative_control():
            net = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU()).eval()
            x = torch.from_numpy(rng.standard_normal((4, 3), dtype=np.float32))
            direct = import_model(net, example_inputs=x).run(x.numpy())
            changed = np.array(direct, copy=True)
            changed.view(np.uint32).flat[0] ^= 1
            try:
                same_bits(changed, direct, "negative control")
            except AssertionError:
                return
            raise AssertionError("word comparator failed to detect injected output corruption")
        checked("output_word_negative_control", negative_control)
        report["complete"] = not report["failures"]
    except Exception:
        report["failures"].append({"name": "setup", "traceback": traceback.format_exc()})
    (out / "adapter-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": report["complete"], "passed": len(report["checks"]),
                      "failures": [f["name"] for f in report["failures"]]}))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
