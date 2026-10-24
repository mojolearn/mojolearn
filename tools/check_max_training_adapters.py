#!/usr/bin/env python3
"""External MAX whole-training checks against independent SGD and retained Torch results.

TWO PROCESSES, ON PURPOSE. Importing mojolearn loads its bundled Mojo runtime, which creates the AsyncRT CPU
device with its own options; a MAX InferenceSession in the same process then asks for the CPU device with
different options and LLVM aborts the whole process ("AsyncRT::getOrCreateCPUDevice called requesting
different options to those used to create the existing CPUDevice", exit -6; the 0.8.56 Apple gate). So:

    parent  never imports MAX. Fits the Python API case (native_training_fixtures.api_case: mojolearn.Sequential)
            and writes its adapter inputs to OUT/max-side/python_api-input.npz + .json, keeping the API's five
            outputs; runs the child; compares the child's python_api outputs with the API's, byte for byte.
    child   this file with --max-side, run with the same interpreter (sys.executable): MAX only, never imports
            mojolearn. Runs every training case (Torch exact, resume exact, analytic SGD) and the python_api
            input, writing OUT/max-side/python_api.npz and OUT/max-side/max-side.json (checks, failures).
Writes OUT/max-training-results.json; exit 0 only when the child exited 0 and completed and the bytes match.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

SIDE = "max-side"


def byte_diff(actual, expected):
    """None when the output lists hold the same shapes, dtypes and bytes, else what differs: the slot, and for
    differing words how many and the first few (index, got, want) in hex. Test oracle/inspection only."""
    import numpy as np
    if len(actual) != len(expected):
        return "%d outputs != %d outputs" % (len(actual), len(expected))
    for slot, (a, b) in enumerate(zip(actual, expected)):
        if a.shape != b.shape or a.dtype != b.dtype:
            return "output %d: %s %s != %s %s" % (slot, a.shape, a.dtype, b.shape, b.dtype)
        got = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
        want = np.ascontiguousarray(b).view(np.uint8).reshape(-1)
        if got.size % 4 == 0:
            got, want = got.view(np.uint32), want.view(np.uint32)
        bad = np.flatnonzero(got != want)
        if bad.size:
            first = ", ".join("[%d] %x != %x" % (i, int(got[i]), int(want[i])) for i in bad[:4])
            return "output %d: %d of %d words differ: %s" % (slot, bad.size, got.size, first)
    return None


def max_side(args):
    """The child: MAX graphs only (mojolearn is never imported here)."""
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    report = {"complete": False, "checks": [], "failures": []}
    try:
        import numpy as np
        from max.driver import Accelerator, Buffer
        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, TensorType, ops
        from native_training_fixtures import CASES, fixture, identical, analytic
        device, dev = Accelerator(0), DeviceRef.GPU(0)
        session = InferenceSession(devices=[device])
        def run(name, arrays, program, options, scalars):
            types = [TensorType(DType.int32 if a.dtype == np.int32 else DType.float32,
                                a.shape, device=dev) for a in arrays]
            capacity = int(options[7]) + (arrays[-1].size if len(arrays) == 7 else 0)
            outputs = [TensorType(DType.float32, [arrays[2].size], device=dev) for _ in range(3)]
            outputs += [TensorType(DType.int32, [24], device=dev), TensorType(DType.float32, [capacity], device=dev)]
            with Graph(name, input_types=types, custom_extensions=[args.library.resolve()]) as graph:
                result = ops.custom("mojolearn_train_resume" if len(arrays) == 7 else "mojolearn_train", dev,
                    [v.tensor for v in graph.inputs], out_types=outputs,
                    parameters={"library": str(args.native_library.resolve()),
                                "model": ",".join(map(str, program)),
                                "options": ",".join(map(str, options)),
                                "scalars": ",".join(map(str, scalars))})
                graph.output(*[v.tensor for v in result])
            compiled = session.load(graph)
            values = [Buffer.from_numpy(a).to(device) for a in arrays]
            actual = [v.to_numpy().copy() for v in compiled.execute(*values)]
            identical([v.to_numpy() for v in values], arrays)
            identical([v.to_numpy() for v in compiled.execute(*values)], actual)
            return actual
        for name in (("analytic_sgd",) if args.smoke else CASES):
            arrays, program, options, scalars = fixture(name)
            actual = run(name, arrays, program, options, scalars)
            np.savez(out / (name + ".npz"), **{f"output_{i}": v for i, v in enumerate(actual)})
            with np.load(args.torch_results / (name + ".npz")) as saved:
                identical(actual, [saved[f"output_{i}"] for i in range(5)])
            if name == "analytic_sgd":
                analytic(actual)
            else:
                half = options.copy(); half[7] = 2
                first = run(name + "_first", arrays, program, half, scalars)
                half[12] = 0
                resumed = run(name + "_resume", arrays[:2] + first, program, half, scalars)
                identical(resumed, actual)
                report["checks"].append(name + ": resume exact")
            report["checks"].append(name + ": compile, execute, repeat, input ownership, Torch exact")
        # The Python API case's inputs, written by the parent (which fitted mojolearn.Sequential); the parent
        # compares these outputs with the API's. Program, options and scalars are kept as the exact strings the
        # parent's values print as, so the op receives what it would have in one process.
        spec = json.loads((out / "python_api-input.json").read_text())
        with np.load(out / "python_api-input.npz") as saved:
            arrays = [saved[f"input_{i}"] for i in range(spec["arrays"])]
        actual = run("python_api", arrays, spec["program"], spec["options"], spec["scalars"])
        np.savez(out / "python_api.npz", **{f"output_{i}": v for i, v in enumerate(actual)})
        report["checks"].append("python_api: MAX op executed on the Sequential.fit inputs")
        report["complete"] = True
    except Exception:
        report["failures"].append(traceback.format_exc())
    (out / "max-side.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["complete"] else 1


def child_command(args, side):
    cmd = [sys.executable, str(Path(__file__).resolve()), "--max-side", "--library", str(args.library),
           "--native-library", str(args.native_library), "--torch-results", str(args.torch_results),
           "--out", str(side)]
    return cmd + (["--smoke"] if args.smoke else [])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--torch-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-side", action="store_true", help="internal: the MAX-only child process")
    args = parser.parse_args(argv)
    if args.max_side:
        return max_side(args)
    args.out.mkdir(parents=True, exist_ok=True)
    side = args.out / SIDE
    side.mkdir(parents=True, exist_ok=True)
    report = {"complete": False, "checks": [], "failures": []}
    try:
        import numpy as np
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from native_training_fixtures import api_case
        # The MAX op is the Python API in another format: mojolearn.Sequential's fit, same bytes.
        expected = None
        try:
            arrays, program, options, scalars, expected = api_case()
            np.savez(side / "python_api-input.npz", **{f"input_{i}": a for i, a in enumerate(arrays)})
            (side / "python_api-input.json").write_text(json.dumps(dict(
                arrays=len(arrays), program=[str(v) for v in program], options=[str(v) for v in options],
                scalars=[str(v) for v in scalars])) + "\n")
        except Exception:
            report["failures"].append(traceback.format_exc())
        # The child shares this process group, so the smoke's per-phase bound kills it too.
        with open(args.out / (SIDE + ".log"), "w") as log:
            child = subprocess.run(child_command(args, side), stdout=log, stderr=subprocess.STDOUT)
        report["max_side"] = {"exit": child.returncode, "log": SIDE + ".log"}
        try:
            doc = json.loads((side / "max-side.json").read_text())
        except (OSError, ValueError):
            doc = {}
        report["checks"] += list(doc.get("checks") or [])
        report["failures"] += [str(f) for f in doc.get("failures") or []]
        if child.returncode != 0 or not doc.get("complete"):
            tail = (args.out / (SIDE + ".log")).read_text(errors="replace").splitlines()[-5:]
            report["failures"].append("MAX side exit %s, complete=%s; last log lines: %s"
                                      % (child.returncode, doc.get("complete"), " | ".join(tail)))
        if expected is not None:
            try:
                with np.load(side / "python_api.npz") as saved:
                    actual = [saved[f"output_{i}"] for i in range(len(saved.files))]
                np.savez(args.out / "python_api.npz", **{f"output_{i}": v for i, v in enumerate(actual)})
                diff = byte_diff(actual, expected)
                assert diff is None, "python_api: MAX op and Sequential.fit bytes differ: " + diff
                report["checks"].append("python_api: Sequential.fit and the MAX op return the same bytes "
                                        "(five outputs)")
            except Exception:
                report["failures"].append(traceback.format_exc())
        report["complete"] = not report["failures"]
    except Exception:
        report["failures"].append(traceback.format_exc())
    (args.out / "max-training-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": report["complete"], "passed": len(report["checks"]), "failures": len(report["failures"])}))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
