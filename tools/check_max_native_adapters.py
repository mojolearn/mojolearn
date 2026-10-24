#!/usr/bin/env python3
"""External MAX graph qualification controller; run only on the owned test box.

TWO PROCESSES, ON PURPOSE. Importing mojolearn loads its bundled Mojo runtime, which creates the AsyncRT CPU
device with its own options; a MAX InferenceSession in the same process then asks for the CPU device with
different options and LLVM aborts the whole process ("AsyncRT::getOrCreateCPUDevice called requesting
different options to those used to create the existing CPUDevice", exit -6; the 0.8.56 Apple gate). So:

    child   this file with --max-side, run with the same interpreter (sys.executable): MAX only, never imports
            mojolearn. Builds and runs the graphs, checks them against the numpy references, and writes every
            output to OUT/max-side/<name>.npy plus OUT/max-side/max-side.json (checks, failures, outputs).
    parent  never imports MAX. Runs the child, then computes the same operations through mojolearn.ops.Session
            (the plain Python API on this GPU) and compares the child's output bytes with the API's, word for
            word, reporting the word diff on a mismatch. Writes OUT/max-results.json; exit 0 only when the child
            exited 0 and completed and every comparison passed.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

SIDE = "max-side"


def fixtures():
    """The test inputs both processes use (x, w, ids, bias)."""
    import numpy as np
    x = np.array([[1., -2., .5], [-.25, 2., -1.]], dtype=np.float32)
    w = np.array([[.5, 1.], [-1., 2.], [.25, -.5]], dtype=np.float32)
    ids = np.array([1, 0], dtype=np.int32)
    bias = np.array([[.5, -.25]], dtype=np.float32)
    return x, w, ids, bias


def api_references(api, x, w, ids, bias):
    """name -> the same operation through mojolearn.ops.Session (a handle); the names the child writes."""
    X, W = api.tensor(x), api.tensor(w)
    return {
        "matmul_relu_composition": lambda: api.relu(api.matmul(X, W)),
        "full_no_inputs": lambda: api.full((2, 3), .5),
        "scale_scalar_option": lambda: api.scale(X, .5),
        "transpose_view": lambda: api.transpose(X),
        "embedding_mixed_dtypes": lambda: api.embedding(X, ids),
        "take_mixed_dtypes": lambda: api.take(X, ids, axis=1),
        "argmax_integer_output": lambda: api.argmax(X, axis=1, int32=True),
        "linear_three_inputs": lambda: api.linear(X, W, api.tensor(bias)),
        "restore_composition": lambda: api.add(X, X),
    }


def word_diff(got, want):
    """None when the two arrays hold the same 32-bit words (row-major), else the byte diff: sizes, how many
    words differ and the first few (index, got, want) in hex. Test oracle/inspection only."""
    import numpy as np
    got = np.ascontiguousarray(got).view(np.int32).reshape(-1)
    want = np.ascontiguousarray(want).view(np.int32).reshape(-1)
    if got.size != want.size:
        return "%d words != %d words" % (got.size, want.size)
    bad = np.flatnonzero(got != want)
    if bad.size == 0:
        return None
    first = ", ".join("[%d] %08x != %08x" % (i, int(got[i]) & 0xFFFFFFFF, int(want[i]) & 0xFFFFFFFF) for i in bad[:4])
    return "%d of %d words differ: %s" % (bad.size, got.size, first)


def max_side(args):
    """The child: MAX graphs only (mojolearn is never imported here)."""
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    report = {"complete": False, "scope": "MAX custom operation execution on visible GPU zero", "checks": [],
              "failures": [], "outputs": {}}
    try:
        import numpy as np
        import max
        from max.driver import Accelerator, Buffer
        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, TensorType, ops
        report["max"] = getattr(max, "__version__", "unknown")
        device = Accelerator(0)
        dev = DeviceRef.GPU(0)
        session = InferenceSession(devices=[device])
        x, w, ids, bias = fixtures()

        def keep(name, actual):
            # The parent compares these bytes with the Python API's.
            np.save(out / (name + ".npy"), np.ascontiguousarray(actual))
            report["outputs"][name] = name + ".npy"
        input_types = [TensorType(DType.float32, a.shape, device=dev) for a in (x, w)]
        output_type = TensorType(DType.float32, [2, 2], device=dev)
        with Graph("mojolearn_native_matmul_relu", input_types=input_types,
                   custom_extensions=[args.library.resolve()]) as graph:
            product = ops.custom("mojolearn_matmul", dev, [v.tensor for v in graph.inputs],
                                 out_types=[output_type],
                                 parameters={"library": str(args.native_library.resolve())})[0].tensor
            result = ops.custom("mojolearn_relu", dev, [product], out_types=[output_type],
                                parameters={"library": str(args.native_library.resolve())})[0].tensor
            graph.output(result)
        model = session.load(graph)
        buffers = [Buffer.from_numpy(a).to(device) for a in (x, w)]
        expected = np.maximum(x @ w, np.float32(0))
        for iteration in range(4):
            result = model.execute(*buffers)[0].to_numpy()
            assert np.array_equal(result.view(np.int32), expected.view(np.int32)), result
        report["checks"].append("compiled_matmul_relu_composition_repeated")
        keep("matmul_relu_composition", result)
        # Retain the exact test inputs/output, not only their hashes.
        np.savez(out / "max-fixtures.npz", x=x, w=w, expected=expected, actual=result)
        changed = result.copy()
        changed.view(np.int32).flat[0] ^= 1
        assert not np.array_equal(changed.view(np.int32), expected.view(np.int32))
        report["checks"].append("output_corruption_comparator_control")

        def evaluate(name, arrays, code, expected, options="", scalars=""):
            types = [TensorType(DType.int32 if a.dtype == np.int32 else DType.float32,
                                a.shape, device=dev) for a in arrays]
            output_type = TensorType(DType.int32 if expected.dtype == np.int32 else DType.float32,
                                     expected.shape, device=dev)
            with Graph(name, input_types=types, custom_extensions=[args.library.resolve()]) as graph:
                result = ops.custom(f"mojolearn_operation{len(arrays)}", dev,
                    [v.tensor for v in graph.inputs], out_types=[output_type],
                    parameters={"library": str(args.native_library.resolve()), "code": code,
                                "options": options, "scalars": scalars})[0].tensor
                graph.output(result)
            compiled = session.load(graph)
            values = [Buffer.from_numpy(a).to(device) for a in arrays]
            actual = compiled.execute(*values)[0].to_numpy()
            assert actual.dtype == expected.dtype and actual.shape == expected.shape
            assert np.array_equal(actual.view(np.int32), expected.view(np.int32)), name
            np.savez(out / f"{name}.npz", expected=expected, actual=actual,
                     **{f"input_{i}": a for i, a in enumerate(arrays)})
            report["checks"].append(name)
            keep(name, actual)
        evaluate("full_no_inputs", [], 30, np.full((2, 3), .5, np.float32), "2,3", ".5")
        evaluate("scale_scalar_option", [x], 12, x * np.float32(.5), scalars=".5")
        evaluate("transpose_view", [x], 13, x.T.copy())
        evaluate("embedding_mixed_dtypes", [x, ids], 32, x[[1, 0]], "-1")
        evaluate("take_mixed_dtypes", [x, ids], 41, x[:, [1, 0]].copy(), "1")
        evaluate("argmax_integer_output", [x], 72, x.argmax(axis=1).astype(np.int32), "1,0,0")
        evaluate("linear_three_inputs", [x, w, bias], 19, x @ w + bias, "0")
        evaluate("restore_composition", [x, x], 2, x + x)
        report["complete"] = True
    except Exception:
        report["failures"].append(traceback.format_exc())
    (out / "max-side.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["complete"] else 1


def child_command(args, side):
    return [sys.executable, str(Path(__file__).resolve()), "--max-side", "--library", str(args.library),
            "--native-library", str(args.native_library), "--out", str(side)]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-side", action="store_true", help="internal: the MAX-only child process")
    args = parser.parse_args(argv)
    if args.max_side:
        return max_side(args)
    args.out.mkdir(parents=True, exist_ok=True)
    side = args.out / SIDE
    report = {"complete": False, "scope": "MAX custom operation execution on visible GPU zero", "checks": [],
              "failures": []}
    try:
        # The child shares this process group, so the smoke's per-phase bound kills it too.
        with open(args.out / (SIDE + ".log"), "w") as log:
            child = subprocess.run(child_command(args, side), stdout=log, stderr=subprocess.STDOUT)
        report["max_side"] = {"exit": child.returncode, "log": SIDE + ".log"}
        try:
            doc = json.loads((side / "max-side.json").read_text())
        except (OSError, ValueError):
            doc = {}
        report["max"] = doc.get("max")
        report["checks"] += list(doc.get("checks") or [])
        report["failures"] += [str(f) for f in doc.get("failures") or []]
        if child.returncode != 0 or not doc.get("complete"):
            tail = (args.out / (SIDE + ".log")).read_text(errors="replace").splitlines()[-5:]
            report["failures"].append("MAX side exit %s, complete=%s; last log lines: %s"
                                      % (child.returncode, doc.get("complete"), " | ".join(tail)))
        # The MAX ops are the Python API in another format: each output the child kept is compared, word for
        # word, with the same operation through mojolearn.ops.Session (the plain Python API on this GPU).
        import numpy as np
        from mojolearn import ops as ml_ops
        api = ml_ops.Session(numeric_mode="identical")
        outputs = doc.get("outputs") or {}
        for name, reference in api_references(api, *fixtures()).items():
            try:
                assert name in outputs, name + ": the MAX side kept no output"
                got = np.load(side / outputs[name])
                diff = word_diff(got, api.numpy(reference()))
                assert diff is None, name + ": MAX and Python API words differ: " + diff
                report["checks"].append("python_api:" + name)
            except Exception:
                report["failures"].append(traceback.format_exc())
        report["complete"] = not report["failures"]
    except Exception:
        report["failures"].append(traceback.format_exc())
    (args.out / "max-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": report["complete"], "passed": len(report["checks"]),
                      "failures": len(report["failures"])}))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
