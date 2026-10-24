#!/usr/bin/env python3
"""External MAX whole-training checks against independent SGD and retained Torch results."""
import argparse
import json
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--torch-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
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
            capacity = options[7] + (arrays[-1].size if len(arrays) == 7 else 0)
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
            np.savez(args.out / (name + ".npz"), **{f"output_{i}": v for i, v in enumerate(actual)})
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
        report["complete"] = True
    except Exception:
        report["failures"].append(traceback.format_exc())
    (args.out / "max-training-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": report["complete"], "passed": len(report["checks"]), "failures": len(report["failures"])}))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
