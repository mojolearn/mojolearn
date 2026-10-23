#!/usr/bin/env python3
"""External MAX graph qualification controller; run only on the owned test box."""
import argparse
import json
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"complete": False, "scope": "MAX AMD custom operation execution", "checks": [], "failures": []}
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
        x = np.array([[1., -2., .5], [-.25, 2., -1.]], dtype=np.float32)
        w = np.array([[.5, 1.], [-1., 2.], [.25, -.5]], dtype=np.float32)
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
        # Retain the exact test inputs/output, not only their hashes.
        np.savez(args.out / "max-fixtures.npz", x=x, w=w, expected=expected, actual=result)
        changed = result.copy()
        changed.view(np.int32).flat[0] ^= 1
        assert not np.array_equal(changed.view(np.int32), expected.view(np.int32))
        report["checks"].append("output_corruption_comparator_control")
        report["complete"] = True
    except Exception:
        report["failures"].append(traceback.format_exc())
    (args.out / "max-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": report["complete"], "passed": len(report["checks"]),
                      "failures": len(report["failures"])}))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
