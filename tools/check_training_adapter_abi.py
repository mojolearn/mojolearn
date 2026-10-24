#!/usr/bin/env python3
"""External direct ABI checks for transport refusal and output bounds."""
import argparse
import ctypes as c
import json
from pathlib import Path
import traceback
import numpy as np
from native_training_fixtures import fixture, analytic


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--native-library", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    report = {"complete": False, "checks": [], "failures": []}
    try:
        lib = c.CDLL(str(a.native_library.resolve()))
        call = lib.mojolearn_adapter_train
        call.argtypes, call.restype = [c.c_int64] * 13, c.c_int64
        arrays, model, options, scalars = fixture("analytic_sgd")
        output = [np.full(2, -99., np.float32) for _ in range(3)]
        output += [np.zeros(24, np.int32), np.full(1, -99., np.float32)]
        def descriptors(values):
            return np.array([[v.ctypes.data, int(v.dtype == np.int32), v.ndim] + list(v.shape) + [1] * (8 - v.ndim)
                             for v in values], dtype=np.int64)
        ins, outs = descriptors(arrays), descriptors(output)
        program, opts, floats = np.array(model, np.int64), np.array(options, np.int64), np.array(scalars, np.float64)
        def invoke(device, host, desc=ins, dest=outs, capacity=2048):
            error = c.create_string_buffer(capacity)
            rc = call(device, host, desc.ctypes.data, len(arrays), dest.ctypes.data,
                      program.ctypes.data, len(program), opts.ctypes.data, len(opts),
                      floats.ctypes.data, len(floats), c.addressof(error), capacity)
            return rc, error.value.decode()
        rc, error = invoke(-1, 0)
        assert rc == 0, error
        analytic(output)
        np.savez(a.out / "cpu-analytic-sgd.npz", **{f"output_{i}": v for i, v in enumerate(output)})
        report["checks"].append("explicit CPU native whole fit analytic oracle")
        for device, host, phrase in ((-1, 1, "visible GPU zero"), (1, 0, "visible GPU zero"), (0, 2, "transport flag")):
            rc, error = invoke(device, host)
            assert rc == 1 and phrase in error, (rc, error)
            report["checks"].append(f"refused device={device}, host_buffers={host}")
        before = [v.copy() for v in output]
        bad_output = outs.copy(); bad_output[0, 3] += 1
        rc, error = invoke(-1, 0, dest=bad_output)
        assert rc == 1 and "vector shape" in error
        assert all(np.array_equal(x.view(np.uint32), y.view(np.uint32)) for x, y in zip(output, before))
        report["checks"].append("invalid output capacity refused before writes")
        rc, error = invoke(-1, 1, capacity=1)
        assert rc == 1 and error == ""
        report["checks"].append("one-byte error buffer is terminated")
        report["complete"] = True
    except Exception:
        report["failures"].append(traceback.format_exc())
    (a.out / "training-abi-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"complete": report["complete"], "passed": len(report["checks"]), "failures": len(report["failures"])}))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
