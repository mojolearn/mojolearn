#!/usr/bin/env python3
"""External test: compare GPU host-buffer transport with native device inputs.

No Python from this file is imported by product code. Inputs and raw outputs
are retained for subsequent comparison across identically frozen vendor builds.
This run proves only the selected vendor; it is not cross-vendor qualification.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--torch-library", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"complete": False, "cases": [], "failures": [], "expected_cases": 48,
              "scope": "same-GPU host-buffer versus device-buffer transport"}
    try:
        import numpy as np
        import torch
        assert torch.cuda.is_available(), "requires a real NVIDIA or AMD GPU"
        report.update(torch=torch.__version__, hip=torch.version.hip,
                      hardware=torch.cuda.get_device_name(0),
                      native_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest())
        command = json.loads(args.torch_library.with_name('torch-build-command.json').read_text())
        assert '-DMOJOLEARN_ADAPTER_TEST_HOST_STAGING=1' not in command, 'reference must use the production device route'
        torch.ops.load_library(str(args.torch_library.resolve()))
        lib = ctypes.CDLL(str(args.library.resolve()))
        call = lib.mojolearn_adapter_operation_host_buffers
        call.argtypes = [ctypes.c_int64] * 11
        call.restype = ctypes.c_int64

        def descriptor(a):
            assert a.dtype in (np.dtype('float32'), np.dtype('int32')) and a.flags.c_contiguous
            return [a.ctypes.data, int(a.dtype == np.int32), a.ndim, *a.shape, *([1] * (8 - a.ndim))]

        def invoke(arrays, code, result, options=(), device=0):
            inputs = np.array([descriptor(a) for a in arrays], dtype=np.int64)
            output = np.array(descriptor(result), dtype=np.int64)
            ints = np.array(options, dtype=np.int64)
            error = ctypes.create_string_buffer(2048)
            rc = call(device, code, inputs.ctypes.data, len(arrays), output.ctypes.data,
                      ints.ctypes.data, len(ints), 0, 0, ctypes.addressof(error), len(error))
            return rc, error.value.decode()

        def values(shape, salt):
            count = int(np.prod(shape))
            return ((1 + (np.arange(count, dtype=np.int64) * 43 + salt * 71) % 113).astype(np.float32)
                    / np.float32(37)).reshape(shape)

        for rows, features, columns in ((2, 3, 4), (17, 19, 7), (3, 65, 5)):
            for code in range(1, 19):
                if code in (12, 14):
                    continue
                name = f"op{code}_{rows}_{features}_{columns}"
                x = values((rows, features), 3)
                y = values((features, columns) if code == 1 else x.shape, 7)
                arrays = [x, y] if code <= 5 else [x]
                shape = ((rows, columns) if code == 1 else (features, rows) if code == 13
                         else (1, 1) if code in (15, 16) else x.shape)
                options = [-1] if code in (15, 16) else []
                actual = np.empty(shape, dtype=np.float32)
                expected = torch.ops.mojolearn.operation(
                    [torch.from_numpy(a).to('cuda:0') for a in arrays], code, list(shape), options).cpu().numpy()
                rc, error = invoke(arrays, code, actual, options)
                np.savez(args.out / f"{name}.npz", actual=actual, expected=expected,
                         **{f"input_{i}": a for i, a in enumerate(arrays)})
                assert rc == 0, f"{name}: {error}"
                assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32)), name
                report['cases'].append({"name": name, "words": actual.size,
                                        "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest()})

        # The GPU-only entry must refuse CPU selection even with valid buffers.
        a = np.array([1., 2.], dtype=np.float32)
        out = np.empty_like(a)
        rc, error = invoke([a], 31, out, device=-1)
        assert rc != 0 and 'requires visible GPU zero' in error
        report['cpu_fallback_refused'] = True
        changed = actual.copy()
        changed.view(np.uint32).flat[0] ^= np.uint32(1)
        assert not np.array_equal(changed.view(np.uint32), expected.view(np.uint32))
        report['one_bit_negative_control'] = True
        assert len(report['cases']) == report['expected_cases']
        report['complete'] = True
    except Exception:
        report['failures'].append(traceback.format_exc())
    (args.out / 'transport-results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'complete': report['complete'], 'passed': len(report['cases']),
                      'failures': len(report['failures'])}))
    return 0 if report['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
