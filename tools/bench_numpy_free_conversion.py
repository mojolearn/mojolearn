#!/usr/bin/env python3
"""Interleaved large host-conversion benchmark; NumPy is a test oracle only.

No device learning is timed. NumPy baseline includes the large C->F row tiling
used before the NumPy-free migration, so comparisons do not use its slow
untiled fallback. Run with PYTHONPATH=python and a built base extension.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=2_000_000)
    parser.add_argument('--cols', type=int, default=20)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows, args.cols, args.samples) <= 0:
        parser.error('dimensions and sample count must be positive')
    import numpy as np
    from mojolearn import _buffer, _backend
    native = _backend.binding('_mojolearn')
    for name in ('cast_f64_to_f32', 'cast_colmajor_f64_to_f32', 'transpose_f32'):
        _buffer._native(name)  # raises by name on a stale binary; no Python fallback exists
    if args.rows * args.cols < 10_000_000:
        print('Reminder: use representative large data for performance decisions.', flush=True)
    results = []
    cases = [('float64', 'C', 'C'), ('float64', 'C', 'F'),
             ('float32', 'C', 'F'), ('float64', 'F', 'C'), ('float32', 'F', 'C')]
    for dtype, source, target in cases:
        x = np.arange(args.rows * args.cols, dtype=dtype).reshape(args.rows, args.cols)
        x *= .0001
        if source == 'F':
            x = np.asfortranarray(x)
        x.flags.writeable = False

        def baseline():
            if (target == 'F' and x.flags.c_contiguous and x.nbytes >= 8 * 1024**2
                    and args.cols >= 8):
                out = np.empty(x.shape, dtype=np.float32, order='F')
                step = max(1, 262144 // (args.cols * x.itemsize))
                for start in range(0, len(x), step):
                    out[start:start + step] = x[start:start + step]
                return out
            return (np.asfortranarray if target == 'F' else np.ascontiguousarray)(x, dtype=np.float32)

        def candidate():
            if target == 'F':
                return _buffer.as_f32_colmajor(x, name='X')[0]
            return _buffer.as_f32_c(x, ndim=2, name='X')[0]

        reference, got = baseline(), candidate()
        raw = reference.tobytes(order=target)
        assert got.tobytes() == raw
        digest = hashlib.sha256(raw).hexdigest()
        del reference, got, raw
        samples = {'numpy': [], 'native': []}
        for iteration in range(args.samples + 2):
            for arm in (('numpy', 'native') if iteration % 2 == 0 else ('native', 'numpy')):
                fn = baseline if arm == 'numpy' else candidate
                start = time.perf_counter_ns()
                out = fn()
                elapsed = (time.perf_counter_ns() - start) / 1e6
                if iteration >= 2:
                    samples[arm].append(elapsed)
                del out
        med = {arm: statistics.median(v) for arm, v in samples.items()}
        record = dict(dtype=dtype, source_order=source, target_order=target,
                      shape=list(x.shape), samples_ms=samples, median_ms=med,
                      native_over_numpy=med['native'] / med['numpy'], output_sha256=digest,
                      native_path=('cast_f64_to_f32' if source == target else
                                   'cast_colmajor_f64_to_f32' if dtype == 'float64' else 'transpose_f32'),
                      allocation='PyMem_RawMalloc, released after each call')
        results.append(record)
        print(json.dumps(record), flush=True)
    artifact = Path(native.__file__)
    report = dict(scope='host input conversion only; not GPU or whole-fit performance',
                  platform=platform.platform(), numpy_version=np.__version__,
                  native_artifact=artifact.name,
                  native_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                  results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
