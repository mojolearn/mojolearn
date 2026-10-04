#!/usr/bin/env python3
"""M3-only manager worker: quality OUTDIR or timing OUTDIR.

No repetitions, warmups, opponent calls, builds, SSH or queue operations.
A fresh exclusive output directory prevents replay under the same tag.
Quality oracle and fixtures are verification only, outside timing boundaries.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
# (name,m,k,n,ta,tb): three changed non-split shapes, two split controls.
SHAPES = [('dense', 4096, 4096, 4096, False, False),
          ('update', 4096, 256, 4096, False, False),
          ('transpose', 1024, 1024, 1024, True, True),
          ('gram-control', 220, 65536, 220, True, False),
          ('thin-control', 64, 65536, 220, True, False)]


def fixtures(m, k, n, ta, tb, kind, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, k)).astype('float32')
    b = rng.standard_normal((k, n)).astype('float32')
    if kind == 'rank1':
        a[:] = a[:, :1]
        b[:] = b[:1, :]
    elif kind == 'cancellation':
        a[:, 1::2] = -a[:, :k // 2 * 2:2]
        b[1::2, :] = b[:k // 2 * 2:2, :]
    return np.ascontiguousarray(a.T if ta else a), np.ascontiguousarray(b.T if tb else b)


def capture(action, out):
    import numpy as np
    assert os.environ.get('MOJOLEARN_VENDOR') == 'apple'
    assert os.environ.get('MOJOLEARN_NUMERIC_MODE') == 'fast'
    from mojolearn import _mojolearn_x_decomp as binding
    out.mkdir(parents=True, exist_ok=False)
    rows = {}
    shapes = SHAPES if action == 'timing' else [
        (f'{kind}-{int(ta)}{int(tb)}', 257, 271, 259, ta, tb)
        for kind in ('random', 'rank1', 'cancellation')
        for ta in (False, True) for tb in (False, True)]
    for name, m, k, n, ta, tb in shapes:
        kind = name.split('-')[0] if action == 'quality' else 'random'
        a, b = fixtures(m, k, n, ta, tb, kind, 742)
        c = np.empty((m, n), dtype='float32')
        args = [m, k, n, int(ta), int(tb)]
        if action == 'quality':
            binding.x_decomp_gemm(a.ctypes.data, b.ctypes.data, c.ctypes.data, args)
            ref = (a.T if ta else a).astype('float64') @ (b.T if tb else b).astype('float64')
            # Scale by ||A|| ||B||: well-defined for exact cancellation too.
            scale = max(float(np.linalg.norm(a.astype('float64')) * np.linalg.norm(b.astype('float64'))), np.finfo(float).tiny)
            rows[name] = dict(error=float(np.linalg.norm(c.astype('float64') - ref) / scale),
                              finite=bool(np.isfinite(c).all()),
                              output_sha256=hashlib.sha256(c.tobytes()).hexdigest())
            np.save(out / (name + '.npy'), c)
        else:
            # One scored host-facing call, including allocations and transfers.
            start = time.perf_counter_ns()
            binding.x_decomp_gemm(a.ctypes.data, b.ctypes.data, c.ctypes.data, args)
            returned = time.perf_counter_ns()
            checksum = float(np.sum(c, dtype='float64'))  # first full read
            consumed = time.perf_counter_ns()
            rows[name] = dict(call_ms=(returned-start)/1e6,
                              call_read_ms=(consumed-start)/1e6,
                              effective_tflops=2*m*n*k/(consumed-start)/1000,
                              checksum=checksum, finite=bool(np.isfinite(c).all()))
            # Separate resident route, also one scored call. A one-element
            # download provides a completion fence; this is not a pure GPU timer.
            ids = [binding.x_decomp_dev_alloc(size) for size in (a.size, b.size, c.size)]
            try:
                binding.x_decomp_dev_upload(ids[0], a.ctypes.data, a.size)
                binding.x_decomp_dev_upload(ids[1], b.ctypes.data, b.size)
                first = np.empty(1, dtype='float32')
                start = time.perf_counter_ns()
                binding.x_decomp_dev_gemm(*ids, args)
                binding.x_decomp_dev_download(ids[2], first.ctypes.data, 1)
                complete = time.perf_counter_ns()
                binding.x_decomp_dev_download(ids[2], c.ctypes.data, c.size)
                checksum = float(np.sum(c, dtype='float64'))
                consumed = time.perf_counter_ns()
                rows[name]['resident'] = dict(completion_ms=(complete-start)/1e6,
                    call_read_ms=(consumed-start)/1e6,
                    effective_tflops=2*m*n*k/(complete-start)/1000,
                    checksum=checksum, finite=bool(np.isfinite(c).all()))
            finally:
                for handle in ids:
                    binding.x_decomp_dev_free(handle)
        assert rows[name]['finite'], 'nonfinite output: ' + name
        if action == 'timing':
            assert rows[name]['resident']['finite'], 'nonfinite resident output: ' + name
        rows[name]['shape'] = dict(m=m, k=k, n=n, ta=ta, tb=tb)
        print('MMA-K16 ' + json.dumps(dict(case=name, **rows[name])), flush=True)
    so = ROOT / 'python/mojolearn/_mojolearn_x_decomp.so'
    (out / 'metrics.json').write_text(json.dumps(dict(action=action, cases=rows,
        binding_sha256=hashlib.sha256(so.read_bytes()).hexdigest()), indent=2) + '\n')


def compare(a, b):
    A = json.loads((a / 'metrics.json').read_text())['cases']
    B = json.loads((b / 'metrics.json').read_text())['cases']
    assert set(A) == set(B) and len(A) == 12
    # Fixed before measuring: no worse oracle error than A, no tolerance.
    failures = [name for name in A if not B[name]['finite'] or not
                (0 <= B[name]['error'] <= A[name]['error'])]
    print('MMA-K16-QUALITY ' + json.dumps(dict(status='FAIL' if failures else 'PASS', failures=failures)))
    return bool(failures)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('quality', 'timing', 'compare'))
    p.add_argument('paths', nargs='+', type=Path)
    args = p.parse_args()
    if args.action == 'compare':
        assert len(args.paths) == 2
        return compare(*args.paths)
    assert len(args.paths) == 1
    capture(args.action, args.paths[0])
    return 0


if __name__ == '__main__':
    sys.exit(main())
