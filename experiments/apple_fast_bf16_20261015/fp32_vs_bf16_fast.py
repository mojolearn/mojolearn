"""Apple FAST primitive A/B: FP32 FAST matmul against BF16 FAST matmul.

Testing code, never product execution. One process per arm, one excluded
warmup and one scored call per shape (one run per arm). Inputs are float32
standard-normal data, not BF16-representable, so the BF16 arms carry their
real rounding error against the FP64 oracle of the float32 inputs.

Arms:
  fp32-fast       linalg.matmul(a, b, identical=False) on float32 inputs.
  bf16-ready      linalg.matmul_bf16 on BF16 bits prepared outside the clock.
  bf16-from-fp32  linalg.to_bf16 on both float32 inputs inside the clock, then
                  linalg.matmul_bf16 (to_bf16 round-trips through the host, so
                  this bounds a model-side cast from above).

Usage:
  python fp32_vs_bf16_fast.py prepare OUT [--shape M,N,K ...]
  MOJOLEARN_NUMERIC_MODE=fast python fp32_vs_bf16_fast.py ARM OUT
"""
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

SHAPES = [(2048, 2048, 2048), (4096, 4096, 4096), (3000, 5000, 2500)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bits(x):
    u = x.view(np.uint32)
    return ((u + np.uint32(0x7fff) + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def prepare(root, shapes):
    root.mkdir(parents=True)
    cases = []
    for m, n, k in shapes:
        rng = np.random.default_rng(7)
        a = rng.standard_normal((m, k)).astype(np.float32)
        b = rng.standard_normal((k, n)).astype(np.float32)
        oracle = a.astype(np.float64) @ b.astype(np.float64)
        path = root / ('input-%d-%d-%d.npz' % (m, n, k))
        np.savez(path, a=a, b=b, a_bits=bits(a), b_bits=bits(b), oracle=oracle)
        cases.append(dict(m=m, n=n, k=k, path=path.name, sha256=digest(path)))
    (root / 'inputs.json').write_text(json.dumps(dict(seed=7, cases=cases), indent=1))


def run(root, arm):
    os.environ.get('MOJOLEARN_NUMERIC_MODE') == 'fast' or sys.exit('set MOJOLEARN_NUMERIC_MODE=fast')
    from mojolearn import linalg, _linalg_impl
    binding = _linalg_impl._load()
    assert int(binding.linalg_numeric_mode()) == 0, 'not the FAST build'
    assert linalg._fast_bf16_available(), 'BF16 FAST not enabled; refuse a substitute'
    if arm == 'fp32-fast':
        call = lambda d: linalg.matmul(d['a'], d['b'], identical=False)
    elif arm == 'bf16-ready':
        call = lambda d: linalg.matmul_bf16(d['a_bits'], d['b_bits'])
    elif arm == 'bf16-from-fp32':
        call = lambda d: linalg.matmul_bf16(linalg.to_bf16(d['a']), linalg.to_bf16(d['b']))
    else:
        raise ValueError(arm)
    record = dict(arm=arm, platform=platform.platform(), binary=str(Path(binding.__file__).resolve()),
                  binary_sha256=digest(binding.__file__), warmup=1, scored=1, cases=[])
    inputs = json.loads((root / 'inputs.json').read_text())
    for case in inputs['cases']:
        path = root / case['path']
        assert digest(path) == case['sha256']
        with np.load(path, allow_pickle=False) as z:
            d = {key: z[key] for key in z.files}
        oracle = d.pop('oracle')
        times = []
        for _ in range(2):
            start = time.perf_counter_ns()
            out = np.asarray(call(d), dtype=np.float32).copy(order='C')
            times.append((time.perf_counter_ns() - start) / 1e6)
        err = np.abs(out.astype(np.float64) - oracle)
        scale = float(np.sqrt(np.mean(oracle * oracle)))
        record['cases'].append(dict(shape=[case['m'], case['n'], case['k']], warmup_ms=times[0],
                                    scored_ms=times[1], finite=bool(np.isfinite(out).all()),
                                    max_abs_error=float(err.max()),
                                    rel_rms_error=float(np.sqrt(np.mean(err * err))) / scale))
    (root / (arm + '.json')).write_text(json.dumps(record, indent=1))
    print(json.dumps([dict(arm=arm, shape=c['shape'], ms=round(c['scored_ms'], 2),
                           rel_rms=c['rel_rms_error']) for c in record['cases']]))


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        extra = [tuple(int(v) for v in s.split(',')) for s in sys.argv[3:] if s != '--shape']
        prepare(Path(sys.argv[2]), extra or SHAPES)
    else:
        run(Path(sys.argv[2]), sys.argv[1])
