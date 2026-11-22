"""Owner-requested local screening, not full-workload/default-promotion evidence.

Testing controller only. Each arm starts with the same retained uint16 BF16
host operands and returns a consumed float32 host result. Copies, allocation,
upload, required widening, matmul, synchronization and download are timed.
NumPy preparation and independent float64 oracle work are outside the clock.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent / 'local-medium'
SHAPES = [(256, 384, 320), (768, 1024, 640), (769, 1025, 641)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prepare():
    ROOT.mkdir(exist_ok=True)
    assert not (ROOT / 'inputs.json').exists(), 'Never overwrite a retained attempt'
    rng = np.random.default_rng(82471)
    records = []
    for m, n, k in SHAPES:
        a = rng.uniform(-1, 1, (m, k)).astype(np.float32)
        b = rng.uniform(-1, 1, (k, n)).astype(np.float32)
        def packed(x):
            u = x.view(np.uint32)
            return ((u + np.uint32(0x7fff) + ((u >> 16) & 1)) >> 16).astype(np.uint16)
        ab, bb = packed(a), packed(b)
        aw = (ab.astype(np.uint32) << 16).view(np.float32)
        bw = (bb.astype(np.uint32) << 16).view(np.float32)
        oracle = aw.astype(np.float64) @ bw.astype(np.float64)
        scale = np.abs(aw).astype(np.float64) @ np.abs(bw).astype(np.float64)
        path = ROOT / ('input-%d-%d-%d.npz' % (m, n, k))
        np.savez(path, a=ab, b=bb, oracle=oracle, abs_product_sum=scale)
        records.append(dict(m=m, n=n, k=k, path=str(path), sha256=digest(path), bytes=path.stat().st_size))
    save(ROOT / 'inputs.json', dict(seed=82471, cases=records, shapes_reason='small screening, medium rectangular workload and neighboring tail dimensions; no dispatch changes',
        input='prepacked BF16 uint16 host buffers', accumulation='float32', output='float32 host array',
        scope='local preliminary operation screening; no training/default-promotion/full-board claim'))


def run(arm):
    directory = ROOT / arm
    directory.mkdir()
    inputs = json.loads((ROOT / 'inputs.json').read_text())
    record = dict(arm=arm, status='running', python=sys.version, platform=platform.platform(),
                  input_manifest_sha256=digest(ROOT / 'inputs.json'), cases=[],
                  boundary='ready BF16 host operands through fully materialized float32 host output; native packing, allocation, upload, widening, compute, sync and download included',
                  excluded='library import, test-data generation, oracle, hashing, evidence serialization',
                  first_call='first operation in this process, not a claim of empty system caches',
                  warmup=1, scored_samples=5, cross_vendor_identity='not compared',
                  scope='local medium-sized screening only')
    if arm == 'candidate-fast':
        from mojolearn import linalg_fast as api
        assert api.has_bf16(), 'Candidate not enabled; refuse to time a substitute'
        function = api.matmul_bf16
        import mojolearn._mojolearn_linalg as binding
        record.update(module=str(Path(binding.__file__).resolve()), binary_sha256=digest(binding.__file__),
                      numeric_mode='fast', source=os.environ['MOJOLEARN_BENCH_SOURCE_COMMIT'])
    elif arm == 'released-identical':
        from mojolearn import linalg as api
        from mojolearn import _linalg_impl
        function = api.matmul_bf16
        binding = _linalg_impl._load()
        record.update(version=importlib.metadata.version('mojolearn'), numeric_mode='identical',
                      module=str(Path(binding.__file__).resolve()), binary_sha256=digest(binding.__file__),
                      wheel_manifest=str(Path(__file__).resolve().parents[1] / 'reviews/hour-4/apple-moe-0845/wheel-manifest.json'))
    elif arm == 'torch-mps':
        import torch
        assert torch.backends.mps.is_available(), 'No MPS GPU; CPU fallback is forbidden'
        assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
        record.update(version=torch.__version__, device='mps', matmul='torch.mm float32',
                      mps_environment={k: os.environ.get(k) for k in ('PYTORCH_ENABLE_MPS_FALLBACK', 'PYTORCH_MPS_FAST_MATH', 'PYTORCH_MPS_PREFER_METAL')},
                      precision='BF16 input bits widened exactly on GPU; FP32 multiplication, accumulation and output')
        def function(a, b):
            # Real library operations, not a harness-written GEMM. Starting
            # from BF16 buffers and widening is inside the opponent's clock.
            ta = torch.from_numpy(a).view(torch.bfloat16).to('mps').to(torch.float32)
            tb = torch.from_numpy(b).view(torch.bfloat16).to('mps').to(torch.float32)
            out = torch.mm(ta, tb).cpu().numpy()
            torch.mps.synchronize()
            return out
    else:
        raise ValueError(arm)
    save(directory / 'summary.json', record)
    failures = []
    for case in inputs['cases']:
        assert digest(case['path']) == case['sha256']
        with np.load(case['path'], allow_pickle=False) as data:
            a, b, oracle, scale = (data[k] for k in ('a', 'b', 'oracle', 'abs_product_sum'))
        m, n, k = case['m'], case['n'], case['k']
        row = dict(m=m, n=n, k=k, samples=[], input_sha256=case['sha256'])
        for index in range(-2, 5):
            start = time.perf_counter_ns()
            output = np.asarray(function(a, b), dtype=np.float32).copy(order='C')
            elapsed = (time.perf_counter_ns() - start) / 1e6
            assert output.shape == (m, n)
            error = np.abs(output.astype(np.float64) - oracle)
            bound = 4 * k * np.finfo(np.float32).eps * scale + 1e-6
            finite = bool(np.isfinite(output).all())
            passed = finite and bool(np.all(error <= bound))
            sample = dict(index=index, role='first' if index == -2 else 'excluded-warmup' if index == -1 else 'scored',
                          milliseconds=elapsed, accuracy_pass=passed, finite=finite,
                          max_abs_error=float(error.max()), rms_error=float(np.sqrt(np.mean(error * error))))
            file = directory / ('output-%d-%d-%d-%d.npy' % (m, n, k, index))
            np.save(file, output, allow_pickle=False)
            sample.update(path=str(file), sha256=digest(file), bytes=file.stat().st_size)
            row['samples'].append(sample)
            if not passed:
                failures.append(dict(shape=[m,n,k], sample=index, reason='accuracy or finite output failed'))
        scored = [s['milliseconds'] for s in row['samples'] if s['role'] == 'scored']
        row.update(median_ms=statistics.median(scored), min_ms=min(scored), max_ms=max(scored))
        record['cases'].append(row)
        save(directory / 'summary.json', record)
    record.update(status='finished', complete=len(record['cases']) == len(SHAPES), failures=failures)
    save(directory / 'summary.json', record)
    print(json.dumps(dict(arm=arm, complete=record['complete'], failures=failures,
                         timings=[dict(shape=[r['m'],r['n'],r['k']], median_ms=r['median_ms']) for r in record['cases']])))
    return 1 if failures else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('arm', choices=['prepare', 'candidate-fast', 'released-identical', 'torch-mps'])
    args = parser.parse_args()
    if args.arm == 'prepare':
        prepare()
    else:
        raise SystemExit(run(args.arm))
