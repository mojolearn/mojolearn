"""Portable testing controller for local BF16 screening; not product execution.

Adapted from the retained executed_controller.py; historical results were
produced by those original bytes, not this portable adaptation.
Owner-requested local screening, not full-workload/default-promotion evidence.

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

SHAPES = [(256, 384, 320), (768, 1024, 640), (769, 1025, 641)]
# Full primitive recipe: tools/bench_board_neural.py GEMM_SHAPES["full"]
# and its seed-7 standard-normal float32 input generator. Test data only;
# these dimensions never control product dispatch or establish estimator coverage.
FULL_SHAPES = [(4096, 4096, 4096)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prepare(root, shapes, preset="screen"):
    # A partial preparation also contains unique evidence: require a new path.
    root.mkdir(parents=True, exist_ok=False)
    seed = 7 if preset == "full" else 82471
    rng = np.random.default_rng(seed)
    records = []
    for m, n, k in shapes:
        if preset == "full":
            a = rng.standard_normal((m, k)).astype(np.float32)
            b = rng.standard_normal((k, n)).astype(np.float32)
        else:
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
        path = root / ('input-%d-%d-%d.npz' % (m, n, k))
        np.savez(path, a=ab, b=bb, oracle=oracle, abs_product_sum=scale)
        records.append(dict(m=m, n=n, k=k, path=path.name, sha256=digest(path), bytes=path.stat().st_size))
    save(root / 'inputs.json', dict(seed=seed, preset=preset, cases=records,
        distribution='standard_normal float32' if preset == 'full' else 'uniform [-1,1] float32',
        recipe_source='tools/bench_board_neural.py GEMM_SHAPES.full and SEED' if preset == 'full' else 'local operation screening',
        shapes_reason='explicit test shapes; full is the declared full GEMM primitive, not an estimator dataset; no dispatch changes',
        input='prepacked BF16 uint16 host buffers', accumulation='float32', output='float32 host array',
        scope='local preliminary operation screening; no training/default-promotion/full-board claim'))


def run(root, arm, *, source_commit, wheel_manifest, samples, warmup):
    directory = root / arm
    directory.mkdir()
    inputs = json.loads((root / 'inputs.json').read_text())
    record = dict(arm=arm, status='running', python=sys.version, platform=platform.platform(),
                  input_manifest_sha256=digest(root / 'inputs.json'), cases=[],
                  controller_sha256=digest(__file__), controller_source='controller.py',
                  boundary='ready BF16 host operands through fully materialized float32 host output; native packing, allocation, upload, widening, compute, sync and download included',
                  excluded='library import, test-data generation, oracle, hashing, evidence serialization',
                  first_call='first recorded call per shape; only first shape is first in process; system caches not claimed empty',
                  warmup=warmup, scored_samples=samples, cross_vendor_identity='not compared',
                  scope='local operation screening only')
    (directory / 'controller.py').write_bytes(Path(__file__).read_bytes())
    save(directory / 'summary.json', record)
    if arm == 'candidate-fast':
        from mojolearn import linalg as api
        assert api._fast_bf16_available(), 'Candidate not enabled; refuse to time a substitute'
        function = api.matmul_bf16
        import mojolearn._mojolearn_linalg as binding
        record.update(module=str(Path(binding.__file__).resolve()), binary_sha256=digest(binding.__file__),
                      numeric_mode='fast', source=source_commit, public_api='mojolearn.linalg.matmul_bf16')
    elif arm == 'released-identical':
        from mojolearn import linalg as api
        from mojolearn import _linalg_impl
        function = api.matmul_bf16
        binding = _linalg_impl._load()
        record.update(version=importlib.metadata.version('mojolearn'), numeric_mode='identical',
                      module=str(Path(binding.__file__).resolve()), binary_sha256=digest(binding.__file__),
                      wheel_manifest=str(wheel_manifest) if wheel_manifest else None,
                      wheel_manifest_sha256=digest(wheel_manifest) if wheel_manifest else None)
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
        input_path = Path(case['path'])
        if not input_path.is_absolute():
            input_path = root / input_path
        assert digest(input_path) == case['sha256']
        with np.load(input_path, allow_pickle=False) as data:
            a, b, oracle, scale = (data[k] for k in ('a', 'b', 'oracle', 'abs_product_sum'))
        m, n, k = case['m'], case['n'], case['k']
        row = dict(m=m, n=n, k=k, samples=[], input_sha256=case['sha256'])
        for index in range(-warmup - 1, samples):
            start = time.perf_counter_ns()
            output = np.asarray(function(a, b), dtype=np.float32).copy(order='C')
            elapsed = (time.perf_counter_ns() - start) / 1e6
            assert output.shape == (m, n)
            error = np.abs(output.astype(np.float64) - oracle)
            bound = 4 * k * np.finfo(np.float32).eps * scale + 1e-6
            finite = bool(np.isfinite(output).all())
            passed = finite and bool(np.all(error <= bound))
            sample = dict(index=index, role='first' if index == -warmup - 1 else 'excluded-warmup' if index < 0 else 'scored',
                          milliseconds=elapsed, accuracy_pass=passed, finite=finite,
                          max_abs_error=float(error.max()), rms_error=float(np.sqrt(np.mean(error * error))))
            file = directory / ('output-%d-%d-%d-%d.npy' % (m, n, k, index))
            np.save(file, output, allow_pickle=False)
            sample.update(path=file.name, sha256=digest(file), bytes=file.stat().st_size)
            row['samples'].append(sample)
            if not passed:
                failures.append(dict(shape=[m,n,k], sample=index, reason='accuracy or finite output failed'))
        scored = [s['milliseconds'] for s in row['samples'] if s['role'] == 'scored']
        row.update(median_ms=statistics.median(scored), min_ms=min(scored), max_ms=max(scored))
        record['cases'].append(row)
        save(directory / 'summary.json', record)
    record.update(status='finished', complete=len(record['cases']) == len(inputs['cases']), failures=failures)
    save(directory / 'summary.json', record)
    print(json.dumps(dict(arm=arm, complete=record['complete'], failures=failures,
                         timings=[dict(shape=[r['m'],r['n'],r['k']], median_ms=r['median_ms']) for r in record['cases']])))
    return 1 if failures else 0


def shape(value):
    try:
        result = tuple(int(part) for part in value.split(','))
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use positive M,N,K integers') from exc
    if len(result) != 3 or min(result) <= 0:
        raise argparse.ArgumentTypeError('Use positive M,N,K integers')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm', choices=['prepare', 'candidate-fast', 'released-identical', 'torch-mps'])
    parser.add_argument('--output', type=Path, required=True, help='Retained inputs and separate arm directories')
    parser.add_argument('--shape', type=shape, action='append', help='Preparation only: repeat M,N,K; defaults to the retained screening shapes')
    parser.add_argument('--preset', choices=['screen', 'full'], default='screen', help='Preparation only: full selects the board 4096-cubed GEMM primitive, seed 7')
    parser.add_argument('--source-commit', default=os.environ.get('MOJOLEARN_BENCH_SOURCE_COMMIT'), help='Exact candidate source commit')
    parser.add_argument('--wheel-manifest', type=Path, help='Optional released wheel provenance, retained by hash')
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1, help='Excluded calls after the separately recorded first call')
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0:
        parser.error('samples must be positive and warmup nonnegative')
    if args.shape and args.arm != 'prepare':
        parser.error('--shape belongs to prepare; all arms consume the retained manifest')
    if args.preset != 'screen' and (args.arm != 'prepare' or args.shape):
        parser.error('--preset full belongs to prepare and cannot be combined with --shape')
    if args.arm == 'candidate-fast' and not args.source_commit:
        parser.error('--source-commit is required for candidate-fast')
    if args.arm == 'prepare':
        shapes = args.shape or (FULL_SHAPES if args.preset == 'full' else SHAPES)
        if len(set(shapes)) != len(shapes):
            parser.error('Preparation shapes must be unique')
        prepare(args.output, shapes, args.preset)
        return 0
    attempt_existed = (args.output / args.arm).exists()
    try:
        return run(args.output, args.arm, source_commit=args.source_commit,
                   wheel_manifest=args.wheel_manifest, samples=args.samples, warmup=args.warmup)
    except Exception:
        # A partial record remains incomplete. Keep the traceback as well as the
        # caller's full redirected log; never turn interrupted work into a pass.
        import traceback
        directory = args.output / args.arm
        if not attempt_existed and directory.exists() and not (directory / 'failure.txt').exists():
            (directory / 'failure.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    raise SystemExit(main())
