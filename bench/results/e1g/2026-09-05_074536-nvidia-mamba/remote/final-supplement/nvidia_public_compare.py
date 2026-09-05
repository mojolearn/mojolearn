#!/usr/bin/env python3
"""Bounded host-array API comparison; NOT the old kernel-only identity-cost fixture.

Run with installed CUDA mojolearn fast/identical extensions, numpy and torch.
cuML comparisons additionally require cupy/cuML; --knn-external torch uses
CUDA FP32 cdist + topk instead, explicitly labeled separately from cuML.
No builds or rentals are performed.
Only the main lane should execute this harness. Each lane has three idle
workers, but exactly one receives a timed command at a time.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '2')


def arrays(args):
    import numpy as np
    rng = np.random.default_rng(20260905)
    if args.lane == 'umap':
        # A fixed continuous two-dimensional manifold in 16 dimensions.
        # No label supervision or external dataset download is involved.
        t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, args.umap_rows)
        z = rng.uniform(-8, 8, args.umap_rows)
        latent = np.column_stack((t * np.cos(t), z, t * np.sin(t)))
        projection = rng.normal(size=(3, 16)) / 4
        x = (latent @ projection).astype('float32')
        return x, np.empty((0,), dtype='float32')
    if args.lane == 'knn':
        return (rng.uniform(-1, 1, (args.index, 32)).astype('float32'),
                rng.uniform(-1, 1, (args.queries, 32)).astype('float32'))
    shapes = {'gemv': ((args.dim, args.dim), (1, args.dim)),
              'nt': ((args.rows, 64), (64, 64)),
              'gram': ((args.gram_rows, 32), (args.gram_rows, 32))}
    sa, sb = shapes[args.lane]
    a = rng.uniform(-1, 1, sa).astype('float32')
    b = a.copy() if args.lane == 'gram' else rng.uniform(-1, 1, sb).astype('float32')
    return a, b


def digest(a):
    return hashlib.sha256(a.tobytes(order='C')).hexdigest()


def worker(args):
    import numpy as np
    import torch
    torch.set_num_threads(2)
    if not torch.cuda.is_available() or torch.version.hip:
        raise RuntimeError('Requires NVIDIA CUDA')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    a, b = arrays(args)
    info = {'arm': args.worker, 'lane': args.lane,
            'inputs': [digest(a), digest(b)], 'shapes': [a.shape, b.shape],
            'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'precision': 'float32; TF32 disabled in torch'}
    if args.lane == 'knn':
        info['knn_external'] = args.knn_external
    info['python'] = sys.version
    info['numpy'] = np.__version__
    info['source_commit'] = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], text=True, capture_output=True,
        cwd=Path(__file__).resolve().parents[1]).stdout.strip()
    if not info['source_commit']:
        marker = Path(__file__).resolve().parents[1] / 'commit.txt'
        info['source_commit'] = os.environ.get('MOJOLEARN_COMMIT') or (
            marker.read_text().strip() if marker.exists() else 'unknown')
    driver = subprocess.run(['nvidia-smi', '--query-gpu=driver_version,name',
                             '--format=csv,noheader'], text=True, capture_output=True)
    info['nvidia_smi'] = driver.stdout.strip()
    if args.worker in ('fast', 'identical'):
        import mojolearn as ml
        info['mojolearn'] = ml.__version__
        info['package'] = ml.__file__
        if args.lane == 'knn':
            model = ml.NearestNeighbors(n_neighbors=args.k)
            model.fit(a)
            call = lambda: model.kneighbors(b)
            binding = model._bind()
            if not hasattr(binding, 'mojolearn_numeric_mode'):
                raise RuntimeError('kNN binary lacks compiled mode witness; rebuild core binding')
            raw = int(binding.mojolearn_numeric_mode())
            info['mode'] = {0: 'fast', 1: 'identical', 2: 'deterministic'}.get(raw, 'unknown')
            info['binding_file'] = binding.__file__
        elif args.lane == 'umap':
            def call():
                model = ml.UMAP(**umap_parameters(args), numeric_mode=args.worker)
                return (model.fit_transform(a),)
            # Read this surface's compiled witness, not another extension.
            probe = ml.UMAP(**umap_parameters(args), numeric_mode=args.worker)
            raw = int(probe._bind('_mojolearn_metrics').umap_numeric_mode())
            # UMAP binding uses its own explicit fast=0/identical=1/deterministic=2 map.
            info['mode'] = {0: 'fast', 1: 'identical', 2: 'deterministic'}.get(raw, 'unknown')
            info['parameters'] = umap_parameters(args)
            info['arm_label'] = 'mojolearn-umap-fit-transform-host-request'
        else:
            info['mode'] = str(ml.linalg.numeric_mode())
            call = lambda: (ml.linalg.matmul(
                a, b, transpose_a=args.lane == 'gram',
                transpose_b=args.lane != 'gram',
                identical=args.worker == 'identical'),)
        if info['mode'].lower() != args.worker:
            raise RuntimeError('Compiled mode witness mismatch: ' + repr(info))
        info['precision'] = ('mojolearn IDENTICAL FP32' if args.worker == 'identical'
                             else 'mojolearn FAST; backend precision policy not inferred')
    elif args.lane == 'umap':
        import cupy as cp
        import cuml
        from cuml.manifold import UMAP
        parameters = dict(umap_parameters(args), output_type='numpy',
                          build_algo='brute_force_knn', force_serial_epochs=False)
        # An unsupported explicit build_algo is a refusal, never a silent
        # switch to an approximate graph at a different workload.
        probe = UMAP(**parameters)
        def call():
            model = UMAP(**parameters)
            result = np.asarray(model.fit_transform(a))
            cp.cuda.runtime.deviceSynchronize()
            return (result,)
        info['cuml'] = cuml.__version__
        info['parameters'] = parameters
        info['arm_label'] = 'cuml-umap-fit-transform-host-request'
        info['repeatability'] = 'cuML parallel epochs; no cross-round bitwise promise'
    elif args.lane == 'knn' and args.knn_external == 'torch':
        # Exact exhaustive search, not an approximate index. Re-upload BOTH
        # host arrays and return both outputs each request, as in the cuML
        # arm below. The dense query-by-index distance workspace is explicit.
        if args.queries * args.index > 100_000_000:
            raise ValueError('torch kNN distance workspace is capped at 100M float32 cells')
        def call():
            index = torch.from_numpy(a).to('cuda')
            query = torch.from_numpy(b).to('cuda')
            distances = torch.cdist(query, index, p=2,
                                   compute_mode='use_mm_for_euclid_dist')
            values, indices = torch.topk(distances, args.k, dim=1,
                                         largest=False, sorted=True)
            return values.cpu().numpy(), indices.cpu().numpy()
        info['arm_label'] = 'torch-cuda-cdist-topk-host-request'
        info['precision'] = 'torch CUDA FP32 cdist matrix-product path; TF32 disabled'
        info['cdist_compute_mode'] = 'use_mm_for_euclid_dist'
        info['topk_sorted'] = True
        info['distance_workspace_bytes'] = 4 * args.queries * args.index
        info['transfer_scope'] = 'host index + query uploaded and distances + indices downloaded each request'
    elif args.lane == 'knn':
        import cupy as cp
        import cuml
        from cuml.neighbors import NearestNeighbors
        model = NearestNeighbors(n_neighbors=args.k, algorithm='brute',
                                 metric='euclidean', output_type='cupy')
        # Mojo stores a host index and uploads it per call. Include cuML's
        # fit/index upload each call too, to compare host-input requests.
        def call():
            model.fit(cp.asarray(a))
            d, i = model.kneighbors(cp.asarray(b))
            return cp.asnumpy(d), cp.asnumpy(i)
        info['cuml'] = cuml.__version__
        info['arm_label'] = 'cuml-brute-host-request'
    else:
        def call():
            ta = torch.from_numpy(a).to('cuda')
            tb = torch.from_numpy(b).to('cuda')
            value = ta.T @ tb if args.lane == 'gram' else ta @ tb.T
            return (value.cpu().numpy(),)
        info['arm_label'] = 'torch-cublas-fp32-host-request'
    print('RESULT ' + json.dumps({'ready': info}), flush=True)
    for line in sys.stdin:
        command = json.loads(line)
        if command['action'] == 'stop':
            break
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        result = tuple(np.ascontiguousarray(x) for x in call())
        torch.cuda.synchronize()
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if not all(np.isfinite(x).all() for x in result):
            raise RuntimeError('Nonfinite output')
        target = Path(command['output'])
        np.savez(target, **{'x' + str(i): x for i, x in enumerate(result)})
        print('RESULT ' + json.dumps({'ms': elapsed,
              'hashes': [digest(x) for x in result]}), flush=True)


def receive(proc):
    # Native runtime diagnostics may also use stdout; retain them visibly.
    for line in proc.stdout:
        if line.startswith('RESULT '):
            return json.loads(line[7:])
        print(line.rstrip(), file=sys.stderr)
    raise RuntimeError('Worker exited: ' + str(proc.wait()))


def umap_parameters(args):
    return dict(n_neighbors=15, n_components=2, metric='euclidean',
                init='spectral', random_state=19, n_epochs=args.umap_epochs,
                min_dist=0.1, spread=1.0)


def neighborhood_quality(x, embedding, k=10):
    """Trustworthiness and k-neighbor retention; CPU work outside timing.

    Exact distances with stable index tie breaks. Small fixtures only:
    two n-by-n distance/rank matrices, never a production-scale score.
    """
    import numpy as np
    n = len(x)
    if embedding.shape != (n, 2) or not np.isfinite(embedding).all():
        raise ValueError('Invalid UMAP embedding')
    def order(v):
        v = np.asarray(v, dtype='float64')
        d = ((v[:, None, :] - v[None, :, :]) ** 2).sum(axis=2)
        np.fill_diagonal(d, np.inf)
        return np.argsort(d, axis=1, kind='stable')
    original, reduced = order(x), order(embedding)
    ranks = np.empty((n, n), dtype=np.int32)
    ranks[np.arange(n)[:, None], original] = np.arange(1, n + 1)
    neighbor_ranks = ranks[np.arange(n)[:, None], reduced[:, :k]]
    penalty = np.maximum(neighbor_ranks - k, 0).sum(dtype=np.int64)
    trust = 1.0 - 2.0 * float(penalty) / (n * k * (2 * n - 3 * k - 1))
    retention = float((neighbor_ranks <= k).mean())
    return {'trustworthiness': trust, 'neighbor_retention': retention, 'k': k}


def main(args):
    import numpy as np
    if args.rounds < 7:
        raise ValueError('At least seven timed rounds required')
    if args.lane == 'umap' and not 32 <= args.umap_rows <= 1024:
        raise ValueError('Bounded UMAP quality harness requires 32..1024 rows')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    processes = {}
    records = []
    metadata = {}
    status = {'status': 'INCOMPLETE', 'scope': 'host-array public APIs, including transfers and outputs; not historical kernel-only fixtures',
              'args': vars(args), 'metadata': metadata, 'records': records}
    try:
        for arm in ('fast', 'identical', 'external'):
            env = dict(os.environ, MOJOLEARN_NUMERIC_MODE=arm if arm != 'external' else 'fast')
            cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', arm,
                   '--lane', args.lane, '--dim', str(args.dim), '--rows', str(args.rows),
                   '--gram-rows', str(args.gram_rows), '--index', str(args.index),
                   '--queries', str(args.queries), '--k', str(args.k),
                   '--knn-external', args.knn_external,
                   '--umap-rows', str(args.umap_rows), '--umap-epochs', str(args.umap_epochs)]
            processes[arm] = subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE,
                                              stdout=subprocess.PIPE, text=True)
            metadata[arm] = receive(processes[arm])['ready']
        if len({tuple(v['inputs']) for v in metadata.values()}) != 1:
            raise RuntimeError('Input witness mismatch')
        for r in range(args.rounds + 1):
            # Rotate order to reduce systematic position bias.
            arms = ['fast', 'identical', 'external']
            arms = arms[r % 3:] + arms[:r % 3]
            for arm in arms:
                p = processes[arm]
                p.stdin.write(json.dumps({'action': 'run', 'output': str(out / f'{arm}-{r}.npz')}) + '\n')
                p.stdin.flush()
                row = dict(receive(p), arm=arm, round=r, warmup=r == 0)
                records.append(row)
                print(json.dumps(row), flush=True)
        hashes = {tuple(x['hashes']) for x in records if x['arm'] == 'identical'}
        if len(hashes) != 1:
            raise RuntimeError('IDENTICAL changed bytes between rounds')
        accuracy = []
        if args.lane == 'umap':
            x, _ = arrays(args)
            for r in range(args.rounds + 1):
                scores = {}
                for arm in ('fast', 'identical', 'external'):
                    with np.load(out / f'{arm}-{r}.npz') as actual:
                        scores[arm] = neighborhood_quality(x, actual['x0'])
                for arm, quality in scores.items():
                    passed = quality['trustworthiness'] >= args.quality_min
                    if arm != 'external':
                        passed = passed and quality['trustworthiness'] >= (
                            scores['external']['trustworthiness'] - args.quality_gap)
                    accuracy.append(dict(quality, arm=arm, round=r, passed=bool(passed)))
            status['quality_policy'] = {
                'minimum_trustworthiness': args.quality_min,
                'maximum_trustworthiness_loss_vs_cuml': args.quality_gap,
                'coordinate_equality': 'not required across different implementations',
                'retention': 'reported separately; no equality required'}
        for r in range(args.rounds + 1):
            if args.lane == 'umap':
                break
            with np.load(out / f'external-{r}.npz') as reference:
                for arm in ('fast', 'identical'):
                    with np.load(out / f'{arm}-{r}.npz') as actual:
                        av, rv = actual['x0'].astype('float64'), reference['x0'].astype('float64')
                        rel = float(np.linalg.norm(av - rv) / max(np.linalg.norm(rv), 1e-30))
                        maxerr = float(np.max(np.abs(av - rv)))
                        # FP32 library accumulation orders differ. Thresholds
                        # are admission bounds, not a numerical contract.
                        passed = rel <= args.rtol
                        if args.lane == 'knn':
                            passed = passed and np.array_equal(actual['x1'], reference['x1'])
                        accuracy.append({'round': r, 'arm': arm, 'relative_l2': rel,
                                         'max_abs': maxerr, 'passed': bool(passed)})
        status['accuracy'] = accuracy
        if not all(row['passed'] for row in accuracy):
            raise RuntimeError('Accuracy admission failed; no competitive conclusion')
        values = {arm: np.array([row['ms'] for row in records if row['arm'] == arm and not row['warmup']])
                  for arm in processes}
        status['summary'] = {arm: {'median_ms': float(np.median(v)),
                                    'iqr_ms': float(np.quantile(v, .75) - np.quantile(v, .25))}
                             for arm, v in values.items()}
        status['ratios'] = {}
        for num, den in [('identical', 'fast'), ('identical', 'external'), ('fast', 'external')]:
            ratios = values[num] / values[den]
            status['ratios'][num + '/' + den] = {
                'median_paired': float(np.median(ratios)),
                'min_paired': float(ratios.min()), 'max_paired': float(ratios.max())}
            status['ratios'][num + '/' + den]['separated_from_one'] = bool(
                ratios.min() > 1 or ratios.max() < 1)
        status['status'] = 'PASSED'
    except Exception as exc:
        status['status'] = 'REFUSED'
        status['reason'] = repr(exc)
        raise
    finally:
        for p in processes.values():
            if p.poll() is None:
                try:
                    p.stdin.write('{"action":"stop"}\n')
                    p.stdin.flush()
                    p.wait(timeout=15)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    p.kill()
                    p.wait()
        (out / 'results.json').write_text(json.dumps(status, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['fast', 'identical', 'external'])
    parser.add_argument('--lane', choices=['knn', 'gemv', 'nt', 'gram', 'umap'], required=True)
    parser.add_argument('--out', default='bench/results/nvidia-public-comparison')
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--index', type=int, default=10000)
    parser.add_argument('--queries', type=int, default=128)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--knn-external', choices=['cuml', 'torch'], default='cuml')
    parser.add_argument('--dim', type=int, default=2048)
    parser.add_argument('--rows', type=int, default=16384)
    parser.add_argument('--gram-rows', type=int, default=65536)
    parser.add_argument('--rtol', type=float, default=5e-4)
    parser.add_argument('--umap-rows', type=int, default=256)
    parser.add_argument('--umap-epochs', type=int, default=50)
    parser.add_argument('--quality-min', type=float, default=.85)
    parser.add_argument('--quality-gap', type=float, default=.05)
    arguments = parser.parse_args()
    worker(arguments) if arguments.worker else main(arguments)
