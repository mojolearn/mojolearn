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
import signal
import sys
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '2')


def arrays(args):
    import numpy as np
    rng = np.random.default_rng(20260905)
    if args.lane == 'gbdt':
        x = rng.uniform(-1, 1, (1024, 8)).astype('float32')
        y = (2 * x[:, 0] - x[:, 1] + .5 * x[:, 2] * x[:, 3]).astype('float32')
        return x, y
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


def loaded_cuda_libraries():
    """Record actual DSO mappings, including lazily loaded CUDA libraries."""
    prefixes = ('libcuda', 'libcublas', 'libcusolver', 'libcusparse',
                'libcufft', 'libcurand', 'libnvrtc', 'libnvJitLink',
                'libnvfatbin', 'libcudnn', 'libnccl', 'libcuml', 'libcuvs',
                'libraft', 'librmm')
    try:
        paths = set()
        for line in Path('/proc/self/maps').read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and Path(fields[5]).name.startswith(prefixes):
                paths.add(fields[5])
        return sorted(paths)
    except OSError as exc:
        return {'unavailable': repr(exc)}


def worker(args):
    import numpy as np
    if args.worker == 'external' and args.lane == 'umap':
        # Importing the image's older Torch first can preload an incompatible
        # libcublas into this process before the newer RAPIDS wheels load.
        # Keep this worker entirely on its venv's CuPy/CUDA runtime.
        import cupy as cp
        if cp.cuda.runtime.is_hip or cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('Requires NVIDIA CUDA')
        properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        device_name = properties['name']
        if isinstance(device_name, bytes):
            device_name = device_name.decode('utf-8', errors='replace')
        runtime_info = {'device': device_name, 'cupy': cp.__version__,
                        'cuda': cp.cuda.runtime.runtimeGetVersion(),
                        'cuda_driver_version': cp.cuda.runtime.driverGetVersion(),
                        'precision': 'float32 input; cuML backend precision policy',
                        'synchronization': 'cupy.cuda.runtime.deviceSynchronize'}
        synchronize = cp.cuda.runtime.deviceSynchronize
    else:
        import torch
        torch.set_num_threads(2)
        if not torch.cuda.is_available() or torch.version.hip:
            raise RuntimeError('Requires NVIDIA CUDA')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        runtime_info = {'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
                        'cuda': torch.version.cuda, 'precision': 'float32; TF32 disabled in torch'}
        synchronize = torch.cuda.synchronize
    a, b = arrays(args)
    info = {'arm': args.worker, 'lane': args.lane,
            'inputs': [digest(a), digest(b)], 'shapes': [a.shape, b.shape],
            'dtypes': [a.dtype.str, b.dtype.str],
            **runtime_info}
    if args.lane == 'knn':
        info['knn_external'] = args.knn_external
    info['python'] = sys.version
    info['numpy'] = np.__version__
    info['ld_library_path'] = os.environ.get('LD_LIBRARY_PATH', '')
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
        if args.lane == 'gbdt':
            parameters = gbdt_parameters()
            def call():
                model = ml.GradientBoosting(**parameters, numeric_mode=args.worker)
                model.fit(a[:768], b[:768])
                return (model.predict(a[768:]),)
            probe = ml.GradientBoosting(**parameters, numeric_mode=args.worker)
            binding = probe._bind('_mojolearn_gbdt')
            raw = int(binding.gbdt_numeric_mode())
            info['mode'] = {0: 'fast', 1: 'identical', 2: 'deterministic'}.get(raw, 'unknown')
            info['binding_file'] = binding.__file__
            info['parameters'] = parameters
            info['arm_label'] = 'mojolearn-symmetric-rmse-fit-predict-host-request'
        elif args.lane == 'knn':
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
            binding = probe._bind('_mojolearn_metrics')
            raw = int(binding.umap_numeric_mode())
            info['binding_file'] = binding.__file__
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
        if 'binding_file' in info:
            info['binding_sha256'] = hashlib.sha256(Path(info['binding_file']).read_bytes()).hexdigest()
        info['precision'] = ('mojolearn IDENTICAL FP32' if args.worker == 'identical'
                             else 'mojolearn FAST; backend precision policy not inferred')
    elif args.lane == 'gbdt':
        import catboost
        parameters = dict(loss_function='RMSE', iterations=16, depth=4,
                          learning_rate=.1, l2_leaf_reg=3., border_count=32,
                          random_seed=19, random_strength=0., bootstrap_type='No',
                          boosting_type='Plain', grow_policy='SymmetricTree',
                          leaf_estimation_method='Newton', leaf_estimation_iterations=1,
                          boost_from_average=False, score_function='L2',
                          task_type='GPU', devices='0', thread_count=2,
                          gpu_ram_part=.25,
                          allow_writing_files=False, verbose=False)
        def call():
            model = catboost.CatBoostRegressor(**parameters)
            model.fit(a[:768], b[:768])
            return (model.predict(a[768:], task_type='GPU'),)
        info['catboost'] = catboost.__version__
        info['parameters'] = parameters
        info['arm_label'] = 'catboost-cuda-symmetric-rmse-fit-predict-host-request'
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
    info['loaded_cuda_libraries'] = loaded_cuda_libraries()
    print('RESULT ' + json.dumps({'ready': info}), flush=True)
    for line in sys.stdin:
        command = json.loads(line)
        if command['action'] == 'stop':
            break
        synchronize()
        start = time.perf_counter_ns()
        result = tuple(np.ascontiguousarray(x) for x in call())
        synchronize()
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if not all(np.isfinite(x).all() for x in result):
            raise RuntimeError('Nonfinite output')
        target = Path(command['output'])
        np.savez(target, **{'x' + str(i): x for i, x in enumerate(result)})
        print('RESULT ' + json.dumps({'ms': elapsed,
              'hashes': [digest(x) for x in result],
              'loaded_cuda_libraries': loaded_cuda_libraries()}), flush=True)


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


def gbdt_parameters():
    return dict(loss='RMSE', n_estimators=16, max_depth=4,
                learning_rate=.1, l2_leaf_reg=3., border_count=32,
                random_state=19, random_strength=0., bootstrap_type='No',
                grow_policy='SymmetricTree', leaf_estimation_method='Newton',
                leaf_estimation_iterations=1, boost_from_average=False,
                score_function='L2')


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


def worker_environment(arm, lane):
    env = dict(os.environ, MOJOLEARN_NUMERIC_MODE=arm if arm != 'external' else 'fast')
    if arm == 'external' and lane == 'umap':
        # Set the loader path before this isolated worker imports CUDA.
        # The native workers' runtime environment is unchanged.
        site = Path(sys.prefix) / 'lib' / ('python%d.%d' % sys.version_info[:2]) / 'site-packages'
        candidates = sorted(site.glob('nvidia/*/lib'))
        candidates += sorted(site.glob('nvidia/*/lib64'))
        candidates += sorted(site.glob('lib*/lib64'))
        candidates += sorted(site.glob('rapids_logger/lib64'))
        candidates += sorted(site.glob('treelite/lib'))
        # Matching venv CUDA/RAPIDS DSOs take precedence over Pixi fallback.
        candidates += [Path(__file__).resolve().parents[1] / '.pixi/envs/default/lib']
        paths = [str(p) for p in candidates if p.is_dir()]
        if env.get('LD_LIBRARY_PATH'):
            paths.append(env['LD_LIBRARY_PATH'])
        env['LD_LIBRARY_PATH'] = ':'.join(paths)
    return env


def worker_command(args, arm):
    return [sys.executable, str(Path(__file__).resolve()), '--worker', arm,
            '--lane', args.lane, '--dim', str(args.dim), '--rows', str(args.rows),
            '--gram-rows', str(args.gram_rows), '--index', str(args.index),
            '--queries', str(args.queries), '--k', str(args.k),
            '--knn-external', args.knn_external,
            '--umap-rows', str(args.umap_rows), '--umap-epochs', str(args.umap_epochs)]


def probe_external(args):
    """Import/construct external UMAP only; EOF prevents all fit/timed calls."""
    if args.lane != 'umap' or args.worker:
        raise ValueError('--probe-external requires --lane umap and no --worker')
    if not 32 <= args.umap_rows <= 1024:
        raise ValueError('Bounded UMAP probe requires 32..1024 rows')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    record = {'status': 'INCOMPLETE', 'scope': 'external UMAP runtime readiness only; no fit or timings',
              'metadata': {}, 'returncode': None}
    environment = worker_environment('external', args.lane)
    record['ld_library_path'] = environment.get('LD_LIBRARY_PATH', '')

    def capture_metadata():
        path = out / 'probe.stdout.log'
        if path.exists():
            for line in path.read_text().splitlines():
                if line.startswith('RESULT '):
                    try:
                        record['metadata'].update(json.loads(line[7:]))
                    except json.JSONDecodeError:
                        record['metadata_parse_error'] = 'Incomplete RESULT line; raw log retained'

    def interrupted(signum, _frame):
        # Raising through subprocess.run kills/reaps its child; the external
        # root guard can also terminate our shared process group directly.
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        raise InterruptedError('External probe interrupted by signal ' + str(signum))

    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
    for sig in previous_handlers:
        signal.signal(sig, interrupted)
    try:
        with (out / 'probe.stdout.log').open('w') as stdout, (out / 'probe.stderr.log').open('w') as stderr:
            # The child inherits our process group for the outer root guard.
            # An empty stdin means the worker exits immediately after readiness.
            result = subprocess.run(worker_command(args, 'external'),
                                    env=environment,
                                    input='', text=True, stdout=stdout, stderr=stderr,
                                    timeout=90)
        record['returncode'] = result.returncode
        capture_metadata()
        if result.returncode != 0 or 'failed' in record['metadata'] or 'ready' not in record['metadata']:
            raise RuntimeError('External UMAP readiness failed; see probe metadata/logs')
        record['status'] = 'READY'
    except BaseException as exc:
        record.update(status='REFUSED', reason=repr(exc),
                      timed_out=isinstance(exc, subprocess.TimeoutExpired))
        raise
    finally:
        try:
            capture_metadata()
            (out / 'probe.json').write_text(json.dumps(record, indent=2) + '\n')
        finally:
            for sig, previous in previous_handlers.items():
                signal.signal(sig, previous)
    print(json.dumps(record), flush=True)


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
    def interrupted(signum, frame):
        raise InterruptedError('Comparison interrupted by signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    try:
        # Freeze exact request bytes before initializing any worker or timing
        # an arm. Each independent worker must reproduce this saved witness.
        a, b = arrays(args)
        np.savez(out / 'inputs.npz', a=a, b=b)
        archive_digest = hashlib.sha256()
        with (out / 'inputs.npz').open('rb') as archive:
            for chunk in iter(lambda: archive.read(1024 * 1024), b''):
                archive_digest.update(chunk)
        status['inputs'] = {
            'file': 'inputs.npz', 'file_sha256': archive_digest.hexdigest(),
            'keys': ['a', 'b'], 'raw_sha256': [digest(a), digest(b)],
            'shapes': [list(a.shape), list(b.shape)],
            'dtypes': [a.dtype.str, b.dtype.str],
            'hash_order': 'raw C-order bytes, one digest per array',
        }
        del a, b
        (out / 'results.json').write_text(json.dumps(status, indent=2) + '\n')
        for arm in ('fast', 'identical', 'external'):
            env = worker_environment(arm, args.lane)
            cmd = worker_command(args, arm)
            processes[arm] = subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE,
                                              stdout=subprocess.PIPE, text=True)
            startup = receive(processes[arm])
            if 'failed' in startup:
                metadata[arm] = startup['failed']
                raise RuntimeError('Worker initialization failed: ' + repr(startup['failed']))
            metadata[arm] = startup['ready']
        if any(v['inputs'] != status['inputs']['raw_sha256']
               or v['shapes'] != status['inputs']['shapes']
               or v['dtypes'] != status['inputs']['dtypes'] for v in metadata.values()):
            raise RuntimeError('Worker inputs differ from the retained input archive')
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
                if 'failed' in row:
                    raise RuntimeError('Worker execution failed: ' + repr(row['failed']))
        hashes = {tuple(x['hashes']) for x in records if x['arm'] == 'identical'}
        if len(hashes) != 1:
            raise RuntimeError('IDENTICAL changed bytes between rounds')
        accuracy = []
        if args.lane == 'gbdt':
            _, y = arrays(args)
            baseline = float(np.mean((y[768:] - y[:768].mean()) ** 2) ** .5)
            for r in range(args.rounds + 1):
                errors = {}
                for arm in ('fast', 'identical', 'external'):
                    with np.load(out / f'{arm}-{r}.npz') as actual:
                        prediction = actual['x0']
                        if prediction.shape != y[768:].shape:
                            raise RuntimeError('GBDT prediction shape mismatch')
                        errors[arm] = float(np.mean((prediction.astype('float64') - y[768:]) ** 2) ** .5)
                for arm, rmse in errors.items():
                    passed = rmse < .8 * baseline
                    if arm != 'external':
                        passed = passed and rmse <= errors['external'] * 1.10
                    accuracy.append(dict(arm=arm, round=r, rmse=rmse, passed=bool(passed)))
            status['quality_policy'] = {
                'held_out_rows': 256, 'baseline_rmse': baseline,
                'maximum_baseline_fraction': .8, 'maximum_external_rmse_ratio': 1.10,
                'scope': 'numeric plain symmetric RMSE fit plus GPU prediction; independent binning',
                'external_bitwise_equality': False}
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
            if args.lane in ('umap', 'gbdt'):
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
        # Preserve partial evidence before cleanup, including a guard signal
        # that may be followed by SIGKILL if a worker does not exit promptly.
        (out / 'results.json').write_text(json.dumps(status, indent=2) + '\n')
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
    parser.add_argument('--probe-external', action='store_true',
                        help='UMAP only: check external runtime readiness with no fit or timing')
    parser.add_argument('--lane', choices=['knn', 'gemv', 'nt', 'gram', 'umap', 'gbdt'], required=True)
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
    if arguments.probe_external:
        probe_external(arguments)
    elif arguments.worker == 'external' and arguments.lane == 'umap':
        try:
            worker(arguments)
        except Exception as exc:
            # Retain the loader's actual choices even when importing cuML
            # fails before the ordinary readiness metadata can be published.
            print('RESULT ' + json.dumps({'failed': {
                'arm': arguments.worker, 'lane': arguments.lane,
                'reason': repr(exc),
                'ld_library_path': os.environ.get('LD_LIBRARY_PATH', ''),
                'loaded_cuda_libraries': loaded_cuda_libraries(),
            }}), flush=True)
            raise
    else:
        worker(arguments) if arguments.worker else main(arguments)
