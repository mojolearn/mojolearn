#!/usr/bin/env python3
"""Root-only NVIDIA digits quality qualification, never a speed comparison.

Requires NumPy, scikit-learn, umap-learn and source-compatible MojoLearn
FAST/IDENTICAL metrics extensions. Run remotely under nvidia_serial_guard.py.
The CPU umap-learn reference and the two NVIDIA modes run sequentially in
separate processes. Fixed bounds: 1024 training rows, 256 held-out rows,
64 features, 100 epochs, two output dimensions, one CPU thread per worker.
FAST training therefore crosses the public GPU optimizer's 1024-row boundary.
No wall-clock measurements or CPU/GPU speed ratios are emitted.

Thresholds below are declared before execution. A failure must be retained
and diagnosed, never repaired by tuning the threshold against these results.
Dataset bytes, split indices, binaries, and raw embeddings are preserved.
Use --expected-data-sha256 from an approved capture to enforce a dataset pin.
An unpinned successful quality capture returns CAPTURED_UNPINNED, not PASS.
IDENTICAL additionally repeats fit and transform with a fresh estimator and
requires raw byte equality for both arrays. The native vendor must be CUDA.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'MOJOLEARN_CPU_THREADS'):
    os.environ[_key] = '1'

PARAMETERS = dict(n_neighbors=15, n_components=2, n_epochs=100,
                  metric='euclidean', init='spectral', random_state=19,
                  min_dist=0.1, spread=1.0, set_op_mix_ratio=1.0,
                  local_connectivity=1.0, learning_rate=1.0,
                  repulsion_strength=1.0, negative_sample_rate=5)
POLICY = dict(k=10, minimum_trustworthiness=0.85,
              maximum_trustworthiness_loss_vs_reference=0.08,
              minimum_retention=0.15, minimum_control_trustworthiness_margin=0.15)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_witness(array):
    import numpy as np
    value = np.ascontiguousarray(array)
    return dict(shape=list(value.shape), dtype=value.dtype.str,
                raw_c_order_sha256=hashlib.sha256(value.tobytes()).hexdigest())


def write_json(path, record):
    Path(path).write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')


def nvidia_witness():
    if sys.platform != 'linux':
        raise RuntimeError('This quality runner requires a remote Linux NVIDIA host')
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=uuid,name,driver_version',
         '--format=csv,noheader'], check=True, capture_output=True, text=True,
        timeout=10)
    rows = result.stdout.strip().splitlines()
    if len(rows) != 1:
        raise RuntimeError('Requires exactly one NVIDIA GPU')
    return rows[0]


def prepare_dataset(out, expected):
    import numpy as np
    import sklearn
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    data = load_digits()
    # Canonical unscaled values and labels define the dataset pin. Pixels are
    # exact integers; division by 16 is exact float32 scaling, with no fitting
    # on held-out data. Labels only stratify the split; UMAP never receives y.
    raw = np.asarray(data.data, dtype='<f4', order='C')
    labels = np.asarray(data.target, dtype='<i8', order='C')
    digest = hashlib.sha256(raw.tobytes() + labels.tobytes()).hexdigest()
    if expected is not None and digest != expected:
        raise RuntimeError('Digits dataset SHA256 differs from required pin')
    ids = np.arange(len(raw), dtype=np.int64)
    train, query = train_test_split(ids, train_size=1024, test_size=256,
                                    random_state=19, stratify=labels)
    x = np.asarray(raw[train] / np.float32(16), dtype=np.float32, order='C')
    q = np.asarray(raw[query] / np.float32(16), dtype=np.float32, order='C')
    np.savez(out / 'inputs.npz', train=x, query=q, train_indices=train,
             query_indices=query, train_labels=labels[train], query_labels=labels[query])
    return dict(name='sklearn.datasets.load_digits', sklearn=sklearn.__version__,
                data_sha256=digest, expected_data_sha256=expected,
                pin_format='C-order little-endian float32 pixels followed by int64 labels',
                pin_verified=expected is not None,
                scaling='divide pixels by 16; no fitted preprocessing',
                split='sklearn stratified train_test_split seed 19; 1024/256 disjoint rows',
                labels_used_for_embedding=False, train=array_witness(x),
                query=array_witness(q), train_indices=array_witness(train),
                query_indices=array_witness(query), inputs_file_sha256=sha(out / 'inputs.npz'))


def worker(args):
    import numpy as np
    out = args.out.resolve()
    with np.load(out / 'inputs.npz') as inputs:
        train, query = inputs['train'], inputs['query']
    if train.shape != (1024, 64) or query.shape != (256, 64):
        raise ValueError('Refusing a dataset outside the fixed workload bounds')
    before = (array_witness(train), array_witness(query))
    record = dict(arm=args.worker, device=nvidia_witness(), parameters=PARAMETERS,
                  numpy=np.__version__, python=sys.version, status='INITIALIZING',
                  train=before[0], query=before[1])
    write_json(out / (args.worker + '-metadata.json'), record)
    if args.worker == 'reference':
        import umap
        model = umap.UMAP(**PARAMETERS, n_jobs=1, transform_seed=19,
                          force_approximation_algorithm=False)
        record.update(implementation='umap-learn CPU quality reference',
                      version=umap.__version__, module_file=umap.__file__,
                      transform_seed=19, n_jobs=1,
                      force_approximation_algorithm=False)
    else:
        import mojolearn
        from mojolearn import _backend
        model = mojolearn.UMAP(**PARAMETERS, numeric_mode=args.worker)
        binding = model._bind('_mojolearn_metrics')
        witness = int(binding.umap_numeric_mode())
        vendor = _backend.read_vendor(binding)
        record.update(implementation='mojolearn NVIDIA', version=mojolearn.__version__,
                      module_file=mojolearn.__file__, binding_file=binding.__file__,
                      binding_sha256=sha(binding.__file__), numeric_mode_witness=witness,
                      native_vendor=vendor)
        write_json(out / (args.worker + '-metadata.json'), record)
        if vendor != 'cuda':
            raise RuntimeError('UMAP native vendor witness must be cuda')
        if witness != {'fast': 0, 'identical': 1}[args.worker]:
            raise RuntimeError('UMAP compiled mode witness mismatch')
    implementation_source = inspect.getfile(model.__class__)
    record.update(estimator_source_file=implementation_source,
                  estimator_source_sha256=sha(implementation_source), status='FITTING')
    write_json(out / (args.worker + '-metadata.json'), record)
    fitted = np.ascontiguousarray(model.fit_transform(train))
    frozen = fitted.copy()
    heldout = np.ascontiguousarray(model.transform(query))
    if (fitted.shape != (1024, 2) or heldout.shape != (256, 2)
            or not np.isfinite(fitted).all() or not np.isfinite(heldout).all()):
        raise RuntimeError('Invalid fit or held-out transform embedding')
    if before != (array_witness(train), array_witness(query)):
        raise RuntimeError('UMAP changed its input data')
    if array_witness(frozen) != array_witness(model.embedding_):
        raise RuntimeError('Transform changed the fitted training embedding')
    np.savez(out / (args.worker + '-embeddings.npz'), fit=fitted, transform=heldout)
    record.update(fit=array_witness(fitted), transform=array_witness(heldout),
                  artifact_sha256=sha(out / (args.worker + '-embeddings.npz')))
    write_json(out / (args.worker + '-metadata.json'), record)
    if args.worker == 'identical':
        record['status'] = 'REPEATING_WITH_FRESH_ESTIMATOR'
        write_json(out / (args.worker + '-metadata.json'), record)
        # Independent estimator and independent inputs. The first fit is
        # already on disk; never reset or mutate its private fitted state.
        repeated_model = mojolearn.UMAP(**PARAMETERS, numeric_mode='identical')
        repeated_train, repeated_query = train.copy(), query.copy()
        repeated_fit = np.ascontiguousarray(repeated_model.fit_transform(repeated_train))
        repeated_frozen = repeated_fit.copy()
        repeated_transform = np.ascontiguousarray(repeated_model.transform(repeated_query))
        np.savez(out / 'identical-repeat-embeddings.npz',
                 fit=repeated_fit, transform=repeated_transform)
        with np.load(out / 'identical-embeddings.npz') as original:
            fit_equal = (array_witness(original['fit']) == array_witness(repeated_fit)
                         and original['fit'].tobytes() == repeated_fit.tobytes())
            transform_equal = (
                array_witness(original['transform']) == array_witness(repeated_transform)
                and original['transform'].tobytes() == repeated_transform.tobytes())
        state_unchanged = array_witness(repeated_frozen) == array_witness(repeated_model.embedding_)
        inputs_unchanged = before == (array_witness(repeated_train), array_witness(repeated_query))
        record['repeatability'] = dict(
            scope='same NVIDIA host, independent fresh estimators; no cross-vendor claim',
            fit=array_witness(repeated_fit), transform=array_witness(repeated_transform),
            artifact_sha256=sha(out / 'identical-repeat-embeddings.npz'),
            fit_raw_bytes_equal=fit_equal, transform_raw_bytes_equal=transform_equal,
            frozen_model_unchanged=state_unchanged, inputs_unchanged=inputs_unchanged)
        record['status'] = 'PASS' if all((fit_equal, transform_equal, state_unchanged,
                                         inputs_unchanged)) else 'FAIL'
        write_json(out / (args.worker + '-metadata.json'), record)
        if record['status'] != 'PASS':
            raise RuntimeError('IDENTICAL fresh-estimator repeatability failed')
    record['status'] = 'PASS'
    write_json(out / (args.worker + '-metadata.json'), record)


def order(query, train, self_excluding):
    import numpy as np
    from sklearn.metrics import pairwise_distances
    # Bounded 1024x1024 distance/rank matrices; avoid n*n*64 broadcasting.
    distances = pairwise_distances(np.asarray(query, dtype=np.float64),
                                  np.asarray(train, dtype=np.float64),
                                  metric='euclidean', n_jobs=1)
    if self_excluding:
        np.fill_diagonal(distances, np.inf)
    return np.argsort(distances, axis=1, kind='stable')


def score(original_order, embedded_order, self_excluding):
    import numpy as np
    rows, n = original_order.shape
    k = POLICY['k']
    ranks = np.empty_like(original_order)
    np.put_along_axis(ranks, original_order, np.arange(1, n + 1)[None, :], axis=1)
    selected = np.take_along_axis(ranks, embedded_order[:, :k], axis=1)
    penalty = int(np.maximum(selected - k, 0).sum())
    # Fit excludes each point itself; held-out rows have n candidate anchors.
    correction = -1 if self_excluding else 1
    trust = 1 - 2 * penalty / (rows * k * (2 * n - 3 * k + correction))
    return dict(trustworthiness=float(trust), retention=float((selected <= k).mean()))


def run(args):
    import numpy as np
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists():
        raise RuntimeError('Use a new output directory; existing evidence is preserved')
    record = dict(schema='umap.digits.real-data-quality.v1', status='RUNNING',
                  scope='fit and held-out quality only; no speed or cross-library bit parity claim',
                  harness_sha256=sha(__file__), policy=POLICY, arms={}, quality={},
                  metrics={'fit': 'self-excluding Euclidean rank trustworthiness and top-k retention',
                           'transform': 'held-out query-to-frozen-training rank trustworthiness and top-k retention',
                           'ties': 'stable ascending training-row index',
                           'control': 'rotate training embedding rows by 512, preserving geometry'},
                  threads={key: os.environ[key] for key in
                           ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                            'NUMBA_NUM_THREADS', 'MOJOLEARN_CPU_THREADS')})
    def interrupted(signum, _frame):
        # subprocess.run catches BaseException, kills and reaps its child.
        # Ignore repeated termination during cleanup/artifact preservation.
        for termination in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(termination, signal.SIG_IGN)
        raise RuntimeError('Quality runner interrupted by signal ' + str(signum))

    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
    for sig in previous_handlers:
        signal.signal(sig, interrupted)
    try:
        record['device'] = nvidia_witness()
        record['dataset'] = prepare_dataset(out, args.expected_data_sha256)
        write_json(out / 'results.json', record)
        for arm in ('reference', 'fast', 'identical'):
            env = dict(os.environ, MOJOLEARN_NUMERIC_MODE='fast' if arm == 'reference' else arm)
            # Children retain this process group so the external root guard
            # accounts for all memory and can terminate the whole workload.
            with (out / (arm + '.log')).open('w') as log:
                try:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                    '--worker', arm, '--out', str(out)], env=env,
                                   stdout=log, stderr=subprocess.STDOUT, check=True,
                                   timeout=600)
                except BaseException as exc:
                    record['failed_arm'] = dict(
                        arm=arm, reason=repr(exc), exit_code=getattr(exc, 'returncode', None),
                        timed_out=isinstance(exc, subprocess.TimeoutExpired))
                    raise
                finally:
                    metadata_path = out / (arm + '-metadata.json')
                    if metadata_path.exists():
                        record['arms'][arm] = json.loads(metadata_path.read_text())
            write_json(out / 'results.json', record)
        with np.load(out / 'inputs.npz') as inputs:
            train, query = inputs['train'], inputs['query']
        originals = {'fit': order(train, train, True),
                     'transform': order(query, train, False)}
        for arm in ('reference', 'fast', 'identical'):
            with np.load(out / (arm + '-embeddings.npz')) as embeddings:
                fitted, heldout = embeddings['fit'], embeddings['transform']
            scores = {}
            # Rotate training correspondences by one half. All pairwise
            # geometry is retained while input-to-layout identities are broken.
            scrambled = np.roll(fitted, 512, axis=0)
            for stage, values, control_values in (
                ('fit', fitted, scrambled), ('transform', heldout, heldout)):
                is_fit = stage == 'fit'
                actual = score(originals[stage], order(values, fitted, is_fit), is_fit)
                control = score(originals[stage], order(control_values, scrambled, is_fit), is_fit)
                reference = (actual if arm == 'reference' else
                             record['quality']['reference'][stage])
                actual['control_trustworthiness'] = control['trustworthiness']
                actual['passed'] = bool(
                    actual['trustworthiness'] >= POLICY['minimum_trustworthiness']
                    and actual['retention'] >= POLICY['minimum_retention']
                    and actual['trustworthiness'] >= reference['trustworthiness']
                    - POLICY['maximum_trustworthiness_loss_vs_reference']
                    and actual['trustworthiness'] - control['trustworthiness']
                    >= POLICY['minimum_control_trustworthiness_margin'])
                scores[stage] = actual
            record['quality'][arm] = scores
        quality_passed = all(row['passed'] for stages in record['quality'].values()
                             for row in stages.values())
        record['quality_passed'] = quality_passed
        record['status'] = ('FAIL' if not quality_passed else
                            'PASS' if record['dataset']['pin_verified'] else 'CAPTURED_UNPINNED')
    except BaseException as exc:
        record.update(status='REFUSED', reason=repr(exc))
        raise
    finally:
        try:
            write_json(out / 'results.json', record)
        finally:
            for sig, previous in previous_handlers.items():
                signal.signal(sig, previous)
    print(json.dumps(dict(status=record['status'], quality=record['quality'])), flush=True)
    return 0 if record['status'] == 'PASS' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expected-data-sha256')
    parser.add_argument('--capture-data-only', action='store_true',
                        help='Freeze dataset bytes before any model execution')
    parser.add_argument('--worker', choices=('reference', 'fast', 'identical'),
                        help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.capture_data_only:
        nvidia_witness()
        options.out.mkdir(parents=True, exist_ok=False)
        dataset = prepare_dataset(options.out, options.expected_data_sha256)
        write_json(options.out / 'dataset.json', dataset)
        print(dataset['data_sha256'], flush=True)
    elif options.worker:
        worker(options)
    else:
        raise SystemExit(run(options))
