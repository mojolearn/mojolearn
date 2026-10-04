#!/usr/bin/env python3
"""Pinned actual PCA.fit quality and separately admitted one-call timing.

COMPILED_SOURCE TAG quality|timing DATA_NPZ DATA_SHA256
Timing additionally requires --quality-report PATH --quality-sha SHA256.
Only the public covariance_eigh fit on full, unscaled big-istella X is covered.
No builds, queue edits, opponents, warmups, repeats or default promotion.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

if not __debug__:
    raise RuntimeError('quality assertions must be enabled')
ROOT = Path(__file__).resolve().parents[1]
BINDING = 'estimators'
MODULE = '_mojolearn_estimators'
CONTRACT = 'scoped-pca-public-fit-istella-v1'
A_FLAGS = '-D MOJOLEARN_SCOPED_GEMM_AUDIT'
B_FLAGS = A_FLAGS + ' -D MOJOLEARN_SCOPED_GEMM_G1_GRAM -D MOJOLEARN_SCOPED_GEMM_SPLIT -D MOJOLEARN_SCOPED_GEMM_PCA'
BOUND = 5e-6
HARNESS_ONLY = {'tools/scoped_pca_fit.py', 'tools/scoped_pca_fit_spec.py',
                'tools/test_scoped_pca_fit_contract.py', 'tools/apple_fast_job_policy.py',
                'docs/apple-fast/ab/scoped-pca-fit.md'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def record(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def source_contract(compiled):
    assert re.fullmatch('[0-9a-f]{40}', compiled)
    harness = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    subprocess.run(['git', 'merge-base', '--is-ancestor', compiled, harness], check=True)
    paths = set(subprocess.check_output(['git', 'diff', '--name-only', compiled, harness], text=True).splitlines())
    assert paths <= HARNESS_ONLY, ('native/production/config drift', sorted(paths - HARNESS_ONLY))
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'], check=True)
    return harness


def install(source, destination):
    fd, name = tempfile.mkstemp(prefix='.scoped-pca-', suffix='.so', dir=destination.parent)
    os.close(fd)
    try:
        shutil.copy2(source, name)
        os.replace(name, destination)
    finally:
        Path(name).unlink(missing_ok=True)


def native(binary_sha, mask):
    from mojolearn import _backend
    module = _backend.binding(MODULE, 'fast')
    path = Path(module.__file__).resolve()
    assert path == (ROOT / 'python/mojolearn' / (MODULE + '.so')).resolve()
    assert sha(path) == binary_sha
    assert module.estimators_numeric_mode() == 0
    assert module.estimators_vendor() == 'metal'
    assert module.scoped_gemm_flags() == mask
    for name in ('pca_fit', 'scoped_gemm_count', 'scoped_gemm_metadata'):
        assert callable(getattr(module, name, None)), name
    return module


def load_x(path, expected_sha):
    import numpy as np
    assert sha(path) == expected_sha, 'data pin mismatch'
    with np.load(path, allow_pickle=False) as packet:
        x = np.array(packet['X'], copy=True, order='C')
    assert x.dtype == np.dtype('float32') and x.ndim == 2
    assert x.shape[1] == 220 and x.shape[0] >= 128, 'istella full-feature covariance reach required'
    assert x.size <= 2147483647 and np.isfinite(x).all()
    return x


def outputs(model):
    import numpy as np
    names = ('components_', 'mean_', 'explained_variance_',
             'explained_variance_ratio_', 'singular_values_')
    arrays = {n: np.array(getattr(model, n), copy=True, order='C') for n in names}
    arrays['noise_variance_'] = np.asarray(model.noise_variance_, dtype='float64')
    return arrays


def capture(binary_sha, mask, data, data_sha, out, scored, preflight=False):
    import numpy as np
    import time
    sys.path.insert(0, str(ROOT / 'python'))
    module = native(binary_sha, mask)
    if preflight:
        return
    from mojolearn import PCA
    x = load_x(data, data_sha)
    before = [module.scoped_gemm_count(r, a) for r in range(3) for a in range(3)]
    # Same public call as classical_two_datasets.OursPCA; output copy completes span.
    start = time.perf_counter_ns() if scored else None
    model = PCA(n_components=10, svd_solver='covariance_eigh', whiten=False, random_state=7)
    model.fit(x)
    arrays = outputs(model)
    elapsed = (time.perf_counter_ns() - start) / 1e6 if scored else None
    counts = [module.scoped_gemm_count(r, a) - before[r * 3 + a]
              for r in range(3) for a in range(3)]
    expected = [0] * 9
    expected[6 + int(mask == 52)] = 1
    assert counts == expected, ('NO_REACH or unexpected route', counts, expected)
    meta = [module.scoped_gemm_metadata(i) for i in range(15)]
    nr, nf = x.shape
    initial = max(1, 640 // (((nf + 63) // 64) ** 2))
    per = ((nr + initial - 1) // initial + 31) // 32 * 32
    splits = (nr + per - 1) // per
    assert meta[:13] == [2, int(mask == 52), nf, nf, nr, splits, per, 1, 1, nf, nf, 1, 1]
    assert not model.input_copied_, 'prepared input unexpectedly converted'
    assert all(np.isfinite(a).all() for a in arrays.values())
    # Public fit must preserve caller input, including under skipped device restore.
    with np.load(data, allow_pickle=False) as original:
        assert np.array_equal(x.view('uint32'), original['X'].view('uint32'))
    np.savez(out, **arrays)
    record(str(out) + '.json', dict(binary_sha=binary_sha, mask=mask, shape=list(x.shape),
           counts=counts, metadata=meta, elapsed_ms=elapsed, scored=scored,
           span='public PCA constructor + fit + first complete fitted-output copies',
           input_preserved=True, input_copied=False))


def oracle(data, data_sha, path):
    import numpy as np
    x = load_x(data, data_sha)
    # All rows; chunking only limits FP64 memory. This is untimed host verification.
    nr, nf = x.shape
    mean = np.zeros(nf, dtype='float64')
    for i in range(0, nr, 8192):
        mean += x[i:i + 8192].astype('float64').sum(axis=0)
    mean /= nr
    cov = np.zeros((nf, nf), dtype='float64')
    for i in range(0, nr, 8192):
        z = x[i:i + 8192].astype('float64') - mean
        cov += z.T @ z
    cov /= nr - 1
    eig = np.linalg.eigvalsh(cov)[::-1]
    np.savez(path, mean=mean, cov=cov, eigenvalues=eig, rows=np.asarray(nr))


def metrics(packet, reference):
    import numpy as np
    with np.load(packet, allow_pickle=False) as p, np.load(reference, allow_pickle=False) as ref:
        c, mu, ev, ratio, singular = [p[k].astype('float64') for k in
            ('components_', 'mean_', 'explained_variance_', 'explained_variance_ratio_', 'singular_values_')]
        assert c.shape == (10, 220) and mu.shape == (220,)
        assert ev.shape == ratio.shape == singular.shape == (10,)
        assert all(np.isfinite(p[k]).all() for k in p.files)
        cov, mean, eig, nr = ref['cov'], ref['mean'], ref['eigenvalues'], int(ref['rows'])
        trace = float(np.trace(cov))
        assert trace > 0
        scale = max(1.0, float(np.linalg.norm(cov)))
        top = eig[:10]
        expected_s = np.sqrt(np.maximum(top, 0) * (nr - 1))
        expected_ratio = top / trace
        expected_noise = float(np.maximum(eig[10:], 0).mean())
        errors = {}
        for name, a, b in [('mean', mu, mean), ('variance', ev, top),
                           ('ratio', ratio, expected_ratio), ('singular', singular, expected_s),
                           ('noise', p['noise_variance_'], np.asarray(expected_noise))]:
            errors[name + '_relative'] = float(np.linalg.norm(a - b)) / max(1.0, float(np.linalg.norm(b)))
            errors[name + '_maxabs'] = float(np.max(np.abs(a - b)))
        residual = cov @ c.T - c.T * ev
        errors['eigen_residual_relative'] = float(np.linalg.norm(residual)) / scale
        errors['eigen_residual_maxabs'] = float(np.max(np.abs(residual)))
        ortho = c @ c.T - np.eye(10)
        errors['orthogonality_relative'] = float(np.linalg.norm(ortho)) / np.sqrt(10)
        errors['orthogonality_maxabs'] = float(np.max(np.abs(ortho)))
        # Reconstruction error, independent of component signs/degenerate rotations.
        projector = c.T @ c
        errors['reconstruction_relative'] = float(np.trace(cov - 2 * cov @ projector + projector @ cov @ projector)) / trace
        return errors


def compare(directory):
    import math
    a, b = [metrics(directory / (arm + '.npz'), directory / 'oracle.npz') for arm in 'AB']
    rows = {key: dict(A=a[key], B=b[key], pass_no_worse=math.isfinite(a[key]) and math.isfinite(b[key]) and b[key] <= a[key]) for key in a}
    # Keep 5e-6 independently referenced error bound and zero regression allowance.
    bounded = [key for key in a if key.endswith('_relative') and key != 'reconstruction_relative']
    ok = all(v['pass_no_worse'] for v in rows.values()) and all(a[k] <= BOUND and b[k] <= BOUND for k in bounded)
    return ok, rows


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '_capture':
        _, _, binary_sha, mask, data, data_sha, out, scored, preflight = sys.argv
        capture(binary_sha, int(mask), data, data_sha, Path(out), scored == '1', preflight == '1')
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source'); p.add_argument('tag'); p.add_argument('action', choices=('quality', 'timing'))
    p.add_argument('data'); p.add_argument('data_sha')
    p.add_argument('--quality-report'); p.add_argument('--quality-sha')
    args = p.parse_args()
    assert re.fullmatch('[A-Za-z0-9_.-]+', args.tag)
    assert re.fullmatch('[0-9a-f]{64}', args.data_sha)
    assert subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip() == 'Apple M3 Ultra'
    os.chdir(ROOT)
    harness = source_contract(args.source)
    data = Path(args.data).expanduser().resolve()
    assert sha(data) == args.data_sha
    arms = Path.home() / 'mq/verified-arms' / args.source / BINDING
    manifest = json.loads((arms / 'manifest.json').read_text())
    expected = dict(source_sha=args.source, binding=BINDING, numeric_mode='fast', defines_A=A_FLAGS, defines_B=B_FLAGS)
    assert all(manifest[k] == v for k, v in expected.items())
    hashes = {arm: sha(arms / (arm + '.so')) for arm in 'AB'}
    assert manifest['hashes'] == hashes
    identity = dict(source_sha=args.source, harness_source=harness, contract=CONTRACT,
                    helper_sha=sha(__file__), data_sha=args.data_sha, data_path=str(data),
                    manifest=manifest, bound=BOUND, error_regression_allowance=0)
    prior = None
    if args.action == 'timing':
        assert args.quality_report and args.quality_sha
        prior_path = Path(args.quality_report).expanduser().resolve()
        assert sha(prior_path) == args.quality_sha
        prior = json.loads(prior_path.read_text())
        assert prior['status'] == 'PASS' and all(prior[k] == v for k, v in identity.items())
        assert all(row['pass_no_worse'] and row['B'] <= row['A'] for row in prior['metrics'].values())
        for name, digest in prior['packets'].items():
            assert sha(prior_path.parent / name) == digest
        assert prior['A']['counts'] == [0,0,0,0,0,0,1,0,0]
        assert prior['B']['counts'] == [0,0,0,0,0,0,0,1,0]
    else:
        assert not args.quality_report and not args.quality_sha
    directory = Path.home() / 'mq/out' / (args.tag + '-' + args.action)
    directory.mkdir(parents=True, exist_ok=False)
    target = ROOT / 'python/mojolearn' / (MODULE + '.so')
    assert not target.is_symlink()
    old = target.exists()
    if old:
        shutil.copy2(target, directory / 'original.so')
    env = dict(os.environ, MOJOLEARN_NUMERIC_MODE='fast', MOJOLEARN_VENDOR='apple',
               MOJOLEARN_BENCH_INSTALLED='0', PYTHONPATH=str(ROOT / 'python'),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    env.pop('PYTHONOPTIMIZE', None)
    def run(arm, preflight):
        install(arms / (arm + '.so'), target)
        assert sha(target) == hashes[arm]
        command = [sys.executable, str(Path(__file__).resolve()), '_capture', hashes[arm],
                   '0' if arm == 'A' else '52', str(data), args.data_sha,
                   str(directory / (arm + '.npz')), '1' if args.action == 'timing' else '0',
                   '1' if preflight else '0']
        log = directory / (arm + ('.preflight.log' if preflight else '.log'))
        with log.open('x') as stream:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            print('\n'.join(log.read_text(errors='replace').splitlines()[-12:]))
            raise RuntimeError('capture infrastructure/reach failure: ' + str(log))
        assert sha(target) == hashes[arm] and sha(arms / (arm + '.so')) == hashes[arm]
    try:
        for arm in 'AB':
            run(arm, True)  # Resolve both real loaded APIs before fitting either.
        if args.action == 'timing':
            # Cross-tag reservation: never silently replay a scored pair after failure.
            key = hashlib.sha256(json.dumps([CONTRACT, args.source, args.data_sha, hashes], sort_keys=True).encode()).hexdigest()
            record(Path.home() / 'mq/out' / ('scoped-pca-fit-scored-' + key + '.json'),
                   dict(tag=args.tag, quality_sha=args.quality_sha, **identity))
            shutil.copy2(prior_path.parent / 'oracle.npz', directory / 'oracle.npz')
        else:
            oracle(data, args.data_sha, directory / 'oracle.npz')
        for arm in 'AB':
            run(arm, False)
        assert sha(data) == args.data_sha
        ok, measured = compare(directory)
        packets = {name: sha(directory / name) for name in ('A.npz', 'B.npz', 'A.npz.json', 'B.npz.json', 'oracle.npz')}
        report = dict(identity, status='PASS' if ok else 'HOLD', metrics=measured, packets=packets,
                      A=json.loads((directory / 'A.npz.json').read_text()),
                      B=json.loads((directory / 'B.npz.json').read_text()), scored=args.action == 'timing',
                      quality_report_sha=args.quality_sha, board_admitted=False)
        record(directory / 'report.json', report)
        print(json.dumps(dict(status=report['status'], report=str(directory / 'report.json'), scored=report['scored'])))
        if not ok:
            raise SystemExit(1)
    finally:
        if old:
            install(directory / 'original.so', target)
        else:
            target.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
