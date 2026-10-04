#!/usr/bin/env python3
"""M3-only unscored actual decomp/PCA quality. No timing or installs.

SOURCE TAG PROFILE validates staged A/B manifests, captures each isolated arm,
then gates against FP64 and A with zero error-regression allowance.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BINDING = 'scoped_gemm_probe'
AUDIT = 'MOJOLEARN_SCOPED_GEMM_AUDIT'
FLAGS = ['MOJOLEARN_SCOPED_GEMM_' + x for x in
         ['G1_TALL', 'G1_DENSE', 'G1_GRAM', 'G2_NARROW', 'SPLIT', 'PCA']]
PROFILES = {'tall': 1, 'dense': 2, 'gram': 4, 'narrow': 8,
            'gram-split': 20, 'pca': 52, 'all': 63}
BOUND = 5e-6
# (name, m, n, k, transpose A/B, aliased inputs, data kind)
CASES = [
    ('tall-anchor', 32768, 64, 220, 0, 0, 0, 'random'),
    ('tall-neighbor', 4097, 71, 221, 0, 0, 0, 'cancel'),
    ('tall-low-bound', 4096, 32, 128, 0, 0, 0, 'dynamic'),
    ('tall-outside', 4095, 32, 128, 0, 0, 0, 'random'),
    ('dense-anchor', 2048, 512, 512, 0, 0, 0, 'random'),
    ('dense-neighbor', 1025, 257, 257, 0, 0, 0, 'cancel'),
    ('dense-outside', 1023, 256, 256, 0, 0, 0, 'random'),
    ('narrow-anchor', 32768, 8, 220, 0, 1, 0, 'random'),
    ('narrow-neighbor', 4097, 9, 221, 0, 1, 0, 'dynamic'),
    ('narrow-low-bound', 4096, 2, 128, 0, 1, 0, 'cancel'),
    ('narrow-outside', 4096, 17, 128, 0, 1, 0, 'random'),
    ('vector-control', 4096, 1, 128, 0, 1, 0, 'random'),
    ('square-control', 1024, 1024, 220, 0, 0, 0, 'random'),
    ('gram-nt-anchor', 1024, 1024, 220, 0, 1, 1, 'random'),
    ('gram-nt-neighbor', 221, 221, 257, 0, 1, 1, 'dynamic'),
    ('gram-tn-new-orientation', 220, 220, 257, 1, 0, 1, 'random'),
    ('gram-nonalias-control', 220, 220, 257, 1, 0, 0, 'random'),
    ('gram-128-control', 128, 128, 220, 0, 1, 1, 'random'),
    ('gram-before-split', 220, 220, 1023, 0, 1, 1, 'cancel'),
    ('gram-at-split', 220, 220, 1024, 0, 1, 1, 'random'),
    ('gram-after-split', 220, 220, 1025, 0, 1, 1, 'dynamic'),
    ('gram-tn-split', 220, 220, 1025, 1, 0, 1, 'random'),
    ('tt-control', 129, 131, 257, 1, 1, 0, 'random'),
]
# Covariance actual entrance including its separate split policy and input lifecycle.
PCA_CASES = [('pca-fused-control', 257, 128, 1), ('pca-boundary', 257, 129, 1),
             ('pca-neighbor', 1025, 221, 1), ('pca-no-restore', 1025, 220, 0),
             ('pca-one-split-atomic-control', 31, 129, 1)]


def inputs(case):
    import numpy as np
    _, m, n, k, ta, tb, alias, kind = case
    rng = np.random.default_rng(711)
    a = rng.standard_normal((k, m) if ta else (m, k)).astype('float32')
    b = a if alias else rng.standard_normal((n, k) if tb else (k, n)).astype('float32')
    if kind == 'dynamic':
        a *= np.float32(16)
        if not alias:
            b *= np.float32(1 / 16)
    elif kind == 'cancel':
        a.flat[1::2] *= np.float32(-1)
    return a, b


def plan(case):
    _, m, n, k, ta, tb, alias, _ = case
    tiles = ((m + 63) // 64) * ((n + 63) // 64)
    splits = min(640 // tiles, k // 512) if tiles < 640 and k >= 1024 else 1
    if splits > 1:
        per = ((k + splits - 1) // splits + 31) // 32 * 32
        return 1, (k + per - 1) // per, per
    return 0, 1, k


def chosen(case, mask):
    _, m, n, k, ta, tb, alias, _ = case
    result = 0
    if mask & 1 and not ta and not tb and m >= 4096 and 32 <= n <= 128 and 128 <= k <= 512:
        result = 1
    if mask & 2 and not ta and not tb and 1024 <= m <= 8192 and 256 <= n <= 768 and 256 <= k <= 768 and m >= 2 * n:
        result = 1
    if mask & 4 and alias and m == n and 129 <= m <= 1024 and k >= 128 and ta != tb:
        result = 1
    if mask & 8 and not ta and tb and m >= 4096 and 2 <= n <= 16 and 128 <= k <= 512:
        result = 2
    if plan(case)[0] and not mask & 16:
        result = 0
    return result


def capture(so, destination, mask):
    import numpy as np
    spec = importlib.util.spec_from_file_location('_mojolearn_scoped_gemm_probe', so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.abi_version() == 1 and mod.enabled() == 1 and mod.numeric_mode() == 0
    assert mod.flags() == mask
    arrays, rows = {}, []
    def counts():
        return [mod.route_count(r, a) for r in range(3) for a in range(3)]
    for case in CASES:
        name, m, n, k, ta, tb, alias, _ = case
        a, b = inputs(case)
        c = np.full((m, n), np.nan, dtype='float32')
        before = counts()
        mod.gemm(a.ctypes.data, b.ctypes.data, c.ctypes.data, [m, n, k, ta, tb, alias])
        delta = [x-y for x, y in zip(counts(), before)]
        route, splits, per = plan(case)
        arm = chosen(case, mask)
        expected = [0] * 9
        expected[route * 3 + arm] = 1
        assert delta == expected, (name, delta, expected)
        meta = [mod.metadata(i) for i in range(15)]
        assert meta[:8] == [route, arm, m, n, k, splits, per, route], (name, meta)
        assert meta[8:13] == [1 if ta else k, m if ta else 1,
                              1 if tb else n, k if tb else 1, alias]
        arrays[name] = c
        rows.append({'case': name, 'metadata': meta, 'counts': delta, 'selected': arm})
    for name, nr, nc, restore in PCA_CASES:
        rng = np.random.default_rng(818)
        x = rng.standard_normal((nr, nc)).astype('float32')
        c = np.full((nc, nc), np.nan, dtype='float32')
        mu = np.full(nc, np.nan, dtype='float32')
        after = np.full_like(x, np.nan)
        before = counts()
        mod.covariance(x.ctypes.data, c.ctypes.data, mu.ctypes.data, after.ctypes.data, [nr, nc, restore])
        delta = [a-b for a, b in zip(counts(), before)]
        expected = [0] * 9
        arm = int(mask & 52 == 52 and nc > 128 and nr >= 128)
        meta = None
        if nc > 128:
            expected[6 + arm] = 1
            tiles = ((nc + 63) // 64) ** 2
            initial = max(1, 640 // tiles)
            per = ((nr + initial - 1) // initial + 31) // 32 * 32
            splits = (nr + per - 1) // per
            meta = [mod.metadata(i) for i in range(15)]
            assert meta[:13] == [2, arm, nc, nc, nr, splits, per, 1, 1, nc, nc, 1, 1]
        assert delta == expected, (name, delta, expected)
        arrays[name] = c
        arrays[name + '-mean'] = mu
        arrays[name + '-input-after'] = after
        rows.append({'case': name, 'metadata': meta, 'counts': delta, 'selected': arm})
    np.savez(destination, **arrays)
    Path(str(destination) + '.json').write_text(json.dumps(rows, indent=2) + '\n')


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '_capture':
        capture(Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('tag')
    parser.add_argument('profile', choices=PROFILES)
    args = parser.parse_args()
    assert re.fullmatch('[0-9a-f]{40}', args.source)
    assert re.fullmatch('[A-Za-z0-9_.-]+', args.tag)
    assert subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip() == 'Apple M3 Ultra'
    os.chdir(ROOT)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == args.source
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'bindings/', 'core/', 'gemm/', 'x_decomp/', 'decomposition/', 'experiments/', 'tools/scoped_gemm_quality.py'], check=True)
    arms = Path.home() / 'mq/verified-arms' / args.source / BINDING
    manifest = json.loads((arms / 'manifest.json').read_text())
    mask = PROFILES[args.profile]
    expected_a = '-D ' + AUDIT
    expected_b = expected_a + ''.join(' -D ' + flag for i, flag in enumerate(FLAGS) if mask & (1 << i))
    assert manifest['source_sha'] == args.source and manifest['binding'] == BINDING
    assert manifest['numeric_mode'] == 'fast'
    assert manifest['defines_A'] == expected_a and manifest['defines_B'] == expected_b
    for arm in 'AB':
        assert hashlib.sha256((arms / (arm + '.so')).read_bytes()).hexdigest() == manifest['hashes'][arm]
    directory = Path.home() / 'mq/out' / (args.tag + '-quality')
    directory.mkdir(parents=True, exist_ok=False)
    os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    # Isolated Python processes load explicit verified paths; no installed .so assumptions.
    for arm, flagmask in [('A', 0), ('B', mask)]:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '_capture', str(arms / (arm + '.so')), str(directory / (arm + '.npz')), str(flagmask)], check=True)
    import numpy as np
    a, b = np.load(directory / 'A.npz'), np.load(directory / 'B.npz')
    reached = json.loads((directory / 'B.npz.json').read_text())
    references = {}
    for case in CASES:
        x, y = inputs(case)
        ta, tb = case[4:6]
        references[case[0]] = (x.T if ta else x).astype('float64') @ (y.T if tb else y).astype('float64')
    for name, nr, nc, restore in PCA_CASES:
        x = np.random.default_rng(818).standard_normal((nr, nc)).astype('float32').astype('float64')
        centered = x - x.mean(axis=0)
        references[name] = centered.T @ centered / (nr - 1)
        references[name + '-mean'] = x.mean(axis=0)
        references[name + '-input-after'] = x if restore or nc <= 128 else centered
    rows, failures = [], []
    for key, ref in references.items():
        scale = max(1.0, float(np.linalg.norm(ref)))
        ma = float(np.max(np.abs(a[key].astype('float64') - ref)))
        mb = float(np.max(np.abs(b[key].astype('float64') - ref)))
        ea = float(np.linalg.norm(a[key].astype('float64') - ref)) / scale
        eb = float(np.linalg.norm(b[key].astype('float64') - ref)) / scale
        finite = bool(np.isfinite(a[key]).all() and np.isfinite(b[key]).all())
        ok = finite and ea <= BOUND and eb <= BOUND and eb <= ea and mb <= ma
        if not ok:
            failures.append(key)
        rows.append({'case': key, 'A_relative_fro': ea, 'B_relative_fro': eb,
                     'A_maxabs': ma, 'B_maxabs': mb, 'same_words': bool(np.array_equal(a[key].view('uint32'), b[key].view('uint32'))), 'pass': ok})
    report = {'source_sha': args.source, 'binding': BINDING, 'profile': args.profile,
              'contract': 'scoped-actual-decomp-pca-quality-v1',
              'baseline': 'current-main AFN decomp/PCA; small PCA fused Gram unchanged, NOT SDK screen G0',
              'manifest': manifest, 'bound': BOUND, 'error_regression_allowance': 0,
              'reach': reached, 'cases': rows, 'failures': failures,
              'status': 'PASS' if not failures else 'HOLD'}
    (directory / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'report': str(directory / 'report.json'), 'status': report['status'], 'failures': failures}))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
