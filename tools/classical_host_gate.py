#!/usr/bin/env python3
"""Classical host inference gate (the classical host inference lane,
2026-09-13): LinearRegression, Ridge, TruncatedSVD, LogisticRegression and
PCA predicted on a CPU from a model fitted on a GPU, compared bit for bit;
since the kde svc host lane (2026-09-14) also KernelDensity, SVC and the
whitened PCA (`pca-whiten`, an identity_break lane of its own), and since the
knn host inference lane (2026-09-14) NearestNeighbors (`kneighbors`: distances
and indices), KNeighborsClassifier (`predict`, `predict_proba`) and
KNeighborsRegressor (`predict`), lanes knn, knn-clf and knn-reg, through
`mojolearn/host/_mojolearn_core_host.so`; since the neighbors and density
inference lane (2026-09-15) also the k-NN metric, ball cover and weighted
lanes, RadiusNeighbors (lanes radius, radius-manhattan, radius-chebyshev,
radius-minkowski-p3) and the KDE kernel, metric and weight lanes.

Two halves over tools/identity_break.py's own nine fixtures, so the
held-out rows here ARE the rows behind the `infer` column of the committed
GPU JSONs, and the digest this tool prints as `identity_hash` is that
column's cell (`identity_break._h` over the lane's probe outputs).

  record <dir> --lanes ols,ridge,...   on a GPU box, through the normal
           package: for every lane and every fixture, fit exactly what the
           identity_break lane fits (`LANES[lane]`), save the model as
           <dir>/<lane>/<fixture>/model.npz, predict the held-out rows on
           the GPU, reload the file through the class's own `load` and
           require the reload to predict the same bits, and write
           expected.json with the SHA-256, dtype and shape of every probe
           output, the identity_break hash, the vendor, GPU arch, numeric
           mode and model file hash. Refuses a CPU-only install, so the
           host binding can never record its own answer as the reference.
  check <dir>... [--gpu-column JSON ...]   on the CPU box, or on a GPU box
           through the host subclasses of `mojolearn._classical_host`
           (`_bind` answers the CPU binding, everything else is the GPU
           class's own Python): regenerate the held-out rows, verify them
           against the recorded hash, load model.npz through
           `mojolearn.host_model`, predict, and require every SHA-256,
           dtype and shape to equal the recording. With --gpu-column, the
           identity_break hash is also compared with that JSON's
           `cells[lane/fixture].infer` cell, one line per vendor, so one
           Mac run is judged against every committed GPU column.

Exit 0: every byte equal. 1: any mismatch. 2: the gate could not run.
--expect-mismatch inverts the verdict for the sabotage build
(MOJOLEARN_HOST_DIR pointing at a set built with
-D MOJOLEARN_HOST_SABOTAGE=1 and MOJOLEARN_HOST_ALLOW_SABOTAGE=1): exit 0
only if something differs.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))

from forest_host_gate import (  # noqa: E402
    digest_prediction, git_commit, host_info, sha256_bytes,
)

#: Held-out rows each lane probes, `identity_break.py`'s `Xh[:256]`.
PROBE_ROWS = 256
#: lane -> (estimator, identity_break probe, extra surfaces). The identity
#: probe is the tuple `identity_break` hashes for the `infer` column, in its
#: order (the knn lanes probe `Xh[:64]`, as `identity_break` does); the
#: first element is digested under PROBE_NAMES, the extras are hashed on
#: their own and are not part of that cell.
LANES = {
    'ols': ('LinearRegression', lambda e, X: (e.predict(X),), {}),
    'ridge': ('Ridge', lambda e, X: (e.predict(X),), {}),
    'tsvd': ('TruncatedSVD', lambda e, X: (e.transform(X),), {}),
    'logistic': ('LogisticRegression', lambda e, X: (e.predict_proba(X),),
                 {'predict': lambda e, X: e.predict(X),
                  'decision_function': lambda e, X: e.decision_function(X)}),
    'pca': ('PCA', lambda e, X: (e.transform(X),), {}),
    # The kde svc host lane (2026-09-14). kde's identity_break lane fits and
    # probes the first four columns; svc's probe is the pair (decision,
    # predict), the labels hashed as an extra too; pca-whiten adds the
    # whitened inverse as an extra, the second half of the host pair.
    'kde': ('KernelDensity', lambda e, X: (e.score_samples(X[:, :4]),), {}),
    'svc': ('SVC', lambda e, X: (e.decision_function(X), e.predict(X)),
            {'predict': lambda e, X: e.predict(X)}),
    'pca-whiten': ('PCA', lambda e, X: (e.transform(X),),
                   {'inverse_transform': lambda e, X: e.inverse_transform(e.transform(X))}),
    'knn': ('NearestNeighbors', lambda e, X: e.kneighbors(X[:64]),
            {'kneighbors_indices': lambda e, X: e.kneighbors(X[:64])[1]}),
    'knn-clf': ('KNeighborsClassifier',
                lambda e, X: (e.predict(X[:64]), e.predict_proba(X[:64])),
                {'predict_proba': lambda e, X: e.predict_proba(X[:64])}),
    'knn-reg': ('KNeighborsRegressor', lambda e, X: (e.predict(X[:64]),), {}),
    # lane/logistic-multiclass (2026-09-14): three classes through the
    # softmax loss; the probe is the pair (predict_proba, predict), the
    # order of the identity_break lane body in
    # docs/lanes/BRIEF_logistic_multiclass_2026-09-14.md section 5.
    'logistic-multiclass': ('LogisticRegression',
                            lambda e, X: (e.predict_proba(X), e.predict(X)),
                            {'predict': lambda e, X: e.predict(X),
                             'decision_function': lambda e, X: e.decision_function(X)}),
    # The neighbors and density inference lane (2026-09-15): every k-NN
    # metric and the ball cover arm, the distance-weighted vote and mean, the
    # radius query on its four metrics and the KDE kernel and metric pairs,
    # each probed exactly as its identity_break lane probes the held-out
    # rows (the radius probe is identity_break's `_ragged` over the sorted
    # query; the cosine KDE pair shifts its rows by 8, as the lane does).
    **{f'knn-{name}': ('NearestNeighbors', lambda e, X: e.kneighbors(X[:64]),
                       {'kneighbors_indices': lambda e, X: e.kneighbors(X[:64])[1]})
       for name in ('sqeuclidean', 'manhattan', 'chebyshev', 'cosine', 'minkowski-p3', 'rbc')},
    'knn-clf-distance': ('KNeighborsClassifier',
                         lambda e, X: (e.predict(X[:64]), e.predict_proba(X[:64])),
                         {'predict_proba': lambda e, X: e.predict_proba(X[:64])}),
    'knn-reg-distance': ('KNeighborsRegressor', lambda e, X: (e.predict(X[:64]),), {}),
    **{lane: ('RadiusNeighbors',
              lambda e, X: identity_tool()._ragged(e.radius_neighbors(X[:64], sort_results=True)), {})
       for lane in ('radius', 'radius-manhattan', 'radius-chebyshev', 'radius-minkowski-p3')},
    **{f'kde-{kernel}-{metric}': ('KernelDensity',
                                  (lambda shift: lambda e, X: (e.score_samples(X[:, :4] + shift),))(
                                      8.0 if metric == 'cosine' else 0.0), {})
       for kernel, metric in (('tophat', 'sqeuclidean'), ('epanechnikov', 'l1'), ('exponential', 'chebyshev'),
                              ('linear', 'cosine'), ('cosine', 'minkowski'))},
    'kde-weighted': ('KernelDensity', lambda e, X: (e.score_samples(X[:, :4]),), {}),
    # IsolationForest (same lane): identity_break scores every held-out row
    # and predicts the first 512, so these two lanes probe the whole
    # held-out draw (LANE_PROBE_ROWS) rather than its first PROBE_ROWS.
    **{lane: ('IsolationForest', lambda e, X: (e.score_samples(X), e.predict(X[:512])),
              {'predict': lambda e, X: e.predict(X[:512]),
               'decision_function': lambda e, X: e.decision_function(X)})
       for lane in ('iforest', 'iforest-tuned')},
    # GaussianMixture (same lane), through the inference-only mixture
    # binding: 64 held-out rows of the first four columns, as the lanes ask.
    'gmm': ('GaussianMixture',
            lambda e, X: (e.score_samples(X[:64, :4]), e.predict(X[:64, :4]), e.predict_proba(X[:64, :4])),
            {'predict': lambda e, X: e.predict(X[:64, :4]),
             'predict_proba': lambda e, X: e.predict_proba(X[:64, :4])}),
    'gmm-random-init': ('GaussianMixture', lambda e, X: (e.score_samples(X[:64, :4]),),
                        {'predict': lambda e, X: e.predict(X[:64, :4])}),
    # HDBSCAN (same lane): approximate_predict's labels and probabilities on
    # the first 256 held-out rows of four columns, as both lanes ask.
    **{lane: ('HDBSCAN',
              lambda e, X: tuple(__import__('mojolearn.hdbscan', fromlist=['approximate_predict'])
                                 .approximate_predict(e, X[:, :4])),
              {'probabilities': lambda e, X: __import__('mojolearn.hdbscan', fromlist=['approximate_predict'])
                                 .approximate_predict(e, X[:, :4])[1]})
       for lane in ('hdbscan', 'hdbscan-leaf')},
}
PROBE_NAMES = {'ols': 'predict', 'ridge': 'predict', 'tsvd': 'transform',
               'logistic': 'predict_proba', 'pca': 'transform',
               'kde': 'score_samples', 'svc': 'decision_function',
               'pca-whiten': 'transform',
               'knn': 'kneighbors_distances', 'knn-clf': 'predict',
               'knn-reg': 'predict', 'logistic-multiclass': 'predict_proba'}
PROBE_NAMES.update({lane: {'NearestNeighbors': 'kneighbors_distances', 'KNeighborsClassifier': 'predict',
                           'KNeighborsRegressor': 'predict', 'RadiusNeighbors': 'radius_neighbors_counts',
                           'KernelDensity': 'score_samples', 'IsolationForest': 'score_samples',
                           'GaussianMixture': 'score_samples', 'HDBSCAN': 'approximate_predict_labels'}[spec[0]]
                    for lane, spec in LANES.items() if lane not in PROBE_NAMES})


def _fit_logistic_multiclass(ml, X, yc, yr, Xh=None):
    """The `logistic-multiclass` lane body, word for word the one handed to
    tools/identity_break.py's owner (the brief, section 5), carried here
    until that file has it; `do_record` uses it when `ib.LANES` lacks the
    lane. Three classes from the fixture's own labels: the binary rule plus
    one for rows whose column 5 is above its median (a column no fixture
    perturbs; the median split keeps all three classes on every fixture)."""
    import numpy as np
    ib = identity_tool()
    y3 = (yc + (X[:, 5] > np.median(X[:, 5]))).astype(np.int32)
    m = ml.LogisticRegression(max_iter=50).fit(X, y3)
    return ib._fit(dict(coef=ib._h(m.coef_), proba=ib._h(m.predict_proba(X[:256]))), m,
                   lambda e: (e.predict_proba(Xh[:256]), e.predict(Xh[:256])))


#: lane -> fit, for a lane the gate knows before identity_break does.
LOCAL_FITS = {'logistic-multiclass': _fit_logistic_multiclass}


def identity_tool():
    import identity_break
    return identity_break


def package_root(args):
    if args.package_root is None:
        sys.path.insert(0, str(ROOT / 'python'))
    elif args.package_root:
        sys.path.insert(0, os.path.abspath(args.package_root))


#: Lanes whose identity_break probe reads the whole held-out draw; every
#: other lane reads its first PROBE_ROWS rows.
LANE_PROBE_ROWS = {'iforest': None, 'iforest-tuned': None}


def probe_rows(ib, lane, kind):
    """The held-out row count `lane` probes on fixture `kind`."""
    rows = LANE_PROBE_ROWS.get(lane, PROBE_ROWS)
    return int(ib.heldout(kind).shape[0]) if rows is None else rows


def held_out(ib, kind, lane=None):
    """The identity_break held-out slice the lane probes, as a numpy
    array, and its bytes' SHA-256."""
    Xh = ib.heldout(kind)[:probe_rows(ib, lane, kind)]
    return Xh, sha256_bytes(Xh.tobytes())


def digests_for(lane, model, Xh, ib):
    """Every surface's `(sha256, dtype, shape)`, the identity_break hash of
    the identity probe, and the seconds spent."""
    _, probe, extras = LANES[lane]
    out = {}
    started = time.perf_counter()
    outputs = probe(model, Xh)
    out['identity_hash'] = ib._h(*outputs)
    out[PROBE_NAMES[lane]] = dict(zip(('sha256', 'dtype', 'shape'), digest_prediction(outputs[0])))
    for name, fn in extras.items():
        out[name] = dict(zip(('sha256', 'dtype', 'shape'), digest_prediction(fn(model, Xh))))
    out['seconds'] = round(time.perf_counter() - started, 6)
    return out


def do_record(args):
    package_root(args)
    ib = identity_tool()
    import mojolearn
    vendor = mojolearn.vendor()
    if vendor == 'cpu':
        print('gate: record must run on a GPU box; this install is CPU-only', file=sys.stderr)
        return 2
    if mojolearn.numeric_mode() != 'identical':
        print(f'gate: record requires MOJOLEARN_NUMERIC_MODE=identical, this process is '
              f'{mojolearn.numeric_mode()}', file=sys.stderr)
        return 2
    lanes = [l for l in args.lanes.split(',') if l]
    unknown = [l for l in lanes if l not in LANES]
    if unknown:
        print(f'gate: unknown lanes {unknown}; this gate knows {sorted(LANES)}', file=sys.stderr)
        return 2
    fixtures = [f for f in ib.FIXTURES if not args.fixtures or f in args.fixtures.split(',')]
    for lane in lanes:
        for kind in fixtures:
            directory = args.fixture_dir / lane / kind
            if directory.exists() and any(directory.iterdir()) and not args.overwrite:
                print(f'gate: {directory} exists and is not empty; pass --overwrite', file=sys.stderr)
                return 2
    for lane in lanes:
        estimator = LANES[lane][0]
        for kind in fixtures:
            directory = args.fixture_dir / lane / kind
            directory.mkdir(parents=True, exist_ok=True)
            X, yc, yr = ib.fixture(kind)
            Xh_full = ib.heldout(kind)
            fit = (ib.LANES[lane] if lane in ib.LANES else LOCAL_FITS[lane])(mojolearn, X, yc, yr, Xh_full)
            model = fit.est
            if type(model).__name__ != estimator:
                print(f'gate: lane {lane} fitted {type(model).__name__}, not {estimator}', file=sys.stderr)
                return 2
            Xh, x_sha = held_out(ib, kind, lane)
            gpu = digests_for(lane, model, Xh, ib)
            # The identity_break probe on the fitted model must agree with
            # the tool's own infer cell for this fit, or the probe here is
            # not that column's.
            tool_hash = ib._h(*fit.probe(model))
            if tool_hash != gpu['identity_hash']:
                print(f'gate: {lane}/{kind} probe hash {gpu["identity_hash"]} is not the '
                      f'identity_break probe hash {tool_hash}', file=sys.stderr)
                return 2
            model_path = directory / 'model.npz'
            model.save(str(model_path))
            back = type(model).load(str(model_path))
            reload = digests_for(lane, back, Xh, ib)
            for key in gpu:
                if key == 'seconds':
                    continue
                if gpu[key] != reload[key]:
                    print(f'gate: {lane}/{kind} {key} differs between the fitted model and its '
                          f'reload on the GPU path: {gpu[key]} vs {reload[key]}', file=sys.stderr)
                    return 1
            spec = dict(lane=lane, kind=kind, estimator=estimator, heldout_seed=ib.HELDOUT_SEED,
                        probe_rows=probe_rows(ib, lane, kind), x_sha256=x_sha)
            (directory / 'fixture.json').write_text(json.dumps(spec, indent=2, sort_keys=True) + '\n')
            report = dict(
                status='RECORDED', lane=lane, kind=kind, estimator=estimator, vendor=vendor,
                gpu_arch=mojolearn.gpu_arch(), numeric_mode=mojolearn.numeric_mode(),
                model_sha256=sha256_bytes(model_path.read_bytes()), x_sha256=x_sha,
                host=host_info(), commit=git_commit(),
                recorded_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), predictions=gpu,
                reload_equal=True,
            )
            (directory / 'expected.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
            print(f"record {lane} {kind} {estimator} identity_hash {gpu['identity_hash']} "
                  f"{PROBE_NAMES[lane]} {gpu[PROBE_NAMES[lane]]['sha256']} (vendor {vendor})")
    return 0


def gpu_columns(paths):
    cols = []
    for p in paths or []:
        with open(p) as fh:
            j = json.load(fh)
        cols.append((os.path.basename(p), j))
    return cols


def do_check(args):
    package_root(args)
    try:
        ib = identity_tool()
        import mojolearn
        from mojolearn._classical_host import binary_path, binary_paths, host_model
    except Exception as exc:
        print(f'gate: import failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
    columns = gpu_columns(args.gpu_column)
    dirs = []
    for root in args.fixture_dir:
        if (root / 'expected.json').exists():
            dirs.append(root)
        else:
            dirs.extend(sorted(p.parent for p in root.glob('*/*/expected.json')))
    if not dirs:
        print('gate: no fixture directory with an expected.json under the paths given', file=sys.stderr)
        return 2
    results = []
    verdict_ok = True
    for directory in dirs:
        expected = json.loads((directory / 'expected.json').read_text())
        if expected.get('status') != 'RECORDED' or 'predictions' not in expected:
            print(f'gate: {directory} status is {expected.get("status")!r}; the GPU-side recording '
                  'is OWED (tools/classical_host_gate.py record on a GPU box)', file=sys.stderr)
            return 2
        spec = json.loads((directory / 'fixture.json').read_text())
        lane, kind = spec['lane'], spec['kind']
        if lane not in LANES or kind not in ib.FIXTURES or int(spec.get('probe_rows', 0)) != probe_rows(ib, lane, kind):
            print(f'gate: {directory} fixture.json names a lane, fixture or probe size this gate '
                  'does not know', file=sys.stderr)
            return 2
        Xh, x_sha = held_out(ib, kind, lane)
        if x_sha != spec.get('x_sha256') or x_sha != expected.get('x_sha256'):
            print(f'gate: {directory} regenerated held-out rows hash {x_sha}, the fixture records '
                  f'{spec.get("x_sha256")}', file=sys.stderr)
            return 2
        model_path = directory / 'model.npz'
        model_sha = sha256_bytes(model_path.read_bytes())
        if expected.get('model_sha256') != model_sha:
            print(f'gate: {model_path} hashes {model_sha}, expected.json records '
                  f'{expected.get("model_sha256")}', file=sys.stderr)
            return 2
        try:
            model = host_model(str(model_path))
            got = digests_for(lane, model, Xh, ib)
        except Exception as exc:
            print(f'gate: {directory} host predict failed: {type(exc).__name__}: {exc}', file=sys.stderr)
            return 2
        if model.estimator != expected.get('estimator'):
            print(f'gate: {directory} model is {model.estimator}, expected.json says '
                  f'{expected.get("estimator")}', file=sys.stderr)
            return 2
        want = expected['predictions']
        cases = []
        for key in sorted(k for k in want if k not in ('seconds', 'identity_hash')):
            if key not in got:
                print(f'gate: {directory} {key} recorded but not computed here', file=sys.stderr)
                return 2
            equal = (want[key]['sha256'] == got[key]['sha256'] and want[key]['dtype'] == got[key]['dtype']
                     and list(want[key]['shape']) == list(got[key]['shape']))
            verdict_ok = verdict_ok and equal
            cases.append(dict(case=key, want=want[key]['sha256'], got=got[key]['sha256'],
                              want_dtype=want[key]['dtype'], got_dtype=got[key]['dtype'],
                              want_shape=want[key]['shape'], got_shape=got[key]['shape'], equal=equal))
            print(f"check {lane} {kind} {key} {'EQUAL' if equal else 'DIFFER'} "
                  f"gpu {want[key]['sha256']} host {got[key]['sha256']}")
        ih_equal = want['identity_hash'] == got['identity_hash']
        verdict_ok = verdict_ok and ih_equal
        print(f"check {lane} {kind} identity_hash {'EQUAL' if ih_equal else 'DIFFER'} "
              f"gpu {want['identity_hash']} host {got['identity_hash']}")
        vendors = []
        for label, j in columns:
            cell = j.get('cells', {}).get(f'{lane}/{kind}')
            infer = (cell or {}).get('infer') or []
            theirs = infer[0] if infer else None
            equal = theirs == got['identity_hash']
            if theirs is None:
                status = 'ABSENT'
            elif isinstance(theirs, str) and theirs.startswith('n/a:'):
                # A record older than the lane's infer probe carries its
                # reason (`n/a:transductive` on hdbscan before 2026-09-15),
                # not a hash: nothing to compare, so it cannot differ.
                status = 'N/A'
                equal = None
            else:
                status = 'EQUAL' if equal else 'DIFFER'
                verdict_ok = verdict_ok and equal
            vendors.append(dict(column=label, vendor=j.get('vendor'), infer=theirs, equal=equal if theirs else None))
            print(f"column {lane} {kind} {label} {status} theirs {theirs} host {got['identity_hash']}")
        results.append(dict(lane=lane, kind=kind, estimator=model.estimator,
                            recorded_vendor=expected.get('vendor'), recorded_gpu_arch=expected.get('gpu_arch'),
                            recorded_numeric_mode=expected.get('numeric_mode'), model_sha256=model_sha,
                            identity_hash=dict(want=want['identity_hash'], got=got['identity_hash'], equal=ih_equal),
                            columns=vendors, seconds=got['seconds'], cases=cases))
    if args.expect_mismatch:
        verdict = 'EXPECTED MISMATCH SEEN' if not verdict_ok else 'SABOTAGE NOT CAUGHT'
        code = 0 if not verdict_ok else 1
    else:
        verdict = 'IDENTICAL' if verdict_ok else 'MISMATCH'
        code = 0 if verdict_ok else 1
    report = dict(verdict=verdict, expect_mismatch=bool(args.expect_mismatch), exit=code,
                  binary=binary_path(), binaries=binary_paths(), vendor=mojolearn.vendor(), host=host_info(),
                  gpu_columns=[label for label, _ in columns], commit=git_commit(),
                  checked_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), fixtures=results)
    if args.report:
        if args.report.exists():
            print(f'gate: {args.report} exists; refusing to overwrite a report', file=sys.stderr)
            return 2
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(f'gate verdict {verdict} ({len(results)} fixtures, {len(columns)} GPU columns, exit {code})')
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--package-root', default=None,
                        help="directory to import mojolearn from (default: this checkout's python/; "
                             "'' for the installed package)")
    sub = parser.add_subparsers(dest='command', required=True)
    rec = sub.add_parser('record', help='on a GPU box, fit, save and record every lane and fixture under a directory')
    rec.add_argument('fixture_dir', type=Path)
    rec.add_argument('--lanes', required=True, help='comma separated: ' + ','.join(LANES))
    rec.add_argument('--fixtures', default='', help='comma separated identity_break fixtures (default all nine)')
    rec.add_argument('--overwrite', action='store_true')
    chk = sub.add_parser('check', help='on the CPU box, compare the host predictions with every expected.json')
    chk.add_argument('fixture_dir', type=Path, nargs='+',
                     help='a <lane>/<fixture> directory, or a root holding <lane>/<fixture>/ directories')
    chk.add_argument('--gpu-column', action='append', default=[],
                     help='an identity_break JSON whose infer cells are compared too (repeatable)')
    chk.add_argument('--report', type=Path, help='new exclusive JSON report')
    chk.add_argument('--expect-mismatch', action='store_true')
    args = parser.parse_args()
    if args.command == 'record':
        return do_record(args)
    return do_check(args)


if __name__ == '__main__':
    sys.exit(main())
