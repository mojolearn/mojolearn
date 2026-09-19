# SPDX-License-Identifier: Apache-2.0
"""One/two-GPU CV numerical and placement captures from source or a wheel.

Five uneven folds for classifier/regressor, one/two/reversed device schedules,
two repeats, saved-model replay and comparator controls. No physical execution
trace or native arithmetic sabotage is inferred from these checks.
"""
import argparse
import copy
import functools
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import subprocess
import time

from ._parallel_cv_witness import array_digest, score_with_witness, read_records, compare_records
from ._verify_distributed import verify_distribution_records

__all__ = ['main', 'source_identity', 'validate_receipt', 'compare']
PROTOCOL = 'parallel-cv-v2'
PROFILE_FILES = frozenset(('__init__.py', '_verify_parallel_cv.py', '_parallel_cv_witness.py',
                           'model_selection.py', 'parallel_model_selection.py', '_parallel_pool.py',
                           '_parallel_worker.py', '_gpu_witness.py', '_verify_distributed.py'))
PARTS = ('model', 'predict', 'raw_predict', 'reload_predict', 'loss_curve', 'score')
FAULTS = ('dropped-fold', 'changed-model', 'changed-prediction', 'aliased-device')


def source_identity(root=None):
    """Hash the actual shipped profile; distinguish checkout, archive and wheel.

    An explicit repository root requires a clean checkout or a full guarded
    archive witness. Installed packages legitimately have no Git commit.
    """
    explicit = root is not None
    package = Path(root) / 'python' / 'mojolearn' if explicit else Path(__file__).resolve().parent
    root = Path(root) if explicit else package.parent.parent
    commit, guarded, kind = None, os.environ.get('MOJOLEARN_COMMIT'), 'installed-package'
    if guarded is not None and not re.fullmatch('[0-9a-f]{40}', guarded):
        raise RuntimeError('CV capture requires a full source commit witness when MOJOLEARN_COMMIT is set')
    if (root / '.git').exists():
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        if subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
            raise RuntimeError('record from a committed, clean source checkout')
        kind = 'clean-git-checkout'
    elif guarded:
        kind = 'guarded-source-archive'
    elif explicit:
        raise RuntimeError('CV capture requires a full source commit witness')
    return {'commit': commit, 'guarded_source_commit': guarded, 'kind': kind,
            'source_sha256': {name: hashlib.sha256((package / name).read_bytes()).hexdigest()
                              for name in sorted(PROFILE_FILES)}}


def distribution_identity(package):
    try:
        dist = metadata.distribution('mojolearn')
    except metadata.PackageNotFoundError:
        return None, None
    direct = dist.read_text('direct_url.json')
    info = {'version': dist.version, 'root': str(Path(dist.locate_file('')).resolve()),
            'import_matches_distribution': Path(dist.locate_file('mojolearn/__init__.py')).resolve() == package / '__init__.py',
            'editable': bool(direct and json.loads(direct).get('dir_info', {}).get('editable'))}
    return dist, info


def _atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def validate_receipt(report):
    if (report.get('protocol') != PROTOCOL or report.get('status') != 'NUMERICS_AND_PLACEMENT_PASS'
            or report.get('vendor') not in ('cuda', 'hip') or report.get('folds') != 5
            or report.get('repeats') != 2):
        raise ValueError('incomplete or incompatible CV capture')
    devices = report.get('devices', [])
    if len(devices) != 2 or len(set(devices)) != 2 or any(type(d) is not int or d < 0 for d in devices):
        raise ValueError('CV capture needs two distinct device indices')
    profile = report.get('source', {}).get('source_sha256', {})
    if set(profile) != PROFILE_FILES or any(not re.fullmatch('[0-9a-f]{64}', str(h)) for h in profile.values()):
        raise ValueError('CV source profile is missing or malformed')
    bindings = report.get('bindings', {})
    if set(bindings) != {'_mojolearn_gbdt'} or any(
        v.get('vendor') != report['vendor'] or not re.fullmatch('[0-9a-f]{64}', str(v.get('sha256')))
        for v in bindings.values()):
        raise ValueError('CV native provenance is missing or malformed')
    inputs = report.get('inputs', {})
    if set(inputs) != {'X', 'classifier', 'regressor'} or any(not re.fullmatch('[0-9a-f]{64}', str(v)) for v in inputs.values()):
        raise ValueError('CV fixture input witnesses are missing')
    layouts = [[devices[0]], devices, devices[::-1]]
    expected = {(model, tuple(order), repeat) for model in ('regressor', 'classifier')
                for order in layouts for repeat in range(2)}
    seen, baseline, scores = set(), {}, {}
    for run in report.get('runs', []):
        key = (run['model'], tuple(run['devices']), run['repeat'])
        if type(run['repeat']) is not int or key not in expected or key in seen:
            raise ValueError('CV run is duplicate or unexpected')
        seen.add(key)
        actual = run['fold_records']
        if len(actual) != 5:
            raise ValueError('CV run lacks five fold witnesses')
        for fixture, record in actual.items():
            if record.get('fixture') != fixture or not re.fullmatch('[0-9a-f]{64}', fixture):
                raise ValueError('CV fixture witness is malformed')
            for part in PARTS:
                length = 16 if part == 'score' else 64
                if not re.fullmatch('[0-9a-f]{' + str(length) + '}', str(record.get(part))):
                    raise ValueError('CV numerical witness is malformed')
            if record['raw_predict'] != record['reload_predict']:
                raise ValueError('CV saved-model replay differs')
            if record['binding_sha256'] != bindings['_mojolearn_gbdt']['sha256']:
                raise ValueError('CV worker binding differs from parent')
        reference = baseline.setdefault(run['model'], actual)
        compare_records(reference, actual, vendor=report['vendor'], workers=len(run['devices']))
        value = run['scores_hex']
        if not re.fullmatch('[0-9a-f]{80}', value) or scores.setdefault(run['model'], value) != value:
            raise ValueError('CV fold scores differ or are malformed')
    if seen != expected:
        raise ValueError('CV capture is missing required runs')
    controls = report.get('controls', [])
    wanted = {model + ':' + fault for model in ('classifier', 'regressor') for fault in FAULTS}
    if len(controls) != len(wanted) or set(controls) != wanted:
        raise ValueError('CV comparator controls are missing')
    return True


def compare(left, right):
    """Compare complete numerical profiles, including independent GPU vendors."""
    validate_receipt(left)
    validate_receipt(right)
    if left['source']['source_sha256'] != right['source']['source_sha256'] or left['inputs'] != right['inputs']:
        raise ValueError('CV source profiles or fixture inputs differ')
    def numerics(report):
        return {run['model']: {'scores_hex': run['scores_hex'], 'folds': {key: {part: row[part] for part in PARTS}
                for key, row in run['fold_records'].items()}} for run in report['runs']}
    return numerics(left) == numerics(right)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-backend', choices=('cuda', 'hip'))
    parser.add_argument('--devices')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--require-installed', action='store_true')
    parser.add_argument('--compare', type=Path, nargs=2, metavar=('LEFT', 'RIGHT'))
    args = parser.parse_args(argv)
    if args.compare:
        if args.devices or args.out or args.require_backend or args.require_installed:
            parser.error('--compare cannot be combined with capture options')
        matched = compare(*(json.loads(path.read_text()) for path in args.compare))
        print(json.dumps({'status': 'MATCH_EXECUTION_TRACE_OWED' if matched else 'NUMERICAL_MISMATCH'}))
        return 0 if matched else 1
    if args.out is None:
        parser.error('capture requires --out NEW_DIRECTORY')
    try:
        devices = tuple(int(part) for part in (args.devices or '0,1').split(','))
    except ValueError:
        parser.error('devices must be comma-separated integers')
    if len(devices) != 2 or len(set(devices)) != 2 or min(devices) < 0:
        parser.error('this gate requires exactly two distinct nonnegative device indices')
    out = args.out.resolve()
    if out.exists():
        parser.error('out already exists; preserve previous capture')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    os.environ['MOJOLEARN_NUMERIC_MODE'] = 'identical'
    try:
        import numpy as np
    except ImportError:
        parser.error('CV capture requires NumPy; install the optional mojolearn[test] extra')
    import mojolearn as ml
    from . import _backend
    from .parallel_model_selection import cross_val_score
    if _backend.numeric_mode() != 'identical':
        parser.error('CV capture requires identical numeric mode in a fresh process')
    if ml.vendor() not in ('cuda', 'hip') or (args.require_backend and ml.vendor() != args.require_backend):
        parser.error('requested GPU backend is not active; CPU fallback is not evidence')
    package = Path(ml.__file__).resolve().parent
    source = source_identity()
    dist, distribution = distribution_identity(package)
    native = _backend.binding('_mojolearn_gbdt', 'identical')
    binding_path = Path(native.__file__).resolve()
    bindings = {'_mojolearn_gbdt': {'path': str(binding_path), 'vendor': _backend.read_vendor(native),
                                  'sha256': hashlib.sha256(binding_path.read_bytes()).hexdigest()}}
    if args.require_installed:
        if not distribution or not distribution['import_matches_distribution'] or distribution['editable']:
            parser.error('import does not match a noneditable installed mojolearn wheel')
        try:
            verify_distribution_records(dist, package, source['source_sha256'], bindings)
        except ValueError as exc:
            parser.error(str(exc))
        distribution['source_record_hashes_match'] = distribution['native_record_hashes_match'] = True
    from ._verify import environment_json
    report = dict(protocol=PROTOCOL, status='INCOMPLETE', source_commit=source['commit'], source=source,
                  package_origin=ml.__file__, distribution=distribution, bindings=bindings, environment=environment_json(),
                  vendor=ml.vendor(), devices=list(devices), repeats=2, folds=5, runs=[], controls=[],
                  physical_execution_trace='OWED', native_fault_controls='OWED',
                  scope='CV numerical, save/reload and placement checks; no capacity, throughput or complete physical execution qualification')
    out.mkdir(parents=True, exist_ok=False)
    def save():
        _atomic_json(out / 'report.json', report)
    save()
    try:
        rng = np.random.default_rng(21871)
        X = rng.normal(size=(257, 6)).astype('<f4')
        X[20:40] = X[:20]  # repeated values plus uneven five-fold partitioning
        report['inputs'] = {'X': array_digest(X)}
        for classifier in (False, True):
            name = 'classifier' if classifier else 'regressor'
            y = (X[:, 0] > X[:, 1]).astype('<i4') if classifier else (
                X[:, 0] * np.float32(.5) + X[:, 1] * np.float32(.25)).astype('<f4')
            report['inputs'][name] = array_digest(y)
            model = (ml.GradientBoostingClassifier if classifier else ml.GradientBoostingRegressor)(
                n_estimators=4, max_depth=3, border_count=16, random_state=7, numeric_mode='identical')
            baseline = scores0 = None
            for order in ((devices[0],), devices, devices[::-1]):
                for repeat in range(2):
                    directory = out / (name + '-' + '-'.join(map(str, order)) + f'-r{repeat}')
                    directory.mkdir()
                    scorer = functools.partial(score_with_witness, directory=str(directory))
                    start = time.monotonic()
                    scores = cross_val_score(model, X, y, devices=order, cv=5, scoring=scorer)
                    records = read_records(directory, 5)
                    if baseline is None:
                        baseline, scores0 = records, scores.tobytes()
                    compare_records(baseline, records, vendor=ml.vendor(), workers=len(order))
                    if scores.tobytes() != scores0:
                        raise ValueError('fold score order or bits changed')
                    report['runs'].append(dict(model=name, devices=list(order), repeat=repeat,
                                               scores_hex=scores.tobytes().hex(), fold_records=records,
                                               total_seconds_including_witnesses=time.monotonic() - start))
                    save()
            # Comparator controls are separate from native computational faults.
            for fault in ('dropped-fold', 'changed-model', 'changed-prediction', 'aliased-device'):
                changed = copy.deepcopy(records)
                key = next(iter(changed))
                if fault == 'dropped-fold': changed.pop(key)
                elif fault == 'changed-model': changed[key]['model'] = '0' * 64
                elif fault == 'changed-prediction': changed[key]['predict'] = '0' * 64
                else:
                    first = changed[key]['inventory']['devices']
                    for record in changed.values(): record['inventory']['devices'] = first
                try:
                    compare_records(baseline, changed, vendor=ml.vendor(), workers=2)
                except (ValueError, RuntimeError):
                    report['controls'].append(name + ':' + fault)
                else:
                    raise AssertionError('comparator failed to catch ' + fault)
        report['status'] = 'NUMERICS_AND_PLACEMENT_PASS'
        validate_receipt(report)
        save()
    except BaseException as exc:
        report.update(status='FAILED', reason=repr(exc))
        save()
        raise
    print(json.dumps({'status': report['status'], 'runs': len(report['runs']), 'out': str(out)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
