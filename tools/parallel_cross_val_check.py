#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One/two-GPU CV numerical and placement evidence; full execution trace still owed."""
import argparse
import copy
import functools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-backend', choices=('cuda', 'hip'), required=True)
    parser.add_argument('--devices', default='0,1')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    devices = tuple(int(part) for part in args.devices.split(','))
    if len(devices) != 2 or len(set(devices)) != 2 or min(devices) < 0:
        parser.error('this gate requires exactly two distinct nonnegative device indices')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))  # tools scorer must be importable by children too.
    os.environ['MOJOLEARN_NUMERIC_MODE'] = 'identical'
    import numpy as np
    import mojolearn as ml
    from mojolearn.parallel_model_selection import cross_val_score
    from tools.parallel_cv_witness import score_with_witness, read_records, compare_records
    if ml.vendor() != args.require_backend:
        raise RuntimeError('requested GPU backend is not active; CPU fallback is not evidence')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('record from a committed, clean source checkout')
    report = dict(status='INCOMPLETE', source_commit=commit, package_origin=ml.__file__,
                  vendor=ml.vendor(), runs=[], controls=[], physical_execution_trace='OWED',
                  scope='CV source numerical and placement checks only; no installed-wheel, '
                        'capacity, throughput or complete physical execution qualification')
    def save():
        (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    try:
        rng = np.random.default_rng(21871)
        X = rng.normal(size=(257, 6)).astype('<f4')
        X[20:40] = X[:20]  # repeated values plus uneven five-fold partitioning
        for classifier in (False, True):
            name = 'classifier' if classifier else 'regressor'
            y = (X[:, 0] > X[:, 1]).astype('<i4') if classifier else (
                X[:, 0] * np.float32(.5) + X[:, 1] * np.float32(.25)).astype('<f4')
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
                                               scores_hex=scores.tobytes().hex(),
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
        save()
    except BaseException as exc:
        report.update(status='FAILED', reason=repr(exc))
        save()
        raise
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
