#!/usr/bin/env python3
"""Bounded public CPU/GPU-table routing/bit probe; root executes after compile."""
from contextlib import contextmanager
import argparse
import json
import os
from pathlib import Path
import traceback


def run():
    import numpy as np
    from mojolearn import (OneClassSVM, LocalOutlierFactor, NearestCentroid,
                          KNNImputer, LabelPropagation, PageRank, _backend)
    binding = _backend.binding('_mojolearn_x_neighbors', 'identical')
    old_entries = {name: getattr(binding, name) for name in dir(binding) if name.startswith('xn_') or name in ('x_neighbors_ocsvm_alpha_init', 'x_neighbors_unit_ff', 'x_neighbors_ocsvm_resident_idn')}
    @contextmanager
    def arm(enabled):
        os.environ['MOJOLEARN_CLASSICAL_NATIVE_DISPATCH'] = str(enabled)
        def refused(*args, **kwargs):
            raise AssertionError('packed operation redispatched through a Python-facing entry')
        if enabled:
            for name in old_entries:
                setattr(binding, name, refused)
        try:
            yield
        finally:
            if enabled:
                for name, value in old_entries.items():
                    setattr(binding, name, value)
    rng = np.random.default_rng(195)
    x = rng.normal(size=(18, 4)).astype('f4')
    y = (x[:, 0] > 0).astype('i4')
    missing = x.copy(); missing[::4, 1] = np.nan
    semi = y.copy(); semi[2:] = -1
    graph = np.abs(rng.normal(size=(8, 8))).astype('f4')
    np.fill_diagonal(graph, 0)
    cases = [
        ('OneClassSVM', lambda: OneClassSVM(kernel='rbf', gamma=.25, max_iter=12), x, None,
         ('support_vectors_', 'dual_coef_'), ('decision_function', 'predict')),
        ('LocalOutlierFactor', lambda: LocalOutlierFactor(n_neighbors=3, novelty=True), x, None,
         ('negative_outlier_factor_',), ('decision_function', 'predict')),
        ('NearestCentroid', NearestCentroid, x, y, ('centroids_',), ('predict',)),
        ('KNNImputer', lambda: KNNImputer(n_neighbors=3), missing, None, (), ('transform',)),
        ('LabelPropagation', lambda: LabelPropagation(gamma=.25, max_iter=5), x, semi,
         ('transduction_',), ('predict_proba', 'predict')),
        ('PageRank', lambda: PageRank(max_iter=5), graph, None, ('pagerank_',), ()),
    ]
    def same(a,b):
        a,b=np.asarray(a),np.asarray(b)
        assert a.dtype==b.dtype and a.shape==b.shape
        assert a.tobytes()==b.tobytes()
    records=[]
    for name,make,data,target,attributes,methods in cases:
        try:
            models=[make(),make()]
            for enabled,model in enumerate(models):
                model.numeric_mode='identical'
                with arm(enabled):
                    model.fit(data) if target is None else model.fit(data,target)
            for attr in attributes:
                same(getattr(models[0],attr),getattr(models[1],attr))
            for method in methods:
                outputs=[]
                for enabled,model in enumerate(models):
                    with arm(enabled):
                        outputs.append(getattr(model,method)(data))
                same(*outputs)
            records.append(dict(case=name,status='passed'))
        except Exception:
            records.append(dict(case=name,status='failed',error=traceback.format_exc()))
    return dict(scope='77 packed operations through shared provider; selected public cases',
                records=records,full_runtime_migration_qualified=False,
                pending=['remaining kernel-feature/custom capability entries','full metadata admission','all algorithms/refusals',
                         'CPU runtime unless explicitly selected','other devices','performance'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    try: report=run()
    except Exception: report=dict(fatal=traceback.format_exc())
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(1 if 'fatal' in report or any(r['status']!='passed' for r in report['records']) else 0)
