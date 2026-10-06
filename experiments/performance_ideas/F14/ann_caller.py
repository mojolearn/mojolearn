#!/usr/bin/env python3
"""Distinct approximate IVF task: compact list chunks at fixed search knobs.

NumPy supplies fixtures and exhaustive quality references only. The public
IVF build and search remain native GPU operations. Exact API claims are not
inferred from approximate recall; full-probe controls are labelled separately.
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'apple_fast'))
from support import binding_check, capture_main, consumed


def exercise(args):
    import numpy as np
    from mojolearn import IVFIndex, _mojolearn_ivf as binding

    cases = {}
    for rows, dim, probes, skew in ((1031, 7, 5, False), (4093, 33, 8, True), (1031, 7, 17, False)):
        rng = np.random.default_rng(141)
        centers = rng.normal(size=(17, dim)).astype('float32') * 3
        labels = rng.integers(0, 17, rows)
        if skew:
            labels[rng.random(rows) < .8] = 0
        x = np.asarray(centers[labels] + rng.normal(size=(rows, dim)) * .3, 'float32')
        q = np.asarray(x[:67] + rng.normal(size=(67, dim)) * .02, 'float32')
        model = IVFIndex(n_lists=17, n_probes=probes, n_neighbors=8,
                         kmeans_n_iters=8, random_state=141, metric='sqeuclidean')
        _, fit_ms = consumed(lambda: (model.fit(x).centers_,))
        index_hash = hashlib.sha256(b''.join(np.asarray(getattr(model, field)).tobytes()
            for field in ('centers_', 'center_norms_', 'list_offsets_', 'list_indices_', 'list_data_'))).hexdigest()
        full = np.sum((q.astype(float)[:, None, :] - x.astype(float)[None, :, :]) ** 2, axis=2)
        for filtered in (False, True):
            keep = np.ones(rows, dtype=bool)
            if filtered:
                keep[::11] = False
            allowed = full.copy()
            allowed[:, ~keep] = np.inf
            reference = np.argsort(allowed, axis=1, kind='stable')[:, :8]
            model.search(q, filter=keep if filtered else None)  # matched warmup
            before = int(binding.ivf_fast_balanced_hits())
            (distance, index), query_ms = consumed(lambda: model.search(q, filter=keep if filtered else None))
            hits = int(binding.ivf_fast_balanced_hits()) - before
            if args.arm == 'B' and hits <= 0:
                raise AssertionError('approximate candidate did not reach bounded list tasks')
            if args.arm == 'A' and hits != 0:
                raise AssertionError('baseline unexpectedly reached candidate')
            distance, index = np.asarray(distance), np.asarray(index)
            if np.any(index < 0) or np.any(index >= rows) or not keep[index].all():
                raise AssertionError('returned excluded or invalid candidate')
            if any(len(set(row.tolist())) != 8 for row in index):
                raise AssertionError('returned duplicate candidates')
            recall = np.mean([len(set(got.tolist()) & set(want.tolist())) / 8
                              for got, want in zip(index, reference)])
            if probes == 17 and recall != 1:
                raise AssertionError('full-probe control lost exhaustive neighbors')
            oracle = np.take_along_axis(full, index, axis=1)
            error = float(np.max(np.abs(distance - oracle)))
            key = f'{rows}-{dim}-probes{probes}-skew{skew}-filter{filtered}'
            cases[key] = dict(
                contract=dict(api='approximate IVF' if probes < 17 else 'IVF full-probe control',
                              rows=rows, dim=dim, probes=probes, lists=17, k=8,
                              iterations=8, seed=141, skew=skew, filtered=filtered,
                              index_sha256=index_hash),
                metrics=dict(recall_error=dict(value=float(1-recall), rtol=0, atol=0),
                             distance_error=dict(value=error, rtol=.1, atol=1e-4)),
                fit_ms=fit_ms, query_ms=query_ms, candidate_hits=hits,
                candidate_counts=np.asarray(model.n_candidates_).tolist(),
                indices=index.tolist(), distances=distance.tolist())
    return dict(binding=binding_check(binding, 'ivf'), cases=cases)


if __name__ == '__main__':
    capture_main(exercise)
