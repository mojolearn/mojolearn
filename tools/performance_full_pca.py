#!/usr/bin/env python3
"""Full saved-dataset PCA A/B arm; benchmark glue, never library runtime.

Use separate processes/packages for the frozen arms. The operation includes
input preparation, fitting, transformation, inverse and host consumption.
Dataset disk loading and evidence hashes are outside the operation clock.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Full big-DATASET.npz cache')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=['A', 'B'], required=True)
    parser.add_argument('--phase', choices=['warmup', 'scored'], required=True)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--dataset-sha256', required=True)
    args = parser.parse_args()
    import numpy as np
    from mojolearn import PCA, _mojolearn_estimators
    from bench_board_state import canonical_hash, model_receipt
    meta = json.loads(args.data.with_suffix('.json').read_text())
    with np.load(args.data) as arrays:
        x, query = arrays['X'], arrays['Xq']
    # Existing classical taxi recipes contain a 4M cap; only uncapped input
    # is admitted, never treating --rows full as proof by itself.
    if meta.get('smoke_max_rows') or len(x) != meta['fit_rows_available']:
        raise ValueError('Dataset does not contain the entire declared training split')
    if list(x.shape) != meta['arrays']['X']['shape'] or list(query.shape) != meta['arrays']['Xq']['shape']:
        raise ValueError('Dataset dimensions differ from saved full-workload recipe')
    fit_rows = meta.get('fit_rows')
    if fit_rows != [0, len(x)]:
        raise ValueError('Full contiguous train split is required')
    rank = meta['pca']['n_components']
    binding_path = Path(_mojolearn_estimators.__file__)
    before = [_mojolearn_estimators.scoped_gemm_count(route, arm)
              for route in range(3) for arm in range(3)]
    start = time.perf_counter_ns()
    x = np.ascontiguousarray(x, dtype=np.float32)
    query = np.ascontiguousarray(query, dtype=np.float32)
    model = PCA(n_components=rank, svd_solver='covariance_eigh', whiten=False,
                random_state=7, numeric_mode='fast')
    prepared = time.perf_counter_ns()
    model.fit(x)
    # Consume public fit state; native public calls synchronize their results.
    components = np.array(model.components_, copy=True)
    mean = np.array(model.mean_, copy=True)
    singular = np.array(model.singular_values_, copy=True)
    fit_end = time.perf_counter_ns()
    projected = np.array(model.transform(query), copy=True)
    transform_end = time.perf_counter_ns()
    restored = np.array(model.inverse_transform(projected), copy=True)
    operation_end = time.perf_counter_ns()
    repeat_start = time.perf_counter_ns()
    repeated = np.array(model.transform(query), copy=True)
    repeat_end = time.perf_counter_ns()
    repeated_inverse = np.array(model.inverse_transform(repeated), copy=True)
    repeat_inverse_end = time.perf_counter_ns()
    after = [_mojolearn_estimators.scoped_gemm_count(route, arm)
             for route in range(3) for arm in range(3)]
    reached = [int(right - left) for left, right in zip(before, after)]
    state = dict(components=components, mean=mean, singular_values=singular,
                 explained_variance=np.asarray(model.explained_variance_),
                 explained_variance_ratio=np.asarray(model.explained_variance_ratio_),
                 noise_variance=float(model.noise_variance_))
    export = model_receipt(model)
    # Public fitted-array hash remains explicit even if a model has no save API.
    public_state_hash = canonical_hash(state)
    boundary = 'host input preparation + public fit + full query transform + inverse + host copies; disk load and hashes excluded'
    packet = dict(schema='mojolearn.full-ab-result/1', status='PASS', idea='F01', arm=args.arm, phase=args.phase,
                  dataset_sha256=args.dataset_sha256, mode='fast',
                  dimensions=dict(train=list(x.shape), query=list(query.shape)),
                  full_dataset_coverage=True,
                  model_state=dict(status='CAPTURED', sha256=public_state_hash,
                                   scope='public numerical PCA fitted arrays and noise variance'),
                  loaded_artifacts={str(binding_path): hashlib.sha256(binding_path.read_bytes()).hexdigest()},
                  timings=dict(full_operation_seconds=(operation_end-start)/1e9,
                               preparation_seconds=(prepared-start)/1e9, fit_seconds=(fit_end-prepared)/1e9,
                               cold_transform_seconds=(transform_end-fit_end)/1e9, cold_inverse_seconds=(operation_end-transform_end)/1e9,
                               repeated_transform_seconds=(repeat_end-repeat_start)/1e9, repeated_inverse_seconds=(repeat_inverse_end-repeat_end)/1e9),
                  source_sha=args.source_sha, numeric_mode=model.numeric_mode_used(),
                  vendor=model.vendor_used(), model_sha256=public_state_hash,
                  model_hash_scope='public numerical PCA fitted arrays and noise variance',
                  public_model_export=export if export.get('status') == 'ok' else dict(status='UNAVAILABLE', reason=export.get('reason', 'public export unavailable')), 
                  output_sha256=canonical_hash(dict(projected=projected, restored=restored)),
                  repeated_output_sha256=canonical_hash(dict(projected=repeated, restored=repeated_inverse)),
                  binding=dict(path=str(binding_path), sha256=hashlib.sha256(binding_path.read_bytes()).hexdigest()),
                  dataset=dict(path=str(args.data), recipe=meta, dimensions=dict(train=list(x.shape), query=list(query.shape)),
                               full_dataset_coverage=True, intrinsic_cap=None),
                  settings=dict(n_components=rank, svd_solver='covariance_eigh', whiten=False, random_state=7),
                  timing_ms=dict(preparation=(prepared-start)/1e6, fit=(fit_end-prepared)/1e6,
                                 cold_transform=(transform_end-fit_end)/1e6,
                                 cold_inverse=(operation_end-transform_end)/1e6,
                                 operation=(operation_end-start)/1e6,
                                 repeated_transform=(repeat_end-repeat_start)/1e6,
                                 repeated_inverse=(repeat_inverse_end-repeat_end)/1e6),
                  timed_boundary=boundary,
                  reach=reached, candidate_reached=any(reached[i] for i in (1, 2, 4, 5, 7, 8)),
                  quality=dict(fitted_state_finite=bool(all(np.isfinite(v).all() for v in (components,mean,singular))),
                               output_finite=bool(np.isfinite(restored).all()),
                               reconstruction_squared_error=float(np.sum((restored.astype(np.float64)-query)**2)),
                               query_squared_norm=float(np.sum(query.astype(np.float64)**2))),
                  promotion=False, acceptance='pending complete A/B quality and full affected-workload coverage')
    if packet['numeric_mode'] != 'fast' or packet['vendor'] != 'apple':
        raise RuntimeError('Unexpected measured route: ' + str((packet['numeric_mode'],packet['vendor'])))
    if not packet['quality']['fitted_state_finite'] or not packet['quality']['output_finite'] or (args.arm == 'B' and not packet['candidate_reached']):
        packet['status'] = 'FAIL'
        packet['quality_status'] = 'FAIL'
    args.output.write_text(json.dumps(packet, indent=2, allow_nan=False)+'\n')
    print('FULL_PCA arm='+args.arm+' phase='+args.phase+' rows='+str(len(x))+' output='+str(args.output), flush=True)


if __name__ == '__main__':
    main()
