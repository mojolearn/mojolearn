#!/usr/bin/env python3
"""Full saved-dataset F14 exact brute-force neighbor A/B measurement.

Preserve the original F14 caller's k values (5, 11, 13), Euclidean metric,
brute-force algorithm, automatic query tile, and distance/index tie semantics.
Use every training and saved query row. The grouped-query kernel only applies
to <=32 features and <=16 neighbors; reject other workloads as no distinct
candidate route, rather than altering dimensions or k to make them applicable.
The retained pair must also explicitly disable the preceding FAST MMA route;
without MOJOLEARN_KNN_FAST_MMA_OFF the grouped-query path is shadowed.

Existing native stage-time lines witness fused top-k execution. They are not a
query-group counter: the accepted binding receipt establishes the group define.
No compilation, preliminary device validation, synthetic input or opponent runs.
"""
import argparse
from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from performance_full_resample import file_hash, saved_array_hash

BOUNDARY = ('host input preparation + public fit + all saved queries kneighbors '
            '+ native synchronization + host copies; disk load and hashes excluded')


def validate_inputs(x, query, meta, k):
    if meta.get('smoke_max_rows') or len(x) != meta.get('fit_rows_available'):
        raise ValueError('Expected complete uncapped saved training split')
    if meta.get('fit_rows') != [0, len(x)]:
        raise ValueError('Expected complete contiguous training rows')
    for name, array in (('X', x), ('Xq', query)):
        spec = meta['arrays'][name]
        if (list(array.shape) != spec['shape'] or str(array.dtype) != spec['dtype']
                or not array.flags.c_contiguous or saved_array_hash(array) != spec['sha256']):
            raise ValueError('Saved full dataset array differs: ' + name)
    if x.ndim != 2 or query.ndim != 2 or x.shape[1] != query.shape[1] or len(query) < 1:
        raise ValueError('Expected nonempty compatible train/query matrices')
    # These are the compiled kernel's documented register-capacity guards,
    # not benchmark shape choices. Preserve all features and report no route.
    if not (1 <= x.shape[1] <= 32 and 1 <= k <= 16):
        raise ValueError('NO_DISTINCT_CANDIDATE_ROUTE: grouped queries require features<=32 and k<=16')


def validate_artifact_controls(defines, arm):
    def enabled(name):
        return any(value.split('=')[0] == name and value.split('=')[-1] != '0'
                   for value in defines)
    group = enabled('MOJOLEARN_KNN_FAST_QUERY_GROUP4')
    if group != (arm == 'B'):
        raise ValueError('Accepted artifact defines do not describe F14 A/B controls')
    # In the retained source, fast_mma_knn is dispatched first and covers all
    # feature/k combinations admitted by fast_topk_knn. Shape eligibility alone
    # therefore cannot establish candidate reach. Never time this no-op pair.
    if not enabled('MOJOLEARN_KNN_FAST_MMA_OFF'):
        raise ValueError('NO_DISTINCT_CANDIDATE_ROUTE: preceding FAST MMA shadows grouped queries; retained pair must explicitly disable it')
    return group


@contextmanager
def native_log(path):
    """Retain existing native dispatch diagnostics without sending them to chat."""
    libc = ctypes.CDLL(None)
    sys.stdout.flush()
    libc.fflush(None)
    original = os.dup(1)
    with Path(path).open('xb') as stream:
        try:
            os.dup2(stream.fileno(), 1)
            yield
        finally:
            libc.fflush(None)
            os.dup2(original, 1)
            os.close(original)


def dispatch_evidence(path):
    count, examples = 0, []
    with Path(path).open(errors='replace') as stream:
        for line in stream:
            if line.startswith('FAST_TOPK_KNN '):
                count += 1
                if len(examples) < 8:
                    examples.append(line.strip())
    return dict(native_dispatch_records=count, examples=examples, log=str(path),
                log_sha256=file_hash(path),
                interpretation='Actual fused top-k dispatch; group width comes from accepted artifact defines')


def check_output(distances, indices, rows, queries, k, np):
    if distances.shape != (queries, k) or indices.shape != (queries, k):
        raise ValueError('Output omitted queries or neighbors')
    if not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError('Nonfinite or negative neighbor distance')
    if np.any(indices < 0) or np.any(indices >= rows):
        raise ValueError('Neighbor index outside complete training split')
    if np.any(distances[:, 1:] < distances[:, :-1]):
        raise ValueError('Nearest-neighbor distances are not sorted')
    # Compare exact A/B indices later. Rounded sqrt distances may coincide even
    # when squared selection keys differ; do not invent a tie reorder here.
    if np.any(np.diff(np.sort(indices, axis=1), axis=1) == 0):
        raise ValueError('Duplicate selected neighbor within one query')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=['A', 'B'], required=True)
    parser.add_argument('--phase', choices=['warmup', 'scored'], required=True)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--dataset-sha256', required=True)
    parser.add_argument('--neighbors', type=int, choices=[5, 11, 13], default=11)
    parser.add_argument('--artifact-defines-json', required=True,
                        help='Accepted binding receipt defines, checked against the paired experiment')
    parser.add_argument('--baseline-result', type=Path)
    args = parser.parse_args()
    if args.arm == 'B' and args.baseline_result is None:
        args.baseline_result = args.output.with_name(args.phase + '-A.json')
    defines = json.loads(args.artifact_defines_json)
    group_define = validate_artifact_controls(defines, args.arm)
    if os.environ.get('MOJOLEARN_VENDOR') != 'apple' or os.environ.get('MOJOLEARN_NUMERIC_MODE') != 'fast':
        raise RuntimeError('Explicit Apple FAST mode required')
    chip = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    if 'Apple M3 Ultra' not in chip:
        raise RuntimeError('Full FAST timing requires retained M3 Ultra')
    # This uses already compiled diagnostic output during the scored operation.
    os.environ['MOJOLEARN_STAGE_TIMES'] = '1'
    import numpy as np
    from mojolearn import NearestNeighbors, _mojolearn as binding
    from bench_board_state import canonical_hash, model_receipt
    sidecar = args.data.with_suffix('.json')
    if file_hash(sidecar) != args.dataset_sha256:
        raise ValueError('Saved full dataset recipe hash differs')
    meta = json.loads(sidecar.read_text())
    with np.load(args.data, allow_pickle=False) as arrays:
        x, query = arrays['X'], arrays['Xq']
    validate_inputs(x, query, meta, args.neighbors)
    if int(binding.mojolearn_numeric_mode()) != 0 or str(binding.mojolearn_vendor()) != 'metal':
        raise RuntimeError('Loaded binding is not Apple FAST')
    binding_path = Path(binding.__file__).resolve()
    binding_sha = file_hash(binding_path)
    cold_log = args.output.with_suffix('.cold-native.log')
    repeated_log = args.output.with_suffix('.repeated-native.log')
    with native_log(cold_log):
        start = time.perf_counter_ns()
        prepared_x = np.ascontiguousarray(x, dtype=np.float32)
        prepared_query = np.ascontiguousarray(query, dtype=np.float32)
        model = NearestNeighbors(n_neighbors=args.neighbors, metric='euclidean', algorithm='brute', p=2)
        prepared = time.perf_counter_ns()
        model.fit(prepared_x)
        fitted = time.perf_counter_ns()
        distances, indices = (np.array(value, copy=True) for value in model.kneighbors(prepared_query))
        finished = time.perf_counter_ns()
    with native_log(repeated_log):
        repeat_start = time.perf_counter_ns()
        repeated_query = np.ascontiguousarray(query, dtype=np.float32)
        repeated_distances, repeated_indices = (np.array(value, copy=True)
                                                for value in model.kneighbors(repeated_query))
        repeat_end = time.perf_counter_ns()
    dispatch = dict(cold=dispatch_evidence(cold_log), repeated=dispatch_evidence(repeated_log))
    if any(evidence['native_dispatch_records'] < 1 for evidence in dispatch.values()):
        raise RuntimeError('Expected actual fused top-k native dispatch evidence is absent')
    for d, i in ((distances, indices), (repeated_distances, repeated_indices)):
        check_output(d, i, len(x), len(query), args.neighbors, np)
    output_hash = canonical_hash(dict(distances=distances, indices=indices))
    repeat_hash = canonical_hash(dict(distances=repeated_distances, indices=repeated_indices))
    if output_hash != repeat_hash:
        raise RuntimeError('Repeated seeded-independent exact query output differs')
    distance_hash, index_hash = canonical_hash(distances), canonical_hash(indices)
    state = model_receipt(model)
    model_state = (dict(status='CAPTURED', sha256=state['sha256'], scope='public NearestNeighbors.save numerical state')
                   if state.get('status') == 'ok' else
                   dict(status='UNAVAILABLE', reason=state.get('reason', state.get('error', 'public export unavailable'))))
    dimensions = dict(train=list(x.shape), query=list(query.shape))
    settings = dict(n_neighbors=args.neighbors, algorithm='brute', metric='euclidean', p=2, query_tile=0)
    baseline_match = None
    if args.baseline_result is not None:
        baseline = json.loads(args.baseline_result.read_text())
        expected = dict(status='PASS', idea='F14', arm='A', phase=args.phase,
                        source_sha=args.source_sha, dataset_sha256=args.dataset_sha256,
                        mode='fast', vendor='apple', dimensions=dimensions, settings=settings,
                        timed_boundary=BOUNDARY, output_sha256=output_hash,
                        repeated_output_sha256=repeat_hash, indices_sha256=index_hash,
                        distances_sha256=distance_hash)
        if any(baseline.get(key) != value for key, value in expected.items()):
            raise ValueError('A/B full query indices/distances or workload provenance differ')
        if baseline.get('model_state') != model_state:
            raise ValueError('A/B public fitted state differs')
        baseline_match = True
    packet = dict(schema='mojolearn.full-ab-result/1', status='PASS', idea='F14', variant='default',
                  arm=args.arm, phase=args.phase, source_sha=args.source_sha,
                  dataset_sha256=args.dataset_sha256, mode='fast', vendor='apple',
                  dimensions=dimensions, full_dataset_coverage=True, timed_boundary=BOUNDARY,
                  timings=dict(full_operation_seconds=(finished-start)/1e9,
                               preparation_seconds=(prepared-start)/1e9, fit_seconds=(fitted-prepared)/1e9,
                               cold_query_seconds=(finished-fitted)/1e9,
                               repeated_query_seconds=(repeat_end-repeat_start)/1e9),
                  model_state=model_state, output_sha256=output_hash, repeated_output_sha256=repeat_hash,
                  indices_sha256=index_hash, distances_sha256=distance_hash,
                  loaded_artifacts={str(binding_path): binding_sha},
                  accepted_artifact_defines=defines, settings=settings,
                  dataset=dict(path=str(args.data), recipe=meta, intrinsic_caps=[]),
                  query_pairs_per_call=len(x)*len(query), native_dispatch=dispatch,
                  grouping=dict(queries_per_thread_from_artifact=4 if group_define else 2,
                                actual_group_counter='UNAVAILABLE: retained binary exports none'),
                  quality=dict(all_outputs_finite_and_in_range=True, neighbors_unique_and_distance_sorted=True,
                               repeated_output_exact=True, baseline_exact_match=baseline_match,
                               accepted_prior_quality_reused=True,
                               independent_full_fp64_oracle='not rerun; exact complete A/B output comparison retained'),
                  recipe_sources=['experiments/performance_ideas/F14/caller.py', 'saved full classical train/query split'],
                  recipe_changes=['All saved train/query rows replace synthetic caller fixtures; original k and API preserved'],
                  promotion=False, acceptance='Pending other affected full workloads and original k coverage')
    args.output.write_text(json.dumps(packet, indent=2, allow_nan=False) + '\n')
    print('FULL_KNN arm=' + args.arm + ' phase=' + args.phase + ' k=' + str(args.neighbors)
          + ' rows=' + str(len(x)) + ' queries=' + str(len(query)) + ' output=' + str(args.output), flush=True)


if __name__ == '__main__':
    main()
