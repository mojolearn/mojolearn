#!/usr/bin/env python3
"""Full saved-dataset F04/F19 paired resampling; benchmark glue only.

Recipe: tools/bench_board_algos.py algos/resample, with its capped regression
block replaced by the complete saved training split, and seed 109 retained from
F04/caller.py. Both X and y are sampled with replacement, n_samples=None.
Cold and repeated complete calls include input preparation, index generation,
uploads, both gathers, native synchronization and host copies of every output.
Disk loading and evidence/quality computations occur outside those intervals.

F19's permutation variant is not admitted here: its retained baseline uses the
public NumPy gather fallback, contrary to that card's no-hybrid quality gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


BOUNDARY = ('host input preparation + public paired resample index draw and both gathers '
            '+ native synchronization + host copies; disk load and hashes excluded')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def saved_array_hash(array):
    """Match classical_two_datasets.py's saved-array identity schema."""
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(array.shape).encode())
    h.update(memoryview(array).cast('B'))
    return h.hexdigest()


def validate_dataset(x, y, meta):
    if meta.get('smoke_max_rows') or len(x) != meta.get('fit_rows_available'):
        raise ValueError('Expected complete uncapped saved training split')
    if meta.get('fit_rows') != [0, len(x)] or len(y) != len(x):
        raise ValueError('Expected complete contiguous paired training rows')
    if x.ndim != 2 or y.ndim != 1 or len(x) < 1:
        raise ValueError('Expected nonempty X matrix and y vector')
    for name, array in (('X', x), ('y', y)):
        spec = meta['arrays'][name]
        if (list(array.shape) != spec['shape'] or str(array.dtype) != spec['dtype']
                or not array.flags.c_contiguous or saved_array_hash(array) != spec['sha256']):
            raise ValueError('Saved dataset array differs: ' + name)


def check_draw(x, y, indices, outputs, np):
    """Quality oracle outside the operation timer, covering every output row."""
    if indices.shape != (len(x),) or np.any(indices < 0) or np.any(indices >= len(x)):
        raise ValueError('Invalid full-length row-index draw')
    if outputs[0].shape != x.shape or outputs[1].shape != y.shape:
        raise ValueError('Public output omitted rows or columns')
    # Bound temporary reference gathers to approximately 64 MiB for any width;
    # this affects only the untimed oracle, never product dispatch or workload.
    batch_rows = max(1, (64 * 1024 * 1024) // max(1, x.shape[1] * x.dtype.itemsize))
    for start in range(0, len(x), batch_rows):
        stop = min(start + batch_rows, len(x))
        selected = indices[start:stop]
        if not np.array_equal(outputs[0][start:stop], x[selected]):
            raise ValueError('Feature gather differs from exact public index contract')
        if not np.array_equal(outputs[1][start:stop], y[selected]):
            raise ValueError('Label gather differs from exact public index contract')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--idea', choices=['F04', 'F19'], required=True)
    parser.add_argument('--arm', choices=['A', 'B'], required=True)
    parser.add_argument('--phase', choices=['warmup', 'scored'], required=True)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--dataset-sha256', required=True,
                        help='SHA256 of saved sidecar recipe including all array hashes')
    parser.add_argument('--baseline-result', type=Path,
                        help='B baseline JSON; defaults to sibling PHASE-A.json written by the serial queue')
    args = parser.parse_args()
    if args.arm == 'B' and args.baseline_result is None:
        args.baseline_result = args.output.with_name(args.phase + '-A.json')
    if os.environ.get('MOJOLEARN_VENDOR') != 'apple' or os.environ.get('MOJOLEARN_NUMERIC_MODE') != 'fast':
        raise RuntimeError('Explicit Apple FAST execution required')
    chip = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    if 'Apple M3 Ultra' not in chip:
        raise RuntimeError('Full FAST timing requires the retained Apple M3 Ultra')
    import numpy as np
    from mojolearn.resample import resample, resample_indices
    from mojolearn import _mojolearn_resample as binding
    from bench_board_state import canonical_hash

    sidecar = args.data.with_suffix('.json')
    if file_hash(sidecar) != args.dataset_sha256:
        raise ValueError('Saved dataset recipe hash differs')
    meta = json.loads(sidecar.read_text())
    with np.load(args.data, allow_pickle=False) as arrays:
        x, y = arrays['X'], arrays['y']
    validate_dataset(x, y, meta)
    if int(binding.resample_numeric_mode()) != 0 or str(binding.resample_vendor()) != 'metal':
        raise RuntimeError('Loaded binding is not Apple FAST')
    if int(binding.resample_gpu_gather_enabled()) != 1:
        raise RuntimeError('Retained artifact lacks GPU gather')
    binding_path = Path(binding.__file__).resolve()
    binding_sha = file_hash(binding_path)
    original = binding.resample_gather_gpu
    route_results = []

    def observe(addresses, params):
        result = original(addresses, params)
        route_results.append(int(result))
        return result

    binding.resample_gather_gpu = observe
    durations = {}
    retained = []
    try:
        for label in ('cold', 'repeated'):
            before_calls = len(route_results)
            start = time.perf_counter_ns()
            prepared_x = np.ascontiguousarray(x, dtype=np.float32)
            prepared_y = np.ascontiguousarray(y, dtype=np.float32)
            prepared = time.perf_counter_ns()
            values = resample(prepared_x, prepared_y, replace=True, n_samples=None,
                              random_state=109, numeric_mode='fast')
            # Copy every element, making output consumption part of the operation.
            consumed = tuple(np.array(value, copy=True) for value in values)
            end = time.perf_counter_ns()
            if route_results[before_calls:] != [1]:
                raise RuntimeError('GPU paired gather did not execute; fallback is unqualified')
            durations[label + '_operation_seconds'] = (end - start) / 1e9
            durations[label + '_preparation_seconds'] = (prepared - start) / 1e9
            durations[label + '_draw_gather_consume_seconds'] = (end - prepared) / 1e9
            retained.append(consumed)
    finally:
        binding.resample_gather_gpu = original

    # Capture/quality after measured work, with no separate device preverification.
    indices = np.asarray(resample_indices(len(x), n_samples=None, replace=True,
                                         random_state=109, numeric_mode='fast'))
    for outputs in retained:
        check_draw(x, y, indices, outputs, np)
    cold_hash = canonical_hash(dict(X=retained[0][0], y=retained[0][1]))
    repeated_hash = canonical_hash(dict(X=retained[1][0], y=retained[1][1]))
    if cold_hash != repeated_hash:
        raise RuntimeError('Repeated operation changed seeded output')
    indices_hash = canonical_hash(indices)
    dimensions = dict(train=list(x.shape), labels=list(y.shape))
    baseline_match = None
    if args.baseline_result is not None:
        baseline = json.loads(args.baseline_result.read_text())
        expected = dict(status='PASS', idea=args.idea, arm='A', phase=args.phase,
                        source_sha=args.source_sha, dataset_sha256=args.dataset_sha256,
                        mode='fast', vendor='apple', dimensions=dimensions,
                        timed_boundary=BOUNDARY, output_sha256=cold_hash,
                        repeated_output_sha256=repeated_hash, indices_sha256=indices_hash)
        if any(baseline.get(key) != value for key, value in expected.items()):
            raise ValueError('A/B result provenance, exact indices or complete output differs')
        baseline_match = True
    packet = dict(schema='mojolearn.full-ab-result/1', status='PASS', idea=args.idea,
                  variant='default', arm=args.arm, phase=args.phase, source_sha=args.source_sha,
                  dataset_sha256=args.dataset_sha256, mode='fast', vendor='apple',
                  dimensions=dimensions, full_dataset_coverage=True, timed_boundary=BOUNDARY,
                  timings=dict(full_operation_seconds=durations['cold_operation_seconds'], **durations),
                  model_state=dict(status='UNAVAILABLE', reason='Stateless resampling has no fitted model'),
                  output_sha256=cold_hash, repeated_output_sha256=repeated_hash,
                  indices_sha256=indices_hash, loaded_artifacts={str(binding_path): binding_sha},
                  dataset=dict(path=str(args.data), recipe=meta, intrinsic_caps=[],
                               original_board_regression_cap_removed=True),
                  settings=dict(replace=True, n_samples=None, effective_n_samples=len(x),
                                random_state=109, paired_arrays=['X', 'y']),
                  recipe_sources=['tools/bench_board_algos.py:resample',
                                  'experiments/performance_ideas/F04/caller.py'],
                  recipe_changes=['Use complete saved train split instead of capped board regression block',
                                  'Preserve candidate caller seed109 instead of board seed7'],
                  reach=dict(gpu_paired_gather_returncodes=route_results),
                  quality=dict(exact_gather_matches_public_indices=True,
                               repeated_output_exact=True, baseline_exact_match=baseline_match),
                  timing_scope=dict(fit='not applicable: stateless operation', inference='not applicable',
                                    cold_samples=1, repeated_samples=1),
                  promotion=False, acceptance='Pending complete affected-workload/combination coverage')
    args.output.write_text(json.dumps(packet, indent=2, allow_nan=False) + '\n')
    print('FULL_RESAMPLE idea=' + args.idea + ' arm=' + args.arm + ' phase=' + args.phase
          + ' rows=' + str(len(x)) + ' output=' + str(args.output), flush=True)


if __name__ == '__main__':
    main()
