#!/usr/bin/env python3
"""Large real-HIGGS public RF IDENTICAL tile A/B, on one NVIDIA GPU.

Uses existing forest benchmark loaders/configuration/arms. Supplied modules must
come from the same source/toolchain, differing only in column-tile definitions.
Route probes run in separate processes because RF_LAUNCH_LOG is latched.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np
import forest_speed_arm as forest
from nvidia_identical_trees import digest_arrays

VARIANTS = ('reference', 'columns2', 'columns4')


def load_arm(path, name, config, data):
    spec = importlib.util.spec_from_file_location('tile_' + name + '._mojolearn_rf', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.rf_numeric_mode() != 1 or module.rf_vendor() != 'cuda':
        raise RuntimeError('requires actual IDENTICAL CUDA binding: ' + str(path))
    arm = forest.our_rf_arm('rf', config, data)
    original_fit = arm.fit
    def fit(model, current_data):
        # _refresh_config clears instance attributes, so bind at the class
        # temporarily for this blocking fit and subsequent scoring.
        cls = type(model)
        old = cls._bind
        cls._bind = lambda self, name=None: module
        try:
            return original_fit(model, current_data)
        finally:
            cls._bind = old
    arm.fit = fit
    score = arm.score
    def score_bound(model, current_data):
        cls = type(model)
        old = cls._bind
        cls._bind = lambda self, name=None: module
        try:
            return score(model, current_data)
        finally:
            cls._bind = old
    arm.score = score_bound
    arm.name = name
    return arm


def fingerprint(model):
    return digest_arrays([(name, getattr(model, name)) for name in
        ('_offsets', '_colid', '_quesval', '_left_child', '_leaves')])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bindings', type=Path, required=True,
                        help='directory containing reference/columns2/columns4/_mojolearn_rf.so')
    parser.add_argument('--rows', type=int, default=1000000)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe-arm', choices=VARIANTS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.rows < 2 or args.rounds < 1:
        parser.error('rows>=2 and rounds>=1 required')
    performance_eligible = args.rows >= 1000000 and args.rounds >= 5
    if not performance_eligible:
        print('WARNING: reduced rows/rounds are smoke only, not performance evidence', flush=True)
    if os.environ.get('MOJOLEARN_NUMERIC_MODE') != 'identical':
        parser.error('set MOJOLEARN_NUMERIC_MODE=identical')
    if os.environ.get('MOJOLEARN_SPEED_FORTRAN'):
        parser.error('unset MOJOLEARN_SPEED_FORTRAN; packing belongs inside fit')
    if not args.probe_arm and os.environ.get('RF_LAUNCH_LOG'):
        parser.error('unset RF_LAUNCH_LOG for timed process')
    if os.environ.get('MOJOLEARN_BUILD_LOCK_HELD') != '1':
        parser.error('run through tools/with_build_lock.sh')
    paths = {v: (args.bindings / v / '_mojolearn_rf.so').resolve() for v in VARIANTS}
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.probe_arm and (args.output / 'fits.jsonl').exists():
        parser.error('choose a fresh output directory; existing evidence is preserved')
    if not args.probe_arm:
        # Probe binaries on actual HIGGS, then launch-free timing in this process.
        for variant in VARIANTS:
            log = args.output / (variant + '.launches.log')
            log.write_text('')
            env = dict(os.environ, RF_LAUNCH_LOG=str(log.resolve()))
            command = [sys.executable, str(Path(__file__).resolve()),
                '--bindings', str(args.bindings.resolve()), '--rows', str(args.rows),
                '--rounds', str(args.rounds), '--output', str(args.output.resolve()),
                '--probe-arm', variant]
            with (args.output / (variant + '.probe.log')).open('w') as stream:
                subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            lines = log.read_text().splitlines()
            tiled = [line for line in lines if line.startswith('histogram_binned_columns')]
            if variant == 'reference':
                assert not tiled, 'reference unexpectedly uses column tiling'
            else:
                assert tiled and all(line.startswith('histogram_binned_' + variant + '_') for line in tiled), 'wrong/missing tile route'
    spec = forest.spec
    data = spec.load_dataset('higgs', 'shipped', args.rows)
    assert data.X_train.shape[0] == args.rows, 'loader did not supply requested training rows'
    forest.prepare_our_inputs(data)
    config = spec.lane_config('rf', 'shipped')
    if args.probe_arm:
        config['n_estimators'] = 1  # Reachability only, never performance evidence.
        arm = load_arm(paths[args.probe_arm], args.probe_arm, config, data)
        model = arm.make()
        arm.fit(model, data)
        arm.sync()
        print('PROBE_PASS', args.probe_arm, fingerprint(model))
        return 0
    arms = {v: load_arm(paths[v], v, config, data) for v in VARIANTS}
    records = []
    expected = None
    def run(name, phase, round_index):
        nonlocal expected
        arm = arms[name]
        start = time.perf_counter()
        model = arm.make()
        arm.fit(model, data)
        arm.sync()
        ms = (time.perf_counter() - start) * 1000
        model_hash = fingerprint(model)
        scores = arm.score(model, data)
        assert scores and all(np.isfinite(value) for _, value, _ in scores)
        vectors = [(metric, vector) for metric, _, vector in scores if vector is not None]
        assert vectors and all(np.isfinite(vector).all() for _, vector in vectors)
        identity = (model_hash, digest_arrays(vectors))
        if expected is None:
            expected = identity
        assert identity == expected, (phase, name, 'model/prediction mismatch')
        record = dict(arm=name, phase=phase, round=round_index, fit_ms=ms,
            model_sha256=identity[0], prediction_sha256=identity[1],
            scores={metric: value for metric, value, _ in scores})
        records.append(record)
        with (args.output / 'fits.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
    (args.output / 'fits.jsonl').write_text('')
    for v in VARIANTS:
        run(v, 'warmup', -1)
    for i in range(args.rounds):
        for v in (VARIANTS if i % 2 == 0 else VARIANTS[::-1]):
            run(v, 'timed', i)
    samples = {v: [r['fit_ms'] for r in records if r['arm'] == v and r['phase'] == 'timed'] for v in VARIANTS}
    spread = {v: max(values) / min(values) for v, values in samples.items()}
    medians = {v: statistics.median(values) for v, values in samples.items()}
    result = dict(numeric_mode='identical', vendor='cuda', rows=args.rows,
        config=config, records=records, spread=spread, median_ms=medians,
        performance_eligible=performance_eligible,
        timing_valid=performance_eligible and all(value <= 1.10 for value in spread.values()),
        candidate_over_reference={v: medians[v] / medians['reference'] for v in VARIANTS[1:]},
        expected_definitions={'reference': [], 'columns2': ['MOJOLEARN_RF_HIST_COLUMNS2=1'],
                              'columns4': ['MOJOLEARN_RF_HIST_COLUMNS4=1']},
        binary_sha256={v: hashlib.sha256(p.read_bytes()).hexdigest() for v, p in paths.items()},
        data_sha256=digest_arrays([(k, getattr(data, k)) for k in ('X_train', 'y_train', 'X_test', 'y_test')]),
        contract='constructor+packing+upload+full fit+sync; scoring/hash excluded; separate route probes')
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print('RF_HIGGS_COLUMNS_PASS timing_valid=', result['timing_valid'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
