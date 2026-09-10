#!/usr/bin/env python3
"""Focused NVIDIA IDENTICAL baselines using the existing forest benchmark arms.

RF: cuML default streams and one stream. Symmetric GBDT: CatBoost GPU.
ET: our baseline only; no claim that cuML RF is an equivalent ET learner.
Run under an externally bounded GPU lease/timeout. No dataset fallback.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import statistics
import sys

import numpy as np
import forest_speed_arm as forest


def digest_arrays(arrays):
    digest = hashlib.sha256()
    for name, value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class Tee(io.StringIO):
    def write(self, value):
        sys.__stdout__.write(value)
        sys.__stdout__.flush()
        return super().write(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane', choices=['rf', 'et', 'gbdt-symmetric'], required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--rows', type=int, required=True)
    parser.add_argument('--size', choices=['smoke', 'shipped'], default='shipped')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--symmetric-profile',
                        choices=['matched-no-noise', 'native-defaults'],
                        default='matched-no-noise',
                        help='matched-no-noise sets CatBoost random_strength=0, matching ours; '
                             'native-defaults retains the historical mismatch')
    parser.add_argument('--nvtx', action='store_true',
                        help='diagnostic NVTX fit ranges for Nsight Systems; rerun without profiling for timing claims')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 1 or args.rows < 1:
        parser.error('rounds and rows must be positive')
    if os.environ.get('MOJOLEARN_NUMERIC_MODE') != 'identical':
        parser.error('set MOJOLEARN_NUMERIC_MODE=identical explicitly')
    if os.environ.get('MOJOLEARN_SPEED_EXPECTED_VENDOR') != 'cuda':
        parser.error('set MOJOLEARN_SPEED_EXPECTED_VENDOR=cuda explicitly')
    if forest._fortran_requested():
        parser.error("leave MOJOLEARN_SPEED_FORTRAN unset: this run includes host packing")
    import cupy
    cupy.cuda.runtime.deviceSynchronize()  # Fail if no real CUDA runtime.
    @contextlib.contextmanager
    def fit_range(arm_name, round_index):
        cupy.cuda.nvtx.RangePush(
            "mojolearn-fit/%s/%s/round-%d" % (args.lane, arm_name, round_index))
        try:
            yield
        finally:
            cupy.cuda.nvtx.RangePop()

    spec = forest.spec
    data = spec.load_dataset(args.dataset, args.size, args.rows)
    config = spec.lane_config(args.lane, args.size)
    spec.prepare_cuml_labels(data)
    forest.prepare_our_inputs(data)
    ours = forest.OUR_BUILDERS[args.lane](args.lane, config, data)
    witness = forest.verify_our_arm(ours)
    arms = [ours]
    if args.lane == 'rf':
        default = spec.cuml_rf_arm('rf', config, data)
        single = spec.cuml_rf_arm('rf', config, data)
        make_single = single.make
        single.make = lambda: make_single().set_params(n_streams=1)
        single.name = 'cuml-rf-gpu-stream1'
        arms += [default, single]
    elif args.lane == 'gbdt-symmetric':
        competitors = spec.catboost_arms(args.lane, config, data, ['gpu'])
        if args.symmetric_profile == 'matched-no-noise':
            for arm in competitors:
                original_make = arm.make
                arm.make = lambda make=original_make: make().set_params(random_strength=0.0)
        arms += competitors

    model_hashes = []
    original_score = ours.score

    def score(model, current_data):
        if args.lane.startswith('gbdt'):
            arrays = [('model_', model.model_)]
        else:
            arrays = [(name, getattr(model, name)) for name in
                      ('_offsets', '_colid', '_quesval', '_left_child', '_leaves')]
        model_hashes.append(digest_arrays(arrays))  # Outside fit timer.
        return original_score(model, current_data)

    ours.score = score
    fitted_parameters = {}
    # Resolved CatBoost defaults are available only after fit. Capture outside
    # timers; constructor get_params alone hides leaf-estimation work.
    for arm in arms:
        if arm.library == "catboost":
            base_score = arm.score

            def score_catboost(model, current_data, name=arm.name, base=base_score):
                fitted_parameters[name] = model.get_all_params()
                return base(model, current_data)

            arm.score = score_catboost
    parameters = {}
    for arm in arms:
        estimator = arm.make()
        parameters[arm.name] = (estimator.get_params() if hasattr(estimator, 'get_params')
                                else config.copy())
    metadata = dict(lane=args.lane, dataset=data.tag, config=config,
                    dataset_scale=spec.dataset_scale(data),
                    numeric_mode='identical', vendor='cuda', rounds=args.rounds,
                    diagnostic_nvtx=args.nvtx,
                    cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                    thread_environment={key: os.environ.get(key) for key in
                        ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                         'CUDA_VISIBLE_DEVICES')},
                    symmetric_profile=args.symmetric_profile if args.lane == 'gbdt-symmetric' else None,
                    binding=witness,
                    binding_sha256=hashlib.sha256(Path(witness['path']).read_bytes()).hexdigest(),
                    benchmark_source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (Path(__file__), Path(forest.__file__), Path(spec.__file__))},
                    source_commit=Path('SHIPPED_COMMIT.txt').read_text().strip()
                    if Path('SHIPPED_COMMIT.txt').exists() else None,
                    data_sha256=digest_arrays([(name, getattr(data, name)) for name in
                        ('X_train', 'y_train', 'X_test', 'y_test')]),
                    versions={name: importlib.metadata.version(name)
                        for name in ('numpy', 'catboost', 'cuml-cu12')},
                    arm_parameters=parameters)
    capture = Tee()
    with contextlib.redirect_stdout(capture):
        spec.run(args.lane, arms, data, args.rounds, args.size, rotate_order=True,
                 fit_context=fit_range if args.nvtx else None)
    log = capture.getvalue()
    samples = {}
    predictions = {}
    for arm, ms, prediction in re.findall(
            r'^FSPEED lane=\S+ arm=(\S+) shape=\S+ round=\d+ ms=([\d.]+) hash=(\S+)',
            log, re.MULTILINE):
        samples.setdefault(arm, []).append(float(ms))
        predictions.setdefault(arm, []).append(prediction)
    summary = {}
    for arm, values in samples.items():
        spread = max(values) / min(values)
        summary[arm] = dict(fit_ms=values, median_ms=statistics.median(values),
                            max_min_spread=spread,
                            stable=(spread <= 1.10) if len(values) >= 5 else None)
    complete = all(len(samples.get(arm.name, [])) == args.rounds for arm in arms)
    hashes = predictions.get(ours.name, [])
    repeatable = (len(model_hashes) == args.rounds + 1 and len(set(model_hashes)) == 1
                  and len(hashes) == args.rounds and '-' not in hashes and len(set(hashes)) == 1)
    quality = [dict(arm=arm, metric=metric, value=float(value))
               for arm, metric, value in re.findall(
                   r'^FSPEED-ACC lane=\S+ arm=(\S+) metric=(\S+) value=(\S+)',
                   log, re.MULTILINE)]
    quality_valid = (all(np.isfinite(row['value']) for row in quality)
                     and all(any(row['arm'] == arm.name for row in quality) for arm in arms))
    metadata.update(summary=summary, complete=complete, quality=quality,
                    fitted_parameters=fitted_parameters,
                    arm_order_policy="rotate first arm each measured round",
                    arm_orders=re.findall(r'^FSPEED-ORDER .*$', log, re.MULTILINE),
                    quality_valid=quality_valid,
                    repeated_model_and_prediction_equal=repeatable if args.rounds >= 2 else None,
                    model_hashes=model_hashes, prediction_hashes=predictions,
                    scores=re.findall(r'^FSPEED-ACC .*$', log, re.MULTILINE),
                    refusals=re.findall(r'^FSPEED-REFUSED .*$', log, re.MULTILINE),
                    timing_contract='constructor + host packing + fit + synchronization; scoring excluded')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2, default=str) + '\n')
    print('BASELINE_RESULT', args.output, 'complete=', complete, 'repeatable=', repeatable if args.rounds >= 2 else 'one-round smoke')
    return 0 if complete and repeatable and quality_valid and not metadata['refusals'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
