"""Matched old/new sampled fits, interleaved in one process on one GPU.

Reference directory contains {mode}/_mojolearn_gbdt.so built at e05e889b
on THIS machine/vendor/toolchain. Current imports use the candidate bindings.
Do not copy reference binaries from a different device platform.
"""
import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
from mojolearn import GradientBoosting, vendor
from mojolearn._backend import binding


def digest(model, x):
    return (hashlib.sha256(model.model_.encode()).hexdigest(),
            hashlib.sha256(model.predict(x).tobytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--expected-vendor', choices=['metal', 'cuda', 'hip'], required=True)
    parser.add_argument('--rows', type=int, nargs='+', default=[8192, 65536, 262144])
    parser.add_argument('--modes', nargs='+', choices=['fast', 'deterministic', 'identical'], default=['fast', 'identical'])
    parser.add_argument('--policies', nargs='+', choices=['SymmetricTree', 'Depthwise', 'Lossguide'], default=['SymmetricTree', 'Depthwise', 'Lossguide'])
    parser.add_argument('--fractions', type=float, nargs='+', default=[.5, .25])
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if vendor() != args.expected_vendor:
        raise RuntimeError(f'expected {args.expected_vendor}, loaded {vendor()}')
    if args.rounds < 1 or any(n < 2 for n in args.rows):
        raise ValueError('positive rounds and at least two rows required')
    records = []
    for mode in args.modes:
        path = args.reference_root / mode / '_mojolearn_gbdt.so'
        spec = importlib.util.spec_from_file_location(f'feature_reference_{mode}._mojolearn_gbdt', path)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        expected_code = {'fast': 0, 'deterministic': 2, 'identical': 1}[mode]
        assert reference.gbdt_numeric_mode() == expected_code
        assert reference.gbdt_vendor() == args.expected_vendor
        candidate_binding = binding('_mojolearn_gbdt', mode=mode)
        assert candidate_binding.gbdt_vendor() == args.expected_vendor
        candidate_hash = hashlib.sha256(Path(candidate_binding.__file__).read_bytes()).hexdigest()
        for n in args.rows:
            rng = np.random.default_rng(20260910)
            x = rng.standard_normal((n, 32)).astype(np.float32)
            y = (x[:, 0] + x[:, 3] * .5 + (x[:, 5] > 0).astype(np.float32) + x[:, 9] * .2).astype(np.float32)
            data_hash = hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
            for policy in args.policies:
                for fraction in args.fractions:
                    opts = dict(loss='RMSE', n_estimators=8, max_depth=6, learning_rate=.15,
                                random_state=13, bootstrap_type='No', score_function='L2',
                                border_count=32, numeric_mode=mode, grow_policy=policy,
                                max_leaves=64 if policy == 'Lossguide' else None,
                                feature_fraction=fraction)
                    def fit(arm):
                        model = GradientBoosting(**opts)
                        if arm == 'reference':
                            model._bind = lambda name: reference
                        assert model._bind('_mojolearn_gbdt').gbdt_numeric_mode() == expected_code
                        begin = time.perf_counter()
                        model.fit(x, y)
                        elapsed = time.perf_counter() - begin
                        return elapsed, digest(model, x)
                    _, expected = fit('reference')
                    _, candidate = fit('candidate')
                    assert candidate == expected, (mode, n, policy, fraction, 'warmup model/prediction mismatch')
                    times = {'reference': [], 'candidate': []}
                    for repeat in range(args.rounds):
                        order = ('reference', 'candidate') if repeat % 2 == 0 else ('candidate', 'reference')
                        for arm in order:
                            elapsed, fingerprint = fit(arm)
                            assert fingerprint == expected, (mode, n, policy, fraction, arm, 'model/prediction mismatch')
                            times[arm].append(elapsed)
                    spread = {arm: max(values)/min(values) for arm, values in times.items()}
                    record = dict(mode=mode, policy=policy, rows=n, features=32,
                                  fraction=fraction, rounds=args.rounds, vendor=vendor(),
                                  dataset_sha256=data_hash, reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                  candidate_sha256=candidate_hash,
                                  model_sha256=expected[0], prediction_sha256=expected[1],
                                  seconds=times, spread=spread,
                                  stable=all(v <= 1.1 for v in spread.values()),
                                  candidate_over_reference=float(np.median(times['candidate'])/np.median(times['reference'])))
                    records.append(record)
                    print(json.dumps(record), flush=True)
                    # Preserve completed cells if a later profile fails.
                    args.output.write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    main()
