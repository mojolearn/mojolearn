#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Time ET shared counts against private accumulators with exact model checks.

First run extratrees/tools/check_shared_score.sh to build the six binaries.
Then use tools/with_build_lock.sh python3 extratrees/bench/shared_score_ab.py OUTPUT.
The build lock must span timing; this driver additionally takes the bench lock.
"""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def synthetic_main():
    if os.environ.get("MOJOLEARN_BUILD_LOCK_HELD") != "1":
        raise SystemExit("Run via tools/with_build_lock.sh")
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=True)
    cases = [(65536, 2, 16, 10), (262144, 2, 16, 10),
             (65536, 5, 8, 8), (65536, 9, 8, 8),
             (65536, 17, 8, 8), (65536, 32, 8, 8)]
    if os.environ.get("MOJOLEARN_ET_BENCH_MULTICLASS_ONLY") == "1":
        cases = [case for case in cases if case[1] >= 5]
    wanted_classes = os.environ.get("MOJOLEARN_ET_BENCH_CLASSES")
    if wanted_classes:
        selected = {int(v) for v in wanted_classes.split(",")}
        cases = [case for case in cases if case[1] in selected]
    modes = os.environ.get("MOJOLEARN_ET_BENCH_MODES", "fast,identical").split(",")
    if not cases or any(mode not in ("fast", "identical", "deterministic") for mode in modes):
        raise SystemExit("Invalid or empty benchmark mode/class selection")
    results = []
    subprocess.run(["tools/bench_lock.sh", "acquire", "et-shared-counts", "exact model ABBA fits", "2 minutes"], check=True)
    try:
        for mode in modes:
            for rows, classes, trees, depth in cases:
                samples = {"baseline": [], "candidate": []}
                medians_by_pass = {"baseline": [], "candidate": []}
                fingerprints = set()
                for rep, arm in enumerate(("baseline", "candidate", "candidate", "baseline")):
                    cmd = [f"/tmp/et-shared-{mode}-{arm}", str(rows), str(classes), str(trees), str(depth), "3"]
                    run = subprocess.run(cmd, text=True, capture_output=True)
                    (output / f"{mode}-{rows}-{classes}-{arm}-{rep}.log").write_text(run.stdout + run.stderr)
                    run.check_returncode()
                    lines = run.stdout.splitlines()
                    assert f"numeric_mode {mode.upper()}" in lines
                    assert f"shared_class_counts_mask {15 if arm == 'candidate' else 0}" in lines
                    hashes = [s for s in lines if s.startswith("fingerprint ")]
                    times = [float(s.split()[-1]) for s in lines if s.startswith("fit_ms ")]
                    assert len(hashes) == 1 and len(times) == 3
                    fingerprints.update(hashes)
                    samples[arm].extend(times)
                    medians_by_pass[arm].append(statistics.median(times))
                assert len(fingerprints) == 1, fingerprints
                medians = {arm: statistics.median(values) for arm, values in samples.items()}
                drift = max(medians_by_pass["baseline"]) / min(medians_by_pass["baseline"])
                candidate_drift = max(medians_by_pass["candidate"]) / min(medians_by_pass["candidate"])
                result = dict(mode=mode, warmup_fits=int(os.environ.get("MOJOLEARN_ET_WARMUP_FITS", "1")), rows=rows, classes=classes, cols=13, trees=trees, depth=depth,
                              medians_ms=medians, speedup=medians["baseline"] / medians["candidate"],
                              baseline_drift_ratio=drift, candidate_drift_ratio=candidate_drift,
                              stable=max(drift, candidate_drift) <= 1.10,
                              samples_ms=samples, fingerprint=fingerprints.pop())
                results.append(result)
                print(json.dumps(result), flush=True)
                (output / "timing.json").write_text(json.dumps(results, indent=2) + "\n")
    finally:
        subprocess.run(["tools/bench_lock.sh", "release"], check=True)



# Real-data A/B uses isolated extension artifacts; no installed binary is
# overwritten by this driver. Keep the small synthetic gate above unchanged.
ROOT = Path(__file__).resolve().parents[2]


def file_hash(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def digest_array(digest, name, array):
    import numpy as np
    array = np.ascontiguousarray(array)
    digest.update(name.encode() + str(array.dtype).encode() + str(array.shape).encode())
    raw = memoryview(array).cast('B')
    for start in range(0, len(raw), 8 * 1024 * 1024):
        digest.update(raw[start:start + 8 * 1024 * 1024])


def verify_binding(native, mode, vendor, mask):
    witness = dict(mode=int(native.trees_numeric_mode()),
                   vendor=str(native.trees_vendor()),
                   shared_counts_mask=int(native.trees_shared_counts_mask()))
    expected = dict(mode={'fast': 0, 'identical': 1, 'deterministic': 2}[mode],
                    vendor=vendor, shared_counts_mask=mask)
    if witness != expected:
        raise RuntimeError(f'Compiled binding witness mismatch: {witness} != {expected}')
    return witness


def real_worker(config_path, arm, result_path):
    import hashlib
    import importlib.util
    import time
    import numpy as np
    config = json.loads(Path(config_path).read_text())
    artifact = config['artifacts'][arm]
    if file_hash(artifact['path']) != artifact['sha256']:
        raise RuntimeError('Binding artifact changed since benchmark preparation')
    spec = importlib.util.spec_from_file_location('_mojolearn_trees', artifact['path'])
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    witness = verify_binding(native, config['mode'], config['vendor'],
                             0 if arm == 'baseline' else 15)
    sys.path.insert(0, str(ROOT / 'python'))
    from mojolearn.extratrees import ExtraTreesClassifier, _ExtraTreesBase

    def exact_binding(self, name=None):
        if self._effective_mode() != config['mode']:
            raise RuntimeError('Estimator mode differs from benchmark mode')
        return native
    # Process-local selection survives fit's constructor-state refresh.
    _ExtraTreesBase._bind = exact_binding
    data = {}
    for name, item in config['data'].items():
        if file_hash(item['path']) != item['sha256']:
            raise RuntimeError(f'Prepared data changed: {name}')
        data[name] = np.load(item['path'], mmap_mode='r', allow_pickle=False)
    samples = []
    reference = None
    for repeat in range(config['warmups'] + config['repeats']):
        model = ExtraTreesClassifier(numeric_mode=config['mode'], **config['parameters'])
        start = time.perf_counter_ns()
        model.fit(data['X_train'], data['y_train'])
        fit_ms = (time.perf_counter_ns() - start) / 1e6
        # fit returns host model arrays, so all training work has completed.
        # Hashing and existing prediction traversal are deliberately untimed.
        model_digest = hashlib.sha256()
        for name in ('_offsets', '_colid', '_quesval', '_left_child', '_leaves', 'classes_'):
            digest_array(model_digest, name, getattr(model, name))
        digest_array(model_digest, 'metadata', np.asarray([
            model.n_features_in_, model._n_trees, model._num_outputs,
            model.depth_cap_bound_, model.max_depth_resolved_, model.max_features_], dtype=np.int64))
        prediction_digest = hashlib.sha256()
        accuracy = {}
        for split in ('train', 'test'):
            X, y = data['X_' + split], data['y_' + split]
            correct = 0
            for begin in range(0, len(X), config['prediction_batch_rows']):
                end = min(len(X), begin + config['prediction_batch_rows'])
                probabilities = model.predict_proba(X[begin:end])
                if not np.isfinite(probabilities).all():
                    raise RuntimeError('Non-finite probabilities')
                digest_array(prediction_digest, f'{split}:{begin}', probabilities)
                predicted = model.classes_[np.argmax(probabilities, axis=1)]
                correct += int(np.count_nonzero(predicted == y[begin:end]))
            accuracy[split] = correct / len(X)
        fingerprints = dict(model=model_digest.hexdigest(), predictions=prediction_digest.hexdigest())
        if reference is not None and fingerprints != reference:
            raise RuntimeError(f'Within-arm model/prediction bits changed: {arm}, repeat {repeat}')
        reference = fingerprints
        item = dict(repeat=repeat, warmup=repeat < config['warmups'], fit_ms=fit_ms,
                    fingerprints=fingerprints, accuracy=accuracy)
        samples.append(item)
        # Flush partial evidence after each completed fit, even if a later fit fails.
        Path(result_path).write_text(json.dumps(dict(arm=arm, witness=witness,
            artifact=artifact, samples=samples), indent=2) + '\n')
        print(json.dumps(item), flush=True)


def real_summary(passes, expected_repeats):
    samples = {arm: [] for arm in ('baseline', 'candidate')}
    medians = {arm: [] for arm in samples}
    hashes = set()
    for result in passes:
        values = [item['fit_ms'] for item in result['samples'] if not item['warmup']]
        if len(values) != expected_repeats or any(not (0 < v < float('inf')) for v in values):
            raise RuntimeError('Missing or invalid timing samples')
        for item in result['samples']:
            hashes.add(tuple(sorted(item['fingerprints'].items())))
        samples[result['arm']].extend(values)
        medians[result['arm']].append(statistics.median(values))
    if len(hashes) != 1:
        raise RuntimeError('A/B full-model or full-prediction fingerprints differ')
    summaries = {}
    for arm, values in samples.items():
        if not values:
            raise RuntimeError('Both arms need completed samples')
        center = statistics.median(values)
        summaries[arm] = dict(samples_ms=values, median_ms=center, min_ms=min(values),
            max_ms=max(values), stdev_ms=statistics.stdev(values) if len(values) > 1 else 0,
            mad_ms=statistics.median(abs(v-center) for v in values),
            pass_medians_ms=medians[arm], drift_ratio=max(medians[arm])/min(medians[arm]),
            spread_ratio=max(values)/min(values), sample_count=len(values))
    stable = all(item['drift_ratio'] <= 1.10 and item['spread_ratio'] <= 1.10
                 and item['sample_count'] >= 5 for item in summaries.values())
    ratio = summaries['baseline']['median_ms']/summaries['candidate']['median_ms']
    return dict(arms=summaries, stable=stable, observed_speedup=ratio,
                qualified_speedup=ratio if stable else None,
                fingerprints=dict(next(iter(hashes))))



def source_provenance():
    result = {'script_sha256': file_hash(Path(__file__).resolve())}
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                          capture_output=True, text=True)
    if head.returncode == 0:
        result['git_head'] = head.stdout.strip()
        status = subprocess.run(['git', 'status', '--short'], cwd=ROOT,
                                capture_output=True, text=True, check=True)
        result['dirty_paths'] = status.stdout.splitlines()
    else:
        marker = ROOT / 'SHIPPED_COMMIT.txt'
        result['shipped_commit'] = marker.read_text().strip() if marker.exists() else None
        result['git_unavailable'] = True
    result['source_hashes'] = {
        str(path): file_hash(ROOT / path) for path in (
            Path('tools/speed_gbdt_arm.py'), Path('python/mojolearn/extratrees.py'),
            Path('python/mojolearn/_forest_protocol.py'), Path('bindings/_mojolearn_trees.mojo'),
            Path('extratrees/impl/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo'))
    }
    return result

def real_main(argv):
    import argparse
    import numpy as np
    parser = argparse.ArgumentParser(description='Real-data full-public-fit ET shared-count A/B')
    parser.add_argument('output', type=Path)
    parser.add_argument('--dataset', choices=('higgs', 'covtype', 'covtype2'), required=True)
    parser.add_argument('--rows', type=int, default=1000000, help='Shared-loader cap; actual shapes are recorded')
    parser.add_argument('--baseline-binding', type=Path, required=True)
    parser.add_argument('--candidate-binding', type=Path, required=True)
    parser.add_argument('--mode', choices=('fast', 'deterministic', 'identical'), default='identical')
    parser.add_argument('--vendor', choices=('cuda', 'metal', 'hip'), default='cuda')
    parser.add_argument('--trees', type=int, default=100)
    parser.add_argument('--depth', type=int, default=16)
    parser.add_argument('--max-features', choices=('sqrt', 'all'), default='sqrt')
    parser.add_argument('--criterion', choices=('gini', 'entropy'), default='gini')
    parser.add_argument('--bootstrap', action='store_true')
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--cycles', type=int, default=1, help='ABBA then BAAB cycles')
    parser.add_argument('--prediction-batch-rows', type=int, default=65536)
    args = parser.parse_args(argv)
    for name in ('rows', 'trees', 'depth', 'warmups', 'repeats', 'cycles', 'prediction_batch_rows'):
        if getattr(args, name) < 1:
            parser.error(f'{name} must be positive')
    if os.environ.get('MOJOLEARN_BUILD_LOCK_HELD') != '1':
        parser.error('Run via tools/with_build_lock.sh')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for arm in ('baseline', 'candidate'):
        path = getattr(args, arm + '_binding').resolve(strict=True)
        artifacts[arm] = dict(path=str(path), sha256=file_hash(path))
    if artifacts['baseline']['sha256'] == artifacts['candidate']['sha256']:
        parser.error('Baseline and candidate must be distinct compiled artifacts')
    sys.path.insert(0, str(ROOT / 'tools'))
    from speed_gbdt_arm import load_dataset, dataset_scale, emit_scale_reminder
    dataset = load_dataset(args.dataset, 'shipped', args.rows)
    scale = dataset_scale(dataset)
    emit_scale_reminder(dataset, 'et-shared-real')
    prepared = {}
    shapes = {}
    for name in ('X_train', 'X_test', 'y_train', 'y_test'):
        array = np.ascontiguousarray(getattr(dataset, name), dtype=np.float32)
        if not len(array) or not np.isfinite(array).all():
            raise ValueError(f'Empty/non-finite real data: {name}')
        target = output / (name + '.npy')
        np.save(target, array, allow_pickle=False)
        prepared[name] = dict(path=str(target), sha256=file_hash(target))
        shapes[name] = list(array.shape)
    classes = np.unique(dataset.y_train)
    if not 2 <= len(classes) <= 32:
        raise ValueError('This ET benchmark requires 2–32 observed classes')
    if not np.isin(dataset.y_test, classes).all():
        raise ValueError('Test labels are outside training vocabulary')
    del dataset
    config = dict(dataset=args.dataset, requested_rows=args.rows, shapes=shapes,
        classes=classes.tolist(), mode=args.mode, vendor=args.vendor, artifacts=artifacts,
        data=prepared, warmups=args.warmups, repeats=args.repeats,
        prediction_batch_rows=args.prediction_batch_rows,
        parameters=dict(n_estimators=args.trees, max_depth=args.depth,
            max_features=None if args.max_features == 'all' else 'sqrt',
            random_state=0xACC2021, bootstrap=args.bootstrap, criterion=args.criterion),
        timer='full Python fit including packing and host model export; prediction/hash excluded',
        dataset_scale=scale, source_provenance=source_provenance())
    if args.vendor == 'cuda':
        config['nvidia_smi'] = subprocess.check_output(['nvidia-smi'], text=True)
    config_path = output / 'config.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    env = dict(os.environ, MOJOLEARN_NUMERIC_MODE=args.mode, BENCH_LOCK_PID=str(os.getpid()))
    lock = str(ROOT / 'tools/bench_lock.sh')
    subprocess.run([lock, 'acquire', 'et-real-shared-counts', args.dataset, 'full-fit ABBA'], env=env, check=True)
    passes = []
    try:
        for cycle in range(args.cycles):
            order = ('baseline', 'candidate', 'candidate', 'baseline') if cycle % 2 == 0 else (
                     'candidate', 'baseline', 'baseline', 'candidate')
            for arm in order:
                index = len(passes)
                result_path = output / f'pass-{index:02d}-{arm}.json'
                command = [sys.executable, str(Path(__file__).resolve()), '--real-worker',
                           str(config_path), arm, str(result_path)]
                with (output / f'pass-{index:02d}-{arm}.log').open('w') as log:
                    log.write('COMMAND: ' + json.dumps(command) + '\n'); log.flush()
                    subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                passes.append(json.loads(result_path.read_text()))
        summary = real_summary(passes, args.repeats)
        summary['pass_order'] = [item['arm'] for item in passes]
        summary['config'] = str(config_path)
        (output / 'timing.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary), flush=True)
    finally:
        subprocess.run([lock, 'release'], env=env, check=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--real-worker':
        real_worker(*sys.argv[2:])
    elif '--dataset' in sys.argv or '--help' in sys.argv or any(a.startswith('--dataset=') for a in sys.argv):
        real_main(sys.argv[1:])
    else:
        synthetic_main()
