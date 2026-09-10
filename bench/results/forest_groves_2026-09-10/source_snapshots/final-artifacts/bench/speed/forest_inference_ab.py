#!/usr/bin/env python3
"""Same-forest transient/resident grove A/B: FAST Metal or IDENTICAL CUDA, RF or ET.

Use large real datasets for performance decisions. Synthetic fixtures are only
smoke/diagnostics. No rental, download fallback, CPU training or default change.
"""
import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import importlib.metadata
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import forest_speed_arm as forest
from nvidia_identical_trees import digest_arrays


def summarize(records):
    result = {}
    for name in dict.fromkeys(row['arm'] for row in records):
        samples = [row['ms'] for row in records
                   if row['arm'] == name and not row['warmup']]
        if not samples:
            continue
        spread = max(samples) / min(samples)
        result[name] = dict(samples_ms=samples, median_ms=statistics.median(samples),
                            spread=spread, stable=len(samples) >= 5 and spread <= 1.10)
    return result


def staged_call(model, method, X):
    """Separate instrumented public call, never a GPU kernel-only timer.

    Shared dispatch includes cache lookup/preparation and the binding. Older RF
    wrappers expose only the binding selector; record that different boundary.
    ET input checking lives inside _vote, so its Python remainder includes it.
    """
    stages = {}
    hook = '_predict_forest' if hasattr(model, '_predict_forest') else '_prediction_function'
    originals = {hook: getattr(model, hook)}
    if hasattr(model, '_check_predict_input'):
        originals['_check_predict_input'] = model._check_predict_input
    previous = {name: model.__dict__.get(name) for name in originals}
    existed = {name: name in model.__dict__ for name in originals}

    def checked(*args, **kwargs):
        start = time.perf_counter()
        value = originals['_check_predict_input'](*args, **kwargs)
        stages['input_validation_packing_ms'] = 1000 * (time.perf_counter() - start)
        return value

    def timed(fn, *args, **kwargs):
        start = time.perf_counter()
        value = fn(*args, **kwargs)
        stages['dispatch_ms'] = 1000 * (time.perf_counter() - start)
        return value

    def dispatch(*args, **kwargs):
        if hook == '_predict_forest':
            return timed(originals[hook], *args, **kwargs)
        fn = originals[hook](*args, **kwargs)
        return lambda *a, **kw: timed(fn, *a, **kw)

    if '_check_predict_input' in originals:
        model._check_predict_input = checked
    setattr(model, hook, dispatch)
    try:
        start = time.perf_counter()
        prediction = getattr(model, method)(X)
        stages['instrumented_public_ms'] = 1000 * (time.perf_counter() - start)
        if 'dispatch_ms' not in stages:
            raise RuntimeError('public prediction bypassed instrumented dispatch boundary')
        stages['dispatch_boundary'] = hook
        stages['python_remainder_ms'] = (stages['instrumented_public_ms']
            - stages.get('input_validation_packing_ms', 0) - stages['dispatch_ms'])
        return prediction, stages
    finally:
        for name in originals:
            if existed[name]:
                setattr(model, name, previous[name])
            else:
                delattr(model, name)


@contextmanager
def prediction_arm(model, name):
    """Transient diagnostic still invokes the public response method.

    Override only its shared native dispatcher to reach the original stateless
    GPU ABI. Input checking/packing and output construction remain public.
    """
    model.inference_engine = 'sequential' if name == 'sequential' else 'parallel_groves'
    if name in ('parallel_groves_resident', 'parallel_groves_borrowed'):
        hook = '_resident_prediction_function'
        if not callable(getattr(model, hook, None)):
            if name == 'parallel_groves_borrowed':
                raise RuntimeError('rebuild Python wrapper for borrowed-buffer inference')
            yield
            return
        native_name = ('forest_predict_resident_into_gpu' if name == 'parallel_groves_borrowed'
                       else 'forest_predict_resident_gpu')
        if not callable(getattr(model._bind(), native_name, None)):
            raise RuntimeError('rebuild binding for ' + native_name)
        previous = model.__dict__.get(hook)
        existed = hook in model.__dict__
        setattr(model, hook, lambda native: getattr(native, native_name))
        try:
            yield
        finally:
            if existed:
                setattr(model, hook, previous)
            else:
                delattr(model, hook)
        return
    if name != 'parallel_groves_transient':
        yield
        return
    previous = model.__dict__.get('_predict_forest')
    existed = '_predict_forest' in model.__dict__

    def transient(sequential_name, X, out):
        native = model._bind()
        fn = getattr(native, sequential_name + '_gpu_parallel')
        arrays = [getattr(model, key) for key in
                  ('_offsets', '_colid', '_quesval', '_left_child', '_leaves')]
        rows, features = X.shape
        return fn(*(int(a.ctypes.data) for a in arrays), int(X.ctypes.data),
                  int(out.ctypes.data), [int(rows), int(features),
                                        int(model._n_trees), int(model._num_outputs)])
    model._predict_forest = transient
    try:
        yield
    finally:
        if existed:
            model._predict_forest = previous
        else:
            del model._predict_forest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lane', choices=['rf', 'et'], default='rf')
    parser.add_argument('--dataset', choices=['higgs', 'year', 'covtype', 'covtype2',
                                             'synth', 'synthclf'], default='higgs')
    parser.add_argument('--rows', type=int, default=1000000)
    parser.add_argument('--predict-rows', type=int, help='cap held-out rows; never repeat rows')
    parser.add_argument('--size', choices=['smoke', 'shipped'], default='shipped')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--calls-per-sample', type=int, default=1,
                        help='public calls per measured block; normalize block time to ms/call')
    parser.add_argument('--staged-rounds', type=int, default=0)
    parser.add_argument('--trees', type=int, help='explicit model-complexity diagnostic override')
    parser.add_argument('--depth', type=int, help='explicit model-complexity diagnostic override')
    parser.add_argument('--borrowed-buffers', action='store_true',
                        help='add resident borrowed-output ABI arm; requires new binding and wrapper hook')
    parser.add_argument('--include-sequential', action='store_true',
                        help='also time existing host inference; can be expensive on large models')
    parser.add_argument('--cuml-context', action='store_true',
                        help='CUDA RF only; separately trained cached GPU model, not identical forest')
    parser.add_argument('--rtol', type=float, default=2e-6)
    parser.add_argument('--atol', type=float, default=2e-6)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.rows < 2 or args.rounds < 1 or args.staged_rounds < 0 or args.calls_per_sample < 1:
        parser.error('rows>=2, rounds>=1, staged-rounds>=0 and calls-per-sample>=1 required')
    if any(value is not None and value < 1 for value in
           (args.predict_rows, args.trees, args.depth)):
        parser.error('prediction rows, trees and depth must be positive')
    if not all(np.isfinite(v) and v >= 0 for v in (args.rtol, args.atol)):
        parser.error('tolerances must be finite and nonnegative')
    mode = os.environ.get('MOJOLEARN_NUMERIC_MODE')
    vendor = os.environ.get('MOJOLEARN_SPEED_EXPECTED_VENDOR')
    if (mode, vendor) not in [('fast', 'metal'), ('identical', 'cuda')]:
        parser.error('select FAST/metal or IDENTICAL/cuda through explicit environment variables')
    if args.cuml_context and (vendor != 'cuda' or args.lane != 'rf'):
        parser.error('cuML context requires CUDA RF; it is not an equivalent ET learner')
    if forest._fortran_requested():
        parser.error('leave MOJOLEARN_SPEED_FORTRAN unset: public calls include input packing')
    if args.output.exists():
        parser.error('choose fresh output to preserve evidence')
    cupy = None
    if vendor == 'cuda':
        import cupy
        cupy.cuda.runtime.deviceSynchronize()
    data = forest.spec.load_dataset(args.dataset, args.size, args.rows)
    if args.predict_rows is not None:
        if args.predict_rows > len(data.y_test):
            parser.error('predict-rows exceeds held-out split; increasing by repetition is refused')
        data.X_test = data.X_test[:args.predict_rows]
        data.y_test = data.y_test[:args.predict_rows]
    forest.spec.emit_scale_reminder(data, 'inference-before')
    forest.prepare_our_inputs(data)
    cfg = forest.spec.lane_config(args.lane, args.size)
    for name, value in [('n_estimators', args.trees), ('max_depth', args.depth)]:
        if value is not None:
            cfg[name] = value
    arm = forest.OUR_BUILDERS[args.lane](args.lane, cfg, data)
    witness = forest.verify_our_arm(arm)
    ours = arm.make()
    start = time.perf_counter()
    arm.fit(ours, data)
    arm.sync()
    fit_ms = 1000 * (time.perf_counter() - start)
    vector_getter = getattr(ours._bind(), 'forest_vector_groves', None)
    if not callable(vector_getter):
        raise RuntimeError('rebuild binding: compiled vector dispatch readback is missing')
    vector_dispatch = bool(vector_getter(int(ours._num_outputs)))
    print('INFERENCE_DISPATCH', mode, vendor, 'outputs=', int(ours._num_outputs),
          'compiled_vector_groves=', vector_dispatch, flush=True)
    method = 'predict' if data.task == 'regression' else 'predict_proba'
    arrays = [(name, getattr(ours, name)) for name in
              ('_offsets', '_colid', '_quesval', '_left_child', '_leaves')]
    names = ['parallel_groves_transient', 'parallel_groves_resident']
    if args.borrowed_buffers:
        if (not callable(getattr(ours, '_resident_prediction_function', None))
                or not callable(getattr(ours._bind(), 'forest_predict_resident_into_gpu', None))):
            raise RuntimeError('borrowed-buffers requires resident selector hook and into binding')
        names.append('parallel_groves_borrowed')
    if args.include_sequential:
        names.append('sequential')
    cuml_model = None
    if args.cuml_context:
        forest.spec.prepare_cuml_labels(data)
        competitor = forest.spec.cuml_rf_arm('rf', cfg, data)
        cuml_model = competitor.make()
        competitor.fit(cuml_model, data)
        competitor.sync()
        names.append('cuml_gpu')
    versions = {}
    for name in ('numpy', 'mojolearn', 'cuml-cu12'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    records, stages, hashes = [], [], {}
    result = dict(harness_schema=2, complete=False, records=records, staged_records=stages,
        calls_per_sample=args.calls_per_sample,
        timing_scope='batched_public_throughput' if args.calls_per_sample > 1 else 'single_public_call',
        lane=args.lane, dataset=data.tag, task=data.task, response_method=method,
        rows_fit=len(data.y_train), rows_predict=len(data.y_test),
        features=data.X_train.shape[1], outputs=int(ours._num_outputs),
        numeric_mode=mode, vendor=vendor, binding=witness, config=cfg, versions=versions,
        compiled_vector_groves=vector_dispatch, borrowed_buffers=args.borrowed_buffers,
        first_call_contract='round0 includes lazy preparation for each engine; measured repeats may reuse resident state',
        trees=int(ours._n_trees), nodes=int(ours._colid.size),
        host_model_bytes=sum(value.nbytes for _, value in arrays),
        host_predict_input_bytes=int(data._ours_Xtest.nbytes),
        model_sha256=digest_arrays(arrays),
        data_sha256=digest_arrays([(name, getattr(data, name)) for name in
                                 ('X_train', 'y_train', 'X_test', 'y_test')]),
        binding_sha256=hashlib.sha256(Path(witness['path']).read_bytes()).hexdigest(),
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (Path(__file__), Path(forest.__file__), Path(forest.spec.__file__),
                                   Path('core/forest_inference.mojo'),
                                   Path('core/forest_inference_model.mojo'),
                                   Path('bindings/forest_inference_binding.mojo'),
                                   Path('bindings/_mojolearn_rf.mojo'),
                                   Path('bindings/_mojolearn_trees.mojo'),
                                   Path('python/mojolearn/_forest_protocol.py'),
                                   Path('python/mojolearn/randomforest.py'),
                                   Path('python/mojolearn/extratrees.py'))},
        source_commit=Path('SHIPPED_COMMIT.txt').read_text().strip()
            if Path('SHIPPED_COMMIT.txt').exists() else None,
        preparation_fit_ms=fit_ms, rtol=args.rtol, atol=args.atol,
        contract='primary public calls include host packing, native staging/transfers and host output; '
                 'fit excluded; same MojoLearn forest, transient public dispatch overridden to stateless GPU ABI; resident/transient groves must match bits; '
                 'optional sequential reference may differ in association; '
                 'optional cuML context has a separately trained cached GPU model',
        stages_contract='separate instrumented public calls, no kernel-only timing; '
                        'dispatch includes cache lookup, native validation, allocation, transfer, traversal and output; '
                        'ET input checking is included in Python remainder',
        memory_contract='reported byte counts are host arrays, not peak device/host memory; collect telemetry separately')
    # Do not retain pre-cache host arrays after the resident snapshot freezes
    # them; an extra benchmark-only owner would distort model memory pressure.
    del arrays
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        result['summary'] = summarize(records)
        args.output.write_text(json.dumps(result, indent=2, default=str) + '\n')

    grove_reference = None
    try:
        save()
        for round_index in range(args.rounds + 1):
            offset = max(0, round_index - 1) % len(names)
            for name in names[offset:] + names[:offset]:
                count = 1 if round_index == 0 else args.calls_per_sample
                predictions = []
                arm_context = nullcontext() if name == 'cuml_gpu' else prediction_arm(ours, name)
                with arm_context:
                    start = time.perf_counter()
                    for _ in range(count):
                        if name == 'cuml_gpu':
                            pred = getattr(cuml_model, method)(data.X_test)
                            if isinstance(pred, cupy.ndarray):
                                pred = cupy.asnumpy(pred)
                            pred = np.asarray(pred)
                            cupy.cuda.runtime.deviceSynchronize()
                        else:
                            pred = getattr(ours, method)(data._ours_Xtest)
                        predictions.append(pred)
                    block_ms = 1000 * (time.perf_counter() - start)
                row = dict(arm=name, round=round_index, warmup=round_index == 0,
                           block_ms=block_ms, calls_per_sample=count, ms=block_ms/count,
                           timing_scope='batched_public_throughput' if count > 1 else 'single_public_call',
                           prediction_hashes=[])
                records.append(row)
                expected = ((len(data.y_test),) if method == 'predict'
                            else (len(data.y_test), int(ours._num_outputs)))
                for pred in predictions:
                    if pred.shape != expected or not np.isfinite(pred).all():
                        raise AssertionError('invalid prediction shape or nonfinite output: ' + name)
                    digest = digest_arrays([('prediction', pred)])
                    row['prediction_hashes'].append(digest)
                    row['hash'] = digest
                    if name != 'cuml_gpu' and name in hashes and digest != hashes[name]:
                        raise AssertionError('within-engine repeated bits changed: ' + name)
                    hashes[name] = digest
                    if name.startswith('parallel_groves'):
                        if grove_reference is None:
                            grove_reference = pred.copy()
                        elif digest_arrays([('prediction', grove_reference)]) != digest:
                            raise AssertionError('grove arms differ in output bits')
                    if name == 'sequential' and grove_reference is not None:
                        error = float(np.max(np.abs(pred.astype(np.float64) - grove_reference.astype(np.float64))))
                        row['max_abs_error_vs_groves'] = max(row.get('max_abs_error_vs_groves', 0), error)
                        np.testing.assert_allclose(pred, grove_reference, rtol=args.rtol, atol=args.atol)
                del predictions, pred
                print(json.dumps(row), flush=True)
                save()
        for index in range(args.staged_rounds):
            order = [name for name in names if name != 'cuml_gpu']
            if index % 2:
                order.reverse()
            for name in order:
                with prediction_arm(ours, name):
                    pred, spans = staged_call(ours, method, data._ours_Xtest)
                stages.append(dict(arm=name, round=index, **spans))
                if digest_arrays([('prediction', pred)]) != hashes[name]:
                    raise AssertionError('instrumentation changed outputs: ' + name)
                save()
        result['complete'] = True
        result['timing_stable'] = all(row['stable'] for row in summarize(records).values())
        result['repeated_engine_hashes'] = hashes
        save()
    except Exception as exc:
        result['error'] = type(exc).__name__ + ': ' + str(exc)
        save()
        raise
    forest.spec.emit_scale_reminder(data, 'inference-after')
    print('PASS inference comparison; stable=', result['timing_stable'],
          '(completion is not performance qualification)')


if __name__ == '__main__':
    main()
