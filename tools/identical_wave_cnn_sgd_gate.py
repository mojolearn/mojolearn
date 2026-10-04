#!/usr/bin/env python3
"""Untimed CNN/SGD boundary fixtures; one fit per training configuration.

Run each suite separately for GPU and CPU columns of the SAME frozen ON/OFF
builds. Compare case digests across vendors within each arm; CNN ON/OFF may
intentionally differ. This file runs no compiler, timer, or opponent.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback


def dump(path, doc):
    temporary = path.with_suffix(path.suffix+'.new')
    temporary.write_text(json.dumps(doc, indent=2)+'\n'); temporary.replace(path)


def source_check(root, sha):
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != sha or subprocess.run(['git', '-C', str(root), 'diff', '--quiet', 'HEAD', '--']).returncode:
        raise RuntimeError('frozen source SHA/clean-tree check failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--sha', required=True)
    parser.add_argument('--harness-sha', help='Optional exact commit supplying this harness, separately from numerical source')
    parser.add_argument('--vendor', required=True, choices=('cuda', 'hip', 'metal', 'cpu'))
    parser.add_argument('--arm', required=True, choices=('on', 'off'))
    parser.add_argument('--suite', required=True, choices=('cnn-primitives', 'cnn-training', 'sgd'))
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{40}', args.sha): parser.error('exact full source SHA required')
    if os.environ.get('MOJOLEARN_NUMERIC_MODE') != 'identical': parser.error('IDENTICAL environment required')
    if os.environ.get('MOJOLEARN_VENDOR') != args.vendor: parser.error('explicit vendor environment mismatch')
    root = args.source.resolve(); source_check(root, args.sha)
    harness = Path(__file__).read_bytes()
    harness_sha = args.harness_sha or args.sha
    if not re.fullmatch('[0-9a-f]{40}', harness_sha): parser.error('exact harness commit required')
    committed_harness = subprocess.check_output(['git', '-C', str(root), 'show', harness_sha+':tools/identical_wave_cnn_sgd_gate.py'])
    if harness != committed_harness:
        raise RuntimeError('gate harness differs from recorded harness commit')
    args.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root/'python'))
    import numpy as np
    import mojolearn as ml
    if not Path(ml.__file__).resolve().is_relative_to(root/'python/mojolearn'):
        raise RuntimeError('non-source package refused')
    if ml.vendor() != args.vendor: raise RuntimeError('selected vendor differs from requested vendor')
    report = {'schema': 1, 'status': 'INCOMPLETE', 'sha': args.sha, 'vendor': args.vendor,
              'arm': args.arm, 'suite': args.suite, 'timing_samples': 0, 'opponents_executed': 0,
              'harness_sha256': hashlib.sha256(harness).hexdigest(), 'harness_source_sha': harness_sha,
              'cases': {}, 'bindings': {}, 'package': str(Path(ml.__file__).resolve())}
    receipt = args.out/'gate.json'; dump(receipt, report)

    def bind(module, prefix):
        if int(getattr(module, prefix+'_numeric_mode')()) != 1:
            raise RuntimeError('binding is not IDENTICAL: '+prefix)
        if str(getattr(module, prefix+'_vendor')()) != args.vendor:
            raise RuntimeError('binding vendor mismatch: '+prefix)
        if args.vendor == 'cpu':
            from mojolearn import _backend
            native = _backend.load_host_module('_mojolearn_'+prefix+'_host')
        else:
            native = module
        path = Path(vars(native).get('__file__') or native.__spec__.origin).resolve()
        if not path.is_relative_to(root/'python/mojolearn') or path.suffix != '.so':
            raise RuntimeError('source-built binding provenance absent: '+str(path))
        report['bindings'][prefix] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                      'numeric_mode': 1, 'vendor': args.vendor}

    def case(name, function):
        try:
            values = {key: np.ascontiguousarray(value) for key, value in function().items()}
            if not values: raise AssertionError('empty fixture output')
            digest = hashlib.sha256(); outputs = {}
            for key, value in sorted(values.items()):
                if value.dtype.hasobject: raise AssertionError('object output cannot establish bits')
                metadata = {'shape': list(value.shape), 'dtype': value.dtype.str}
                raw = value.tobytes()
                digest.update(json.dumps([key, metadata], sort_keys=True).encode()+b'\0'+raw)
                outputs[key] = dict(metadata, sha256=hashlib.sha256(raw).hexdigest())
            np.savez(args.out/(name+'.npz'), **values)
            report['cases'][name] = {'status': 'PASS', 'digest': digest.hexdigest(), 'outputs': outputs}
        except Exception as exc:
            report['cases'][name] = {'status': 'FAIL', 'error': str(exc), 'traceback': traceback.format_exc()}
        dump(receipt, report)
        print('CNN_SGD_GATE', name, report['cases'][name]['status'], flush=True)

    if args.suite.startswith('cnn'):
        from mojolearn import _expansion_cnn as cnn
        binding = ml.CNNClassifier(input_shape=(1, 4, 4))._binding()
        bind(binding, 'x_cnn')
        flags = int(binding.x_cnn_idn2_flags())
        report['cnn_idn2_flags'] = flags
        if args.arm == 'on' and flags & 3 != 3: raise RuntimeError('ON fold/epoch routing not enabled')
        if args.arm == 'off' and flags & 3: raise RuntimeError('master-OFF fold/epoch routing still enabled')

    if args.suite == 'cnn-primitives':
        for n in (33, 1024, 1025):
            def xent(n=n):
                logits = (((np.arange(n*7, dtype=np.int32) % 31)-15).astype(np.float32)/np.float32(8)).reshape(n, 7)
                labels = np.arange(n, dtype=np.int32) % 7
                loss, gradient, probabilities = cnn._softmax_xent(binding, logits, labels)
                shifted = logits.astype(np.float64)-logits.max(axis=1, keepdims=True)
                expected = np.exp(shifted); expected /= expected.sum(axis=1, keepdims=True)
                expected_loss = -np.log(expected[np.arange(n), labels]).mean()
                expected_gradient = expected.copy(); expected_gradient[np.arange(n), labels] -= 1; expected_gradient /= n
                np.testing.assert_allclose(probabilities, expected, rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(loss, expected_loss, rtol=1e-5, atol=1e-6)
                return {'loss': np.asarray(loss, dtype=np.float64), 'gradient': gradient, 'probabilities': probabilities}
            case('xent-'+str(n), xent)
        for n in (1, 2, 3, 1000, 65537):
            def permutation(n=n):
                values = {}
                for epoch in (0, 1):
                    order = np.empty(n, dtype=np.int32)
                    binding.x_cnn_epoch_rows(order.ctypes.data, [n, epoch, 1, 0x12345678, 0x9ABCDEF0])
                    np.testing.assert_array_equal(np.sort(order), np.arange(n, dtype=np.int32))
                    values['epoch-'+str(epoch)] = order
                return values
            case('epoch-'+str(n), permutation)
        for index, betas in enumerate(((0.9, 0.999), (0.5, 0.9999))):
            def hyper(betas=betas):
                values = {}
                for step in (1, 2, 3, 31, 1024, 99999, 100000):
                    output = np.empty((1, 9), dtype=np.float64)
                    binding.x_cnn_adam_hyper_d(output.ctypes.data, [step, 1], [0.001, *betas, 1e-8, 0.01, 1.0])
                    if not np.isfinite(output).all(): raise AssertionError('nonfinite Adam hyperparameter')
                    # Primary requirement here is current-version cross-column bits;
                    # this deliberately does not require legacy float64 scalar bits.
                    values['step-'+str(step)] = output
                return values
            case('adam-hyper-'+str(index), hyper)
        expected_count = 10
    elif args.suite == 'cnn-training':
        for optimizer in ('sgd', 'adam'):
            def train(optimizer=optimizer):
                x = (((np.arange(65*16, dtype=np.int32) % 37)-18).astype(np.float32)/np.float32(16)).reshape(65, 16)
                y = np.arange(65, dtype=np.int32) % 3
                model = ml.CNNClassifier(input_shape=(1, 4, 4), conv_channels=(2,), kernel_size=3,
                                        pool_size=2, batch_size=33, max_iter=2, random_state=7,
                                        optimizer=optimizer, learning_rate=0.01, shuffle=True).fit(x, y)
                values = {'losses': np.asarray(model.losses_, dtype=np.float64),
                          'loss_curve': np.asarray(model.loss_curve_, dtype=np.float64),
                          'probabilities': np.asarray(model.predict_proba(x))}
                for index, array in enumerate(model.weights()): values['weight-'+str(index)] = np.asarray(array)
                for index, array in enumerate(model._bufs): values['optimizer-'+str(index)] = np.asarray(array)
                if not all(np.isfinite(v).all() for v in values.values()): raise AssertionError('nonfinite CNN state')
                return values
            case('cnn-'+optimizer, train)
        expected_count = 2
    else:
        from mojolearn import _expansion_linear as linear
        x = (((np.arange(513*17, dtype=np.int32) % 43)-21).astype(np.float32)/np.float32(16)).reshape(513, 17)
        y = np.arange(513, dtype=np.int32) % 3
        weights = ((np.arange(513, dtype=np.int32) % 4)+1).astype(np.float32)/np.float32(2)
        parameters = dict(max_iter=3, tol=None, shuffle=True, random_state=7, class_weight={0: 0.5, 1: 1.0, 2: 2.0})
        configs = [('sgd', ml.SGDClassifier, dict(batch_size=128, learning_rate='constant', eta0=0.01)),
                   ('perceptron', ml.Perceptron, dict(batch_size=256, eta0=0.01)),
                   ('pa', ml.PassiveAggressiveClassifier, dict(batch_size=256, C=0.1))]
        for name, constructor, extra in configs:
            def train(constructor=constructor, extra=extra):
                model = constructor(**parameters, **extra)
                bind(linear._fit_module(model, linear.ALGO_SGD), 'x_linear')
                model.fit(x, y, sample_weight=weights)
                values = {'coef': np.asarray(model.coef_), 'intercept': np.asarray(model.intercept_),
                          'n_iter': np.asarray(model.n_iter_, dtype=np.int64), 't': np.asarray(model.t_, dtype=np.float64),
                          'prediction': np.asarray(model.predict(x[:33]))}
                if values['coef'].shape != (3, 17) or not 1 <= model.n_iter_ <= 3:
                    raise AssertionError('SGD OVR shape/iteration coverage missing')
                if not all(np.isfinite(v).all() for v in values.values()):
                    raise AssertionError('nonfinite SGD fitted state')
                return values
            case('sgd-ovr-'+name, train)
        def refuse_nan():
            bad = x.copy(); bad[-1, -1] = np.nan
            try: ml.SGDClassifier(max_iter=1, batch_size=128, random_state=7).fit(bad, y)
            except ValueError: return {'refused': np.asarray(1, dtype=np.int32)}
            raise AssertionError('SGD accepted NaN input')
        case('sgd-nan-refusal', refuse_nan)
        def refuse_overflow():
            # Finite inputs pass the public finite-input scan. Arithmetic overflow
            # must reach the SGD status/error path, not silently return a model.
            huge = np.full((513, 17), 1e20, dtype=np.float32)
            target = np.full(513, 1e20, dtype=np.float32)
            try:
                ml.SGDRegressor(max_iter=2, tol=None, batch_size=256, learning_rate='constant',
                                eta0=1e30, random_state=7).fit(huge, target)
            except ValueError as exc:
                if 'Floating-point' not in str(exc): raise AssertionError('wrong overflow refusal: '+str(exc))
                return {'refused': np.asarray(1, dtype=np.int32)}
            raise AssertionError('SGD accepted nonfinite fitted state')
        case('sgd-overflow-refusal', refuse_overflow)
        expected_count = 5
    source_check(root, args.sha)
    for binding in report['bindings'].values():
        if hashlib.sha256(Path(binding['path']).read_bytes()).hexdigest() != binding['sha256']:
            raise RuntimeError('binding changed during fixture execution')
    report['expected_cases'] = expected_count
    report['status'] = 'PASS' if len(report['cases']) == expected_count and all(c['status'] == 'PASS' for c in report['cases'].values()) else 'FAIL'
    dump(receipt, report)
    print('CNN_SGD_GATE', args.suite, report['status'], 'cases='+str(len(report['cases'])), 'timing_samples=0', flush=True)
    return 0 if report['status'] == 'PASS' else 1

if __name__ == '__main__': sys.exit(main())
