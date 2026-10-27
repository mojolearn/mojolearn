#!/usr/bin/env python3
"""Bounded native lock/public block ownership check; root runs built Mac."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import threading
import traceback


def qualify():
    os.environ['MOJOLEARN_NEURAL_NATIVE_DISPATCH'] = '1'
    os.environ['MOJOLEARN_GENERATION_DEVICE'] = 'gpu'
    import numpy as np
    from mojolearn import TransformerBlock
    from mojolearn.models import causal_lm  # connect native runtime and lock class
    from mojolearn._native_operation_lock import _OperationLock
    from mojolearn.tests.test_neural_block_native import block_inputs
    from mojolearn._verify_native_runtime import audit_native_call
    lock = _OperationLock()
    assert lock.acquire() and lock.acquire() and lock._is_owned()
    events = []
    entered = threading.Event()
    done = threading.Event()
    failures = []
    def contender():
        try:
            assert lock.acquire(False) is False
            assert lock.acquire(True, .01) is False
            try:
                lock.release()
            except RuntimeError:
                events.append('wrong-thread-release-refused')
            else:
                raise AssertionError('wrong thread released owner')
            entered.set()
            with lock:
                events.append('contender-acquired')
            done.set()
        except BaseException:
            failures.append(traceback.format_exc())
            entered.set(); done.set()
    thread = threading.Thread(target=contender)
    thread.start()
    assert entered.wait(10), 'contender did not test blocking'
    assert not failures, failures
    lock.release()
    assert not done.is_set(), 'one release lost recursive ownership'
    lock.release()
    assert done.wait(10), 'native lock did not wake waiting owner'
    thread.join(10)
    assert not failures and not thread.is_alive(), failures
    assert events == ['wrong-thread-release-refused','contender-acquired']
    try:
        with lock:
            raise ValueError('test cleanup')
    except ValueError:
        pass
    assert not lock._is_owned()
    with tempfile.TemporaryDirectory(prefix='ml-native-lock-') as folder:
        _, weights, options, x = block_inputs(Path(folder), 'llama')
        block = TransformerBlock(weights, numeric_mode='identical', **options)
        assert type(block._runtime_lock) is _OperationLock
        expected = np.asarray(block.forward(x)).copy()
        calls = []
        def poison(*args, **kwargs):
            calls.append('Python lock callback')
            raise AssertionError('Python lock hook invoked')
        old = _OperationLock.acquire
        try:
            _OperationLock.acquire = poison
            # Internal product calls use the admitted native owner directly;
            # changing the Python-facing method must not drive execution.
            audit = audit_native_call(block.forward, x)
            actual = np.asarray(audit.assert_native())
        finally:
            _OperationLock.acquire = old
        assert expected.tobytes() == actual.tobytes()
        assert not calls
        owner = block._runtime_lock
        class UnsupportedLock:
            def acquire(self):
                return poison()
            def release(self):
                return poison()
        block._runtime_lock = UnsupportedLock()
        try:
            try:
                block.forward(x)
            except TypeError:
                pass
            else:
                raise AssertionError('unsupported lock protocol admitted')
        finally:
            block._runtime_lock = owner
        assert not calls
        block.forward(x)
    return dict(status='passed', ownership_events=events, public_native_audit=audit.report(),
                pending=['native mixed-layer engine integration', 'OS signal interruption',
                         'long contention stress/performance', 'other targets'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:
        report=qualify()
    except BaseException:
        report=dict(status='failed',traceback=traceback.format_exc())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],evidence=str(a.output))))
    return report['status']!='passed'


if __name__=='__main__':
    raise SystemExit(main())
