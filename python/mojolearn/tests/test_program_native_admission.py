# SPDX-License-Identifier: Apache-2.0
"""Actual public candidate tests; authored only until root compiles the binding."""
import gc

import numpy as np
import pytest

from mojolearn import Array, ops
from mojolearn._verify_native_runtime import audit_native_call


def refused_without_callbacks(call, *args):
    audit = audit_native_call(call, *args)
    report = audit.report()
    assert isinstance(audit.exception, TypeError), report
    for name in ('python_frames', 'wrapper_work', 'forbidden_native_calls', 'unobserved_existing_threads'):
        assert not report[name], report


def test_program_and_tensor_refuse_foreign_protocols_before_touching_them():
    seen = []

    class Protocol:
        def __getattribute__(self, name):
            seen.append(name)
            raise AssertionError('foreign attribute callback')

        def __float__(self):
            seen.append('float')
            raise AssertionError('foreign scalar callback')

    class FloatHook(float):
        def __float__(self):
            seen.append('float subclass')
            raise AssertionError('foreign scalar callback')

    class ArrayHook(np.ndarray):
        def __getattribute__(self, name):
            seen.append('ndarray ' + name)
            raise AssertionError('foreign ndarray hook')

    class ListHook(list):
        def __len__(self):
            seen.append('list length')
            raise AssertionError('foreign container hook')

    bad_array = np.ones((2, 3), 'f4').view(ArrayHook)
    values = (Protocol(), FloatHook(1.0), bad_array, ListHook([[1.0, 2.0, 3.0]]))
    with ops.Session() as session:
        tensor = session.tensor(np.ones((2, 3), 'f4'))
        program = session.capture(inputs=[tensor], outputs=[session.relu(tensor)])
        for value in values:
            refused_without_callbacks(session.tensor, value)
            refused_without_callbacks(program.run, value)
        refused_without_callbacks(session.tensor, [[FloatHook(1.0)]])
        assert not seen
        actual = audit_native_call(program.run, np.full((2, 3), 2.0, 'f4')).assert_native()
        np.testing.assert_array_equal(actual, 2.0)


def test_program_native_array_and_lists_preserve_values_ownership_and_refresh():
    x = np.arange(6, dtype='f4').reshape(2, 3)
    with ops.Session() as session:
        tensor = audit_native_call(session.tensor, Array.from_list(x.tolist(), '<f4')).assert_native()
        np.testing.assert_array_equal(session.numpy(tensor), x)
        program = session.capture(inputs=[tensor], outputs=[session.relu(tensor)])
        for current in (x.tolist(), Array.from_list((x + 1).tolist(), '<f4'), np.asfortranarray(x + 2, dtype='f8')):
            actual = audit_native_call(program.run, current).assert_native()
            expected = np.asarray(current, dtype='f4')
            assert actual.tobytes() == expected.tobytes()
        retained = actual.tobytes()
        del current
        gc.collect()
        assert actual.tobytes() == retained
        with pytest.raises(ValueError, match='rectangular'):
            session.tensor([[1.0], [2.0, 3.0]])
        np.testing.assert_array_equal(program.run(x), x)
