# SPDX-License-Identifier: Apache-2.0
"""Direct Program construction shares the capture owner's native boundary.

The owner (providers/python/autograd_program_owner_pyapi.mojo
program_initialize_owner) admits exact metadata only: the handle, shapes,
parameters and kinds are exact native integers (or NumPy integer scalars) in
native lists or tuples, so no ``__int__``/``__index__``/``__iter__`` of a
caller's object runs inside the call; anything else is a TypeError."""
import numpy as np
import pytest

from mojolearn import ops
from mojolearn._verify_native_runtime import audit_native_call


@pytest.mark.parametrize("legacy", [False, True])
def test_direct_program_constructor_admits_exact_metadata_and_runs_native(legacy):
    inputs = np.array([[-2, 1], [3, -4]], dtype="f4")
    with ops.Session() as session:
        x = session.tensor(inputs)
        result = session.relu(x)
        captured = session.capture(inputs=[x], outputs=[result])
        # Lists and tuples mix freely; the four-item ABI keeps value inputs.
        info = [captured._id, [[2, 2]], ((2, np.int64(2)),), []]
        if not legacy:
            info.append((0,))
        program = ops.Program.__new__(ops.Program)
        audit_native_call(program.__init__, session, info).assert_native()
        assert program._s is session
        assert program._id == captured._id
        assert program.input_shapes == program.output_shapes == ((2, 2),)
        assert program.input_kinds == (0,)
        assert program.params == ()
        actual = audit_native_call(program.run, inputs).assert_native()
        np.testing.assert_array_equal(actual, np.maximum(inputs, np.float32(0)))


@pytest.mark.parametrize("info", [
    lambda pid: [str(pid), [[2, 2]], [[2, 2]], []],
    lambda pid: [pid, [[2.0, 2]], [[2, 2]], []],
    lambda pid: [pid, [["2", 2]], [[2, 2]], []],
    lambda pid: [pid, [[2, 2]], [[2, 2]], [], ["0"]],
])
def test_direct_program_constructor_refuses_converted_metadata(info):
    inputs = np.array([[-2, 1], [3, -4]], dtype="f4")
    with ops.Session() as session:
        x = session.tensor(inputs)
        captured = session.capture(inputs=[x], outputs=[session.relu(x)])
        program = ops.Program.__new__(ops.Program)
        result = audit_native_call(program.__init__, session, info(captured._id))
        assert isinstance(result.exception, TypeError), result.report()
        assert "must be an integer" in str(result.exception)
        assert not result.report()["python_frames"], result.report()


def test_direct_program_constructor_preserves_integer_input_metadata():
    weights = np.array([[1, 2], [3, 4], [5, 6]], dtype="f4")
    inputs = np.array([[0, 2], [1, 0]], dtype="i4")
    with ops.Session() as session:
        ids = session.ids(inputs)
        w = session.tensor(weights)
        output = session.embedding(w, ids)
        captured = session.capture(inputs=[ids], outputs=[output])
        info = [captured._id, captured.input_shapes, captured.output_shapes,
                tuple(captured.params), (1,)]
        program = ops.Program.__new__(ops.Program)
        audit_native_call(program.__init__, session, info).assert_native()
        assert program.params == captured.params
        assert program.input_kinds == (1,)
        actual = audit_native_call(program.run, inputs).assert_native()
        # Session.embedding documents one row per id, flattened row-major.
        np.testing.assert_array_equal(actual, weights[inputs.reshape(-1)])


@pytest.mark.parametrize("legacy", [False, True])
def test_direct_program_constructor_refuses_metadata_iterators(legacy):
    inputs = np.array([[-2, 1], [3, -4]], dtype="f4")
    with ops.Session() as session:
        x = session.tensor(inputs)
        output = session.relu(x)
        captured = session.capture(inputs=[x], outputs=[output])
        info = [captured._id, iter([iter([2, 2])]),
                iter([iter([2, 2])]), iter(())]
        if not legacy:
            info.append(iter([0]))
        program = ops.Program.__new__(ops.Program)
        result = audit_native_call(program.__init__, session, info)
        assert isinstance(result.exception, TypeError), result.report()
        assert "native list or tuple" in str(result.exception)
        assert not result.report()["python_frames"], result.report()
        assert "_id" not in vars(program)


@pytest.mark.parametrize("invalid, expected", [
    ("invalid", TypeError), (None, TypeError),
    (float("nan"), TypeError), (float("inf"), TypeError), (0, ValueError),
])
def test_direct_program_constructor_invalid_metadata_has_no_python_runtime(invalid, expected):
    with ops.Session() as session:
        value = ops.Program.__new__(ops.Program)
        result = audit_native_call(value.__init__, session, [invalid, [], [], []])
        assert isinstance(result.exception, expected), result.report()
        for key in ("python_frames", "forbidden_native_calls", "wrapper_work",
                    "unobserved_existing_threads"):
            assert not result.report()[key], result.report()
