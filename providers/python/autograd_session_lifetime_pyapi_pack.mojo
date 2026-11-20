# SPDX-License-Identifier: Apache-2.0
# Python door for SessionLifetime: opens the native session and hands the
# fixed [sid, owner] ABI result to the binding.
from std.python import Python, PythonObject
from sequence.exec_trait import Exec
from autograd.ops_api import ops_session_open, ops_session
from providers.python.autograd_program_arrays_pyapi_door import initialize_program_arrays
from autograd.session_lifetime import SessionLifetime


def ops_session_create_py[E: Exec & Movable & Deinitable, KIND: Int](
    var ex: E, record: PythonObject,
) raises -> PythonObject:
    initialize_program_arrays()
    var sid = ops_session_open[E, KIND](ex^, Bool(py=record))
    # A public Session's device executor is on the sequence context, not the
    # x_cnn one: its CNN block ops drain the session queue before each x_cnn
    # entry (autograd/block_step_ops.mojo BlockStepScratch.cnn_other_queue).
    # Nothing on a host session reads the flag.
    var opened = ops_session[E, KIND](sid)
    opened[].steps.cnn_other_queue = True
    var owner = SessionLifetime[E, KIND](sid)
    # Construct the fixed ABI result directly with the supported native
    # list constructor, transferring the native owner exactly once.
    return Python.list(PythonObject(sid), PythonObject(alloc=owner^))
