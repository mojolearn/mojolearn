# SPDX-License-Identifier: Apache-2.0
# PythonModuleBuilder owns this native value. Releasing a Session's last
# reference closes its executor without a Python __del__ callback.
from std.python import Python, PythonObject
from sequence.exec_trait import Exec
from autograd.ops_api import ops_session_open, ops_session_close
from autograd.program_arrays import initialize_program_arrays


struct SessionLifetime[E: Exec & Movable & Deinitable, KIND: Int](Movable):
    var sid: Int

    def __init__(out self, sid: Int):
        self.sid = sid

    def close(mut self) raises:
        var sid = self.sid
        self.sid = 0
        if sid != 0:
            ops_session_close[E, KIND](sid)

    def __deinit__(deinit self):
        if self.sid != 0:
            try:
                ops_session_close[E, KIND](self.sid)
            except:
                pass


def ops_session_create_py[E: Exec & Movable & Deinitable, KIND: Int](
    var ex: E, record: PythonObject,
) raises -> PythonObject:
    initialize_program_arrays()
    var sid = ops_session_open[E, KIND](ex^, Bool(py=record))
    var owner = SessionLifetime[E, KIND](sid)
    var result = Python.list()
    result.append(sid)
    result.append(PythonObject(alloc=owner^))
    return result
