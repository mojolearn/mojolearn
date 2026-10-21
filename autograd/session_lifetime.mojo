# SPDX-License-Identifier: Apache-2.0
# The binding registers this native value with PythonModuleBuilder
# (session_lifetime_pyapi.mojo builds it). Releasing a Session's last
# reference closes its executor without a Python __del__ callback.
from std.sys.compile import is_defined
from sequence.exec_trait import Exec
from autograd.ops_api import ops_session_close


# MODE is part of the type's name on purpose (2026-10-21): the Python type
# registry is one per process, and `import mojolearn` loads the fast AND the
# identical build of the x_sequence binding. With the same type id in both,
# the second `add_type` aborted the import ("multiple Python type objects
# bound to Mojo type"). The numeric mode in the id keeps the two apart.
comptime SESSION_MODE: Int = 1 if is_defined["MOJOLEARN_NUMERIC_IDENTICAL"]() else 0


struct SessionLifetime[E: Exec & Movable & Deinitable, KIND: Int, MODE: Int = SESSION_MODE](Movable, Writable):
    var sid: Int

    def __init__(out self, sid: Int):
        self.sid = sid

    def write_to(self, mut writer: Some[Writer]):
        writer.write("_OpsSessionLifetime(open=", self.sid != 0, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("_OpsSessionLifetime(open=", self.sid != 0, ")")

    def close(mut self) raises:
        var sid = self.sid
        self.sid = 0
        if sid != 0:
            ops_session_close[Self.E, Self.KIND](sid)

    def __deinit__(deinit self):
        if self.sid != 0:
            try:
                ops_session_close[Self.E, Self.KIND](self.sid)
            except:
                pass
