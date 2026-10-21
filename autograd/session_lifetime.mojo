# SPDX-License-Identifier: Apache-2.0
# The binding registers this native value with PythonModuleBuilder
# (session_lifetime_pyapi.mojo builds it). Releasing a Session's last
# reference closes its executor without a Python __del__ callback.
from sequence.exec_trait import Exec
from autograd.ops_api import ops_session_close


struct SessionLifetime[E: Exec & Movable & Deinitable, KIND: Int](Movable, Writable):
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
