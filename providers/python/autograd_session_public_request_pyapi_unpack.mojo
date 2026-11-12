# SPDX-License-Identifier: Apache-2.0
# The provider boundary retains Python owners, but selects no Python callable.
# Its synchronous descriptor forwards one whole Session operation to the exact
# compiled image that owns the Session registry. The GIL remains held here;
# existing native operation owners decide where execution can release it.
from std.python import Python, PythonObject
from bindings.py_balanced import py_type
from providers.python.neural_descriptor_pyapi import (
    NeuralDescriptorProvider, NativeDescriptorResource, neural_descriptor_entry,
    neural_descriptor_connect, neural_descriptor_execute,
)
from providers.python.autograd_session_frontend_pyapi import ops_session_fields

comptime SESSION_PUBLIC_ABI_NAME = "mojolearn.session.public.v1"


struct NativeSessionPublicRequest(Movable):
    var session: PythonObject
    var fields: PythonObject
    var operation: PythonObject
    var args: PythonObject
    var result: PythonObject

    def __init__(out self, session: PythonObject, fields: PythonObject,
                 operation: PythonObject, args: PythonObject):
        self.session = session
        self.fields = fields
        self.operation = operation
        self.args = args
        self.result = PythonObject(None)


def ops_session_public_connect_py(module: PythonObject) raises -> PythonObject:
    # Import-time registration only. Session construction retains this owner.
    return neural_descriptor_connect(neural_descriptor_entry(module, "session_public_api_v1"),
                                     String(SESSION_PUBLIC_ABI_NAME), 1)
