# SPDX-License-Identifier: Apache-2.0
# The provider boundary retains Python owners, but selects no Python callable.
# One public Session operation: when the calling image owns the Session
# registry (the resource's scope words equal this image's registry token and
# executor kind), the door calls the typed session operation directly; only a
# session owned by another compiled image (the other numeric mode, or the
# host image) is forwarded through that image's synchronous descriptor. The
# GIL remains held here either way, exactly as on the descriptor route;
# existing native operation owners decide where execution can release it.
# The PACK half of the former `providers/python/autograd_session_public_request_pyapi.mojo` (fence-flip-2 split, 2026-11-12);
# the UNPACK half is `providers/python/autograd_session_public_request_pyapi_unpack.mojo`.
#
# Lane restore-executor-fast-run (2026-11-13): the same-image direct call
# restores the 0.8.55 shape of one native call per public Session operation.
# Since d2f14a3de4 / ec806eaf6f / 1edde63a6a (2026-10-27) every call, including
# a call into this very image, went door -> descriptor record -> C ABI entry
# (`session_public_native_v1`) -> `ops_session_call_fields_py`, with no A/B.
# Board cell ops-session-step NV 2.30 ms (0.8.55) -> 2.71 (0.8.66) -> 2.60
# (candidate 4c2340386abc, 1.13x). A/B owed (owed row
# restore-executor-session-call, b arm session-call-descriptor):
# -D MOJOLEARN_SESSION_CALL_DESCRIPTOR forwards every call through the
# descriptor as before (B). Same session operation either way: no bit change.
from std.python import Python, PythonObject
from std.sys import is_defined
from bindings.py_balanced import py_type
from sequence.exec_trait import Exec
from autograd.ops_api import ops_session_registry_token
from providers.python.neural_descriptor_pyapi_door import (
    NeuralDescriptorProvider, NativeDescriptorResource, neural_descriptor_execute,
)
from providers.python.autograd_session_frontend_pyapi_door import ops_session_fields
from providers.python.autograd_ops_frontend_pyapi_door import ops_session_call_fields_py

from providers.python.autograd_session_public_request_pyapi_unpack import (
    NativeSessionPublicRequest,
    SESSION_PUBLIC_ABI_NAME,
)

comptime SESSION_CALL_DESCRIPTOR = is_defined["MOJOLEARN_SESSION_CALL_DESCRIPTOR"]()


def ops_session_public_call_py[E: Exec & Movable & Deinitable, KIND: Int](
    session: PythonObject, operation: PythonObject, args: PythonObject,
) raises -> PythonObject:
    if py_type(operation) is not py_type(PythonObject("")) or py_type(args) is not py_type(Python.tuple()):
        raise Error("TypeError: Session requires an exact operation string and argument tuple")
    var fields = ops_session_fields(session)
    var provider = fields["_public_provider"]
    var public_owner = provider.downcast_value_ptr[NeuralDescriptorProvider]()
    var lifetime = fields["_lifetime"]
    var resource = lifetime.downcast_value_ptr[NativeDescriptorResource]()
    var model_owner = resource[].provider.downcast_value_ptr[NeuralDescriptorProvider]()
    if public_owner[].record[0].name != SESSION_PUBLIC_ABI_NAME or public_owner[].module is not model_owner[].module:
        raise Error("TypeError: Session public provider differs from its native registry owner")
    comptime if not SESSION_CALL_DESCRIPTOR:
        # The same admission `session_public_native_v1` applies on entry:
        # the scope words name this registry image and this executor kind.
        if (len(resource[].context) == 2 and resource[].context[0] == ops_session_registry_token()
                and resource[].context[1] == KIND):
            var answer = ops_session_call_fields_py[E, KIND](session, operation, args, fields)
            _ = lifetime  # the resource owner stays pinned through the call
            _ = provider
            return answer
    # Stable native storage across the synchronous image boundary; each field
    # keeps its original owner alive until the completed result is returned.
    var requests = List[NativeSessionPublicRequest]()
    requests.append(NativeSessionPublicRequest(session, fields, operation, args))
    var addresses: List[Int] = [Int(MutPointer(to=requests[0]))]
    _ = neural_descriptor_execute(provider, 0, addresses, resource[].context, List[Float64](), False)
    return requests[0].result
