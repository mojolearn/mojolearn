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
# (candidate 4c2340386abc, 1.13x). Same session operation either way: no bit
# change. The B arm (every call through the descriptor) lost A/B abn9 and is
# deleted; a Session of another image still takes the descriptor below.
from std.python import Python, PythonObject
from bindings.py_balanced import py_item, py_type
from sequence.exec_trait import Exec
from bindings.autograd_keys_unpack import autograd_keys
from autograd.ops_api import ops_session_registry_token
from providers.python.neural_descriptor_pyapi_door import (
    NeuralDescriptorProvider, NativeDescriptorResource, neural_descriptor_execute,
)
from providers.python.autograd_session_frontend_pyapi_door import ops_session_fields
from providers.python.autograd_program_arrays_pyapi_door import program_builtin_types
from providers.python.autograd_ops_frontend_pyapi_door import ops_session_call_admitted_py

from providers.python.autograd_session_public_request_pyapi_unpack import (
    NativeSessionPublicRequest,
    SESSION_PUBLIC_ABI_NAME,
)

# TOMBSTONE: MOJOLEARN_SESSION_CALL_DESCRIPTOR (unmeasured B arm: every Session call through the descriptor; EXP:2395 RESTORED the direct typed call as default) deleted 2026-11-15 by lane/expdel2-08-base; code recoverable at 8fe6eaaa01.
# Restore: git apply experiments/removed/MOJOLEARN_SESSION_CALL_DESCRIPTOR.patch; record in docs/TOMBSTONES.md.
# RESTORED 2026-11-15 by lane/expdel2-restore-owed: owed B arm (owed-pre0867.json), under four weeks old; the deletion was premature.
# TOMBSTONE: MOJOLEARN_SESSION_CALL_DESCRIPTOR (B arm loses: every same-image Session call through the descriptor, zero-filled read-back arrays; A/B abn9 NVIDIA L40S, ops-session-step 2.63 -> 2.93 ms 1.11, program-run 1.00 arm/base) deleted 2026-11-21 by lane/fx-verdicts7; code recoverable at 519388d23a.
# Restore: git apply experiments/removed/MOJOLEARN_SESSION_CALL_DESCRIPTOR.fx-verdicts7.patch; record in docs/TOMBSTONES.md.


def ops_session_public_call_py[E: Exec & Movable & Deinitable, KIND: Int](
    session: PythonObject, operation: PythonObject, args: PythonObject,
) raises -> PythonObject:
    # Lane fix-program-run-per-call (2026-11-14): the exact builtin types
    # connected once at Session setup (was a throwaway str, tuple and int
    # built per Session operation to read their types; about 25 operations
    # per ops-session-step round).
    var types = program_builtin_types()
    if py_type(operation) is not types[].str_type or py_type(args) is not types[].tuple_type:
        raise Error("TypeError: Session requires an exact operation string and argument tuple")
    var fields = ops_session_fields(session)
    # Held keys (bindings/autograd_keys_unpack.mojo): the three reads below
    # each built, hashed and freed a Python str per Session operation.
    var nk = autograd_keys()
    var lifetime = py_item(fields, nk[].k__lifetime)
    var resource = lifetime.downcast_value_ptr[NativeDescriptorResource]()
    # TOMBSTONE: MOJOLEARN_SESSION_CALL_DESCRIPTOR (B arm loses: every same-image Session call through the descriptor, zero-filled read-back arrays; A/B abn9 NVIDIA L40S, ops-session-step 2.63 -> 2.93 ms 1.11, program-run 1.00 arm/base) deleted 2026-11-21 by lane/fx-verdicts7; code recoverable at 519388d23a.
    # Restore: git apply experiments/removed/MOJOLEARN_SESSION_CALL_DESCRIPTOR.fx-verdicts7.patch; record in docs/TOMBSTONES.md.
    # The same admission `session_public_native_v1` applies on entry:
    # the scope words name this registry image and this executor kind.
    if (len(resource[].context) == 2 and resource[].context[0] == ops_session_registry_token()
            and resource[].context[1] == KIND):
        # Lane fix-executor-session-rows (2026-11-13): the handle
        # admission of `ops_session_call_fields_py`, read from the
        # resource already in hand (one downcast per call, not two),
        # and no public-provider read: the descriptor below is the only
        # user of `_public_provider`, so its downcasts, ABI-name string
        # compare and module identity check run only on that route.
        var sid = py_item(fields, nk[].k__id)
        if sid is not None:
            if py_type(sid) is not types[].int_type:
                raise Error("TypeError: Session handle must be a native integer")
            var admitted_id = resource[].handle
            if admitted_id == 0 or admitted_id != Int(py=sid):
                raise Error("TypeError: Session handle differs from its native registry owner")
        var answer = ops_session_call_admitted_py[E, KIND](session, operation, args, fields)
        _ = lifetime  # the resource owner stays pinned through the call
        return answer
    var provider = py_item(fields, nk[].k__public_provider)
    var public_owner = provider.downcast_value_ptr[NeuralDescriptorProvider]()
    var model_owner = resource[].provider.downcast_value_ptr[NeuralDescriptorProvider]()
    if public_owner[].record[0].name != SESSION_PUBLIC_ABI_NAME or public_owner[].module is not model_owner[].module:
        raise Error("TypeError: Session public provider differs from its native registry owner")
    # Stable native storage across the synchronous image boundary; each field
    # keeps its original owner alive until the completed result is returned.
    var requests = List[NativeSessionPublicRequest]()
    requests.append(NativeSessionPublicRequest(session, fields, operation, args))
    var addresses: List[Int] = [Int(MutPointer(to=requests[0]))]
    _ = neural_descriptor_execute(provider, 0, addresses, resource[].context, List[Float64](), False)
    _ = lifetime
    return requests[0].result
