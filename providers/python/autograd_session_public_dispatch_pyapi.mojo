# SPDX-License-Identifier: Apache-2.0
# Provider-boundary dispatch, deliberately separate from the arithmetic ABI.
from core.native_descriptor_abi import NativeDescriptorCall, NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report, native_descriptor_words
from autograd.ops_api import ops_session_registry_token
from sequence.exec_trait import Exec
from providers.python.autograd_ops_frontend_pyapi import ops_session_call_fields_py
from providers.python.autograd_session_public_request_pyapi import NativeSessionPublicRequest, SESSION_PUBLIC_ABI_NAME


def session_public_native_v1[E: Exec & Movable & Deinitable, KIND: Int](
    op: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError,
) abi("C") -> Int:
    try:
        var addresses = native_descriptor_words(aa, na)
        var scope = native_descriptor_words(pp, np)
        if op != 0 or na != 1 or np != 2 or nf != 0 or result == 0:
            raise Error("native Session: invalid public request")
        if addresses[0] == 0 or scope[0] != ops_session_registry_token() or scope[1] != KIND:
            raise Error("native Session: public request belongs to another registry or executor")
        var request = MutPointer[NativeSessionPublicRequest, MutUntrackedOrigin](unsafe_from_address=addresses[0])
        request[].result = ops_session_call_fields_py[E, KIND](
            request[].session, request[].operation, request[].args, request[].fields)
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = 0
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1


struct NativeSessionPublicAPI[E: Exec & Movable & Deinitable, KIND: Int](Defaultable, Movable):
    var name: String
    var entries: List[NativeDescriptorCall]

    def __init__(out self):
        self.name = String(SESSION_PUBLIC_ABI_NAME)
        self.entries = List[NativeDescriptorCall]()
        self.entries.append(session_public_native_v1[E, KIND])
