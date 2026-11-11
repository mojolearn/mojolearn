# SPDX-License-Identifier: Apache-2.0
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.hdbscan_query_api import (
    HdbscanQueryState, HDBSCAN_QUERY_MODE, HDBSCAN_QUERY_HOST,
    HDBSCAN_QUERY_APPROXIMATE, HDBSCAN_QUERY_MEMBERSHIP, HDBSCAN_QUERY_ALL_POINTS,
)


trait HdbscanQueryBackend:
    @staticmethod
    def mode() -> Int: ...

    @staticmethod
    def host() -> Bool: ...

    @staticmethod
    def run(mut state: HdbscanQueryState) raises -> Int: ...


def hdbscan_query_dispatch[K: HdbscanQueryBackend](operation: Int,
    aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) -> Int:
    try:
        if result == 0:
            raise Error("HDBSCAN query ABI: missing result span")
        var answer = 0
        if operation == HDBSCAN_QUERY_MODE or operation == HDBSCAN_QUERY_HOST:
            if na != 0 or np != 0 or nf != 0:
                raise Error("HDBSCAN query ABI: metadata takes no arguments")
            answer = K.mode() if operation == HDBSCAN_QUERY_MODE else Int(K.host())
        elif operation == HDBSCAN_QUERY_APPROXIMATE or operation == HDBSCAN_QUERY_MEMBERSHIP or operation == HDBSCAN_QUERY_ALL_POINTS:
            if na != 1 or np != 0 or nf != 0 or aa == 0:
                raise Error("HDBSCAN query ABI: invalid prepared request")
            var address = Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=aa)[0]
            if address == 0:
                raise Error("HDBSCAN query ABI: missing prepared owner")
            var state = MutPointer[HdbscanQueryState, MutUntrackedOrigin](unsafe_from_address=address)
            if state[].operation != operation or state[].host != K.host():
                raise Error("HDBSCAN query ABI: prepared request belongs to a different route")
            answer = K.run(state[])
        else:
            raise Error("HDBSCAN query ABI: unknown operation")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = answer
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1
