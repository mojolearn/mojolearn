# SPDX-License-Identifier: Apache-2.0
"""Named OPTICS output fields over its unchanged native execution entry."""
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.optics_fit_api import OpticsFitResult, OPTICS_FIT_MODE, OPTICS_FIT_RUN
from x_cluster.variants_dispatch import KMeansVariantsBackend
from x_cluster.entries import ENTRY_OPTICS


def _word(address: Int, index: Int) -> Int:
    return Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=address)[index]


def optics_fit_dispatch[K: KMeansVariantsBackend](operation: Int,
    aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) -> Int:
    try:
        if result == 0 or (na > 0 and aa == 0) or (np > 0 and pp == 0) or (nf > 0 and ff == 0):
            raise Error("OPTICS fit ABI: missing request or result span")
        var answer = 0
        if operation == OPTICS_FIT_MODE:
            if na != 0 or np != 0 or nf != 0:
                raise Error("OPTICS fit ABI: mode takes no arguments")
            answer = K.mode()
        elif operation == OPTICS_FIT_RUN:
            if na != 2 or np != 8 or nf != 4 or _word(aa, 1) == 0:
                raise Error("OPTICS fit ABI: invalid fit request")
            var ip = List[Int]()
            for i in range(1, 8):  # metadata: original OPTICS entry's seven words
                ip.append(_word(pp, i))
            var fp = List[Float64]()
            for i in range(4):  # metadata: max_eps, xi, eps, metric power
                fp.append(Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)[i])
            # The backend supplies transport only. ENTRY_OPTICS still calls
            # optics_graph then optics_xi_ops or optics_dbscan_ops unchanged.
            var previous = K.execute(ENTRY_OPTICS, _word(aa, 0), _word(pp, 0), 0, 0, ip, fp)
            if len(previous.f) != 2 or len(previous.i) != 4 or len(previous.s) != 0:
                raise Error("OPTICS fit ABI: invalid result slots")
            var value = OpticsFitResult()
            value.core_distances = previous.f.pop(0)
            value.reachability = previous.f.pop(0)
            value.ordering = previous.i.pop(0)
            value.predecessor = previous.i.pop(0)
            value.labels = previous.i.pop(0)
            value.hierarchy = previous.i.pop(0)
            MutPointer[OpticsFitResult, MutUntrackedOrigin](unsafe_from_address=_word(aa, 1))[] = value^
            _ = ip^
            _ = fp^
        else:
            raise Error("OPTICS fit ABI: unknown operation")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = answer
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1
