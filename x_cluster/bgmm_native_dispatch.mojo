# SPDX-License-Identifier: Apache-2.0
"""The BGMM descriptor over the unchanged cluster execution owner
(core/bgmm_native_api.mojo). Transport only: ENTRY_BGMM and
ENTRY_BGMM_SCORE run exactly as `x_cluster_call` runs them."""
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.bgmm_native_api import (
    BGMM_NATIVE_MODE, BGMM_NATIVE_FIT, BGMM_NATIVE_SCORE, BGMM_FIT_INTS, BGMM_FIT_FLOATS, BGMM_SCORE_INTS,
)
from x_cluster.variants_dispatch import KMeansVariantsBackend
from x_cluster.entries import ENTRY_BGMM, ENTRY_BGMM_SCORE
from x_cluster.out import ClusterOut


def _word(address: Int, index: Int) -> Int:
    return Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=address)[index]


def bgmm_native_dispatch[K: KMeansVariantsBackend](operation: Int,
    aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) -> Int:
    try:
        if result == 0 or (na > 0 and aa == 0) or (np > 0 and pp == 0) or (nf > 0 and ff == 0):
            raise Error("BGMM native ABI: missing request or result span")
        var answer = 0
        if operation == BGMM_NATIVE_MODE:
            if na != 0 or np != 0 or nf != 0:
                raise Error("BGMM native ABI: mode takes no arguments")
            answer = K.mode()
        elif operation == BGMM_NATIVE_FIT or operation == BGMM_NATIVE_SCORE:
            var fit = operation == BGMM_NATIVE_FIT
            var integers = BGMM_FIT_INTS if fit else BGMM_SCORE_INTS
            var scalars = BGMM_FIT_FLOATS if fit else 0
            if na != 3 or np != integers + 2 or nf != scalars or _word(aa, 2) == 0:
                raise Error("BGMM native ABI: invalid request")
            var ip = List[Int]()
            for i in range(integers):  # driver-loop: fixed native entry option words
                ip.append(_word(pp, i + 2))
            var fp = List[Float64]()
            for i in range(scalars):  # driver-loop: the six fixed float options
                fp.append(Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)[i])
            var count = _word(pp, 0)
            var auxiliary_count = _word(pp, 1)
            # The `x_cluster_call` address rule: an empty span has no address.
            var source = _word(aa, 0) if count > 0 else 0
            var auxiliary = _word(aa, 1) if auxiliary_count > 0 else 0
            var value = K.execute(ENTRY_BGMM if fit else ENTRY_BGMM_SCORE, source, count,
                                  auxiliary, auxiliary_count, ip, fp)
            MutPointer[ClusterOut, MutUntrackedOrigin](unsafe_from_address=_word(aa, 2))[] = value^
            _ = ip^
            _ = fp^
        else:
            raise Error("BGMM native ABI: unknown operation")
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = answer
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1
