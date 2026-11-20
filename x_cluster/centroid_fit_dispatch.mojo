# SPDX-License-Identifier: Apache-2.0
"""Typed transport into the existing MeanShift and AP execution entries."""
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.cluster_centroid_fit_api import (
    CentroidFitResult, CENTROID_FIT_MODE, CENTROID_FIT_MEANSHIFT, CENTROID_FIT_AFFINITY,
)
from x_cluster.variants_dispatch import KMeansVariantsBackend
from x_cluster.entries import ENTRY_MEANSHIFT, ENTRY_AFFINITY


def _word(address: Int, index: Int) -> Int:
    return Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=address)[index]


def centroid_fit_dispatch[K: KMeansVariantsBackend](operation: Int,
    aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) -> Int:
    try:
        if result == 0 or (na > 0 and aa == 0) or (np > 0 and pp == 0) or (nf > 0 and ff == 0):
            raise Error("cluster centroid fit ABI: missing request or result span")
        var answer = 0
        if operation == CENTROID_FIT_MODE:
            if na != 0 or np != 0 or nf != 0:
                raise Error("cluster centroid fit ABI: mode takes no arguments")
            answer = K.mode()
        else:
            if operation != CENTROID_FIT_MEANSHIFT and operation != CENTROID_FIT_AFFINITY:
                raise Error("cluster centroid fit ABI: unknown operation")
            var scalars = 1 if operation == CENTROID_FIT_MEANSHIFT else 2
            if na != 3 or np != 9 or nf != scalars or _word(aa, 2) == 0:
                raise Error("cluster centroid fit ABI: invalid fit request")
            var ip = List[Int]()
            for i in range(2, 9):  # metadata: the original seven entry words
                ip.append(_word(pp, i))
            var fp = List[Float64]()
            for i in range(scalars):  # metadata: bandwidth or damping/preference
                fp.append(Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=ff)[i])
            var which = ENTRY_MEANSHIFT if operation == CENTROID_FIT_MEANSHIFT else ENTRY_AFFINITY
            # These are the original run_entry routes and tuned kernels, not
            # any KMeans fit. The backend supplies their shared transport only.
            var old = K.execute(which, _word(aa, 0), _word(pp, 0), _word(aa, 1), _word(pp, 1), ip, fp)
            var value = CentroidFitResult()
            if operation == CENTROID_FIT_MEANSHIFT:
                if len(old.f) != 1 or len(old.i) != 1 or len(old.s) != 3:
                    raise Error("cluster centroid fit ABI: invalid MeanShift result slots")
                value.centers = old.f.pop(0)
                value.labels = old.i.pop(0)
                value.bandwidth = old.s[0]
                value.iterations = Int(old.s[1])
                value.clusters = Int(old.s[2])
            else:
                if len(old.f) != 3 or len(old.i) != 2 or len(old.s) != 1:
                    raise Error("cluster centroid fit ABI: invalid affinity result slots")
                value.affinity = old.f.pop(0)
                value.message_diagonal = old.f.pop(0)
                value.centers = old.f.pop(0)
                value.center_indices = old.i.pop(0)
                value.labels = old.i.pop(0)
                value.iterations = Int(old.s[0])
                value.clusters = len(value.center_indices)
            var dst = MutPointer[CentroidFitResult, MutUntrackedOrigin](unsafe_from_address=_word(aa, 2))
            dst[] = value^
            _ = ip^
            _ = fp^
        MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = answer
        return 0
    except problem:
        native_descriptor_report(problem, context, sink)
        return 1
