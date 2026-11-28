# SPDX-License-Identifier: Apache-2.0
"""Typed transport of the Harmony fit (core/harmony_api.mojo) into x_cluster/harmony.mojo `harmony_fit` on one
column's ops (lane sc-stats-1). The device and host tables (harmony_device.mojo, harmony_host.mojo) call these."""
from std.memory import unsafe_memcpy
from core.hostptr import read_f32
from core.harmony_api import HARMONY_MODE, HARMONY_FIT, HarmonyFitResult
from x_cluster.ops import ClusterOps
from x_cluster.harmony import HarmonyParams, harmony_fit


def _word(address: Int, index: Int) -> Int:
    return Pointer[Int, ImmUntrackedOrigin](unsafe_from_address=address)[index]


def _real(address: Int, index: Int) -> Float64:
    return Pointer[Float64, ImmUntrackedOrigin](unsafe_from_address=address)[index]


def harmony_check(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int, result: Int) raises:
    if result == 0 or (na > 0 and aa == 0) or (np > 0 and pp == 0) or (nf > 0 and ff == 0):
        raise Error("harmony ABI: missing request or result span")
    if operation == HARMONY_MODE:
        if na != 0 or np != 0 or nf != 0:
            raise Error("harmony ABI: mode takes no arguments")
        return
    if operation != HARMONY_FIT:
        raise Error("harmony ABI: unknown operation")
    if na != 6 or np != 8 or nf != 5 or _word(aa, 5) == 0:
        raise Error("harmony ABI: invalid fit request")


def harmony_answer(result: Int, value: Int):
    MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[0] = value


def harmony_run[O: ClusterOps](mut ops: O, aa: Int, pp: Int, ff: Int) raises:
    """One fit: the request words read into the driver's typed parameters, the result moved into the caller's
    slot (`HarmonyResultSlot`)."""
    var n = _word(pp, 0)
    var d = _word(pp, 1)
    var k = _word(pp, 2)
    var c = _word(pp, 3)
    var nb = _word(pp, 4)
    if n < 1 or d < 1 or k < 1 or c < 1 or nb < 1:
        raise Error("harmony ABI: invalid sizes")
    var z = read_f32(_word(aa, 0), n * d)
    var codes = List[Int32](length=c * n, fill=Int32(0))
    unsafe_memcpy(dest=codes.unsafe_ptr(), src=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=_word(aa, 1)),
                  count=c * n)
    var offsets = List[Int32](length=c, fill=Int32(0))
    unsafe_memcpy(dest=offsets.unsafe_ptr(),
                  src=MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=_word(aa, 2)), count=c)
    var theta = List[Float64](length=nb, fill=0.0)
    var lamb = List[Float64](length=nb, fill=0.0)
    for b in range(nb):  # metadata: one theta and lamb per batch level
        theta[b] = _real(_word(aa, 3), b)
        lamb[b] = _real(_word(aa, 4), b)
    var params = HarmonyParams(n, d, k, c, nb, _word(pp, 5), _word(pp, 6), UInt64(_word(pp, 7)), _real(ff, 0),
                               _real(ff, 1), _real(ff, 2), _real(ff, 3), _real(ff, 4), theta^, lamb^)
    var fitted = harmony_fit(ops, z, codes, offsets, params)
    var dst = MutPointer[HarmonyFitResult, MutUntrackedOrigin](unsafe_from_address=_word(aa, 5))
    dst[] = fitted^
    _ = z^
    _ = codes^
    _ = offsets^
