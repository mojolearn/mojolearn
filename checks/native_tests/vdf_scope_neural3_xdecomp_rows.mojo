# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""THE RESIDENT ROUTE of the XDecomp members (lane vdf-scope-neural3, 2026-11-24): a fold taken ON THE DEVICE out
of the whole X held in the x_decomp device pool (x_decomp/member_store.mojo `x_decomp_store_v1`, the transport the
member door puts in `route.fifth`), read in place by the family's GPU operations through their X pool word, gives
the same bytes as the same GPU operation over the fold's rows gathered on the host and handed as a host X (the
host-input route, whose bytes the CPU column and the cross-vendor verification already pin). Device column only:
the pool lives on the device.

    case 0  the store transport: PUT of the whole X, TAKE_ROWS of a fold with a repeat and an out-of-order row,
            the taken id's words downloaded == the host gather, word for word; LIVE counts the two ids; FREE
    case 1  random projection transform (x_decomp/rp_owner.mojo `rp_project_run`, tiled; XD_RP_PROJECT word 5)
            over the fold's pool id == over the host-gathered fold
    case 2  FactorAnalysis fit (x_decomp/fa_owner.mojo `fa_fit_run`, `x_dev`): components, noise variances,
            mean and the log-likelihood curve, bit for bit
    case 3  FastICA transform (x_decomp/ica_owner.mojo `ica_project_run`, `x_dev`, whitened)
    case 4  refusals: TAKE_ROWS past X's rows (the exact put size, not the pool class's padding); a pool id the
            store did not put

Fail-proof edits (each must fail its case): in `xd_store_take_rows` hand the kernel `row_words` + 1 (case 0); in
`rp_project_run` add the X matrix as `kit.add(Int(x), -1, ...)` regardless of `x_dev` (case 1 faults: null X);
in `xd_store_take_rows` bound by `len(src) // row_words` (case 4: the padding row is read).

Not covered here (OWED): the decomposition kit (DKit.borrow_device: DictionaryLearning, SparsePCA, SparseCoder
transform, MinCovDet) and the manifold kit (MRun.adopt: Isomap, LLE, MDS) need an x_decomp entry reference; their
end-to-end check is the msel search RUN with `plan.scope` / `plan.pin` over those members.
"""
from std.memory import bitcast
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases_serial, require
from core.native_descriptor_call import native_descriptor_call
from x_prep.transport_abi import XP_DEV_PUT, XP_DEV_FREE, XP_DEV_TAKE_ROWS, XP_DEV_LIVE
from x_decomp.cells import F32Ptr
from x_decomp.device import DevExec
from x_decomp.pca_randomized_dev import DevRDev
from x_decomp.decomp_owner_dev import DevDecomp
from x_decomp.resident import dev_download_run, pool_alloc, pool_free
from x_decomp.member_store import x_decomp_store_v1
from x_decomp.rp_owner import rp_project_run
from x_decomp.fa_owner import fa_fit_run
from x_decomp.ica_owner import ica_project_run

comptime CASES = 5
comptime _ROWS = 24
comptime _D = 5


def _values(n: Int, salt: Int) -> List[Float32]:
    var out = List[Float32](length=n, fill=Float32(0))
    for i in range(n):  # small-loop(fixture values)
        out[i] = Float32((i * 7 + salt * 5) % 13 - 6) / Float32(8)
    return out^


def _fold() -> List[Int64]:
    """Twelve rows of a 24-row X: a repeat and an out-of-order run."""
    var idx: List[Int64] = [5, 0, 17, 3, 3, 22, 9, 11, 1, 14, 20, 7]
    return idx^


def _gather(whole: List[Float32], row_words: Int, idx: List[Int64]) -> List[Float32]:
    var out = List[Float32](length=len(idx) * row_words, fill=Float32(0))
    for r in range(len(idx)):  # small-loop(oracle: the fold's rows)
        for c in range(row_words):  # small-loop(oracle: one row's words)
            out[r * row_words + c] = whole[Int(idx[r]) * row_words + c]
    return out^


def _same(a: List[Float32], b: List[Float32], what: String) raises:
    require(len(a) == len(b), what + ": lengths differ")
    for i in range(len(a)):  # small-loop(oracle: word compare)
        require(bitcast[DType.uint32](a[i]) == bitcast[DType.uint32](b[i]),
                what + ": word " + String(i) + " differs (" + String(a[i]) + " vs " + String(b[i]) + ")")


def _store(op: Int, a: List[Int], p: List[Int]) raises -> Int:
    return native_descriptor_call(x_decomp_store_v1, op, a, p, List[Float64]())


def _put(whole: List[Float32]) raises -> Int:
    var a: List[Int] = [Int(whole.unsafe_ptr())]
    var p: List[Int] = [len(whole)]
    return _store(XP_DEV_PUT, a, p)


def _take(base: Int, row_words: Int, idx: List[Int64]) raises -> Int:
    var a: List[Int] = [Int(idx.unsafe_ptr())]
    var p: List[Int] = [base, row_words, len(idx)]
    return _store(XP_DEV_TAKE_ROWS, a, p)


def _free(id: Int) raises:
    var p: List[Int] = [id]
    _ = _store(XP_DEV_FREE, List[Int](), p)


def _taken_fold(whole: List[Float32], idx: List[Int64]) raises -> Tuple[Int, Int]:
    var base = _put(whole)
    var taken = _take(base, _D, idx)
    return (base, taken)


def _transport() raises:
    var whole = _values(_ROWS * _D, 1)
    var idx = _fold()
    var before = _store(XP_DEV_LIVE, List[Int](), List[Int]())
    var ids = _taken_fold(whole, idx)
    require(_store(XP_DEV_LIVE, List[Int](), List[Int]()) == before + 2, "x_decomp store: LIVE counts put + take")
    var got = List[Float32](length=len(idx) * _D, fill=Float32(0))
    _ = dev_download_run(ids[1], F32Ptr(unsafe_from_address=Int(got.unsafe_ptr())), len(got))
    _same(got, _gather(whole, _D, idx), "x_decomp store take_rows")
    _free(ids[1])
    _free(ids[0])
    require(_store(XP_DEV_LIVE, List[Int](), List[Int]()) == before, "x_decomp store: FREE returns both ids")
    _ = whole^
    _ = idx^


def _rp() raises:
    var whole = _values(_ROWS * _D, 2)
    var idx = _fold()
    var host = _gather(whole, _D, idx)
    var kc = 3
    var matrix = _values(kc * _D, 5)
    var n = len(idx)
    var oracle = List[Float32](length=n * kc, fill=Float32(0))
    var resident = List[Float32](length=n * kc, fill=Float32(0))
    _ = rp_project_run[DevExec, DevRDev, True, DevDecomp](
        F32Ptr(unsafe_from_address=Int(host.unsafe_ptr())), n, _D, F32Ptr(unsafe_from_address=Int(matrix.unsafe_ptr())),
        kc, True, F32Ptr(unsafe_from_address=Int(oracle.unsafe_ptr())))
    var ids = _taken_fold(whole, idx)
    _ = rp_project_run[DevExec, DevRDev, True, DevDecomp](
        F32Ptr(unsafe_from_address=0), n, _D, F32Ptr(unsafe_from_address=Int(matrix.unsafe_ptr())), kc, True,
        F32Ptr(unsafe_from_address=Int(resident.unsafe_ptr())), -1, False, ids[1])
    _free(ids[1])
    _free(ids[0])
    _same(resident, oracle, "random projection transform over the device fold")
    _ = whole^
    _ = host^
    _ = matrix^


def _fa() raises:
    var whole = _values(_ROWS * _D, 3)
    var idx = _fold()
    var host = _gather(whole, _D, idx)
    var n = len(idx)
    var nc = 2
    var iters = 6
    var psi0 = List[Float32](length=_D, fill=Float32(1.0))
    var comps = List[List[Float32]]()
    var psis = List[List[Float32]]()
    var means = List[List[Float32]]()
    var curves = List[List[Float64]]()
    var ids = _taken_fold(whole, idx)
    for arm in range(2):  # small-loop(the two arms: host fold, device fold)
        comps.append(List[Float32](length=nc * _D, fill=Float32(0)))
        psis.append(List[Float32](length=_D, fill=Float32(0)))
        means.append(List[Float32](length=_D, fill=Float32(0)))
        curves.append(List[Float64](length=iters, fill=Float64(0)))
        var x = Int(host.unsafe_ptr()) if arm == 0 else 0
        _ = fa_fit_run[DevExec, DevRDev, True, DevDecomp](
            F32Ptr(unsafe_from_address=x), n, _D, nc, iters, 1.0e-2, F32Ptr(unsafe_from_address=Int(psi0.unsafe_ptr())),
            0, F32Ptr(unsafe_from_address=Int(comps[arm].unsafe_ptr())),
            F32Ptr(unsafe_from_address=Int(psis[arm].unsafe_ptr())),
            F32Ptr(unsafe_from_address=Int(means[arm].unsafe_ptr())),
            MutPointer[Float64, MutAnyOrigin](unsafe_from_address=Int(curves[arm].unsafe_ptr())), 0,
            -1 if arm == 0 else ids[1])
    _free(ids[1])
    _free(ids[0])
    _same(comps[1], comps[0], "FactorAnalysis components_ over the device fold")
    _same(psis[1], psis[0], "FactorAnalysis noise_variance_ over the device fold")
    _same(means[1], means[0], "FactorAnalysis mean_ over the device fold")
    for k in range(iters):  # small-loop(oracle: the EM curve)
        require(bitcast[DType.uint64](curves[1][k]) == bitcast[DType.uint64](curves[0][k]),
                "FactorAnalysis loglike_ word " + String(k) + " differs over the device fold")
    _ = whole^
    _ = host^
    _ = psi0^


def _ica() raises:
    var whole = _values(_ROWS * _D, 4)
    var idx = _fold()
    var host = _gather(whole, _D, idx)
    var n = len(idx)
    var nc = 3
    var matrix = _values(nc * _D, 6)
    var mean = _values(_D, 7)
    var oracle = List[Float32](length=n * nc, fill=Float32(0))
    var resident = List[Float32](length=n * nc, fill=Float32(0))
    ica_project_run[DevExec, DevRDev, True](
        F32Ptr(unsafe_from_address=Int(host.unsafe_ptr())), n, _D, False,
        F32Ptr(unsafe_from_address=Int(matrix.unsafe_ptr())), nc, F32Ptr(unsafe_from_address=Int(mean.unsafe_ptr())),
        True, F32Ptr(unsafe_from_address=Int(oracle.unsafe_ptr())))
    var ids = _taken_fold(whole, idx)
    ica_project_run[DevExec, DevRDev, True](
        F32Ptr(unsafe_from_address=0), n, _D, False, F32Ptr(unsafe_from_address=Int(matrix.unsafe_ptr())), nc,
        F32Ptr(unsafe_from_address=Int(mean.unsafe_ptr())), True,
        F32Ptr(unsafe_from_address=Int(resident.unsafe_ptr())), ids[1])
    _free(ids[1])
    _free(ids[0])
    _same(resident, oracle, "FastICA transform over the device fold")
    _ = whole^
    _ = host^
    _ = matrix^
    _ = mean^


def _refusals() raises:
    # 23 rows of 5 words: the pool class holds 128 words, so row 23 is padding the take must refuse.
    var whole = _values(23 * _D, 1)
    var bad: List[Int64] = [0, 23]
    var base = _put(whole)
    var refused = False
    try:
        _ = _take(base, _D, bad)
    except error:
        refused = String(error).find("out of range") >= 0
    _free(base)
    require(refused, "x_decomp store: take_rows refuses a row past X's rows")
    var stranger = pool_alloc(8)
    var foreign = False
    try:
        _ = _take(stranger, 1, bad)
    except error:
        foreign = String(error).find("was not put through this store") >= 0
    pool_free(stranger)
    require(foreign, "x_decomp store: take_rows refuses a pool id it did not put")
    _ = whole^
    _ = bad^


def _case(unit: Int) raises:
    if unit == 0:
        _transport()
    elif unit == 1:
        _rp()
    elif unit == 2:
        _fa()
    elif unit == 3:
        _ica()
    else:
        _refusals()


def test_xdecomp_resident_rows() raises:
    check_cases_serial("vdf_scope_neural3_xdecomp_rows", CASES, _case)


def main() raises:
    var suite = TestSuite()
    suite.test[test_xdecomp_resident_rows]()
    suite^.run()
