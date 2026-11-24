# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""THE RESIDENT ROUTE, round 5 (lane vdf-scope-neural5, 2026-11-24): the TruncatedSVD fold lease, inverse_transform
on a fold, the randomized TruncatedSVD fit over a fold taken on the device, and the IncrementalPCA resident bound.
Each device-resident form gives the same bytes as the same GPU operation over the fold's rows gathered on the host
and handed as a host X (the host-input route) and, for the randomized fit, as the host column's code (HostExec /
NoRDev / RES False: the CPU image's) over the same host rows.

    case 0  IncrementalPCA: transform reading more rows than the taken id holds is refused (`_resident_bound`,
            `DevRDev.held_words`); an id the member store did not make is refused; the exact row count passes
    case 1  TruncatedSVD.transform on a fold through a fold lease (core/tsvd_native_fold.mojo open / query with
            `resident` / close: the components' device copy) == without a lease == the device entry over host rows;
            a second lease query (the kept copy) gives the same bytes
    case 2  inverse_transform on a fold of scores: PCA unwhitened (PCAFoldKernels.query, `inverse`) == the device
            entry's host-input inverse; TruncatedSVD with and without a lease == the device entry's host-input
            inverse; an out-of-range index answers 2 (TruncatedSVD) / in_range 1 (PCA)
    case 3  TruncatedSVD randomized fit (x_decomp/tsvd_owner.mojo `tsvd_randomized_fit_run`, `x_dev`: the pool id of
            a fold taken on the device) == the host-input GPU fit == the host column
    case 4  refusals: a resident X on the host column (randomized TruncatedSVD); a resident source on the TSQR
            route of the decomposition provider

Fail-proof edits (each must fail its case): in `_resident_bound` compare `held < rows * d - d` (case 0: one row past
the id passes); in `tsvd_transform_fold_resident` pass `n_components - 1` to gemm_nt (case 1); in
`inverse_transform_fold` gather with `n_features` columns instead of `n_components` (case 2); in
`tsvd_randomized_fit_run` borrow `x_dev` with `rows - 1` rows (case 3).

Not covered here (OWED): the member and door wiring (core/decomposition_member.mojo `_fold_lease`, `_rows_open_xd`;
providers/python/member_decomposition_boundary_door.mojo `_xd_store_route`), which only the whole search RUN with
scope + pin exercises (checks/native_tests/msel_search_run.mojo pattern).
"""
from std.memory import bitcast
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases_serial, require
from core.native_descriptor_call import native_descriptor_call
from x_prep.transport_abi import XP_DEV_PUT, XP_DEV_FREE, XP_DEV_TAKE_ROWS
from x_decomp.cells import F32Ptr
from x_decomp.device import DevExec
from x_decomp.host import HostExec
from x_decomp.pca_randomized import NoRDev
from x_decomp.pca_randomized_dev import DevRDev
from x_decomp.member_store import x_decomp_store_v1
from x_decomp.ipca_owner import ipca_transform_run
from x_decomp.tsvd_owner import tsvd_randomized_fit_run
from core.msel_device_x import msel_fold_store, msel_device_x_context
from core.pca_native_api import PCAQueryRequest, PCAFoldStore
from core.pca_native_fold import PCAFoldKernels
from core.pca_native_device import PCADeviceKernels
from core.tsvd_native_api import TSVDQueryRequest, TSVDFitRequest, TSVD_TSQR
from core.tsvd_native_fold import TSVDFoldKernels
from core.tsvd_native_device import TSVDDeviceKernels
from core.tsvd_native_decomp import TSVDDecompKernels

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


def _p(values: List[Float32]) -> F32Ptr:
    return F32Ptr(unsafe_from_address=Int(values.unsafe_ptr()))


def _store(op: Int, a: List[Int], p: List[Int]) raises -> Int:
    return native_descriptor_call(x_decomp_store_v1, op, a, p, List[Float64]())


def _taken_fold(whole: List[Float32], idx: List[Int64], d: Int) raises -> Tuple[Int, Int]:
    """The whole X put in the x_decomp store and the fold taken on the device: (base id, taken id)."""
    var a: List[Int] = [Int(whole.unsafe_ptr())]
    var p: List[Int] = [len(whole)]
    var base = _store(XP_DEV_PUT, a, p)
    var at: List[Int] = [Int(idx.unsafe_ptr())]
    var pt: List[Int] = [base, d, len(idx)]
    var taken = _store(XP_DEV_TAKE_ROWS, at, pt)
    return (base, taken)


def _free(id: Int) raises:
    var p: List[Int] = [id]
    _ = _store(XP_DEV_FREE, List[Int](), p)


struct _MselFold(Movable):
    """The whole matrix and the fold's int64 index put in the model-selection fold store."""
    var base: Int
    var index: Int
    var rows: Int

    def __init__(out self, whole: List[Float32], rows: Int, d: Int, idx: List[Int64]) raises:
        var ctx = msel_device_x_context()
        var store = msel_fold_store()
        self.base = store[].put(ctx, Int(whole.unsafe_ptr()), rows * d * 4)
        self.index = store[].put(ctx, Int(idx.unsafe_ptr()), len(idx) * 8)
        self.rows = rows

    def fold(self) -> PCAFoldStore:
        return PCAFoldStore(self.base, self.rows, self.index)

    def close(mut self) raises:
        var ctx = msel_device_x_context()
        var store = msel_fold_store()
        store[].free(ctx, self.index)
        store[].free(ctx, self.base)


def _refused(message: String, needle: String) -> Bool:
    return message.find(needle) >= 0


def _ipca_bound() raises:
    var k = 2
    var whole = _values(_ROWS * _D, 3)
    var idx = _fold()
    var n = len(idx)
    var ids = _taken_fold(whole, idx, _D)
    var mean = _values(_D, 1)
    var comps = _values(k * _D, 2)
    var ev: List[Float32] = [2.0, 1.0]
    var dst = List[Float32](length=(n + 1) * k, fill=Float32(0))
    var over = False
    try:
        ipca_transform_run[DevExec, DevRDev, True](F32Ptr(unsafe_from_address=0), n + 1, _D, k, _p(mean), _p(comps),
            _p(ev), False, _p(dst), List[Int](), ids[1])
    except error:
        over = _refused(String(error), "the resident X holds")
    require(over, "IncrementalPCA transform: one row past the taken id is refused")
    ipca_transform_run[DevExec, DevRDev, True](F32Ptr(unsafe_from_address=0), n, _D, k, _p(mean), _p(comps),
        _p(ev), False, _p(dst), List[Int](), ids[1])
    var stray = DevRDev.alloc(n * _D)
    var unknown = False
    try:
        ipca_transform_run[DevExec, DevRDev, True](F32Ptr(unsafe_from_address=0), n, _D, k, _p(mean), _p(comps),
            _p(ev), False, _p(dst), List[Int](), stray)
    except error:
        unknown = _refused(String(error), "was not put through this store")
    DevRDev.free(stray)
    require(unknown, "IncrementalPCA transform: an id the store did not make is refused")
    _free(ids[1])
    _free(ids[0])
    _ = whole^
    _ = idx^


def _tsvd_lease() raises:
    var d = 4
    var k = 2
    var idx = _fold()
    var n = len(idx)
    var whole = _values(_ROWS * d, 29)
    var host_x = _gather(whole, d, idx)
    var comps = _values(k * d, 11)
    var store = _MselFold(whole, _ROWS, d, idx)
    var ta = List[Float32](length=n * k, fill=Float32(0))
    var tb = List[Float32](length=n * k, fill=Float32(0))
    var tc = List[Float32](length=n * k, fill=Float32(0))
    var td = List[Float32](length=n * k, fill=Float32(0))
    var finite = TSVDDeviceKernels.query(TSVDQueryRequest(Int(host_x.unsafe_ptr()), Int(comps.unsafe_ptr()),
        Int(ta.unsafe_ptr()), n, d, k, False, True, False, 0))
    require(finite == 0, "TruncatedSVD host-input transform finite word")
    var plain = TSVDQueryRequest(0, Int(comps.unsafe_ptr()), Int(tb.unsafe_ptr()), n, d, k, False, True, False, 0)
    plain.fold = store.fold()
    require(TSVDFoldKernels.query(plain) == 0, "TruncatedSVD fold transform (no lease) status")
    var lease = TSVDFoldKernels.open()
    require(lease > 0, "TruncatedSVD fold provider opens a lease")
    var first = TSVDQueryRequest(0, Int(comps.unsafe_ptr()), Int(tc.unsafe_ptr()), n, d, k, False, True, True, lease)
    first.fold = store.fold()
    require(TSVDFoldKernels.query(first) == 0, "TruncatedSVD fold transform (lease, first) status")
    var again = TSVDQueryRequest(0, Int(comps.unsafe_ptr()), Int(td.unsafe_ptr()), n, d, k, False, True, True, lease)
    again.fold = store.fold()
    require(TSVDFoldKernels.query(again) == 0, "TruncatedSVD fold transform (lease, kept copy) status")
    TSVDFoldKernels.close(lease)
    store.close()
    _same(ta, tb, "TruncatedSVD fold transform, no lease")
    _same(ta, tc, "TruncatedSVD fold transform, lease upload")
    _same(ta, td, "TruncatedSVD fold transform, lease kept copy")
    _ = whole^
    _ = host_x^
    _ = comps^


def _inverse() raises:
    var d = 4
    var k = 2
    var idx = _fold()
    var n = len(idx)
    var scores_whole = _values(_ROWS * k, 31)
    var host_scores = _gather(scores_whole, k, idx)
    var mean = _values(d, 4)
    var comps = _values(k * d, 13)
    var store = _MselFold(scores_whole, _ROWS, k, idx)
    # PCA unwhitened inverse
    var pa = List[Float32](length=n * d, fill=Float32(0))
    var pb = List[Float32](length=n * d, fill=Float32(0))
    _ = PCADeviceKernels.query(PCAQueryRequest(Int(host_scores.unsafe_ptr()), Int(mean.unsafe_ptr()),
        Int(comps.unsafe_ptr()), 0, Int(pa.unsafe_ptr()), n, d, k, _ROWS, True, False, PCAFoldStore()))
    var pw = PCAFoldKernels.query(PCAQueryRequest(0, Int(mean.unsafe_ptr()), Int(comps.unsafe_ptr()), 0,
        Int(pb.unsafe_ptr()), n, d, k, _ROWS, True, False, store.fold()))
    require(pw.in_range == 0, "PCA fold inverse_transform range word")
    _same(pa, pb, "PCA inverse_transform over the fold")
    # TruncatedSVD inverse, with and without a lease
    var ta = List[Float32](length=n * d, fill=Float32(0))
    var tb = List[Float32](length=n * d, fill=Float32(0))
    var tc = List[Float32](length=n * d, fill=Float32(0))
    _ = TSVDDeviceKernels.query(TSVDQueryRequest(Int(host_scores.unsafe_ptr()), Int(comps.unsafe_ptr()),
        Int(ta.unsafe_ptr()), n, d, k, True, False, False, 0))
    var plain = TSVDQueryRequest(0, Int(comps.unsafe_ptr()), Int(tb.unsafe_ptr()), n, d, k, True, False, False, 0)
    plain.fold = store.fold()
    require(TSVDFoldKernels.query(plain) == 0, "TruncatedSVD fold inverse_transform (no lease) status")
    var lease = TSVDFoldKernels.open()
    var leased = TSVDQueryRequest(0, Int(comps.unsafe_ptr()), Int(tc.unsafe_ptr()), n, d, k, True, False, True, lease)
    leased.fold = store.fold()
    require(TSVDFoldKernels.query(leased) == 0, "TruncatedSVD fold inverse_transform (lease) status")
    TSVDFoldKernels.close(lease)
    store.close()
    _same(ta, tb, "TruncatedSVD inverse_transform over the fold, no lease")
    _same(ta, tc, "TruncatedSVD inverse_transform over the fold, lease")
    # an index past the base's rows
    var bad: List[Int64] = [0, Int64(_ROWS)]
    var bad_store = _MselFold(scores_whole, _ROWS, k, bad)
    var dst = List[Float32](length=2 * d, fill=Float32(0))
    var off = TSVDQueryRequest(0, Int(comps.unsafe_ptr()), Int(dst.unsafe_ptr()), 2, d, k, True, False, False, 0)
    off.fold = bad_store.fold()
    var word = TSVDFoldKernels.query(off)
    var pca_off = PCAFoldKernels.query(PCAQueryRequest(0, Int(mean.unsafe_ptr()), Int(comps.unsafe_ptr()), 0,
        Int(dst.unsafe_ptr()), 2, d, k, _ROWS, True, False, bad_store.fold()))
    bad_store.close()
    require(word == 2, "TruncatedSVD fold inverse_transform: an out-of-range index answers 2")
    require(pca_off.in_range != 0, "PCA fold inverse_transform: an out-of-range index sets the range word")
    _ = scores_whole^
    _ = host_scores^
    _ = mean^
    _ = comps^
    _ = bad^


struct _TsvdRoom(Movable):
    var comps: List[Float32]
    var sing: List[Float32]
    var var_: List[Float32]
    var ratio: List[Float32]

    def __init__(out self, k: Int, d: Int):
        self.comps = List[Float32](length=k * d, fill=Float32(0))
        self.sing = List[Float32](length=k, fill=Float32(0))
        self.var_ = List[Float32](length=k, fill=Float32(0))
        self.ratio = List[Float32](length=k, fill=Float32(0))

    def same(self, other: _TsvdRoom, what: String) raises:
        _same(self.comps, other.comps, what + " components")
        _same(self.sing, other.sing, what + " singular values")
        _same(self.var_, other.var_, what + " explained variance")
        _same(self.ratio, other.ratio, what + " explained variance ratio")


def _tsvd_randomized() raises:
    var k = 2
    var idx = _fold()
    var n = len(idx)
    var whole = _values(_ROWS * _D, 37)
    var host_x = _gather(whole, _D, idx)
    var ids = _taken_fold(whole, idx, _D)
    var resident = _TsvdRoom(k, _D)
    var device = _TsvdRoom(k, _D)
    var cpu = _TsvdRoom(k, _D)
    tsvd_randomized_fit_run[DevExec, DevRDev, True](F32Ptr(unsafe_from_address=0), _p(resident.comps),
        _p(resident.sing), _p(resident.var_), _p(resident.ratio), n, _D, k, 3, 2, False, 7, 0, ids[1])
    tsvd_randomized_fit_run[DevExec, DevRDev, True](_p(host_x), _p(device.comps), _p(device.sing),
        _p(device.var_), _p(device.ratio), n, _D, k, 3, 2, False, 7)
    tsvd_randomized_fit_run[HostExec, NoRDev, False](_p(host_x), _p(cpu.comps), _p(cpu.sing),
        _p(cpu.var_), _p(cpu.ratio), n, _D, k, 3, 2, False, 7)
    _free(ids[1])
    _free(ids[0])
    resident.same(device, "randomized TruncatedSVD, resident fold vs host-input GPU:")
    device.same(cpu, "randomized TruncatedSVD, GPU vs the host column:")
    _ = whole^
    _ = host_x^


def _refusals() raises:
    var k = 2
    var whole = _values(_ROWS * _D, 41)
    var idx = _fold()
    var n = len(idx)
    var ids = _taken_fold(whole, idx, _D)
    var cpu = _TsvdRoom(k, _D)
    var host_refused = False
    try:
        tsvd_randomized_fit_run[HostExec, NoRDev, False](F32Ptr(unsafe_from_address=0), _p(cpu.comps),
            _p(cpu.sing), _p(cpu.var_), _p(cpu.ratio), n, _D, k, 3, 2, False, 7, 0, ids[1])
    except error:
        host_refused = _refused(String(error), "needs a GPU binding")
    require(host_refused, "randomized TruncatedSVD: the host column refuses a resident X")
    var tsqr_refused = False
    try:
        var request = TSVDFitRequest(-(ids[1] + 1), Int(cpu.comps.unsafe_ptr()), Int(cpu.sing.unsafe_ptr()),
            Int(cpu.var_.unsafe_ptr()), Int(cpu.ratio.unsafe_ptr()), n, _D, k, 0, 0, False, 0, True, TSVD_TSQR,
            PCAFoldStore())
        _ = TSVDDecompKernels[DevExec, DevRDev, True].fit(request)
    except error:
        tsqr_refused = _refused(String(error), "only in a GPU randomized fit")
    require(tsqr_refused, "TruncatedSVD decomposition provider: a resident source on the TSQR route is refused")
    _free(ids[1])
    _free(ids[0])
    _ = whole^
    _ = idx^


def _case(unit: Int) raises:
    if unit == 0:
        _ipca_bound()
    elif unit == 1:
        _tsvd_lease()
    elif unit == 2:
        _inverse()
    elif unit == 3:
        _tsvd_randomized()
    else:
        _refusals()


def test_vdf_scope_neural5_rows() raises:
    check_cases_serial("vdf_scope_neural5_rows", CASES, _case)


def main() raises:
    var suite = TestSuite()
    suite.test[test_vdf_scope_neural5_rows]()
    suite^.run()
