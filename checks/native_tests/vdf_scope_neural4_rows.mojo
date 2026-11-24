# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""THE RESIDENT ROUTE, round 4 (lane vdf-scope-neural4, 2026-11-24): IncrementalPCA, NMF and the PCA / TruncatedSVD
fold forms read a fold taken ON THE DEVICE and give the same bytes as the same GPU operation over the fold's rows
gathered on the host and handed as a host X (the host-input route), and, for IncrementalPCA and NMF, as the host
column's kernels (HostExec / NoRDev / RES False: the CPU image's code) over the same host rows.

    case 0  IncrementalPCA fit (x_decomp/ipca_owner.mojo `ipca_fit_run`, `x_dev`: batches of 5 rows of a 12-row
            fold sliced ON THE DEVICE by `_resident_rows`, the remainder batch at a row offset) == the host-input
            GPU fit == the host column; then transform (`ipca_transform_run`, `x_dev`) over the same fold
    case 1  NMF fit_transform, MU and CD solvers (x_decomp/nmf_owner.mojo `nmf_run`, `x_dev`; the solve reads the
            pool id in place, `DecompDev.nmf_solve_pool`) == the host-input GPU fit (whose solve now also reads the
            kit's device X) == the host column
    case 2  PCA fit_transform on a model-selection fold (core/pca_native_fold.mojo `PCAFoldKernels.fit` with a
            projection: decomposition/pca_fold.mojo `_project_x`), unwhitened and whitened, == the device entry's
            host-input fit_transform (PCADeviceKernels.fit, `pca_fit_owned` / `pca_project_owned`)
    case 3  PCA whitened transform (`pca_whiten_transform_fold`) and TruncatedSVD.transform
            (`tsvd_transform_fold`, TSVDFoldKernels.query) on the fold == the device entries over the host rows
    case 4  refusals: a resident X on the host column (IncrementalPCA); the TruncatedSVD fold query's
            out-of-range status (2)

Fail-proof edits (each must fail its case): in `_resident_rows` pass `0` for the move's a2 word (case 0: every
batch reads the first rows); in `nmf_run` call `nmf_solve_pool` with `n * d - 1` columns' worth by passing `d - 1`
(case 1); in `_project_x` use `n_components` = 1 (case 2: the projection's second column stays 0).

Not covered here (OWED): the manifold kinds' `MRun.adopt_copy` (needs an x_decomp entry reference) and the whole
search RUN with scope + pin over the DKit and manifold members (checks/native_tests/msel_search_run.mojo pattern).
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
from x_decomp.decomp_owner_dev import DevDecomp
from x_decomp.decomp_owner_host import HostDecomp
from x_decomp.member_store import x_decomp_store_v1
from x_decomp.ipca_owner import NoIpcaDev, ipca_fit_run, ipca_transform_run
from x_decomp.ipca_owner_dev import DevIpcaDev
from x_decomp.nmf import NmfArgs
from x_decomp.nmf_owner import NMF_INIT_NNDSVDA, nmf_run
from core.msel_device_x import msel_fold_store, msel_device_x_context
from core.pca_native_api import PCAFitRequest, PCAQueryRequest, PCAFoldStore
from core.pca_native_fold import PCAFoldKernels
from core.pca_native_device import PCADeviceKernels
from core.tsvd_native_api import TSVDQueryRequest
from core.tsvd_native_fold import TSVDFoldKernels
from core.tsvd_native_device import TSVDDeviceKernels

comptime CASES = 5
comptime _ROWS = 24
comptime _D = 5


def _values(n: Int, salt: Int) -> List[Float32]:
    var out = List[Float32](length=n, fill=Float32(0))
    for i in range(n):  # small-loop(fixture values)
        out[i] = Float32((i * 7 + salt * 5) % 13 - 6) / Float32(8)
    return out^


def _positive(n: Int, salt: Int) -> List[Float32]:
    """NMF's nonnegative fixture."""
    var out = List[Float32](length=n, fill=Float32(0))
    for i in range(n):  # small-loop(fixture values)
        out[i] = Float32((i * 5 + salt * 3) % 11 + 1) / Float32(8)
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


def _taken_fold(whole: List[Float32], idx: List[Int64]) raises -> Tuple[Int, Int]:
    """The whole X put in the x_decomp store and the fold taken on the device: (base id, taken id)."""
    var a: List[Int] = [Int(whole.unsafe_ptr())]
    var p: List[Int] = [len(whole)]
    var base = _store(XP_DEV_PUT, a, p)
    var at: List[Int] = [Int(idx.unsafe_ptr())]
    var pt: List[Int] = [base, _D, len(idx)]
    var taken = _store(XP_DEV_TAKE_ROWS, at, pt)
    return (base, taken)


def _free(id: Int) raises:
    var p: List[Int] = [id]
    _ = _store(XP_DEV_FREE, List[Int](), p)


struct _IpcaRoom(Movable):
    var comps: List[Float32]
    var sing: List[Float32]
    var mean: List[Float32]
    var var_: List[Float32]
    var ev: List[Float32]
    var evr: List[Float32]

    def __init__(out self):
        self.comps = List[Float32](length=_D * _D, fill=Float32(0))
        self.sing = List[Float32](length=_D, fill=Float32(0))
        self.mean = List[Float32](length=_D, fill=Float32(0))
        self.var_ = List[Float32](length=_D, fill=Float32(0))
        self.ev = List[Float32](length=_D, fill=Float32(0))
        self.evr = List[Float32](length=_D, fill=Float32(0))

    def same(self, other: _IpcaRoom, what: String) raises:
        _same(self.comps, other.comps, what + " components")
        _same(self.sing, other.sing, what + " singular values")
        _same(self.mean, other.mean, what + " mean")
        _same(self.var_, other.var_, what + " var")
        _same(self.ev, other.ev, what + " explained variance")
        _same(self.evr, other.evr, what + " ratio")


def _ipca() raises:
    var whole = _values(_ROWS * _D, 3)
    var idx = _fold()
    var host = _gather(whole, _D, idx)
    var n = len(idx)
    var nc = 3
    var batch = 5
    var cpu = _IpcaRoom()
    var gpu = _IpcaRoom()
    var res = _IpcaRoom()
    var kept = List[Int]()
    var got_cpu = ipca_fit_run[HostExec, NoRDev, False, NoIpcaDev](_p(host), n, _D, nc, batch,
        _p(cpu.comps), _p(cpu.sing), _p(cpu.mean), _p(cpu.var_), _p(cpu.ev), _p(cpu.evr), False, kept)
    var got_gpu = ipca_fit_run[DevExec, DevRDev, True, DevIpcaDev](_p(host), n, _D, nc, batch,
        _p(gpu.comps), _p(gpu.sing), _p(gpu.mean), _p(gpu.var_), _p(gpu.ev), _p(gpu.evr), False, kept)
    var ids = _taken_fold(whole, idx)
    var got_res = ipca_fit_run[DevExec, DevRDev, True, DevIpcaDev](F32Ptr(unsafe_from_address=0), n, _D, nc, batch,
        _p(res.comps), _p(res.sing), _p(res.mean), _p(res.var_), _p(res.ev), _p(res.evr), False, kept, ids[1])
    require(got_res[0] == got_gpu[0] and got_gpu[0] == got_cpu[0], "IncrementalPCA kept count")
    require(bitcast[DType.uint64](got_res[1]) == bitcast[DType.uint64](got_gpu[1]), "IncrementalPCA noise (device)")
    require(bitcast[DType.uint64](got_gpu[1]) == bitcast[DType.uint64](got_cpu[1]), "IncrementalPCA noise (column)")
    res.same(gpu, "IncrementalPCA device fold vs host rows:")
    gpu.same(cpu, "IncrementalPCA GPU vs host column:")
    var count = got_res[0]
    var ya = List[Float32](length=n * count, fill=Float32(0))
    var yb = List[Float32](length=n * count, fill=Float32(0))
    var yc = List[Float32](length=n * count, fill=Float32(0))
    var none = List[Int]()
    ipca_transform_run[HostExec, NoRDev, False](_p(host), n, _D, count, _p(cpu.mean), _p(cpu.comps), _p(cpu.ev),
        True, _p(yc), none)
    ipca_transform_run[DevExec, DevRDev, True](_p(host), n, _D, count, _p(gpu.mean), _p(gpu.comps), _p(gpu.ev),
        True, _p(ya), none)
    ipca_transform_run[DevExec, DevRDev, True](F32Ptr(unsafe_from_address=0), n, _D, count, _p(gpu.mean),
        _p(gpu.comps), _p(gpu.ev), True, _p(yb), none, ids[1])
    _free(ids[1])
    _free(ids[0])
    _same(yb, ya, "IncrementalPCA whitened transform over the device fold")
    _same(ya, yc, "IncrementalPCA whitened transform GPU vs host column")
    _ = whole^
    _ = host^


def _nmf_one(solver: Int) raises:
    var whole = _positive(_ROWS * _D, 4 + solver)
    var idx = _fold()
    var host = _gather(whole, _D, idx)
    var n = len(idx)
    var nc = 2
    var a = NmfArgs(2.0, True, 40, 1e-4, 0.0, 0.0, 0.0, 0.0, False, 0)
    var unused = List[Float32](length=1, fill=Float32(0))
    var wc = List[Float32](length=n * nc, fill=Float32(0))
    var hc = List[Float32](length=nc * _D, fill=Float32(0))
    var wg = List[Float32](length=n * nc, fill=Float32(0))
    var hg = List[Float32](length=nc * _D, fill=Float32(0))
    var wr = List[Float32](length=n * nc, fill=Float32(0))
    var hr = List[Float32](length=nc * _D, fill=Float32(0))
    var rc = nmf_run[HostExec, NoRDev, False, HostDecomp](_p(host), n, _D, nc, NMF_INIT_NNDSVDA, _p(unused),
        _p(unused), 0, solver, a, _p(wc), _p(hc))
    var rg = nmf_run[DevExec, DevRDev, True, DevDecomp](_p(host), n, _D, nc, NMF_INIT_NNDSVDA, _p(unused),
        _p(unused), 0, solver, a, _p(wg), _p(hg))
    var ids = _taken_fold(whole, idx)
    var rr = nmf_run[DevExec, DevRDev, True, DevDecomp](F32Ptr(unsafe_from_address=0), n, _D, nc, NMF_INIT_NNDSVDA,
        _p(unused), _p(unused), 0, solver, a, _p(wr), _p(hr), 0, 0, ids[1])
    _free(ids[1])
    _free(ids[0])
    var what = String("NMF solver ") + String(solver)
    require(rr[0] == rg[0] and rg[0] == rc[0], what + ": iteration counts")
    require(bitcast[DType.uint64](rr[1]) == bitcast[DType.uint64](rg[1]), what + ": error (device fold)")
    require(bitcast[DType.uint64](rg[1]) == bitcast[DType.uint64](rc[1]), what + ": error (host column)")
    _same(wr, wg, what + ": W over the device fold")
    _same(hr, hg, what + ": H over the device fold")
    _same(wg, wc, what + ": W GPU vs host column")
    _same(hg, hc, what + ": H GPU vs host column")
    _ = whole^
    _ = host^


def _nmf() raises:
    _nmf_one(1)  # multiplicative update
    _nmf_one(0)  # coordinate descent


struct _MselFold(Movable):
    """The whole X and the fold's int64 index put in the model-selection fold store."""
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


def _pca_fit_transform(whiten: Bool) raises:
    var d = 4
    var k = 2
    var idx = _fold()
    var n = len(idx)
    var whole = _values(_ROWS * d, 17 + Int(whiten))
    var host_x = _gather(whole, d, idx)
    var store = _MselFold(whole, _ROWS, d, idx)
    var comp_a = List[Float32](length=k * d, fill=Float32(0))
    var comp_b = List[Float32](length=k * d, fill=Float32(0))
    var mean_a = List[Float32](length=d, fill=Float32(0))
    var mean_b = List[Float32](length=d, fill=Float32(0))
    var var_a = List[Float32](length=k, fill=Float32(0))
    var var_b = List[Float32](length=k, fill=Float32(0))
    var rat_a = List[Float32](length=k, fill=Float32(0))
    var rat_b = List[Float32](length=k, fill=Float32(0))
    var sin_a = List[Float32](length=k, fill=Float32(0))
    var sin_b = List[Float32](length=k, fill=Float32(0))
    var proj_a = List[Float32](length=n * k, fill=Float32(0))
    var proj_b = List[Float32](length=n * k, fill=Float32(0))
    _ = PCADeviceKernels.fit(PCAFitRequest(Int(host_x.unsafe_ptr()), Int(comp_a.unsafe_ptr()), Int(mean_a.unsafe_ptr()),
        Int(var_a.unsafe_ptr()), Int(rat_a.unsafe_ptr()), Int(sin_a.unsafe_ptr()), Int(proj_a.unsafe_ptr()), n, d, k,
        False, 0, whiten, False, 0, 0, False, 0, True, 0.0, PCAFoldStore()))
    var got = PCAFoldKernels.fit(PCAFitRequest(0, Int(comp_b.unsafe_ptr()), Int(mean_b.unsafe_ptr()),
        Int(var_b.unsafe_ptr()), Int(rat_b.unsafe_ptr()), Int(sin_b.unsafe_ptr()), Int(proj_b.unsafe_ptr()), n, d, k,
        False, 0, whiten, False, 0, 0, False, 0, True, 0.0, store.fold()))
    store.close()
    require(got.finite == 0 and got.in_range == 0, "PCA fold fit_transform status")
    var what = String("PCA fit_transform (whiten ") + String(whiten) + "):"
    _same(comp_a, comp_b, what + " components")
    _same(mean_a, mean_b, what + " mean")
    _same(sin_a, sin_b, what + " singular values")
    _same(proj_a, proj_b, what + " projection")
    _ = whole^
    _ = host_x^


def _pca_projection() raises:
    _pca_fit_transform(False)
    _pca_fit_transform(True)


def _queries() raises:
    var d = 4
    var k = 2
    var idx = _fold()
    var n = len(idx)
    var whole = _values(_ROWS * d, 23)
    var host_x = _gather(whole, d, idx)
    var mean = _values(d, 2)
    var comps = _values(k * d, 7)
    var singular: List[Float32] = [3.0, 1.5]
    var store = _MselFold(whole, _ROWS, d, idx)
    # PCA whitened transform
    var wa = List[Float32](length=n * k, fill=Float32(0))
    var wb = List[Float32](length=n * k, fill=Float32(0))
    var request = PCAQueryRequest(Int(host_x.unsafe_ptr()), Int(mean.unsafe_ptr()), Int(comps.unsafe_ptr()),
        Int(singular.unsafe_ptr()), Int(wa.unsafe_ptr()), n, d, k, _ROWS, False, True, PCAFoldStore())
    _ = PCADeviceKernels.query(request)
    var fold_request = PCAQueryRequest(0, Int(mean.unsafe_ptr()), Int(comps.unsafe_ptr()), Int(singular.unsafe_ptr()),
        Int(wb.unsafe_ptr()), n, d, k, _ROWS, False, True, store.fold())
    var status = PCAFoldKernels.query(fold_request)
    require(status.finite == 0 and status.in_range == 0, "PCA whitened fold transform status")
    _same(wa, wb, "PCA whitened transform over the fold")
    # TruncatedSVD.transform
    var ta = List[Float32](length=n * k, fill=Float32(0))
    var tb = List[Float32](length=n * k, fill=Float32(0))
    var finite = TSVDDeviceKernels.query(TSVDQueryRequest(Int(host_x.unsafe_ptr()), Int(comps.unsafe_ptr()),
        Int(ta.unsafe_ptr()), n, d, k, False, True, False, 0))
    require(finite == 0, "TruncatedSVD host-input transform finite word")
    var tsvd_request = TSVDQueryRequest(0, Int(comps.unsafe_ptr()), Int(tb.unsafe_ptr()), n, d, k, False, True,
        False, 0)
    tsvd_request.fold = store.fold()
    var word = TSVDFoldKernels.query(tsvd_request)
    store.close()
    require(word == 0, "TruncatedSVD fold transform status")
    _same(ta, tb, "TruncatedSVD transform over the fold")
    _ = whole^
    _ = host_x^
    _ = mean^
    _ = comps^
    _ = singular^


def _refusals() raises:
    var whole = _values(_ROWS * _D, 1)
    var idx = _fold()
    var ids = _taken_fold(whole, idx)
    var n = len(idx)
    var cpu = _IpcaRoom()
    var kept = List[Int]()
    var host_refused = False
    try:
        _ = ipca_fit_run[HostExec, NoRDev, False, NoIpcaDev](F32Ptr(unsafe_from_address=0), n, _D, 2, 5,
            _p(cpu.comps), _p(cpu.sing), _p(cpu.mean), _p(cpu.var_), _p(cpu.ev), _p(cpu.evr), False, kept, ids[1])
    except error:
        host_refused = String(error).find("needs a GPU binding") >= 0
    require(host_refused, "IncrementalPCA: the host column refuses a resident X")
    _free(ids[1])
    _free(ids[0])
    # TruncatedSVD fold query: an index past the base's rows answers 2
    var d = 4
    var k = 2
    var bad: List[Int64] = [0, Int64(_ROWS)]
    var wide = _values(_ROWS * d, 5)
    var comps = _values(k * d, 7)
    var store = _MselFold(wide, _ROWS, d, bad)
    var dst = List[Float32](length=2 * k, fill=Float32(0))
    var tsvd_request = TSVDQueryRequest(0, Int(comps.unsafe_ptr()), Int(dst.unsafe_ptr()), 2, d, k, False, True,
        False, 0)
    tsvd_request.fold = store.fold()
    var word = TSVDFoldKernels.query(tsvd_request)
    store.close()
    require(word == 2, "TruncatedSVD fold transform: an out-of-range index answers 2")
    _ = whole^
    _ = wide^
    _ = comps^
    _ = bad^


def _case(unit: Int) raises:
    if unit == 0:
        _ipca()
    elif unit == 1:
        _nmf()
    elif unit == 2:
        _pca_projection()
    elif unit == 3:
        _queries()
    else:
        _refusals()


def test_vdf_scope_neural4_rows() raises:
    check_cases_serial("vdf_scope_neural4_rows", CASES, _case)


def main() raises:
    var suite = TestSuite()
    suite.test[test_vdf_scope_neural4_rows]()
    suite^.run()
