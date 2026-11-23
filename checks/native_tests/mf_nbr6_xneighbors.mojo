# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The XNeighbors member routes lane mf-nbr6 added (core/xneighbors_member.mojo), against the family's own
public RUN on the host x_neighbors image (x_neighbors/native_descriptor_host.mojo `xn_native_v1`), exact. The
words are the ones providers/python/member_xneighbors_boundary_door.mojo writes; the rows are a fixed
generator's (no fixture).

    test_ocsvm_precomputed_member_equals_run   OneClassSVM kernel='precomputed' (word [13]): the fit on a
                                               linear Gram matrix equals `oc_fit_run` with precomputed=True
                                               (support indices, dual coefficients, rho); the support-vector
                                               block is supports x 0; decision_function on query Gram columns
                                               equals `oc_decision_run` on OC_QUERY_GRAM_PRECOMPUTED
    test_ocsvm_precomputed_refuses_rectangle    a non-square X is refused in `oc_fit_unpack`'s words
    test_kpca_precomputed_words                 KernelPCA KPCA_PRECOMPUTED: a non-square fit X is refused in
                                               `kpca_fit`'s words before the route; an imported state refuses a
                                               query whose columns are not the fitted rows, in
                                               `kpca_transform`'s words
    test_imputer_k0_fits_then_refuses           KNNImputer n_neighbors=0: NEW and FIT succeed (the door's fit
                                               takes it); transform is refused 'n_neighbors must be >= 1'
    test_schi2_takes_five_words                 SkewedChi2Sampler NEW takes five words (word [4] the one-call
                                               transform) and refuses four
    test_svgp_given_inducing_equals_run         SVGP with caller-given inducing points (words [7..9]): Z_,
                                               alpha, C and the elbo equal `svgp_fit_run` with pick_rows False on
                                               the same points; a column mismatch is refused in words

FAIL-PROOF (each edit makes the named case fail): in `_oc_fit` set `request.precomputed = False` ->
test_ocsvm_precomputed_member_equals_run; in `_oc_query` drop the OC_QUERY_GRAM_PRECOMPUTED branch ->
test_ocsvm_precomputed_member_equals_run; delete the square check in `_oc_fit` ->
test_ocsvm_precomputed_refuses_rectangle; delete the precomputed column check in `_kp_transform` (or move
`_kp_fit`'s square check after the route check) -> test_kpca_precomputed_words; restore
`local_check_n_neighbors(Int(words[1]))` in `_new`'s KNNImputer branch (or drop it from `_im_transform`) ->
test_imputer_k0_fits_then_refuses; set the SkewedChi2Sampler count back to 4 -> test_schi2_takes_five_words;
pass `True` for pick_rows in `_sv_fit` (or skip the given copy) -> test_svgp_given_inducing_equals_run.
"""
from std.testing import TestSuite
from core.entry_table import EntryRef, entry_ref_word
from core.device_route import DeviceRoute, DEVICE_CPU
from core.member_abi import (
    MemberCall, MemberMatrix, MemberRef, MEMBER_F64, MEMBER_I32, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT,
    MEMBER_OP_DECISION, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_OP_TRANSFORM, MEMBER_IMPORT_DONE, member_call,
)
from core.xneighbors_member import (
    xneighbors_member_v2, XNEIGHBORS_KIND_OCSVM, XNEIGHBORS_KIND_KERNEL_PCA, XNEIGHBORS_KIND_IMPUTER,
    XNEIGHBORS_KIND_SCHI2, XNEIGHBORS_KIND_SVGP, OC_BLOCK_VECTORS, OC_BLOCK_DUAL, OC_BLOCK_SUPPORT,
    OC_BLOCK_SCALARS, OC_GAMMA_VALUE, KP_BLOCK_ROWS, KP_BLOCK_VALUES, KP_BLOCK_VECTORS, KP_BLOCK_COLUMNS,
    KP_BLOCK_TOTAL, KP_BLOCK_GAMMA, SV_BLOCK_Z, SV_BLOCK_ALPHA, SV_BLOCK_C, SV_BLOCK_SCALARS,
)
from core.kpca_native_plan import KPCA_PRECOMPUTED
from core.neighbors_native_driver import XnDriver
from core.ocsvm_resident_api import OcResidentDriver
from svm.svc_ocsvm_native_run import (
    OC_SOLVE_GRAM, OC_QUERY_GRAM_PRECOMPUTED, OcFitRequest, OcQueryRequest, oc_fit_run, oc_decision_run,
)
from neighbors.svgp_run import SvgpFitRequest, svgp_fit_run
from x_neighbors.native_descriptor_host import xn_native_v1
from x_neighbors.descriptor_schema import XN_OPERATION_COUNT
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.native_tests.testsuite_cases import require

comptime _D = 3
comptime _N = 24
comptime _Q = 7


def _route() -> DeviceRoute:
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = mode
    route.vendor = 0
    route.entry = EntryRef(xn_native_v1, entry_ref_word(mode, 0, XN_OPERATION_COUNT))
    return route


def _member() -> MemberRef:
    return MemberRef(EntryRef(xneighbors_member_v2, 0), 0)


def _rows(count: Int, width: Int, seed: Int) -> List[Float32]:
    """`count` x `width` float32 words in [0, 1) from a fixed linear congruential generator."""
    var values = List[Float32](length=count * width, fill=Float32(0.0))
    var s = UInt64(seed)
    for i in range(count * width):  # small-loop: the fixed test rows (at most a few hundred words)
        s = s * 6364136223846793005 + 1442695040888963407
        values[i] = Float32(Float64(Int(s >> 40)) / 16777216.0)
    return values^


def _gram(a: List[Float32], na: Int, b: List[Float32], nb: Int) -> List[Float32]:
    """The linear kernel a b^T (na x nb float32), the caller's precomputed matrix."""
    var k = List[Float32](length=na * nb, fill=Float32(0.0))
    for i in range(na):  # small-loop: the fixed test Gram (at most 24 x 24)
        for j in range(nb):  # small-loop: one Gram row
            var acc = Float32(0.0)
            for f in range(_D):  # small-loop: three features
                acc += a[i * _D + f] * b[j * _D + f]
            k[i * nb + j] = acc
    return k^


def _new(words: List[Float64]) raises -> Int:
    var call = MemberCall(_route())
    call.params = Int(words.unsafe_ptr())
    call.param_count = len(words)
    _ = member_call(_member(), MEMBER_OP_NEW, call)
    _ = words
    return call.state


def _free(state: Int):
    var call = MemberCall(_route(), state)
    try:
        _ = member_call(_member(), MEMBER_OP_FREE, call)
    except:
        pass


def _export(state: Int, block: Int) raises -> MemberMatrix:
    var call = MemberCall(_route(), state)
    call.block = block
    _ = member_call(_member(), MEMBER_OP_EXPORT, call)
    return call.dst


def _f32_words(view: MemberMatrix) -> List[Float32]:
    var count = view.rows * view.cols
    var out = List[Float32](length=count, fill=Float32(0.0))
    var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=view.data)
    for i in range(count):  # small-loop: one exported block copy
        out[i] = src[i]
    return out^


def _ocsvm_words() -> List[Float64]:
    """`_ocsvm_admit`'s precomputed words: nu 0.5, max_iter -1 (cap 10000000), tol 1e-3, the Gram solve."""
    var words: List[Float64] = [Float64(XNEIGHBORS_KIND_OCSVM), 0.0, 0.5, 0.0, 0.0, Float64(OC_GAMMA_VALUE), 0.0,
        10000000.0, 0.001, Float64(OC_SOLVE_GRAM), 0.0, 0.0, 0.0, 1.0]
    return words^


def test_ocsvm_precomputed_member_equals_run() raises:
    var x = _rows(_N, _D, 11)
    var q = _rows(_Q, _D, 29)
    var k = _gram(x, _N, x, _N)
    var kq = _gram(q, _Q, x, _N)
    var driver = XnDriver(xn_native_v1)
    var request = OcFitRequest()
    request.x = Int(k.unsafe_ptr())
    request.rows = _N
    request.features = _N
    request.precomputed = True
    request.nu = 0.5
    request.cap = 10000000
    request.tol = 0.001
    request.solve = OC_SOLVE_GRAM
    var run = oc_fit_run(driver, Optional[OcResidentDriver](), request)
    var supports = run.state.supports
    var want_support = List[Int32](length=max(supports, 1), fill=Int32(0))
    var want_dual = List[Float32](length=max(supports, 1), fill=Float32(0.0))
    for i in range(supports):  # small-loop: the solve's support prefix
        want_support[i] = run.state.support_rows[i]
        want_dual[i] = run.state.support_values[i]
    var rho = run.rho
    var query = OcQueryRequest()
    query.path = OC_QUERY_GRAM_PRECOMPUTED
    query.query = Int(kq.unsafe_ptr())
    query.rows = _Q
    query.features = _N
    query.supports = supports
    query.support = Int(want_support.unsafe_ptr())
    query.support_count = supports
    query.coefficients = Int(want_dual.unsafe_ptr())
    query.offset = rho
    var want_decision = List[Float32](length=_Q, fill=Float32(0.0))
    oc_decision_run(driver, query, Int(want_decision.unsafe_ptr()))
    var state = _new(_ocsvm_words())
    try:
        var call = MemberCall(_route(), state)
        call.x = MemberMatrix(Int(k.unsafe_ptr()), _N, _N)
        _ = member_call(_member(), MEMBER_OP_FIT, call)
        var support = _export(state, OC_BLOCK_SUPPORT)
        require(support.cols == supports, "precomputed: the support count equals the RUN's")
        var support_words = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=support.data)
        var dual = _f32_words(_export(state, OC_BLOCK_DUAL))
        for i in range(supports):  # small-loop: exact support words
            require(support_words[i] == want_support[i], "precomputed: support index " + String(i))
            require(dual[i] == want_dual[i], "precomputed: dual coefficient " + String(i))
        var vectors = _export(state, OC_BLOCK_VECTORS)
        require(vectors.rows == supports and vectors.cols == 0, "precomputed: support_vectors_ is supports x 0")
        var scalars = _export(state, OC_BLOCK_SCALARS)
        var words = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=scalars.data)
        require(words[1] == rho, "precomputed: offset_ equals the RUN's rho")
        require(Int(words[3]) == _N, "precomputed: n_features_in_ is the Gram's columns")
        var got = List[Float32](length=_Q, fill=Float32(-7.0))
        var ask = MemberCall(_route(), state)
        ask.x = MemberMatrix(Int(kq.unsafe_ptr()), _Q, _N)
        ask.dst = MemberMatrix(Int(got.unsafe_ptr()), _Q, 1)
        _ = member_call(_member(), MEMBER_OP_DECISION, ask)
        for i in range(_Q):  # small-loop: exact decision words
            require(got[i] == want_decision[i], "precomputed: decision_function row " + String(i))
        _ = got^
    finally:
        _free(state)
    _ = run^
    _ = x^
    _ = q^
    _ = k^
    _ = kq^
    _ = want_support^
    _ = want_dual^
    _ = want_decision^


def test_ocsvm_precomputed_refuses_rectangle() raises:
    var x = _rows(_N, _D, 5)
    var state = _new(_ocsvm_words())
    var refused = String("")
    try:
        var call = MemberCall(_route(), state)
        call.x = MemberMatrix(Int(x.unsafe_ptr()), _N, _D)
        _ = member_call(_member(), MEMBER_OP_FIT, call)
    except e:
        refused = String(e)
    finally:
        _free(state)
    require("Precomputed matrix must be a square matrix." in refused, "precomputed: a rectangle is refused")
    _ = x^


def test_kpca_precomputed_words() raises:
    var words: List[Float64] = [Float64(XNEIGHBORS_KIND_KERNEL_PCA), Float64(KPCA_PRECOMPUTED), -1.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    var x = _rows(_N, _D, 7)
    var state = _new(words)
    var refused = String("")
    try:
        var call = MemberCall(_route(), state)   # no eigensolver entries: the square check comes first
        call.x = MemberMatrix(Int(x.unsafe_ptr()), _N, _D)
        _ = member_call(_member(), MEMBER_OP_FIT, call)
    except e:
        refused = String(e)
    finally:
        _free(state)
    require("Precomputed matrix must be a square matrix." in refused, "kpca precomputed: a rectangle is refused")
    # An imported fitted state of 4 rows: a query of 3 columns is refused in `kpca_transform`'s words.
    var fit_x = _rows(4, 4, 3)
    var values: List[Float32] = [2.0, 1.0]
    var vectors = _rows(4, 2, 9)
    var columns = _rows(1, 4, 13)
    var total: List[Float32] = [0.5]
    var gamma: List[Float64] = [0.25]
    state = _new(words)
    refused = String("")
    try:
        var call = MemberCall(_route(), state)
        call.block = KP_BLOCK_ROWS
        call.x = MemberMatrix(Int(fit_x.unsafe_ptr()), 4, 4)
        _ = member_call(_member(), MEMBER_OP_IMPORT, call)
        call.block = KP_BLOCK_VALUES
        call.x = MemberMatrix(Int(values.unsafe_ptr()), 1, 2)
        _ = member_call(_member(), MEMBER_OP_IMPORT, call)
        call.block = KP_BLOCK_VECTORS
        call.x = MemberMatrix(Int(vectors.unsafe_ptr()), 4, 2)
        _ = member_call(_member(), MEMBER_OP_IMPORT, call)
        call.block = KP_BLOCK_COLUMNS
        call.x = MemberMatrix(Int(columns.unsafe_ptr()), 1, 4)
        _ = member_call(_member(), MEMBER_OP_IMPORT, call)
        call.block = KP_BLOCK_TOTAL
        call.x = MemberMatrix(Int(total.unsafe_ptr()), 1, 1)
        _ = member_call(_member(), MEMBER_OP_IMPORT, call)
        call.block = KP_BLOCK_GAMMA
        call.x = MemberMatrix(Int(gamma.unsafe_ptr()), 1, 1, MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(_member(), MEMBER_OP_IMPORT, call)
        var q = _rows(2, 3, 17)
        var dst = List[Float32](length=4, fill=Float32(0.0))
        var ask = MemberCall(_route(), state)
        ask.x = MemberMatrix(Int(q.unsafe_ptr()), 2, 3)
        ask.dst = MemberMatrix(Int(dst.unsafe_ptr()), 2, 2)
        _ = member_call(_member(), MEMBER_OP_TRANSFORM, ask)
        _ = q^
        _ = dst^
    except e:
        refused = String(e)
    finally:
        _free(state)
    require("KernelPCA: a precomputed kernel needs 4 columns (the fitted rows), got 3" in refused,
        "kpca precomputed: a query of the wrong width is refused in the door's words")
    _ = x^
    _ = fit_x^
    _ = values^
    _ = vectors^
    _ = columns^
    _ = total^
    _ = gamma^


def test_imputer_k0_fits_then_refuses() raises:
    # `_imputer_admit`'s words for n_neighbors=0, NaN missing, every arm off.
    var words: List[Float64] = [Float64(XNEIGHBORS_KIND_IMPUTER), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    var x = _rows(_N, _D, 23)
    var state = _new(words)
    var refused = String("")
    try:
        var call = MemberCall(_route(), state)
        call.x = MemberMatrix(Int(x.unsafe_ptr()), _N, _D)
        _ = member_call(_member(), MEMBER_OP_FIT, call)
        var dst = List[Float32](length=_N * _D, fill=Float32(0.0))
        var ask = MemberCall(_route(), state)
        ask.x = MemberMatrix(Int(x.unsafe_ptr()), _N, _D)
        ask.dst = MemberMatrix(Int(dst.unsafe_ptr()), _N, _D)
        try:
            _ = member_call(_member(), MEMBER_OP_TRANSFORM, ask)
        except e:
            refused = String(e)
        _ = dst^
    finally:
        _free(state)
    require("n_neighbors must be >= 1" in refused, "imputer: n_neighbors 0 fits and is refused at transform")
    _ = x^


def test_schi2_takes_five_words() raises:
    var five: List[Float64] = [Float64(XNEIGHBORS_KIND_SCHI2), 10.0, 1.0, 42.0, 0.0]
    var state = _new(five)
    _free(state)
    var four: List[Float64] = [Float64(XNEIGHBORS_KIND_SCHI2), 10.0, 1.0, 42.0]
    var refused = String("")
    try:
        _free(_new(four))
    except e:
        refused = String(e)
    require("five parameters" in refused, "schi2: four words are refused")
    _ = five^
    _ = four^


def test_svgp_given_inducing_equals_run() raises:
    var m = 5
    var x = _rows(_N, _D, 31)
    var y = _rows(_N, 1, 37)
    var z = _rows(m, _D, 41)
    var gamma = Float64(0.5)
    var kv = Float64(1.0)
    var nv = Float64(0.1)
    var jitter = Float64(1e-6)
    var want_z = z.copy()
    var want_alpha = List[Float32](length=m, fill=Float32(0.0))
    var want_c = List[Float32](length=m * m, fill=Float32(0.0))
    var want_qmu = List[Float32](length=m, fill=Float32(0.0))
    var want_qsqrt = List[Float32](length=m * m, fill=Float32(0.0))
    var fitted = svgp_fit_run(XnDriver(xn_native_v1), SvgpFitRequest(Int(x.unsafe_ptr()), Int(y.unsafe_ptr()),
        Int(want_z.unsafe_ptr()), Int(want_alpha.unsafe_ptr()), Int(want_c.unsafe_ptr()),
        Int(want_qmu.unsafe_ptr()), Int(want_qsqrt.unsafe_ptr()), _N, m, _D, False, gamma, kv, nv, jitter))
    var words: List[Float64] = [Float64(XNEIGHBORS_KIND_SVGP), 100.0, gamma, kv, nv, jitter, 0.0,
        Float64(Int(z.unsafe_ptr())), Float64(m), Float64(_D)]
    var state = _new(words)
    try:
        var call = MemberCall(_route(), state)
        call.x = MemberMatrix(Int(x.unsafe_ptr()), _N, _D)
        call.y = MemberMatrix(Int(y.unsafe_ptr()), _N, 1)
        _ = member_call(_member(), MEMBER_OP_FIT, call)
        var got_z = _f32_words(_export(state, SV_BLOCK_Z))
        var got_alpha = _f32_words(_export(state, SV_BLOCK_ALPHA))
        var got_c = _f32_words(_export(state, SV_BLOCK_C))
        require(len(got_z) == m * _D and len(got_alpha) == m and len(got_c) == m * m,
            "svgp given: m is the given rows")
        for i in range(m * _D):  # small-loop: exact Z words
            require(got_z[i] == want_z[i], "svgp given: Z_ word " + String(i))
        for i in range(m):  # small-loop: exact alpha words
            require(got_alpha[i] == want_alpha[i], "svgp given: alpha word " + String(i))
        for i in range(m * m):  # small-loop: exact C words
            require(got_c[i] == want_c[i], "svgp given: C word " + String(i))
        var scalars = _export(state, SV_BLOCK_SCALARS)
        require(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=scalars.data)[0] == fitted[0],
            "svgp given: elbo equals the RUN's")
    finally:
        _free(state)
    # Given points of another width are refused in words.
    var wrong: List[Float64] = [Float64(XNEIGHBORS_KIND_SVGP), 100.0, gamma, kv, nv, jitter, 0.0,
        Float64(Int(z.unsafe_ptr())), Float64(m - 1), Float64(_D + 1)]
    state = _new(wrong)
    var refused = String("")
    try:
        var call = MemberCall(_route(), state)
        call.x = MemberMatrix(Int(x.unsafe_ptr()), _N, _D)
        call.y = MemberMatrix(Int(y.unsafe_ptr()), _N, 1)
        _ = member_call(_member(), MEMBER_OP_FIT, call)
    except e:
        refused = String(e)
    finally:
        _free(state)
    require("inducing_points has 4 features, but X has 3 features" in refused, "svgp given: a width mismatch")
    _ = x^
    _ = y^
    _ = z^
    _ = want_z^
    _ = want_alpha^
    _ = want_c^
    _ = want_qmu^
    _ = want_qsqrt^
    _ = words^
    _ = wrong^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
