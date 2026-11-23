# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The LabelPropagation / LabelSpreading member entries (core/xneighbors_member.mojo kinds 7 and 8, lane
mf-entries5) against the family's own public RUN (neighbors/label_propagation_run.mojo `lp_fit_run` /
`lp_query_run`, the bodies the estimator's door `_lp_fit` / `_lp_query` runs), on the host x_neighbors image
(x_neighbors/native_descriptor_host.mojo `xn_native_v1`), exact.

The member is handed the parameter words providers/python/member_xneighbors_boundary_door.mojo `_lp_admit` writes
for the estimator's defaults (LabelPropagation kernel='knn', n_neighbors 7, tol 1e-3, max_iter 1000;
LabelSpreading kernel='rbf', gamma 20, alpha 0.2, max_iter 30), every arm off (unfused, old items, raw graph,
dense proba). The direct RUN is called with the same words and the marker word = the class count (every row
labeled), as the door's fit of the same codes computes it. The rows are a fixed generator's (no fixture).

    test_lp_knn_member_equals_run        fit (label_distributions_, transduction codes, n_iter_), predict and
                                         predict_proba (float32 and the float64 cast) equal the RUN's words
    test_ls_rbf_member_equals_run        the same for LabelSpreading on the RBF graph
    test_lp_member_import_round_trip     the exported rows / distributions / scalars imported into a new state
                                         predict the same codes and probabilities
    test_lp_member_refuses_absent_class  a fit whose codes miss a class of the count is refused in words

FAIL-PROOF (each edit makes the named case fail): in `_lp_fit` pass marker 0 instead of `classes` (code 0 is
read as unlabeled) -> test_lp_knn_member_equals_run; pass 1.0 for `slot[].lp_weight` in `_lp_fit` (the
spreading clamp weight) -> test_ls_rbf_member_equals_run;
in `_lp_import` skip the distributions copy -> test_lp_member_import_round_trip; delete the `_lp_codes_check`
call -> test_lp_member_refuses_absent_class.
"""
from std.testing import TestSuite
from core.entry_table import EntryRef, entry_ref_word
from core.device_route import DeviceRoute, DEVICE_CPU
from core.member_abi import (
    MemberCall, MemberMatrix, MemberRef, MEMBER_F64, MEMBER_I32, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT,
    MEMBER_OP_PREDICT, MEMBER_OP_PREDICT_PROBA, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_IMPORT_DONE,
    member_call,
)
from core.xneighbors_member import (
    xneighbors_member_v2, XNEIGHBORS_KIND_LABEL_PROPAGATION, XNEIGHBORS_KIND_LABEL_SPREADING,
    LP_BLOCK_ROWS, LP_BLOCK_DISTRIBUTIONS, LP_BLOCK_TRANSDUCTION, LP_BLOCK_SCALARS,
)
from core.neighbors_native_driver import XnDriver
from neighbors.label_propagation_run import LpFitRequest, LpQueryRequest, lp_fit_run, lp_query_run
from neighbors.neighbors_semi_run import xn_semi_knn_k, xn_lp_static_weight
from neighbors.neighbors_graph_run import xn_f32_word
from x_neighbors.native_descriptor_host import xn_native_v1
from x_neighbors.descriptor_schema import XN_OPERATION_COUNT
from checks.numerics import GLOBAL_NUMERIC_MODE
from checks.native_tests.testsuite_cases import require

comptime _N = 60
comptime _D = 3
comptime _C = 3
comptime _Q = 17
comptime _RBF = 2   # python/mojolearn/_expansion_neighbors.py `_KERNELS["rbf"]`


def _route() -> DeviceRoute:
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var route = DeviceRoute(DEVICE_CPU)
    route.mode = mode
    route.vendor = 0
    route.entry = EntryRef(xn_native_v1, entry_ref_word(mode, 0, XN_OPERATION_COUNT))
    return route


def _member() -> MemberRef:
    return MemberRef(EntryRef(xneighbors_member_v2, 0), 0)


def _rows(count: Int, seed: Int) -> List[Float32]:
    """`count` x _D float32 rows in [0, 1) from a fixed linear congruential generator."""
    var values = List[Float32](length=count * _D, fill=Float32(0.0))
    var s = UInt64(seed)
    for i in range(count * _D):  # small-loop: the fixed test rows (at most 180 words)
        s = s * 6364136223846793005 + 1442695040888963407
        values[i] = Float32(Float64(Int(s >> 40)) / 16777216.0)
    return values^


def _codes(x: List[Float32], count: Int) -> List[Int32]:
    """Three classes by the first coordinate's thirds; rows 0, 1, 2 force every class present."""
    var codes = List[Int32](length=count, fill=Int32(0))
    for i in range(count):  # small-loop: the fixed test codes
        var v = x[i * _D]
        codes[i] = Int32(0) if v < Float32(0.33) else (Int32(1) if v < Float32(0.66) else Int32(2))
    codes[0] = 0
    codes[1] = 1
    codes[2] = 2
    return codes^


def _words(kind: Int, knn: Bool) -> List[Float64]:
    """The words `_lp_admit` writes for the defaults (every arm off)."""
    var spreading = kind == XNEIGHBORS_KIND_LABEL_SPREADING
    var weight = xn_lp_static_weight(0.2) if spreading else 1.0
    var alpha = xn_f32_word(0.2) if spreading else 0.0
    var words: List[Float64] = [Float64(kind), 1.0 if knn else 0.0, 7.0 if knn else 0.0,
        0.0 if knn else Float64(_RBF), 0.0 if knn else 20.0, 0.0, weight, alpha, 0.001,
        30.0 if spreading else 1000.0, 0.0, 0.0, 0.0]
    return words^


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


def _fit(state: Int, x: List[Float32], codes: List[Int32], count: Int, classes: Int) raises:
    var call = MemberCall(_route(), state)
    call.x = MemberMatrix(Int(x.unsafe_ptr()), count, _D)
    call.y = MemberMatrix(Int(codes.unsafe_ptr()), count, 1, MEMBER_I32)
    call.classes = classes
    _ = member_call(_member(), MEMBER_OP_FIT, call)
    _ = x
    _ = codes


def _block_f32(state: Int, block: Int) raises -> List[Float32]:
    var call = MemberCall(_route(), state)
    call.block = block
    _ = member_call(_member(), MEMBER_OP_EXPORT, call)
    var count = call.dst.rows * call.dst.cols
    var out = List[Float32](length=count, fill=Float32(0.0))
    var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    for i in range(count):  # small-loop: one exported block copy
        out[i] = src[i]
    return out^


def _block_i32(state: Int, block: Int) raises -> List[Int32]:
    var call = MemberCall(_route(), state)
    call.block = block
    _ = member_call(_member(), MEMBER_OP_EXPORT, call)
    var count = call.dst.rows * call.dst.cols
    var out = List[Int32](length=count, fill=Int32(0))
    var src = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    for i in range(count):  # small-loop: one exported block copy
        out[i] = src[i]
    return out^


def _scalars(state: Int) raises -> List[Float64]:
    var call = MemberCall(_route(), state)
    call.block = LP_BLOCK_SCALARS
    _ = member_call(_member(), MEMBER_OP_EXPORT, call)
    var src = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=call.dst.data)
    var out: List[Float64] = [src[0], src[1]]
    return out^


def _predict(state: Int, q: List[Float32], count: Int) raises -> List[Int32]:
    var dst = List[Int32](length=count, fill=Int32(-7))
    var call = MemberCall(_route(), state)
    call.x = MemberMatrix(Int(q.unsafe_ptr()), count, _D)
    call.dst = MemberMatrix(Int(dst.unsafe_ptr()), count, 1, MEMBER_I32)
    _ = member_call(_member(), MEMBER_OP_PREDICT, call)
    _ = q
    return dst^


def _proba(state: Int, q: List[Float32], count: Int) raises -> List[Float32]:
    var dst = List[Float32](length=count * _C, fill=Float32(-7.0))
    var call = MemberCall(_route(), state)
    call.x = MemberMatrix(Int(q.unsafe_ptr()), count, _D)
    call.dst = MemberMatrix(Int(dst.unsafe_ptr()), count, _C)
    _ = member_call(_member(), MEMBER_OP_PREDICT_PROBA, call)
    _ = q
    return dst^


def _proba64(state: Int, q: List[Float32], count: Int) raises -> List[Float64]:
    var dst = List[Float64](length=count * _C, fill=-7.0)
    var call = MemberCall(_route(), state)
    call.x = MemberMatrix(Int(q.unsafe_ptr()), count, _D)
    call.dst = MemberMatrix(Int(dst.unsafe_ptr()), count, _C, MEMBER_F64)
    _ = member_call(_member(), MEMBER_OP_PREDICT_PROBA, call)
    _ = q
    return dst^


struct _Direct(Movable):
    var dist: List[Float32]
    var codes: List[Int32]
    var iterations: Int
    var predicted: List[Int32]
    var proba: List[Float32]

    def __init__(out self):
        self.dist = List[Float32]()
        self.codes = List[Int32]()
        self.iterations = 0
        self.predicted = List[Int32]()
        self.proba = List[Float32]()


def _direct(kind: Int, knn: Bool, x: List[Float32], codes: List[Int32], q: List[Float32]) raises -> _Direct:
    """The family's own RUN with the door's words for the same codes (marker = the class count)."""
    var spreading = kind == XNEIGHBORS_KIND_LABEL_SPREADING
    var words = _words(kind, knn)
    var k = xn_semi_knn_k(7, _N) if knn else 0
    var out = _Direct()
    out.dist = List[Float32](length=_N * _C, fill=Float32(0.0))
    out.codes = List[Int32](length=_N, fill=Int32(0))
    var driver = XnDriver(xn_native_v1)
    var request = LpFitRequest(Int(x.unsafe_ptr()), Int(codes.unsafe_ptr()), Int(out.dist.unsafe_ptr()),
        Int(out.codes.unsafe_ptr()), _N, _D, _C, _C, spreading, knn, k, 0 if knn else _RBF,
        0.0 if knn else 20.0, False, words[6], words[7], 0.001, Int(words[9]), False, False)
    var outcome = lp_fit_run(driver, request)
    require(outcome.status == 0, "the direct fit refused finite rows")
    out.iterations = outcome.iterations
    out.predicted = List[Int32](length=_Q, fill=Int32(0))
    out.proba = List[Float32](length=_Q * _C, fill=Float32(0.0))
    var query = LpQueryRequest(Int(q.unsafe_ptr()), Int(x.unsafe_ptr()), Int(out.dist.unsafe_ptr()), 0,
        Int(out.predicted.unsafe_ptr()), _Q, _N, _D, _C, knn, k, 0 if knn else _RBF, 0.0 if knn else 20.0,
        False, True, False, False)
    require(lp_query_run(driver, query) == 0, "the direct predict refused finite rows")
    query.want_labels = False
    query.labels = 0
    query.dst = Int(out.proba.unsafe_ptr())
    require(lp_query_run(driver, query) == 0, "the direct predict_proba refused finite rows")
    _ = x
    _ = codes
    _ = q
    _ = words^
    return out^


def _check_kind(kind: Int, knn: Bool, name: String) raises:
    var x = _rows(_N, 11)
    var codes = _codes(x, _N)
    var q = _rows(_Q, 23)
    var expected = _direct(kind, knn, x, codes, q)
    var state = _new(_words(kind, knn))
    try:
        _fit(state, x, codes, _N, _C)
        var dist = _block_f32(state, LP_BLOCK_DISTRIBUTIONS)
        var trans = _block_i32(state, LP_BLOCK_TRANSDUCTION)
        var scalars = _scalars(state)
        require(len(dist) == _N * _C, name + ": label_distributions_ shape")
        for i in range(_N * _C):  # small-loop: exact words
            require(dist[i] == expected.dist[i], name + ": label_distributions_ word " + String(i))
        for i in range(_N):  # small-loop: exact codes
            require(trans[i] == expected.codes[i], name + ": transduction code " + String(i))
        require(Int(scalars[0]) == expected.iterations, name + ": n_iter_")
        require(Int(scalars[1]) == _D, name + ": n_features_in_")
        var predicted = _predict(state, q, _Q)
        for i in range(_Q):  # small-loop: exact codes
            require(predicted[i] == expected.predicted[i], name + ": predict code " + String(i))
        var proba = _proba(state, q, _Q)
        var proba64 = _proba64(state, q, _Q)
        for i in range(_Q * _C):  # small-loop: exact words
            require(proba[i] == expected.proba[i], name + ": predict_proba word " + String(i))
            require(proba64[i] == Float64(expected.proba[i]), name + ": float64 predict_proba word " + String(i))
    finally:
        _free(state)
    _ = x^
    _ = codes^
    _ = q^


def test_lp_knn_member_equals_run() raises:
    _check_kind(XNEIGHBORS_KIND_LABEL_PROPAGATION, True, "LabelPropagation knn")


def test_ls_rbf_member_equals_run() raises:
    _check_kind(XNEIGHBORS_KIND_LABEL_SPREADING, False, "LabelSpreading rbf")


def test_lp_member_import_round_trip() raises:
    var x = _rows(_N, 11)
    var codes = _codes(x, _N)
    var q = _rows(_Q, 23)
    var words = _words(XNEIGHBORS_KIND_LABEL_PROPAGATION, True)
    var fitted = _new(words)
    var restored = 0
    try:
        _fit(fitted, x, codes, _N, _C)
        var rows = _block_f32(fitted, LP_BLOCK_ROWS)
        var dist = _block_f32(fitted, LP_BLOCK_DISTRIBUTIONS)
        var scalars = _scalars(fitted)
        restored = _new(words)
        var call = MemberCall(_route(), restored)
        call.block = LP_BLOCK_ROWS
        call.x = MemberMatrix(Int(rows.unsafe_ptr()), _N, _D)
        _ = member_call(_member(), MEMBER_OP_IMPORT, call)
        call.block = LP_BLOCK_DISTRIBUTIONS
        call.x = MemberMatrix(Int(dist.unsafe_ptr()), _N, _C)
        _ = member_call(_member(), MEMBER_OP_IMPORT, call)
        call.block = LP_BLOCK_SCALARS
        call.x = MemberMatrix(Int(scalars.unsafe_ptr()), 1, 2, MEMBER_F64)
        call.flags = MEMBER_IMPORT_DONE
        _ = member_call(_member(), MEMBER_OP_IMPORT, call)
        var a = _predict(fitted, q, _Q)
        var b = _predict(restored, q, _Q)
        for i in range(_Q):  # small-loop: exact codes
            require(a[i] == b[i], "imported state: predict code " + String(i))
        var pa = _proba(fitted, q, _Q)
        var pb = _proba(restored, q, _Q)
        for i in range(_Q * _C):  # small-loop: exact words
            require(pa[i] == pb[i], "imported state: predict_proba word " + String(i))
        _ = rows^
        _ = dist^
        _ = scalars^
    finally:
        _free(fitted)
        if restored != 0:
            _free(restored)
    _ = x^
    _ = codes^
    _ = q^
    _ = words^


def test_lp_member_refuses_absent_class() raises:
    var x = _rows(_N, 11)
    var codes = List[Int32](length=_N, fill=Int32(0))
    for i in range(_N):  # small-loop: codes 0 and 1 only
        codes[i] = Int32(i % 2)
    var state = _new(_words(XNEIGHBORS_KIND_LABEL_PROPAGATION, True))
    var refused = False
    try:
        _fit(state, x, codes, _N, _C)
    except error:
        refused = String(error).find("class 2 is absent") >= 0
    _free(state)
    require(refused, "a fit missing class 2 of three is refused in words")
    _ = x^
    _ = codes^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
