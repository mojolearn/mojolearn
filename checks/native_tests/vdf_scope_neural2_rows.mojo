# SPDX-License-Identifier: Apache-2.0
# native-check: gpu
"""THE RESIDENT ROUTE of the neural, scaler and decomposition members (lanes vdf-scope-neural, vdf-scope-neural2,
2026-11-24): a fold taken ON THE DEVICE out of the whole X held in a family's member store, read in place by the
family's GPU fit / predict / transform, gives the same bytes as the same GPU operation over the fold's rows
gathered on the host and uploaded (the host-input route, whose bytes the CPU column and the cross-vendor
verification already pin). Device column only: the stores live on the device.

    case 0  MLP (sequence/ops_impl.mojo `mlp_fit_impl` / `mlp_predict_impl`, `x_device`): X slot of the x_sequence
            member store (sequence/seq_member_store.mojo, `take_rows` of a fold with a repeat and an out-of-order
            row) == host-gathered X: the trained parameters, the loss curve and the predictions, bit for bit
    case 1  RNN (LSTM, `rnn_fit_impl`, `x_device`): rows of T x D words; the parameters and every minibatch loss
    case 2  CNN (sequence/cnn_classifier_program.mojo `cnn_fit_program` / `cnn_predict_program`, `x_device`):
            the slot of the x_sequence store read on the x_cnn context; the parameters, the epoch curve, every
            minibatch loss and the resident predict's probabilities
    case 3  StandardScaler and MinMaxScaler (preprocessing/prep_member_store.mojo `prep_slot_fit` /
            `prep_slot_transform`) == the direct entries (`standard_fit_direct`, `standard_transform_direct`,
            `minmax_fit_direct`, `minmax_transform_direct`) over the host-gathered fold
    case 4  PCA (covariance) and TruncatedSVD: the resident-fold entries (core/pca_native_fold.mojo,
            core/tsvd_native_fold.mojo) over (base, index) ids of the model-selection fold store (the ids the
            members put through core/msel_fold_transport.mojo) == the device entries (PCADeviceKernels,
            TSVDDeviceKernels gram + explained) over the host-gathered fold: components, mean, variances,
            singular values and the PCA transform
    case 5  the store transports refuse what they do not hold: an out-of-range row index on `take_rows`; the fold
            store's TAKE_ROWS

Fail-proof edits (each must fail its case): in `mlp_fit` read `ex.upload` into a fresh buffer from the slot's
address as if it were host memory (case 0 faults or differs); in `_resident_predict` keep the upload under
`x_device` (case 2); in `prep_slot_fit` hand `standard_fit_on` the whole slot's rows (case 3: n is the fold's);
in `_rows_open` put the index as int32 words (case 4 through the member; here: put 4-byte indices, case 4 fails).
"""
from std.memory import bitcast
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases_serial, require
from core.neural_context import process_ctx
from sequence.exec_device import DeviceExec, sequence_ctx
from sequence.seq_member_store import SEQ_MEMBER_STORE, seq_member_store_ptr
from sequence.ops import CELL_LSTM, OPT_ADAM
from sequence.ops_impl import rnn_fit_impl, mlp_fit_impl, mlp_predict_impl
from sequence.recurrent import Net
from sequence.cnn_classifier_program import cnn_fit_program, cnn_open_program, cnn_predict_program
from autograd.ops_api import OPS_DEVICE, ops_session_close
from x_cnn.device import cnn_ctx
from x_cnn.classifier_run import (
    CnnClfPlan, CnnClfFitSettings, cnnclf_run_layout, cnnclf_run_fit_addresses, cnnclf_run_fit_ints,
    cnnclf_run_fit_floats, cnnclf_run_optimizer_word, cnnclf_run_fit_steps, cnnclf_run_open_addresses,
    cnnclf_run_open_ints, cnnclf_run_predict_ints,
)
from preprocessing.estimator import (
    _DEVCTX_SLOT, standard_fit_direct, standard_transform_direct, minmax_fit_direct, minmax_transform_direct,
)
from preprocessing.prep_member_store import PREP_MEMBER_STORE, prep_slot_fit, prep_slot_transform
from core.msel_device_x import msel_fold_store, msel_device_x_context
from core.msel_fold_transport import msel_fold_store_v1
from core.pca_native_api import PCAFitRequest, PCAQueryRequest, PCAFoldStore
from core.pca_native_fold import PCAFoldKernels
from core.pca_native_device import PCADeviceKernels
from core.tsvd_native_api import TSVDFitRequest, TSVD_GRAM, TSVD_EXPLAINED, TSVD_FOLD
from core.tsvd_native_fold import TSVDFoldKernels
from core.tsvd_native_device import TSVDDeviceKernels
from core.native_descriptor_call import native_descriptor_call
from x_prep.transport_abi import XP_DEV_TAKE_ROWS

comptime CASES = 6
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]


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
    """The fold's rows gathered on the host (the oracle's input)."""
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


def _seq_fold(whole: List[Float32], row_words: Int, idx: List[Int64]) raises -> Tuple[Int, Int]:
    """The whole X put in the x_sequence member store and the fold taken on
    the device: (whole slot, fold slot)."""
    var store = SEQ_MEMBER_STORE.get_or_create_ptr()
    var base = store[].put(sequence_ctx(), Int(whole.unsafe_ptr()), len(whole))
    var taken = store[].take_rows(sequence_ctx(), base, row_words, Int(idx.unsafe_ptr()), len(idx))
    return (base, taken)


def _seq_free(ids: Tuple[Int, Int]) raises:
    var store = SEQ_MEMBER_STORE.get_or_create_ptr()
    store[].free(sequence_ctx(), ids[1])
    store[].free(sequence_ctx(), ids[0])


def _mlp() raises:
    var n_whole = 24
    var d = 3
    var h = 4
    var idx = _fold()
    var n = len(idx)
    var whole = _values(n_whole * d, 1)
    var y = _values(n, 2)
    var host_x = _gather(whole, d, idx)
    var ids = _seq_fold(whole, d, idx)
    var slot_x = Int(seq_member_store_ptr(ids[1], n * d))
    var ip: List[Int] = [n, d, 1, 3, 0, 0, 0, 0, 0, 4, 3, 1, 7, 1000000, 1, h]
    var fp: List[Float64] = [1e-3, 0.9, 0.999, 1e-8, 0.9, 0.5, 0.0, 0.0]
    var pa = _values(d * h + h + h + 1, 3)
    var pb = pa.copy()
    var ca = List[Float32](length=3, fill=Float32(0))
    var cb = List[Float32](length=3, fill=Float32(0))
    var ea = DeviceExec()
    _ = mlp_fit_impl(ea, [Int(host_x.unsafe_ptr()), Int(y.unsafe_ptr()), Int(pa.unsafe_ptr()), Int(ca.unsafe_ptr())],
                     ip, fp)
    var eb = DeviceExec()
    _ = mlp_fit_impl(eb, [slot_x, Int(y.unsafe_ptr()), Int(pb.unsafe_ptr()), Int(cb.unsafe_ptr())], ip, fp, True)
    _same(pa, pb, "MLP fit parameters")
    _same(ca, cb, "MLP loss curve")
    var pi: List[Int] = [n, d, 1, 3, 0, 256, 1, h]
    var oa = List[Float32](length=n, fill=Float32(0))
    var ob = List[Float32](length=n, fill=Float32(0))
    _ = mlp_predict_impl(ea, [Int(host_x.unsafe_ptr()), Int(pa.unsafe_ptr()), Int(oa.unsafe_ptr())], pi)
    _ = mlp_predict_impl(eb, [slot_x, Int(pa.unsafe_ptr()), Int(ob.unsafe_ptr())], pi, True)
    _ = pa^
    _same(oa, ob, "MLP predictions")
    _seq_free(ids)
    _ = whole^
    _ = host_x^
    _ = y^


def _rnn() raises:
    var n_whole = 24
    var t = 4
    var d = 3
    var hidden = 5
    var bs = 4
    var idx = _fold()
    var n = len(idx)
    var net = Net(CELL_LSTM, d, hidden, 1, 1)
    var whole = _values(n_whole * t * d, 4)
    var y = _values(n, 5)
    var host_x = _gather(whole, t * d, idx)
    var ids = _seq_fold(whole, t * d, idx)
    var slot_x = Int(seq_member_store_ptr(ids[1], n * t * d))
    var steps = 2 * ((n + bs - 1) // bs)
    var pa = _values(net.n_params(), 6)
    var pb = pa.copy()
    var la = List[Float32](length=steps, fill=Float32(0))
    var lb = List[Float32](length=steps, fill=Float32(0))
    var ra = List[Float32](length=steps, fill=Float32(1e-2))
    var rb = List[Float32](length=steps, fill=Float32(1e-2))
    var ints: List[Int] = [CELL_LSTM, d, hidden, 1, 1, 0, n, t, 2, bs, OPT_ADAM, 0, 1, 11, 0]
    var floats: List[Float64] = [0.9, 0.999, 1e-8, 0.0, 0.0, 0.0]
    var ea = DeviceExec()
    _ = rnn_fit_impl(ea, [Int(host_x.unsafe_ptr()), Int(y.unsafe_ptr()), Int(pa.unsafe_ptr()),
                          Int(la.unsafe_ptr()), Int(ra.unsafe_ptr())], ints, floats)
    _ = ra^
    var eb = DeviceExec()
    _ = rnn_fit_impl(eb, [slot_x, Int(y.unsafe_ptr()), Int(pb.unsafe_ptr()), Int(lb.unsafe_ptr()),
                          Int(rb.unsafe_ptr())], ints, floats, True)
    _ = rb^
    _same(pa, pb, "RNN fit parameters")
    _same(la, lb, "RNN minibatch losses")
    _seq_free(ids)
    _ = whole^
    _ = host_x^
    _ = y^


def _cnn_fit(x: Int, y: Int, mut params: List[Float32], counts: List[Int], mut curve: List[Float32],
             mut steps: List[Float32], plan: CnnClfPlan, settings: CnnClfFitSettings, x_device: Bool) raises:
    var addresses_of = List[Int]()
    var offset = 0
    for i in range(len(counts)):  # driver-loop: over the layer parameter arrays (model structure)
        addresses_of.append(Int(params.unsafe_ptr()) + 4 * offset)
        offset += counts[i]
    var addresses = cnnclf_run_fit_addresses(x, y, Int(curve.unsafe_ptr()), Int(steps.unsafe_ptr()), addresses_of)
    _ = cnn_fit_program[DeviceExec](DeviceExec(cnn_ctx()), addresses, cnnclf_run_fit_ints(plan, settings),
                                    cnnclf_run_fit_floats(settings), x_device)


def _cnn_proba(x: Int, n: Int, mut params: List[Float32], counts: List[Int], classes: Int, row: Int,
               x_device: Bool) raises -> List[Float32]:
    var addresses_of = List[Int]()
    var offset = 0
    for i in range(len(counts)):  # driver-loop: over the layer parameter arrays (model structure)
        addresses_of.append(Int(params.unsafe_ptr()) + 4 * offset)
        offset += counts[i]
    var layout = cnnclf_run_layout(1, 6, 6, 3, 2, [2], classes, 0)
    var plan1 = CnnClfPlan(1, 1, 6, 6, classes, 3, 2, [2], layout.pools.copy())
    var scope = List[Int](length=2, fill=0)
    var sid = cnn_open_program[DeviceExec, OPS_DEVICE](DeviceExec(cnn_ctx()),
        cnnclf_run_open_addresses(Int(scope.unsafe_ptr()), addresses_of), cnnclf_run_open_ints(plan1))
    var out = List[Float32](length=n * classes, fill=Float32(0))
    try:
        var ints = cnnclf_run_predict_ints(scope[0], scope[1], sid, n, False)
        if x_device:
            ints.append(row)
        var addresses: List[Int] = [x, Int(out.unsafe_ptr())]
        _ = cnn_predict_program[DeviceExec, OPS_DEVICE](addresses, ints, x_device)
    finally:
        ops_session_close[DeviceExec, OPS_DEVICE](sid)
    return out^


def _cnn() raises:
    var n_whole = 24
    var row = 36
    var classes = 3
    var batch = 4
    var epochs = 2
    var idx = _fold()
    var n = len(idx)
    var layout = cnnclf_run_layout(1, 6, 6, 3, 2, [2], classes, 0)
    var plan = CnnClfPlan(n, 1, 6, 6, classes, 3, 2, [2], layout.pools.copy())
    var settings = CnnClfFitSettings(batch, epochs, True, UInt64(7), cnnclf_run_optimizer_word("sgd"), False,
                                     0.01, 0.9, 0.9, 0.999, 1e-8, 0.0, 0.0)
    var whole = _values(n_whole * row, 9)
    var y = List[Int32](length=n, fill=Int32(0))
    for i in range(n):  # small-loop(fixture labels)
        y[i] = Int32(i % classes)
    var host_x = _gather(whole, row, idx)
    var ids = _seq_fold(whole, row, idx)
    var slot_x = Int(seq_member_store_ptr(ids[1], n * row))
    var total = 0
    for i in range(len(layout.counts)):  # driver-loop: over the layer parameter arrays (model structure)
        total += layout.counts[i]
    var pa = _values(total, 10)
    var pb = pa.copy()
    var ca = List[Float32](length=epochs, fill=Float32(0))
    var cb = List[Float32](length=epochs, fill=Float32(0))
    var sa = List[Float32](length=epochs * cnnclf_run_fit_steps(n, batch), fill=Float32(0))
    var sb = List[Float32](length=epochs * cnnclf_run_fit_steps(n, batch), fill=Float32(0))
    _cnn_fit(Int(host_x.unsafe_ptr()), Int(y.unsafe_ptr()), pa, layout.counts, ca, sa, plan, settings, False)
    _cnn_fit(slot_x, Int(y.unsafe_ptr()), pb, layout.counts, cb, sb, plan, settings, True)
    _same(pa, pb, "CNN fit parameters")
    _same(ca, cb, "CNN epoch curve")
    _same(sa, sb, "CNN minibatch losses")
    var qa = _cnn_proba(Int(host_x.unsafe_ptr()), n, pa, layout.counts, classes, row, False)
    var qb = _cnn_proba(slot_x, n, pa, layout.counts, classes, row, True)
    _same(qa, qb, "CNN resident predict probabilities")
    _seq_free(ids)
    _ = whole^
    _ = host_x^
    _ = y^


def _scalers() raises:
    var n_whole = 24
    var d = 5
    var idx = _fold()
    var n = len(idx)
    var whole = _values(n_whole * d, 11)
    var host_x = _gather(whole, d, idx)
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var store = PREP_MEMBER_STORE.get_or_create_ptr()
    var base = store[].put(ctx, Int(whole.unsafe_ptr()), n_whole * d)
    var taken = store[].take_rows(ctx, base, d, Int(idx.unsafe_ptr()), n)
    _ = idx^
    # StandardScaler (with_mean, with_std): the three rows, then the transform with them.
    var sa = List[Float32](length=3 * d, fill=Float32(0))
    var sb = List[Float32](length=3 * d, fill=Float32(0))
    require(standard_fit_direct(_F32P(unsafe_from_address=Int(host_x.unsafe_ptr())), n, d, 1, 1,
                                _F32P(unsafe_from_address=Int(sa.unsafe_ptr()))) == 1, "StandardScaler direct fit")
    require(prep_slot_fit(taken, n, d, False, 1, 1, 0.0, 1.0, Int(sb.unsafe_ptr())) == 1, "StandardScaler slot fit")
    _same(sa, sb, "StandardScaler fitted rows")
    var ta = List[Float32](length=n * d, fill=Float32(0))
    var tb = List[Float32](length=n * d, fill=Float32(0))
    var mean = Int(sa.unsafe_ptr())
    var scale = Int(sa.unsafe_ptr()) + 8 * d
    require(standard_transform_direct(_F32P(unsafe_from_address=Int(host_x.unsafe_ptr())),
                                      _F32P(unsafe_from_address=mean), _F32P(unsafe_from_address=scale),
                                      _F32P(unsafe_from_address=Int(ta.unsafe_ptr())), n, d, 0, 1, 1) == 1,
            "StandardScaler direct transform")
    require(prep_slot_transform(taken, n, d, False, 0, 1, 1, 0, 0.0, 1.0, mean, scale, Int(tb.unsafe_ptr())) == 1,
            "StandardScaler slot transform")
    _ = sa^
    _same(ta, tb, "StandardScaler transform")
    # MinMaxScaler (feature_range (0, 1), no clip): the five rows, then the transform with scale_ / min_.
    var ma = List[Float32](length=5 * d, fill=Float32(0))
    var mb = List[Float32](length=5 * d, fill=Float32(0))
    require(minmax_fit_direct(_F32P(unsafe_from_address=Int(host_x.unsafe_ptr())), n, d, 0.0, 1.0,
                              _F32P(unsafe_from_address=Int(ma.unsafe_ptr()))) == 1, "MinMaxScaler direct fit")
    require(prep_slot_fit(taken, n, d, True, 1, 1, 0.0, 1.0, Int(mb.unsafe_ptr())) == 1, "MinMaxScaler slot fit")
    _same(ma, mb, "MinMaxScaler fitted rows")
    var mscale = Int(ma.unsafe_ptr()) + 12 * d
    var moffset = Int(ma.unsafe_ptr()) + 16 * d
    require(minmax_transform_direct(_F32P(unsafe_from_address=Int(host_x.unsafe_ptr())),
                                    _F32P(unsafe_from_address=mscale), _F32P(unsafe_from_address=moffset),
                                    _F32P(unsafe_from_address=Int(ta.unsafe_ptr())), n, d, 0, 0, 0.0, 1.0) == 1,
            "MinMaxScaler direct transform")
    require(prep_slot_transform(taken, n, d, True, 0, 1, 1, 0, 0.0, 1.0, mscale, moffset, Int(tb.unsafe_ptr())) == 1,
            "MinMaxScaler slot transform")
    _ = ma^
    _same(ta, tb, "MinMaxScaler transform")
    store[].free(ctx, taken)
    store[].free(ctx, base)
    _ = ctx^
    _ = whole^
    _ = host_x^


def _decomposition() raises:
    var n_whole = 24
    var d = 4
    var k = 2
    var idx = _fold()
    var n = len(idx)
    var whole = _values(n_whole * d, 13)
    var host_x = _gather(whole, d, idx)
    var ctx = msel_device_x_context()
    var store = msel_fold_store()
    var base = store[].put(ctx, Int(whole.unsafe_ptr()), n_whole * d * 4)
    var index = store[].put(ctx, Int(idx.unsafe_ptr()), n * 8)
    _ = idx^
    var fold = PCAFoldStore(base, n_whole, index)
    # PCA, covariance route, integer n_components.
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
    _ = PCADeviceKernels.fit(PCAFitRequest(Int(host_x.unsafe_ptr()), Int(comp_a.unsafe_ptr()), Int(mean_a.unsafe_ptr()),
        Int(var_a.unsafe_ptr()), Int(rat_a.unsafe_ptr()), Int(sin_a.unsafe_ptr()), 0, n, d, k, False, 0, False,
        False, 0, 0, False, 0, True, 0.0, PCAFoldStore()))
    var got = PCAFoldKernels.fit(PCAFitRequest(0, Int(comp_b.unsafe_ptr()), Int(mean_b.unsafe_ptr()),
        Int(var_b.unsafe_ptr()), Int(rat_b.unsafe_ptr()), Int(sin_b.unsafe_ptr()), 0, n, d, k, False, 0, False,
        False, 0, 0, False, 0, True, 0.0, fold))
    require(got.finite == 0 and got.in_range == 0, "PCA fold fit status")
    _same(comp_a, comp_b, "PCA components")
    _same(mean_a, mean_b, "PCA mean")
    _same(var_a, var_b, "PCA explained variance")
    _same(rat_a, rat_b, "PCA explained variance ratio")
    _same(sin_a, sin_b, "PCA singular values")
    var pa = List[Float32](length=n * k, fill=Float32(0))
    var pb = List[Float32](length=n * k, fill=Float32(0))
    _ = PCADeviceKernels.query(PCAQueryRequest(Int(host_x.unsafe_ptr()), Int(mean_a.unsafe_ptr()),
        Int(comp_a.unsafe_ptr()), 0, Int(pa.unsafe_ptr()), n, d, k, n, False, False, PCAFoldStore()))
    var status = PCAFoldKernels.query(PCAQueryRequest(0, Int(mean_a.unsafe_ptr()), Int(comp_a.unsafe_ptr()), 0,
        Int(pb.unsafe_ptr()), n, d, k, n, False, False, fold))
    _ = comp_a^
    _ = mean_a^
    require(status.finite == 0 and status.in_range == 0, "PCA fold transform status")
    _same(pa, pb, "PCA transform")
    # TruncatedSVD: the device gram fit then its explained step == the fold fit.
    var tc_a = List[Float32](length=k * d, fill=Float32(0))
    var tc_b = List[Float32](length=k * d, fill=Float32(0))
    var ts_a = List[Float32](length=k, fill=Float32(0))
    var ts_b = List[Float32](length=k, fill=Float32(0))
    var tv_a = List[Float32](length=k, fill=Float32(0))
    var tv_b = List[Float32](length=k, fill=Float32(0))
    var tr_a = List[Float32](length=k, fill=Float32(0))
    var tr_b = List[Float32](length=k, fill=Float32(0))
    var request = TSVDFitRequest(Int(host_x.unsafe_ptr()), Int(tc_a.unsafe_ptr()), Int(ts_a.unsafe_ptr()),
        Int(tv_a.unsafe_ptr()), Int(tr_a.unsafe_ptr()), n, d, k, 0, 0, False, 0, True, TSVD_GRAM, PCAFoldStore())
    _ = TSVDDeviceKernels.fit(request)
    request.route = TSVD_EXPLAINED
    request.test_finite = False
    _ = TSVDDeviceKernels.fit(request)
    var words = TSVDFoldKernels.fit(TSVDFitRequest(0, Int(tc_b.unsafe_ptr()), Int(ts_b.unsafe_ptr()),
        Int(tv_b.unsafe_ptr()), Int(tr_b.unsafe_ptr()), n, d, k, 0, 0, False, 0, True, TSVD_FOLD, fold))
    require(words.finite == 0 and words.in_range == 0, "TruncatedSVD fold fit status")
    _same(tc_a, tc_b, "TruncatedSVD components")
    _same(ts_a, ts_b, "TruncatedSVD singular values")
    _same(tv_a, tv_b, "TruncatedSVD explained variance")
    _same(tr_a, tr_b, "TruncatedSVD explained variance ratio")
    store[].free(ctx, index)
    store[].free(ctx, base)
    _ = whole^
    _ = host_x^


def _refusals() raises:
    var whole = _values(24 * 3, 1)
    var bad: List[Int64] = [0, 24]
    var store = SEQ_MEMBER_STORE.get_or_create_ptr()
    var base = store[].put(sequence_ctx(), Int(whole.unsafe_ptr()), len(whole))
    var refused = False
    try:
        _ = store[].take_rows(sequence_ctx(), base, 3, Int(bad.unsafe_ptr()), 2)
    except error:
        refused = String(error).find("out of range") >= 0
    store[].free(sequence_ctx(), base)
    require(refused, "x_sequence store: take_rows refuses an out-of-range row")
    var a: List[Int] = [Int(bad.unsafe_ptr())]
    var p: List[Int] = [0, 3, 2]
    var fold_refused = False
    try:
        _ = native_descriptor_call(msel_fold_store_v1, XP_DEV_TAKE_ROWS, a, p, List[Float64]())
    except error:
        fold_refused = String(error).find("no take_rows") >= 0
    require(fold_refused, "model-selection fold store: TAKE_ROWS is refused")
    _ = bad^
    _ = whole^


def _case(unit: Int) raises:
    if unit == 0:
        _mlp()
    elif unit == 1:
        _rnn()
    elif unit == 2:
        _cnn()
    elif unit == 3:
        _scalers()
    elif unit == 4:
        _decomposition()
    else:
        _refusals()


def test_resident_rows() raises:
    check_cases_serial("vdf_scope_neural2_rows", CASES, _case)


def main() raises:
    var suite = TestSuite()
    suite.test[test_resident_rows]()
    suite^.run()
