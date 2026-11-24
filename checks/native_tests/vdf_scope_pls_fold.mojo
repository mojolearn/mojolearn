# SPDX-License-Identifier: Apache-2.0
# native-check: cpu gpu
"""THE RESIDENT ROUTE of the PLS members (lane vdf-scope-pls, 2026-11-24; core/pls_member.mojo), in a process with
no Python: a fold taken ON THE DEVICE out of the whole X held in the x_decomp device pool (x_decomp/member_store.mojo
`x_decomp_store_v1`, the transport the PLS member door puts in `route.fifth`) and read in place by the typed kit
(bindings/decomp_kit_run.mojo `borrow_device`) gives the same bytes as the same kit over the fold's rows gathered on
the host (the host-input route) and as the host image's kit over that gather (the CPU column's code). The fit's
XD_PLS_FIT reads the centred X and Y as pool ids (int words 9 / 10), so no X words come down.

Host column (CPU_COLUMN_BUILD):
    case 0   the host image refuses a pool word in XD_PLS_FIT; the host kit fits the fixture (PLSRegression)

Device column (each kind: PLSRegression nipals scaled, PLSCanonical svd, CCA mode B):
    case 0   XD_PLS_FIT itself: centred X / Y as pool ids == as host words (every block, done, iteration counts)
    case 1   pls_fit_run over the taken fold (borrow_device) == over the host gather on the device kit == over the
             host gather on the host kit: coef, means, stds, rotations, the six blocks, iteration counts
    case 2   predict (pls_predict_run) and transform (pls_project) over a taken fold == over the gather, device
             and host kits
    case 3   inverse_transform (pls_rebuild) over taken rows of a scores matrix == over the gather

FAIL-PROOF EDITS (each must fail its case): in x_decomp/pls_dev.mojo `pls_fit_dev_run` build `xk` from `pxa`
whatever `x_dev` says (device 0 faults: a null X); in bindings/pls_run.mojo pass `x_dev` for both pool words
(device 1: Y is X); delete the pool-word refusal of x_decomp/descriptor_owner_host.mojo (host 0).

Not covered here (OWED): the member entry itself over a search RUN with `plan.scope` / `plan.pin`
(core/pls_member.mojo `_rows`, the door's store route); its end-to-end check is the msel search RUN.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, check_cases_serial, CPU_COLUMN_BUILD, require
from core.native_descriptor_call import native_descriptor_call
from core.entry_table import EntryRef, entry_ref_word
from x_prep.transport_abi import XP_DEV_PUT, XP_DEV_FREE, XP_DEV_TAKE_ROWS
from x_decomp.descriptor_schema import (
    XD_MODE, XD_VENDOR, XD_CAPABILITIES, XD_CAP_DEV_SVD, XD_OPERATION_COUNT, XD_PLS_FIT, XD_DEV_ALLOC, XD_DEV_FREE,
    XD_DEV_UPLOAD,
)
from x_decomp.descriptor_mani_schema import XD_CAP_DEV_LU, XD_CAP_DEV_EIGH
from x_decomp.descriptor_host import x_decomp_host_v1
from x_decomp.descriptor_device import x_decomp_device_v1
from x_decomp.member_store import x_decomp_store_v1
from bindings.decomp_kit_run import (
    DKit, DK_HAS_DEV_SVD, DK_HAS_DEV_LU, DK_HAS_DEV_EIGH_TO, DK_HAS_S_FLAGS, DK_HAS_QFIX_FLAGS, DK_HAS_IDN_FLAGS,
)
from bindings.pls_run import PlsFitWords, pls_fit_run, pls_project, pls_rebuild, pls_predict_run

comptime _ROWS = 24
comptime _D = 5
comptime _Q = 2
comptime _NC = 2
comptime _KINDS = 3


def _values(n: Int, salt: Int) -> List[Float32]:
    var out = List[Float32](length=n, fill=Float32(0))
    for i in range(n):  # small-loop(fixture values)
        out[i] = Float32((i * 7 + salt * 5 + (i * i) % 11) % 17 - 8) / Float32(8)
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
        require(a[i].to_bits() == b[i].to_bits(),
                what + ": word " + String(i) + " differs (" + String(a[i]) + " vs " + String(b[i]) + ")")


def _words(mut kit: DKit, m: Int) raises -> List[Float32]:
    var n = kit.size(m)
    var out = List[Float32](length=n, fill=Float32(0))
    var p = kit.ptr(m)
    for i in range(n):  # small-loop(oracle: a kit matrix's words)
        out[i] = p[i]
    return out^


def _kind(k: Int) -> PlsFitWords:
    # (components, max_iter, mode_b, canonical, svd_algo, scale, tol)
    if k == 0:
        return PlsFitWords(_NC, 500, False, False, False, True, 1e-06)    # PLSRegression
    if k == 1:
        return PlsFitWords(_NC, 500, False, True, True, True, 1e-06)      # PLSCanonical, svd
    return PlsFitWords(_NC, 500, True, True, False, True, 1e-06)          # CCA, mode B


def _host_kit() raises -> DKit:
    var mode = native_descriptor_call(x_decomp_host_v1, XD_MODE, List[Int](), List[Int](), List[Float64]())
    return DKit(EntryRef(x_decomp_host_v1, entry_ref_word(mode, 0, XD_OPERATION_COUNT)), True, False, 0)


def _device_kit() raises -> DKit:
    var mode = native_descriptor_call(x_decomp_device_v1, XD_MODE, List[Int](), List[Int](), List[Float64]())
    var vendor = native_descriptor_call(x_decomp_device_v1, XD_VENDOR, List[Int](), List[Int](), List[Float64]())
    var caps = native_descriptor_call(x_decomp_device_v1, XD_CAPABILITIES, List[Int](), List[Int](), List[Float64]())
    var present = DK_HAS_S_FLAGS | DK_HAS_QFIX_FLAGS | DK_HAS_IDN_FLAGS
    if (caps & XD_CAP_DEV_SVD) != 0:
        present |= DK_HAS_DEV_SVD
    if (caps & XD_CAP_DEV_LU) != 0:
        present |= DK_HAS_DEV_LU
    if (caps & XD_CAP_DEV_EIGH) != 0:
        present |= DK_HAS_DEV_EIGH_TO
    return DKit(EntryRef(x_decomp_device_v1, entry_ref_word(mode, vendor, XD_OPERATION_COUNT)), True, True, present)


def _store(op: Int, a: List[Int], p: List[Int]) raises -> Int:
    return native_descriptor_call(x_decomp_store_v1, op, a, p, List[Float64]())


def _taken(whole: List[Float32], row_words: Int, idx: List[Int64]) raises -> Tuple[Int, Int]:
    """(the whole X's pool id, the fold's pool id), as `member_rows_open` takes them."""
    var a: List[Int] = [Int(whole.unsafe_ptr())]
    var p: List[Int] = [len(whole)]
    var base = _store(XP_DEV_PUT, a, p)
    var ia: List[Int] = [Int(idx.unsafe_ptr())]
    var ip: List[Int] = [base, row_words, len(idx)]
    var taken = _store(XP_DEV_TAKE_ROWS, ia, ip)
    return (base, taken)


def _free(id: Int) raises:
    var p: List[Int] = [id]
    _ = _store(XP_DEV_FREE, List[Int](), p)


struct _Fitted(Movable):
    """One fit's public words, in a fixed order."""
    var blocks: List[List[Float32]]
    var done: Int
    var iterations: List[Int32]

    def __init__(out self):
        self.blocks = List[List[Float32]]()
        self.done = 0
        self.iterations = List[Int32]()


def _fit(mut kit: DKit, x: Int, ym: Int, k: Int) raises -> _Fitted:
    var fit = pls_fit_run(kit, x, ym, _kind(k), False)
    var out = _Fitted()
    out.done = fit.done
    out.iterations = fit.iterations.copy()
    out.blocks.append(_words(kit, fit.coef))
    out.blocks.append(_words(kit, fit.x_mean))
    out.blocks.append(_words(kit, fit.x_std))
    out.blocks.append(_words(kit, fit.y_mean))
    out.blocks.append(_words(kit, fit.y_std))
    out.blocks.append(_words(kit, fit.x_rotations))
    out.blocks.append(_words(kit, fit.y_rotations))
    for b in range(len(fit.outs)):  # small-loop(the six fitted blocks)
        out.blocks.append(_words(kit, fit.outs[b]))
    return out^


def _same_fit(a: _Fitted, b: _Fitted, what: String) raises:
    require(a.done == b.done, what + ": fitted component counts differ")
    require(len(a.iterations) == len(b.iterations), what + ": iteration count lengths differ")
    for i in range(len(a.iterations)):  # small-loop(oracle: iteration counts)
        require(a.iterations[i] == b.iterations[i], what + ": iteration count " + String(i) + " differs")
    require(len(a.blocks) == len(b.blocks), what + ": block counts differ")
    for i in range(len(a.blocks)):  # small-loop(oracle: the fitted blocks)
        _same(a.blocks[i], b.blocks[i], what + " block " + String(i))


def _host_case(unit: Int) raises:
    # A pool word on the host image is refused in words.
    var x = _values(_ROWS * _D, 1)
    var y = _values(_ROWS * _Q, 2)
    var blocks = List[List[Float32]]()
    var sizes: List[Int] = [_D, _Q, _ROWS, _ROWS, _D, _Q]
    var a: List[Int] = [Int(x.unsafe_ptr()), Int(y.unsafe_ptr())]
    for b in range(6):  # small-loop(the six output blocks)
        blocks.append(List[Float32](length=sizes[b] * _NC, fill=Float32(0)))
        a.append(Int(blocks[b].unsafe_ptr()))
    var its = List[Int32](length=_NC, fill=0)
    var nits = List[Int](length=1, fill=0)
    a.append(Int(its.unsafe_ptr()))
    a.append(Int(nits.unsafe_ptr()))
    var p: List[Int] = [_ROWS, _D, _Q, _NC, 500, 0, 0, 0, 0, 0, -1]
    var refused = False
    try:
        _ = native_descriptor_call(x_decomp_host_v1, XD_PLS_FIT, a, p, [1e-06])
    except:
        refused = True
    require(refused, "the host image read a PLS pool word")
    # The host kit fits (its words -1: the host addresses).
    var kit = _host_kit()
    try:
        var xm = kit.borrow(Int(x.unsafe_ptr()), _ROWS, _D)
        var ym = kit.borrow(Int(y.unsafe_ptr()), _ROWS, _Q)
        var fitted = _fit(kit, xm, ym, 0)
        require(fitted.done == _NC, "the host kit fitted " + String(fitted.done) + " components")
    finally:
        kit.close()
    _ = x^
    _ = y^
    _ = blocks^
    _ = its^
    _ = nits^


def test_host_fixtures() raises:
    comptime if CPU_COLUMN_BUILD:
        def body(unit: Int) raises:
            _host_case(unit)

        check_cases("vdf_scope_pls_fold host", 1, body)


def _pls_fit_entry(k: Int) raises:
    """XD_PLS_FIT with the centred X / Y as pool ids == as host words."""
    var n = 12
    var x = _values(n * _D, 3 + k)
    var y = _values(n * _Q, 4 + k)
    var w = _kind(k)
    var sizes: List[Int] = [_D, _Q, n, n, _D, _Q]
    var results = List[List[Float32]]()
    var counts = List[List[Int32]]()
    var dones = List[Int]()
    var empty = List[Float64]()
    for form in range(2):  # small-loop(the two input forms)
        var ids = List[Int]()
        if form == 1:
            for m in range(2):  # small-loop(X then Y up into the pool)
                var words = n * (_D if m == 0 else _Q)
                var id = native_descriptor_call(x_decomp_device_v1, XD_DEV_ALLOC, List[Int](), [words], empty)
                var src = Int(x.unsafe_ptr()) if m == 0 else Int(y.unsafe_ptr())
                _ = native_descriptor_call(x_decomp_device_v1, XD_DEV_UPLOAD, [src], [id, words], empty)
                ids.append(id)
        var blocks = List[List[Float32]]()
        var a = List[Int]()
        a.append(0 if form == 1 else Int(x.unsafe_ptr()))
        a.append(0 if form == 1 else Int(y.unsafe_ptr()))
        for b in range(6):  # small-loop(the six output blocks)
            blocks.append(List[Float32](length=sizes[b] * _NC, fill=Float32(0)))
            a.append(Int(blocks[b].unsafe_ptr()))
        var its = List[Int32](length=_NC, fill=0)
        var nits = List[Int](length=1, fill=0)
        a.append(Int(its.unsafe_ptr()))
        a.append(Int(nits.unsafe_ptr()))
        var p: List[Int] = [n, _D, _Q, _NC, w.max_iter, 1 if w.mode_b else 0, Int(w.canonical), Int(w.canonical),
                            Int(w.svd_algo), ids[0] if form == 1 else -1, ids[1] if form == 1 else -1]
        var done = native_descriptor_call(x_decomp_device_v1, XD_PLS_FIT, a, p, [w.tol])
        dones.append(done)
        for b in range(6):  # small-loop(the six output blocks)
            results.append(blocks[b].copy())
        counts.append(its.copy())
        for m in range(len(ids)):  # small-loop(the two pool ids)
            _ = native_descriptor_call(x_decomp_device_v1, XD_DEV_FREE, List[Int](), [ids[m]], empty)
        _ = blocks^
        _ = nits^
    require(dones[0] == dones[1], "XD_PLS_FIT: fitted counts differ between pool ids and host words")
    for b in range(6):  # small-loop(oracle: the six blocks)
        _same(results[6 + b], results[b], "XD_PLS_FIT kind " + String(k) + " block " + String(b))
    for i in range(_NC):  # small-loop(oracle: iteration counts)
        require(counts[0][i] == counts[1][i], "XD_PLS_FIT: iteration count " + String(i) + " differs")
    _ = x^
    _ = y^


def _pls_fit_fold(k: Int) raises:
    """pls_fit_run over the taken fold == the gather on the device kit == the gather on the host kit."""
    var whole = _values(_ROWS * _D, 5 + k)
    var y_whole = _values(_ROWS * _Q, 6 + k)
    var idx = _fold()
    var n = len(idx)
    var x_host = _gather(whole, _D, idx)
    var y_host = _gather(y_whole, _Q, idx)
    var ids = _taken(whole, _D, idx)
    var dev = _device_kit()
    var gathered = _device_kit()
    var host = _host_kit()
    try:
        var xd = dev.borrow_device(ids[1], n, _D)
        var yd = dev.borrow(Int(y_host.unsafe_ptr()), n, _Q)
        var a = _fit(dev, xd, yd, k)
        var xg = gathered.borrow(Int(x_host.unsafe_ptr()), n, _D)
        var yg = gathered.borrow(Int(y_host.unsafe_ptr()), n, _Q)
        var b = _fit(gathered, xg, yg, k)
        var xh = host.borrow(Int(x_host.unsafe_ptr()), n, _D)
        var yh = host.borrow(Int(y_host.unsafe_ptr()), n, _Q)
        var c = _fit(host, xh, yh, k)
        _same_fit(a, b, "PLS fit kind " + String(k) + ", device fold vs device gather")
        _same_fit(b, c, "PLS fit kind " + String(k) + ", device vs host column")
    finally:
        dev.close()
        gathered.close()
        host.close()
        _free(ids[1])
        _free(ids[0])
    _ = whole^
    _ = y_whole^
    _ = x_host^
    _ = y_host^


struct _Operands(Movable):
    """One host fit's query operands (host words)."""
    var x_mean: List[Float32]
    var x_std: List[Float32]
    var coef: List[Float32]
    var y_mean: List[Float32]
    var rot: List[Float32]
    var components: Int

    def __init__(out self, mut kit: DKit, x: Int, ym: Int, k: Int) raises:
        var fit = pls_fit_run(kit, x, ym, _kind(k), False)
        self.x_mean = _words(kit, fit.x_mean)
        self.x_std = _words(kit, fit.x_std)
        self.coef = _words(kit, fit.coef)
        self.y_mean = _words(kit, fit.y_mean)
        self.rot = _words(kit, fit.x_rotations)
        self.components = fit.done


def _answers(mut kit: DKit, x: Int, ops: _Operands, mut out: List[List[Float32]]) raises:
    """predict then transform of kit matrix `x`, their words appended to `out`."""
    var xm = kit.borrow(Int(ops.x_mean.unsafe_ptr()), 1, _D)
    var xs = kit.borrow(Int(ops.x_std.unsafe_ptr()), 1, _D)
    var cf = kit.borrow(Int(ops.coef.unsafe_ptr()), _Q, _D)
    var ym = kit.borrow(Int(ops.y_mean.unsafe_ptr()), 1, _Q)
    var rt = kit.borrow(Int(ops.rot.unsafe_ptr()), _D, ops.components)
    var predicted = pls_predict_run(kit, x, xm, cf, ym)
    out.append(_words(kit, predicted))
    var projected = pls_project(kit, x, xm, xs, rt)
    out.append(_words(kit, projected))


def _pls_queries(k: Int) raises:
    """predict and transform over a taken fold == over the gather (device and host kits), with one host fit's
    operands."""
    var fit_x = _values(_ROWS * _D, 7 + k)
    var fit_y = _values(_ROWS * _Q, 8 + k)
    var whole = _values(_ROWS * _D, 9 + k)
    var idx = _fold()
    var n = len(idx)
    var x_host = _gather(whole, _D, idx)
    var ids = _taken(whole, _D, idx)
    var src = _host_kit()
    var dev = _device_kit()
    var gathered = _device_kit()
    var host = _host_kit()
    try:
        var fx = src.borrow(Int(fit_x.unsafe_ptr()), _ROWS, _D)
        var fy = src.borrow(Int(fit_y.unsafe_ptr()), _ROWS, _Q)
        var ops = _Operands(src, fx, fy, k)
        var answers = List[List[Float32]]()
        var xd = dev.borrow_device(ids[1], n, _D)
        _answers(dev, xd, ops, answers)
        var xg = gathered.borrow(Int(x_host.unsafe_ptr()), n, _D)
        _answers(gathered, xg, ops, answers)
        var xh = host.borrow(Int(x_host.unsafe_ptr()), n, _D)
        _answers(host, xh, ops, answers)
        _same(answers[0], answers[2], "PLS predict kind " + String(k) + ", device fold vs device gather")
        _same(answers[1], answers[3], "PLS transform kind " + String(k) + ", device fold vs device gather")
        _same(answers[2], answers[4], "PLS predict kind " + String(k) + ", device vs host column")
        _same(answers[3], answers[5], "PLS transform kind " + String(k) + ", device vs host column")
        _ = ops^
    finally:
        src.close()
        dev.close()
        gathered.close()
        host.close()
        _free(ids[1])
        _free(ids[0])
    _ = fit_x^
    _ = fit_y^
    _ = whole^
    _ = x_host^


def _rebuilt(mut kit: DKit, z: Int, loadings: List[Float32], std: List[Float32],
             mean: List[Float32]) raises -> List[Float32]:
    var l = kit.borrow(Int(loadings.unsafe_ptr()), _D, _NC)
    var s = kit.borrow(Int(std.unsafe_ptr()), 1, _D)
    var m = kit.borrow(Int(mean.unsafe_ptr()), 1, _D)
    var rebuilt = pls_rebuild(kit, z, l, s, m)
    return _words(kit, rebuilt)


def _pls_inverse() raises:
    """inverse_transform over taken rows of a scores matrix == over the gather (device and host kits)."""
    var z_whole = _values(_ROWS * _NC, 11)
    var loadings = _values(_D * _NC, 12)
    var std = _values(_D, 13)
    var mean = _values(_D, 14)
    var idx = _fold()
    var n = len(idx)
    var z_host = _gather(z_whole, _NC, idx)
    var ids = _taken(z_whole, _NC, idx)
    var dev = _device_kit()
    var gathered = _device_kit()
    var host = _host_kit()
    try:
        var zd = dev.borrow_device(ids[1], n, _NC)
        var a = _rebuilt(dev, zd, loadings, std, mean)
        var zg = gathered.borrow(Int(z_host.unsafe_ptr()), n, _NC)
        var b = _rebuilt(gathered, zg, loadings, std, mean)
        var zh = host.borrow(Int(z_host.unsafe_ptr()), n, _NC)
        var c = _rebuilt(host, zh, loadings, std, mean)
        _same(a, b, "PLS inverse_transform, device fold vs device gather")
        _same(b, c, "PLS inverse_transform, device vs host column")
    finally:
        dev.close()
        gathered.close()
        host.close()
        _free(ids[1])
        _free(ids[0])
    _ = z_whole^
    _ = loadings^
    _ = std^
    _ = mean^
    _ = z_host^


def _device_case(unit: Int) raises:
    if unit == 0:
        for k in range(_KINDS):  # small-loop(the three kinds)
            _pls_fit_entry(k)
    elif unit == 1:
        for k in range(_KINDS):  # small-loop(the three kinds)
            _pls_fit_fold(k)
    elif unit == 2:
        for k in range(_KINDS):  # small-loop(the three kinds)
            _pls_queries(k)
    else:
        _pls_inverse()


def test_device_fixtures() raises:
    comptime if not CPU_COLUMN_BUILD:
        # One device context and its pool: in order.
        check_cases_serial("vdf_scope_pls_fold device", 4, _device_case)


def main() raises:
    var suite = TestSuite()
    suite.test[test_host_fixtures]()
    suite.test[test_device_fixtures]()
    suite^.run()
