# SPDX-License-Identifier: Apache-2.0
# The clustering metrics of python/mojolearn/_expansion_metrics.py, one
# native call each: contingency_matrix, pair_confusion_matrix,
# normalized_mutual_info_score, adjusted_mutual_info_score,
# calinski_harabasz_score and davies_bouldin_score.
#
# This owner is the Python `_clusterings` / `_contingency` and
# `_cluster_inputs` / `_cluster_score`: the same refusals, then the same
# x_metrics PROGRAM (arena layout, stage records, parameter words, order),
# run by the shared program runner, and the same binary64 epilogue
# (`_generalized_average`, the AMI guards) in the same order. The codes of
# each clustering into its own sorted classes are the encoder's codes
# themselves (the Python `_codes(a, ca)` gathered the identity table). Same
# kernels: the bits do not move.
from bindings.operation_names_mexp import (
    MEXP_OPN_FAMILY_MEXP_CLUSTER_AMI,
    MEXP_OPN_FAMILY_MEXP_CLUSTER_CALINSKI_HARABASZ,
    MEXP_OPN_FAMILY_MEXP_CLUSTER_CONTINGENCY,
    MEXP_OPN_FAMILY_MEXP_CLUSTER_DAVIES_BOULDIN,
    MEXP_OPN_FAMILY_MEXP_CLUSTER_NMI,
    MEXP_OPN_FAMILY_MEXP_CLUSTER_PAIR_CONFUSION,
)
from bindings.metrics_exp_common_run import nmi_finish, ami_finish, contingency_dtype_code, contingency_stages
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family
from bindings.py_balanced import py_call
from bindings.family_common_unpack import FAMILY_F32P, FAMILY_I64P
from bindings.family_common_pack import family_array
from bindings.classical_arrays_door import classical_address
from core.host_finite_scan import host_all_finite_f32
from core.metrics_exp_impl import mexp_cast_i64
from bindings.metrics_exp_common_run import MX_NONE
from bindings.metrics_program_pack import metric_arena_array
from bindings.metrics_program_door import (
    MetricDoorProg, metric_door_run, MetricRowsSlot, metric_door_rows_slot, metric_slot_first_nonfinite,
    metric_slot_release,
)
from bindings.arena_prog_run import MetricArenaOwner
from bindings.metrics_exp_common_unpack import (
    MxLab, mx_b, mx_len, mx_truthy, mx_is_str, mx_float, mx_refuse_multilabel, mx_encode,
)

comptime _CL_CENT = 0
comptime _CL_CH_ROW = 1
comptime _CL_CH_FIN = 2
comptime _CL_DB_DIST = 3
comptime _CL_DB_FIN = 4


struct _Clusterings(Movable):
    var a: MxLab
    var b: MxLab

    def __init__(out self, var a: MxLab, var b: MxLab):
        self.a = a^
        self.b = b^


def _clusterings(labels_true: PythonObject, labels_pred: PythonObject, caller: String,
                 context: PythonObject) raises -> _Clusterings:
    mx_refuse_multilabel(labels_true, "labels_true", caller, context)
    mx_refuse_multilabel(labels_pred, "labels_pred", caller, context)
    var a = mx_encode(labels_true, "labels_true", context)
    var b = mx_encode(labels_pred, "labels_pred", context)
    if a.n != b.n:
        raise Error("mojolearn " + caller + ": labels_true and labels_pred must have the same length")
    return _Clusterings(a^, b^)


struct _Cont(Movable):
    """`_contingency`'s program offsets and arena."""
    var arena: MetricArenaOwner
    var ka: Int
    var kb: Int
    var C: Int
    var F: Int
    var R: Int
    var K: Int
    var P: Int
    var E: Int
    var MI: Int
    var EMI: Int

    def __init__(out self, var arena: MetricArenaOwner, ka: Int, kb: Int, C: Int, F: Int, R: Int, K: Int, P: Int, E: Int,
                 MI: Int, EMI: Int):
        self.arena = arena^
        self.ka = ka
        self.kb = kb
        self.C = C
        self.F = F
        self.R = R
        self.K = K
        self.P = P
        self.E = E
        self.MI = MI
        self.EMI = EMI


def _contingency(c: _Clusterings, mode: PythonObject, has_eps: Bool, eps: Float64, mi: Bool, emi: Bool,
                 context: PythonObject) raises -> _Cont:
    var ka = mx_len(c.a.classes)
    var kb = mx_len(c.b.classes)
    var prog = MetricDoorProg()
    # the typed builder (bindings/metrics_exp_common_run.mojo
    # `contingency_stages`, lane mf-score5): its inputs A, B pinned in that order
    var r = contingency_stages(prog.plan, c.a.n, ka, kb, has_eps, eps, mi, emi)
    prog.pins.append(c.a.codes)
    prog.pins.append(c.b.codes)
    var arena = metric_door_run(prog, mode, context)
    return _Cont(arena^, ka, kb, r.C, r.F, r.R, r.K, r.P, r.E, r.MI, r.EMI)


def _contingency_matrix(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (labels_true, labels_pred, eps, sparse, dtype, mode)."""
    var b = mx_b()
    var eps = args[2]
    var sparse = mx_truthy(args[3])
    if eps is not None and sparse:
        raise Error("Cannot set 'eps' when sparse=True")
    if sparse:
        raise Error("NotImplementedError: mojolearn contingency_matrix: sparse=True is NOT IMPLEMENTED; this "
                    + "package has no sparse matrix type (metrics/NOT_IMPLEMENTED.tsv)")
    var c = _clusterings(args[0], args[1], "contingency_matrix", context)
    var has_eps = eps is not None
    var r = _contingency(c, args[5], has_eps, mx_float(eps, context) if has_eps else 0.0, False, False, context)
    var shape = Python.tuple(PythonObject(r.ka), PythonObject(r.kb))
    var cells = r.ka * r.kb
    if has_eps:
        return metric_arena_array(r.arena, r.F, 2 * cells, shape, "<f8", context)
    var counts = metric_arena_array(r.arena, r.C, 2 * cells, shape, "<i8", context)
    # the dtype's name: a string, or a type's __name__, or its str()
    var dtype = args[4]
    var name: String
    if mx_is_str(dtype):
        name = String(py=dtype)
    else:
        name = String(py=py_call(b.getattr, dtype, PythonObject("__name__"), py_call(b.str, dtype)))
    var code = contingency_dtype_code(name)
    if code < 0:
        return counts
    var typestr = String("<i4") if code == 0 else (String("<f8") if code == 1 else String("<f4"))
    var out = family_array(shape, typestr, context)
    mexp_cast_i64(FAMILY_I64P(unsafe_from_address=Int(py=classical_address(counts))), cells,
                  Int(py=classical_address(out)), code)
    _ = counts  # keep the Python object alive through the native call above
    _ = out  # keep the Python object alive through the native call above
    _ = counts^
    return out


def _pair_confusion(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (labels_true, labels_pred, mode)."""
    var c = _clusterings(args[0], args[1], "pair_confusion_matrix", context)
    var r = _contingency(c, args[2], False, 0.0, False, False, context)
    return metric_arena_array(r.arena, r.P, 8, Python.tuple(PythonObject(2), PythonObject(2)), "<i8", context)


def _average_method(method: PythonObject) raises -> Int:
    """`_generalized_average`'s method check: 0 min, 1 geometric, 2 arithmetic, 3 max."""
    if mx_is_str(method):
        var m = String(py=method)
        if m == "min":
            return 0
        if m == "geometric":
            return 1
        if m == "arithmetic":
            return 2
        if m == "max":
            return 3
    raise Error("'average_method' must be 'min', 'geometric', 'arithmetic', or 'max'")


def _nmi(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (labels_true, labels_pred, average_method, mode)."""
    var method = _average_method(args[2])
    var c = _clusterings(args[0], args[1], "normalized_mutual_info_score", context)
    var ka = mx_len(c.a.classes)
    var kb = mx_len(c.b.classes)
    if (ka == kb and kb == 1) or (ka == kb and kb == 0):
        return PythonObject(1.0)
    var r = _contingency(c, args[3], False, 0.0, True, False, context)
    return PythonObject(nmi_finish(r.arena.f64(r.MI), r.arena.f64(r.E), r.arena.f64(r.E + 2), method))


def _ami(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """args = (labels_true, labels_pred, average_method, mode)."""
    var method = _average_method(args[2])
    var c = _clusterings(args[0], args[1], "adjusted_mutual_info_score", context)
    var ka = mx_len(c.a.classes)
    var kb = mx_len(c.b.classes)
    if (ka == kb and kb == 1) or (ka == kb and kb == 0):
        return PythonObject(1.0)
    if ka == 1 or kb == 1:
        return PythonObject(0.0)
    var r = _contingency(c, args[3], False, 0.0, True, True, context)
    return PythonObject(ami_finish(r.arena.f64(r.MI), r.arena.f64(r.EMI), r.arena.f64(r.E), r.arena.f64(r.E + 2),
                                   method))


def _cluster_score(args: PythonObject, caller: String, db: Bool, context: PythonObject) raises -> PythonObject:
    """calinski_harabasz_score (db False) or davies_bouldin_score (db True);
    args = (X, labels, mode).

    A DeviceRows X on a GPU image (lane vdf-xm-transport, 2026-11-24) is
    gathered on the device (`metric_door_rows_slot`), its finiteness scanned
    there (`metric_slot_first_nonfinite`, the same refusal words) and read by
    the program from the slot: its rows never come to the host. Every other
    X is admitted on the host as before."""
    var owned = List[Int]()
    var rx = metric_door_rows_slot(args[0], args[2], context, owned)
    var result = 0.0
    try:
        result = _cluster_score_run(args, caller, db, rx, context)
    except error:
        var cleanup = metric_slot_release(rx, owned)
        if cleanup.byte_length() > 0:
            raise Error(String(error) + "; cleanup also failed: " + cleanup)
        raise error
    var cleanup = metric_slot_release(rx, owned)
    if cleanup.byte_length() > 0:
        raise Error("RuntimeError: x_metrics slot cleanup failed: " + cleanup)
    return PythonObject(result)


def _cluster_score_run(args: PythonObject, caller: String, db: Bool, rx: MetricRowsSlot,
                       context: PythonObject) raises -> Float64:
    # `_cluster_inputs`
    var Xa = PythonObject(None)
    var n = rx.rows
    var d = rx.cols
    if rx.slot >= 0:
        if metric_slot_first_nonfinite(rx) >= 0:
            raise Error("mojolearn " + caller + ": X contains NaN or infinity")
    else:
        Xa = admit_family(args[0], 2, "X", context).array
        n = Int(py=Xa.shape[0])
        d = Int(py=Xa.shape[1])
        if not host_all_finite_f32(FAMILY_F32P(unsafe_from_address=Int(py=classical_address(Xa))), n * d):
            raise Error("mojolearn " + caller + ": X contains NaN or infinity")
        _ = Xa  # keep the Python object alive through the native calls above
    var lab = mx_encode(args[1], "labels", context)
    if lab.n != n:
        raise Error("mojolearn " + caller + ": X and labels have different numbers of rows")
    var k = mx_len(lab.classes)
    if not (1 < k and k < n):
        raise Error("Number of labels is " + String(k) + ". Valid values are 2 to n_samples - 1 (inclusive)")
    var prog = MetricDoorProg()
    var X0 = -1
    if rx.slot >= 0:
        X0 = prog.put_resident(args[0], n * d, rx.slot)
    else:
        X0 = prog.put(Xa)
    var L = prog.put_i32(lab.codes)
    var g = prog.group(L, n, k, X0, d, MX_NONE, d)
    var off = g[0]
    var sums = g[1]
    var C32 = prog.scratch(k * d)
    var CB = prog.scratch(2 * k * d)
    prog.stage("cl_epi", k * d, [_CL_CENT, off, sums, k, d, C32, CB])
    var dist = prog.scratch(n)
    prog.stage("row_centroid_dist", n, [X0, d, L, C32, dist, 1 if db else 0])
    var per = prog.group(L, n, k, dist, 1, MX_NONE, 1)[1]
    var res = prog.want(prog.alloc(2), 2)
    if db:
        var D = prog.scratch(2 * k * k)
        prog.stage("cl_epi", k * k, [_CL_DB_DIST, CB, k, d, D])
        prog.stage("cl_epi", 1, [_CL_DB_FIN, off, per, D, k, res])
    else:
        var zero = prog.scratch(n)
        prog.stage("pair_key", n, [0, 0, zero, 1, 2])
        var gsum = prog.group(zero, n, 1, X0, d, MX_NONE, d)[1]
        var inner = prog.scratch(2 * k)
        prog.stage("cl_epi", k, [_CL_CH_ROW, CB, gsum, n, d, k, inner])
        prog.stage("cl_epi", 1, [_CL_CH_FIN, off, inner, per, n, k, res])
    var arena = metric_door_run(prog, args[2], context)
    return arena.f64(res)


def metrics_exp_cluster_operation(operation: Int, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """The clustering metrics (argument orders: see python/mojolearn/_expansion_metrics.py)."""
    if operation == MEXP_OPN_FAMILY_MEXP_CLUSTER_CONTINGENCY:
        return _contingency_matrix(args, context)
    if operation == MEXP_OPN_FAMILY_MEXP_CLUSTER_PAIR_CONFUSION:
        return _pair_confusion(args, context)
    if operation == MEXP_OPN_FAMILY_MEXP_CLUSTER_NMI:
        return _nmi(args, context)
    if operation == MEXP_OPN_FAMILY_MEXP_CLUSTER_AMI:
        return _ami(args, context)
    if operation == MEXP_OPN_FAMILY_MEXP_CLUSTER_CALINSKI_HARABASZ:
        return _cluster_score(args, "calinski_harabasz_score", False, context)
    if operation == MEXP_OPN_FAMILY_MEXP_CLUSTER_DAVIES_BOULDIN:
        return _cluster_score(args, "davies_bouldin_score", True, context)
    raise Error("RuntimeError: unknown clustering metric " + String(operation))
