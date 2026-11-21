# SPDX-License-Identifier: Apache-2.0
# The two `score` metrics of the SVM family (python/mojolearn/_svm_impl.py,
# svm.py): the x_metrics programs of `_r2_sums` and `accuracy_fraction`,
# rebuilt here stage for stage (the `_Prog` of bindings/metrics_reg_door.mojo,
# run by the shared runner). The family's shared reads are
# bindings/svc_common_unpack.mojo, its shared public arrays and decode
# bindings/svc_common_pack.mojo, the typed checks bindings/svc_common_run.mojo.
#
# Both metrics follow the estimator's device (lane label-steps, 2026-11-21):
# the caller hands them its label scope (bindings/labels_descriptor_boundary_door.mojo
# `labels_scope(model, context)`). A `device="cpu"` estimator's scope names the
# CPU x_metrics runner entries (`labels_metrics_entries`), which run the R^2
# sums program and the accuracy COUNT tail; any other context names none and
# runs the process image through the shared runner, as before.
from std.python import Python, PythonObject
from bindings.admitted_family_door import admit_family
from bindings.classical_arrays_door import classical_address
from bindings.classical_labels_door import classical_score_from_codes
from bindings.metrics_reg_run import _Reg, _RE_DIV
from bindings.arena_prog_run import MetricArenaOwner
from bindings.metrics_program_door import metric_plan_run
from bindings.arena_prog_native_door import metrics_run_plan_entries
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.metrics_keys_unpack import metrics_keys
from bindings.py_balanced import py_get, py_item, py_type
from core.metrics_program_plan import metric_term
from bindings.svc_common_run import svc_r2_check
from bindings.svc_common_unpack import svc_shape_of


# ---------------------------------------------------------------- score metrics


def svc_r2_sums(pred: PythonObject, y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`linear_model._r2_sums(pred, y)`: (SS_res, SS_tot) from the one
    x_metrics program `_r2_sums_of` ran: the y mean (Float32 column), then
    both sums of squares, binary64 out."""
    var t = admit_family(y, 1, "y", context).array
    var p = admit_family(pred, 1, "predictions", context).array
    var n = Int(py=t.size)
    var m = Int(py=p.size)
    svc_r2_check(Int(py=classical_address(t)), n, Int(py=classical_address(p)), m)
    var pins = List[PythonObject]()
    pins.append(t)
    pins.append(p)
    var r = _Reg(n, 1, False, "uniform_average", List[Float64]())
    var sq = metric_term("sq")
    r.program(0.0)
    var y_off = r.y_off
    var p_off = r.p_off
    var y_sums = r.sums(y_off)
    var y_mean = r.fin32(y_sums, _RE_DIV)
    var res_terms = r.term(sq, y_off, p_off, False)
    var res_sums = r.sums(res_terms)
    var num = r.fin(res_sums, 0)
    var tot_terms = r.term(sq, y_off, y_mean, True)
    var tot_sums = r.sums(tot_terms)
    var den = r.fin(tot_sums, 0)
    _ = r.plan.want(num, 2, -1, 1)
    _ = r.plan.want(den, 2, -1, 1)
    # The scope's CPU x_metrics entries, or None (the process image): one
    # dictionary read per score.
    var entries = py_get(context, "labels_metrics_entries")
    if entries is None:
        var arena = metric_plan_run(r.plan, pins, PythonObject(None), context)
        return _svc_r2_pack(arena, num, den, t, p)
    if py_type(entries) is py_type(PythonObject("")):
        raise Error(String(entries))
    var prog_context = py_item(context, metrics_keys()[].k_family_metrics_prog_context)
    classical_connect_numpy(prog_context)
    var arena = metrics_run_plan_entries(r.plan, pins, entries, prog_context)
    return _svc_r2_pack(arena, num, den, t, p)


def _svc_r2_pack(arena: MetricArenaOwner, num: Int, den: Int, t: PythonObject, p: PythonObject) raises -> PythonObject:
    var ss_res = arena.f64(num)
    var ss_tot = arena.f64(den)
    _ = t
    _ = p
    return Python.tuple(PythonObject(ss_res), PythonObject(ss_tot))


def svc_accuracy_codes(y_true: PythonObject, classes: PythonObject, codes: PythonObject, n_pred: Int,
                       context: PythonObject) raises -> Float64:
    """`svc_accuracy(y_true, classes[codes])` without the decode: the same
    shape refusal, then y encoded once against `classes` and compared with
    the predicted int64 codes in the x_metrics COUNT program (unweighted,
    the process mode). `classical_score_from_codes` keeps the decode route
    for classes that are neither all str nor all integers, so every refusal
    is the one the decoded prediction raised."""
    # N382 (lane ne3-g-svc-gp-neighbors). classical_score_from_codes is the
    # shared helper of lane ne1-h-labels (bindings/classical_labels_door.mojo).
    var shape = svc_shape_of(y_true, context)
    if len(shape) != 1 or Int(py=shape[0]) != n_pred:
        raise Error("mojolearn: y must be 1-D with one entry per row of X")
    return classical_score_from_codes(y_true, classes, codes, PythonObject(None), PythonObject(None), context)
