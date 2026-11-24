# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The `mx_targets` / `mx_codes` forms of a y_true encoded on the device (lane
vdf-xm-wire3): the metrics_exp doors' single-vector label consumers ask
bindings/metric_resident_labels_door.mojo `metric_resident_labels` first and
take these, which answer the host route unchanged when it is not live.

The label PAIR (lane vdf-xm-wire4): `mx_resident_pair` is `mx_pair` with each
side asked first (`MxDevPair`): a resident side is encoded on the device and
mapped through the door's label order there (MS_GATHER), a host side takes the
host encode; the union of the two class lists is the host route's `present`.
The pair consumers of bindings/metrics_exp_cls_door.mojo (`_sums`, `_kappa`,
`_likelihood_ratios`, `_report`) and bindings/metrics_exp_cluster_door.mojo
(`_contingency`) pin each side with `MxDevPair.pin`."""
from std.python import PythonObject
from bindings.metric_resident_labels_door import MetricResidentLabels, metric_resident_labels, metric_pin_codes
from bindings.metrics_exp_common_unpack import (
    MxLab, MxPair, mx_b, mx_targets, mx_codes, mx_map_eq, mx_raise_obj, mx_encode, mx_refuse_multilabel,
    mx_pair_encoded,
)
from bindings.metrics_program_door import MetricDoorProg


def mx_resident_targets(dev: MetricResidentLabels, y_true: PythonObject, caller: String,
                        context: PythonObject) raises -> MxLab:
    """`mx_targets`, or for a y_true encoded on the device (`dev.live()`,
    lane vdf-xm-wire3) its kind, classes and length with no host codes."""
    if dev.live():
        return MxLab("integer", dev.classes, PythonObject(None), dev.n)
    return mx_targets(y_true, caller, context)


def mx_resident_codes(mut dev: MetricResidentLabels, lab: MxLab, order: PythonObject,
                      context: PythonObject) raises -> Tuple[PythonObject, Int]:
    """`mx_codes(lab, order, True)` on either side: (host codes, -1) or
    (None, the device slot); a row outside `order` raises the host gather's
    KeyError with its label, found on the device (`row_class`)."""
    if not dev.live():
        return (mx_codes(lab, order, True, context), Int(-1))
    var got = dev.order(order, context)
    if got[1] >= 0:
        mx_raise_obj(mx_b().KeyError, dev.row_class(order, got[1], context))
    return (PythonObject(None), got[0])


def mx_resident_flags(mut dev: MetricResidentLabels, lab: MxLab, pos: PythonObject,
                      context: PythonObject) raises -> Tuple[PythonObject, Int]:
    """`mx_map_eq(lab, pos)` on either side: (host flags, -1) or (None, the
    device slot of the 0/1 flags)."""
    if not dev.live():
        return (mx_map_eq(lab, pos, context), Int(-1))
    return (PythonObject(None), dev.equal(pos, context))


def mx_resident_lab(dev: MetricResidentLabels, values: PythonObject, name: String,
                    context: PythonObject) raises -> MxLab:
    """`mx_encode(values, name)`, or for a vector encoded on the device
    (`dev.live()`) its kind, classes and length with no host codes (lane
    vdf-xm-wire4)."""
    if dev.live():
        return MxLab("integer", dev.classes, PythonObject(None), dev.n)
    return mx_encode(values, name, context)


struct MxDevPair(Movable):
    """`mx_pair`'s answer with each side's device encode (lane vdf-xm-wire4):
    the host `MxPair` (a live side's `codes` is None), the two
    `MetricResidentLabels` (not live: that side's host route), the two label
    vectors (each a live side's slot owner, `pin_resident`), and the door
    call's arguments with the weights' argument index (`pin_arg`, -1:
    unmarked). Kept until after the program run; `close` frees the slots, and
    the destructors free them on an error path."""
    var pair: MxPair
    var t: MetricResidentLabels
    var p: MetricResidentLabels
    var y_true: PythonObject
    var y_pred: PythonObject
    var args: PythonObject
    var w_arg: Int

    def __init__(out self, var pair: MxPair, var t: MetricResidentLabels, var p: MetricResidentLabels,
                 y_true: PythonObject, y_pred: PythonObject, args: PythonObject, w_arg: Int):
        self.pair = pair^
        self.t = t^
        self.p = p^
        self.y_true = y_true
        self.y_pred = y_pred
        self.args = args
        self.w_arg = w_arg

    def codes(mut self, side: Int, order: PythonObject, context: PythonObject) raises -> Tuple[PythonObject, Int]:
        """`mx_codes(side, order, False)` (side 0 y_true, 1 y_pred): (host
        codes, -1), or (None, the device slot of the codes mapped through
        `order`, -1 outside it)."""
        if side == 0:
            if self.t.live():
                return (PythonObject(None), self.t.order(order, context)[0])
            return (mx_codes(self.pair.true, order, False, context), Int(-1))
        if self.p.live():
            return (PythonObject(None), self.p.order(order, context)[0])
        return (mx_codes(self.pair.pred, order, False, context), Int(-1))

    def pin(mut self, mut prog: MetricDoorProg, side: Int, order: PythonObject, context: PythonObject) raises:
        """Pin one side's codes in `order` as the program's next buffer input:
        the device slot read in place (its label vector the owner), else the
        host codes as this call's temporary (never kept resident)."""
        var got = self.codes(side, order, context)
        metric_pin_codes(prog, got, self.y_true if side == 0 else self.y_pred)

    def pin_weights(mut self, mut prog: MetricDoorProg):
        """Pin the admitted weights (`pair.w`, not None) naming the door
        argument they came from (`pin_arg`), or unmarked when none is known."""
        if self.w_arg >= 0:
            prog.pin_arg(self.pair.w, self.args, self.w_arg)
        else:
            prog.pin(self.pair.w, True)

    def close(mut self) raises:
        """Free both sides' device slots after the program run."""
        self.t.close()
        self.p.close()


def mx_resident_pair_of(var t: MetricResidentLabels, var p: MetricResidentLabels, var true: MxLab, var pred: MxLab,
                        y_true: PythonObject, y_pred: PythonObject, sample_weight: PythonObject, caller: String,
                        args: PythonObject, w_arg: Int, context: PythonObject) raises -> MxDevPair:
    """`mx_pair_encoded` of two sides already encoded (each on the device or
    on the host), kept with their device state."""
    var pair = mx_pair_encoded(true^, pred^, sample_weight, caller, context)
    return MxDevPair(pair^, t^, p^, y_true, y_pred, args, w_arg)


def mx_resident_pair(y_true: PythonObject, y_pred: PythonObject, sample_weight: PythonObject, mode: PythonObject,
                     caller: String, args: PythonObject, w_arg: Int, context: PythonObject) raises -> MxDevPair:
    """`mx_pair` with each side encoded on the device when it is resident
    (`metric_resident_labels`); the multilabel refusal of a host side in
    `mx_pair`'s order (a live side is a 1-D vector: it cannot apply)."""
    var t = metric_resident_labels(y_true, mode, caller, context)
    if not t.live():
        mx_refuse_multilabel(y_true, "y_true", caller, context)
    var p = metric_resident_labels(y_pred, mode, caller, context)
    if not p.live():
        mx_refuse_multilabel(y_pred, "y_pred", caller, context)
    var true = mx_resident_lab(t, y_true, "y_true", context)
    var pred = mx_resident_lab(p, y_pred, "y_pred", context)
    return mx_resident_pair_of(t^, p^, true^, pred^, y_true, y_pred, sample_weight, caller, args, w_arg, context)
