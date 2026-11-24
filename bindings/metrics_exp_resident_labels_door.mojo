# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The `mx_targets` / `mx_codes` forms of a y_true encoded on the device (lane
vdf-xm-wire3): the metrics_exp doors' single-vector label consumers ask
bindings/metric_resident_labels_door.mojo `metric_resident_labels` first and
take these, which answer the host route unchanged when it is not live."""
from std.python import PythonObject
from bindings.metric_resident_labels_door import MetricResidentLabels
from bindings.metrics_exp_common_unpack import MxLab, mx_b, mx_targets, mx_codes, mx_map_eq, mx_raise_obj


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
