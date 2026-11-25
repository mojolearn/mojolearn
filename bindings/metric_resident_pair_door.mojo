# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CALLER'S LABEL PAIR encoded on the device inside `resident()` (lane
gd-resident-functions, 2026-11-25; moved here from
bindings/mcls_classify_door.mojo `_resident_labels`, lane vdf-xm-wire).

The classification pair metrics (confusion_matrix, precision / recall / F)
and the union-code cluster metrics (mutual information, Fowlkes-Mallows,
homogeneity, completeness, v-measure and the three at once,
bindings/mcls_cluster_door.mojo) take the CALLER'S two label vectors, never
the int32 copies the door's admission makes: the scope is keyed by buffer
identity, and a copy has a new address on every call (the union-code
metrics passed their copies, so the GPU verify of 2026-11-25 showed
`stats {'x_prep': (4|8, 0)}`: an upload per call, never a hit). The raw
4- or 8-byte labels go to the device once and MS_ENCODE_PAIR encodes them
there; the union codes of the raw labels are the codes of their int32
narrowing (the admission refuses a label outside int32 first), so the
answer is the host route's, bit for bit.
"""
from std.python import PythonObject
from bindings.metrics_slot_door import McSlots, McResidentPair, mc_resident_pair, mc_resident_none
from bindings.mcls_common_unpack import mc_integer_view
from bindings.mcls_common_run import MC_INT32_MAX
from bindings.classical_arrays_door import classical_address
from bindings.metric_resident_source_door import metric_exact_buffer


def mc_resident_caller_pair(binding: PythonObject, y_true: PythonObject, y_pred: PythonObject, want_classes: Bool,
                            mut state: McSlots, context: PythonObject) raises -> McResidentPair:
    """The caller's label pair encoded on the device (bindings/metrics_slot_door.mojo
    `mc_resident_pair`): both are exact Arrays or NumPy ndarrays, contiguous
    1-D signed integer vectors of one width (4 or 8 bytes) and length, at
    least one resident in the active `resident()` scope. Their union codes
    stay on the device; k (and, with `want_classes`, the sorted union) come
    back once. Anything else (outside a scope, lists, strings, bool /
    unsigned / mixed widths, a strided view, a length mismatch the host route
    words) is not live, and the caller takes its host route, unchanged."""
    if not state.resident.active():
        return mc_resident_none()
    # exact Arrays or NumPy ndarrays (the gate took Arrays only, so NumPy
    # labels, the common input, never went to the device)
    if not metric_exact_buffer(y_true) or not metric_exact_buffer(y_pred):
        return mc_resident_none()
    var vt = mc_integer_view(y_true, context)
    var vp = mc_integer_view(y_pred, context)
    if not vt or not vp:
        return mc_resident_none()
    var it = vt.value().item
    var n = vt.value().count
    if (not vt.value().signed or not vp.value().signed or it != vp.value().item or vt.value().stride != it
            or vp.value().stride != it or n != vp.value().count or n > MC_INT32_MAX):
        return mc_resident_none()
    var ta = Int(py=classical_address(vt.value().view))
    var pa = Int(py=classical_address(vp.value().view))
    var pair = mc_resident_pair(binding, y_true, ta, y_pred, pa, n, it, want_classes, state)
    _ = vt^  # the views' buffers are read by address inside the step above
    _ = vp^
    return pair^
