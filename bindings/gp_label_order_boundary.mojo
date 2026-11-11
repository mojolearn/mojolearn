# SPDX-License-Identifier: Apache-2.0
"""Exact builtin integer class ordering over the shared native key index."""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.classical_labels import _TextCodeStorage, _TextLabels
from bindings.labels_score_boundary import score_keys_unpack, score_keys_index
from bindings.p3_labels_metrics_native import p3_text_classes
from core.labels_score_keys import score_key_codes


def gp_integer_classes[S: _TextCodeStorage](labels: List[PythonObject],
                                           context: PythonObject) raises -> _TextLabels[S]:
    # The caller has completed the original label admission and admits only
    # exact builtin int/bool here. Their arbitrary-precision mathematical order
    # is exactly the shared native key order. NumPy numeric promotion is not
    # interchangeable with this rule and stays with its own admitted boundary.
    var captured = score_keys_unpack(labels, False, context)
    var index = score_keys_index(captured.keys)
    # This PACK selects the original first object at each distinct index;
    # normalized bool/int key storage never replaces observable class objects.
    var classes = p3_text_classes(labels, index.sorted_rows)
    if len(classes) > 2147483648:
        raise Error("TypeError: mojolearn: cannot build a <i4 Array from these values: signed integer is greater than maximum")
    var codes = S(len(labels), context)
    var destination = codes.address()
    var py = Python()
    try:
        with GILReleased(py):
            score_key_codes(index, captured.keys, destination)
    finally:
        _ = captured
        _ = index
        _ = codes
        _ = labels
    return _TextLabels[S](classes, codes^, len(labels))
