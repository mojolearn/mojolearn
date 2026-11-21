# SPDX-License-Identifier: Apache-2.0
"""Typed part of the public parallel output boundary (RUN): which results a
driver widens to float64, and the widening itself over two admitted blocks.
`bindings/parallel_public_result_door.mojo` unpacks the result's storage and
packs the widened block; no Python object reaches this file."""
from core.admitted_cast_elements import admitted_cast_elements
from bindings.classical_labels_run import _native_code


def parallel_result_widens(kind: String, method: String) -> Bool:
    """True where the public driver answers float64: a regressor's `predict`
    and an outlier detector's `decision_function` / `score_samples`. `kind`
    is "" for a driver that names no estimator kind."""
    if kind == "regressor":
        return method == "predict"
    if kind == "outlier_detector":
        return method == "decision_function" or method == "score_samples"
    return False


def parallel_result_widen(src: Int, dst: Int, count: Int) raises:
    """float32 block at `src` to the float64 block at `dst`, `count` elements
    in storage order (`core/admitted_cast_elements.mojo`: element tasks across
    the host's cores). Exact for every float32 value."""
    if count == 0:
        return
    var refused = admitted_cast_elements(src, _native_code("<f4"), dst, _native_code("<f8"), count)
    if refused != 0:
        raise Error("TypeError: mojolearn: an element does not fit <f8")
