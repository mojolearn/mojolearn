# SPDX-License-Identifier: Apache-2.0
"""Python conversion/ignored-field seams of the installed legacy label entries."""
import pytest


@pytest.fixture(params=("host", "device"))
def legacy_labels(request):
    from mojolearn import _backend

    if request.param == "host":
        return _backend.load_host_module("_mojolearn_core_host")
    from mojolearn._native_labels_bridge import _base_binding

    module = _base_binding()
    if module is False:
        pytest.skip("CPU-only install has no device label entry")
    if isinstance(module, str):
        raise RuntimeError(module)
    return module


@pytest.mark.parametrize(
    "entry,args,expected",
    [
        ("unique_inverse", (None, 0, 0, None, None), "unique_inverse: n must be positive, got 0"),
        ("unique_inverse", (None, 1, 9, None, None), "unique_inverse: kind must be 0 (float64) or 1 (int64)"),
        ("unique_inverse", (0, 1, 0, None, None), "unique_inverse: null buffer address"),
        ("unique_inverse", (1, 1, 0, 0, None), "unique_inverse: null buffer address"),
        ("prepare_targets", (None, None, -1, None, None, None, None, None), "prepare_targets: n must be non-negative, got -1"),
        ("prepare_targets", (1, None, 0, None, None, None, None, None), "prepare_targets: labels need n >= 1"),
        ("prepare_targets", (1, None, 1, 0, None, None, None, None), "prepare_targets: null label output address"),
        ("prepare_targets", (1, None, 1, 1, 0, None, None, None), "prepare_targets: null label output address"),
        ("prepare_targets", (1, 99, 1, 1, 1, None, None, None), "prepare_targets: label dtype code must be 0..5, got 99"),
        ("gather_i64", (None, 0, None, 0, None), "gather_i64: n must be non-negative and the table non-empty"),
        ("gather_f64", (None, 1, None, -1, None), "gather_f64: n must be non-negative and the table non-empty"),
        ("gather_i64", (0, 1, None, 1, None), "gather_i64: null buffer address"),
        ("gather_f64", (1, 1, 0, 1, None), "gather_f64: null buffer address"),
        ("argmax_rows_f32", (None, -1, 1, None), "argmax_rows_f32: n_rows must be non-negative and n_cols positive"),
        ("argmax_rows_f64", (None, 0, 0, None), "argmax_rows_f64: n_rows must be non-negative and n_cols positive"),
        ("argmax_rows_f32", (0, 1, 1, None), "buffer address"),
        ("class_ratio_f64", (None, 1, 1, 9, None), "class_ratio_f64: mode is 0 (balanced) or 1 (prior)"),
        ("class_ratio_f64", (None, 1, 0, 1, None), "class_ratio_f64: the prior of zero rows"),
        ("class_ratio_f64", (0, 1, 1, 0, None), "null buffer address"),
    ],
)
def test_earlier_native_refusal_precedes_ignored_python_conversion(legacy_labels, entry, args, expected):
    # None is deliberately not convertible to an address. It must remain
    # untouched after the earlier native refusal, rather than mask that error.
    with pytest.raises(Exception) as captured:
        getattr(legacy_labels, entry)(*args)
    assert expected in str(captured.value)


@pytest.mark.parametrize(
    "entry,args",
    [
        ("prepare_targets", (0, None, 2, None, None, 0, None, None)),
        ("prepare_targets", (0, None, 0, None, None, 1, None, None)),
        ("gather_i64", (None, 1, None, 0, None)),
        ("gather_f64", (None, 1, None, 0, None)),
        ("argmax_rows_f32", (None, 0, 1, None)),
        ("argmax_rows_f64", (None, 0, 1, None)),
        ("class_ratio_f64", (None, 0, 1, 9, None)),
    ],
)
def test_empty_legacy_operation_skips_unused_python_fields(legacy_labels, entry, args):
    assert getattr(legacy_labels, entry)(*args) == 0


def test_label_content_refusal_precedes_weight_metadata_conversion(legacy_labels):
    import numpy as np

    labels = np.array([np.nan], dtype=np.float64)
    classes = np.empty(1, dtype=np.float64)
    codes = np.empty(1, dtype=np.int32)
    with pytest.raises(Exception) as captured:
        legacy_labels.prepare_targets(
            labels.ctypes.data, 1, 1, classes.ctypes.data, codes.ctypes.data,
            None, None, None,
        )
    assert "y contains a NaN label; NaN is not a class" in str(captured.value)
