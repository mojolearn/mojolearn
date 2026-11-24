"""Public RFE validation coverage for the shared native finite-check entries."""
import numpy as np
import pytest
import mojolearn as ml


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["contiguous", "strided"])
def test_rfe_accepts_finite_target_storage(dtype, layout):
    x = np.arange(24, dtype=np.float32).reshape(8, 3) / 10
    y = np.arange(8, dtype=dtype)
    if layout == "strided":
        backing = np.zeros(16, dtype=dtype)
        backing[::2] = y
        y = backing[::2]
    model = ml.RFE(ml.Ridge(), n_features_to_select=2)
    assert model.fit(x, y) is model
    assert model.n_features_in_ == 3
    assert sum(model.get_support()) == 2


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("argument", ["X", "y"])
def test_rfe_preserves_nonfinite_input_refusal(dtype, argument):
    x = np.arange(24, dtype=dtype).reshape(8, 3) / 10
    y = np.arange(8, dtype=dtype)
    if argument == "X":
        x[3, 1] = np.inf
    else:
        y[3] = np.nan
    model = ml.RFE(ml.Ridge(), n_features_to_select=2)
    with pytest.raises(ValueError, match="NaN or infinity"):
        model.fit(x, y)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rfe_refuses_non_native_target_bytes_in_its_estimators_fit_words(dtype):
    # Commit 06d7af5776 (lane w3-fix-a, row 13) removed RFE's own x_prep target scan, which read byte-swapped
    # words and refused them in MaxAbsScaler's words: RFE's target is admitted once, by its member UNPACK, in
    # the words its estimator's fit door uses (scikit-learn's RFE validates y with ensure_all_finite=False and
    # leaves the target to the estimator's fit). Ridge's fit door refuses a non-native byte order before it
    # reads a value, so RFE over Ridge raises that refusal, exactly as Ridge.fit does on the same y.
    x = np.arange(24, dtype=np.float32).reshape(8, 3) / 10
    y = np.arange(8, dtype=dtype)
    y[3] = np.inf
    y = y.astype(y.dtype.newbyteorder(">"))
    with pytest.raises(TypeError) as own:
        ml.Ridge().fit(x, y)
    assert str(own.value) == "y must be in native byte order"
    with pytest.raises(TypeError) as through_rfe:
        ml.RFE(ml.Ridge(), n_features_to_select=2).fit(x, y)
    assert str(through_rfe.value) == str(own.value)
