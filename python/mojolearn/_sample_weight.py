"""The one `sample_weight` check every estimator's `fit` shares.

scikit-learn's `sklearn.utils.validation._check_sample_weight`, with its
wording (its estimator checks match on it): a scalar becomes a constant
vector; a list, tuple, pandas Series or buffer becomes a 1-D float vector of
exactly n_samples entries; negative, NaN and infinite entries are refused,
and so is an all-zero vector. The scans are the base binding's native
reductions (`Array.min`, `Array.max`, `all_finite`): no Python loop over
the weights.

The returned Array may BORROW the caller's buffer (a float32 C-contiguous
input is not copied). A caller that writes into the weights must pass
`copy=True`, so the caller's array is never changed.
"""

from ._buffer import all_finite, as_f32_c, as_f64_c, full

#: scikit-learn's own message for an all-zero weight vector; its
#: check_all_zero_sample_weights_error matches "weight" ... "zero".
ZERO_WEIGHTS_MSG = "Sample weights must contain at least one non-zero number."
NEGATIVE_WEIGHTS_MSG = "Negative values in data passed to `sample_weight`"


def check_sample_weight(sample_weight, n_samples, *, dtype="<f4", copy=False,
                        allow_all_zero=False, allow_negative=False):
    """`sample_weight` as a checked 1-D float Array of `n_samples` entries.

    Parameters
    ----------
    sample_weight : None, scalar, sequence or buffer
        None gives a vector of ones.
    n_samples : int
        The number of rows of X.
    dtype : {'<f4', '<f8'}
        The Array's dtype.
    copy : bool
        Always return fresh storage (for a caller that writes into it).
    allow_all_zero : bool
        Accept an all-zero vector.
    allow_negative : bool
        Accept negative entries.

    Raises ``ValueError`` with scikit-learn's wording for a wrong shape,
    non-finite or negative entries, and an all-zero vector.
    """
    n = int(n_samples)
    if sample_weight is None:
        return full((n,), 1.0, dtype)
    if isinstance(sample_weight, (int, float)) and not isinstance(sample_weight, bool):
        w = full((n,), float(sample_weight), dtype)
    else:
        if hasattr(sample_weight, "ndim") and hasattr(sample_weight, "shape") \
                and int(sample_weight.ndim) == 0:
            w = full((n,), float(sample_weight), dtype)
        else:
            conv = as_f32_c if dtype == "<f4" else as_f64_c
            try:
                w, copied = conv(sample_weight, ndim=None, name="sample_weight")
            except ValueError as exc:
                if "is empty" in str(exc) and n != 0:
                    raise ValueError(
                        f"sample_weight.shape == (0,), expected ({n},)!") from None
                raise
            if w.ndim != 1:
                raise ValueError("Sample weights must be 1D array or scalar")
            if w.shape[0] != n:
                raise ValueError(
                    f"sample_weight.shape == {tuple(w.shape)}, expected ({n},)!")
            if copy and not copied:
                w = w.copy()
    if n == 0:
        return w
    if not all_finite(w):
        raise ValueError("Input sample_weight contains NaN or infinity.")
    if not allow_negative and w.min() < 0:  # glue: native min of the weights
        raise ValueError(NEGATIVE_WEIGHTS_MSG)
    if not allow_all_zero and not (w.max() > 0 or w.min() < 0):  # glue: native reductions
        raise ValueError(ZERO_WEIGHTS_MSG)
    return w

