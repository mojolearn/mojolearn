"""FROZEN ORACLE (test tool): the former `mojolearn._sample_weight.check_sample_weight`.

Deleted from the package 2026-11-14 by lane py-admission: it had no product
caller; every door admits sample_weight natively
(bindings/admitted_input_unpack.mojo `admit_weights`, the same wording).
The legacy oracles below still read it.

The one `sample_weight` check every estimator's `fit` shared.

scikit-learn's `sklearn.utils.validation._check_sample_weight`, with its
wording (its estimator checks match on it): a scalar becomes a constant
vector; a list, tuple, pandas Series or buffer becomes a 1-D float vector of
exactly n_samples entries; negative, NaN and infinite entries are refused,
and so is an all-zero vector. The scans are ONE native call, the shared
argument-and-label entry `prepare_targets` (`_labels.prepare_targets`):
the finiteness scan and the min/max reductions it ran as three calls
before, in the same order, with no Python loop over the weights.

The returned Array may BORROW the caller's buffer (a float32 C-contiguous
input is not copied). A caller that writes into the weights must pass
`copy=True`, so the caller's array is never changed.
"""

from mojolearn.tests._oracle_support import full, require_numpy
from mojolearn.tests._typed_input import as_f32_c, as_f64_c


def _check_weights(w, *, allow_negative, allow_all_zero):
    """The weight refusals of the product's `_labels.prepare_targets(None, w, n, ...)`
    (bindings/labels_admission_unpack.mojo `labels_prepare_arrays`), in its order and wording, computed with
    NumPy: non-finite, then negative (unless `allow_negative`), then all-zero (unless `allow_all_zero`)."""
    np = require_numpy("the oracle sample_weight check")
    v = np.asarray(w)
    if not np.isfinite(v).all():
        raise ValueError("Input sample_weight contains NaN or infinity.")
    if not allow_negative and (v < 0).any():
        raise ValueError("Negative values in data passed to `sample_weight`")
    if not allow_all_zero and not (v != 0).any():
        raise ValueError("Sample weights must contain at least one non-zero number.")


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
        elif isinstance(sample_weight, bool):
            # The deleted `_buffer.as_f32_c` made a bool a 0-D array, refused just below as not 1-D; the
            # tests/_typed_input converters (7b3c18d435) refuse a bool with a TypeError instead.
            raise ValueError("Sample weights must be 1D array or scalar")
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
    _check_weights(w, allow_negative=allow_negative, allow_all_zero=allow_all_zero)
    return w

