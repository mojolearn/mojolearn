# SPDX-License-Identifier: Apache-2.0
"""Independent cross-validation folds assigned to explicit native workers."""
from . import model_selection as _selection

__all__ = ['cross_val_score']


def cross_val_score(estimator, X, y, *, devices, cv=None, scoring=None,
                    groups=None, error_score='raise', device=None):
    """Fit independent native estimator clones in explicit worker processes.

    In `identical` mode, mojolearn returns the same bits on Apple,
    NVIDIA and AMD GPUs.

    Parameters
    ----------
    estimator : native mojolearn estimator
        The constructor configuration to clone for each fold.
    X, y : dense native-compatible buffers
        Features and target values with matching row counts.
    devices : list or tuple of int
        Distinct CUDA/HIP device indices, or ``(0,)`` on Metal. On a
        CPU-only install each index selects an independent CPU process.
    cv : int, native splitter, or list of index pairs, optional
        Fold specification; every fold is validated before workers start.
    scoring : str or native scorer, optional
        A scorer name or a scorer built from a native mojolearn metric.
        None selects the estimator's native score.
    groups : native-compatible buffer, optional
        Group labels for a group-aware splitter.
    error_score : 'raise' or float, default='raise'
        'raise' raises a failed fit or score after worker cleanup. A number
        is the score of a fold whose fit or score failed, with
        scikit-learn's warnings; when every fit fails a ValueError is
        raised.
    device : {None, 'gpu', 'cpu'}, default None
        Where the computation runs. None and 'gpu' run on the GPUs in
        ``devices``; 'cpu' raises ValueError on a GPU installation, since this
        driver has no CPU form. Other values raise ValueError. An estimator
        constructed with a different ``device`` raises ValueError.

    Returns
    -------
    Array
        Float64 scores in fold order. At most one fold runs per worker;
        each fit must fit within its worker's memory. Metal uses one child
        process and makes no multi-device isolation claim.

    Examples
    --------
    >>> from mojolearn import LinearRegression
    >>> from mojolearn.parallel_model_selection import cross_val_score
    >>> scores = cross_val_score(LinearRegression(), X, y, devices=(0,), cv=3)
    """
    return _selection._NATIVE(None, 'family_msel_parallel_score',
                              (estimator, X, y, devices, cv, scoring, groups, error_score, device),
                              _selection._NATIVE_CONTEXT)


def _capture(estimator, X, y, *, devices, cv=None, scoring=None,
             groups=None, error_score='raise', device=None):
    # Testing entry: the same native fold path returns completed model and
    # numerical witnesses; no controller/scorer callback executes in a fold.
    return _selection._NATIVE(None, 'family_msel_parallel_capture',
                              (estimator, X, y, devices, cv, scoring, groups, error_score, device),
                              _selection._NATIVE_CONTEXT)
