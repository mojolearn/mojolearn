# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Coordinate descent on the GPU: Lasso and ElasticNet. Reference: cuML's
`solver='cd'` arm (`cuml/cpp/src/solver/cd.cuh::cdFit`).

The implementation is `solver/` (DEVIATIONS 610-613 and 880); `solver/README.md`,
`solver/NOT_IMPLEMENTED.tsv` are the record. The
reference is cuML pinned at `v26.08.00` = `265b9da`, and every line number
cited in this file was read in that checkout.

These classes are not re-exported from `mojolearn/__init__.py` by this file;
whoever owns that file decides the public namespace.
"""

from . import _mojolearn_solver, _native_archive_bridge
from ._mode import ParamsMixin
from ._buffer import addr, addr_ro, as_f32_c, as_f32_colmajor, empty, zeros, _materialize
from ._sample_weight import check_sample_weight
from .linear_model import _r2_host, _shape_of

# cuML's `loss_funct` / DistanceType style codes this surface uses.
_SELECTION_SHUFFLE = {"cyclic": 0, "random": 1}

#: The saved-model format of ElasticNet and Lasso (lane/inference-linear-svm,
#: 2026-09-15): `mojolearn.host_model(path)` predicts from it on a CPU.
_CD_FORMAT = "mojolearn-cd-1"

# The compiled saved-model owner, connected once at import: `save` and
# `load` are one native call each (bindings/archive_schema_native.mojo
# writes the schema registered below; host_archive_native.mojo builds the
# loaded model).
_ARCHIVE, _ARCHIVE_CONTEXT = _native_archive_bridge.connect()


# THE IMPORT-TIME MODE GUARD THAT STOOD HERE IS DELETED. DEVIATION 1931,
# 2026-08-28, and it is deleted rather than corrected because both halves of
# it were wrong by the time it fired.
#
# It read: "`_mojolearn_solver` is not in `_backend._MODULES` yet, so under
# `identical` the FAST binary would be imported here under the identical
# label", and it ended "Delete it when `_backend._MODULES` lists this
# module." `_MODULES` HAS LISTED IT since DEVIATION 869 (2026-08-24), so the
# reason the message gave a reader was false, and its own instruction said to
# remove it.
#
# What it actually did after that was worse than nothing. `_backend.select()`
# hands out a `_MissingIdentical` stub for any binding whose identical binary
# was not built -- import succeeds, touching it raises BY NAME with the build
# command -- so one unbuilt binding degrades to one broken estimator. A stub
# has no `__file__`, so this guard's `os.path.basename(os.path.dirname(...))`
# check could never pass for it, and a missing solver binary became an
# ImportError at `from . import _mojolearn_solver`: the whole package failed
# to import, and with it tools/e1_traced_fit.py and the entire E2 matrix.
# That is how phases 3 and 4 went dark on every vendor while the bootstrap
# reported two unrelated-looking driver findings.
#
# The protection is not lost. `select()` is the thing that redirects these
# modules and it refuses to fall back to a FAST binary under the identical
# name -- that refusal is the mechanism, and it is the one this guard was
# written to duplicate before the tuple was fixed.


# Maintainer notes (moved from the docstring, 2026-10-12):
# Reference: cuml.linear_model.ElasticNet(solver='cd') and
# cuml/cpp/src/solver/cd.cuh::cdFit; the Mojo implementation is
# solver/impl/cd.mojo and the host surface solver/estimator.mojo.
# THE OBJECTIVE: cd.cuh:84-86 documents an unscaled objective, and the code
# (cd.cuh:201-202, v26.08.00 checkout) scales both penalties by the row count:
#     l2_alpha = (1 - l1_ratio) * alpha * n_rows
#     l1_alpha =      l1_ratio  * alpha * n_rows
# so what is minimized is n_rows times scikit-learn's documented objective
# (sklearn/linear_model/_coordinate_descent.py:499-501; scikit-learn's solver
# applies the same n_samples factor, :781-782). A positive scale does not move
# the minimizer, so alpha and l1_ratio mean the same in sklearn, cuML and here;
# cuML's docstring is off by the factor n.
# Differences from scikit-learn, all matching cuML on purpose:
#   stopping rule  after each epoch, coefMax < tol OR diffMax / coefMax < tol
#                  (cd.cuh:271), a coefficient-change test only; no duality gap,
#                  so dual_gap_ is absent, not zero
#   tol default    1e-3 (cuML's); cuML's CPU-to-GPU converter multiplies a
#                  scikit-learn tol by 10 (elastic_net.py:154)
#   zero-column    `squared > 1e-5 ? r / squared : 0` (cd.cuh:68), an absolute
#   guard          threshold on a sum of squares; on the identity card as
#                  `cd.squared`. scikit-learn skips only an exactly-zero column
# Refusals and their sources:
#   alpha / l1_ratio  non-finite refused by name (DEVIATION 613, cd.mojo): cuML's
#                  guard `alpha < 0` is false for NaN, and a NaN alpha zeroes
#                  every coefficient silently
#   fit_intercept  centers on the device through the implemented preProcessData /
#                  postProcessData (cuML's kernels; on the identity card as
#                  `cd.mu_input`, `cd.mu_labels`), unlike LinearRegression/Ridge
#   selection      'random' refused (cd.mojo): cuML seeds std::mt19937(0) and calls
#                  std::shuffle, whose algorithm the C++ standard does not
#                  specify, so it cannot be gated bitwise; an exact
#                  implementation is DEVIATION 611, reserved and not spent
#   solver 'qn'    refused: cuML's 'auto' picks 'qn' for sparse input
#                  (elastic_net.py:243-244); 'qn' under l1 is OWL-QN on that
#                  surface (glm/NOT_IMPLEMENTED.tsv)
#   sample_weight  implemented (lane sk-kernels): solver/impl/cd_weighted.mojo,
#                  scikit-learn's rescale to sum n, weighted centering, sqrt rows
#   positive / warm_start / precompute  cuML raises UnsupportedOnGPU too
#                  (elastic_net.py:143, :146, :149); cdFit reads coef as its
#                  starting point and solver/estimator.mojo zeroes it every fit
#   copy_X=False   refused: the design is copied to the device and the caller's
#                  array is never written (DEVIATION 880; cuML's cdFit centers
#                  and un-centers input in place and does not restore the bits)
#   random_state   accepted and inert (only 'random' reads it); refusing it broke
#                  nested Lasso(random_state=...) in voting/stacking members
#   n_rows <= 1 / n_cols <= 0  refused by cdFit itself (cd.cuh:145, :144), in fit
#                  and in predict
#   sparse X       cdFit is dense only (elastic_net.py:265-269)
# THE DESIGN MATRIX IS COPIED AT MOST ONCE (DEVIATION 2367): cdFit wants cuML's F
# order and `_buffer.as_f32_colmajor` converts dtype and layout in one pass; a
# float32 C-order design crosses as is and the device transposes it
# (lane/gap-nv-classical2). `input_copied_` and `fortran_copied_` report the same
# flag.
# n_iter_: the stopping test branches on reduction bits; under identical it is a
# function of the inputs alone and is the card's `cd.n_iter` stage (bit-identical
# on Apple M4, NVIDIA H100 and AMD MI325X at leg 11).
# IDENTITY TRACE: with MOJOLEARN_IDENTITY_TRACE=<path> set, a fit writes the same
# `cd.*` identity card solver/cd_main.mojo writes (solver/estimator.mojo hands the
# implemented entry a live trace).
class ElasticNet(ParamsMixin):
    """Linear regression with a combined l1 and l2 penalty, fitted by
    coordinate descent.

    Minimizes scikit-learn's elastic-net objective::

        (1 / (2 * n_samples)) * ||y - X w||^2
        + alpha * l1_ratio * ||w||_1
        + 0.5 * alpha * (1 - l1_ratio) * ||w||^2

    so ``alpha`` and ``l1_ratio`` mean the same as in scikit-learn's and
    cuML's ``ElasticNet``. The solver is cuML's coordinate descent
    (``solver='cd'``), and it differs from scikit-learn in ways that can
    change results:

    - It stops when the largest coefficient, or the largest coefficient
      change relative to it, falls below ``tol`` after an epoch. There is
      no duality-gap check, so there is no ``dual_gap_`` attribute.
    - ``tol`` defaults to 1e-3 (cuML's); scikit-learn's is 1e-4. Set it
      explicitly for a like-for-like comparison.
    - A column whose sum of squares is at most 1e-5 gets a zero
      coefficient. This threshold is absolute, so very small-valued
      features can be zeroed where the same data in larger units would
      not be. scikit-learn zeroes only an all-zero column.

    ``X`` must be dense and is read as float32; the caller's array is
    never modified. ``y`` must be one-dimensional. At least two rows are
    required, in ``fit`` and in ``predict``. With ``fit_intercept=True``
    the data are centered on the GPU. ``ElasticNet`` has no
    ``numeric_mode`` parameter.

    Parameters
    ----------
    alpha : float, default 1.0
        Overall penalty strength, >= 0. A negative value raises
        ``ValueError``; a non-finite value is refused by ``fit``.
    l1_ratio : float, default 0.5
        The l1 share of the penalty, in [0, 1] (``ValueError``
        otherwise); 1.0 is the lasso.
    fit_intercept : bool, default True
        Whether to fit an (unpenalized) intercept.
    max_iter : int, default 1000
        Maximum number of coordinate-descent epochs.
    tol : float, default 1e-3
        Stopping tolerance (see above).
    solver : {'auto', 'cd'}, default 'auto'
        Both mean coordinate descent. ``'qn'`` raises
        ``NotImplementedError``; any other value raises ``ValueError``.
    selection : {'cyclic'}, default 'cyclic'
        Coordinates are updated in order. ``'random'`` raises
        ``NotImplementedError``; any other value raises ``ValueError``.
    precompute : False
        Only False is supported; anything else raises
        ``NotImplementedError``.
    copy_X : True
        Only True is supported; False raises ``NotImplementedError``.
    warm_start : False
        Only False is supported (every fit starts from zero); True raises
        ``NotImplementedError``.
    positive : False
        Only False is supported; True raises ``NotImplementedError``.
    random_state : None, int or NumPy random state, default None
        Accepted for compatibility and unused, as in scikit-learn with
        ``selection='cyclic'``.

    Attributes
    ----------
    coef_ : array of shape (n_features,), float32
        The fitted coefficients.
    intercept_ : float
        The fitted intercept; exactly 0.0 when ``fit_intercept`` is False.
    n_iter_ : int
        The number of epochs run. In ``identical`` mode it is the same on
        Apple, NVIDIA and AMD GPUs; in other modes it can differ between GPUs.
    n_features_in_ : int
        The number of features seen in ``fit``.
    input_copied_ : bool
        Whether ``fit`` had to copy ``X``. A float32 array in C or Fortran
        order is used without a copy.
    fortran_copied_ : bool
        The same flag as ``input_copied_``, kept for compatibility.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = X @ np.arange(8, dtype=np.float32) + 1.0
    >>> model = mojolearn.ElasticNet(alpha=0.01, l1_ratio=0.7).fit(X, y)
    >>> pred = model.predict(X)
    """

    #: scikit-learn's estimator kind: `cross_val_score` stratifies a
    #: classifier's default folds, as scikit-learn's does.
    _estimator_type = "regressor"

    def __init__(self, alpha=1.0, *, l1_ratio=0.5, fit_intercept=True,
                 max_iter=1000, tol=1e-3, solver="auto", selection="cyclic",
                 precompute=False, copy_X=True, warm_start=False,
                 positive=False, random_state=None):
        # Stored verbatim (scikit-learn's convention); `_validate_params`
        # checks them when `fit` runs.
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.selection = selection
        self.precompute = precompute
        self.copy_X = copy_X
        self.warm_start = warm_start
        self.positive = positive
        self.random_state = random_state


    # sample_weight (lane sk-kernels): checked here with scikit-learn's
    # wording and handed to the weighted coordinate descent
    # (solver/impl/cd_weighted.mojo: weights rescaled to sum to n_samples,
    # weighted centering, sqrt-weight row scaling, all in Mojo).
    def fit(self, X, y, sample_weight=None):
        """Fit the model by coordinate descent.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Dense training data, read as float32. Sparse input raises
            ``NotImplementedError``.
        y : array-like of shape (n_samples,)
            Target values. A 2-D ``y`` or a length mismatch raises
            ``ValueError``.
        sample_weight : float or array-like of shape (n_samples,), default None
            Per-sample weights, non-negative and not all zero. As in
            scikit-learn, they are rescaled to sum to ``n_samples``, so the
            data term is the weighted mean of the squared residuals.

        Returns
        -------
        self : ElasticNet
            The fitted estimator.
        """
        return _CD_CALL(self, "cd_fit", (X, y, sample_weight), _CD_CONTEXT)

    def predict(self, X):
        """Predict with the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples, read as float32.

        Returns
        -------
        y_pred : array of shape (n_samples,), float32
            ``X @ coef_ + intercept_``.

        Raises
        ------
        ValueError
            If the model is not fitted, or ``X`` has a different number of
            features than the training data.
        """
        return _CD_CALL(self, "cd_predict", (X,), _CD_CONTEXT)

    # The file: `coef_` as fitted, the intercept <f8, `hyper` <f8 [alpha,
    # l1_ratio, tol], `meta` <i8 [n_features_in_, fit_intercept, max_iter,
    # n_iter_] and the numeric mode (lane/inference-linear-svm, 2026-09-15).
    def save(self, path):
        """Save the fitted model to ``path`` as an ``.npz`` file.

        The file stores ``coef_``, the intercept, ``alpha``, ``l1_ratio``,
        ``tol``, ``max_iter``, ``fit_intercept``, ``n_iter_`` and the
        numeric mode. ``mojolearn.host_model(path)`` can predict from the
        file on a machine without a GPU.

        Parameters
        ----------
        path : str or path-like
            Destination file.

        Raises
        ------
        RuntimeError
            If the model is not fitted.
        """
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

    @classmethod
    def load(cls, path):
        """Load a model written by ``save``.

        The loaded model predicts with the saved coefficients and
        intercept and carries the saved hyperparameters and ``n_iter_``.

        Parameters
        ----------
        path : str or path-like
            A file written by ``save`` of the same class.

        Returns
        -------
        model : ElasticNet or Lasso

        Raises
        ------
        ValueError
            If the file was saved by a different estimator or its fields do
            not match.
        """
        return _ARCHIVE(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)

    # `linear_model._r2_host` (formerly a sequential float64 host reduction,
    # DEVIATION 2365).
    def score(self, X, y):
        """Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True target values.

        Returns
        -------
        score : float
            ``1 - SS_res / SS_tot``, computed in float64. A constant ``y``
            scores 0.0.
        """
        return _CD_CALL(self, "cd_score", (X, y), _CD_CONTEXT)


# Maintainer note: cuML spells Lasso as ElasticNet(l1_ratio=1.0)
# (cuml/linear_model/lasso.py:129-140, "Lasso is just a special case of
# ElasticNet"), and so does scikit-learn; the l2 term is absent and
# l1_alpha = alpha * n_rows. See ElasticNet's notes.
class Lasso(ElasticNet):
    """Linear regression with an l1 penalty, fitted by coordinate descent.

    ``Lasso(alpha)`` is ``ElasticNet(alpha, l1_ratio=1.0)``: it minimizes
    ``(1 / (2 * n_samples)) * ||y - X w||^2 + alpha * ||w||_1``, the same
    objective as scikit-learn's ``Lasso``. Everything in ``ElasticNet``'s
    description applies, including the stopping rule, the absolute
    zero-column threshold and the refused options.

    ``tol`` defaults to 1e-3 (cuML's); scikit-learn's is 1e-4.

    Parameters
    ----------
    alpha : float, default 1.0
        Penalty strength, finite and >= 0.
    fit_intercept : bool, default True
        Whether to fit an (unpenalized) intercept.
    max_iter : int, default 1000
        Maximum number of coordinate-descent epochs.
    tol : float, default 1e-3
        Stopping tolerance.
    solver, selection, precompute, copy_X, warm_start, positive, random_state
        As on ``ElasticNet``.

    Attributes
    ----------
    coef_ : array of shape (n_features,), float32
        The fitted coefficients; the l1 penalty sets some exactly to zero.
    intercept_ : float
        The fitted intercept; exactly 0.0 when ``fit_intercept`` is False.
    n_iter_ : int
        The number of epochs run.
    n_features_in_ : int
        The number of features seen in ``fit``.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((1000, 8), dtype=np.float32)
    >>> y = X @ np.arange(8, dtype=np.float32) + 1.0
    >>> model = mojolearn.Lasso(alpha=0.01).fit(X, y)
    >>> pred = model.predict(X)
    """

    def __init__(self, alpha=1.0, *, fit_intercept=True, max_iter=1000,
                 tol=1e-3, solver="auto", selection="cyclic",
                 precompute=False, copy_X=True, warm_start=False,
                 positive=False, random_state=None):
        super().__init__(
            alpha=alpha, l1_ratio=1.0, fit_intercept=fit_intercept,
            max_iter=max_iter, tol=tol, solver=solver, selection=selection,
            precompute=precompute, copy_X=copy_X, warm_start=warm_start,
            positive=positive, random_state=random_state,
        )


# The file: `coef_` as fitted, the intercept <f8, `hyper` <f8 [alpha,
# l1_ratio, tol] and `meta` <i8 [n_features_in_, fit_intercept, max_iter,
# n_iter_]. Lasso registers the same schema so its file names Lasso and
# its load applies the l1_ratio refusal.
_CD_SCHEMA = (
    _native_archive_bridge.fitted("coef_"),
    *_native_archive_bridge.head(_CD_FORMAT),
    ("array", "coef", "coef_"),
    ("f8", "intercept", (("float", "intercept_"),)),
    ("f8", "hyper", (("float", "alpha"), ("float", "l1_ratio"), ("float", "tol"))),
    ("i8", "meta", (("int", "n_features_in_"), ("bool", "fit_intercept"),
                    ("int", "max_iter"), ("int", "n_iter_"))),
)
_native_archive_bridge.register(ElasticNet, _CD_FORMAT, _CD_SCHEMA)
_native_archive_bridge.register(Lasso, _CD_FORMAT, _CD_SCHEMA)
