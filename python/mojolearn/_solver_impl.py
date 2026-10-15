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

from . import _mojolearn_solver, _serialize
from ._array import Array
from ._mode import ParamsMixin
from ._buffer import addr, addr_ro, as_f32_c, as_f32_colmajor, empty, zeros, _materialize
from ._sample_weight import check_sample_weight
from .linear_model import _check_saved_by, _r2_host, _restore_mode, _saved_mode, _shape_of

# cuML's `loss_funct` / DistanceType style codes this surface uses.
_SELECTION_SHUFFLE = {"cyclic": 0, "random": 1}

#: The saved-model format of ElasticNet and Lasso (lane/inference-linear-svm,
#: 2026-09-15): `mojolearn.host_model(path)` predicts from it on a CPU.
_CD_FORMAT = "mojolearn-cd-1"


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

    def _validate_params(self):
        """The constructor checks, run at the start of `fit`."""
        alpha, l1_ratio, solver, selection = self.alpha, self.l1_ratio, self.solver, self.selection
        precompute, copy_X, warm_start, positive = self.precompute, self.copy_X, self.warm_start, self.positive
        random_state = self.random_state
        # cuML's own guards first, in their order and with their messages
        # (`elastic_net.py:233-240`), so a script that catches theirs
        # catches these.
        if alpha < 0.0:
            raise ValueError(f"Expected alpha >= 0, got {alpha}")
        if selection not in _SELECTION_SHUFFLE:
            raise ValueError(f"selection {selection!r} is not supported")
        if l1_ratio < 0.0 or l1_ratio > 1.0:
            raise ValueError(f"Expected 0.0 <= l1_ratio <= 1.0, got {l1_ratio}")
        if solver not in ("auto", "cd", "qn"):
            raise ValueError(f"solver={solver!r} is not supported")
        # Then the refusals that are this implementation's, each by name.
        if solver == "qn":
            raise NotImplementedError(
                "mojolearn ElasticNet: solver='qn' is not implemented. cuML's "
                "solver='auto' picks 'qn' only for SPARSE input "
                "(elastic_net.py:243-244), and 'qn' under an l1 penalty is "
                "OWL-QN (min_owlqn, qn_solvers.cuh), which glm/NOT_IMPLEMENTED.tsv "
                "lists as not implemented. solver='cd' (cuML's 'auto' for dense "
                "input) is the implemented arm; see solver/NOT_IMPLEMENTED.tsv"
            )
        if selection == "random":
            raise NotImplementedError(
                "mojolearn ElasticNet: selection='random' is REFUSED BY "
                "NAME. cuML seeds std::mt19937(0) and shuffles the "
                "coordinate order with std::shuffle each epoch "
                "(cd.cuh:198-199, :222), and the C++ standard does not specify "
                "std::shuffle's algorithm, so their permutation is a "
                "function of the toolchain rather than of the seed and "
                "cannot be gated bitwise or certified across vendors. An "
                "exact implementation is DEVIATION 611, reserved and not spent; see "
                "solver/NOT_IMPLEMENTED.tsv"
            )
        if positive:
            raise NotImplementedError(
                "mojolearn ElasticNet: positive=True is not supported "
                "(cuML raises UnsupportedOnGPU for it too, "
                "elastic_net.py:143)"
            )
        if warm_start:
            raise NotImplementedError(
                "mojolearn ElasticNet: warm_start=True is not supported "
                "(cuML raises UnsupportedOnGPU for it too, "
                "elastic_net.py:146). cdFit reads coef as its starting "
                "point and solver/estimator.mojo zeroes it on every fit"
            )
        if precompute is not False:
            raise NotImplementedError(
                "mojolearn ElasticNet: precompute is not supported (cuML "
                "raises UnsupportedOnGPU for it too, elastic_net.py:149)"
            )
        if not copy_X:
            raise NotImplementedError(
                "mojolearn ElasticNet: copy_X=False is refused. The design "
                "crosses to the device as a copy and the caller's array is "
                "never written (DEVIATION 880); there is no in-place arm to "
                "select"
            )
        # random_state is accepted and stored, scikit-learn's semantics:
        # it seeds the coordinate draw of selection='random' only and is
        # inert under 'cyclic' (sklearn ElasticNet: "Used when selection ==
        # 'random'"). 'random' is refused by name above (DEVIATION 611), so
        # no fit reads it and no output bit depends on it. Refusing it broke
        # every nested Lasso(random_state=...) (the board's voting-reg and
        # stacking-reg members), so it is now accepted.
        if random_state is not None and (
                isinstance(random_state, bool)
                or not (isinstance(random_state, int)
                        or hasattr(random_state, "randint")
                        or hasattr(random_state, "integers"))):
            raise ValueError(
                f"random_state={random_state!r}: None, an int or a "
                "numpy random state"
            )

    def _as_fortran(self, X, name):
        """A float32 column-major Array over X, and whether that cost the
        one copy it may cost (DEVIATION 2367).

        `cdFit` reads the design in cuML's `F` order (element (i, j) at
        `j * n_rows + i`). Returned alongside the flag so the caller can
        keep the Array alive across the Mojo call, which `_buffer.py`'s
        borrow contract requires.
        """
        f, copied = as_f32_colmajor(X, name=name)
        # THE ONE THING A SHAPE CHECK CANNOT CATCH LATER. `cd_fit` takes a
        # bare address plus n_rows and n_cols, and a C-order buffer of the
        # same size is a VALID transposed design: the fit would run and
        # return plausible coefficients for the wrong matrix. Two flags of
        # reading here is the only place that can be caught.
        if not f.flags["F_CONTIGUOUS"] or f.dtype != "<f4":
            raise AssertionError(
                "mojolearn: internal -- the design handed to cd_fit must be "
                f"float32 and Fortran-ordered, got {f.dtype} "
                f"F_CONTIGUOUS={f.flags['F_CONTIGUOUS']}"
            )
        return f, copied

    def _solver(self):
        """The binding `fit` and `predict` call; the host subclass of
        `_classical_host.py` answers the CPU binding instead."""
        return _mojolearn_solver

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
        # The CPU inference boundary: ElasticNet does not inherit
        # NumericModeMixin's guard, so on a CPU-only install the fit refuses
        # outside mojolearn._cpu_reference.reference_training().
        from ._cpu_reference import require_training
        require_training(self)
        self._validate_params()
        if hasattr(X, "toarray") or hasattr(X, "tocsr"):
            raise NotImplementedError(
                "mojolearn ElasticNet: sparse X is refused. cdFit is dense "
                "only; cuML routes sparse input to solver='qn' and raises "
                "for solver='cd' (elastic_net.py:265-269), and 'qn' is not "
                "implemented"
            )
        # lane/gap-nv-classical2: a float32 C-order design crosses AS IT IS
        # and the device transposes it (`cd_fit_host`'s row_major arm), so
        # no host transpose sits inside the fit. Every other input takes the
        # one F-order copy as before.
        a, a_copied = _materialize(X, "X")
        row_major = bool(a.ndim == 2 and a.dtype == "<f4" and a.size
                         and a.order == "C" and not a._both_orders())
        if row_major:
            work_x, copied = a, a_copied
        else:
            work_x, copied = self._as_fortran(a, "X")
            copied = copied or a_copied
        self.input_copied_ = copied
        self.fortran_copied_ = copied
        n_rows, n_cols = work_x.shape
        if n_rows < 2:
            # cdFit needs two rows (its refusal is a bare Exception from the
            # binding); scikit-learn's check_array wording, raised first
            raise ValueError(
                f"Found array with {n_rows} sample(s) (shape=({n_rows}, {n_cols})) "
                "while a minimum of 2 is required.")
        if len(_shape_of(y)) != 1:
            raise ValueError(
                "mojolearn ElasticNet currently requires one target")
        target, _ = as_f32_c(y, ndim=1, name="y")
        if target.shape[0] != n_rows:
            raise ValueError("mojolearn ElasticNet X and y lengths differ")

        sw = None
        if sample_weight is not None:
            sw = check_sample_weight(sample_weight, n_rows)

        self.coef_ = zeros((n_cols,), "<f4")
        info = zeros((1,), "<f4")
        params = [
            n_rows, n_cols, 1 if self.fit_intercept else 0,
            int(self.max_iter), float(self.alpha), float(self.l1_ratio),
            float(self.tol), _SELECTION_SHUFFLE[self.selection],
            0 if sw is None else 1, 1 if row_major else 0,
        ]
        if sw is not None:
            params.append(addr_ro(sw, name="sample_weight"))
        n_iter = self._solver().cd_fit(
            addr_ro(work_x, name="X"), addr_ro(target, name="y"),
            addr(self.coef_, name="coef_"), addr(info, name="info"),
            # ORDER MATCHES bindings/_mojolearn_solver.mojo::cd_fit_binding.
            # n_rows, n_cols, fit_intercept, max_iter, alpha, l1_ratio, tol,
            # shuffle, has_sample_weight, row_major[, sample_weight address]
            params,
        )
        self.intercept_ = float(info[0])
        self.n_iter_ = int(n_iter)
        self.n_features_in_ = n_cols
        # `work_x` (and `sw`) are the objects the addresses above belong to;
        # they live in this frame for the whole call, the borrow contract.
        return self

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
        if not hasattr(self, "coef_"):
            raise ValueError("mojolearn ElasticNet: call fit before predict")
        work_x, _ = self._as_fortran(X, "X")
        if work_x.shape[1] != self.n_features_in_:
            raise ValueError(
                "mojolearn ElasticNet feature count differs from fit")
        out = empty((work_x.shape[0],), "<f4")
        self._solver().cd_predict(
            addr_ro(work_x, name="X"), addr_ro(self.coef_, name="coef_"),
            addr(out, name="predictions"),
            # ORDER MATCHES bindings/_mojolearn_solver.mojo::cd_predict_binding.
            # n_rows, n_cols, intercept
            [work_x.shape[0], work_x.shape[1], float(self.intercept_)],
        )
        return out

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
        if not hasattr(self, "coef_"):
            raise RuntimeError("this estimator is not fitted yet")
        arrays = {
            "format": _CD_FORMAT,
            "estimator": type(self).__name__,
            "numeric_mode": _saved_mode(self),
            "coef": self.coef_,
            "intercept": Array.from_list([float(self.intercept_)], "<f8"),
            "hyper": Array.from_list(
                [float(self.alpha), float(self.l1_ratio), float(self.tol)], "<f8"),
            "meta": Array.from_list(
                [int(self.n_features_in_), 1 if self.fit_intercept else 0,
                 int(self.max_iter), int(self.n_iter_)], "<i8"),
        }
        return _serialize.write_npz(path, arrays)

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
        arrays = _serialize.read_npz(path, _CD_FORMAT)
        _check_saved_by(arrays, path, cls)
        meta = _serialize.exact(arrays, "meta", "<i8")
        if meta.size != 4:
            raise ValueError(f"mojolearn: {path!r} meta holds {meta.size} fields, 4 are needed")
        hyper = _serialize.exact(arrays, "hyper", "<f8")
        if hyper.size != 3:
            raise ValueError(f"mojolearn: {path!r} hyper holds {hyper.size} fields, 3 are needed")
        kwargs = dict(alpha=float(hyper[0]), fit_intercept=bool(int(meta[1])),
                      max_iter=int(meta[2]), tol=float(hyper[2]))
        if not issubclass(cls, Lasso):
            kwargs["l1_ratio"] = float(hyper[1])
        elif float(hyper[1]) != 1.0:
            raise ValueError(f"mojolearn: {path!r} carries l1_ratio {float(hyper[1])}, a Lasso is 1.0")
        obj = cls(**kwargs)
        _restore_mode(obj, arrays)
        obj.n_features_in_ = int(meta[0])
        obj.coef_ = _serialize.exact(arrays, "coef", "<f4")
        if obj.coef_.ndim != 1 or obj.coef_.size != obj.n_features_in_:
            raise ValueError(f"mojolearn: {path!r} coef does not match n_features_in_")
        intercept = _serialize.exact(arrays, "intercept", "<f8")
        if intercept.size != 1:
            raise ValueError(f"mojolearn: {path!r} intercept must hold one value")
        obj.intercept_ = float(intercept[0])
        obj.n_iter_ = int(meta[3])
        return obj

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
        return _r2_host(self.predict(X), y)


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
