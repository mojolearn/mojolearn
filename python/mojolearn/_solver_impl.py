# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Coordinate descent on the GPU: Lasso and ElasticNet, mirroring cuML's
`solver='cd'` arm (`cuml/cpp/src/solver/cd.cuh::cdFit`).

The port is `solver/` (DEVIATIONS 610-613 and 880); `solver/README.md`,
`solver/DERIVATION_MAP.tsv` and `solver/NOT_IMPLEMENTED.tsv` are the record. The
upstream is cuML pinned at `v26.08.00` = `265b9da`, and every line number
cited in this file was read in that checkout.

These classes are not re-exported from `mojolearn/__init__.py` by this file;
whoever owns that file decides the public namespace.
"""

import numpy as np

from . import _mojolearn_solver
from ._arrays import _addr, _addr_ro, as_f32_c

# cuML's `loss_funct` / DistanceType style codes this surface uses.
_SELECTION_SHUFFLE = {"cyclic": 0, "random": 1}


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



class ElasticNet:
    """Elastic-net regression by coordinate descent on the GPU.

    Mirrors `cuml.linear_model.ElasticNet(solver='cd')` on top of
    `cuml/cpp/src/solver/cd.cuh::cdFit`; the Mojo port is
    `solver/impl/solver/cd.mojo` and the host surface is
    `solver/estimator.mojo`.

    THE OBJECTIVE, AND WHOSE `alpha` THIS IS
    ----------------------------------------
    `cd.cuh:84-86` DOCUMENTS an unscaled objective, and the CODE
    (`cd.cuh:201-202`, read in the `v26.08.00` checkout) scales both
    penalties by the row count::

        l2_alpha = (1 - l1_ratio) * alpha * n_rows
        l1_alpha =      l1_ratio  * alpha * n_rows

    so what is minimized is `n_rows` times ::

        (1 / (2 n)) ||y - X w||^2
        + alpha * l1_ratio * ||w||_1
        + 0.5 * alpha * (1 - l1_ratio) * ||w||^2

    which is exactly scikit-learn's documented ElasticNet objective
    (`sklearn/linear_model/_coordinate_descent.py:499-501`), and
    scikit-learn's own solver applies the same `n_samples` factor to both
    penalties before its coordinate loop (`:781-782`). Scaling an objective
    by a positive constant does not move its minimizer, so **`alpha` and
    `l1_ratio` mean the same thing in `sklearn.linear_model.ElasticNet`, in
    `cuml.linear_model.ElasticNet` and here.** Their docstring is off by the
    factor `n`; the code is what was ported.

    WHERE THE THREE STILL DIFFER, all of it carried from cuML on purpose:

        stopping rule   after each epoch, `coefMax < tol` OR
                        `diffMax / coefMax < tol` (`cd.cuh:271`) -- a
                        coefficient-CHANGE test only. scikit-learn runs the
                        same change test and THEN requires a duality gap
                        below `tol * ||y||^2`; the gap is what it certifies
                        and there is none here. `dual_gap_` is therefore
                        ABSENT, not zero.
        tol default     1e-3 (cuML's). scikit-learn's is 1e-4, and cuML's
                        own CPU-to-GPU converter multiplies a scikit-learn
                        `tol` by 10 (`elastic_net.py:154`) to line the two
                        up. A like-for-like comparison must set them.
        zero-column     `squared > 1e-5 ? r / squared : 0` (`cd.cuh:68`), an
        guard           ABSOLUTE threshold on a sum of squares that scales
                        with `n` and with the square of the data, so a
                        column of tiny-unit data can be zeroed where the
                        same design in larger units is not. Carried as
                        theirs (COPY, DO NOT IMPROVE) and recorded on the
                        identity card as `cd.squared`. scikit-learn instead
                        skips only a column whose norm is exactly 0.

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY -- one line per parameter,
    because a parameter accepted and ignored is a wrong answer waiting for a
    caller. Every "refused by name" below names the file that raises it:

        alpha           honored   >= 0. Non-finite is REFUSED BY NAME
                                  (DEVIATION 613, cd.mojo): cuML's guard is
                                  `alpha < 0`, which is false for a NaN, and
                                  a NaN alpha makes every coefficient 0
                                  silently.
        l1_ratio        honored   0 <= l1_ratio <= 1; NaN refused by name
                                  (DEVIATION 613)
        fit_intercept   honored   True centers X and y ON THE DEVICE through
                                  the ported `preProcessData` /
                                  `postProcessData`. NOTE this is unlike
                                  `mojolearn.LinearRegression` and
                                  `mojolearn.Ridge`, whose centering is a
                                  host reimplementation in numpy; here the
                                  centering is cuML's own kernels and is on
                                  the identity card (`cd.mu_input`,
                                  `cd.mu_labels`).
        max_iter        honored   `cdFit`'s `epochs`; default 1000, theirs
        tol             honored   default 1e-3, theirs (see above)
        selection       honored   'cyclic' only. 'random' is REFUSED BY NAME
                                  (cd.mojo): cuML seeds `std::mt19937(0)`
                                  and calls `std::shuffle`, whose algorithm
                                  the C++ standard does not specify, so
                                  their permutation is a function of the
                                  toolchain and not of the seed and cannot
                                  be gated bitwise. An exact port is
                                  DEVIATION 611, reserved and NOT spent.
        solver          honored   'auto' and 'cd'. 'qn' is REFUSED BY NAME:
                                  cuML's 'auto' picks 'qn' for SPARSE input
                                  (`elastic_net.py:243-244`), and 'qn' under
                                  an l1 penalty is OWL-QN, which is not
                                  ported (`glm/NOT_IMPLEMENTED.tsv`). Sparse input
                                  is refused with it.
        sample_weight   refused   the weighted `preProcessData`, the
                                  sqrt-weight scaling of X and y and its
                                  undo are not ported (cd.mojo raises)
        positive        refused   cuML raises `UnsupportedOnGPU` too
                                  (`elastic_net.py:143`)
        warm_start      refused   cuML raises `UnsupportedOnGPU` too
                                  (`elastic_net.py:146`). `cdFit` READS
                                  `coef` as its starting point, and
                                  `solver/estimator.mojo` zeroes it on every
                                  fit precisely so that a fit is a function
                                  of its inputs.
        precompute      refused   cuML raises `UnsupportedOnGPU` too
                                  (`elastic_net.py:149`)
        copy_X          honored   True only, and it is what happens: the
                                  design is copied to the device and the
                                  caller's array is never written
                                  (DEVIATION 880 -- cuML's `cdFit` centers
                                  and un-centers `input` IN PLACE and does
                                  not restore the bits). False is refused
                                  because this surface cannot offer it.
        random_state    refused   it only ever selected the `std::shuffle`
                                  stream, and `selection='random'` is
                                  refused
        y 2-D           refused   one target only, at this boundary
        n_rows <= 1     refused by name by `cdFit` itself (`cd.cuh:145`),
                        in `fit` AND in `predict` -- so `predict(X[:1])`
                        raises rather than returning one number
        n_cols <= 0     refused by name (`cd.cuh:144`)
        sparse X        refused   `cdFit` is dense only; cuML routes sparse
                                  to 'qn' and raises for `solver='cd'`
                                  (`elastic_net.py:265-269`)

    THE DESIGN MATRIX IS COPIED TWICE, AND THE SECOND COPY IS NAMED.
    `cdFit` wants cuML's `F` (column-major) order and `_arrays.as_f32_c`
    hands out C order, so `fit` and `predict` build a Fortran-ordered copy
    with `np.asfortranarray`. On a large X that copy is the dominant cost of
    the call. `input_copied_` reports the dtype/contiguity copy and
    `fortran_copied_` this one; pass a float32 array that is already
    Fortran-ordered and both are False.

    Attributes
    ----------
    coef_ : ndarray (n_features,) float32
    intercept_ : float
        `mean(y) - mu_X . coef` from the ported `postProcessData`, computed
        on the DEVICE; 0.0 exactly when `fit_intercept` is False.
    n_iter_ : int
        Epochs actually run. **This number is a property of the arithmetic,
        not only of the data**: the stopping test branches on reduction
        bits, so under the default FAST mode it can differ between GPUs.
        Under `MOJOLEARN_NUMERIC_MODE=identical` it is a function of the
        inputs alone and is the card's `cd.n_iter` stage, which was
        bit-identical on Apple M4, NVIDIA H100 and AMD MI325X at leg 11.
    n_features_in_ : int

    IDENTITY: with `MOJOLEARN_IDENTITY_TRACE=<path>` set, a `fit` through
    this class writes the same `cd.*` identity card
    `solver/cd_main.mojo` writes (`solver/estimator.mojo` hands the ported
    entry a live trace). Under FAST the reductions are the vendor's and no
    cross-vendor claim is made.
    """

    def __init__(self, alpha=1.0, *, l1_ratio=0.5, fit_intercept=True,
                 max_iter=1000, tol=1e-3, solver="auto", selection="cyclic",
                 precompute=False, copy_X=True, warm_start=False,
                 positive=False, random_state=None):
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
        # Then the refusals that are this port's, each by name.
        if solver == "qn":
            raise NotImplementedError(
                "mojolearn ElasticNet: solver='qn' is not ported. cuML's "
                "solver='auto' picks 'qn' only for SPARSE input "
                "(elastic_net.py:243-244), and 'qn' under an l1 penalty is "
                "OWL-QN (min_owlqn, qn_solvers.cuh), which glm/NOT_IMPLEMENTED.tsv "
                "lists as not ported. solver='cd' (cuML's 'auto' for dense "
                "input) is the ported arm; see solver/NOT_IMPLEMENTED.tsv"
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
                "exact port is DEVIATION 611, reserved and not spent; see "
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
        if random_state is not None:
            raise NotImplementedError(
                "mojolearn ElasticNet: random_state is refused. It selects "
                "nothing here -- the only randomness in cdFit is the "
                "coordinate shuffle, and selection='random' is refused by "
                "name (DEVIATION 611 reserved)"
            )
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.selection = selection
        self.precompute = False
        self.copy_X = True
        self.warm_start = False
        self.positive = False
        self.random_state = None

    def _as_fortran(self, X, name):
        """A float32 column-major view of X, and the two copies it may cost.

        `cdFit` reads the design in cuML's `F` order (element (i, j) at
        `j * n_rows + i`). Returned alongside the flags so the caller can
        keep the array alive across the Mojo call, which `_arrays.py`'s
        contract requires.
        """
        a, copied = as_f32_c(X, name)
        f = np.asfortranarray(a)
        # THE ONE THING A SHAPE CHECK CANNOT CATCH LATER. `cd_fit` takes a
        # bare address plus n_rows and n_cols, and a C-order buffer of the
        # same size is a VALID transposed design: the fit would run and
        # return plausible coefficients for the wrong matrix. Two floats of
        # flag reading here is the only place that can be caught.
        if not f.flags["F_CONTIGUOUS"] or f.dtype != np.float32:
            raise AssertionError(
                "mojolearn: internal -- the design handed to cd_fit must be "
                f"float32 and Fortran-ordered, got {f.dtype} "
                f"F_CONTIGUOUS={f.flags['F_CONTIGUOUS']}"
            )
        return a, f, copied, f is not a

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise NotImplementedError(
                "mojolearn ElasticNet: sample_weight is not ported "
                "(cd.cuh:156-194 and :274-287 -- the weighted "
                "preProcessData, the sqrt-weight scaling of input and "
                "labels and its undo; solver/NOT_IMPLEMENTED.tsv). "
                "solver/impl/solver/cd.mojo refuses it by name"
            )
        if hasattr(X, "toarray") or hasattr(X, "tocsr"):
            raise NotImplementedError(
                "mojolearn ElasticNet: sparse X is refused. cdFit is dense "
                "only; cuML routes sparse input to solver='qn' and raises "
                "for solver='cd' (elastic_net.py:265-269), and 'qn' is not "
                "ported"
            )
        _keep, work_x, self.input_copied_, self.fortran_copied_ = (
            self._as_fortran(X, "X"))
        n_rows, n_cols = work_x.shape
        target = np.asarray(y)
        if target.ndim != 1:
            raise ValueError(
                "mojolearn ElasticNet currently requires one target")
        if target.shape[0] != n_rows:
            raise ValueError("mojolearn ElasticNet X and y lengths differ")
        target = np.ascontiguousarray(target, dtype=np.float32)

        self.coef_ = np.zeros(n_cols, dtype=np.float32)
        info = np.zeros(1, dtype=np.float32)
        n_iter = _mojolearn_solver.cd_fit(
            _addr_ro(work_x), _addr_ro(target), _addr(self.coef_),
            _addr(info),
            # ORDER MATCHES bindings/_mojolearn_solver.mojo::cd_fit_binding.
            # n_rows, n_cols, fit_intercept, max_iter, alpha, l1_ratio, tol,
            # shuffle, has_sample_weight
            [
                n_rows, n_cols, 1 if self.fit_intercept else 0,
                int(self.max_iter), float(self.alpha), float(self.l1_ratio),
                float(self.tol), _SELECTION_SHUFFLE[self.selection], 0,
            ],
        )
        self.intercept_ = float(info[0])
        self.n_iter_ = int(n_iter)
        self.n_features_in_ = n_cols
        # `_keep` held the C-order array alive across the call; naming it
        # here is what stops a reader from deleting the binding above.
        del _keep
        return self

    def predict(self, X):
        if not hasattr(self, "coef_"):
            raise ValueError("mojolearn ElasticNet: call fit before predict")
        _keep, work_x, _, _ = self._as_fortran(X, "X")
        if work_x.shape[1] != self.n_features_in_:
            raise ValueError(
                "mojolearn ElasticNet feature count differs from fit")
        out = np.empty(work_x.shape[0], dtype=np.float32)
        _mojolearn_solver.cd_predict(
            _addr_ro(work_x), _addr_ro(self.coef_), _addr(out),
            # ORDER MATCHES bindings/_mojolearn_solver.mojo::cd_predict_binding.
            # n_rows, n_cols, intercept
            [work_x.shape[0], work_x.shape[1], float(self.intercept_)],
        )
        del _keep
        return out

    def score(self, X, y):
        target = np.asarray(y, dtype=np.float64)
        residual = target - self.predict(X)
        denom = np.sum((target - target.mean()) ** 2)
        return 1.0 - float(np.sum(residual ** 2) / denom) if denom else 0.0


class Lasso(ElasticNet):
    """l1-regularized least squares by coordinate descent on the GPU.

    `Lasso` is `ElasticNet(l1_ratio=1.0)` and nothing else -- that is how
    cuML spells it (`cuml/linear_model/lasso.py:129-140`, "Lasso is just a
    special case of ElasticNet") and how scikit-learn does. Read
    `ElasticNet`'s docstring for the objective, the honored/refused table
    and the identity note; every line of it applies here with
    `l1_ratio` pinned at 1.0, so the l2 term is absent and `l1_alpha =
    alpha * n_rows`.

    `tol` defaults to 1e-3, which is cuML's and NOT scikit-learn's 1e-4.
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
