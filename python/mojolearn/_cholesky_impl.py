# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python door of the Cholesky lane (workstream D, 2026-09-14).

`cholesky/estimator.mojo`'s one-shot host entries, `cholesky_factor_host`
and `cholesky_solve_host`, reached through `bindings/_mojolearn_gp.mojo`,
which already links every Cholesky kernel because the Gaussian process
factors through them. No new binary: `Cholesky` binds `_mojolearn_gp`.

PUBLIC CPU INFERENCE (lane/inference-embedding-ivf-cholesky, 2026-09-15).
On a CPU-only install `Cholesky` binds `_mojolearn_linalg` instead, routed
to `_mojolearn_linalg_host`, which carries the same three door names over
`cholesky/host/chol_oracle.mojo` and ships in the inference wheel. Both
halves are public there: `fit` factors a GIVEN matrix (an inference answer,
not a trained model, so `_CPU_FIT_IS_INFERENCE` exempts it from the CPU
training refusal) and `solve` answers from a factor, including one written
by `save` on a GPU box and read back by `load` (or `mojolearn.host_model`).
`HostCholesky` (`_classical_host`, the CPU side) is the same class bound
to the host binding on a box that also has a GPU, which is how a GPU factor
and a CPU solve meet in one process.

WHAT CROSSES. The matrix goes down as `n * n` float32 row-major and the
factor comes back the same way, lower triangle `L` with the strict upper
triangle `+0.0` (`CholeskyFactor.l`). Four scalars come back beside it,
`info`, `nb`, `logdet` and `jitter`, because `cholesky/estimator.mojo`'s
header argues that a factor without them cannot be compared with another
factor: `info` is DATA-DEPENDENT (DEVIATION 1634) and `nb` and `jitter` are
the two numeric parameters of the profile (DEVIATIONS 1630 and 1637).
`solve` sends `info` back down unjudged so the Mojo refusal to solve
against a FAILED factor fires from Python exactly as it fires from Mojo.
The saved file carries exactly those: `L` `<f4`, `meta` `<i8` [n, info,
nb], `scalars` `<f8` [logdet, jitter] (each a float32 widened exactly), the
`jitter` parameter and the tier. Nothing is cast on load.

WHAT IS REFUSED, AND WHERE. Here, by name: a matrix that is not 2-D and
square, a right-hand side whose row count is not `n`, `solve` or `logdet_`
before `fit`, a `jitter` that is not a real number, and a model file whose
format, estimator, dtypes or shapes disagree. On the Mojo host, by name,
before any upload: a non-finite or non-symmetric matrix
(`chol_validate_matrix`, DEVIATION 1638), a jitter that is neither `0.0`
nor the profile's pinned ridge (`chol_validate_jitter`, DEVIATION 1637),
and a solve against `info != 0` (DEVIATION 1634). `nb` is not a parameter
at this boundary and cannot be: `cholesky_factor_host` takes none, on
purpose (its header, point 3).

NO SPEED CLAIM. The lane has no published number and this door adds none.
"""
from . import _backend, _serialize
from ._buffer import Array, addr, addr_ro, as_f32_c, empty
from ._mode import NumericModeMixin

#: `checks/numerics.mojo` codes, duplicated from `_backend._MODE_CODE` on
#: purpose (the GP's reason): the read-back must not share a table with
#: the thing it checks.
_MODE_CODE = {"fast": 0, "identical": 1, "deterministic": 2}

#: The saved-model format tag (`save`, `load`, `mojolearn.host_model`).
_CHOLESKY_FORMAT = "mojolearn-cholesky-1"

#: The GPU binding and the host route. A CPU-only install binds the linalg
#: route because its host binding ships; the gp host binding does not.
_GPU_BINDING = "_mojolearn_gp"
_CPU_BINDING = "_mojolearn_linalg"


# Binds `_mojolearn_gp` (which links every cholesky/ kernel; entries cholesky_factor_host /
# cholesky_solve_host of cholesky/estimator.mojo). CPU-only installs bind `_mojolearn_linalg`
# routed to `_mojolearn_linalg_host` (cholesky/host/chol_oracle.mojo); fit is inference there
# (_CPU_FIT_IS_INFERENCE). `HostCholesky` is this class on the host binding. Four scalars cross
# with L: info (data-dependent, DEVIATION 1634), nb and jitter (profile parameters, DEVIATIONS
# 1630, 1637), logdet. Host-side refusals: chol_validate_matrix (DEVIATION 1638),
# chol_validate_jitter (1637), solve against info != 0 (1634). Saved npz: L <f4, meta <i8
# [n, info, nb], scalars <f8 [logdet, jitter], jitter, tier.
class Cholesky(NumericModeMixin):
    """Cholesky factorization ``A + jitter * I = L @ L.T`` of a symmetric positive-definite matrix, then solves.

    `fit` factors a given float32 matrix once; `solve` answers ``A X = B`` from the
    factor and `logdet_` gives the log-determinant. A failed factorization is not an
    exception: check `info_` (0 means success), and a failed factor's `solve` and
    `logdet_` raise `ValueError`. The factor can be saved with `save` and loaded with
    `load`, including on a CPU-only install, where `fit` and `solve` also run.

    Parameters
    ----------
    jitter : float or None, default None
        Ridge added to the diagonal before factoring. None uses the profile's fixed
        ridge (see `profile_jitter`); 0.0 adds none. Any other value is refused,
        because the identical mode fixes the ridge to those two values. A non-number
        raises `TypeError`.

    Attributes
    ----------
    L_ : mojolearn.Array, shape (n, n), float32
        Lower-triangular factor, strict upper triangle zero. Meaningful only when
        ``info_ == 0``.
    info_ : int
        LAPACK-style status: 0 for a complete factor; ``k > 0`` means the leading minor
        of order k is not positive definite and `L_` is partial.
    nb_ : int
        Panel width used by the factorization.
    logdet_ : float
        ``log |A + jitter * I|``. Raises `ValueError` after a failed fit.
    jitter_ : float
        The ridge that was added.
    n_ : int
        Matrix size.

    Notes
    -----
    Non-finite or non-symmetric matrices, a non-square `A`, and a right-hand side whose
    row count is not n are refused with `ValueError`.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> M = rng.random((64, 64), dtype=np.float32)
    >>> A = (M @ M.T + 64 * np.eye(64, dtype=np.float32)).astype(np.float32)
    >>> chol = mojolearn.Cholesky(jitter=0.0).fit(A)
    >>> x = np.asarray(chol.solve(np.ones(64, dtype=np.float32)))
    """

    #: This class's GPU binding: the GP's, which already links `cholesky/`.
    _BINDING = _GPU_BINDING

    #: `fit` factors a given matrix; that is inference, so a CPU-only
    #: install runs it publicly (`_cpu_reference.require_training`).
    _CPU_FIT_IS_INFERENCE = True

    def __init__(self, jitter=None):
        if jitter is not None:
            if isinstance(jitter, bool) or not isinstance(jitter, (int, float)):
                raise TypeError(
                    "mojolearn Cholesky: jitter must be a float, 0.0 or None "
                    f"(the profile's ridge), got {type(jitter).__name__}"
                )
            jitter = float(jitter)
        self.jitter = jitter

    # -- the binding, and the tier it really is -----------------------------

    def _door(self):
        """(the module that serves the three door names for THIS object, the
        name of its tier read-back). A CPU-only route answers attributes it
        lacks with a by-name ImportError, so the read-back is chosen by
        route, never probed."""
        if _backend._CPU_ONLY is not None:
            return self._bind(_CPU_BINDING), "linalg_numeric_mode"
        return self._bind(), "gp_numeric_mode"

    def _extension(self):
        """The door's binding for THIS object's tier, with the binary's
        compile-time answer cross-checked against it (the GP's
        `_extension`, for its reason)."""
        mod, readback = self._door()
        want = getattr(self, "numeric_mode", None) or _backend.default_mode()
        fn = getattr(mod, readback, None)
        if fn is not None:
            got = int(fn())
            if got != _MODE_CODE.get(want):
                raise RuntimeError(
                    f"mojolearn Cholesky: numeric_mode={want!r} was requested "
                    f"but {getattr(mod, '__name__', mod)} reports compile-time mode code {got}; "
                    "the binary and the directory it sits in disagree, rebuild "
                    "it with bash bindings/build_gp.sh (a GPU install) or "
                    "bindings/build_linalg_host.sh (a CPU-only install)"
                )
        return mod

    def vendor_used(self):
        """Accelerator API of the loaded binary, read back from it; "cpu" on a CPU-only install."""
        return _backend.read_vendor(self._door()[0])

    def profile_jitter(self):
        """The profile's fixed ridge, used when ``jitter=None``."""
        return float(self._extension().cholesky_profile_jitter())

    # -- fit ----------------------------------------------------------------

    def fit(self, A):
        """Factor ``A + jitter * I``; returns self.

        `A` is a square 2-D float32 matrix, otherwise `ValueError`. A failed factorization
        sets ``info_ != 0`` instead of raising.
        """
        a, _ = as_f32_c(A, ndim=2, name="A")
        n, m = a.shape
        if n != m:
            raise ValueError(
                f"mojolearn Cholesky: A must be square, got shape {a.shape}"
            )
        jitter = self.profile_jitter() if self.jitter is None else self.jitter
        l_out = empty((n * n,), "<f4")
        # info, nb, logdet, jitter -- in that order.
        scalars = empty((4,), "<f8")
        # Every array addressed below is bound to a local in this frame,
        # which is what keeps the addresses alive (`_arrays.py`).
        info = self._extension().cholesky_factor(
            # ORDER MATCHES bindings/_mojolearn_gp.mojo::cholesky_factor_binding.
            # a, l_out, scalars_out
            [addr_ro(a, name="A"), addr(l_out, name="L_"), addr(scalars, name="scalars")],
            # n, jitter
            [n, jitter],
        )
        self.n_ = n
        self.L_ = l_out.reshape((n, n))
        self.info_ = int(info)
        self.nb_ = int(scalars[1])
        self._logdet = float(scalars[2])
        self.jitter_ = float(scalars[3])
        return self

    @property
    def logdet_(self):
        """``log |A + jitter * I|``; raises `ValueError` after a failed fit."""
        if not hasattr(self, "info_"):
            raise AttributeError("mojolearn Cholesky: logdet_ is set by fit")
        if self.info_ != 0:
            raise ValueError(
                f"mojolearn Cholesky: the factorization failed (info={self.info_}), "
                "so there is no determinant to report (DEVIATION 1634)"
            )
        return self._logdet

    # -- solve --------------------------------------------------------------

    def solve(self, B):
        """Solve ``A X = B`` from the factor.

        Parameters
        ----------
        B : float32 array, shape (n,) or (n, nrhs)

        Returns
        -------
        X : float32 array, the shape of `B`

        Raises
        ------
        ValueError
            Before `fit`, against a failed factor, or when `B` has the wrong row count or
            no columns.
        """
        if not hasattr(self, "L_"):
            raise ValueError("mojolearn Cholesky: call fit before solve")
        b, _ = as_f32_c(B, ndim=None, name="B")
        if b.ndim == 1:
            rows, nrhs, squeeze = b.shape[0], 1, True
        elif b.ndim == 2:
            rows, nrhs, squeeze = b.shape[0], b.shape[1], False
        else:
            raise ValueError(
                f"mojolearn Cholesky: B must be 1-D or 2-D, got {b.ndim}-D"
            )
        if rows != self.n_:
            raise ValueError(
                f"mojolearn Cholesky: B has {rows} rows, the factor is {self.n_} x {self.n_}"
            )
        if nrhs < 1:
            raise ValueError("mojolearn Cholesky: B must have at least one column")
        flat_l = self.L_.reshape((self.n_ * self.n_,))
        flat_b = b.reshape((self.n_ * nrhs,))
        x_out = empty((self.n_ * nrhs,), "<f4")
        self._extension().cholesky_solve(
            # ORDER MATCHES bindings/_mojolearn_gp.mojo::cholesky_solve_binding.
            # l, b, x_out
            [addr_ro(flat_l, name="L_"), addr_ro(flat_b, name="B"), addr(x_out, name="X")],
            # n, nrhs, info, nb, logdet, jitter
            [self.n_, nrhs, self.info_, self.nb_, self._logdet, self.jitter_],
        )
        if squeeze:
            return x_out
        return x_out.reshape((self.n_, nrhs))

    # -- save and load ------------------------------------------------------

    def save(self, path):
        """Write the factor to `path` as an .npz file.

        Stores the factor, ``info_``, ``nb_``, ``logdet_``, the jitter and the numeric mode.
        A failed factor saves too; its `solve` still refuses. Raises `RuntimeError` before
        `fit`.
        """
        if not hasattr(self, "L_"):
            raise RuntimeError("mojolearn Cholesky: call fit before save")
        mode = getattr(self, "numeric_mode", None) or _backend.default_mode()
        if not isinstance(mode, str) or mode.strip().lower() not in _MODE_CODE:
            raise ValueError(f"mojolearn Cholesky: cannot save invalid numeric_mode {mode!r}")
        l_arr = self.L_ if isinstance(self.L_, Array) else as_f32_c(self.L_, ndim=2, name="L_")[0]
        if l_arr.dtype != "<f4" or tuple(l_arr.shape) != (self.n_, self.n_):
            raise ValueError(
                f"mojolearn Cholesky: L_ must be float32 ({self.n_}, {self.n_}), "
                f"got {l_arr.dtype} {tuple(l_arr.shape)}"
            )
        arrays = {
            "format": _CHOLESKY_FORMAT,
            "estimator": "Cholesky",
            "numeric_mode": mode.strip().lower(),
            "L": l_arr,
            "meta": Array.from_list([int(self.n_), int(self.info_), int(self.nb_)], "<i8"),
            "scalars": Array.from_list([float(self._logdet), float(self.jitter_)], "<f8"),
            # -1.0 means "the profile's ridge" (None); the fitted ridge is
            # in `scalars` either way.
            "jitter_param": Array.from_list([-1.0 if self.jitter is None else float(self.jitter)], "<f8"),
        }
        return _serialize.write_npz(path, arrays)

    @classmethod
    def load(cls, path):
        """Load a factor written by `save`.

        The result can `solve` and report `logdet_`; it holds no matrix and does not
        refactor. A file saved by another estimator, or with mismatched shapes or dtypes,
        raises `ValueError`.
        """
        arrays = _serialize.read_npz(path, _CHOLESKY_FORMAT)
        saved_as = _serialize.scalar_str(arrays, "estimator")
        if saved_as not in (c.__name__ for c in cls.__mro__):  # glue: checks the saved class name
            raise ValueError(f"mojolearn: {path!r} was saved by {saved_as}, not {cls.__name__}")
        mode = _serialize.scalar_str(arrays, "numeric_mode")
        if mode not in _MODE_CODE:
            raise ValueError(f"mojolearn: invalid saved numeric_mode {mode!r}")
        meta = _serialize.exact(arrays, "meta", "<i8")
        scalars = _serialize.exact(arrays, "scalars", "<f8")
        jp = _serialize.exact(arrays, "jitter_param", "<f8")
        if meta.size != 3 or scalars.size != 2 or jp.size != 1:
            raise ValueError(
                f"mojolearn: {path!r} holds meta {meta.size}, scalars {scalars.size} and "
                f"jitter_param {jp.size} fields; 3, 2 and 1 are needed"
            )
        n = int(meta[0])
        l_arr = _serialize.exact(arrays, "L", "<f4")
        if n < 1 or tuple(l_arr.shape) != (n, n):
            raise ValueError(f"mojolearn: {path!r} L has shape {tuple(l_arr.shape)}, the factor is {n} x {n}")
        jitter = float(jp[0])
        obj = cls(jitter=None if jitter == -1.0 else jitter)
        obj.numeric_mode = mode
        obj.n_ = n
        obj.L_ = l_arr
        obj.info_ = int(meta[1])
        obj.nb_ = int(meta[2])
        obj._logdet = float(scalars[0])
        obj.jitter_ = float(scalars[1])
        return obj


__all__ = ["Cholesky"]
