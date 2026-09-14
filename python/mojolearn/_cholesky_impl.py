# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The Python door of the Cholesky lane (workstream D, 2026-09-14).

`cholesky/estimator.mojo`'s one-shot host entries, `cholesky_factor_host`
and `cholesky_solve_host`, reached through `bindings/_mojolearn_gp.mojo`,
which already links every Cholesky kernel because the Gaussian process
factors through them. No new binary: `Cholesky` binds `_mojolearn_gp`.

WHAT CROSSES. The matrix goes down as `n * n` float32 row-major and the
factor comes back the same way, lower triangle `L` with the strict upper
triangle `+0.0` (`CholeskyFactor.l`). Four scalars come back beside it,
`info`, `nb`, `logdet` and `jitter`, because `cholesky/estimator.mojo`'s
header argues that a factor without them cannot be compared with another
factor: `info` is DATA-DEPENDENT (DEVIATION 1634) and `nb` and `jitter` are
the two numeric parameters of the profile (DEVIATIONS 1630 and 1637).
`solve` sends `info` back down unjudged so the Mojo refusal to solve
against a FAILED factor fires from Python exactly as it fires from Mojo.

WHAT IS REFUSED, AND WHERE. Here, by name: a matrix that is not 2-D and
square, a right-hand side whose row count is not `n`, `solve` or `logdet_`
before `fit`, and a `jitter` that is not a real number. On the Mojo host,
by name, before any upload: a non-finite or non-symmetric matrix
(`chol_validate_matrix`, DEVIATION 1638), a jitter that is neither `0.0`
nor the profile's pinned ridge (`chol_validate_jitter`, DEVIATION 1637),
and a solve against `info != 0` (DEVIATION 1634). `nb` is not a parameter
at this boundary and cannot be: `cholesky_factor_host` takes none, on
purpose (its header, point 3).

NO SPEED CLAIM. The lane has no published number and this door adds none.
"""
from . import _backend
from ._buffer import addr, addr_ro, as_f32_c, empty
from ._mode import NumericModeMixin

#: `checks/numerics.mojo` codes, duplicated from `_backend._MODE_CODE` on
#: purpose (the GP's reason): the read-back must not share a table with
#: the thing it checks.
_MODE_CODE = {"fast": 0, "identical": 1, "deterministic": 2}


class Cholesky(NumericModeMixin):
    """`A + jitter I = L L^T` on the GPU, one shot, then `solve`.

    Parameters
    ----------
    jitter : float or None, default None
        The ridge added to the diagonal before factoring. `None` means the
        profile's pinned ridge, read from the binding
        (`cholesky_profile_jitter`, DEVIATION 1637); `0.0` means none. Any
        other value is refused BY NAME on the Mojo host, because the
        identical tier pins the ridge to those two values. It is a
        parameter of the PROFILE and is carried on the fitted object.

    Attributes
    ----------
    L_ : Array (n, n) float32
        The factor, lower triangle `L`, strict upper triangle `+0.0`.
        Meaningful only when `info_ == 0`.
    info_ : int
        LAPACK's `info`. READ IT BEFORE YOU BELIEVE `L_` (DEVIATION 1634):
        0 means a complete factor; `k > 0` means the leading minor of
        order `k` was not positive definite and `L_` is partial.
    nb_ : int
        The panel width that ran. Part of the profile, not a tuning record.
    logdet_ : float
        `log |A + jitter I|`, computed on the device by the one pinned
        fold. Reading it on a failed fit raises by name, as
        `cholesky_logdet_host` does.
    jitter_ : float
        The ridge that was added, by value.
    n_ : int
    """

    #: This class's binding: the GP's, which already links `cholesky/`.
    _BINDING = "_mojolearn_gp"

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

    def _extension(self):
        """The `_mojolearn_gp` binding for THIS object's tier, with the
        binary's compile-time answer cross-checked against it (the GP's
        `_extension`, for its reason)."""
        mod = self._bind()
        want = getattr(self, "numeric_mode", None) or _backend.default_mode()
        fn = getattr(mod, "gp_numeric_mode", None)
        if fn is not None:
            got = int(fn())
            if got != _MODE_CODE.get(want):
                raise RuntimeError(
                    f"mojolearn Cholesky: numeric_mode={want!r} was requested "
                    f"but {mod.__name__} reports compile-time mode code {got}; "
                    "the binary and the directory it sits in disagree, rebuild "
                    "it with bash bindings/build_gp.sh"
                )
        return mod

    def profile_jitter(self):
        """The profile's pinned ridge, read from the binding by name."""
        return float(self._extension().cholesky_profile_jitter())

    # -- fit ----------------------------------------------------------------

    def fit(self, A):
        """Factor `A + jitter I`. Returns `self`. `info_` is a RESULT, not
        an exception; a failed factorization is a fitted object whose
        `solve` and `logdet_` refuse by name."""
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
        """`A X = B` from the factor. `B` is `(n,)` or `(n, nrhs)` and `X`
        comes back in the same shape, float32. A failed factor is refused
        BY NAME IN MOJO; `info_` goes down with the call so that refusal
        stays reachable from here."""
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


__all__ = ["Cholesky"]
