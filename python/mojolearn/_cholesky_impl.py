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
from . import _backend, _native_archive_bridge, _native_family_bridge
from ._buffer import addr, addr_ro, as_f32_c, empty
from ._mode import NumericModeMixin

#: `checks/numerics.mojo` codes, duplicated from `_backend._MODE_CODE` on
#: purpose (the GP's reason): the read-back must not share a table with
#: the thing it checks.
_MODE_CODE = {"fast": 0, "identical": 1}

# The compiled family owner (bindings/linalg_native.mojo), connected once at
# import: each public operation selects its binding inside the native owner.
_NATIVE, _NATIVE_CONTEXT = _native_family_bridge.connect()
_native_family_bridge.tables(family_chol_mode_codes=dict(_MODE_CODE))

#: The saved-model format tag (`save`, `load`, `mojolearn.host_model`).
_CHOLESKY_FORMAT = "mojolearn-cholesky-1"

# The compiled saved-model owner, connected once at import: `save` and
# `load` are one native call each (bindings/archive_schema_native.mojo
# writes the schema registered after the class; host_archive_native.mojo
# builds the loaded model).
_ARCHIVE, _ARCHIVE_CONTEXT = _native_archive_bridge.connect()

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
    exception: check `info_` (0 means success); a failed factor's `solve` and
    `logdet_` refuse. The factor can be saved with `save` and loaded with
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
    A non-square `A` or a right-hand side whose row count is not n raises
    `ValueError`; non-finite or non-symmetric matrices are refused before any work.

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
    _mojolearn_native_public = True
    _BINDING = _GPU_BINDING

    #: `fit` factors a given matrix; that is inference, so a CPU-only
    #: install runs it publicly (`_cpu_reference.require_training`).
    _CPU_FIT_IS_INFERENCE = True

    def __init__(self, jitter=None, *, numeric_mode=None):
        return _NATIVE(self, "family_chol_initialize", (jitter, numeric_mode), _NATIVE_CONTEXT)

    # -- the binding, and the tier it really is -----------------------------

    def _door(self):
        return _NATIVE(self, "family_chol_door", (), _NATIVE_CONTEXT)

    def _door_binding(self):
        return _NATIVE(self, "family_chol_binding", (), _NATIVE_CONTEXT)

    def _extension(self):
        return _NATIVE(self, "family_chol_binding", (), _NATIVE_CONTEXT)

    def vendor_used(self):
        """Accelerator API of the loaded binary, read back from it; "cpu" on a CPU-only install."""
        return _NATIVE(self, "family_chol_vendor", (), _NATIVE_CONTEXT)

    def profile_jitter(self):
        """The profile's fixed ridge, used when ``jitter=None``."""
        return _NATIVE(self, "family_chol_profile_jitter", (), _NATIVE_CONTEXT)

    # -- fit ----------------------------------------------------------------

    def fit(self, A):
        """Factor ``A + jitter * I``; returns self.

        `A` is a square 2-D float32 matrix, otherwise `ValueError`. A failed factorization
        sets ``info_ != 0`` instead of raising.
        """
        return _NATIVE(self, "family_chol_fit", (None, A), _NATIVE_CONTEXT)

    @property
    def logdet_(self):
        """``log |A + jitter * I|``; raises `ValueError` after a failed fit."""
        return _NATIVE(self, "family_chol_logdet", (), _NATIVE_CONTEXT)

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
            Before `fit`, or when `B` has the wrong row count or no columns. Solving
            against a failed factor is also refused.
        """
        return _NATIVE(self, "family_chol_solve", (None, B), _NATIVE_CONTEXT)

    # -- save and load ------------------------------------------------------

    def save(self, path):
        """Write the factor to `path` as an .npz file.

        Stores the factor, ``info_``, ``nb_``, ``logdet_``, the jitter and the numeric mode.
        A failed factor saves too; its `solve` still refuses. Raises `RuntimeError` before
        `fit`.
        """
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

    @classmethod
    def load(cls, path):
        """Load a factor written by `save`.

        The result can `solve` and report `logdet_`; it holds no matrix and does not
        refactor. A file saved by another estimator, or with mismatched shapes or dtypes,
        raises `ValueError`.
        """
        return _ARCHIVE(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)


# The 0.8.55 writer's members and refusals, in its order: the fit check, the
# numeric mode, the factor's dtype and shape, then the field vectors. The
# saved estimator is always "Cholesky"; -1.0 in jitter_param means "the
# profile's ridge" (None), the fitted ridge is in `scalars` either way.
_native_archive_bridge.register(Cholesky, _CHOLESKY_FORMAT, (
    ("need", "L_", "RuntimeError", "mojolearn Cholesky: call fit before save"),
    ("format", "format", _CHOLESKY_FORMAT),
    ("format", "estimator", "Cholesky"),
    ("mode", "numeric_mode", "mojolearn Cholesky: "),
    ("check_array", "L_", "<f4", (("int", "n_"), ("int", "n_")), "ValueError",
     "mojolearn Cholesky: L_ must be float32 {want}, got {dtype} {shape}"),
    ("array", "L", "L_"),
    ("i8", "meta", (("int", "n_"), ("int", "info_"), ("int", "nb_"))),
    ("f8", "scalars", (("float", "_logdet"), ("float", "jitter_"))),
    ("f8", "jitter_param", (("none_real", "jitter", -1.0),)),
))


__all__ = ["Cholesky"]


# The class the multi-GPU Cholesky drivers admit (bindings/linalg_native.mojo
# `family_chol_par_*`).
_native_family_bridge.tables(family_chol_par_class=Cholesky)

# Import-time connections; the native operation selects the current mode/route.
_native_family_bridge.connect_bindings(_GPU_BINDING, _CPU_BINDING)
