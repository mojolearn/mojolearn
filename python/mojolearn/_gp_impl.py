# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Exact dense Gaussian process regression on the GPU.

PRIVATE MODULE. `GaussianProcessRegressor` and the four kernel classes
(`RBF`, `Matern`, `ConstantKernel`, `WhiteKernel`) are re-exported from
`mojolearn/__init__.py`.

EXPOSED 2026-09-01, AND THE HISTORY OF THE WITHHOLDING IS PART OF THIS
FILE'S CONTRACT. This estimator sat in `__init__.py`'s `_NOT_YET` -- the
only entry ever placed there for a reason other than a missing surface --
because its IDENTICAL card was believed to diverge Apple against AMD on 8
of 3,494 stages. That reading was WITHDRAWN at `9835094e` (2026-09-01):
the eight differing lines are the SABOTAGED half of one `GP_SAB_STD_EXP`
clean-then-sabotaged pair, an arm that exists precisely because a device
`exp` is a vendor's choice in its last bit, and `check_gp_sabotages`
writes both halves of every pair into the same card. The SHIPPED path is
byte-identical Apple M4 against AMD MI325X on the other 3,486 lines. See
`bench/results/e1/GP_CROSS_VENDOR_DIVERGENCE.md` (corrected in place) and
`gaussian_process/README.md`. With the sole blocker withdrawn, the
estimator is exposed.

**THERE IS NO REFERENCE GAUSSIAN PROCESS.** cuML, cuVS and RAFT implement
none at the pinned commits, so this lane adds the capability per `ok-to-add-capability`.
scikit-learn's `sklearn/gaussian_process/_gpr.py` is the SEMANTICS
reference and the oracle, never the design source; every name this surface
shares with scikit-learn means what scikit-learn means by it.

THE BINDING THIS FILE CALLS IS `_mojolearn_gp`, AND ITS ABI WAS
RENEGOTIATED ONCE, ON MEASUREMENT. This file landed (22a5b550) with the
binding OWED and a contract at each call site in which every buffer
address was its own positional argument -- ten arguments for `gpr_fit`,
thirteen for `gpr_predict`. The binding was written to that spelling
verbatim (8c449b70) and the first `bash bindings/build_gp.sh` FAILED AT
def_function ELABORATION (2026-09-01, log /tmp/build_gp_fast.log),
exactly as `bindings/_mojolearn.mojo`'s header predicts above roughly
nine arguments. The maintainer renegotiated BOTH sides the same day to
the tree's own fold -- the one `knn_search` took when its ten arguments
hit the same wall: each entry point takes exactly TWO Python lists,
`addrs` (every buffer address, order written in the binding docstring
and at the call site below in the same words) and `params` (the scalar
list 22a5b550 spelled, unchanged to the word). EVERY ARRAY WHOSE ADDRESS
GOES INTO `addrs` MUST BE BOUND TO A LOCAL for the duration of the call:
an address inside a list keeps nothing alive (`_buffer.py`). The rest of
the contract is unchanged: function names, and that the kernel spec MUST
be rebuilt through `gaussian_process/checks/kernels.mojo`'s constructors
so their refusals stay reachable. The three-tier build is still RUN
OWED.

WHAT THE KERNEL CLASSES ARE. A covariance function here is a POSTFIX node
list over five flat arrays (`GPKernelSpec`, DEVIATION 1756), which is
exactly what a binding can hand over without a graph representation. The
four classes below and their `+` / `*` operators build that list and
nothing else: they hold no bounds, no `theta` and no gradient, because
hyperparameter optimization is not implemented (DEVIATION 1761) and the
pieces that exist only to serve it are absent rather than present and
unused. HYPERPARAMETERS ARE NOT VALIDATED IN PYTHON: a NaN noise level, a
non-positive length scale, a Matern `nu` outside the three closed forms
each go down to the Mojo constructors unjudged, so those refusals
(DEVIATIONS 1765, 1768) stay reachable from this surface -- the same rule
that keeps `SVR`'s epsilon refusals reachable.

THE LANE'S STANDING: gated on the Apple M4 in both tiers, eleven checks
each, nine sabotage arms driven at run time; the IDENTICAL card
byte-identical Apple M4 against AMD MI325X on every shipped-path line
(at `9835094e`'s reading of the 2026-08-28 cards). The third column came
later: the identity harness's 136-lane record at 4048e1b51
(bench/results/identity_break/2026-09-14_136-lanes/diff.three-columns.txt)
reads IDENTICAL x3 on all 36 gp training cells and 36 infer cells across
the Apple M4, an NVIDIA H100 and an AMD MI325X under IDENTICAL. The gp SPEED ladder is UNRUN:
exposure is a correctness claim, not a
speed claim.
"""

from . import _portable_math as math
from . import _backend, _native_archive_bridge, _native_family_bridge

#: `gaussian_process/gp_optim_items.mojo`'s stop words (GP_STOP_*).
_OPT_STOPS = ("running", "pgtol", "no-descent", "line-search", "ftol", "max-iter", "nonfinite-start")
from ._mode import NumericModeMixin
from .linear_model import _flatten, _shape_of
from .preprocessing import StandardScaler, _STANDARD_CONTEXT

#: `checks/numerics.mojo` codes, duplicated from `_backend._MODE_CODE` on
#: purpose, for `_arima_impl.py`'s reason: the read-back must not share a
#: table with the thing it checks.
_MODE_CODE = {"fast": 0, "identical": 1}

# The GP_K_* node kinds, mirroring gaussian_process/checks/kernels.mojo.
# A silent renumbering on either side is a WRONG KERNEL, not a failure,
# which is why the surface test plants a fixture whose answer moves if the
# kind codes move.
_K_CONST = 0
_K_WHITE = 1
_K_RBF = 2
_K_MATERN = 3
_K_SUM = 4
_K_PROD = 5


def _as_length_scale(length_scale, what):
    """A list of floats from a scalar or a sequence. CONVERSION ONLY, no
    judgement: positivity, finiteness and the isotropic-or-ARD length rule
    are `gp_kernel_rbf` / `gp_kernel_matern` / `gp_validate_kernel`'s
    refusals (kernels.mojo), and a copy of them here would make the named
    Mojo refusal unreachable from Python and therefore untestable."""
    if not _shape_of(length_scale):
        return [float(length_scale)]
    ls = [float(v) for v in _flatten(length_scale)]  # glue: converts the length-scale argument
    if len(ls) == 0:
        raise ValueError(
            f"mojolearn {what}: length_scale is an empty sequence; pass a "
            "scalar (isotropic) or n_features values (ARD)"
        )
    return ls


#: scikit-learn's default hyperparameter bounds, `kernels.py` (every leaf).
_DEFAULT_BOUNDS = (1e-5, 1e5)


def _as_bounds(bounds, n, what):
    """"fixed" or a list of `n` `(lo, hi)` float pairs. A single pair applies
    to every entry of an ARD length scale, scikit-learn's rule. Each bound
    must be finite and positive with `lo <= hi` (theta is its log;
    DEVIATION 2880)."""
    if isinstance(bounds, str):
        if bounds != "fixed":
            raise ValueError(f"mojolearn {what}: bounds must be a (low, high) pair or 'fixed', got {bounds!r}")
        return "fixed"
    shape = _shape_of(bounds)
    if shape == (2,):
        pairs = [tuple(float(v) for v in _flatten(bounds))] * n  # glue: converts the bounds argument
    elif shape == (n, 2):
        flat = [float(v) for v in _flatten(bounds)]  # glue: converts the bounds argument
        pairs = list(zip(flat[0::2], flat[1::2]))
    else:
        raise ValueError(
            f"mojolearn {what}: bounds must be 'fixed', a (low, high) pair, or {n} pairs; got shape {shape}")
    for lo, hi in pairs:  # glue: validates each hyperparameter bound pair
        if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 < lo <= hi):
            raise ValueError(
                f"mojolearn {what}: every bound must be finite and positive with low <= high, got "
                f"({lo!r}, {hi!r}); theta is the natural log of a hyperparameter (DEVIATION 2880)")
    return pairs


# Maintainer notes: the four classes build a postfix node list (`a b +` /
# `a b *`) over GPKernelSpec's flat arrays; nothing is distributed or
# reassociated (DEVIATION 1756). Hyperparameters per DEVIATION 2880: a user's
# kernel computes `theta` with the owned host math helper; the fit uses the
# same pinned logarithm arithmetic through its Mojo binding, and a fitted
# `kernel_` reports the optimizer's own theta.
class Kernel:
    """Base class of the Gaussian process kernels.

    Kernels combine with ``+`` and ``*``: ``k1 + k2`` is the sum and
    ``k1 * k2`` the product of two kernels, as in scikit-learn's ``Sum`` and
    ``Product``. Only mojolearn kernels combine; write a scaled kernel as
    ``ConstantKernel(2.0) * RBF(...)`` rather than ``2.0 * RBF(...)``
    (combining a kernel with anything else raises ``TypeError``).

    The hyperparameter view follows scikit-learn: `theta` is the natural log
    of every free hyperparameter, left operand first, with one entry per
    feature for an anisotropic length scale; `bounds` is the log of each
    entry's ``(low, high)`` pair; `n_dims` is their count. A kernel built
    with bounds ``"fixed"`` contributes no entries.
    """

    def __add__(self, other):
        return _NATIVE(self, "family_gp_kernel_add", (other,), _NATIVE_CONTEXT)

    def __mul__(self, other):
        return _NATIVE(self, "family_gp_kernel_mul", (other,), _NATIVE_CONTEXT)

    def _nodes(self):
        """Postfix list of `(kind, param, length_scales)` tuples."""
        raise NotImplementedError

    def _hyper(self):
        """Postfix list, one per node: `None` for an operator or a fixed leaf,
        else `(values, bounds)` for the leaf's free entries."""
        raise NotImplementedError

    def _replace(self, values):
        """A copy with the free entries replaced by `values` (consumed from
        the front of the list, which is mutated)."""
        raise NotImplementedError

    def _free_flags(self):
        return [0 if h is None else 1 for h in self._hyper()]  # glue: one flag per postfix node

    def _free_values(self):
        return [v for h in self._hyper() if h is not None for v in h[0]]  # glue: collects free kernel hyperparameters (n_dims entries)

    def _free_bounds(self):
        return [b for h in self._hyper() if h is not None for b in h[1]]  # glue: collects bounds of free kernel hyperparameters (n_dims entries)

    def _with_free_values(self, values, theta=None):
        out = self._replace(list(values))
        if theta is not None:
            out._theta_fit = [float(t) for t in theta]  # glue: copies the fitted theta, kernel hyperparameters (n_dims entries)
        return out

    @property
    def n_dims(self):
        """Number of free hyperparameters (the length of `theta`)."""
        return _NATIVE(self, "family_gp_kernel_n_dims", (), _NATIVE_CONTEXT)

    @property
    def theta(self):
        """Natural log of the free hyperparameters, as a list of floats.

        On a fitted ``kernel_`` this is the optimizer's own theta.
        """
        return _NATIVE(self, "family_gp_kernel_theta", (), _NATIVE_CONTEXT)

    @property
    def bounds(self):
        """Natural log of each free hyperparameter's ``(low, high)`` bounds,
        as a list of ``[low, high]`` pairs."""
        return _NATIVE(self, "family_gp_kernel_bounds", (), _NATIVE_CONTEXT)

    def __repr__(self):
        return _NATIVE(self, "family_gp_kernel_repr", (), _NATIVE_CONTEXT)


# Reference: scikit-learn `kernels.py:1187`.
class ConstantKernel(Kernel):
    """Constant kernel: ``k(x, y) = constant_value`` for every pair of points.

    Follows scikit-learn's ``ConstantKernel``. Multiply it with another
    kernel to scale that kernel's magnitude.

    Parameters
    ----------
    constant_value : float, default 1.0
        The constant.
    constant_value_bounds : pair of float or "fixed", default (1e-5, 1e5)
        Bounds for the optimizer, or ``"fixed"`` to keep the value fixed.
        Both bounds must be finite and positive with ``low <= high``;
        otherwise ``ValueError`` is raised.

    Examples
    --------
    >>> import mojolearn
    >>> kernel = mojolearn.ConstantKernel(2.0) * mojolearn.RBF(1.0)
    """

    def __init__(self, constant_value=1.0, constant_value_bounds=_DEFAULT_BOUNDS):
        _NATIVE(self, "family_gp_kernel_initialize", (constant_value, constant_value_bounds), _NATIVE_CONTEXT)

    def _nodes(self):
        return [(_K_CONST, self.constant_value, [])]

    def _hyper(self):
        b = self.constant_value_bounds
        return [None if b == "fixed" else ([self.constant_value], b)]

    def _replace(self, values):
        v = values.pop(0) if self.constant_value_bounds != "fixed" else self.constant_value
        return ConstantKernel(v, self.constant_value_bounds)

    def _name(self):
        return f"ConstantKernel({self.constant_value!r})"


# Reference: scikit-learn `kernels.py:1325`; diagonal-only rule is DEVIATION 1762.
class WhiteKernel(Kernel):
    """White noise kernel: adds ``noise_level`` to the training diagonal.

    Follows scikit-learn's ``WhiteKernel``: the noise appears only on the
    diagonal of the training covariance, and contributes zero to the
    covariance between training and query points. Add it to another kernel
    to model observation noise.

    Parameters
    ----------
    noise_level : float, default 1.0
        The noise variance.
    noise_level_bounds : pair of float or "fixed", default (1e-5, 1e5)
        Bounds for the optimizer, or ``"fixed"`` to keep the value fixed.
        Both bounds must be finite and positive with ``low <= high``;
        otherwise ``ValueError`` is raised.

    Examples
    --------
    >>> import mojolearn
    >>> kernel = mojolearn.RBF(1.0) + mojolearn.WhiteKernel(0.1)
    """

    def __init__(self, noise_level=1.0, noise_level_bounds=_DEFAULT_BOUNDS):
        _NATIVE(self, "family_gp_kernel_initialize", (noise_level, noise_level_bounds), _NATIVE_CONTEXT)

    def _nodes(self):
        return [(_K_WHITE, self.noise_level, [])]

    def _hyper(self):
        b = self.noise_level_bounds
        return [None if b == "fixed" else ([self.noise_level], b)]

    def _replace(self, values):
        v = values.pop(0) if self.noise_level_bounds != "fixed" else self.noise_level
        return WhiteKernel(v, self.noise_level_bounds)

    def _name(self):
        return f"WhiteKernel({self.noise_level!r})"


# Reference: scikit-learn `kernels.py:1448`.
class RBF(Kernel):
    """Radial basis function (squared exponential) kernel.

    Follows scikit-learn's ``RBF``:
    ``k(x, y) = exp(-0.5 * ||(x - y) / length_scale||^2)``.

    Parameters
    ----------
    length_scale : float or sequence of float, default 1.0
        A scalar gives an isotropic kernel; a sequence of ``n_features``
        values gives one length scale per feature (anisotropic). Any other
        length is refused when the model is fitted. Values must be positive
        and finite; they are checked at fit time.
    length_scale_bounds : pair of float, sequence of pairs, or "fixed", default (1e-5, 1e5)
        Bounds for the optimizer: one pair for every length scale, one pair
        per length scale, or ``"fixed"``. Both bounds must be finite and
        positive with ``low <= high``; otherwise ``ValueError`` is raised.

    Examples
    --------
    >>> import mojolearn
    >>> kernel = mojolearn.RBF(length_scale=[1.0, 2.0, 0.5])
    """

    def __init__(self, length_scale=1.0, length_scale_bounds=_DEFAULT_BOUNDS):
        _NATIVE(self, "family_gp_kernel_initialize", (length_scale, length_scale_bounds), _NATIVE_CONTEXT)

    def _nodes(self):
        return [(_K_RBF, 0.0, list(self.length_scale))]

    def _hyper(self):
        b = self.length_scale_bounds
        return [None if b == "fixed" else (list(self.length_scale), b)]

    def _replace(self, values):
        ls = list(self.length_scale)
        if self.length_scale_bounds != "fixed":
            ls = [values.pop(0) for _ in ls]  # glue: consumes this leaf's length scales
        return RBF(ls, self.length_scale_bounds)

    def _name(self):
        return f"RBF({self.length_scale!r})"


# Reference: scikit-learn `kernels.py:1601`. `nu` is not checked here so
# that `gp_kernel_matern`'s refusal (DEVIATION 1765) stays reachable.
class Matern(Kernel):
    """Matern kernel with ``nu`` in {0.5, 1.5, 2.5}.

    Follows scikit-learn's ``Matern``, with the same argument order. Only the
    three closed forms are supported: ``nu=0.5`` (absolute exponential),
    ``1.5`` (once differentiable) and ``2.5`` (twice differentiable). Any
    other ``nu``, including ``inf``, is refused with an error when the model
    is fitted; use `RBF` for the ``nu=inf`` limit.

    Parameters
    ----------
    length_scale : float or sequence of float, default 1.0
        A scalar gives an isotropic kernel; a sequence of ``n_features``
        values gives one length scale per feature (anisotropic). Values must
        be positive and finite; they are checked at fit time.
    length_scale_bounds : pair of float, sequence of pairs, or "fixed", default (1e-5, 1e5)
        Bounds for the optimizer: one pair for every length scale, one pair
        per length scale, or ``"fixed"``. Both bounds must be finite and
        positive with ``low <= high``; otherwise ``ValueError`` is raised.
    nu : float, default 1.5
        Smoothness: 0.5, 1.5 or 2.5.

    Examples
    --------
    >>> import mojolearn
    >>> kernel = mojolearn.Matern(length_scale=1.0, nu=2.5)
    """

    def __init__(self, length_scale=1.0, length_scale_bounds=_DEFAULT_BOUNDS, nu=1.5):
        _NATIVE(self, "family_gp_kernel_initialize", (length_scale, length_scale_bounds, nu), _NATIVE_CONTEXT)

    def _nodes(self):
        return [(_K_MATERN, self.nu, list(self.length_scale))]

    def _hyper(self):
        b = self.length_scale_bounds
        return [None if b == "fixed" else (list(self.length_scale), b)]

    def _replace(self, values):
        ls = list(self.length_scale)
        if self.length_scale_bounds != "fixed":
            ls = [values.pop(0) for _ in ls]  # glue: consumes this leaf's length scales
        return Matern(ls, self.length_scale_bounds, nu=self.nu)

    def _name(self):
        return f"Matern({self.length_scale!r}, nu={self.nu!r})"


class _Combined(Kernel):
    def __init__(self, a, b, op, sym):
        _NATIVE(self, "family_gp_kernel_initialize", (a, b, op, sym), _NATIVE_CONTEXT)

    def _nodes(self):
        return self.a._nodes() + self.b._nodes() + [(self.op, 0.0, [])]

    def _hyper(self):
        return self.a._hyper() + self.b._hyper() + [None]

    def _replace(self, values):
        return _Combined(self.a._replace(values), self.b._replace(values), self.op, self.sym)

    def _name(self):
        return f"({self.a._name()} {self.sym} {self.b._name()})"


class _SavedKernel(Kernel):
    """A kernel read back from a saved model: its postfix `(kind, param,
    length_scales)` nodes exactly as `_nodes()` produced them at save time,
    so `_kernel_arrays()` hands the binding the same four arrays the fit
    handed it (the neighbors and density inference lane, 2026-09-15)."""

    def __init__(self, nodes):
        _NATIVE(self, "family_gp_kernel_initialize", (nodes,), _NATIVE_CONTEXT)

    def _nodes(self):
        return [(k, p, list(ls)) for k, p, ls in self.__nodes]  # glue: copies the kernel's postfix nodes

    def _hyper(self):
        # A saved kernel carries no bounds: every hyperparameter is fixed.
        return [None for _ in self.__nodes]  # glue: one None per postfix node

    def _replace(self, values):
        return _SavedKernel(self.__nodes)

    def _name(self):
        return f"_SavedKernel({len(self.__nodes)} nodes)"


#: The model file (the neighbors and density inference lane, 2026-09-15):
#: `X_train` `<f4` (n, d), `L` `<f4` (n, n), `alpha` `<f4` (n) as fitted; the
#: kernel's postfix spec as `kinds` `<i4`, `params` `<f8`, `ls_len` `<i4` and the
#: `ls` table `<f8` (the Python floats `_nodes()` held, so the binary32 arrays
#: predict builds are the fit's); `ints` `<i8` [n_features_in_, info_,
#: normalize_y_, n_ls]; `reals` `<f8` [alpha, _y_train_mean, _y_train_std,
#: log_marginal_likelihood_value_]. `mojolearn.host_model(path)` predicts the
#: mean and std from it on a CPU through the inference-only gp binding, with
#: normalize_y's scale-back in `predict` as on the GPU.
_GP_FORMAT = "mojolearn-gp-1"

# The compiled saved-model owner, connected once at import:
# GaussianProcessRegressor's `save` and `load` are one native call each
# (bindings/archive_schema_native.mojo writes the schema registered below
# the class; host_archive_native.mojo builds the loaded model).
_ARCHIVE, _ARCHIVE_CONTEXT = _native_archive_bridge.connect()
# The compiled owner of the public methods, connected once at import:
# GaussianProcessRegressor's fit, predict, sample_y, log_marginal_likelihood
# and score are one native call each (bindings/gp_regressor_native.mojo).
_NATIVE, _NATIVE_CONTEXT = _native_family_bridge.connect()
_native_family_bridge.connect_bindings("_mojolearn_gp")

#: Each kernel class's postfix node layout, the fields its `_nodes()` reads:
#: ("leaf", kind, param field or None for 0.0, length-scale field or None),
#: ("pair", left, right, op field) or ("nodes", the saved node list). The
#: compiled save reads a fitted kernel through it
#: (bindings/archive_schema_native.mojo "kernel").
_KERNEL_NODES = {
    ConstantKernel: ("leaf", _K_CONST, "constant_value", None),
    WhiteKernel: ("leaf", _K_WHITE, "noise_level", None),
    RBF: ("leaf", _K_RBF, None, "length_scale"),
    Matern: ("leaf", _K_MATERN, "nu", "length_scale"),
    _Combined: ("pair", "a", "b", "op"),
    _SavedKernel: ("nodes", "_SavedKernel__nodes"),
}


# Maintainer notes (moved from the docstring):
# - The scikit-learn surface over `gaussian_process/estimator.mojo`
#   (DEVIATIONS 1750-1771 are the lane's; 1772 is this surface's). No
#   reference GP exists in cuML/cuVS/RAFT; scikit-learn `_gpr.py` is the
#   semantics oracle only.
# - House rule: a parameter accepted and ignored is a wrong answer waiting
#   for a caller, so each parameter is honored or refused by name.
# - alpha IS the Cholesky profile's jitter; there is no second jitter knob
#   (DEVIATION 1751). Passed through UNCLAMPED so the Mojo refusals stay
#   reachable: NaN, negative and +inf refused by name (DEVIATION 1768);
#   any other value is accepted in both modes (the identical two-value pin
#   of DEVIATION 1637 was removed by lane sk-edges, 2026-10-15).
# - DEVIATION 1772: the default alpha is 2**-20, not sklearn's 1e-10. (a)
#   Every kernel matrix this lane builds has a unit diagonal, and in float32
#   1.0 + 1e-10 rounds to exactly 1.0 (the gap above 1.0 is 2**-23), so
#   sklearn's default ridge would be a bitwise no-op here, measured by
#   `check_duplicate_inputs_need_the_ridge` (DEVIATION 1752). (b) 2**-20 is
#   the smallest float32 ridge on a unit diagonal that does anything and is
#   `gaussian_process/estimator.mojo::gp_profile_alpha()`'s value.
# - optimizer: 'fmin_l_bfgs_b' uses the identical gradient (DEVIATION 2880)
#   and a projected L-BFGS whose every rule is pinned (DEVIATION 2881,
#   gaussian_process/gp_optim_items.mojo, one binding call on the device).
#   None stays the default so a fit that ran before runs the same bits.
#   Restarts are Philox keyed by random_state (DEVIATION 2881).
# - normalize_y: sklearn `_gpr.py:275-285`; the two folds are
#   StandardScaler's pinned float32 reductions in this estimator's tier; the
#   un-normalization is `std * mean + y_mean` and `sqrt(var * std**2)`, each
#   one correctly rounded binary32 operation, flushed like `ftz`.
# - Non-finite X/y/length scales are refused on the Mojo host naming the flat
#   index (DEVIATION 1768), before any launch.
# - A failed Cholesky returns LAPACK info instead of raising (DEVIATION 1634);
#   predict refuses it in Mojo, log_marginal_likelihood here with the Mojo
#   entry's own sentences.
# - return_cov / sample_y: DEVIATION 2793. GP classification (Laplace, Newton
#   iteration) is `_gpc_impl.py` (DEVIATION 2830 closes DEVIATION 1766).
# - Cross-vendor record: the IDENTICAL card is byte-identical Apple M4 vs AMD
#   MI325X on every shipped-path line (the 8-line divergence reading was
#   withdrawn at 9835094e: those lines are the sabotaged half of a
#   GP_SAB_STD_EXP pair). The identity harness's 136-lane record at
#   4048e1b51 reads IDENTICAL x3 on every gp training and infer cell across
#   Apple M4, H100 and MI325X.
class GaussianProcessRegressor(NumericModeMixin):
    """Exact Gaussian process regression on the GPU.

    Follows scikit-learn's ``GaussianProcessRegressor``: ``fit`` builds the
    kernel matrix ``K`` of the training points, adds ``alpha`` to its
    diagonal, factors it by Cholesky and solves for the dual coefficients;
    ``predict`` returns the posterior mean and, on request, the standard
    deviation or the full covariance. X is dense float32 in C order (other
    inputs are converted; sparse input is not accepted) and y is 1-D: one
    target per model.

    Differences from scikit-learn that change results:

    - The default ``alpha`` is ``2**-20``, not ``1e-10``. Kernel matrices
      have a unit diagonal, and in float32 ``1.0 + 1e-10`` is exactly
      ``1.0``, so scikit-learn's default would add nothing.
    - The default ``optimizer`` is ``None``: the kernel you pass is used as
      is. Pass ``optimizer="fmin_l_bfgs_b"`` (scikit-learn's default) to fit
      the hyperparameters by maximizing the log marginal likelihood.
    - Kernel matrices are factored in float32. When ``K + alpha * I`` is
      not positive definite there and ``alpha`` is positive, ``fit`` warns
      and refits with ``alpha`` ten times larger, until the factorization
      succeeds or ``alpha`` reaches 1.0.
    - A failed Cholesky factorization does not raise. ``fit`` returns
      normally and records LAPACK's code in ``info_``; check that it is 0.
      A nonzero ``info_`` means this kernel and ridge do not describe the
      data (common causes: duplicate training rows, a length scale far
      larger than the data's spread, ``alpha=0``). ``predict`` and
      ``log_marginal_likelihood`` then raise an error.

    Under ``numeric_mode="identical"`` (the default), fit, predict and
    sample_y give the same bits on Apple, NVIDIA and AMD GPUs.

    Parameters
    ----------
    kernel : Kernel or None, default None
        A combination of `RBF`, `Matern`, `ConstantKernel` and `WhiteKernel`
        with ``+`` and ``*``. None means ``ConstantKernel(1.0, "fixed") *
        RBF(1.0, "fixed")``, scikit-learn's default. Other kernel types,
        including scikit-learn kernel objects, raise ``TypeError``.
    alpha : float, default 2**-20
        Value added to the diagonal of the training kernel matrix (a ridge).
        NaN, negative and infinite values are refused at fit time. Not to
        be confused with the fitted ``alpha_``.
    optimizer : None or "fmin_l_bfgs_b", default None
        None fits the kernel as passed. ``"fmin_l_bfgs_b"`` maximizes the log
        marginal likelihood over the kernel's free hyperparameters (those
        whose bounds are not ``"fixed"``) with a bounded L-BFGS that runs on
        the device. A callable raises ``NotImplementedError``; any other
        value raises ``ValueError``.
    n_restarts_optimizer : int, default 0
        Number of extra optimizer runs from log-uniform random starting
        points within the bounds. The run with the best likelihood wins; a
        tie goes to the earlier run.
    normalize_y : bool, default False
        Center and scale y by its mean and standard deviation before the fit,
        and scale the predictions back, as in scikit-learn. A zero standard
        deviation scales by one.
    copy_X_train : bool, default True
        Only True is accepted: X is always copied to float32 memory.
        False raises ``NotImplementedError``.
    n_targets : None, default None
        Only None is accepted; multi-target models are not supported.
    random_state : int or None, default None
        Seed for the optimizer's restart points, an int in ``[0, 2**64)``.
        Required when ``optimizer`` is set and ``n_restarts_optimizer > 0``.
        `sample_y` takes its own ``random_state``.
    numeric_mode : str or None, default None
        The numeric mode for this estimator; None uses the process default.
        See the numeric modes guide.

    Attributes
    ----------
    X_train_ : Array of shape (n_train, n_features), float32
        The training inputs.
    y_train_ : Array of shape (n_train,), float32
        The training targets, normalized when ``normalize_y=True`` (as in
        scikit-learn).
    kernel_ : Kernel
        The fitted kernel: the kernel you passed when ``optimizer=None``,
        otherwise a copy with the optimized hyperparameters.
    L_ : Array of shape (n_train, n_train), float32
        Lower Cholesky factor of ``K + alpha * I``.
    alpha_ : Array of shape (n_train,), float32
        Dual coefficients ``(K + alpha * I)^-1 y`` (scikit-learn's
        ``alpha_``; not the ``alpha`` ridge parameter).
    log_marginal_likelihood_value_ : float
        Log marginal likelihood of the fitted kernel. Meaningful only when
        ``info_ == 0``.
    info_ : int
        LAPACK's Cholesky code. 0 means ``L_`` is a valid factor; ``k > 0``
        means the leading minor of order ``k`` was not positive definite.
    nb_ : int
        The Cholesky panel width used.
    n_features_in_ : int
        Number of features seen during fit.
    input_copied_ : bool
        Whether fit had to copy X to reach float32 C order.
    clamped_ : Array of shape (n_query,), int32
        Set by ``predict(X, return_std=True)``: a per-point flag marking
        variances that were clamped to zero.
    n_clamped_ : int
        Set by ``predict(X, return_std=True)``: how many variances were
        clamped to zero.

    Examples
    --------
    >>> import numpy as np, mojolearn
    >>> rng = np.random.default_rng(0)
    >>> X = rng.random((200, 3), dtype=np.float32)
    >>> y = np.sin(X.sum(axis=1)).astype(np.float32)
    >>> kernel = (mojolearn.ConstantKernel(1.0) * mojolearn.RBF([1.0, 1.0, 1.0])
    ...           + mojolearn.WhiteKernel(0.01))
    >>> model = mojolearn.GaussianProcessRegressor(kernel, optimizer="fmin_l_bfgs_b").fit(X, y)
    >>> ok = model.info_ == 0
    >>> mean, std = model.predict(X[:5], return_std=True)
    """

    #: scikit-learn's estimator kind: `cross_val_score` stratifies a
    #: classifier's default folds, as scikit-learn's does.
    _estimator_type = "regressor"

    _BINDING = "_mojolearn_gp"

    # The native public owner replaces _mode lifecycle/output decorators.
    # The shared full-reset installer still owns the field-only constructor.
    _mojolearn_native_public = True

    def __init__(
        self,
        kernel=None,
        *,
        alpha=2.0 ** -20,
        optimizer=None,
        n_restarts_optimizer=0,
        normalize_y=False,
        copy_X_train=True,
        n_targets=None,
        random_state=None,
        numeric_mode=None,
    ):
        # Stored verbatim, as scikit-learn requires; `_validate_params`
        # checks them when `fit` (or `load`) runs.
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.n_targets = n_targets
        self.random_state = random_state
        self.numeric_mode = numeric_mode

    # -- the binding, and the tier it really is -----------------------------

    def _extension(self):
        """The `_mojolearn_gp` binding for THIS estimator's tier, with the
        binary's own compile-time answer cross-checked against it --
        `_arima_impl.py`'s pattern, for its reason: a wrong-arm measurement
        that is correctly labelled by accident is the failure the
        three-tier design exists to prevent."""
        mod = self._bind()
        want = getattr(self, "numeric_mode", None) or _backend.default_mode()
        fn = getattr(mod, "gp_numeric_mode", None)
        if fn is not None:
            got = int(fn())
            if got != _MODE_CODE.get(want):
                raise RuntimeError(
                    "mojolearn GaussianProcessRegressor: numeric_mode="
                    f"{want!r} was requested but {mod.__name__} reports "
                    f"compile-time mode code {got}; the binary and the "
                    "directory it sits in disagree, rebuild it with "
                    "bash bindings/build_gp.sh"
                )
        return mod

    # -- fit ----------------------------------------------------------------

    # Reference: sklearn `_gpr.py:349-367`: `K = kernel(X); K[diag] += alpha;
    # L = cholesky(K); alpha_ = cho_solve(L, y)` plus the log marginal
    # likelihood, in one device call (`gpr_fit_host`). DEVIATION 1634: a
    # failed factorization is reported in `info_`, not raised.
    def fit(self, X, y):
        """Fit the Gaussian process to training data.

        With an optimizer, the kernel hyperparameters are fitted first. Then
        ``K + alpha * I`` is factored and the dual coefficients and log
        marginal likelihood are computed on the device.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training inputs, converted to float32.
        y : array-like of shape (n_samples,)
            Training targets. A 2-D y raises ``ValueError``.

        Returns
        -------
        self : GaussianProcessRegressor
            The fitted model. Check ``info_ == 0``: a failed Cholesky
            factorization is reported there rather than raised.
        """
        return _NATIVE(self, "family_gp_gpr_fit", (None, X, y), _NATIVE_CONTEXT)

    # -- the optimizer (DEVIATIONS 2880, 2881) -------------------------------

    # -- predict ------------------------------------------------------------

    # Reference: sklearn `_gpr.py:446-500`. Variance clamp and flags:
    # DEVIATION 1760. A fit with `info_ != 0` is refused in Mojo
    # (`gpr_predict_host`, DEVIATION 1634); `info_` is passed down so that
    # refusal stays reachable. return_cov: DEVIATION 2793.
    def predict(self, X, return_std=False, return_cov=False):
        """Predict with the posterior of the fitted Gaussian process.

        Parameters
        ----------
        X : array-like of shape (n_query, n_features)
            Query points, converted to float32.
        return_std : bool, default False
            Also return the predictive standard deviation. Negative variances
            from rounding are clamped to zero; ``clamped_`` and
            ``n_clamped_`` record which points were clamped.
        return_cov : bool, default False
            Also return the full predictive covariance ``k(X, X) - V^T V``,
            exactly symmetric and not clamped. At most one of
            ``return_std`` and ``return_cov`` may be True (``RuntimeError``
            otherwise).

        Returns
        -------
        mean : Array of shape (n_query,), float32
        std : Array of shape (n_query,), float32
            Only with ``return_std=True``.
        cov : Array of shape (n_query, n_query), float32
            Only with ``return_cov=True``.

        Raises
        ------
        ValueError
            If the model is not fitted (predicting from the unfitted prior is
            not supported), or X has a different number of features than
            the training data.

        Notes
        -----
        A model whose fit failed (``info_ != 0``) refuses to predict. With
        ``normalize_y=True`` the mean, standard deviation and covariance are
        scaled back to the units of y.
        """
        return _NATIVE(self, "family_gp_gpr_predict", (None, X, return_std, return_cov), _NATIVE_CONTEXT)

    # -- saved models -----------------------------------------------------------

    # Format: `_GP_FORMAT`, layout documented above it.
    def save(self, path):
        """Save the fitted model to ``path`` as an ``.npz`` file.

        The file holds everything `predict` reads: the training inputs, the
        Cholesky factor, the dual coefficients, the fitted kernel and the
        numeric mode. Raises ``RuntimeError`` if the model is not fitted.

        Parameters
        ----------
        path : str or path-like
            Destination file.
        """
        return _ARCHIVE(self, "archive_save", (path,), _ARCHIVE_CONTEXT)

    @classmethod
    def load(cls, path):
        """Load a model written by `save`.

        The loaded model predicts the mean and standard deviation. Every
        array is read at its saved dtype and never cast; a file saved by a
        different estimator or with inconsistent contents raises
        ``ValueError``.

        Parameters
        ----------
        path : str or path-like
            File written by `save`.

        Returns
        -------
        GaussianProcessRegressor
        """
        return _ARCHIVE(cls, "host_archive_load", (path,), _ARCHIVE_CONTEXT)

    # -- the rest of sklearn's surface, honored or refused by name ----------

    # Reference: sklearn `_gpr.py:575`; theta arm per DEVIATION 2880; a
    # failed factorization answers -inf as `_gpr.py:593` does.
    def log_marginal_likelihood(self, theta=None, eval_gradient=False, clone_kernel=True):
        """Log marginal likelihood of the training data.

        Parameters
        ----------
        theta : sequence of float or None, default None
            None returns the value computed during fit, and raises
            ``RuntimeError`` if that fit's factorization failed. Otherwise,
            the log of the free hyperparameters of ``kernel_`` (one entry per
            ``kernel_.n_dims``; ``ValueError`` on a length mismatch): the
            likelihood is evaluated at ``exp(theta)`` in float32. A
            factorization that fails at that theta gives ``-inf`` and a zero
            gradient, as in scikit-learn.
        eval_gradient : bool, default False
            Also return the gradient with respect to theta. Requires a
            theta (``ValueError`` otherwise).
        clone_kernel : bool, default True
            Accepted for scikit-learn compatibility; ``kernel_`` is never
            modified.

        Returns
        -------
        log_likelihood : float
        gradient : list of float
            Only with ``eval_gradient=True``.
        """
        return _NATIVE(self, "family_gp_gpr_lml", (None, theta, eval_gradient, clone_kernel), _NATIVE_CONTEXT)

    # DEVIATION 2793 (`gaussian_process/checks/sample_y.mojo`) names the
    # stream, the jitter and the order: the covariance `k(X, X) - V^T V` is
    # factored by the identical Cholesky with its pinned 2^-20 jitter, the
    # normals are position-mapped Philox keyed by random_state (DEVIATION
    # 2791's construction, tag "GPSY"), and the product is the identical GEMM.
    def sample_y(self, X, n_samples=1, random_state=0):
        """Draw samples from the posterior at ``X``.

        Each sample is the predictive mean plus a factor of the predictive
        covariance times standard normals, scaled back to the units of y
        under ``normalize_y``, as in scikit-learn. The covariance is factored
        with a ``2**-20`` jitter on its diagonal. The same model, ``X``,
        ``n_samples`` and ``random_state`` give the same values on every
        call; they are not scikit-learn's values. A covariance that cannot be
        factored raises an error.

        Parameters
        ----------
        X : array-like of shape (n_query, n_features)
            Query points, converted to float32.
        n_samples : int, default 1
            Number of joint draws.
        random_state : int, default 0
            Seed, an int in ``[0, 2**64)``. None and a NumPy ``RandomState``
            raise ``ValueError``.

        Returns
        -------
        Array of shape (n_query, n_samples), float32
            Column ``s`` is one joint draw over the query points.

        Raises
        ------
        ValueError
            If the model is not fitted (sampling the unfitted prior is not
            supported).
        """
        return _NATIVE(self, "family_gp_gpr_sample_y", (None, X, n_samples, random_state), _NATIVE_CONTEXT)

    # Host float64 summary of float32 predictions, outside the identity
    # claim (the SVR.score rule).
    def score(self, X, y):
        """Coefficient of determination R^2 of the predictions.

        Uses scikit-learn's definition, accumulated in float64 from the
        float32 predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test inputs.
        y : array-like of shape (n_samples,)
            True targets.

        Returns
        -------
        float
        """
        return _NATIVE(self, "family_gp_gpr_score", (None, X, y), _NATIVE_CONTEXT)


# The file: see `_GP_FORMAT`. The kernel step comes first: `save` read the
# fitted kernel's nodes before anything else.
_native_archive_bridge.register(GaussianProcessRegressor, _GP_FORMAT, (
    _native_archive_bridge.fitted("alpha_"),
    ("kernel", "kernel_", _KERNEL_NODES, "<f8", "kinds", "params", "ls_len", "ls"),
    *_native_archive_bridge.head(_GP_FORMAT),
    ("array", "X_train", "X_train_"),
    ("flat", "L", "L_"),
    ("array", "alpha", "alpha_"),
    ("i8", "ints", (("int", "n_features_in_"), ("int", "info_"),
                    ("bool_or", "normalize_y_", False), ("kernel_ls", "kernel_", _KERNEL_NODES))),
    ("f8", "reals", (("float", "_alpha"), ("float_or", "_y_train_mean", 0.0),
                     ("float_or", "_y_train_std", 1.0), ("float", "log_marginal_likelihood_value_"))),
), host_archive_gp_saved_kernel=_SavedKernel)


# The fixed tables the native owner reads (bindings/gp_common_native.mojo):
# the kernel classes' node layouts and bound fields, the default kernel's
# classes, the optimizer's stop words, the mode codes, the seed bound and
# StandardScaler with its connected context (normalize_y).
_native_family_bridge.tables(
    family_gp_kernel_base=Kernel,
    family_gp_kernel_nodes=dict(_KERNEL_NODES),
    family_gp_kernel_bounds={ConstantKernel: "constant_value_bounds", WhiteKernel: "noise_level_bounds",
                             RBF: "length_scale_bounds", Matern: "length_scale_bounds"},
    family_gp_classes={"const": ConstantKernel, "rbf": RBF, "combined": _Combined},
    family_gp_opt_stops=_OPT_STOPS,
    family_gp_mode_codes=dict(_MODE_CODE),
    family_gp_seed_bound=2 ** 64,
    family_gp_scaler=(StandardScaler, _STANDARD_CONTEXT),
)

# Import-time native owner identities for constructor reinitialization.
from ._native_owner_lifetime import connect_owner_context as _connect_kernel_owners
_connect_kernel_owners(_NATIVE_CONTEXT)
