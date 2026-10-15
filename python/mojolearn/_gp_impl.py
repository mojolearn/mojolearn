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
import numbers

from . import _backend
from ._array import Array
from ._buffer import addr, addr_ro, as_f32_c, empty

#: `gaussian_process/gp_optim_items.mojo`'s stop words (GP_STOP_*).
_OPT_STOPS = ("running", "pgtol", "no-descent", "line-search", "ftol", "max-iter", "nonfinite-start")
from ._mode import NumericModeMixin
from .linear_model import _flatten, _r2_sums, _round_f32, _shape_of
from .preprocessing import StandardScaler

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


#: FLT_MIN, the smallest normal binary32; `_ftz` flushes anything below it.
_FLT_MIN = 1.1754943508222875e-38


def _gp_py2mojo(ext):
    """True when the binding `ext` applies normalize_y's un-normalization
    itself (lane apple-fast-py2mojo-cluster: the mean, std, covariance and
    sample_y draws, `gaussian_process/unnorm.mojo`). A
    `-D MOJOLEARN_PY2MOJO_cluster_OFF` build answers 0 and the Python loops
    below run (the A/B switch)."""
    fn = getattr(ext, "gp_py2mojo", None)
    return fn is not None and int(fn()) == 1


def _gpr_ptr(ext):
    """True when the binding fits and predicts from the caller's memory on
    the device (`gpr_fit_ptr` / `gpr_predict_ptr`,
    `gaussian_process/gpr_resident.mojo`, lane fam2-kernel-gp): IDENTICAL
    GPU builds. The host binding and a `-D MOJOLEARN_IDN_GPR_PTR_OFF` build
    answer no and `gpr_fit` / `gpr_predict` run with the same lists."""
    fn = getattr(ext, "gp_idn_caps", None)
    return fn is not None and (int(fn()) & 2) == 2


def _ftz(x):
    """`checks/numerics.mojo::ftz` on a binary32 value held in a Python
    float: a subnormal becomes a zero of the same sign."""
    if x != 0.0 and abs(x) < _FLT_MIN:
        return math.copysign(0.0, x)
    return x

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
        return _Combined(self, other, _K_SUM, "+")

    def __mul__(self, other):
        return _Combined(self, other, _K_PROD, "*")

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
        return len(self._free_values())

    @property
    def theta(self):
        """Natural log of the free hyperparameters, as a list of floats.

        On a fitted ``kernel_`` this is the optimizer's own theta.
        """
        fitted = getattr(self, "_theta_fit", None)
        if fitted is not None:
            return list(fitted)
        return [math.log(v) for v in self._free_values()]  # glue: sklearn theta API, log of free kernel hyperparameters (n_dims entries)

    @property
    def bounds(self):
        """Natural log of each free hyperparameter's ``(low, high)`` bounds,
        as a list of ``[low, high]`` pairs."""
        return [[math.log(lo), math.log(hi)] for lo, hi in self._free_bounds()]  # glue: sklearn bounds API, log of kernel hyperparameters (n_dims entries) bounds

    def __repr__(self):
        return self._name()


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
        self.constant_value = float(constant_value)
        self.constant_value_bounds = _as_bounds(constant_value_bounds, 1, "ConstantKernel")

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
        self.noise_level = float(noise_level)
        self.noise_level_bounds = _as_bounds(noise_level_bounds, 1, "WhiteKernel")

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
        self.length_scale = _as_length_scale(length_scale, "RBF")
        self.length_scale_bounds = _as_bounds(length_scale_bounds, len(self.length_scale), "RBF")

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
        self.length_scale = _as_length_scale(length_scale, "Matern")
        self.length_scale_bounds = _as_bounds(length_scale_bounds, len(self.length_scale), "Matern")
        self.nu = float(nu)

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
        if not isinstance(a, Kernel) or not isinstance(b, Kernel):
            raise TypeError(
                "mojolearn gaussian process kernels compose only with each "
                "other; sklearn's magnitude-scaling shorthand "
                "`2.0 * RBF(...)` is spelled ConstantKernel(2.0) * RBF(...) "
                "here, so the constant that runs is one you constructed"
            )
        self.a, self.b, self.op, self.sym = a, b, op, sym

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
        self.__nodes = [(int(k), float(p), [float(v) for v in ls]) for k, p, ls in nodes]  # glue: copies the kernel's postfix nodes

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

    def _validate_params(self):
        """The constructor checks, run first in `fit` and `load`; the
        resolved and converted values go to private attributes."""
        kernel, alpha, optimizer = self.kernel, self.alpha, self.optimizer
        n_restarts_optimizer, normalize_y, copy_X_train = self.n_restarts_optimizer, self.normalize_y, self.copy_X_train
        n_targets, random_state = self.n_targets, self.random_state
        if kernel is None:
            # sklearn's documented default, `_gpr.py:229-231`, bounds
            # "fixed" as there, so the default kernel is never optimized.
            kernel = ConstantKernel(1.0, "fixed") * RBF(1.0, "fixed")
        if not isinstance(kernel, Kernel):
            raise TypeError(
                "mojolearn GaussianProcessRegressor: kernel must be a "
                "composition of mojolearn's RBF, Matern, ConstantKernel and "
                "WhiteKernel (a scikit-learn kernel object carries bounds "
                "and a theta this implementation deliberately has no use for; "
                "DEVIATION 1761)"
            )
        if callable(optimizer):
            raise NotImplementedError(
                "mojolearn GaussianProcessRegressor: a callable optimizer is "
                "refused; it would run caller code on host floats whose "
                "arithmetic no column can pin. optimizer='fmin_l_bfgs_b' runs "
                "the identical bounded quasi-Newton optimizer (DEVIATION 2881)"
            )
        if optimizer is None or (isinstance(optimizer, str) and optimizer.lower() == "none"):
            optimizer = None
        elif optimizer != "fmin_l_bfgs_b":
            raise ValueError(
                f"mojolearn GaussianProcessRegressor: optimizer={optimizer!r} is "
                "not one of None and 'fmin_l_bfgs_b'"
            )
        if isinstance(n_restarts_optimizer, bool) or not isinstance(n_restarts_optimizer, numbers.Integral) \
                or int(n_restarts_optimizer) < 0:
            raise ValueError(
                "mojolearn GaussianProcessRegressor: n_restarts_optimizer must "
                f"be a non-negative int, got {n_restarts_optimizer!r}"
            )
        n_restarts_optimizer = int(n_restarts_optimizer)
        if not isinstance(normalize_y, (bool, int)) or normalize_y not in (0, 1):
            raise TypeError(
                "mojolearn GaussianProcessRegressor: normalize_y must be a bool, "
                f"got {normalize_y!r}"
            )
        if not copy_X_train:
            raise NotImplementedError(
                "mojolearn GaussianProcessRegressor: copy_X_train=False is "
                "refused. sklearn's False stores a REFERENCE to X; this "
                "surface always copies -- X crosses to float32 C-order host "
                "memory and to the device -- so False has nothing it could "
                "mean here, and accepting it would promise a memory saving "
                "that cannot be delivered"
            )
        if n_targets is not None:
            raise NotImplementedError(
                "mojolearn GaussianProcessRegressor: n_targets is refused; "
                "it shapes sample_y's draws from the unfitted prior, which is "
                "not implemented. y is single-target here"
            )
        if random_state is not None and (
                isinstance(random_state, bool) or not isinstance(random_state, numbers.Integral)
                or not 0 <= int(random_state) < 2 ** 64):
            raise ValueError(
                "mojolearn GaussianProcessRegressor: random_state must be None "
                f"or an int in [0, 2**64), got {random_state!r}; it keys the "
                "restart draws (DEVIATION 2881). sample_y takes its own "
                "random_state argument (DEVIATION 2793)"
            )
        if optimizer is not None and n_restarts_optimizer > 0 and random_state is None:
            raise ValueError(
                "mojolearn GaussianProcessRegressor: n_restarts_optimizer > 0 "
                "needs random_state to be an int; the restart starting points "
                "are position-mapped Philox keyed by it (DEVIATION 2881), and "
                "scikit-learn's None (the global NumPy generator) has no "
                "identical counterpart"
            )
        self._kernel = kernel
        # CONVERSION, NOT POLICY (the SVR epsilon rule): a non-number still
        # raises here, and every value that IS a number goes down unclamped
        # so gp_validate_alpha's refusals -- NaN, negative, +inf, and the
        # identical tier's two-value pin -- stay reachable from Python.
        self._alpha = float(alpha)
        self._optimizer = optimizer
        self._n_restarts_optimizer = n_restarts_optimizer
        self._normalize_y = bool(normalize_y)
        self._random_state = None if random_state is None else int(random_state)

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

    def _kernel_arrays(self, kernel=None):
        """The postfix spec as four flat arrays, the shape a binding can
        take (DEVIATION 1756: GPKernelSpec is five parallel host lists; the
        offsets are the one list NOT sent, because the binding MUST rebuild
        the spec through gp_kernel_const/white/rbf/matern and
        gp_kernel_sum/prod -- which recompute them -- so that every
        constructor refusal stays reachable from this surface). The kernel
        is the FITTED `kernel_` once there is one, else `kernel`."""
        if kernel is None:
            kernel = getattr(self, "kernel_", None) or self._kernel
        nodes = kernel._nodes()
        # `Array.from_list(..., '<f4')` rounds each Python float to binary32
        # exactly as `np.array(dtype=np.float32)` did (DEVIATION 2376).
        kinds = Array.from_list([int(k) for k, _, _ in nodes], "<i4")  # glue: packs the kernel's postfix nodes
        params = Array.from_list([float(p) for _, p, _ in nodes], "<f4")  # glue: packs the kernel's postfix nodes
        ls_len = Array.from_list([len(ls) for _, _, ls in nodes], "<i4")  # glue: packs the kernel's postfix nodes
        table = [v for _, _, ls in nodes for v in ls]  # glue: packs the kernel length-scale hyperparameters
        n_ls = len(table)
        # Never a zero-length buffer: mirror estimator.mojo's
        # _length_scale_table -- one unused 1.0 stands in, and n_ls says so.
        ls = Array.from_list(table if table else [1.0], "<f4")
        return kinds, params, ls_len, ls, n_ls

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
        self._validate_params()
        x, self.input_copied_ = as_f32_c(X, ndim=2, name="X")
        n_rows, n_cols = x.shape
        shape = _shape_of(y)
        if len(shape) != 1:
            raise ValueError(
                "mojolearn GaussianProcessRegressor: y must be 1-D, got "
                f"{len(shape)}-D shape {shape}; multi-target GP fits are not "
                "implemented (the n_targets refusal in __init__ is this same "
                "boundary)"
            )
        # No finiteness check here: check the class table -- non-finite
        # cells are refused BY NAME on the Mojo host (DEVIATION 1768), and
        # a duplicate check here would make that refusal unreachable.
        targets, _ = as_f32_c(y, ndim=1, name="y")
        if targets.shape[0] != n_rows:
            raise ValueError(
                f"mojolearn GaussianProcessRegressor: y has {targets.shape[0]} "
                f"entries, X has {n_rows} rows"
            )
        if self._normalize_y:
            # sklearn `_gpr.py:275-280`. StandardScaler's pinned folds in
            # this estimator's tier; for 1-D y scikit-learn's
            # `_handle_zeros_in_scale` takes its scalar arm (`std == 0 -> 1`),
            # StandardScaler's exact-zero rule.
            column = targets.reshape((n_rows, 1))
            scaler = StandardScaler(numeric_mode=self.numeric_mode_used()).fit(column)
            y_mean = float(scaler.mean_[0])
            y_std = float(scaler.scale_[0])
            targets = scaler.transform(column).reshape((n_rows,))
        else:
            y_mean, y_std = 0.0, 1.0
        kernel_ = self._kernel
        self._optimizer_runs = []
        if self._optimizer is not None and self._kernel.n_dims > 0:
            kernel_ = self._optimize(x, targets, n_rows, n_cols)
        kinds, kparams, ls_len, ls, n_ls = self._kernel_arrays(kernel_)

        # EVERY SIZE IS A FUNCTION OF (n_rows, n_cols) ALONE; nothing here
        # is a worst-case buffer (contrast SVR's DEVIATION 873 note).
        l_out = empty((n_rows * n_rows,), "<f4")
        dual = empty((n_rows,), "<f4")
        # info, nb, logdet, ydotalpha, lml -- in that order.
        scalars = empty((5,), "<f8")
        # TWO LISTS, NOT FOURTEEN ARGUMENTS (the module header's ABI note:
        # the positional spelling failed def_function elaboration, measured
        # 2026-09-01). Every array addressed below is bound to a local in
        # this frame -- x, targets, kinds, kparams, ls_len, ls, l_out,
        # dual, scalars -- which is what keeps the addresses alive.
        ext = self._extension()
        fit_fn = ext.gpr_fit_ptr if _gpr_ptr(ext) else ext.gpr_fit
        while True:
            info = fit_fn(
                # ORDER MATCHES bindings/_mojolearn_gp.mojo::gpr_fit_binding.
                # x, y, kinds, kparams, ls_len, ls, l_out, dual_out, scalars_out
                [
                    addr_ro(x, name="x"),
                    addr_ro(targets, name="targets"),
                    addr_ro(kinds, name="kinds"),
                    addr_ro(kparams, name="kparams"),
                    addr_ro(ls_len, name="ls_len"),
                    addr_ro(ls, name="ls"),
                    addr(l_out, name="l_out"),
                    addr(dual, name="dual"),
                    addr(scalars, name="scalars"),
                ],
                # ORDER MATCHES bindings/_mojolearn_gp.mojo::gpr_fit_binding.
                # n_train, n_features, n_nodes, n_ls, alpha
                [n_rows, n_cols, int(kinds.shape[0]), n_ls, self._alpha],
            )
            # A ridge too small for float32 (scikit-learn factors in float64,
            # where its 1e-10 suffices): K + alpha*I is not positive definite
            # here. A positive alpha is raised tenfold and the fit repeated
            # until the factor exists or the ridge reaches the kernel's unit
            # diagonal, with a warning naming the ridge used (control flow
            # on the binding's status word; the factor is the binding's, so
            # every vendor takes the same steps). alpha=0 asks for no ridge
            # and keeps the failed fit's info_.
            if int(info) == 0 or not self._alpha > 0.0 or self._alpha >= 1.0:
                break
            asked = self._alpha
            self._alpha = min(asked * 10.0, 1.0)
            import warnings
            warnings.warn(
                f"mojolearn GaussianProcessRegressor: K + alpha*I is not positive "
                f"definite in float32 with alpha={asked:g}; refitting with "
                f"alpha={self._alpha:g}. Pass a larger alpha to skip this step.",
                RuntimeWarning,
            )
        self.X_train_ = x
        self.y_train_ = targets
        self._y_train_mean = y_mean
        self._y_train_std = y_std
        self.normalize_y_ = bool(self._normalize_y)
        self.kernel_ = kernel_
        self.n_features_in_ = n_cols
        self.L_ = l_out.reshape((n_rows, n_rows))
        self.alpha_ = dual
        self.info_ = int(info)
        self.nb_ = int(scalars[1])
        self.log_marginal_likelihood_value_ = float(scalars[4])
        self._logdet = float(scalars[2])
        self._ydotalpha = float(scalars[3])
        return self

    # -- the optimizer (DEVIATIONS 2880, 2881) -------------------------------

    def _lml_grad(self, kernel, x, targets, n_rows, n_cols):
        """`(info, lml, grad)` at `kernel` through `gpr_lml_grad`: the float32
        likelihood and gradient over the kernel's free entries, widened."""
        kinds, kparams, ls_len, ls, n_ls = self._kernel_arrays(kernel)
        free = Array.from_list(kernel._free_flags(), "<i4")
        n_free = kernel.n_dims
        grad = empty((max(n_free, 1),), "<f8")
        scalars = empty((2,), "<f8")
        info = self._extension().gpr_lml_grad(
            # ORDER MATCHES bindings/_mojolearn_gp.mojo::gpr_lml_grad_binding.
            # x, y, kinds, kparams, ls_len, ls, free, grad_out, scalars_out
            [addr_ro(x, name="x"), addr_ro(targets, name="targets"), addr_ro(kinds, name="kinds"),
             addr_ro(kparams, name="kparams"), addr_ro(ls_len, name="ls_len"), addr_ro(ls, name="ls"),
             addr_ro(free, name="free"), addr(grad, name="grad"), addr(scalars, name="scalars")],
            # n_train, n_features, n_nodes, n_ls, alpha
            [n_rows, n_cols, int(kinds.shape[0]), n_ls, self._alpha],
        )
        return int(info), float(scalars[1]), [float(grad[i]) for i in range(n_free)]  # glue: reads the gradient of kernel hyperparameters (n_dims entries)

    def _optimize(self, x, targets, n_rows, n_cols):
        """scikit-learn `_gpr.py:299-341`: optimize from the kernel's theta,
        then from `n_restarts_optimizer` log-uniform starts; the smallest
        negative likelihood wins and a tie goes to the earlier run
        (`np.argmin`). ONE binding call (`gpr_optimize`) runs every start on
        the device (DEVIATION 2881, `gaussian_process/gp_optim_items.mojo`);
        Python only hands the kernel over and reads the answer. Returns the
        fitted kernel, whose `theta` is the winning run's. Every run is
        recorded in `_optimizer_runs` as `(n_iter, n_eval, stop, -f)`."""
        base = self._kernel
        kinds, kparams, ls_len, ls, n_ls = self._kernel_arrays(base)
        free = Array.from_list(base._free_flags(), "<i4")
        n_theta = base.n_dims
        bounds = Array.from_list([float(v) for b in base._free_bounds() for v in (b[0], b[1])], "<f4")  # glue: packs bounds of free kernel hyperparameters (n_dims entries)
        n_runs = 1 + int(self._n_restarts_optimizer)
        theta_out = empty((n_theta,), "<f8")
        values_out = empty((n_theta,), "<f8")
        runs_out = empty((n_runs * 4,), "<f8")
        # random_state may be None only with no restarts (checked in __init__);
        # the kernel then draws no restart point and ignores the seed.
        seed = 0 if self._random_state is None else int(self._random_state)
        self._extension().gpr_optimize(
            # ORDER MATCHES bindings/_mojolearn_gp.mojo::gpr_optimize_binding.
            # x, y, kinds, kparams, ls_len, ls, free, bounds, theta_out,
            # values_out, runs_out
            [addr_ro(x, name="x"), addr_ro(targets, name="targets"), addr_ro(kinds, name="kinds"),
             addr_ro(kparams, name="kparams"), addr_ro(ls_len, name="ls_len"), addr_ro(ls, name="ls"),
             addr_ro(free, name="free"), addr_ro(bounds, name="bounds"), addr(theta_out, name="theta_out"),
             addr(values_out, name="values_out"), addr(runs_out, name="runs_out")],
            # n_train, n_features, n_nodes, n_ls, alpha, n_restarts, seed_lo, seed_hi
            [n_rows, n_cols, int(kinds.shape[0]), n_ls, self._alpha, n_runs - 1,
             seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF],
        )
        for r in range(n_runs):  # glue: one record per optimizer restart
            n_iter, n_eval, stop, f = (float(runs_out[r * 4 + j]) for j in range(4))  # glue: four fields per optimizer run
            self._optimizer_runs.append((int(n_iter), int(n_eval), _OPT_STOPS[int(stop)], -f))
        theta = [float(theta_out[i]) for i in range(n_theta)]  # glue: reads the fitted theta, kernel hyperparameters (n_dims entries)
        params = [float(values_out[i]) for i in range(n_theta)]  # glue: reads fitted kernel hyperparameters (n_dims entries)
        return base._with_free_values(params, theta=theta)

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
        if return_cov and return_std:
            raise RuntimeError(
                "mojolearn GaussianProcessRegressor: at most one of return_std "
                "or return_cov can be requested (scikit-learn's rule)"
            )
        if return_cov:
            return self._predict_cov(X)
        if not hasattr(self, "alpha_"):
            raise ValueError(
                "mojolearn GaussianProcessRegressor: call fit() first (the "
                "unfitted-prior arm of sklearn's predict is not implemented)"
            )
        q, _ = as_f32_c(X, ndim=2, name="X")
        if q.shape[1] != self.n_features_in_:
            raise ValueError(
                f"mojolearn GaussianProcessRegressor: X has {q.shape[1]} "
                f"features, fit saw {self.n_features_in_}"
            )
        n_star = q.shape[0]
        n_train = self.X_train_.shape[0]
        kinds, kparams, ls_len, ls, n_ls = self._kernel_arrays()

        mean = empty((n_star,), "<f4")
        var = empty((n_star,), "<f4")
        std = empty((n_star,), "<f4")
        clamped = empty((n_star,), "<i4")
        # Kept in locals so the arrays outlive the call; the Mojo side
        # borrows these addresses and owns nothing, and an address inside
        # the list below keeps nothing alive on its own (_buffer.py). Two
        # lists, not fifteen arguments -- the module header's ABI note.
        xt = self.X_train_
        lf = self.L_
        dual = self.alpha_
        ext = self._extension()
        norm = bool(getattr(self, "normalize_y_", False))
        # the binding un-normalizes (every build; the Python fallback was
        # removed by lane fam2-kernel-gp)
        in_mojo = norm
        tail = [1, float(self._y_train_std), float(self._y_train_mean)] if in_mojo else []
        n_clamped = (ext.gpr_predict_ptr if _gpr_ptr(ext) else ext.gpr_predict)(
            # ORDER MATCHES bindings/_mojolearn_gp.mojo::gpr_predict_binding.
            # xtrain, l, dual, xstar, kinds, kparams, ls_len, ls,
            # mean_out, var_out, std_out, clamped_out
            [
                addr_ro(xt, name="xt"),
                addr_ro(lf, name="lf"),
                addr_ro(dual, name="dual"),
                addr_ro(q, name="q"),
                addr_ro(kinds, name="kinds"),
                addr_ro(kparams, name="kparams"),
                addr_ro(ls_len, name="ls_len"),
                addr_ro(ls, name="ls"),
                addr(mean, name="mean"),
                addr(var, name="var"),
                addr(std, name="std"),
                addr(clamped, name="clamped"),
            ],
            # ORDER MATCHES bindings/_mojolearn_gp.mojo::gpr_predict_binding.
            # n_train, n_features, n_star, n_nodes, n_ls, return_std, info
            [n_train, self.n_features_in_, n_star, int(kinds.shape[0]),
             n_ls, 1 if return_std else 0, self.info_] + tail,
        )
        if return_std:
            self.clamped_ = clamped
            self.n_clamped_ = int(n_clamped)
            return mean, std
        return mean

    def _predict_cov(self, X):
        """`predict(X, return_cov=True)`, `_gpr.py:495-506`: `(mean, cov)`,
        float32 `(n,)` and `(n, n)`. `cov = k(X, X) - V^T V` in the steps
        `sample_y` takes before its factorization (DEVIATION 2793:
        `gpr_predict_cov_host`, `gpr_host_predict_cov`), exactly symmetric,
        not clamped; under `normalize_y` the mean is un-normalized as
        `predict`'s and each covariance cell is scaled by `std**2`, one
        rounding each, on the host."""
        if not hasattr(self, "alpha_"):
            raise ValueError(
                "mojolearn GaussianProcessRegressor: call fit() first (the "
                "unfitted-prior arm of sklearn's predict is not implemented)"
            )
        q, _ = as_f32_c(X, ndim=2, name="X")
        if q.shape[1] != self.n_features_in_:
            raise ValueError(
                f"mojolearn GaussianProcessRegressor: X has {q.shape[1]} "
                f"features, fit saw {self.n_features_in_}"
            )
        n_star = q.shape[0]
        n_train = self.X_train_.shape[0]
        kinds, kparams, ls_len, ls, n_ls = self._kernel_arrays()
        ext = self._extension()
        if not callable(getattr(ext, "gpr_predict_cov", None)):
            raise ImportError(
                f"mojolearn GaussianProcessRegressor: {ext.__name__} does not "
                "export gpr_predict_cov, so return_cov=True cannot run on it; "
                "rebuild the GP binding"
            )
        mean = empty((max(n_star, 1),), "<f4")
        cov = empty((max(n_star * n_star, 1),), "<f4")
        norm = bool(getattr(self, "normalize_y_", False))
        # the binding un-normalizes (every build; the Python fallback was
        # removed by lane fam2-kernel-gp)
        in_mojo = norm
        tail = [1, float(self._y_train_std), float(self._y_train_mean)] if in_mojo else []
        xt = self.X_train_
        lf = self.L_
        dual = self.alpha_
        ext.gpr_predict_cov(
            # ORDER MATCHES bindings/_mojolearn_gp.mojo::gpr_predict_cov_binding.
            # xtrain, l, dual, xstar, kinds, kparams, ls_len, ls, mean_out, cov_out
            [
                addr_ro(xt, name="xt"),
                addr_ro(lf, name="lf"),
                addr_ro(dual, name="dual"),
                addr_ro(q, name="q"),
                addr_ro(kinds, name="kinds"),
                addr_ro(kparams, name="kparams"),
                addr_ro(ls_len, name="ls_len"),
                addr_ro(ls, name="ls"),
                addr(mean, name="mean"),
                addr(cov, name="cov"),
            ],
            # n_train, n_features, n_star, n_nodes, n_ls, info
            # (+ normalize_y, y_std, y_mean: the binding un-normalizes)
            [n_train, self.n_features_in_, n_star, int(kinds.shape[0]), n_ls, self.info_] + tail,
        )
        return mean[:n_star], cov[:n_star * n_star].reshape((n_star, n_star))

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
        if not hasattr(self, "alpha_"):
            raise RuntimeError("this estimator is not fitted yet")
        from . import _serialize
        from .linear_model import _saved_mode
        nodes = self.kernel_._nodes()
        table = [float(v) for _, _, ls in nodes for v in ls]  # glue: packs the kernel length-scale hyperparameters
        n = int(self.X_train_.shape[0])
        return _serialize.write_npz(path, {
            "format": _GP_FORMAT,
            "estimator": type(self).__name__,
            "numeric_mode": _saved_mode(self),
            "X_train": self.X_train_,
            "L": self.L_.reshape((n * n,)),
            "alpha": self.alpha_,
            "kinds": Array.from_list([int(k) for k, _, _ in nodes], "<i4"),  # glue: packs the kernel's postfix nodes
            "params": Array.from_list([float(p) for _, p, _ in nodes], "<f8"),  # glue: packs the kernel's postfix nodes
            "ls_len": Array.from_list([len(ls) for _, _, ls in nodes], "<i4"),  # glue: packs the kernel's postfix nodes
            "ls": Array.from_list(table if table else [1.0], "<f8"),
            "ints": Array.from_list([int(self.n_features_in_), int(self.info_),
                                     1 if getattr(self, "normalize_y_", False) else 0, len(table)], "<i8"),
            "reals": Array.from_list([float(self._alpha), float(getattr(self, "_y_train_mean", 0.0)),
                                      float(getattr(self, "_y_train_std", 1.0)),
                                      float(self.log_marginal_likelihood_value_)], "<f8"),
        })

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
        from . import _serialize
        from .linear_model import _check_saved_by, _restore_mode
        arrays = _serialize.read_npz(path, _GP_FORMAT)
        _check_saved_by(arrays, path, cls)
        ints = _serialize.exact(arrays, "ints", "<i8")
        reals = _serialize.exact(arrays, "reals", "<f8")
        if ints.size != 4 or reals.size != 4:
            raise ValueError(f"mojolearn: {path!r} holds {ints.size} ints and {reals.size} reals, 4 and 4 are needed")
        d, info, norm, n_ls = (int(ints[i]) for i in range(4))  # glue: four saved meta fields
        x = _serialize.exact(arrays, "X_train", "<f4")
        if x.ndim != 2 or x.shape[1] != d or x.shape[0] < 1:
            raise ValueError(f"mojolearn: {path!r} X_train has shape {tuple(x.shape)}, ints say {d} features")
        n = int(x.shape[0])
        l_flat = _serialize.exact(arrays, "L", "<f4")
        dual = _serialize.exact(arrays, "alpha", "<f4")
        if l_flat.size != n * n or dual.ndim != 1 or dual.size != n:
            raise ValueError(f"mojolearn: {path!r} L or alpha does not match {n} training rows")
        kinds = _serialize.exact(arrays, "kinds", "<i4")
        params = _serialize.exact(arrays, "params", "<f8")
        ls_len = _serialize.exact(arrays, "ls_len", "<i4")
        ls = _serialize.exact(arrays, "ls", "<f8")
        if not (kinds.size == params.size == ls_len.size) or kinds.size < 1:
            raise ValueError(f"mojolearn: {path!r} carries a malformed kernel spec")
        lens = [int(ls_len[i]) for i in range(ls_len.size)]  # glue: saved length-scale count per node
        if sum(lens) != n_ls or ls.size != max(n_ls, 1):  # glue: checks the saved kernel node lengths
            raise ValueError(f"mojolearn: {path!r} length scale table does not match its node lengths")
        nodes, off = [], 0
        for i in range(kinds.size):  # glue: rebuilds the kernel's postfix nodes
            nodes.append((int(kinds[i]), float(params[i]), [float(ls[off + j]) for j in range(lens[i])]))  # glue: rebuilds the kernel's postfix nodes
            off += lens[i]
        obj = cls(kernel=_SavedKernel(nodes), alpha=float(reals[0]), normalize_y=bool(norm))
        obj._validate_params()
        _restore_mode(obj, arrays)
        obj.X_train_ = x
        obj.kernel_ = obj._kernel
        obj.n_features_in_ = d
        obj.L_ = l_flat.reshape((n, n))
        obj.alpha_ = dual
        obj.info_ = info
        obj.normalize_y_ = bool(norm)
        obj._y_train_mean = float(reals[1])
        obj._y_train_std = float(reals[2])
        obj.log_marginal_likelihood_value_ = float(reals[3])
        return obj

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
        if not hasattr(self, "alpha_"):
            raise ValueError(
                "mojolearn GaussianProcessRegressor: call fit() first"
            )
        if theta is not None:
            if eval_gradient not in (True, False):
                raise TypeError("mojolearn GaussianProcessRegressor: eval_gradient must be a bool")
            kern = self.kernel_
            theta = [float(t) for t in _flatten(theta)] if _shape_of(theta) else [float(theta)]  # glue: converts the theta argument, kernel hyperparameters (n_dims entries)
            if len(theta) != kern.n_dims:
                raise ValueError(
                    f"mojolearn GaussianProcessRegressor: theta has {len(theta)} entries, "
                    f"kernel_ has {kern.n_dims} free hyperparameters")
            ext = self._extension()
            params = [float(v) for v in ext.gp_theta_params(theta)]  # glue: reads the binding's kernel hyperparameters (n_dims entries)
            n_rows, n_cols = self.X_train_.shape
            info, lml, grad = self._lml_grad(kern._with_free_values(params), self.X_train_,
                                             self.y_train_, n_rows, n_cols)
            if info != 0:
                lml, grad = -math.inf, [0.0] * len(theta)
            return (lml, grad) if eval_gradient else lml
        if eval_gradient:
            raise ValueError(
                "mojolearn GaussianProcessRegressor: eval_gradient=True needs a theta, "
                "scikit-learn's rule (_gpr.py:582)"
            )
        if self.info_ != 0:
            # gpr_log_marginal_likelihood's refusal, restated: this is
            # the one Mojo guard the binding cannot serve (fit already
            # returned), so its sentences live here too. DEVIATION 1634.
            raise RuntimeError(
                "mojolearn GaussianProcessRegressor: the factorization "
                f"failed (info={self.info_}), so the leading minor of order "
                f"{self.info_} of K + alpha*I was not positive definite and "
                "there is no marginal likelihood to report. For a Gaussian "
                "process that is a RESULT and not a bug: this kernel and "
                "this ridge do not describe these points. The usual causes "
                "are duplicate or near-duplicate training rows, a length "
                "scale far larger than the spread of the data, and alpha=0. "
                "sklearn's optimizer answers -inf here (_gpr.py:593) "
                "because it is comparing candidates; there is no optimizer "
                "in this lane (DEVIATION 1761), so this refuses instead. "
                "DEVIATION 1634"
            )
        return self.log_marginal_likelihood_value_

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
        if not hasattr(self, "alpha_"):
            raise ValueError(
                "mojolearn GaussianProcessRegressor: call fit() first (sampling "
                "the unfitted prior is not implemented)"
            )
        if isinstance(n_samples, bool) or not isinstance(n_samples, numbers.Integral):
            raise TypeError("mojolearn GaussianProcessRegressor: n_samples must be an int")
        seed = random_state
        if (isinstance(seed, bool) or not isinstance(seed, numbers.Integral)
                or not 0 <= int(seed) < 2 ** 64):
            raise ValueError(
                "mojolearn GaussianProcessRegressor: sample_y needs random_state to "
                f"be an int in [0, 2**64), got {seed!r}; it is the key of every "
                "draw (DEVIATION 2793)"
            )
        seed = int(seed)
        n = int(n_samples)
        q, _ = as_f32_c(X, ndim=2, name="X")
        if q.shape[1] != self.n_features_in_:
            raise ValueError(
                f"mojolearn GaussianProcessRegressor: X has {q.shape[1]} "
                f"features, fit saw {self.n_features_in_}"
            )
        n_star = q.shape[0]
        n_train = self.X_train_.shape[0]
        kinds, kparams, ls_len, ls, n_ls = self._kernel_arrays()
        # Never a zero-length buffer: a refused n still addresses real memory.
        out = empty((max(n_star * n, 1),), "<f4")
        xt = self.X_train_
        lf = self.L_
        dual = self.alpha_
        ext = self._extension()
        norm = bool(getattr(self, "normalize_y_", False))
        # the binding un-normalizes (every build; the Python fallback was
        # removed by lane fam2-kernel-gp)
        in_mojo = norm
        tail = [1, float(self._y_train_std), float(self._y_train_mean)] if in_mojo else []
        ext.gpr_sample_y(
            # ORDER MATCHES bindings/_mojolearn_gp.mojo::gpr_sample_y_binding.
            # xtrain, l, dual, xstar, kinds, kparams, ls_len, ls, y_out
            [
                addr_ro(xt, name="xt"),
                addr_ro(lf, name="lf"),
                addr_ro(dual, name="dual"),
                addr_ro(q, name="q"),
                addr_ro(kinds, name="kinds"),
                addr_ro(kparams, name="kparams"),
                addr_ro(ls_len, name="ls_len"),
                addr_ro(ls, name="ls"),
                addr(out, name="y_out"),
            ],
            # n_train, n_features, n_star, n_nodes, n_ls, info, n_samples,
            # random_state low 32 bits, random_state high 32 bits
            [n_train, self.n_features_in_, n_star, int(kinds.shape[0]), n_ls,
             self.info_, n, seed & 0xFFFFFFFF, seed >> 32] + tail,
        )
        return out.reshape((n_star, n))

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
        pred = self.predict(X)
        shape = _shape_of(y)
        if len(shape) != 1 or shape[0] != pred.shape[0]:
            raise ValueError(
                "mojolearn GaussianProcessRegressor: y must be 1-D with one "
                f"entry per row of X, got shape {shape} against "
                f"{pred.shape[0]} predictions"
            )
        # Sequential Python float64 (`linear_model._r2_sums`, DEVIATION
        # 2365): a host reduction outside the identity claim.
        ss_res, ss_tot = _r2_sums(pred, y)
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - ss_res / ss_tot
