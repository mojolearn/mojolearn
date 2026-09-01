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
`gaussian_process/README.md`. With the sole blocker withdrawn, Andrew
delegated the exposure decision and the orchestrator took it: expose.

**THERE IS NO UPSTREAM GAUSSIAN PROCESS.** cuML, cuVS and RAFT implement
none at the pinned commits (`gaussian_process/DERIVATION_MAP.tsv` carries
the grep), so this lane is ORIGINAL WORK per `ok-to-add-capability`.
scikit-learn's `sklearn/gaussian_process/_gpr.py` is the SEMANTICS
reference and the oracle, never the design source; every name this surface
shares with scikit-learn means what scikit-learn means by it.

THE BINDING THIS FILE CALLS IS `_mojolearn_gp`, WHICH IS OWED. As of this
file's landing, `bindings/_mojolearn_gp.mojo` and `bindings/build_gp.sh`
do not exist yet: the Mojo lane's host entries (`gaussian_process/
estimator.mojo::gpr_fit_host` / `gpr_predict_host`) have no caller in
`bindings/`. That is `_backend.py`'s designed-for state -- the package
imports fine and this class raises BY NAME with the build command on
first use -- and it is the same state `KNeighbors*`, `SVR` and `ARIMA`
each passed through on their exposure day. The binding contract (function
names, argument order, and that the kernel spec MUST be rebuilt through
`gaussian_process/checks/kernels.mojo`'s constructors so their refusals
stay reachable) is written at each call site below.

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
(two vendors, at `9835094e`'s reading of the 2026-08-28 cards). An NVIDIA
H100 compiled and ran the lane under FAST on 2026-08-26, which is a speed
leg, not an identity one, so there is NO identity card on a third column
and this module does not claim one. The gp SPEED ladder is UNRUN
(HANDOFF_2026-09-01.md section 5): exposure is a correctness claim, not a
speed claim.
"""

import numpy as np

from . import _backend
from ._mode import NumericModeMixin
from ._arrays import _addr, _addr_ro, as_f32_c

#: `checks/numerics.mojo` codes, duplicated from `_backend._MODE_CODE` on
#: purpose, for `_arima_impl.py`'s reason: the read-back must not share a
#: table with the thing it checks.
_MODE_CODE = {"fast": 0, "identical": 1, "deterministic": 2}

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
    if np.ndim(length_scale) == 0:
        return [float(length_scale)]
    ls = [float(v) for v in np.asarray(length_scale).ravel()]
    if len(ls) == 0:
        raise ValueError(
            f"mojolearn {what}: length_scale is an empty sequence; pass a "
            "scalar (isotropic) or n_features values (ARD)"
        )
    return ls


class Kernel:
    """Base of the four kernel spec builders. `k1 + k2` is sklearn's `Sum`
    and `k1 * k2` its `Product`, postfix `a b +` / `a b *`; NOTHING is
    distributed or reassociated (DEVIATION 1756)."""

    def __add__(self, other):
        return _Combined(self, other, _K_SUM, "+")

    def __mul__(self, other):
        return _Combined(self, other, _K_PROD, "*")

    def _nodes(self):
        """Postfix list of `(kind, param, length_scales)` tuples."""
        raise NotImplementedError

    def __repr__(self):
        return self._name()


class ConstantKernel(Kernel):
    """`ConstantKernel(constant_value)`, sklearn `kernels.py:1187`.
    `k(x, y) = constant_value` for every pair."""

    def __init__(self, constant_value=1.0):
        self.constant_value = float(constant_value)

    def _nodes(self):
        return [(_K_CONST, self.constant_value, [])]

    def _name(self):
        return f"ConstantKernel({self.constant_value!r})"


class WhiteKernel(Kernel):
    """`WhiteKernel(noise_level)`, sklearn `kernels.py:1325`. Noise on the
    training diagonal ONLY; a cross-covariance gets ZERO from it, sklearn's
    own structural rule (DEVIATION 1762)."""

    def __init__(self, noise_level=1.0):
        self.noise_level = float(noise_level)

    def _nodes(self):
        return [(_K_WHITE, self.noise_level, [])]

    def _name(self):
        return f"WhiteKernel({self.noise_level!r})"


class RBF(Kernel):
    """`RBF(length_scale)`, sklearn `kernels.py:1448`. One entry is the
    isotropic case, `n_features` entries the ARD (anisotropic) one; there
    is no third spelling."""

    def __init__(self, length_scale=1.0):
        self.length_scale = _as_length_scale(length_scale, "RBF")

    def _nodes(self):
        return [(_K_RBF, 0.0, list(self.length_scale))]

    def _name(self):
        return f"RBF({self.length_scale!r})"


class Matern(Kernel):
    """`Matern(length_scale, nu)`, sklearn `kernels.py:1601`. Only the
    three closed forms run -- `nu` in {0.5, 1.5, 2.5} -- and every other
    value, `nu = inf` included, is refused BY NAME by `gp_kernel_matern`
    at fit time with the closure condition (DEVIATION 1765). `nu` is not
    judged here so that refusal stays reachable."""

    def __init__(self, length_scale=1.0, nu=1.5):
        self.length_scale = _as_length_scale(length_scale, "Matern")
        self.nu = float(nu)

    def _nodes(self):
        return [(_K_MATERN, self.nu, list(self.length_scale))]

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

    def _name(self):
        return f"({self.a._name()} {self.sym} {self.b._name()})"


class GaussianProcessRegressor(NumericModeMixin):
    """Exact dense GP regression on the GPU, the scikit-learn surface over
    `gaussian_process/estimator.mojo` (DEVIATIONS 1750-1771 are the
    lane's; 1772 below is this surface's).

    WHAT IS HONORED, WHAT IS REFUSED, AND WHY -- one line per parameter,
    because a parameter that is accepted and ignored is a wrong answer
    waiting for a caller (the house rule):

        kernel          honored   a composition of RBF, Matern (nu in
                                  {0.5, 1.5, 2.5}), ConstantKernel and
                                  WhiteKernel via + and *. None means
                                  sklearn's documented default,
                                  ConstantKernel(1.0) * RBF(1.0).
                                  DotProduct, RationalQuadratic,
                                  ExpSineSquared and the general Matern
                                  are unported
                                  (gaussian_process/NOT_IMPLEMENTED.tsv)
        alpha           honored   the RIDGE added to the training diagonal,
                                  which IS the Cholesky profile's jitter --
                                  there is no second jitter knob (DEVIATION
                                  1751). Passed through UNCLAMPED so the
                                  Mojo refusals stay reachable: NaN, negative
                                  and +inf are refused by name (DEVIATION
                                  1768), and under numeric_mode='identical'
                                  only +0.0 and 2**-20 are accepted, BY NAME
                                  (DEVIATION 1637). THE DEFAULT IS 2**-20,
                                  NOT sklearn's 1e-10 -- see DEVIATION 1772
                                  below
        optimizer       honored   only None (also spelled 'none'). sklearn's
                                  default 'fmin_l_bfgs_b' is REFUSED: an
                                  optimizer's iteration count is data
                                  dependent, so the convergence test is part
                                  of the arithmetic and nothing identical
                                  exists to run (DEVIATION 1761). The fitted
                                  kernel_ is therefore the kernel you passed
        n_restarts_     refused   anything but 0; it exists to serve the
          optimizer               refused optimizer
        normalize_y     refused   anything truthy. NOT PORTED: it centers
                                  and scales y with host reductions whose
                                  last bits would sit inside a cross-vendor
                                  identity claim (DEVIATION 1764 names it)
        copy_X_train    refused   anything but True. This surface ALWAYS
                                  copies: X crosses to float32 C-order host
                                  memory and then to the device, so the
                                  False spelling ("store a reference") has
                                  nothing it could mean here
        n_targets       refused   anything but None; it exists to shape
                                  sample_y's prior draws, and sample_y is
                                  unported
        random_state    refused   anything but None; only sample_y draws
                                  random numbers upstream, and sample_y is
                                  unported
        sparse X        refused   dense row-major float32 only
                                  (_arrays.py::as_f32_c)
        2-D y           refused   by name; multi-target GP fits are not
                                  ported (single-target only)
        non-finite X/y  refused   on the HOST in Mojo, naming the flat index
        /length scales            (DEVIATION 1768), before any launch

    DEVIATION 1772: THE DEFAULT `alpha` IS `2**-20`, NOT sklearn's `1e-10`.
    Two facts force it. (a) Every kernel matrix this lane builds has a unit
    diagonal, and in float32 `1.0 + 1e-10` rounds to exactly `1.0` -- the
    gap above 1.0 is `2**-23`, three orders of magnitude above the ridge --
    so sklearn's default ridge would be a bitwise NO-OP here, measured by
    `check_duplicate_inputs_need_the_ridge` (DEVIATION 1752). (b) Under
    `identical` the Cholesky profile accepts exactly `+0.0` and `2**-20`
    and refuses anything else by name (DEVIATION 1637). `2**-20` is the
    smallest float32 ridge on a unit diagonal that can do anything, it is
    `gaussian_process/estimator.mojo::gp_profile_alpha()`'s value, and a
    default that silently did nothing is exactly the accepted-and-ignored
    parameter this surface's table exists to prevent.

    A FAILED FACTORIZATION IS A RESULT, NOT AN EXCEPTION -- READ `info_`.
    sklearn raises LinAlgError out of `fit` when the Cholesky fails; this
    lane returns LAPACK's `info` instead (DEVIATION 1634), because for a
    Gaussian process it is an ANSWER: this kernel and this ridge do not
    describe these points (the usual causes are duplicate rows, a length
    scale far larger than the data's spread, and alpha=0). Nothing can
    quietly solve against a partial factor: `predict` and
    `log_marginal_likelihood` on a fit with `info_ != 0` are refused by
    name, the first in Mojo, the second here with the Mojo entry's own
    sentences.

    `predict(X, return_std=True)` is honored; `return_cov=True` is REFUSED:
    the lane computes only the DIAGONAL of the posterior covariance -- a
    full `V^T V` would be an `n_star x n_star` product of which `n_star`
    cells are wanted (DEVIATION 1759). `sample_y` is refused for the same
    reason plus a normal stream inside a reproducibility claim
    (`gaussian_process/estimator.mojo::gpr_sample_y_host` carries the
    closure condition). GP CLASSIFICATION is a different algorithm (a
    Laplace approximation with a data-dependent Newton iteration) and is
    not here at all (DEVIATION 1766).

    CROSS-VENDOR STANDING: the IDENTICAL card is byte-identical Apple M4
    against AMD MI325X on every shipped-path line (the 8-line divergence
    reading was withdrawn at `9835094e`: those lines are the sabotaged half
    of a `GP_SAB_STD_EXP` pair). There is NO identity card on NVIDIA -- the
    H100 leg was a FAST speed run -- so this class claims two vendors, not
    three. That property belongs to `numeric_mode='identical'`; the FAST
    default makes no cross-vendor claim at all.

    Attributes
    ----------
    X_train_ : ndarray (n_train, n_features) float32
    y_train_ : ndarray (n_train,) float32
    kernel_ : Kernel
        The FITTED kernel, and it is the kernel you passed: there is no
        optimizer to clone-and-move it (DEVIATION 1761).
    L_ : ndarray (n_train, n_train) float32
        Lower Cholesky factor of `K + alpha I`.
    alpha_ : ndarray (n_train,) float32
        The dual coefficients `(K + alpha I)^-1 y` -- sklearn's `alpha_`,
        which is NOT the constructor's `alpha` (the ridge). One name, one
        underscore apart, and the collision is scikit-learn's; both
        meanings are spelled out wherever either appears.
    log_marginal_likelihood_value_ : float
        Meaningful only when `info_ == 0`.
    info_ : int
        LAPACK's info. 0 means L_ is a factor; k > 0 means the leading
        minor of order k was not positive definite. CHECK IT.
    nb_ : int
        The Cholesky panel width that ran (part of the profile).
    n_features_in_ : int
    input_copied_ : bool
        Whether fit had to copy X to reach float32 C order.
    """

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
        if kernel is None:
            # sklearn's documented default, `_gpr.py:229-231`. Bounds are
            # "fixed" there only to silence its optimizer; there is no
            # optimizer here, so the two defaults coincide exactly.
            kernel = ConstantKernel(1.0) * RBF(1.0)
        if not isinstance(kernel, Kernel):
            raise TypeError(
                "mojolearn GaussianProcessRegressor: kernel must be a "
                "composition of mojolearn's RBF, Matern, ConstantKernel and "
                "WhiteKernel (a scikit-learn kernel object carries bounds "
                "and a theta this port deliberately has no use for; "
                "DEVIATION 1761)"
            )
        if not (optimizer is None
                or (isinstance(optimizer, str) and optimizer.lower() == "none")):
            raise NotImplementedError(
                f"mojolearn GaussianProcessRegressor: optimizer={optimizer!r} "
                "is refused; only None runs (DEVIATION 1761). An optimizer's "
                "iteration count is data dependent, so the convergence test "
                "is itself part of the arithmetic: two vendors agreeing bit "
                "for bit on every L-BFGS step still return two different "
                "models if one stops at 41 steps and the other at 42. "
                "Pinning that -- the line search, the gradient's fold, the "
                "tolerance comparison -- is the named closure condition in "
                "gaussian_process/estimator.mojo. sklearn's default is "
                "'fmin_l_bfgs_b', so this default DIFFERS from theirs and "
                "the fitted kernel_ is the kernel you passed"
            )
        n_restarts_optimizer = int(n_restarts_optimizer)
        if n_restarts_optimizer != 0:
            raise NotImplementedError(
                "mojolearn GaussianProcessRegressor: n_restarts_optimizer="
                f"{n_restarts_optimizer!r} is refused; it restarts the "
                "optimizer, and the optimizer is refused (DEVIATION 1761)"
            )
        if normalize_y:
            raise NotImplementedError(
                "mojolearn GaussianProcessRegressor: normalize_y=True is "
                "refused; NOT PORTED (DEVIATION 1764). It centers and "
                "scales y with host reductions whose last bits would sit "
                "inside a cross-vendor identity claim. Normalize y yourself "
                "and pass the result, so the numbers that ran are numbers "
                "you made"
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
                "it shapes sample_y's prior draws, and sample_y is not "
                "ported (gaussian_process/estimator.mojo::gpr_sample_y_host "
                "carries the closure condition). y is single-target here"
            )
        if random_state is not None:
            raise NotImplementedError(
                "mojolearn GaussianProcessRegressor: random_state is "
                "refused; the only consumer of randomness upstream is "
                "sample_y, which is not ported. fit and predict draw no "
                "random numbers"
            )
        self.kernel = kernel
        # CONVERSION, NOT POLICY (the SVR epsilon rule): a non-number still
        # raises here, and every value that IS a number goes down unclamped
        # so gp_validate_alpha's refusals -- NaN, negative, +inf, and the
        # identical tier's two-value pin -- stay reachable from Python.
        self.alpha = float(alpha)
        self.optimizer = None
        self.n_restarts_optimizer = 0
        self.normalize_y = False
        self.copy_X_train = True
        self.n_targets = None
        self.random_state = None

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

    def _kernel_arrays(self):
        """The postfix spec as four flat arrays, the shape a binding can
        take (DEVIATION 1756: GPKernelSpec is five parallel host lists; the
        offsets are the one list NOT sent, because the binding MUST rebuild
        the spec through gp_kernel_const/white/rbf/matern and
        gp_kernel_sum/prod -- which recompute them -- so that every
        constructor refusal stays reachable from this surface)."""
        nodes = self.kernel._nodes()
        kinds = np.array([k for k, _, _ in nodes], dtype=np.int32)
        params = np.array([p for _, p, _ in nodes], dtype=np.float32)
        ls_len = np.array([len(ls) for _, _, ls in nodes], dtype=np.int32)
        table = [v for _, _, ls in nodes for v in ls]
        n_ls = len(table)
        # Never a zero-length buffer: mirror estimator.mojo's
        # _length_scale_table -- one unused 1.0 stands in, and n_ls says so.
        ls = np.array(table if table else [1.0], dtype=np.float32)
        return kinds, params, ls_len, ls, n_ls

    # -- fit ----------------------------------------------------------------

    def fit(self, X, y):
        """`_gpr.py:349-367`: `K = kernel(X); K[diag] += alpha;
        L = cholesky(K); alpha_ = cho_solve(L, y)`, plus the log marginal
        likelihood at the fitted kernel, all in one shot on the device via
        `gpr_fit_host`. Returns `self`. READ `info_` BEFORE YOU BELIEVE
        THE MODEL (class docstring, DEVIATION 1634)."""
        x, self.input_copied_ = as_f32_c(X, "X")
        n_rows, n_cols = x.shape
        t = np.asarray(y)
        if t.ndim != 1:
            raise ValueError(
                "mojolearn GaussianProcessRegressor: y must be 1-D, got "
                f"{t.ndim}-D shape {t.shape}; multi-target GP fits are not "
                "ported (the n_targets refusal in __init__ is this same "
                "boundary)"
            )
        if t.shape[0] != n_rows:
            raise ValueError(
                f"mojolearn GaussianProcessRegressor: y has {t.shape[0]} "
                f"entries, X has {n_rows} rows"
            )
        # No finiteness check here: check the class table -- non-finite
        # cells are refused BY NAME on the Mojo host (DEVIATION 1768), and
        # a duplicate check here would make that refusal unreachable.
        targets = np.ascontiguousarray(t, dtype=np.float32)
        kinds, kparams, ls_len, ls, n_ls = self._kernel_arrays()

        # EVERY SIZE IS A FUNCTION OF (n_rows, n_cols) ALONE; nothing here
        # is a worst-case buffer (contrast SVR's DEVIATION 873 note).
        l_out = np.empty(n_rows * n_rows, dtype=np.float32)
        dual = np.empty(n_rows, dtype=np.float32)
        # info, nb, logdet, ydotalpha, lml -- in that order.
        scalars = np.empty(5, dtype=np.float64)
        info = self._extension().gpr_fit(
            _addr_ro(x),
            _addr_ro(targets),
            _addr_ro(kinds),
            _addr_ro(kparams),
            _addr_ro(ls_len),
            _addr_ro(ls),
            _addr(l_out),
            _addr(dual),
            _addr(scalars),
            # ORDER MATCHES bindings/_mojolearn_gp.mojo::gpr_fit_binding.
            # n_train, n_features, n_nodes, n_ls, alpha
            [n_rows, n_cols, int(kinds.shape[0]), n_ls, self.alpha],
        )
        self.X_train_ = x
        self.y_train_ = targets
        self.kernel_ = self.kernel
        self.n_features_in_ = n_cols
        self.L_ = l_out.reshape(n_rows, n_rows)
        self.alpha_ = dual
        self.info_ = int(info)
        self.nb_ = int(scalars[1])
        self.log_marginal_likelihood_value_ = float(scalars[4])
        self._logdet = float(scalars[2])
        self._ydotalpha = float(scalars[3])
        return self

    # -- predict ------------------------------------------------------------

    def predict(self, X, return_std=False, return_cov=False):
        """`_gpr.py:446-500`. Returns `mean`, or `(mean, std)` with
        `return_std=True`; both float32. The variance clamp at zero and its
        per-point flags are the lane's DEVIATION 1760; the flag vector is
        `clamped_` on this instance after any `return_std` call.

        A fit with `info_ != 0` is refused BY NAME IN MOJO
        (`gpr_predict_host`, DEVIATION 1634) -- `info_` goes down with the
        call precisely so that refusal stays reachable from here."""
        if return_cov:
            raise NotImplementedError(
                "mojolearn GaussianProcessRegressor: return_cov=True is "
                "refused; the lane computes only the DIAGONAL of the "
                "posterior covariance (DEVIATION 1759: a full V^T V would "
                "be an n_star x n_star product of which n_star cells are "
                "wanted). return_std=True gives sqrt of that diagonal"
            )
        if not hasattr(self, "alpha_"):
            raise ValueError(
                "mojolearn GaussianProcessRegressor: call fit() first (the "
                "unfitted-prior arm of sklearn's predict is not ported: it "
                "exists to serve sample_y, which is refused)"
            )
        q, _ = as_f32_c(X, "X")
        if q.shape[1] != self.n_features_in_:
            raise ValueError(
                f"mojolearn GaussianProcessRegressor: X has {q.shape[1]} "
                f"features, fit saw {self.n_features_in_}"
            )
        n_star = q.shape[0]
        n_train = self.X_train_.shape[0]
        kinds, kparams, ls_len, ls, n_ls = self._kernel_arrays()

        mean = np.empty(n_star, dtype=np.float32)
        var = np.empty(n_star, dtype=np.float32)
        std = np.empty(n_star, dtype=np.float32)
        clamped = np.empty(n_star, dtype=np.int32)
        # Kept in locals so the arrays outlive the call; the Mojo side
        # borrows these addresses and owns nothing (_arrays.py).
        xt = self.X_train_
        lf = self.L_
        dual = self.alpha_
        n_clamped = self._extension().gpr_predict(
            _addr_ro(xt),
            _addr_ro(lf),
            _addr_ro(dual),
            _addr_ro(q),
            _addr_ro(kinds),
            _addr_ro(kparams),
            _addr_ro(ls_len),
            _addr_ro(ls),
            _addr(mean),
            _addr(var),
            _addr(std),
            _addr(clamped),
            # ORDER MATCHES bindings/_mojolearn_gp.mojo::gpr_predict_binding.
            # n_train, n_features, n_star, n_nodes, n_ls, return_std, info
            [n_train, self.n_features_in_, n_star, int(kinds.shape[0]),
             n_ls, 1 if return_std else 0, self.info_],
        )
        if return_std:
            self.clamped_ = clamped
            self.n_clamped_ = int(n_clamped)
            return mean, std
        return mean

    # -- the rest of sklearn's surface, honored or refused by name ----------

    def log_marginal_likelihood(self, theta=None):
        """`_gpr.py:575`, the `theta is None` arm ONLY: the value computed
        during fit. A non-None theta asks for the likelihood at OTHER
        hyperparameters, which exists to serve the optimizer, and the
        optimizer is refused (DEVIATION 1761)."""
        if theta is not None:
            raise NotImplementedError(
                "mojolearn GaussianProcessRegressor: log_marginal_likelihood"
                "(theta) at non-fitted hyperparameters is refused; it "
                "exists to serve the optimizer, which is refused (DEVIATION "
                "1761). The value at the fitted kernel is "
                "log_marginal_likelihood_value_ or this call with theta=None"
            )
        if not hasattr(self, "alpha_"):
            raise ValueError(
                "mojolearn GaussianProcessRegressor: call fit() first"
            )
        if self.info_ != 0:
            # gpr_log_marginal_likelihood's refusal, transcribed: this is
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

    def sample_y(self, X, n_samples=1, random_state=0):
        raise NotImplementedError(
            "mojolearn GaussianProcessRegressor: sample_y is NOT PORTED "
            "(gaussian_process/NOT_IMPLEMENTED.tsv). It draws from the full "
            "posterior COVARIANCE, and this lane computes only its DIAGONAL "
            "(DEVIATION 1759); it also needs a second Cholesky and a normal "
            "random stream inside a reproducibility claim. "
            "gaussian_process/estimator.mojo::gpr_sample_y_host names the "
            "closure condition"
        )

    def score(self, X, y):
        """R^2, scikit-learn's definition, accumulated in FLOAT64 from
        float32 predictions -- a summary of the answer, not the answer, so
        no part of any identity claim (the SVR.score rule)."""
        pred = np.asarray(self.predict(X), dtype=np.float64)
        t = np.asarray(y, dtype=np.float64)
        if t.ndim != 1 or t.shape[0] != pred.shape[0]:
            raise ValueError(
                "mojolearn GaussianProcessRegressor: y must be 1-D with one "
                f"entry per row of X, got shape {t.shape} against "
                f"{pred.shape[0]} predictions"
            )
        ss_res = float(np.sum((t - pred) ** 2))
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - ss_res / ss_tot
