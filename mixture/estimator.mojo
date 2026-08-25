"""The host-visible `GaussianMixture` surface, and the EM loop that drives it.

**NOT YET WIRED** into `bindings/_mojolearn_estimators.mojo` or
`python/mojolearn/` -- those directories are not this lane's.
`mixture/README.md`'s WHAT THE ORCHESTRATOR MUST WIRE names the tasks; this
file is the entry a binding should reach, shaped like
`kde/estimator.mojo::kde_score_samples_host` and
`cholesky/estimator.mojo::cholesky_factor_host`.

THERE IS NO UPSTREAM AND THIS FILE IS WHERE THAT MATTERS MOST
--------------------------------------------------------------
cuML has no Gaussian mixture model at `upstream/cuml-v26.08.00` (`265b9da`),
verified: no `gmm` directory, no `mixture` module, and the only occurrence of
`gaussian_mixture` anywhere in the tree is in
`python/cuml/cuml_accel_tests/upstream/scikit-learn/xfail-list.yaml`, marking
two scikit-learn tests FLAKY UNDER `cuml.accel` -- which is a record that
cuML does NOT accelerate the estimator and falls through to scikit-learn.
cuVS (`6ba2ce2`) and RAFT (`ebf9268`) have none either. **So
`PORTING_RULES.md`'s COPY DO NOT IMPROVE does not apply to this lane, because
there is nothing to copy.**

What governs instead: `sklearn/mixture/_gaussian_mixture.py` and `_base.py`
define the SEMANTICS and are the ORACLE, and they are never the design
source. **Every parameter this file accepts means exactly what
scikit-learn's parameter of that name means, or is named differently.**
`mixture/README.md` carries the mapping as a table and it is the contract; a
parameter that quietly meant something else would make every comparison
against scikit-learn a comparison of two different estimators.

THE SURFACE, AND WHY IT IS SHAPED LIKE THIS
---------------------------------------------
1. **`GaussianMixtureModel` carries `n_iter` and `converged`.** Not for
   diagnostics: the iteration count is a DATA-DEPENDENT output driven by a
   float comparison, it is the first thing two vendors can disagree about,
   and it is on the identity card. A model that did not carry it could not
   be compared with another model.
2. **`GaussianMixtureModel` carries `lower_bound`, the mean log likelihood
   the loop stopped at.** Same argument: it is the quantity the convergence
   test compared.
3. **A COLLAPSED COMPONENT RAISES.** `gaussian_mixture_fit` does not return
   a model with a reset component and it does not return a model with a flag
   set. DEVIATION 1723; `mixture/README.md`'s hazard 2.
4. **Nothing here takes a threads-per-block argument that reaches a number.**
   The tuning arguments exist so `check_launch_invariance` can drive them,
   and every one of them is SCHEDULING: block shape and grid shape are free
   in both modes (`mojo_only/numerics.mojo::free_block_shape`). The NUMERIC
   parameters -- the fold directions, the row-max rule, `GMM_LOG_2PI`,
   `GMM_TEN_EPS`, the Cholesky's `NB` -- are not arguments at all.

A caller that keeps its data ON THE DEVICE across many fits should call
`mixture/mojo_only/estep.mojo::gmm_e_step` and `mstep.mojo::gmm_m_step`
directly and keep its own `DeviceBuffer`s. This entry is the one-shot form,
which is what the gates and the card use.
"""

from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext

from cholesky.mojo_only.potrf import CHOL_ELEM_TPB, CHOL_PANEL_TPB
from cholesky.mojo_only.trsm import CHOL_SOLVE_TPB
from cluster.estimator import kmeans_fit
from cluster.ported.cluster.kmeans_params import (
    INIT_KMEANS_PLUS_PLUS,
    METRIC_L2_EXPANDED,
)
from core.identity_trace import IdentityTrace
from core.philox import philox4x32_10
from mixture.mojo_only.estep import (
    GMM_COMP_TPB,
    GMM_ELEM_TPB,
    GMM_PROFILE,
    GMM_ROW_TPB,
    gmm_e_step,
    gmm_estep_gemm_workspace_floats,
    gmm_estep_scratch_floats,
    gmm_convergence_change,
    gmm_converged,
    gmm_iter_prefix,
    gmm_n_parameters,
    gmm_neg_inf,
    gmm_predict_labels,
)
from mixture.mojo_only.gmm_sabotage import (
    GMM_SAB_NONE,
    GMM_SAB_TOL_ULP,
    gmm_ulp_down,
)
from mixture.mojo_only.mstep import (
    GmmMStepRun,
    gmm_chol_dwork_floats,
    gmm_chol_workspace_floats,
    gmm_m_step,
    gmm_mstep_gemm_workspace_floats,
    gmm_mstep_scratch_floats,
    gmm_precision_cholesky,
)
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, ftz


# ===========================================================================
# THE PARAMETER VOCABULARY
# ===========================================================================

#: `covariance_type="full"`. The only one this lane implements, and the one
#: with the arithmetic intensity that justifies it: `d^2` FLOPs per point per
#: component, against `d` for the other three.
comptime COV_FULL = 0
#: `covariance_type="tied"`. REFUSED BY NAME.
comptime COV_TIED = 1
#: `covariance_type="diag"`. REFUSED BY NAME.
comptime COV_DIAG = 2
#: `covariance_type="spherical"`. REFUSED BY NAME.
comptime COV_SPHERICAL = 3

#: `init_params="kmeans"`. THE ONE TO SHIP. Runs
#: `cluster/estimator.mojo::kmeans_fit`, which is identity certified in its
#: own lane, and one-hots the labels exactly as `_base.py:120-127` does.
comptime INIT_KMEANS = 0
#: `init_params="random"`. POSITION-MAPPED Philox draws, normalized per row.
#: DEVIATION 1733.
comptime INIT_RANDOM = 1
#: `init_params="k-means++"`. REFUSED BY NAME.
comptime INIT_KMEANS_PP = 2
#: `init_params="random_from_data"`. REFUSED BY NAME.
comptime INIT_RANDOM_FROM_DATA = 3

#: **NUMERIC. DEVIATION 1733.** `2^-24` as float32 bits, the scale that turns
#: a 24-bit Philox draw into a uniform on `[0, 1)`. Written as bits and
#: bitcast for the reason every constant in this lane is:
#: `[[mojo-string-float-roundtrip]]`. The mapping
#: `Float32(u >> 8) * 2^-24` is EXACT at every input -- `u >> 8` is below
#: `2^24` so it converts with no rounding, and the multiply is by a power of
#: two -- so there is no libm, no division and nothing to pin beyond this
#: constant.
comptime GMM_TWO_POW_M24_BITS: UInt32 = 0x33800000


def covariance_type_from_name(name: String) raises -> Int:
    """scikit-learn's `covariance_type` string, or a refusal BY NAME.

    The three refusals name the parameter, name the value, and say what it
    would cost to add it -- because "not supported" without a reason is a
    message that makes a user guess whether they are holding it wrong.
    """
    if name == "full":
        return COV_FULL
    if name == "tied":
        raise Error(
            "GaussianMixture: covariance_type='tied' is NOT PORTED."
            " scikit-learn shares ONE d x d covariance across every"
            " component (_gaussian_mixture.py:200-227), which is one"
            " Cholesky per iteration rather than n_components of them and a"
            " different M-step accumulation. It is a smaller lane than this"
            " one, not a subset of it: the collapse decision (DEVIATION"
            " 1723) becomes global, so a single ill-conditioned component"
            " fails the whole fit, and that is a different identity"
            " argument. mixture/UNPORTED.tsv carries it"
        )
    if name == "diag":
        raise Error(
            "GaussianMixture: covariance_type='diag' is NOT PORTED."
            " scikit-learn stores n_components x d variances and computes"
            " the log probability without any Cholesky at all"
            " (_gaussian_mixture.py:539-544, precisions = precisions_chol"
            " ** 2). That path has d FLOPs per point per component instead"
            " of d^2, so it is BANDWIDTH bound where this lane is compute"
            " bound, and ROADMAP.md's argument for opening the lane does not"
            " apply to it. mixture/UNPORTED.tsv carries it"
        )
    if name == "spherical":
        raise Error(
            "GaussianMixture: covariance_type='spherical' is NOT PORTED."
            " One variance per component (_gaussian_mixture.py:545-552),"
            " which is k-means with a soft assignment and a variance."
            " cluster/ already ships the hard-assignment half, identity"
            " certified. mixture/UNPORTED.tsv carries it"
        )
    raise Error(
        "GaussianMixture: covariance_type='"
        + name
        + "' is not one of scikit-learn's four ('full', 'tied', 'diag',"
        " 'spherical'). Only 'full' is implemented here"
    )


def init_params_from_name(name: String) raises -> Int:
    """scikit-learn's `init_params` string, or a refusal BY NAME."""
    if name == "kmeans":
        return INIT_KMEANS
    if name == "random":
        return INIT_RANDOM
    if name == "k-means++":
        raise Error(
            "GaussianMixture: init_params='k-means++' is NOT PORTED."
            " scikit-learn calls sklearn.cluster.kmeans_plusplus and"
            " one-hots the CHOSEN INDICES rather than a full assignment"
            " (_base.py:148-155), so it needs the seeding routine's own"
            " draw sequence to be reproducible on three vendors."
            " cluster/mojo_only/plus_plus.mojo is cuVS's k-means++ and its"
            " draw order is cuVS's, not scikit-learn's, so wiring it here"
            " would give a DIFFERENT initialization under a scikit-learn"
            " parameter name -- which mixture/README.md's semantics table"
            " forbids. init_params='kmeans' is the supported one"
        )
    if name == "random_from_data":
        raise Error(
            "GaussianMixture: init_params='random_from_data' is NOT PORTED."
            " scikit-learn draws n_components row indices WITHOUT"
            " REPLACEMENT (_base.py:140-147), which is a rejection loop"
            " whose consumed-draw count depends on the data. That is a"
            " reproducibility question of its own and this lane has not"
            " answered it. init_params='random' is the position-mapped"
            " alternative (DEVIATION 1733)"
        )
    raise Error(
        "GaussianMixture: init_params='"
        + name
        + "' is not one of scikit-learn's four ('kmeans', 'k-means++',"
        " 'random', 'random_from_data'). 'kmeans' and 'random' are"
        " implemented here"
    )


@fieldwise_init
struct GmmParams(Copyable, ImplicitlyCopyable, Movable):
    """scikit-learn's `GaussianMixture.__init__` parameters, rung 1.

    Defaults are SCIKIT-LEARN'S, stated rather than quietly matched, because
    a default that differs is a comparison of two estimators:
    `covariance_type="full"`, `tol=1e-3`, `reg_covar=1e-6`, `max_iter=100`,
    `init_params="kmeans"`.

    **`n_init` IS NOT A FIELD, AND `warm_start` IS NOT A FIELD.** DEVIATION
    1734: `n_init > 1` picks the best restart by a FLOAT COMPARISON of lower
    bounds (`_base.py:282`), which is a second data-dependent branch on top
    of the convergence test and would need its own identity argument and its
    own tie rule; `warm_start` makes the fit depend on the object's history
    rather than on its inputs. Both are refused BY ABSENCE -- there is no
    field to set -- and `mixture/UNPORTED.tsv` carries them. scikit-learn's
    `n_init` default is 1 and its `warm_start` default is False, so the
    shipped behavior matches the shipped defaults.

    `means_init`, `weights_init` and `precisions_init` are absent for the
    same reason and are also in `UNPORTED.tsv`.
    """

    var n_components: Int
    var covariance_type: Int
    var tol: Float32
    var reg_covar: Float32
    var max_iter: Int
    var init_params: Int
    var random_state: UInt64

    @staticmethod
    def default() -> Self:
        """scikit-learn's defaults, exactly."""
        return Self(
            1,
            COV_FULL,
            Float32(1.0e-3),
            Float32(1.0e-6),
            100,
            INIT_KMEANS,
            UInt64(0),
        )


@fieldwise_init
struct GaussianMixtureModel(Movable):
    """A fitted model. Everything a caller must not recompute for itself,
    plus the two fields that make two fits comparable."""

    var n_components: Int
    var n_features: Int

    var weights: List[Float32]
    """`weights_`, `n_components`."""

    var means: List[Float32]
    """`means_`, `n_components x n_features` row-major."""

    var covariances: List[Float32]
    """`covariances_`, `n_components x n_features x n_features` row-major
    per component. INCLUDES `reg_covar` on the diagonal, exactly as
    scikit-learn's does."""

    var precisions_cholesky: List[Float32]
    """`precisions_cholesky_`, `(L_k^{-1})^T` per component, UPPER
    triangular. Carried rather than recomputed because `predict`,
    `predict_proba` and `score_samples` all need it and each recomputation
    would be another Cholesky and another `log` fold."""

    var log_det_chol: List[Float32]
    """`_compute_log_det_cholesky`'s answer, `n_components`. DEVIATION 1726
    computes it from `chol_logdet` and a caller must not re-derive it from
    the diagonal above; the two are the same number and not the same bits."""

    var n_iter: Int
    """`n_iter_`. **PART OF THE IDENTITY CARD.** A data-dependent count
    driven by a float comparison; see `mixture/README.md`'s hazard 1."""

    var converged: Bool
    """`converged_`. False means the loop ran out of iterations, which
    scikit-learn reports as a `ConvergenceWarning`. DEVIATION 1746: Mojo has
    no warning channel, so this is a FIELD and `mixture_main.mojo` prints
    it. It is never a raise, because scikit-learn returns a usable model in
    that case and so does this."""

    var lower_bound: Float32
    """`lower_bound_`, the mean log likelihood the loop stopped at. The
    quantity the convergence test compared."""


# ===========================================================================
# VALIDATION: EVERY REFUSAL BY NAME, BEFORE ANY LAUNCH
# ===========================================================================


def gmm_hex32(v: Float32) -> String:
    """A float32 as eight hex digits. Every refusal that names a VALUE names
    its bits, because `[[mojo-string-float-roundtrip]]`: a decimal in an
    error message cannot be trusted to be the number that caused it."""
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


def gmm_validate_params(params: GmmParams, n_samples: Int) raises:
    """`_check_parameters` plus `_parameter_constraints`
    (`_base.py:56-70`, `_gaussian_mixture.py:800-834`), every refusal BY
    NAME. DEVIATION 1738.

    The `n_components > n_samples` refusal is this lane's and not
    scikit-learn's: they let it through and fail later inside the k-means
    initialization with a message about `n_clusters`. Refusing here names the
    parameter the caller set.
    """
    if params.covariance_type != COV_FULL:
        raise Error(
            "GaussianMixture: only covariance_type='full' is implemented"
            " (got type id "
            + String(params.covariance_type)
            + "). See covariance_type_from_name for what each of the other"
            " three would cost and mixture/UNPORTED.tsv for the record"
        )
    if (
        params.init_params != INIT_KMEANS
        and params.init_params != INIT_RANDOM
    ):
        raise Error(
            "GaussianMixture: init_params id "
            + String(params.init_params)
            + " is not implemented; only 'kmeans' and 'random' are. See"
            " init_params_from_name for why 'k-means++' and"
            " 'random_from_data' are refused rather than approximated"
        )
    if params.n_components < 1:
        raise Error(
            "GaussianMixture: n_components must be at least 1, got "
            + String(params.n_components)
        )
    if params.n_components > n_samples:
        raise Error(
            "GaussianMixture: n_components="
            + String(params.n_components)
            + " exceeds n_samples="
            + String(n_samples)
            + ". Every component would need at least one point to have a"
            " covariance at all, so the fit is guaranteed to collapse"
            " (DEVIATION 1723) and refusing here names the parameter the"
            " caller set rather than failing later inside the"
            " initialization"
        )
    if params.max_iter < 0:
        raise Error(
            "GaussianMixture: max_iter must be at least 0, got "
            + String(params.max_iter)
            + ". Zero is legal and means initialization only, which is"
            " scikit-learn's behavior (DEVIATION 1745)"
        )
    if params.tol != params.tol:
        raise Error("GaussianMixture: tol is NaN; refused by name")
    if params.tol < Float32(0.0):
        raise Error(
            "GaussianMixture: tol must be non-negative, got "
            + gmm_hex32(params.tol)
        )
    if params.reg_covar != params.reg_covar:
        raise Error("GaussianMixture: reg_covar is NaN; refused by name")
    if params.reg_covar < Float32(0.0):
        raise Error(
            "GaussianMixture: reg_covar must be non-negative, got "
            + gmm_hex32(params.reg_covar)
        )
    var big = bitcast[DType.float32](UInt32(0x7F800000))
    if params.reg_covar == big:
        raise Error(
            "GaussianMixture: reg_covar is +inf; refused by name"
        )
    if params.tol == big:
        raise Error(
            "GaussianMixture: tol is +inf; every fit would 'converge' at"
            " iteration 1 and the iteration count would stop being a"
            " function of the data. Refused by name"
        )


def gmm_validate_data(
    x: List[Float32], n_samples: Int, n_features: Int
) raises:
    """No NaN and no infinity in `X`, refused BY NAME with the CELL and the
    BITS. DEVIATION 1738.

    scikit-learn's `validate_data` refuses non-finite input the same way
    (`_base.py:236`), so this is their semantics and not an addition. What is
    this lane's is the reason it happens BEFORE ANY UPLOAD: every stage of
    this fit is a recorded card stage, and **a COMPUTED NaN carries the
    vendor's payload** -- Apple `0x7fc00000`, NVIDIA `0x7fffffff`, AMD
    `0xffc00000`, all three measured (IDENTITY_PATHS row 39, FACT 2) -- so a
    NaN that reaches `gmm.iterNNN.mahal` makes three vendors' cards disagree
    while every arithmetic decision in them matched.
    """
    if n_samples < 1:
        raise Error(
            "GaussianMixture: X must have at least one row, got "
            + String(n_samples)
        )
    if n_features < 1:
        raise Error(
            "GaussianMixture: X must have at least one column, got "
            + String(n_features)
        )
    if len(x) != n_samples * n_features:
        raise Error(
            "GaussianMixture: X holds "
            + String(len(x))
            + " floats, "
            + String(n_samples)
            + " x "
            + String(n_features)
            + " needs "
            + String(n_samples * n_features)
        )
    for i in range(len(x)):
        var v = x[i]
        if v != v:
            raise Error(
                "GaussianMixture: X contains NaN at row "
                + String(i // n_features)
                + " column "
                + String(i % n_features)
                + "; refused by name before any upload (DEVIATION 1738)"
            )
        var u = bitcast[DType.uint32](v) & UInt32(0x7FFFFFFF)
        if u == UInt32(0x7F800000):
            raise Error(
                "GaussianMixture: X contains an infinity at row "
                + String(i // n_features)
                + " column "
                + String(i % n_features)
                + " ("
                + gmm_hex32(v)
                + "); refused by name before any upload (DEVIATION 1738)"
            )


# ===========================================================================
# THE INITIAL RESPONSIBILITIES
# ===========================================================================


def gmm_initial_resp(
    ctx: DeviceContext,
    x: List[Float32],
    n_samples: Int,
    n_features: Int,
    params: GmmParams,
) raises -> List[Float32]:
    """`_initialize_parameters` (`_base.py:105-157`): the `n x K` matrix the
    first M-step is handed.

    **`init_params="kmeans"`: `cluster/estimator.mojo::kmeans_fit`, ONE-HOT.**
    `_base.py:120-127` runs `KMeans(n_clusters=K, n_init=1,
    random_state=...)` and sets `resp[arange(n), labels] = 1`. This runs
    `kmeans_fit` at `n_init=1`, `init=INIT_KMEANS_PLUS_PLUS` (scikit-learn's
    `KMeans` default), `metric=METRIC_L2_EXPANDED`, `max_iter=300`,
    `tol=1e-4` -- scikit-learn's `KMeans` defaults -- and one-hots the same
    way. **The k-means is not re-implemented and its identity is not this
    lane's claim**; `cluster/` owns it, `pixi run check-kmeans-identity` is
    its gate, and `mixture/README.md`'s WHAT THIS LANE REUSES RATHER THAN
    REWRITES names the entry point.

    **`init_params="random"`: POSITION-MAPPED Philox. DEVIATION 1733.**
    `_base.py:128-134` draws `n x K` uniforms from a `RandomState` STREAM and
    normalizes each row. A stream is an ORDER, and an order is the one thing
    three vendors cannot be made to agree on for free. So each cell's draw is
    a pure function of ITS OWN POSITION: `philox4x32_10` with the counter set
    from `(i, k)` and the key from `random_state`, taking word 0. Same
    counter, same key, same word, same bits, on every vendor and at every
    launch -- and it is not scikit-learn's draw, which is why the README's
    semantics table marks `random` as MEANING-COMPATIBLE (a uniform Dirichlet
    -like row) and BIT-DIFFERENT from theirs.

    The uniform mapping is `Float32(u >> 8) * 2^-24`, which is EXACT: `u >> 8`
    is below `2^24` so it converts without rounding, and `2^-24` is a power
    of two. No libm, no division, nothing to pin.

    A row whose draws are all zero has no normalizer and RAISES rather than
    producing a NaN row: it cannot happen at any seed anyone will use
    (probability `2^-24K`) and a guard that is cheap and names itself is
    better than a NaN in a certified stage.
    """
    var ncomp = params.n_components
    var resp = List[Float32]()

    if params.init_params == INIT_KMEANS:
        var hx = ctx.enqueue_create_host_buffer[DType.float32](
            n_samples * n_features
        )
        var hc = ctx.enqueue_create_host_buffer[DType.float32](
            ncomp * n_features
        )
        var hl = ctx.enqueue_create_host_buffer[DType.uint32](n_samples)
        ctx.synchronize()
        for i in range(n_samples * n_features):
            hx.unsafe_ptr().unsafe_store(i, x[i])
        var res = kmeans_fit(
            ctx,
            hx.unsafe_ptr(),
            n_samples,
            n_features,
            ncomp,
            hc.unsafe_ptr(),
            hl.unsafe_ptr(),
            hx.unsafe_ptr(),
            0,
            max_iter=300,
            tol=1.0e-4,
            seed=params.random_state,
            n_init=1,
            init=INIT_KMEANS_PLUS_PLUS,
            metric=METRIC_L2_EXPANDED,
        )
        _ = res
        for i in range(n_samples):
            var lab = Int(hl.unsafe_ptr().unsafe_load(i))
            for k in range(ncomp):
                resp.append(Float32(1.0) if k == lab else Float32(0.0))
        _ = hx^
        _ = hc^
        _ = hl^
        return resp^

    # INIT_RANDOM
    var key = SIMD[DType.uint32, 2](
        UInt32(params.random_state & 0xFFFFFFFF),
        UInt32((params.random_state >> 32) & 0xFFFFFFFF),
    )
    for i in range(n_samples):
        var row = List[Float32]()
        var s = Float32(0.0)
        for k in range(ncomp):
            var ctr = SIMD[DType.uint32, 4](
                UInt32(i & 0xFFFFFFFF),
                UInt32((i >> 32) & 0xFFFFFFFF),
                UInt32(k),
                UInt32(0),
            )
            var draw = philox4x32_10(ctr, key)
            var u = Float32(Int(draw[0] >> UInt32(8))) * bitcast[
                DType.float32
            ](GMM_TWO_POW_M24_BITS)
            row.append(u)
            s = ftz(s + u)
        if not (s > Float32(0.0)):
            raise Error(
                "GaussianMixture: init_params='random' produced a row of"
                " responsibilities summing to "
                + gmm_hex32(s)
                + " at row "
                + String(i)
                + ", which has no normalizer. At any seed a caller will"
                " use this cannot happen; it is refused by name rather"
                " than divided through to a NaN, because gmm.init.resp is"
                " a certified card stage and a computed NaN carries the"
                " vendor's payload (IDENTITY_PATHS row 39)"
            )
        for k in range(ncomp):
            resp.append(ftz(row[k] / s))
    return resp^


# ===========================================================================
# THE FIT
# ===========================================================================


def _upload(
    ctx: DeviceContext, values: List[Float32]
) raises -> DeviceBuffer[DType.float32]:
    var n = len(values)
    var buf = ctx.enqueue_create_buffer[DType.float32](n)
    var host = ctx.enqueue_create_host_buffer[DType.float32](n)
    for i in range(n):
        host.unsafe_ptr().unsafe_store(i, values[i])
    ctx.enqueue_copy(dst_buf=buf, src_ptr=host.unsafe_ptr())
    ctx.synchronize()
    _ = host^
    return buf^


def _download(
    ctx: DeviceContext, mut buf: DeviceBuffer[DType.float32], n: Int
) raises -> List[Float32]:
    var h = ctx.enqueue_create_host_buffer[DType.float32](n)
    var view = buf.create_sub_buffer[DType.float32](0, n)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=view)
    ctx.synchronize()
    var out = List[Float32]()
    for i in range(n):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    _ = view^
    return out^


def gaussian_mixture_fit(
    x: List[Float32],
    n_samples: Int,
    n_features: Int,
    params: GmmParams,
    elem_tpb: Int = GMM_ELEM_TPB,
    row_tpb: Int = GMM_ROW_TPB,
    comp_tpb: Int = GMM_COMP_TPB,
    solve_tpb: Int = CHOL_SOLVE_TPB,
    panel_tpb: Int = CHOL_PANEL_TPB,
    chol_elem_tpb: Int = CHOL_ELEM_TPB,
    sabotage: Int = GMM_SAB_NONE,
    card_prefix: String = "gmm",
    trace_path: String = "",
) raises -> GaussianMixtureModel:
    """`GaussianMixture(...).fit(X)`, host in and host out, one shot.

    `trace_path` empty means READ THE ENVIRONMENT (`MOJOLEARN_IDENTITY_TRACE`),
    which is what `mixture/mixture_main.mojo` wants; a non-empty value points
    the card at an explicit file and IGNORES the environment, which is what
    the checks want. `core/identity_trace.mojo::IdentityTrace.to_path` says
    why: "a check whose behavior depends on whether the operator happens to
    have MOJOLEARN_IDENTITY_TRACE exported is a check that passes or fails
    for reasons outside itself."

    `_base.py::fit_predict:241-311` with `n_init = 1` and
    `warm_start = False`, in scikit-learn's order:

        validate; initialize (resp -> moments -> precision Cholesky)
        for it in 1..max_iter:
            prev = lower_bound
            E-step; M-step; precision Cholesky
            lower_bound = mean log likelihood
            if abs(lower_bound - prev) < tol: converged; STOP

    **THE CONVERGENCE TEST IS ON THE HOST AND THAT COSTS ONE DRAIN PER
    ITERATION.** `meanll` is read back so `abs(change) < tol` is decided by
    one float32 comparison in one place, on a value the card has already
    hashed. The alternative -- a device-side test writing a flag -- would put
    the decision behind a kernel launch where nothing can hash it, and
    `PORTING_RULES.md` rule 0c records that this repository has already
    invented one on-device convergence test, cited a line range that was
    actually a function signature, and been wrong: cuVS's own k-means loop
    syncs at `detail/kmeans.cuh:491` and tests on the HOST at `:492`. The
    drain is deliberate and `mixture/README.md`'s WHAT IS OWED prices it.

    **A COLLAPSED COMPONENT RAISES.** DEVIATION 1723. The error names the
    iteration, the component and LAPACK's `info`, and mirrors
    scikit-learn's own message.

    THE CARD, and its ORDER is the product rather than its length:

        <prefix>.input                          X as uploaded
        <prefix>.init.resp                      the initial responsibilities
        <prefix>.init.{resp,nk,weights,means,covariances}
        <prefix>.init.compKKK.{cholesky,precchol}   per component
        <prefix>.init.logdet
        <prefix>.iterNNN.{mahal,wlp,rowmax,lse,logresp,meanll}
        <prefix>.iterNNN.{resp,nk,weights,means,covariances}
        <prefix>.iterNNN.compKKK.{cholesky,precchol}
        <prefix>.iterNNN.logdet
        <prefix>.iterNNN.change                 the convergence quantity
        <prefix>.niter                          [n_iter, converged, max_iter]
    """
    gmm_validate_data(x, n_samples, n_features)
    gmm_validate_params(params, n_samples)

    var n = n_samples
    var d = n_features
    var ncomp = params.n_components
    var dd = d * d

    var ctx = DeviceContext()
    var trace: IdentityTrace
    if trace_path == "":
        trace = IdentityTrace()
    else:
        trace = IdentityTrace.to_path(trace_path)
    trace.header(
        "gmm: profile="
        + GMM_PROFILE
        + " n="
        + String(n)
        + " d="
        + String(d)
        + " n_components="
        + String(ncomp)
        + " tol="
        + gmm_hex32(params.tol)
        + " reg_covar="
        + gmm_hex32(params.reg_covar)
        + " max_iter="
        + String(params.max_iter)
        + " init="
        + String(params.init_params)
        + " seed="
        + String(params.random_state)
    )

    var resp0 = gmm_initial_resp(ctx, x, n, d, params)

    var dx = _upload(ctx, x)
    trace.record_device(ctx, card_prefix + ".input", dx, n * d)

    var logresp = ctx.enqueue_create_buffer[DType.float32](n * ncomp)
    var resp = ctx.enqueue_create_buffer[DType.float32](n * ncomp)
    var mahal = ctx.enqueue_create_buffer[DType.float32](n * ncomp)
    var wlp = ctx.enqueue_create_buffer[DType.float32](n * ncomp)
    var rowmax = ctx.enqueue_create_buffer[DType.float32](n)
    var lse = ctx.enqueue_create_buffer[DType.float32](n)
    var meanll = ctx.enqueue_create_buffer[DType.float32](1)
    var nk = ctx.enqueue_create_buffer[DType.float32](ncomp)
    var weights = ctx.enqueue_create_buffer[DType.float32](ncomp)
    var log_weights = ctx.enqueue_create_buffer[DType.float32](ncomp)
    var log_det_chol = ctx.enqueue_create_buffer[DType.float32](ncomp)
    var means = ctx.enqueue_create_buffer[DType.float32](ncomp * d)
    var cov = ctx.enqueue_create_buffer[DType.float32](ncomp * dd)
    var chol_l = ctx.enqueue_create_buffer[DType.float32](ncomp * dd)
    var linv = ctx.enqueue_create_buffer[DType.float32](ncomp * dd)
    var prec = ctx.enqueue_create_buffer[DType.float32](ncomp * dd)
    var escratch = ctx.enqueue_create_buffer[DType.float32](
        gmm_estep_scratch_floats(n, d)
    )
    var mscratch = ctx.enqueue_create_buffer[DType.float32](
        gmm_mstep_scratch_floats(n, d, ncomp)
    )
    # ONE gemm workspace, sized for the LARGEST plan any of the four products
    # could reach. `identical_gemm_into`'s docstring records that sizing for
    # one plan and letting the dispatcher pick another is an out-of-bounds
    # write a small shape will not show you.
    var need_gws = gmm_estep_gemm_workspace_floats(n, d)
    var mg = gmm_mstep_gemm_workspace_floats(n, d, ncomp)
    if mg > need_gws:
        need_gws = mg
    var gws = ctx.enqueue_create_buffer[DType.float32](need_gws)
    var cws = ctx.enqueue_create_buffer[DType.float32](
        gmm_chol_workspace_floats(d)
    )
    var dwork = ctx.enqueue_create_buffer[DType.float32](
        gmm_chol_dwork_floats(d)
    )
    ctx.synchronize()

    # ---- initialization -------------------------------------------------
    # `_initialize` is handed `resp`, not `log_resp` (`_base.py:157`), and
    # `gmm_m_step` takes logs -- so the logs are formed here, on the host,
    # from a one-hot or a normalized uniform. It is the one place in the fit
    # a `log` of a value we already had is taken, and it is exact for the
    # one-hot case (`log(1) = +0.0`, `log(0) = -inf`, and `exp` of those is
    # `1` and `+0.0` again).
    var loginit = List[Float32]()
    for i in range(n * ncomp):
        var v = resp0[i]
        loginit.append(_safe_log(v))
    var dloginit = _upload(ctx, loginit)
    var dresp0 = _upload(ctx, resp0)
    # `.resp0` and not `.resp`: `gmm_m_step` records `<tag>.resp` for
    # `exp(log_resp)` and `IdentityTrace._emit` RAISES on a duplicate tag
    # (its uniqueness invariant), so the INITIAL responsibilities and the
    # first M-step's recomputed ones have to name themselves apart. They are
    # also different quantities -- one is the initialization's input, the
    # other is a round trip through `log` and `exp` -- and a card that gave
    # them one name would hide that.
    trace.record_device(
        ctx, card_prefix + ".init.resp0", dresp0, n * ncomp
    )

    gmm_m_step(
        ctx, dx, dloginit, resp, nk, weights, log_weights, means, cov,
        mscratch, gws, n, d, ncomp, params.reg_covar, True, trace,
        card_prefix + ".init", elem_tpb, comp_tpb, sabotage,
    )
    var pre = gmm_precision_cholesky(
        ctx, cov, chol_l, linv, prec, log_det_chol, cws, dwork, d, ncomp,
        trace, card_prefix + ".init", elem_tpb, solve_tpb, panel_tpb,
        chol_elem_tpb, sabotage,
    )
    if pre.info != 0:
        raise Error(_collapse_message(0, pre, params))

    var lower_bound = gmm_neg_inf()
    var n_iter = 0
    var converged = False
    var hll = ctx.enqueue_create_host_buffer[DType.float32](1)
    ctx.synchronize()

    for it in range(1, params.max_iter + 1):
        var tag = gmm_iter_prefix(card_prefix, it)
        var prev = lower_bound

        gmm_e_step(
            ctx, dx, means, prec, linv, log_det_chol, log_weights,
            escratch, gws, mahal, wlp, rowmax, lse, logresp, meanll,
            n, d, ncomp, trace, tag, elem_tpb, row_tpb, sabotage,
        )
        gmm_m_step(
            ctx, dx, logresp, resp, nk, weights, log_weights, means, cov,
            mscratch, gws, n, d, ncomp, params.reg_covar, False, trace,
            tag, elem_tpb, comp_tpb, sabotage,
        )
        var p2 = gmm_precision_cholesky(
            ctx, cov, chol_l, linv, prec, log_det_chol, cws, dwork, d,
            ncomp, trace, tag, elem_tpb, solve_tpb, panel_tpb,
            chol_elem_tpb, sabotage,
        )
        if p2.info != 0:
            raise Error(_collapse_message(it, p2, params))

        # THE ONE DRAIN PER ITERATION. See this function's docstring.
        ctx.enqueue_copy(dst_ptr=hll.unsafe_ptr(), src_buf=meanll)
        ctx.synchronize()
        lower_bound = hll.unsafe_ptr().unsafe_load(0)

        var change = gmm_convergence_change(lower_bound, prev)
        var compared = change
        if sabotage == GMM_SAB_TOL_ULP:
            # ARM. THE MOST IMPORTANT ONE IN THE LANE. The value compared
            # against `tol` is moved ONE ULP toward zero. No parameter and no
            # probability changes; only whether the loop stops.
            compared = gmm_ulp_down(change)
        var one = List[Float32]()
        one.append(change)
        trace.record_list_f32(tag + ".change", one)

        n_iter = it
        if gmm_converged(compared, params.tol):
            converged = True
            break

    var card = List[Int32]()
    card.append(Int32(n_iter))
    card.append(Int32(1) if converged else Int32(0))
    card.append(Int32(params.max_iter))
    trace.record_list_i32(card_prefix + ".niter", card)

    var out_w = _download(ctx, weights, ncomp)
    var out_m = _download(ctx, means, ncomp * d)
    var out_c = _download(ctx, cov, ncomp * dd)
    var out_p = _download(ctx, prec, ncomp * dd)
    var out_l = _download(ctx, log_det_chol, ncomp)

    _ = dx^
    _ = dloginit^
    _ = dresp0^
    _ = logresp^
    _ = resp^
    _ = mahal^
    _ = wlp^
    _ = rowmax^
    _ = lse^
    _ = meanll^
    _ = nk^
    _ = weights^
    _ = log_weights^
    _ = log_det_chol^
    _ = means^
    _ = cov^
    _ = chol_l^
    _ = linv^
    _ = prec^
    _ = escratch^
    _ = mscratch^
    _ = gws^
    _ = cws^
    _ = dwork^
    _ = hll^

    return GaussianMixtureModel(
        ncomp, d, out_w^, out_m^, out_c^, out_p^, out_l^, n_iter,
        converged, lower_bound,
    )


def _safe_log(v: Float32) -> Float32:
    """`log` of an initial responsibility, which is `0` or `1` for a one-hot
    and strictly inside `(0, 1)` for the uniform initialization.

    `identical_log(+0.0)` is `-inf`, which is the mathematically right value
    and is what `exp` turns back into `+0.0` on the first M-step. It is
    written through this helper rather than inline so the one-hot case's
    `-inf` is visibly INTENDED: an `-inf` appearing in `gmm.init.resp`'s
    successor stage is otherwise the kind of thing a reader spends an
    afternoon on.
    """
    from mojo_only.numerics import identical_log

    if v == Float32(0.0):
        return bitcast[DType.float32](UInt32(0xFF800000))
    return ftz(identical_log(ftz(v)))


def _collapse_message(
    it: Int, run: GmmMStepRun, params: GmmParams
) -> String:
    """The refusal a collapsed component produces. **DEVIATION 1723.**

    scikit-learn's own wording is kept because it is the message a user has
    already read somewhere else, and the three things it tells them to try
    are the three things that actually work. What is added is the ITERATION,
    the COMPONENT and LAPACK's `info` -- the address of the failure -- and
    the sentence about why the component is not reset.
    """
    var where = String("initialization")
    if it > 0:
        where = String("EM iteration ") + String(it)
    return (
        "GaussianMixture: fitting the mixture model failed because some"
        " components have ill-defined empirical covariance (for instance"
        " caused by singleton or collapsed samples). Try to decrease the"
        " number of components, increase reg_covar, or scale the input"
        " data. FAILED AT "
        + where
        + ", COMPONENT "
        + String(run.failed_component)
        + ", LAPACK info="
        + String(run.info)
        + " (the leading minor of order "
        + String(run.info)
        + " of that component's covariance is not positive definite)."
        " reg_covar was "
        + gmm_hex32(params.reg_covar)
        + ". THE COMPONENT IS NOT RESET AND THE FIT IS NOT CONTINUED"
        " (DEVIATION 1723): the pivot decision is pinned in both halves by"
        " cholesky/mojo_only/potrf.mojo's DEVIATION 1634, so this refusal"
        " is identical on every vendor -- and a silent reset would turn a"
        " decision that is identical into a value that is not, which is"
        " the worst outcome this lane can produce"
    )


# ===========================================================================
# SCORING AGAINST A FITTED MODEL
# ===========================================================================


def gaussian_mixture_score_samples(
    model: GaussianMixtureModel,
    x: List[Float32],
    n_samples: Int,
    elem_tpb: Int = GMM_ELEM_TPB,
    row_tpb: Int = GMM_ROW_TPB,
    card_prefix: String = "gmm.score",
    trace_path: String = "",
) raises -> List[Float32]:
    """`score_samples(X)`, `_base.py:356-373`: `logsumexp_k` of the weighted
    log probabilities, one value per row.

    **ONE POINT SCORED ALONE MUST EQUAL THE SAME POINT INSIDE A BATCH**, and
    that is a property of the arithmetic rather than an assertion: the
    Mahalanobis fold is per row, the logsumexp is per row, and no quantity in
    this function is computed across rows. There is no per-batch
    normalization anywhere and there cannot be one, which is what DEVIATION
    1739 records and `check_launch_invariance` measures.
    """
    var d = model.n_features
    var ncomp = model.n_components
    gmm_validate_data(x, n_samples, d)

    var ctx = DeviceContext()
    # **DISABLED UNLESS ASKED, and this is the opposite default from
    # `gaussian_mixture_fit`.** A scoring entry is called in a LOOP -- once
    # per point by `check_launch_invariance`'s alone-versus-batch arm -- and
    # every call records the same six tags, which `IdentityTrace._emit`
    # refuses as a duplicate. So an environment variable cannot be allowed to
    # turn the card on here; a caller that wants a scoring card names the
    # file and takes responsibility for calling once.
    var trace: IdentityTrace
    if trace_path == "":
        trace = IdentityTrace.disabled()
    else:
        trace = IdentityTrace.to_path(trace_path)
    var dx = _upload(ctx, x)
    var dmeans = _upload(ctx, model.means)
    var dprec = _upload(ctx, model.precisions_cholesky)
    var dlogdet = _upload(ctx, model.log_det_chol)
    var lw = List[Float32]()
    for k in range(ncomp):
        lw.append(_safe_log(model.weights[k]))
    var dlw = _upload(ctx, lw)
    # `linv` is only read by GMM_SAB_VENDOR_MATMUL, which no scoring path
    # drives, so the precision buffer stands in for it rather than a second
    # allocation. Stated because a reader will check.
    var dlinv = _upload(ctx, model.precisions_cholesky)

    var mahal = ctx.enqueue_create_buffer[DType.float32](n_samples * ncomp)
    var wlp = ctx.enqueue_create_buffer[DType.float32](n_samples * ncomp)
    var rowmax = ctx.enqueue_create_buffer[DType.float32](n_samples)
    var lse = ctx.enqueue_create_buffer[DType.float32](n_samples)
    var logresp = ctx.enqueue_create_buffer[DType.float32](
        n_samples * ncomp
    )
    var meanll = ctx.enqueue_create_buffer[DType.float32](1)
    var escratch = ctx.enqueue_create_buffer[DType.float32](
        gmm_estep_scratch_floats(n_samples, d)
    )
    var gws = ctx.enqueue_create_buffer[DType.float32](
        gmm_estep_gemm_workspace_floats(n_samples, d)
    )
    ctx.synchronize()

    gmm_e_step(
        ctx, dx, dmeans, dprec, dlinv, dlogdet, dlw, escratch, gws,
        mahal, wlp, rowmax, lse, logresp, meanll, n_samples, d, ncomp,
        trace, card_prefix, elem_tpb, row_tpb, GMM_SAB_NONE,
    )
    var out = _download(ctx, lse, n_samples)
    _ = dx^
    _ = dmeans^
    _ = dprec^
    _ = dlogdet^
    _ = dlw^
    _ = dlinv^
    _ = mahal^
    _ = wlp^
    _ = rowmax^
    _ = lse^
    _ = logresp^
    _ = meanll^
    _ = escratch^
    _ = gws^
    return out^


def gaussian_mixture_predict_proba(
    model: GaussianMixtureModel,
    x: List[Float32],
    n_samples: Int,
    elem_tpb: Int = GMM_ELEM_TPB,
    row_tpb: Int = GMM_ROW_TPB,
    card_prefix: String = "gmm.proba",
    trace_path: String = "",
) raises -> List[Float32]:
    """`predict_proba(X)`, `_base.py:414-432`: `exp(log_resp)`, `n x K`.

    Computed as `exp` of the log responsibilities and never as a normalized
    ratio of probabilities, which is scikit-learn's route and is the only one
    that survives a point far from every component: the ratio's numerator and
    denominator both underflow and the quotient is `0/0`.
    """
    var d = model.n_features
    var ncomp = model.n_components
    gmm_validate_data(x, n_samples, d)

    var ctx = DeviceContext()
    # **DISABLED UNLESS ASKED, and this is the opposite default from
    # `gaussian_mixture_fit`.** A scoring entry is called in a LOOP -- once
    # per point by `check_launch_invariance`'s alone-versus-batch arm -- and
    # every call records the same six tags, which `IdentityTrace._emit`
    # refuses as a duplicate. So an environment variable cannot be allowed to
    # turn the card on here; a caller that wants a scoring card names the
    # file and takes responsibility for calling once.
    var trace: IdentityTrace
    if trace_path == "":
        trace = IdentityTrace.disabled()
    else:
        trace = IdentityTrace.to_path(trace_path)
    var dx = _upload(ctx, x)
    var dmeans = _upload(ctx, model.means)
    var dprec = _upload(ctx, model.precisions_cholesky)
    var dlogdet = _upload(ctx, model.log_det_chol)
    var lw = List[Float32]()
    for k in range(ncomp):
        lw.append(_safe_log(model.weights[k]))
    var dlw = _upload(ctx, lw)
    var dlinv = _upload(ctx, model.precisions_cholesky)

    var mahal = ctx.enqueue_create_buffer[DType.float32](n_samples * ncomp)
    var wlp = ctx.enqueue_create_buffer[DType.float32](n_samples * ncomp)
    var rowmax = ctx.enqueue_create_buffer[DType.float32](n_samples)
    var lse = ctx.enqueue_create_buffer[DType.float32](n_samples)
    var logresp = ctx.enqueue_create_buffer[DType.float32](
        n_samples * ncomp
    )
    var resp = ctx.enqueue_create_buffer[DType.float32](n_samples * ncomp)
    var meanll = ctx.enqueue_create_buffer[DType.float32](1)
    var escratch = ctx.enqueue_create_buffer[DType.float32](
        gmm_estep_scratch_floats(n_samples, d)
    )
    var gws = ctx.enqueue_create_buffer[DType.float32](
        gmm_estep_gemm_workspace_floats(n_samples, d)
    )
    ctx.synchronize()

    gmm_e_step(
        ctx, dx, dmeans, dprec, dlinv, dlogdet, dlw, escratch, gws,
        mahal, wlp, rowmax, lse, logresp, meanll, n_samples, d, ncomp,
        trace, card_prefix, elem_tpb, row_tpb, GMM_SAB_NONE,
    )
    var out_log = _download(ctx, logresp, n_samples * ncomp)
    var out = List[Float32]()
    for i in range(n_samples * ncomp):
        out.append(_exp_resp(out_log[i]))
    _ = dx^
    _ = dmeans^
    _ = dprec^
    _ = dlogdet^
    _ = dlw^
    _ = dlinv^
    _ = mahal^
    _ = wlp^
    _ = rowmax^
    _ = lse^
    _ = logresp^
    _ = resp^
    _ = meanll^
    _ = escratch^
    _ = gws^
    return out^


def _exp_resp(v: Float32) -> Float32:
    """`exp` of a log responsibility, on the HOST, through the same helper
    the device kernel uses.

    `predict_proba` is the one entry that returns probabilities rather than
    logs, and its exponential is taken here rather than on the device because
    the values are already on their way to the host. `identical_exp` and
    `ftz` all the same, so the answer is the same bits `resp_exp_kernel`
    would have produced -- otherwise a caller comparing `predict_proba`
    against a card's `resp` stage would find a difference with no cause.
    """
    from mojo_only.numerics import identical_exp

    return ftz(identical_exp(ftz(v)))


def gaussian_mixture_predict(
    model: GaussianMixtureModel,
    x: List[Float32],
    n_samples: Int,
    elem_tpb: Int = GMM_ELEM_TPB,
    row_tpb: Int = GMM_ROW_TPB,
    card_prefix: String = "gmm.predict",
    trace_path: String = "",
) raises -> List[Int32]:
    """`predict(X)`, `_base.py:395-412`: `argmax_k` of the weighted log
    probabilities. Ties go to the LOWEST component index, which is
    `xp.argmax`'s documented rule and `argmax_kernel`'s."""
    var d = model.n_features
    var ncomp = model.n_components
    gmm_validate_data(x, n_samples, d)

    var ctx = DeviceContext()
    # **DISABLED UNLESS ASKED, and this is the opposite default from
    # `gaussian_mixture_fit`.** A scoring entry is called in a LOOP -- once
    # per point by `check_launch_invariance`'s alone-versus-batch arm -- and
    # every call records the same six tags, which `IdentityTrace._emit`
    # refuses as a duplicate. So an environment variable cannot be allowed to
    # turn the card on here; a caller that wants a scoring card names the
    # file and takes responsibility for calling once.
    var trace: IdentityTrace
    if trace_path == "":
        trace = IdentityTrace.disabled()
    else:
        trace = IdentityTrace.to_path(trace_path)
    var dx = _upload(ctx, x)
    var dmeans = _upload(ctx, model.means)
    var dprec = _upload(ctx, model.precisions_cholesky)
    var dlogdet = _upload(ctx, model.log_det_chol)
    var lw = List[Float32]()
    for k in range(ncomp):
        lw.append(_safe_log(model.weights[k]))
    var dlw = _upload(ctx, lw)
    var dlinv = _upload(ctx, model.precisions_cholesky)

    var mahal = ctx.enqueue_create_buffer[DType.float32](n_samples * ncomp)
    var wlp = ctx.enqueue_create_buffer[DType.float32](n_samples * ncomp)
    var rowmax = ctx.enqueue_create_buffer[DType.float32](n_samples)
    var lse = ctx.enqueue_create_buffer[DType.float32](n_samples)
    var logresp = ctx.enqueue_create_buffer[DType.float32](
        n_samples * ncomp
    )
    var meanll = ctx.enqueue_create_buffer[DType.float32](1)
    var labels = ctx.enqueue_create_buffer[DType.int32](n_samples)
    var escratch = ctx.enqueue_create_buffer[DType.float32](
        gmm_estep_scratch_floats(n_samples, d)
    )
    var gws = ctx.enqueue_create_buffer[DType.float32](
        gmm_estep_gemm_workspace_floats(n_samples, d)
    )
    ctx.synchronize()

    gmm_e_step(
        ctx, dx, dmeans, dprec, dlinv, dlogdet, dlw, escratch, gws,
        mahal, wlp, rowmax, lse, logresp, meanll, n_samples, d, ncomp,
        trace, card_prefix, elem_tpb, row_tpb, GMM_SAB_NONE,
    )
    gmm_predict_labels(
        ctx, wlp, labels, n_samples, ncomp, trace, card_prefix, row_tpb
    )
    var h = ctx.enqueue_create_host_buffer[DType.int32](n_samples)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=labels)
    ctx.synchronize()
    var out = List[Int32]()
    for i in range(n_samples):
        out.append(h.unsafe_ptr().unsafe_load(i))
    _ = h^
    _ = dx^
    _ = dmeans^
    _ = dprec^
    _ = dlogdet^
    _ = dlw^
    _ = dlinv^
    _ = mahal^
    _ = wlp^
    _ = rowmax^
    _ = lse^
    _ = logresp^
    _ = meanll^
    _ = labels^
    _ = escratch^
    _ = gws^
    return out^


def gaussian_mixture_score(
    model: GaussianMixtureModel, x: List[Float32], n_samples: Int
) raises -> Float32:
    """`score(X)`, `_base.py:375-393`: the mean of `score_samples`.

    Folded ASCENDING through `ftz` and divided by `n` with one
    `identical_div`, which is `meanll_kernel`'s arithmetic on the host. It is
    on the host because `bic` and `aic` need the value as a host scalar and
    a second device fold for a number that is about to be read back is a fold
    shape nobody needs.
    """
    from mojo_only.numerics import identical_div

    var s = gaussian_mixture_score_samples(model, x, n_samples)
    var acc = Float32(0.0)
    for i in range(n_samples):
        acc = ftz(acc + ftz(s[i]))
    return ftz(identical_div(acc, Float32(n_samples)))


def gaussian_mixture_bic(
    model: GaussianMixtureModel, x: List[Float32], n_samples: Int
) raises -> Float32:
    """`bic(X)`, `_gaussian_mixture.py:959-980`:

        -2 * score(X) * n_samples + n_parameters * log(n_samples)

    `identical_log` on `Float32(n_samples)` rather than a host `math.log`
    (IDENTITY_PATHS row 18's class), and the products through
    `identical_mul` so no codegen contracts them into the sum. The parameter
    COUNT is integer arithmetic (`gmm_oracle.mojo::n_parameters`), where
    scikit-learn divides by `2.0` in float and casts.
    """
    from mojo_only.numerics import identical_log, identical_mul

    var sc = gaussian_mixture_score(model, x, n_samples)
    var p = gmm_n_parameters(model.n_features, model.n_components)
    var a = ftz(
        identical_mul(
            Float32(-2.0), ftz(identical_mul(sc, Float32(n_samples)))
        )
    )
    var b = ftz(
        identical_mul(
            Float32(p), ftz(identical_log(Float32(n_samples)))
        )
    )
    return ftz(a + b)


def gaussian_mixture_aic(
    model: GaussianMixtureModel, x: List[Float32], n_samples: Int
) raises -> Float32:
    """`aic(X)`, `_gaussian_mixture.py:982-998`:

        -2 * score(X) * n_samples + 2 * n_parameters
    """
    from mojo_only.numerics import identical_mul

    var sc = gaussian_mixture_score(model, x, n_samples)
    var p = gmm_n_parameters(model.n_features, model.n_components)
    var a = ftz(
        identical_mul(
            Float32(-2.0), ftz(identical_mul(sc, Float32(n_samples)))
        )
    )
    return ftz(a + ftz(identical_mul(Float32(2.0), Float32(p))))


def gmm_mode_name() -> String:
    """`IDENTICAL` or `FAST`, for a banner. The mode is a BUILD DEFINE
    (`-D MOJOLEARN_NUMERIC_IDENTICAL=1`), never an edited line."""
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        return String("IDENTICAL")
    return String("FAST")
