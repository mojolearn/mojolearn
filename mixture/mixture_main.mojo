"""Mixture driver: one fit, the identity card, and the iteration count.

    MOJOLEARN_IDENTITY_TRACE=/tmp/mac.gmm.card \\
        tools/with_build_lock.sh pixi run mojo run -I . mixture/mixture_main.mojo
    MOJOLEARN_IDENTITY_TRACE=/tmp/mac.gmm.identical.card \\
        tools/with_identical_mode.sh pixi run mojo run -I . mixture/mixture_main.mojo

    python3 tools/identity_trace_diff.py \\
        /tmp/mac.gmm.identical.card /tmp/<other>.gmm.identical.card

Environment knobs, all optional: `MOJOLEARN_GMM_FIXTURE` (`separated`,
`overlap`, `collapse`, `duplicates`, `one_d`, `signed_zero`; default
`separated`), `MOJOLEARN_GMM_MAX_ITER` (default 50),
`MOJOLEARN_GMM_INIT` (`kmeans` or `random`; default `kmeans`),
`MOJOLEARN_GMM_SEED` (default 0).

**THE ITERATION COUNT IS THE HEADLINE OF THIS CARD, NOT THE PARAMETERS.**
`gmm.niter` carries `[n_iter, converged, max_iter]` as Int32, and it is the
FIRST thing to compare between two cards. Two runs that took different
numbers of iterations have no parameters worth comparing at all: every stage
after the first differing `gmm.iterNNN.*` belongs to a different iteration of
a different trajectory, and a differ aligning them by tag sequence will
report a structural divergence rather than a numeric one. That is the
correct report and it is why `core/identity_trace.mojo::first_divergence`
says "go run tools/identity_trace_diff.py" on a length mismatch instead of
diffing by position.

THE CARD, and its ORDER is the product rather than its length:

    gmm.input                     X as uploaded, before anything
    gmm.init.resp0                the initial responsibilities (one-hot from
                                  cluster/'s k-means, or the position-mapped
                                  Philox draws)
    gmm.init.resp                 exp(log(the above)), the first M-step's own
                                  input -- a separate tag because
                                  IdentityTrace's uniqueness invariant
                                  forbids two records under one name, and
                                  because they are different quantities
    gmm.init.nk                   the component masses
    gmm.init.weights              nk / n_samples          (_base.py:864)
    gmm.init.means
    gmm.init.covariances
    gmm.init.comp000.cholesky     L_0 of component 0's covariance
    gmm.init.comp000.precchol     (L_0^{-1})^T
    ...                           one pair per component
    gmm.init.logdet               -0.5 * 2 sum log L_jj, per component
    gmm.iter001.mahal             the Mahalanobis distances, n x K
    gmm.iter001.wlp               log P(x|k) + log pi_k
    gmm.iter001.rowmax            the logsumexp's row maxima (row 39's site)
    gmm.iter001.lse               log p(x)
    gmm.iter001.logresp           the log responsibilities
    gmm.iter001.meanll            the mean log likelihood
    gmm.iter001.resp              exp(log_resp), the M-step's input
    gmm.iter001.nk
    gmm.iter001.weights           nk / sum(nk)            (_base.py:898)
    gmm.iter001.means
    gmm.iter001.covariances
    gmm.iter001.comp000.cholesky  ... one pair per component
    gmm.iter001.logdet
    gmm.iter001.change            THE CONVERGENCE QUANTITY
    ...                           one block per iteration
    gmm.niter                     [n_iter, converged, max_iter]

**A CARD THAT DIVERGES HAS AN ADDRESS AND THE ADDRESS IS THE DIAGNOSIS.**

- `gmm.input` moving means the fixture moved, not the fit.
- `gmm.init.resp0` moving means the INITIALIZATION moved. With
  `init_params="kmeans"` that is `cluster/`'s certificate and not this
  lane's; run `pixi run check-kmeans-identity` before reading anything
  below it.
- `gmm.iterNNN.mahal` moving with `precchol` identical means the E-step's
  GEMM or its fold moved. The GEMM is the gemm lane's certificate
  (`mojolearn.identical.gemm.fp32.v1`); the fold is DEVIATION 1728 and is
  this lane's.
- `gmm.iterNNN.rowmax` moving is IDENTITY_PATHS row 39 and nothing else: a
  hardware `max` crept into a positional compare, or a signed zero reached
  the row max. Its sign bit is the whole message.
- `gmm.iterNNN.lse` moving with `rowmax` identical is `identical_exp` or
  `identical_log` (row 12), or the summation order (DEVIATION 1727).
- `gmm.iterNNN.meanll` moving with `lse` identical is the one-thread
  ascending fold, DEVIATION 1732 -- **and it is the one that will change the
  iteration count**.
- `gmm.iterNNN.change` moving with `meanll` identical is impossible unless
  the previous iteration's `meanll` moved, so it localizes to the ITERATION
  rather than to the quantity.
- `gmm.iterNNN.compKKK.cholesky` moving is the Cholesky lane's certificate
  (`mojolearn.identical.cholesky.fp32.v1`), not this one's; run
  `pixi run check-cholesky` first.
- `gmm.niter` differing means the two runs did not take the same number of
  iterations and NOTHING below the first differing iteration is comparable.

**No card has been emitted.** The list above is what the source records, not
a transcript.

Prints every scalar as decimal AND hex, because `String(Float32)` does not
round trip (`[[mojo-string-float-roundtrip]]`).
"""

from std.memory import bitcast
from std.os import getenv

from mixture.estimator import (
    INIT_KMEANS,
    INIT_RANDOM,
    COV_FULL,
    GmmParams,
    gaussian_mixture_aic,
    gaussian_mixture_bic,
    gaussian_mixture_fit,
    gaussian_mixture_predict,
    gaussian_mixture_score,
    gmm_mode_name,
)
from mixture.mojo_only.estep import GMM_PROFILE
from mixture.mojo_only.gmm_fixture import (
    FIX_COLLAPSE,
    FIX_DUPLICATES,
    FIX_ONE_D,
    FIX_OVERLAP,
    FIX_SEPARATED,
    FIX_SIGNED_ZERO,
    gmm_fixture,
    gmm_fixture_d,
    gmm_fixture_k,
    gmm_fixture_n,
    gmm_fixture_name,
)


def _hex32(v: Float32) -> String:
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


def _fixture_from_env() -> Int:
    var s = String(getenv("MOJOLEARN_GMM_FIXTURE"))
    if s == "overlap":
        return FIX_OVERLAP
    if s == "collapse":
        return FIX_COLLAPSE
    if s == "duplicates":
        return FIX_DUPLICATES
    if s == "one_d":
        return FIX_ONE_D
    if s == "signed_zero":
        return FIX_SIGNED_ZERO
    return FIX_SEPARATED


def main() raises:
    var which = _fixture_from_env()
    var n = gmm_fixture_n(which)
    var d = gmm_fixture_d(which)
    var ncomp = gmm_fixture_k(which)

    var params = GmmParams.default()
    params.n_components = ncomp
    params.covariance_type = COV_FULL
    params.max_iter = 50
    var mi = String(getenv("MOJOLEARN_GMM_MAX_ITER"))
    if mi != "":
        params.max_iter = Int(atol(mi))
    params.init_params = INIT_KMEANS
    if String(getenv("MOJOLEARN_GMM_INIT")) == "random":
        params.init_params = INIT_RANDOM
    var sd = String(getenv("MOJOLEARN_GMM_SEED"))
    if sd != "":
        params.random_state = UInt64(atol(sd))
    # THE COLLAPSE FIXTURE RUNS AT reg_covar = +0.0 ON PURPOSE. Its third
    # component is six bit-identical copies of one point, so its
    # maximum-likelihood covariance is exactly the zero matrix and any
    # positive ridge would rescue it. DEVIATION 1723's refusal is what this
    # arm is for, and it fires at the INITIALIZATION rather than at an EM
    # iteration.
    if which == FIX_COLLAPSE:
        params.reg_covar = Float32(0.0)

    print(
        "== mixture/mixture_main.mojo ["
        + gmm_mode_name()
        + "] profile="
        + GMM_PROFILE
        + " fixture="
        + gmm_fixture_name(which)
        + " n="
        + String(n)
        + " d="
        + String(d)
        + " n_components="
        + String(ncomp)
        + " max_iter="
        + String(params.max_iter)
        + " tol="
        + _hex32(params.tol)
        + " reg_covar="
        + _hex32(params.reg_covar)
        + " =="
    )

    var x = gmm_fixture(which)

    var failed = False
    var message = String("")
    var n_iter = 0
    var converged = False
    var lower_bound = Float32(0.0)
    try:
        var model = gaussian_mixture_fit(x, n, d, params)
        n_iter = model.n_iter
        converged = model.converged
        lower_bound = model.lower_bound
        print(
            "  n_iter="
            + String(model.n_iter)
            + "  converged="
            + String(model.converged)
            + "  lower_bound="
            + String(model.lower_bound)
            + "  "
            + _hex32(model.lower_bound)
        )
        if not model.converged and params.max_iter > 0:
            # scikit-learn emits a ConvergenceWarning here (`_base.py:292`).
            # Mojo has no warning channel, so DEVIATION 1746 makes it a
            # printed line and a field, never a raise: their fit returns a
            # usable model in this case and so does ours.
            print(
                "  NOT CONVERGED after "
                + String(params.max_iter)
                + " iterations. scikit-learn raises a ConvergenceWarning"
                " here; DEVIATION 1746 makes it this line plus"
                " GaussianMixtureModel.converged, because the model is"
                " usable and refusing would not be their semantics"
            )
        for k in range(model.n_components):
            print(
                "    weight["
                + String(k)
                + "] = "
                + _hex32(model.weights[k])
                + "  logdet_chol = "
                + _hex32(model.log_det_chol[k])
            )
            var line = String("    mean[") + String(k) + "] ="
            for j in range(model.n_features):
                line += " " + _hex32(model.means[k * model.n_features + j])
            print(line)
        var sc = gaussian_mixture_score(model, x, n)
        var bic = gaussian_mixture_bic(model, x, n)
        var aic = gaussian_mixture_aic(model, x, n)
        print(
            "  score="
            + _hex32(sc)
            + "  bic="
            + _hex32(bic)
            + "  aic="
            + _hex32(aic)
        )
        var labels = gaussian_mixture_predict(model, x, n)
        var head = String("  labels[0:8] =")
        var lim = 8 if n > 8 else n
        for i in range(lim):
            head += " " + String(Int(labels[i]))
        print(head)
    except e:
        failed = True
        message = String(e)

    if failed:
        # A COLLAPSE IS NOT A FAILURE OF THIS RUN. It is the DATA-DEPENDENT
        # refusal DEVIATION 1723 exists to make identical, and on the
        # COLLAPSE fixture it is the EXPECTED outcome. The card written so
        # far holds the partial state, and two vendors must produce the same
        # partial card and the same message.
        print("  REFUSED: " + message)

    var trace_path = String(getenv("MOJOLEARN_IDENTITY_TRACE"))
    if trace_path == "":
        print(
            "  no MOJOLEARN_IDENTITY_TRACE set: everything computed, no"
            " card written"
        )
    else:
        if failed:
            print(
                "  PARTIAL card written to "
                + trace_path
                + " (the fit refused; the stages up to and including the"
                " failing component's are on it, and a run that refuses"
                " identically produces an identical card OF THE SAME"
                " LENGTH)"
            )
        else:
            print(
                "  card written to "
                + trace_path
                + " ("
                + String(1 + 1 + 5 + 2 * ncomp + 1)
                + " header and initialization stages, then "
                + String(n_iter)
                + " iterations x "
                + String(13 + 2 * ncomp)
                + " stages, then gmm.niter)"
            )
        print(
            "  COMPARE gmm.niter FIRST: ["
            + String(n_iter)
            + ", "
            + String(1 if converged else 0)
            + ", "
            + String(params.max_iter)
            + "]. Two cards that disagree there have no comparable"
            " parameters below the first differing iteration"
        )
    _ = lower_bound
