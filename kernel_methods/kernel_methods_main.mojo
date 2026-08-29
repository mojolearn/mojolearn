"""kernel_methods driver: one fit of each estimator, and the identity card.

    MOJOLEARN_IDENTITY_TRACE=/tmp/mac.km.card \\
        tools/with_build_lock.sh pixi run mojo run -I . kernel_methods/kernel_methods_main.mojo
    MOJOLEARN_IDENTITY_TRACE=/tmp/mac.km.identical.card \\
        tools/with_identical_mode.sh pixi run mojo run -I . kernel_methods/kernel_methods_main.mojo

    python3 tools/identity_trace_diff.py /tmp/mac.km.identical.card /tmp/<other>.km.identical.card

Environment knobs, all optional: `MOJOLEARN_KM_FIXTURE` (`ortho`, `rbf`,
`dup`, `signed`, `mixed`; default `rbf`), `MOJOLEARN_KM_KERNEL` (`linear`,
`polynomial`, `rbf`, `sigmoid`, `laplacian`; default `rbf`),
`MOJOLEARN_KM_COMPONENTS` (Nystroem and RBFSampler width; default 8),
`MOJOLEARN_KM_SEED` (default 20260825).

THE CARD, and its ORDER is the point rather than its length. Stages, in the
sequence a divergence would first show up in:

    KERNEL RIDGE
      krr.input               X as uploaded, before anything
      krr.kernel              K(X, X)
      krr.ridged              after K += alpha I (DEVIATION 1660)
      chol.panel000.{factored,solved,trailing} ...    potrf_lower's own
      chol.factor, chol.nb, chol.diag, chol.logdet
      chol.solve.forward, chol.solve.back
      krr.dual_coef           the dual coefficients
      krr.cross_kernel        K(X_new, X_fit)
      krr.predictions         K . dual_coef

    NYSTROEM
      nys.basis_indices       the sampled row ids, Int32, in RANK order
      nys.basis_kernel        K(basis, basis)
      nys.eigenvectors_flipped   after jacobi_eigh_kernel + sign_flip_kernel
      nys.eigenvalues         DESCENDING and clipped
      nys.sqrt_eigenvalues
      nys.eigenvectors        permuted into the eigenvalue order
      nys.scaled              Q / sqrt(s), column by column
      nys.normalization       (Q / sqrt(s)) . Q^T
      nys.cross_kernel        K(X_new, basis)
      nys.embedding           cross_kernel . normalization^T

    RBFSAMPLER
      rf.weights              W, n_features x n_components
      rf.offsets              b
      rf.sigma, rf.scale      the two host constants (DEVIATION 1678)
      rf.projection           X W
      rf.feature_map          sqrt(2/D) cos(X W + b)

**A CARD THAT DIVERGES HAS AN ADDRESS AND THE ADDRESS IS THE DIAGNOSIS.**

- `krr.input` moving means the FIXTURE moved, not the estimator. Nothing
  below it is comparable.
- `krr.kernel` moving with `krr.input` identical is the kernel matrix: the
  GEMM (which is the gemm lane's certificate, not this one's), the RBF
  expansion's `identical_exp`, the sigmoid's `identical_tanh`, the
  polynomial's repeated product, or the laplacian's L1 distance.
- `krr.ridged` moving with `krr.kernel` identical means `alpha` moved, or the
  ridge became relative (DEVIATION 1660's forbidden state).
- Any `chol.*` moving is the CHOLESKY lane's, and its own card documentation
  says which stage means what.
- `krr.predictions` moving with `krr.cross_kernel` and `krr.dual_coef`
  identical is the `OP_NN` product and nothing else.
- `nys.basis_indices` moving means the POSITION MAP moved -- a different
  seed, a different kind byte, or a stream-shaped draw. It is Int32 and it
  cannot move for a floating-point reason at all, so a difference here is a
  difference in `resample/mojo_only/index_map.mojo` or in this lane's rank
  pass.
- `nys.eigenvectors_flipped` moving with `nys.basis_kernel` identical is the
  JACOBI or the SIGN FLIP, both of which are `decomposition/`'s. Check
  `NystroemModel.sweeps` first: **two runs with different sweep counts are
  not comparable below this stage at all**, exactly as two Cholesky runs with
  different `nb` are not, and that is DEVIATION BLOCK 3 of
  `jacobi_eigh_device.mojo` rather than a caveat of ours.
- `nys.eigenvalues` moving with `nys.eigenvectors_flipped` identical is the
  ORDER or the CLIP, and nothing else.
- `nys.normalization` moving with `nys.scaled` identical is the `OP_NT`
  product.
- `nys.embedding` moving with both identical is the transpose (DEVIATION
  1674).
- `rf.weights` moving is the DRAW: the position map, the Box-Muller
  transform's four transcendentals, or `sigma`.
- `rf.feature_map` moving with `rf.projection` identical is `identical_cos`
  and nothing else.

**NO CARD HAS BEEN EMITTED.** The stage list above is what the source
records, not a transcript.

Prints scalars as decimal AND hex, because `String(Float32)` does not round
trip (`[[mojo-string-float-roundtrip]]`).
"""

from std.os import getenv

from core.identity_trace import IdentityTrace
from kernel_methods.estimator import (
    kernel_ridge_fit_host,
    kernel_ridge_predict_host,
    nystroem_fit_host,
    nystroem_transform_host,
    rbf_sampler_fit_host,
    rbf_sampler_transform_host,
)
from kernel_methods.mojo_only.km_fixture import (
    FIX_KM_DUP,
    FIX_KM_MIXED,
    FIX_KM_ORTHO,
    FIX_KM_RBF,
    FIX_KM_SIGNED,
    km_fixture_d,
    km_fixture_n,
    km_fixture_name,
    km_fixture_query,
    km_fixture_x,
    km_fixture_y,
    km_hex32,
)
from kernel_methods.mojo_only.kernel_matrix import (
    km_kernel_from_name,
    km_kernel_name,
    KM_KERNEL_RBF,
)
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name
from svm.ported.svm.svm_parameter import KernelParams


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29. This used to
    be a local two-way `IDENTICAL`-or-`FAST`, written when there were
    two tiers, and it answered "FAST" for a DETERMINISTIC build -- so
    a driver run under the middle tier printed the wrong arm onto
    every line it produced. A correctly-labelled measurement of the
    wrong arm is the failure this tree has been bitten by repeatedly,
    and forty-four copies of a mode label is how it happens.
    """
    return numeric_mode_name()


def _fixture_from_env() -> Int:
    var s = String(getenv("MOJOLEARN_KM_FIXTURE"))
    if s == "ortho":
        return FIX_KM_ORTHO
    if s == "dup":
        return FIX_KM_DUP
    if s == "signed":
        return FIX_KM_SIGNED
    if s == "mixed":
        return FIX_KM_MIXED
    return FIX_KM_RBF


def _kernel_from_env() raises -> Int:
    var s = String(getenv("MOJOLEARN_KM_KERNEL"))
    if s == "":
        return KM_KERNEL_RBF
    return km_kernel_from_name(s)


def main() raises:
    var which = _fixture_from_env()
    var kernel = _kernel_from_env()
    var n = km_fixture_n(which)
    var d = km_fixture_d(which)

    var q = 8
    var q_env = String(getenv("MOJOLEARN_KM_COMPONENTS"))
    if q_env != "":
        q = Int(atol(q_env))
    if q > n:
        q = n

    var seed = UInt64(20260825)
    var seed_env = String(getenv("MOJOLEARN_KM_SEED"))
    if seed_env != "":
        seed = UInt64(atol(seed_env))

    # `gamma` and `alpha` are POWERS OF TWO on purpose: a printed number that
    # is itself a rounding of `1 / n_features` makes every downstream hex
    # value harder to reason about, and the card is meant to be read.
    var gamma = 0.25
    var alpha = Float32(0.5)
    var kp = KernelParams(kernel, 3, gamma, 1.0)

    print(
        "== kernel_methods/kernel_methods_main.mojo ["
        + _mode_name()
        + "] fixture="
        + km_fixture_name(which)
        + " kernel="
        + km_kernel_name(kernel)
        + " n="
        + String(n)
        + " d="
        + String(d)
        + " n_components="
        + String(q)
        + " alpha="
        + km_hex32(alpha)
        + " seed="
        + String(seed)
        + " =="
    )

    var x = km_fixture_x(which, 0)
    var y = km_fixture_y(which, 1, 0)
    var n_query = 3
    var xq = km_fixture_query(which, n_query, 0)

    var trace = IdentityTrace()
    trace.header(
        "kernel_methods: mode="
        + _mode_name()
        + " fixture="
        + km_fixture_name(which)
        + " kernel="
        + km_kernel_name(kernel)
        + " n="
        + String(n)
        + " d="
        + String(d)
        + " q="
        + String(q)
        + " alpha_bits="
        + km_hex32(alpha)
        + " seed="
        + String(seed)
    )

    # ---- KernelRidge ----
    var krr = kernel_ridge_fit_host(x, y, n, d, 1, kp, alpha, trace)
    var pred = kernel_ridge_predict_host(krr, xq, n_query, trace)
    print(
        "  kernel ridge: info="
        + String(krr.info)
        + " dual[0]="
        + km_hex32(krr.dual_coef[0])
        + " ("
        + String(krr.dual_coef[0])
        + ") prediction[0]="
        + km_hex32(pred[0])
        + " ("
        + String(pred[0])
        + ")"
    )
    _ = krr^

    # ---- Nystroem ----
    var nys = nystroem_fit_host(x, n, d, kp, q, seed, trace)
    var emb = nystroem_transform_host(nys, xq, n_query, trace)
    print(
        "  nystroem: n_components="
        + String(nys.n_components)
        + " jacobi_sweeps="
        + String(nys.sweeps)
        + " basis[0]="
        + String(Int(nys.component_indices[0]))
        + " eigenvalue[0]="
        + km_hex32(nys.eigenvalues[0])
        + " ("
        + String(nys.eigenvalues[0])
        + ") embedding[0]="
        + km_hex32(emb[0])
    )
    print(
        "    THE SWEEP COUNT IS A NUMERIC PARAMETER, NOT A DIAGNOSTIC. Two"
        " runs reporting different sweep counts are not comparable below"
        " nys.eigenvectors_flipped at all"
        " (decomposition/mojo_only/jacobi_eigh_device.mojo, DEVIATION BLOCK"
        " 3)."
    )
    _ = nys^

    # ---- RBFSampler ----
    var rf = rbf_sampler_fit_host(d, q, Float32(gamma), seed, trace)
    var z = rbf_sampler_transform_host(rf, xq, n_query, trace)
    print(
        "  rbf sampler: n_components="
        + String(rf.n_components)
        + " sigma="
        + km_hex32(rf.sigma)
        + " scale="
        + km_hex32(rf.scale)
        + " W[0]="
        + km_hex32(rf.random_weights[0])
        + " b[0]="
        + km_hex32(rf.random_offset[0])
        + " z[0]="
        + km_hex32(z[0])
    )
    _ = rf^

    print(
        "  card: set MOJOLEARN_IDENTITY_TRACE to a path to write it;"
        " tools/identity_trace_diff.py compares two."
    )
