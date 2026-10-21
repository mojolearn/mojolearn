# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""What the Gaussian process classifier's GPU path
(`gaussian_process/classifier.mojo`) and its host column
(`gaussian_process/host/gpc_oracle.mojo`, `gpc_steps.mojo`) share that is
neither a kernel nor host arithmetic: the fit and latent records, DEVIATION
2830's stop rule and its constants, and the input checks at the binding
boundary (the role `gaussian_process/estimator.mojo::gp_validate_data` has
for the regressor). Moved out of `host/gpc_steps.mojo` (cpu-gpu-cleanup
c-gp-kernel, 2026-10-02) so the GPU binding imports no host module. GPU-free.
"""

from std.memory import bitcast

from checks.numerics import ftz

#: DEVIATION 2830: float32(1e-10), the reference's tolerance at this width.
comptime GPC_LML_TOL_BITS: UInt32 = 0x2EDBE6FF
comptime GPC_NEG_INF32_BITS: UInt32 = 0xFF800000


@fieldwise_init
struct GPCBinaryFit(Movable):
    """One binary Laplace fit: sklearn's `L_`, `pi_`, `W_sr_` and
    `log_marginal_likelihood_value_`, plus the iteration count (DEVIATION
    2830) and the Cholesky panel width that ran."""

    var l: List[Float32]
    var pi: List[Float32]
    var wsr: List[Float32]
    var lml: Float32
    var n_iter: Int
    var nb: Int


@fieldwise_init
struct GPCLatent(Movable):
    """`latent_mean_and_variance` at the query rows. `variance` is empty
    when only the mean was asked for (`predict`). `proba` is the class-1
    probability (DEVIATION 2832) when the GPU path computed it with the
    variance (`gpc_proba64.mojo`), and empty from the host oracle, whose
    binding calls `gpc_steps.mojo::gpc_proba`."""

    var mean: List[Float32]
    var variance: List[Float32]
    var proba: List[Float64]


def gpc_lml_tol() -> Float32:
    return bitcast[DType.float32](GPC_LML_TOL_BITS)


def gpc_neg_inf32() -> Float32:
    return bitcast[DType.float32](GPC_NEG_INF32_BITS)


# ===========================================================================
# VALIDATION (host, before any launch)
# ===========================================================================


def gpc_validate_labels(y: List[Float32], n_train: Int) raises:
    """The binary targets: `n_train` values, each exactly 0 or 1, both
    present. The Python surface encodes labels (and one-vs-rest columns)
    before this; a single class is refused by name there with the
    reference's sentence (`_gpc.py:198-203`), and again here."""
    if len(y) != n_train:
        raise Error(
            "gpc_fit_host: y holds "
            + String(len(y))
            + " values for "
            + String(n_train)
            + " training rows"
        )
    var zeros = 0
    var ones = 0
    for i in range(n_train):
        var v = y[i]
        if v == Float32(0.0):
            zeros += 1
        elif v == Float32(1.0):
            ones += 1
        else:
            raise Error(
                "gpc_fit_host: the binary target at index "
                + String(i)
                + " is not 0 or 1; the surface encodes the classes before"
                " the binding is reached, so this boundary disagrees with it"
            )
    if zeros == 0 or ones == 0:
        raise Error(
            "gpc_fit_host: a binary Laplace fit requires 2 classes; got 1"
            " class. scikit-learn refuses the same data (_gpc.py:207-212)"
        )


def gpc_validate_max_iter(max_iter_predict: Int) raises:
    if max_iter_predict < 1:
        raise Error(
            "gpc_fit_host: max_iter_predict must be at least 1, got "
            + String(max_iter_predict)
            + " (scikit-learn's Interval(Integral, 1, None))"
        )


def gpc_stop(lml: Float32, previous: Float32) -> Bool:
    """DEVIATION 2830's test, `lml - previous < 1e-10` at float32."""
    return ftz(lml - previous) < gpc_lml_tol()
