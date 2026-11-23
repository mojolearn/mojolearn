# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The score-noise scale, written once (AGENTS.md "One body per operation").

GPU-free: no `max.gpu`, no `DeviceContext`, no kernel. The device fit
(`gbdt/methods/doc_parallel_boosting.mojo`, a host-side driver statement) and
every host GBDT owner (`gbdt/host/gbdt_oracle*.mojo`) call this body, so the
two columns compute the multiplier with the same expression.

Lane gbdt-cpu-rest-2 (2026-11-23): before this module the device called
`random_score_helper.calc_score_model_length_mult` (the seams, the
subtraction left to the compiler's contraction) while the host owners held
five restatements, four of them on the host `std.math` exp and log rather
than the seams. Bits change for every fit with `random_strength != 0` on
the pointwise losses, the multi-output losses, the grouped pointwise
searcher and the non-symmetric driver (CPU column), and possibly on the
device column (the explicit fma below is the one rounding the host
restatement of the symmetric Logloss/RMSE loop already took); every column
together.
"""
from std.math import fma

from checks.numerics import identical_exp64, identical_log64


@always_inline
def score_model_length_mult(
    sample_count: Float64, iteration: Int, learning_rate: Float32
) -> Float64:
    """`CalcScoreModelLengthMult` (`random_score_helper.h:18-22`):

        double modelExpLength = log(sampleCount);
        double modelLeft = exp(modelExpLength - modelSize);
        return modelLeft / (1 + modelLeft);

    with `modelSize = iteration * learningRate`
    (`doc_parallel_boosting.h:358-359`). `log(n) - iteration * lr` is ONE
    rounding (an explicit fma, the contraction the default build applied),
    and exp and log go through the identity seams: the scale multiplies the
    split-score noise, so a last-bit difference can flip a split."""
    var model_left = identical_exp64(
        fma(-Float64(iteration), Float64(learning_rate), identical_log64(sample_count))
    )
    return model_left / (1.0 + model_left)
