# SPDX-License-Identifier: Apache-2.0
"""The typed target normalization of the GP regressor (RUN): enabled flag, standard deviation and mean."""


@fieldwise_init
struct GPRNormalization(Copyable, Movable):
    var enabled: Bool
    var stddev: Float64
    var mean: Float64
