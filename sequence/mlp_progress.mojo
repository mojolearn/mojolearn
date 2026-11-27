# SPDX-License-Identifier: Apache-2.0
"""The portable scalar part of an MLP optimizer checkpoint. Each field occupies one 64-bit word."""

from std.sys.info import size_of


@fieldwise_init
struct MLPProgress(ImplicitlyCopyable, Movable):
    var epochs: Int64
    var samples: Int64
    var no_improvement: Int64
    var seed: Int64
    var shuffle_rng: UInt64
    var rate: Float64
    var beta1_power: Float64
    var beta2_power: Float64
    var best: Float64


comptime MLP_PROGRESS_WORDS = 9

comptime assert size_of[MLPProgress]() == MLP_PROGRESS_WORDS * size_of[Int64](), "invalid MLP checkpoint layout"
