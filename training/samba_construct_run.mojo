# SPDX-License-Identifier: Apache-2.0
"""RUN half of the Samba constructor door (bindings/samba_construct_native_pack.mojo):
the seeded parameter initialization of one stack over typed words. No Python
import (tools/python_free_dirs.sh); the caller releases the GIL around the
one call.

Serial by nature: a driver loop over the stack's registry tensors in saved
order, each iteration one launch of the training image's RNG primitive (op
6), which is the parallel kernel; the stream counter each draw consumes is
part of the generator's public state and advances in registry order.
"""
from std.math import sqrt
from core.entry_table import EntryRef, entry_ref_call
from core.samba_entry_table import SAMBA_PRIM_NEURAL_RNG
from core.samba_construct_native_impl import samba_fill_ones


def samba_initialize_run(entry: EntryRef, names: List[String], addresses: List[Int], sizes: List[Int],
    ranks: List[Int], columns: List[Int], seed_low: Int, seed_high: Int, counter: Int, unused: Int,
    mut drawn: Int) raises:
    """Initialize every registry tensor at `addresses` (one per name, with
    its element count, rank and second dimension). `counter` is the
    generator's stream counter before the call; `drawn` counts the streams
    consumed so far, also when the call raises, so the caller publishes
    `counter + drawn` either way (a stream is consumed before its launch,
    as the per-tensor draw did). `unused` is a one-element float32 scratch
    address the RNG entry's second span requires."""
    # Keep the legacy float64 sequence (including sqrt(5), then gain), rather
    # than simplifying the bound algebraically. Both supported host baselines
    # use correctly rounded hardware sqrt, as portable_math.c does.
    var a = sqrt(Float64(5.0))
    var square = a * a
    var gain = sqrt(Float64(2.0) / (Float64(1.0) + square))
    for i in range(len(names)):  # driver-loop: over initializer calls in the saved registry order
        var kind = 0
        var first = Float32(0.0)
        var second = Float32(0.0)
        var random = True
        if names[i] == "embed.weight" or names[i] == "lm_head.weight":
            kind = 1
            second = Float32(0.02)
        elif names[i].endswith(".weight") and ranks[i] == 2:
            var bound = gain * sqrt(Float64(3.0) / Float64(columns[i]))
            first = Float32(-bound)
            var high_bound = Float32(bound)
            second = Float32(Float64(high_bound) - Float64(first))
        elif names[i].endswith(".dt_bias"):
            first = -4.0
            second = 2.0
        else:
            random = False
            samba_fill_ones(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addresses[i]), sizes[i])
        if random:
            var stream = counter + drawn
            if stream >= 4294967296:
                raise Error("RuntimeError: Generator: the stream counter is exhausted")
            if stream < 0:
                raise Error("Generator: counter must be nonnegative")
            drawn += 1
            _ = entry_ref_call(entry, SAMBA_PRIM_NEURAL_RNG, [addresses[i], unused],
                [sizes[i], 0, seed_low, seed_high, stream, kind], [Float64(first), Float64(second)])
