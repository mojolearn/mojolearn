# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The fit loop record across the model descriptor ABI (lane alm-neural,
2026-11-24; parent ab-loop-mode).

A neural trainer (`mlp_fit`, `rnn_fit`, the Sequential and CNN trainers)
runs inside a connected image and is reached through the descriptor call,
which carries addresses, int words and float words only. The boundary
passes the address of `FIT_LOOP_WORDS` Int64 words; the trainer writes its
`FitLoop` there once, after the loop (`fit_loop_publish`), and the boundary
reads it back (`fit_loop_read`) when it publishes `fit_loop_`. Address 0:
no record wanted (every caller that predates the field), nothing written.

Measurement plumbing only: three word stores per fit, after the loop."""

from core.fit_loop_clock import FitLoop

comptime FIT_LOOP_WORDS = 3
comptime FitLoopWordP = MutPointer[Int64, MutUntrackedOrigin]


def fit_loop_publish(address: Int, loop: FitLoop):
    """Writes `loop` as (iterations, loop_ns, recorded) into the three Int64
    words at `address`; address 0 writes nothing."""
    if address == 0:
        return
    var words = FitLoopWordP(unsafe_from_address=address)
    words.unsafe_store(0, Int64(loop.iterations))
    words.unsafe_store(1, Int64(loop.loop_ns))
    words.unsafe_store(2, Int64(1) if loop.recorded else Int64(0))


def fit_loop_words() -> List[Int64]:
    """Three zeroed words for one fit's record (an unrecorded loop)."""
    return List[Int64](length=FIT_LOOP_WORDS, fill=Int64(0))


def fit_loop_read(words: List[Int64]) -> FitLoop:
    """The record a trainer wrote into `words` (unrecorded when it wrote
    nothing)."""
    return FitLoop(Int(words[0]), Int(words[1]), words[2] != 0)
