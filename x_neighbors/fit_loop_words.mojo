# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The fit loop clock in a resident loop op's int32 info words (lane
ab-loop-mode / alm-iter, 2026-11-24). LabelPropagation and LabelSpreading's
loop ops (`op_lp_iterate`, `op_lp_iterate_knn`, the device and host columns)
write n_iter_ and converged to info words 0 and 1; words 2 and 3 carry the
loop region's nanoseconds (core/fit_loop_clock.mojo), low then high 32 bits,
which the fit publishes as `fit_loop_`. The caller's info buffer holds
XN_LOOP_INFO_WORDS words. Measurement plumbing only: no computed word
changes."""

from core.fit_loop_clock import FitLoop

#: int32 words of a loop op's info buffer: n_iter_, converged, then the loop
#: nanoseconds' low and high halves.
comptime XN_LOOP_INFO_WORDS = 4


def xn_loop_words_store(info: Int, loop: FitLoop):
    """Write the loop's nanoseconds to info words 2 (low 32 bits, two's
    complement) and 3 (high bits)."""
    var words = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=info)
    var low = loop.loop_ns & 0xFFFFFFFF
    words[2] = Int32(low - 4294967296 if low >= 2147483648 else low)
    words[3] = Int32(loop.loop_ns >> 32)
