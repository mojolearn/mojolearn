# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The histogram's per-row quantization: the dither `hist2_dither` and the
dithered fixed-point cell `hist2_quantize`, one body for the device
histogram kernels (`kernel/histogram_utils.mojo` and the hist kernels that
imported them from it) and the CPU column (`gbdt/host/gbdt_oracle.mojo`,
`gbdt_oracle_ordered.mojo`), which restated both (lane gbdt-leaf-body,
2026-11-23). GPU-free, so the host bindings import it."""
from std.math import floor

from checks.numerics import ftz, identical_mul


@always_inline
def hist2_dither(pos: Int) -> Float32:
    """A deterministic hash of the DOCUMENT POSITION, uniform in [0, 1).

    The dither for `hist2_quantize`. Keyed on the row's position so it is
    deterministic run to run and vendor to vendor (it depends on the data
    layout, never on the launch geometry), and so the same row quantizes
    IDENTICALLY at every tree depth, which keeps parent = child + sibling
    exact to the integer."""
    var h = UInt32(pos) * UInt32(2654435761)
    h ^= h >> 16
    h = h * UInt32(2246822519)
    h ^= h >> 13
    # / 2^24, pinned: callers add it (`frac + u`) and the PTX left a plain
    # mul.f32 feeding that add.f32, which ptxas may fuse (lane/pinned-mul-contract-free)
    return identical_mul(Float32(Int(h >> 8)), Float32(5.9604645e-08))


@always_inline
def hist2_quantize(val: Float32, fixed_scale: Float32, u: Float32) -> Int32:
    """DETERMINISTIC DITHERED QUANTIZATION, `floor(val*scale + u(pos))` --
    and every word of that is load-bearing, because both simpler rules
    FAILED MEASURABLY:

      TRUNCATION (`Int32(val * scale)`) loses up to `1/scale` per row,
      always toward zero, so a leaf's histogram total ran ~0.16% short of
      the EXACT `partStats` total that `compute_scores` subtracts it from
      (`sumRight = partStat - sumLeft`, `compute_scores.cu:97-99`). The
      deficit surfaces as phantom mass past every bin, largest at the last
      one; measured at 300k-650k rows x 100 features, the fit locked onto
      (feature 98, bin 127) on three DIFFERENT datasets and stopped at
      depth 1.

      ROUND-TO-NEAREST is still biased whenever a stat VALUE repeats (a
      unit weight plane repeats it 800,000 times): every 1.0 became
      537/536.87, measured +52 phantom weight per 217k rows.

    `E[floor(x + U)] = x` exactly for uniform U in [0,1), for any value
    distribution, so the dithered cell error is zero-mean and grows as
    sqrt(rows), not rows. An integer-valued `val * scale` is unchanged by
    any u < 1, which is what keeps `hist2_check`'s exactness argument
    intact. The +1 worst case per row is accounted for EXACTLY by
    `choose_scale`'s `row_count` allowance (its blanket headroom covers
    callers that state none). Called ONCE per (row, stat) at load time -- computing it
    at every add cost a measured quarter of the tree.

    WRITTEN AS `floor(x)` PLUS A FRACTION COMPARE, not `floor(x + u)`: the
    one-expression form ROUNDS in the addition before `floor` sees it, so an
    integral `x` with `u` within half a Float32 spacing of 1 crossed the
    integer boundary -- 14 cells of `check-hist2`'s exact fixture moved on
    the first run. `x - floor(x)` is exact in Float32 (the operands are
    within a factor of two), the compare never manufactures a crossing, and
    an integral `x` has fraction 0, which no `u < 1` can lift to 1."""
    # IDENTITY_PATHS ROW 10: a DENORMAL `val` diverges here between
    # Metal's FTZ hardware (operand flushed, `scaled` = 0) and a
    # denormal-honoring backend (`scaled` can be a real fraction --
    # `choose_scale` can legally return 2^100-class scales when the plane
    # magnitude is tiny -- and the dither compare then increments `q` on
    # one vendor and not the other). Operand and result flushed through
    # `ftz`, aligning every backend to Metal's measured model; comptime
    # no-ops under FAST, where Apple's hardware did both flushes already.
    var scaled = ftz(ftz(val) * fixed_scale)
    var base = floor(scaled)
    var q = Int32(base)
    if ftz(scaled - base) + u >= Float32(1.0):
        q += Int32(1)
    return q
