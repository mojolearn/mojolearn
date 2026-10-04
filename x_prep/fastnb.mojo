# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CategoricalNB's count table in ONE launch over (row, feature) (lane
apple-fast-nb, 2026-10-02). FAST + Apple ONLY, behind -D MOJOLEARN_NB_CAT_ATOMIC:
the IDENTICAL binding, the other vendors and a FAST build without the define
compile x_prep/device.mojo's unit path unchanged (`cat_hpart_unit` and
`cat_hfold_unit`, x_prep/blocked.mojo).

`cat_hpart_unit` is one thread per (2048-row block, feature) folding its
block's rows serially into a private K*CMAX histogram in device memory
(read-modify-write per row), then `cat_hfold_unit` sums the blocks: a few
thousand threads at the board's 1M rows, most of the GPU idle. Here every
(row, feature) cell is one thread: an integer atomic add of 1 into the
(feature, class, category) slot of the FIRST block of the same histogram
scratch (zeroed by the dispatcher first), and the fold stage becomes the
copy of that block's integer words as float32 into cat_hfold's output.
Integer counts are order-free, so the table is exact and the same on every
run. The weighted form (sample_weight, a float fold) keeps the unit path:
the dispatcher only intercepts W < 0.
"""
from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined
from std.sys.info import has_apple_gpu_accelerator
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST, NUMERIC_IDENTICAL
from x_prep.common import FP, IP, p, ld, st

#: lane idn-int-prep (2026-10-04): IDENTICAL on every vendor, ON by default.
#: The table is an integer count: the atomic adds give the units' words at any
#: row count (`cat_hfold_unit` also sums integers and converts once), and the
#: host column keeps the units. -D MOJOLEARN_IDN_NB_CAT_ATOMIC_OFF restores
#: the unit path on the device.
comptime IDN_NB_CAT_ATOMIC = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and not is_defined["MOJOLEARN_IDN_NB_CAT_ATOMIC_OFF"]()
)
#: The switch: FAST, Apple, and the define (default OFF); or IDENTICAL (above).
comptime NB_CAT_ATOMIC = (
    GLOBAL_NUMERIC_MODE == NUMERIC_FAST and has_apple_gpu_accelerator() and is_defined["MOJOLEARN_NB_CAT_ATOMIC"]()
) or IDN_NB_CAT_ATOMIC


def cat_hist_atomic_kernel(f: FP, q: IP, total: Int32):
    """cat_hpart's q = [X, n, d, Y, K, NCAT, CMAX, W, H, R]; thread t = i*d + j
    (one per (row, feature) cell, total = n*d): row i's feature j holds v;
    when 0 <= v < NCAT[j], H's block-0 word (j*K + Y[i])*CMAX + v gains one
    (an Int32 atomic on the float arena's words). Rows outside the range add
    nothing, as the unit."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var d = p(q, 2)
    var i = t // d
    var j = t - i * d
    var v = Int(ld(f, p(q, 0) + t))
    if v < 0 or v >= Int(ld(f, p(q, 5) + j)):
        return
    var K = p(q, 4)
    var cmax = p(q, 6)
    var k = Int(ld(f, p(q, 3) + i))
    var fi = f.bitcast[Int32]()
    _ = Atomic.fetch_add(fi.unsafe_offset(p(q, 8) + (j * K + k) * cmax + v), Int32(1))


def cat_hist_convert_kernel(f: FP, q: IP, total: Int32):
    """cat_hfold's q = [H, nb, d, K, NCAT, CMAX, W, OUT]; thread t =
    (j*K + k)*CMAX + v: OUT[t] = the integer count in H's block-0 word t as
    float32 (exact below 2^24 rows). Slots v >= NCAT[j] are left as they are,
    as `cat_hfold_unit` leaves them."""
    var t = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if t >= Int(total):
        return
    var K = p(q, 3)
    var cmax = p(q, 5)
    var v = t % cmax
    var j = (t // cmax) // K
    if v >= Int(ld(f, p(q, 4) + j)):
        return
    var fi = f.bitcast[Int32]()
    st(f, p(q, 7) + t, Float32(Int(fi.unsafe_load(p(q, 0) + t))))
