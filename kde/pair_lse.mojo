"""C52 common host/device pair-combine profile, with fixed reference leaves.

NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
Adjacent pairs merge left then right; an odd tail carries without arithmetic.
The left maximum wins ties (including signed zero). Empty pairs are (-inf,0).
The 64-level binary carry stack covers an Int-sized row count without allocating
scratch proportional to the reference set. Geometry never chooses the tree.
"""
from std.memory import bitcast
from checks.numerics import ftz, identical_exp, identical_log, identical_mul, identical_mul_add
from experiments.classical_identical_ideas.stats_controls import C52_ROWS

comptime P = MutPointer[Float32, MutAnyOrigin]

@always_inline
def pair_join(am: Float32, av: Float32, bm: Float32, bv: Float32) -> Tuple[Float32, Float32]:
    var ni = bitcast[DType.float32](UInt32(0xFF800000))
    if am == ni:
        return (bm, bv)
    if bm == ni:
        return (am, av)
    var m = bm if bm > am else am
    var a = ftz(identical_mul(av, ftz(identical_exp(ftz(am - m)))))
    var b = ftz(identical_mul(bv, ftz(identical_exp(ftz(bm - m)))))
    return (m, ftz(a + b))

def pair_finish(m: Float32, s: Float32) -> Tuple[Float32, Float32]:
    var ni = bitcast[DType.float32](UInt32(0xFF800000))
    return (m, m if m == ni else ftz(identical_log(s) + m))

def pair_parts(pm: P, ps: P, base: Int, count: Int) -> Tuple[Float32, Float32]:
    """Consume private per-query chunk scratch in place; no later reader."""
    var width = count
    while width > 1:
        for j in range(width // 2):
            var p = pair_join(pm[base + 2*j], ps[base + 2*j], pm[base + 2*j+1], ps[base + 2*j+1])
            pm[base+j] = p[0]
            ps[base+j] = p[1]
        if width % 2:
            pm[base+width//2] = pm[base+width-1]
            ps[base+width//2] = ps[base+width-1]
        width = (width+1)//2
    if count == 0:
        var ni = bitcast[DType.float32](UInt32(0xFF800000))
        return (ni, ni)
    return pair_finish(pm[base], ps[base])

def pair_row(row: P, n: Int, leaf_rows: Int = C52_ROWS) -> Tuple[Float32, Float32]:
    """Same logical tree as pair_parts, streaming leaves with bounded storage."""
    var ni = bitcast[DType.float32](UInt32(0xFF800000))
    var ms = InlineArray[Float32, 64](fill=ni)
    var ss = InlineArray[Float32, 64](fill=Float32(0))
    var occupied = UInt64(0)
    for lo in range(0, n, leaf_rows):
        var m = ni
        var s = Float32(0)
        for j in range(lo, min(n, lo+leaf_rows)):
            var v = ftz(row[j])
            if v > m:
                s = Float32(1) if m == ni else ftz(identical_mul_add(s, ftz(identical_exp(ftz(m-v))), Float32(1)))
                m = v
            elif v != ni:
                s = ftz(s + ftz(identical_exp(ftz(v-m))))
        var level = 0
        while (occupied & (UInt64(1) << level)) != 0:
            var p = pair_join(ms[level], ss[level], m, s)
            m = p[0]
            s = p[1]
            occupied &= ~(UInt64(1) << level)
            level += 1
        ms[level] = m
        ss[level] = s
        occupied |= UInt64(1) << level
    # Ascending levels join the shorter right tail first, then its left sibling.
    var m = ni
    var s = Float32(0)
    for level in range(64):
        if (occupied & (UInt64(1) << level)) != 0:
            var p = pair_join(ms[level], ss[level], m, s)
            m = p[0]
            s = p[1]
    return pair_finish(m, s)
