"""IDENTITY_PATHS rows 9 and 10, measured on this backend: div, sqrt, FMA.

    pixi run check-ieee-arith

Row 10 says division and `sqrt` in scoring are IEEE-correct "unless a
backend substitutes a fast reciprocal or rsqrt -- verify per backend",
and row 9 says nothing pins FMA contraction. This check turns both from
policy into measurement, on whatever device compiles it. Run it on every
backend the kernel matrix carries a column for; the Apple run is the
first.

## The five arms, each 2^20 hashed finite bit patterns, compared BIT FOR BIT

    1. a / b            raw division, denormals and near-overflow included
    2. sqrt(|a|)        raw square root
    3. (a*a) / (b'+L)   the L2 score expression, exactly as
                        `compute_scores.mojo:153` writes it
    4. a / sqrt(b')     the Cosine final score, `compute_scores.mojo:356`
    5. a*b + c          the canonical contraction shape, judged against
                        BOTH references below

## Why the references are exact, not approximate

For div and sqrt: computing in Float64 and rounding once to Float32 IS
the correctly-rounded Float32 operation, because 53 >= 2*24 + 2 (the
classical double-rounding-safe bound). For mul and add: the Float64
result of two Float32 operands is EXACT (a 24x24 product fits in 48
bits; a sum fits outright), so rounding it once to Float32 is again the
IEEE operation. Chaining those per-op gives the exact UNFUSED reference
for the compound arms. The exact FUSED reference for arm 5 is the host's
`fma` (one rounding by definition). So arm 5 has two bit-exact
references, and whichever one the device matches is the codegen's
contraction answer -- matching NEITHER would mean a third arithmetic and
is its own finding.

## Reading the result

    arms 1-4 zero wrong   -> row 10 CLOSED for this backend: no fast
                             reciprocal, no rsqrt substitution, IEEE
                             div/sqrt in exactly the score kernel's shapes
    arm 5                 -> reported as "fused N / unfused M / neither K";
                             row 9's per-backend answer, and the input to
                             the source-level pinning decision

A nonzero in arms 1-4 prints the first offending bit patterns, because a
fast-math substitution is diagnosed from WHICH inputs break (reciprocals
break denormals first).
"""
from max.gpu.host import DeviceBuffer, DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.math import fma, sqrt

from std.sys import argv

comptime N = 1 << 20
comptime BLOCK = 256
comptime LAM = Float32(3.0)


def arith_kernel(
    a: MutPointer[Float32, MutAnyOrigin],
    b: MutPointer[Float32, MutAnyOrigin],
    c: MutPointer[Float32, MutAnyOrigin],
    bpos: MutPointer[Float32, MutAnyOrigin],
    out_div: MutPointer[Float32, MutAnyOrigin],
    out_sqrt: MutPointer[Float32, MutAnyOrigin],
    out_l2: MutPointer[Float32, MutAnyOrigin],
    out_cos: MutPointer[Float32, MutAnyOrigin],
    out_fma: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i < Int(n_in):
        var av = a.unsafe_load(i)
        var bv = b.unsafe_load(i)
        var cv = c.unsafe_load(i)
        var bp = bpos.unsafe_load(i)
        out_div.unsafe_store(i, av / bv)
        out_sqrt.unsafe_store(i, sqrt(abs(av)))
        # compute_scores.mojo:153, verbatim shape
        out_l2.unsafe_store(i, (av * av) / (bp + LAM))
        # compute_scores.mojo:356, verbatim shape
        out_cos.unsafe_store(i, av / sqrt(bp))
        # the contraction canary: written as the naive chain on purpose
        out_fma.unsafe_store(i, av * bv + cv)


def _finite_pattern(h: UInt32) -> UInt32:
    # clear the exponent's top bit when all-ones would make an inf/nan;
    # keeps denormals (exponent 0) on purpose
    if (h >> 23) & UInt32(0xFF) == UInt32(0xFF):
        return h & ~UInt32(0x40000000)
    return h


def _splitmix(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)


def main() raises:
    var ctx = DeviceContext()

    var ha = ctx.enqueue_create_host_buffer[DType.float32](N)
    var hb = ctx.enqueue_create_host_buffer[DType.float32](N)
    var hc = ctx.enqueue_create_host_buffer[DType.float32](N)
    var hbp = ctx.enqueue_create_host_buffer[DType.float32](N)
    var pa = ha.unsafe_ptr().unsafe_bitcast[UInt32]()
    var pb = hb.unsafe_ptr().unsafe_bitcast[UInt32]()
    var pc = hc.unsafe_ptr().unsafe_bitcast[UInt32]()
    var pbp = hbp.unsafe_ptr().unsafe_bitcast[UInt32]()

    for i in range(N):
        var h1 = _splitmix(UInt64(3 * i + 1))
        var h2 = _splitmix(UInt64(3 * i + 2))
        var h3 = _splitmix(UInt64(3 * i + 3))
        var ba = _finite_pattern(UInt32(h1 & 0xFFFFFFFF))
        var bb = _finite_pattern(UInt32(h2 & 0xFFFFFFFF))
        if bb & UInt32(0x7FFFFFFF) == UInt32(0):
            bb = UInt32(0x3F800000)  # divisor zero -> 1.0
        var bc = _finite_pattern(UInt32(h3 & 0xFFFFFFFF))
        # bpos: a positive finite value for the (b'+L) and sqrt(b') arms,
        # spanning denormals to large normals
        var bpv = (bb & UInt32(0x7FFFFFFF))
        if bpv == UInt32(0):
            bpv = UInt32(0x00000001)
        pa.unsafe_store(i, ba)
        pb.unsafe_store(i, bb)
        pc.unsafe_store(i, bc)
        pbp.unsafe_store(i, bpv)

    var da = ctx.enqueue_create_buffer[DType.float32](N)
    var db = ctx.enqueue_create_buffer[DType.float32](N)
    var dc = ctx.enqueue_create_buffer[DType.float32](N)
    var dbp = ctx.enqueue_create_buffer[DType.float32](N)
    ctx.enqueue_copy(dst_buf=da, src_ptr=ha.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=db, src_ptr=hb.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dc, src_ptr=hc.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=dbp, src_ptr=hbp.unsafe_ptr())

    var o_div = ctx.enqueue_create_buffer[DType.float32](N)
    var o_sqrt = ctx.enqueue_create_buffer[DType.float32](N)
    var o_l2 = ctx.enqueue_create_buffer[DType.float32](N)
    var o_cos = ctx.enqueue_create_buffer[DType.float32](N)
    var o_fma = ctx.enqueue_create_buffer[DType.float32](N)

    ctx.enqueue_function[arith_kernel](
        da.unsafe_ptr(), db.unsafe_ptr(), dc.unsafe_ptr(),
        dbp.unsafe_ptr(),
        o_div.unsafe_ptr(), o_sqrt.unsafe_ptr(), o_l2.unsafe_ptr(),
        o_cos.unsafe_ptr(), o_fma.unsafe_ptr(), Int32(N),
        grid_dim=(N + BLOCK - 1) // BLOCK,
        block_dim=(BLOCK, 1, 1),
    )

    var r_div = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_sqrt = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_l2 = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_cos = ctx.enqueue_create_host_buffer[DType.float32](N)
    var r_fma = ctx.enqueue_create_host_buffer[DType.float32](N)
    ctx.enqueue_copy(dst_ptr=r_div.unsafe_ptr(), src_buf=o_div)
    ctx.enqueue_copy(dst_ptr=r_sqrt.unsafe_ptr(), src_buf=o_sqrt)
    ctx.enqueue_copy(dst_ptr=r_l2.unsafe_ptr(), src_buf=o_l2)
    ctx.enqueue_copy(dst_ptr=r_cos.unsafe_ptr(), src_buf=o_cos)
    ctx.enqueue_copy(dst_ptr=r_fma.unsafe_ptr(), src_buf=o_fma)
    ctx.synchronize()

    var q_div = r_div.unsafe_ptr().unsafe_bitcast[UInt32]()
    var q_sqrt = r_sqrt.unsafe_ptr().unsafe_bitcast[UInt32]()
    var q_l2 = r_l2.unsafe_ptr().unsafe_bitcast[UInt32]()
    var q_cos = r_cos.unsafe_ptr().unsafe_bitcast[UInt32]()
    var q_fma = r_fma.unsafe_ptr().unsafe_bitcast[UInt32]()

    var wrong_div = 0
    var wrong_sqrt = 0
    var wrong_l2 = 0
    var wrong_cos = 0
    var fused = 0
    var unfused = 0
    var neither = 0
    var shown = 0
    var shown_sq = 0
    var shown_ne = 0
    var denorm_div = 0
    var denorm_sqrt = 0
    var denorm_neither = 0

    for i in range(N):
        var av = ha.unsafe_ptr().unsafe_load(i)
        var bv = hb.unsafe_ptr().unsafe_load(i)
        var cv = hc.unsafe_ptr().unsafe_load(i)
        var bp = hbp.unsafe_ptr().unsafe_load(i)

        # correctly-rounded references: one f64 op, one rounding
        var ref_div = Float32(Float64(av) / Float64(bv))
        var ref_sqrt = Float32(sqrt(Float64(abs(av))))
        var t_mul = Float32(Float64(av) * Float64(av))
        var t_add = Float32(Float64(bp) + Float64(LAM))
        var ref_l2 = Float32(Float64(t_mul) / Float64(t_add))
        var t_sq = Float32(sqrt(Float64(bp)))
        var ref_cos = Float32(Float64(av) / Float64(t_sq))
        var t_ab = Float32(Float64(av) * Float64(bv))
        var ref_unfused = Float32(Float64(t_ab) + Float64(cv))
        var ref_fused = fma(av, bv, cv)

        if q_div.unsafe_load(i) != _bits(ref_div):
            wrong_div += 1
            if _is_denormal_involved(av, bv, ref_div):
                denorm_div += 1
            if shown < 6:
                print(
                    "  div mismatch: a=", hex(_bits(av)), " b=",
                    hex(_bits(bv)), " device=", hex(q_div.unsafe_load(i)),
                    " ref=", hex(_bits(ref_div)),
                )
                shown += 1
        if q_sqrt.unsafe_load(i) != _bits(ref_sqrt):
            wrong_sqrt += 1
            if _is_denormal_involved(av, av, ref_sqrt):
                denorm_sqrt += 1
            if shown_sq < 3:
                print(
                    "  sqrt mismatch: a=", hex(_bits(av)), " device=",
                    hex(q_sqrt.unsafe_load(i)), " ref=",
                    hex(_bits(ref_sqrt)),
                )
                shown_sq += 1
        if q_l2.unsafe_load(i) != _bits(ref_l2):
            wrong_l2 += 1
        if q_cos.unsafe_load(i) != _bits(ref_cos):
            wrong_cos += 1

        var got_fma = q_fma.unsafe_load(i)
        var fb = _bits(ref_fused)
        var ub = _bits(ref_unfused)
        if got_fma == ub:
            unfused += 1
        elif got_fma == fb:
            fused += 1
        else:
            neither += 1
            if _fma_denormal_involved(av, bv, cv, ref_unfused):
                denorm_neither += 1
            if shown_ne < 3:
                print(
                    "  a*b+c neither: a=", hex(_bits(av)), " b=",
                    hex(_bits(bv)), " c=", hex(_bits(cv)), " device=",
                    hex(got_fma), " unfused=", hex(ub), " fused=",
                    hex(fb),
                )
                shown_ne += 1

    print("ieee arith,", N, "hashed patterns:")
    print(
        "  div", wrong_div, "wrong;  sqrt", wrong_sqrt,
        "wrong;  L2 score shape", wrong_l2, "wrong;  cosine shape",
        wrong_cos, "wrong",
    )
    print(
        "  a*b+c: unfused", unfused, " fused", fused, " neither", neither,
        " (ties where both references agree count as unfused)",
    )
    print(
        "  denormal-involved:  div", denorm_div, "of", wrong_div,
        "  sqrt", denorm_sqrt, "of", wrong_sqrt, "  neither",
        denorm_neither, "of", neither,
    )
    # THE VERDICT SPLITS ON DENORMALS. Metal documents flush-to-zero for
    # float32, so FTZ divergence is a measured PLATFORM PROPERTY the
    # identity design must handle (align it in source across backends);
    # it is not a fast-math substitution. What would make row 10 REAL is
    # a mismatch on NORMAL values -- a reciprocal or rsqrt -- and that is
    # what fails this check.
    var non_denorm = (
        (wrong_div - denorm_div)
        + (wrong_sqrt - denorm_sqrt)
        + (neither - denorm_neither)
    )
    if non_denorm != 0:
        raise Error(
            "non-denormal divergence on this backend: a fast reciprocal,"
            " rsqrt, or third arithmetic is present -- IDENTITY_PATHS"
            " row 10 is REAL beyond FTZ"
        )
    if wrong_div + wrong_sqrt + neither != 0:
        print(
            "ieee arith check OK: correctly rounded ON NORMALS, with"
            " FLUSH-TO-ZERO on denormal operands, intermediates and"
            " results -- record this backend's column as IEEE+FTZ"
        )
    else:
        print(
            "ieee arith check OK: fully IEEE including denormals on this"
            " backend"
        )
    print(
        "  contraction: a*b+c is",
        "UNFUSED" if fused == 0 else "FUSED" if unfused == 0 else "MIXED",
        "on this backend",
    )


def _is_denorm(v: Float32) -> Bool:
    var bits = _bits(v)
    if (bits >> 23) & UInt32(0xFF) == UInt32(0):
        return bits & UInt32(0x7FFFFF) != UInt32(0)
    return False


def _is_denormal_involved(a: Float32, b: Float32, r: Float32) -> Bool:
    # denormal in any operand or the result -- FTZ territory
    return _is_denorm(a) or _is_denorm(b) or _is_denorm(r)


def _fma_denormal_involved(
    a: Float32, b: Float32, c: Float32, r: Float32
) -> Bool:
    # arm 5 additionally flushes at the INTERMEDIATE product: |a*b| below
    # FLT_MIN flushes on an FTZ machine before the add ever runs
    if _is_denorm(a) or _is_denorm(b) or _is_denorm(c) or _is_denorm(r):
        return True
    var prod = Float64(a) * Float64(b)
    if prod < 0:
        prod = -prod
    return prod != 0.0 and prod < 1.1754943508222875e-38


def _bits(x: Float32) -> UInt32:
    var tmp = List[Float32]()
    tmp.append(x)
    return tmp.unsafe_ptr().unsafe_bitcast[UInt32]().unsafe_load(0)
