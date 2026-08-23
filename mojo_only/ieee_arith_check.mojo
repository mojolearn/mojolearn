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
    var ftz_model_hits = 0
    var ftz_model_misses = 0
    # See THE SEPARATING SUBSET below: the patterns on which a fused and an
    # unfused `a*b+c` actually differ, and the only ones the contraction
    # verdict may be read off.
    var separating = 0
    var sep_fused = 0
    var sep_unfused = 0
    var sep_neither = 0

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

        # THE FTZ MODEL: flush denormal operands, compute the correctly
        # rounded op, flush a denormal result. If this reproduces the
        # device bit-for-bit on every divergence, then a source-level
        # ftz() IS Metal's arithmetic, measured -- which is what lets
        # IDENTICAL mode impose the same flush on non-FTZ backends.
        var ftz_div = _ftz(
            Float32(Float64(_ftz(av)) / Float64(_ftz(bv)))
        )
        var ftz_sqrt = _ftz(Float32(sqrt(Float64(abs(_ftz(av))))))
        if q_div.unsafe_load(i) != _bits(ref_div):
            wrong_div += 1
            if q_div.unsafe_load(i) == _bits(ftz_div):
                ftz_model_hits += 1
            else:
                ftz_model_misses += 1
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
            if q_sqrt.unsafe_load(i) == _bits(ftz_sqrt):
                ftz_model_hits += 1
            else:
                ftz_model_misses += 1
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
        # THE SEPARATING SUBSET, added 2026-08-23, and it changes this
        # check's verdict rather than decorating it. The two references
        # AGREE on the overwhelming majority of hashed patterns -- random
        # exponents put `a * b` and `c` so far apart that one rounding and
        # two round to the same bits -- and every such pattern was being
        # counted as evidence of UNFUSED by the `if got == ub` arm that
        # runs first. A backend that contracts EVERYTHING would therefore
        # still be reported "UNFUSED" as long as the ties dominated, which
        # is exactly what happened: the recorded 'fused 0 / unfused
        # 1,046,394' was 1,046,394 ties and no measurement at all.
        # Only patterns where the references DIFFER carry information.
        if fb != ub:
            separating += 1
            if got_fma == fb:
                sep_fused += 1
            elif got_fma == ub:
                sep_unfused += 1
            else:
                sep_neither += 1
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
        "  a*b+c SEPARATING patterns (the two references differ):",
        separating,
        " of which device fused", sep_fused,
        " unfused", sep_unfused,
        " neither", sep_neither,
    )
    # ---- ARM 6: THE PURPOSE-BUILT CONTRACTION PATTERNS -------------------
    # Arm 5's operands are hashed bit patterns, and the count above says
    # what that costs: ZERO of 2^20 of them separate a fused `a*b+c` from an
    # unfused one, because random exponents put the product and the addend
    # so far apart that both spellings round the same way. A verdict read
    # off those patterns is a verdict about nothing, and the one this file
    # used to print ("UNFUSED on this backend") propagated into
    # IDENTITY_PATHS row 9 and from there into the claim that IDENTICAL's
    # `fma` differs from FAST on Apple.
    #
    # These patterns separate BY CONSTRUCTION. Take `a` and `b` with
    # half-width mantissas so their exact product needs more bits than
    # float32 keeps, and set `c = -fl(a*b)`, the ROUNDED product negated.
    # Then:
    #
    #     unfused:  fl(a*b) + (-fl(a*b))  ==  +0.0, exactly, always
    #     fused:    a*b - fl(a*b)         ==  the rounding error, nonzero
    #
    # so one bit of the answer IS the contraction. Nothing about the
    # magnitudes is delicate and no tie can hide it.
    var sepN = 4096
    var sa = ctx.enqueue_create_buffer[DType.float32](sepN)
    var sb = ctx.enqueue_create_buffer[DType.float32](sepN)
    var sc = ctx.enqueue_create_buffer[DType.float32](sepN)
    var sbp = ctx.enqueue_create_buffer[DType.float32](sepN)
    var s_div = ctx.enqueue_create_buffer[DType.float32](sepN)
    var s_sqrt = ctx.enqueue_create_buffer[DType.float32](sepN)
    var s_l2 = ctx.enqueue_create_buffer[DType.float32](sepN)
    var s_cos = ctx.enqueue_create_buffer[DType.float32](sepN)
    var s_out = ctx.enqueue_create_buffer[DType.float32](sepN)
    var hsa = ctx.enqueue_create_host_buffer[DType.float32](sepN)
    var hsb = ctx.enqueue_create_host_buffer[DType.float32](sepN)
    var hsc = ctx.enqueue_create_host_buffer[DType.float32](sepN)
    var hsbp = ctx.enqueue_create_host_buffer[DType.float32](sepN)
    var hs_out = ctx.enqueue_create_host_buffer[DType.float32](sepN)
    ctx.synchronize()

    for i in range(sepN):
        var h1 = _splitmix(UInt64(i) + UInt64(0xC0FFEE))
        var h2 = _splitmix(UInt64(i) + UInt64(0xBEEF01))
        var av2 = Float32(1.0) + Float32(Int(h1 & UInt64(0xFFF))) / Float32(
            4096.0
        )
        var bv2 = Float32(1.0) + Float32(Int(h2 & UInt64(0xFFF))) / Float32(
            4096.0
        )
        # ONE multiply, standing alone: nothing for a compiler to contract
        # it into, on any host.
        var prod = av2 * bv2
        hsa.unsafe_ptr().unsafe_store(i, av2)
        hsb.unsafe_ptr().unsafe_store(i, bv2)
        hsc.unsafe_ptr().unsafe_store(i, -prod)
        hsbp.unsafe_ptr().unsafe_store(i, Float32(1.0))
    ctx.enqueue_copy(dst_buf=sa, src_ptr=hsa.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=sb, src_ptr=hsb.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=sc, src_ptr=hsc.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=sbp, src_ptr=hsbp.unsafe_ptr())
    ctx.synchronize()
    ctx.enqueue_function[arith_kernel](
        sa.unsafe_ptr(), sb.unsafe_ptr(), sc.unsafe_ptr(), sbp.unsafe_ptr(),
        s_div.unsafe_ptr(), s_sqrt.unsafe_ptr(), s_l2.unsafe_ptr(),
        s_cos.unsafe_ptr(), s_out.unsafe_ptr(), Int32(sepN),
        grid_dim=((sepN + 255) // 256, 1, 1), block_dim=(256, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hs_out.unsafe_ptr(), src_buf=s_out)
    ctx.synchronize()

    var built_sep = 0
    var built_fused = 0
    var built_unfused = 0
    var built_neither = 0
    for i in range(sepN):
        var av2 = hsa.unsafe_ptr().unsafe_load(i)
        var bv2 = hsb.unsafe_ptr().unsafe_load(i)
        var cv2 = hsc.unsafe_ptr().unsafe_load(i)
        var ref_f = fma(av2, bv2, cv2)
        if _bits(ref_f) == _bits(Float32(0.0)):
            # the product was exact; this pattern carries no information
            continue
        built_sep += 1
        var got = _bits(hs_out.unsafe_ptr().unsafe_load(i))
        if got == _bits(ref_f):
            built_fused += 1
        elif got == _bits(Float32(0.0)):
            built_unfused += 1
        else:
            built_neither += 1
    print(
        "  a*b+c BUILT-TO-SEPARATE patterns:", built_sep,
        " device fused", built_fused,
        " unfused", built_unfused,
        " neither", built_neither,
    )
    print(
        "  ftz model: reproduces", ftz_model_hits, "of",
        ftz_model_hits + ftz_model_misses,
        "div/sqrt divergences bit-for-bit",
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
        (
            "UNMEASURED (nothing separated the two spellings)"
            if (separating + built_sep) == 0
            else "FUSED" if (sep_unfused + built_unfused) == 0
            else "UNFUSED" if (sep_fused + built_fused) == 0
            else "MIXED"
        ),
        "-- read off the",
        separating + built_sep,
        "patterns that SEPARATE the two spellings, never the",
        "tie-dominated totals above",
        "on this backend",
    )


def _ftz(x: Float32) -> Float32:
    # the flush Metal's hardware applies: magnitude below FLT_MIN becomes
    # a SIGNED zero
    var bits = _bits(x)
    if (bits >> 23) & UInt32(0xFF) == UInt32(0):
        if bits & UInt32(0x7FFFFF) != UInt32(0):
            var z = List[UInt32]()
            z.append(bits & UInt32(0x80000000))
            return z.unsafe_ptr().unsafe_bitcast[Float32]().unsafe_load(0)
    return x


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
