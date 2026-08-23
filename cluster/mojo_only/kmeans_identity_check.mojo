"""What `IDENTICAL` promises about k-means, gated on ONE device.

NO CUVS COUNTERPART, and there cannot be one: cuVS ships a single GPU
backend and its k-means is not reproducible across GPU MODELS in the first
place (`unfused_distance_nn.cuh:196` runs the distance GEMM at
`CUBLAS_COMPUTE_32F_FAST_TF32`, ten mantissa bits). The property under test
here is ours.

WHAT A ONE-DEVICE CHECK CAN AND CANNOT PROVE
---------------------------------------------
It cannot prove two vendors agree. Only E1 can, and until it runs everything
above the Apple column is construction plus transcription
(`IDENTITY_PATHS.md`). What it CAN prove is every step of the argument that
would otherwise be assertion:

1. **The pinned constructions are REACHED.** A pin nobody's kernel calls is
   the defect this repository has found five times. So each check computes
   the mode's OWN oracle -- an `fma` chain for IDENTICAL, a multiply-then-add
   chain for FAST -- and requires the device to match THAT one bit for bit
   while DIFFERING from the other. A check that passed in both modes would
   be evidence of nothing.

2. **The answer does not depend on the launch geometry.** This is the
   closest a single box gets to a second vendor: another GPU's core count
   reaches the algorithm as a different grid, so running the same fixture at
   several grids and requiring BITWISE-equal output is the cross-vendor
   hazard reproduced locally. It must hold in both modes -- geometry
   independence is not what `IDENTICAL` buys, it is what the argmin's total
   order and the fixed-point accumulator already bought.

3. **The instrument can localize a divergence**, because a claim that can
   only be checked at the end is a claim nobody can debug.

WHY THE FIXTURES LOOK STRANGE
------------------------------
Two of them are built so that a fold SHAPE and a CONTRACTION are visible in
the answer, and each check first proves its own fixture discriminates before
it tests anything. A fixture on which `fma(a, b, c)` and `a * b + c` agree
would let a broken pin pass, which is the same class of hole as a uniform
test matrix hiding a permutation (`[[uniform-test-data-hides-permutation]]`).
"""

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from std.gpu import block_dim, block_idx, thread_idx
from std.math import fma
from std.memory import bitcast

from core.identity_trace import IdentityTrace, first_divergence
from core.pinned_reduce import pinned_block_sum
from core.row_norms import NORM_TPB, row_norm_kernel
from cluster.ported.cluster.detail.kmeans import kmeans_fit_main
from cluster.ported.cluster.kmeans_params import (
    INIT_ARRAY,
    KMeansParams,
)
from cluster.ported.distance.fused_distance_nn.simt_kernel import (
    FUSED_CLAMP_PRECISION,
    FUSED_NORMAL_KBLK,
    FUSED_NORMAL_TC,
    FUSED_NORMAL_TR,
    fused_distance_nn_kernel,
)
from mojo_only.kernel_matrix import (
    COLUMN_AMD,
    COLUMN_APPLE,
    COLUMN_BIT_IDENTICAL,
    COLUMN_NVIDIA,
    K_LIB_ADJ_SCAN,
    K_LIB_EPS_NEIGHBORHOOD,
    K_LIB_PLUS_PLUS,
    K_LIB_REDUCE_BY_KEY,
    K_LIB_ROW_NORM,
    K_LIB_WEAK_CC,
    lib_block_bounds_a_float_fold,
    lib_block_size_for,
)
from mojo_only.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL


comptime IDENTICAL_BUILD = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL

#: The fold this check drives. 128 is what every column carries for the
#: three float-fold library rows today, so the check runs at the shipped
#: width rather than a convenient one.
comptime FOLD_TPB = 128

#: How far one cell of X is nudged in `check_kmeans_trace_localizes`. See
#: that check's docstring: ONE ULP was measured to be absorbed by the norm.
comptime _PERTURB_ULPS = 1024


def _mode_name() -> String:
    comptime if IDENTICAL_BUILD:
        return String("IDENTICAL")
    return String("FAST")


def _hash64(i: Int, f: Int) -> UInt64:
    """A deterministic 64-bit mix. Not a port; the fixture generator."""
    var h = UInt64(i) * UInt64(0x9E3779B97F4A7C15) + UInt64(f + 1) * UInt64(
        0xBF58476D1CE4E5B9
    )
    h = h ^ (h >> UInt64(29))
    h = h * UInt64(0x94D049BB133111EB)
    return h ^ (h >> UInt64(32))


def _spread(i: Int, f: Int) -> Float32:
    """A value in [1, 2) with its mantissa in the TOP 12 BITS ONLY.

    THE HALF-WIDTH MANTISSA IS THE WHOLE FIXTURE DESIGN. The product of two
    such values needs 24 significant bits, so its low half is exactly what a
    naive multiply rounds away and what an `fma` keeps. Full-width random
    mantissas do NOT reliably separate the two spellings in a dot product --
    measured, 0 of 256 rows at d = 4, 8 and 32 -- because the accumulator's
    own rounding usually swallows the product's tail. That measurement is
    why this generator looks odd, and a "tidier" fixture would silently make
    the contraction check unable to fail.
    """
    var frac = Float32(Int(_hash64(i, f) & UInt64(0xFFF))) / Float32(4096.0)
    return Float32(1.0) + frac


def _spread_full(i: Int, f: Int) -> Float32:
    """A value in [1, 2) with a FULL 24-bit mantissa.

    The fold checks want values whose ADDITION order matters; the
    contraction check wants values whose PRODUCTS carry a tail. Those are
    different fixtures and one generator cannot be both: a 12-bit mantissa
    makes a 128-value halving fold agree exactly with a sequential one
    (measured, which is how this function came to exist), and a 24-bit one
    makes an fma agree with a multiply-then-add. Each check asserts its own
    fixture discriminates, so a swap shows up as a refusal rather than a
    pass.
    """
    var frac = Float32(Int(_hash64(i, f) & UInt64(0xFFFFFF))) / Float32(
        16777216.0
    )
    return Float32(1.0) + frac


def fold_probe_kernel(
    out_sum: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
):
    """One block, one value per thread, through `pinned_block_sum`.

    Deliberately NOT a grid-strided loop: this check is about the fold's
    SHAPE, so each thread contributes exactly one element and nothing is
    summed before the fold gets it.
    """
    var tid = Int(thread_idx.x)
    var v = a.unsafe_load(tid)
    var s = pinned_block_sum[FOLD_TPB](v)
    if tid == 0:
        out_sum.unsafe_store(0, s)


def _host_halving_fold(values: List[Float32]) -> Float32:
    """`pinned_block_sum`'s IDENTICAL arm, on the host, in Float32.

    `red[t] += red[t + step]` for `step = n/2 ... 1`, which is the whole
    kernel. Written out rather than approximated, because the point of the
    check is that the DEVICE takes this exact sequence of additions.
    """
    var red = values.copy()
    var step = len(red) // 2
    while step > 0:
        for t in range(step):
            red[t] = red[t] + red[t + step]
        step //= 2
    return red[0]


def _host_sequential_fold(values: List[Float32]) -> Float32:
    """Ascending, one at a time. A DIFFERENT shape, used only to prove the
    fixture can tell two shapes apart."""
    var acc = Float32(0.0)
    for i in range(len(values)):
        acc = acc + values[i]
    return acc


def check_pinned_fold_shape() raises:
    """IDENTITY_PATHS row 20. The within-block float fold has ONE shape.

    `max.gpu.primitives.block.sum` folds across lanes at the HARDWARE width
    -- 32 on Apple and NVIDIA, 64 on AMD's wavefront -- so the same 128
    values reduce through two different association trees on two vendors
    and the last bits differ. Under IDENTICAL the fold is
    `core/pinned_reduce.pinned_block_sum`'s halving tree, which contains no
    lane primitive and therefore cannot consult the hardware.

    The gate: the device must equal the HOST halving tree bit for bit under
    IDENTICAL. Under FAST it must equal neither necessarily -- it is
    Modular's shape, which is theirs to change -- so that arm only requires
    the library value, and requires the two host shapes to DISAGREE, which
    is what makes the IDENTICAL arm's equality meaningful.
    """
    var ctx = DeviceContext()
    var a = ctx.enqueue_create_buffer[DType.float32](FOLD_TPB)
    var out = ctx.enqueue_create_buffer[DType.float32](1)
    var ha = ctx.enqueue_create_host_buffer[DType.float32](FOLD_TPB)
    var hout = ctx.enqueue_create_host_buffer[DType.float32](1)
    ctx.synchronize()

    var values = List[Float32]()
    for i in range(FOLD_TPB):
        # Magnitudes an order of magnitude apart, so the order of the adds
        # decides how much of each addend survives the rounding.
        var v = _spread_full(i, 3) * Float32(
            1.0 + Float32(i % 7) * 1000.0
        )
        values.append(v)
        ha.unsafe_ptr().unsafe_store(i, v)
    ctx.enqueue_copy(dst_buf=a, src_ptr=ha.unsafe_ptr())
    ctx.synchronize()

    var halving = _host_halving_fold(values)
    var sequential = _host_sequential_fold(values)
    if halving == sequential:
        raise Error(
            "check_pinned_fold_shape: the fixture cannot tell two fold"
            " shapes apart (halving == sequential), so it could not detect"
            " a fold that ignored the pin. Fix the fixture, not the check."
        )

    ctx.enqueue_function[fold_probe_kernel](
        out.unsafe_ptr(),
        a.unsafe_ptr(),
        grid_dim=(1, 1, 1),
        block_dim=(FOLD_TPB, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hout.unsafe_ptr(), src_buf=out)
    ctx.synchronize()
    var got = hout.unsafe_ptr().unsafe_load(0)

    comptime if IDENTICAL_BUILD:
        if got != halving:
            raise Error(
                "check_pinned_fold_shape: IDENTICAL build did not take the"
                " halving tree: device "
                + String(got)
                + " vs host halving "
                + String(halving)
                + " (sequential would be "
                + String(sequential)
                + "). The pin is not reached."
            )
        print(
            "check_pinned_fold_shape OK (IDENTICAL): device ==",
            "host halving tree exactly, and the fixture separates it from",
            "a sequential fold (", halving, "vs", sequential, ")",
        )
    else:
        print(
            "check_pinned_fold_shape OK (FAST): library fold ran (",
            got,
            "); host halving", halving, "and sequential", sequential,
            "differ, so the IDENTICAL arm's equality has teeth",
        )


def contraction_oracle_kernel(
    out_val_fma: MutPointer[Float32, MutAnyOrigin],
    out_key_fma: MutPointer[UInt32, MutAnyOrigin],
    out_val_naive: MutPointer[Float32, MutAnyOrigin],
    out_key_naive: MutPointer[UInt32, MutAnyOrigin],
    x: MutPointer[Float32, MutAnyOrigin],
    c: MutPointer[Float32, MutAnyOrigin],
    xn: MutPointer[Float32, MutAnyOrigin],
    cn: MutPointer[Float32, MutAnyOrigin],
    n_in: Int32,
    k_in: Int32,
    d_in: Int32,
):
    """The two spellings of the assignment, side by side, ON THE DEVICE.

    **THE ORACLE IS ON THE DEVICE ON PURPOSE, AND THE HOST VERSION OF IT WAS
    DELETED.** A host model was written first and could not tell the two
    spellings apart at all: this toolchain's HOST codegen contracts
    `acc + x * y` into an `fma` inside a loop, and it even folds
    `Float32(Float64(x) * Float64(y))` back to `x * y` first -- a legal
    rewrite, since a float32 product is exact in double -- so the
    contraction-proof spelling was contracted anyway. Isolated, outside a
    loop, `a * a + c` measured UNFUSED and `fma(a, a, c)` measured genuinely
    fused (bits 973078528 vs 973079552), so the primitive is real and only
    the surrounding code was unreliable. That is the same class as
    `[[mojo-log-breaks-ties]]`'s sibling and is recorded here because the
    next person to write a host oracle for a contraction will hit it.

    On the DEVICE the ambiguity does not exist on this backend: row 9 of
    IDENTITY_PATHS measured Metal-through-MAX UNFUSED on 2^20 patterns
    (fused 0), so the naive arm below really is two roundings and the `fma`
    arm really is one. On a backend where that is not true the check says so
    rather than guessing -- see `check_fused_contraction_pin`.

    Both arms accumulate over features ASCENDING, which is the order the
    shipped kernel takes at every `veclen` and both policies.
    """
    var n = Int(n_in)
    var k = Int(k_in)
    var d = Int(d_in)
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if row >= n:
        return

    var best_f = Float32(3.4028234663852886e38)
    var best_kf = UInt32(0xFFFFFFFF)
    var best_n = Float32(3.4028234663852886e38)
    var best_kn = UInt32(0xFFFFFFFF)
    var xnv = xn.unsafe_load(row)

    for j in range(k):
        var acc_f = Float32(0.0)
        var acc_n = Float32(0.0)
        for f in range(d):
            var xv = x.unsafe_load(row * d + f)
            var cv = c.unsafe_load(j * d + f)
            acc_f = fma(xv, cv, acc_f)
            acc_n = acc_n + xv * cv
        var cnv = cn.unsafe_load(j)
        var df = fma(Float32(-2.0), acc_f, xnv + cnv)
        var dn = Float32(-2.0) * acc_n + (xnv + cnv)
        # `l2_exp.cuh:127-135`, both arms, copied from the shipped epilog.
        if df <= Float32(0.0) or (
            df * df < FUSED_CLAMP_PRECISION and xnv == cnv
        ):
            df = Float32(0.0)
        if dn <= Float32(0.0) or (
            dn * dn < FUSED_CLAMP_PRECISION and xnv == cnv
        ):
            dn = Float32(0.0)
        # `raft::argmin_op`: lowest value, then lowest key.
        if df < best_f or (df == best_f and UInt32(j) < best_kf):
            best_f = df
            best_kf = UInt32(j)
        if dn < best_n or (dn == best_n and UInt32(j) < best_kn):
            best_n = dn
            best_kn = UInt32(j)

    out_val_fma.unsafe_store(row, best_f)
    out_key_fma.unsafe_store(row, best_kf)
    out_val_naive.unsafe_store(row, best_n)
    out_key_naive.unsafe_store(row, best_kn)


def _run_fused(
    ctx: DeviceContext,
    mut x: DeviceBuffer[DType.float32],
    mut c: DeviceBuffer[DType.float32],
    mut xn: DeviceBuffer[DType.float32],
    mut cn: DeviceBuffer[DType.float32],
    mut okey: DeviceBuffer[DType.uint32],
    mut oval: DeviceBuffer[DType.float32],
    n: Int,
    k: Int,
    d: Int,
    grid_y: Int,
) raises:
    """The assignment kernel at the NORMAL policy, veclen 1, at a CHOSEN
    grid. `grid_y` is the knob: it is what a different core count would
    change on another vendor, and nothing about the answer may depend on
    it."""
    comptime kern = fused_distance_nn_kernel[
        1, FUSED_NORMAL_KBLK, FUSED_NORMAL_TR, FUSED_NORMAL_TC
    ]
    comptime threads = FUSED_NORMAL_TR * FUSED_NORMAL_TC
    ctx.enqueue_function[kern](
        okey.unsafe_ptr(),
        oval.unsafe_ptr(),
        x.unsafe_ptr(),
        c.unsafe_ptr(),
        xn.unsafe_ptr(),
        cn.unsafe_ptr(),
        Int32(n),
        Int32(k),
        Int32(d),
        Int32(0),
        grid_dim=(1, grid_y, 1),
        block_dim=(threads, 1, 1),
    )
    ctx.synchronize()


def check_fused_contraction_pin() raises:
    """IDENTITY_PATHS row 19. `acc += x * y` is ONE rounding under IDENTICAL.

    The contraction is a CODEGEN decision and no runtime row reaches it:
    Metal measured UNFUSED on 2^20 patterns (`check-ieee-arith`: fused 0),
    CUDA's compiler contracts by default. So the same source produces two
    different distance matrices on two vendors, and every label downstream
    of a near-tie moves with them.

    THE GATE, and why it is shaped this way:

    - the two ORACLE arms must disagree on at least one row, or this
      fixture cannot see a contraction ON THIS BACKEND and the check
      refuses to report a pass it did not earn;
    - under IDENTICAL the shipped kernel must equal the `fma` arm BIT FOR
      BIT. That is the reach proof: an `identical_mul_add` nobody called
      would leave the kernel matching the naive arm instead;
    - under FAST the check REPORTS which arm the kernel matched rather than
      requiring one. On Apple it is the naive arm (row 9's measurement) and
      the shipped bits are therefore untouched by this lane. On a backend
      whose FAST codegen contracts, matching the `fma` arm is the truth
      about that backend, and it is a finding to record rather than a gate
      to fail.
    """
    var ctx = DeviceContext()
    var n = 256
    var d = 32
    var k = 8

    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var c = ctx.enqueue_create_buffer[DType.float32](k * d)
    var xn = ctx.enqueue_create_buffer[DType.float32](n)
    var cn = ctx.enqueue_create_buffer[DType.float32](k)
    var okey = ctx.enqueue_create_buffer[DType.uint32](n)
    var oval = ctx.enqueue_create_buffer[DType.float32](n)
    var fval = ctx.enqueue_create_buffer[DType.float32](n)
    var fkey = ctx.enqueue_create_buffer[DType.uint32](n)
    var nval = ctx.enqueue_create_buffer[DType.float32](n)
    var nkey = ctx.enqueue_create_buffer[DType.uint32](n)
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var hc = ctx.enqueue_create_host_buffer[DType.float32](k * d)
    ctx.synchronize()

    # SEPARATED clusters, not a cloud. A first attempt drew both sides from
    # the same [1, 2) spread; every distance then fell inside the epilog's
    # positivity clamp, every arm returned exactly 0.0, and the fixture
    # guard below correctly refused to run. The offsets put the distances at
    # O(10) so the arithmetic under test survives to the comparison.
    for i in range(n):
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(
                i * d + f, Float32(3.0 * Float32(i % k)) + _spread(i, f)
            )
    for j in range(k):
        for f in range(d):
            hc.unsafe_ptr().unsafe_store(
                j * d + f,
                Float32(3.0 * Float32(j)) + _spread(j + 500000, f),
            )
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=c, src_ptr=hc.unsafe_ptr())
    ctx.synchronize()

    # The norms come from the DEVICE and are handed to BOTH the kernel and
    # the oracle, so this check is about the contraction inside the distance
    # kernel and not about the norm fold, which `check_pinned_fold_shape`
    # owns.
    ctx.enqueue_function[row_norm_kernel](
        xn.unsafe_ptr(), x.unsafe_ptr(), Int32(d), Int32(0),
        grid_dim=(n, 1, 1), block_dim=(NORM_TPB, 1, 1),
    )
    ctx.enqueue_function[row_norm_kernel](
        cn.unsafe_ptr(), c.unsafe_ptr(), Int32(d), Int32(0),
        grid_dim=(k, 1, 1), block_dim=(NORM_TPB, 1, 1),
    )
    ctx.synchronize()

    ctx.enqueue_function[contraction_oracle_kernel](
        fval.unsafe_ptr(), fkey.unsafe_ptr(),
        nval.unsafe_ptr(), nkey.unsafe_ptr(),
        x.unsafe_ptr(), c.unsafe_ptr(), xn.unsafe_ptr(), cn.unsafe_ptr(),
        Int32(n), Int32(k), Int32(d),
        grid_dim=((n + 127) // 128, 1, 1), block_dim=(128, 1, 1),
    )
    ctx.synchronize()

    _run_fused(ctx, x, c, xn, cn, okey, oval, n, k, d, 4)

    var hval = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hfval = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hnval = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.enqueue_copy(dst_ptr=hval.unsafe_ptr(), src_buf=oval)
    ctx.enqueue_copy(dst_ptr=hfval.unsafe_ptr(), src_buf=fval)
    ctx.enqueue_copy(dst_ptr=hnval.unsafe_ptr(), src_buf=nval)
    ctx.synchronize()

    var separating = 0
    var off_fma = 0
    var off_naive = 0
    for i in range(n):
        var got = hval.unsafe_ptr().unsafe_load(i)
        var vf = hfval.unsafe_ptr().unsafe_load(i)
        var vn = hnval.unsafe_ptr().unsafe_load(i)
        if vf != vn:
            separating += 1
        if got != vf:
            off_fma += 1
        if got != vn:
            off_naive += 1

    # WHAT `separating == 0` MEANS HERE, AND WHY IT IS NOT A FAILURE.
    #
    # It means the BACKEND contracts the naive spelling too, so both oracle
    # arms compile to the same instruction and no fixture can tell them
    # apart. That is the case on Metal through MAX, MEASURED 2026-08-23:
    # `check-ieee-arith`'s new built-to-separate arm reports FUSED on 1,629
    # of 1,629 patterns constructed so that one rounding and two CANNOT
    # agree. (Its old verdict said UNFUSED off 2^20 hashed patterns, of
    # which ZERO separated the two spellings -- the tie-counting bug that
    # sent IDENTITY_PATHS row 9's Apple sentence into the ledger. Both the
    # check and the row are corrected.)
    #
    # So on this backend the contraction pin is BIT-INERT, exactly as
    # `numerics.ftz` is bit-inert on an FTZ backend, and its value is the
    # vendor that does NOT contract by default -- where an unpinned
    # `acc += x * y` would round twice against Metal's once. The pin still
    # has to be REACHED for that to be true, which is what the equality
    # below tests: a kernel computing anything other than the one-rounding
    # answer fails it whether or not the two arms agree.
    comptime if IDENTICAL_BUILD:
        if off_fma != 0:
            raise Error(
                "check_fused_contraction_pin (IDENTICAL): "
                + String(off_fma)
                + " of "
                + String(n)
                + " rows disagree with the FUSED oracle. Under IDENTICAL"
                " every multiply-add on this path goes through"
                " `identical_mul_add`, which is one rounding by"
                " construction, so this is a broken pin or a broken port."
            )
        if separating == 0:
            print(
                "check_fused_contraction_pin OK (IDENTICAL):",
                n,
                "rows match the one-rounding oracle bit for bit. This"
                " backend CONTRACTS the naive spelling too (0 of",
                n,
                "rows separate the arms), so the pin is bit-inert here and"
                " buys its property on a backend that does not.",
            )
        else:
            print(
                "check_fused_contraction_pin OK (IDENTICAL):",
                n,
                "rows match the fused oracle and",
                off_naive,
                "differ from the unfused one, on a backend where",
                separating,
                "rows separate the two spellings",
            )
    else:
        if separating == 0:
            if off_fma != 0:
                raise Error(
                    "check_fused_contraction_pin (FAST): the two oracle"
                    " arms agree (this backend contracts both spellings)"
                    " and the shipped kernel matches NEITHER on "
                    + String(off_fma)
                    + " rows. Something other than the contraction differs"
                    " between the kernel and the oracle."
                )
            print(
                "check_fused_contraction_pin OK (FAST):",
                n,
                "rows match the oracle; this backend contracts the naive"
                " spelling as well, so FAST and IDENTICAL take the same"
                " instruction here and the lane's sweep moved no shipped"
                " bits.",
            )
        else:
            var arm = String("NEITHER")
            if off_naive == 0:
                arm = String("unfused")
            elif off_fma == 0:
                arm = String("FUSED (this backend contracts under FAST)")
            if off_naive != 0 and off_fma != 0:
                raise Error(
                    "check_fused_contraction_pin (FAST): the kernel matches"
                    " NEITHER oracle ("
                    + String(off_fma)
                    + " off fused, "
                    + String(off_naive)
                    + " off unfused) on a backend where the arms differ."
                )
            print(
                "check_fused_contraction_pin OK (FAST): the shipped kernel"
                " is",
                arm + ",",
                "on a backend where",
                separating,
                "of",
                n,
                "rows separate the two spellings",
            )


def check_assignment_geometry_invariance() raises:
    """IDENTITY_PATHS row 21. A different grid is a different MACHINE.

    Another vendor's core count reaches this kernel as a different
    `grid_dim.y` and nothing else. So the same fixture is assigned at four
    grids -- including one block for the whole dataset, which forces the
    outer grid-stride loop -- and every output BIT must agree.

    This must pass in BOTH modes. Geometry independence is not something
    `IDENTICAL` buys: it is what the argmin's `(value, key)` total order
    (`raft::argmin_op`, and OURS in the fused arm where theirs compares the
    value only) and the fixed row ownership already bought. A failure here
    is a defect in the port, not a missing pin.
    """
    var ctx = DeviceContext()
    var n = 512
    var d = 32
    var k = 12

    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var c = ctx.enqueue_create_buffer[DType.float32](k * d)
    var xn = ctx.enqueue_create_buffer[DType.float32](n)
    var cn = ctx.enqueue_create_buffer[DType.float32](k)
    var okey = ctx.enqueue_create_buffer[DType.uint32](n)
    var oval = ctx.enqueue_create_buffer[DType.float32](n)
    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var hc = ctx.enqueue_create_host_buffer[DType.float32](k * d)
    ctx.synchronize()

    for i in range(n):
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(i * d + f, _spread(i, f))
    # PLANTED EXACT TIES: every fourth centroid is a COPY of centroid 0, so
    # a row nearest to it is equidistant from several keys and only a total
    # order can decide. Without these the check would pass on a fixture
    # where no tie ever occurred.
    for j in range(k):
        for f in range(d):
            var src = 0 if (j % 4) == 0 else j
            hc.unsafe_ptr().unsafe_store(
                j * d + f, _spread(src + 500000, f)
            )
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=c, src_ptr=hc.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[row_norm_kernel](
        xn.unsafe_ptr(), x.unsafe_ptr(), Int32(d), Int32(0),
        grid_dim=(n, 1, 1), block_dim=(NORM_TPB, 1, 1),
    )
    ctx.enqueue_function[row_norm_kernel](
        cn.unsafe_ptr(), c.unsafe_ptr(), Int32(d), Int32(0),
        grid_dim=(k, 1, 1), block_dim=(NORM_TPB, 1, 1),
    )
    ctx.synchronize()

    var base_val = List[Float32]()
    var base_key = List[UInt32]()
    var hval = ctx.enqueue_create_host_buffer[DType.float32](n)
    var hkey = ctx.enqueue_create_host_buffer[DType.uint32](n)

    var grids = List[Int]()
    grids.append(1)
    grids.append(2)
    grids.append(5)
    grids.append(8)

    var tied_rows = 0
    for g in range(len(grids)):
        _run_fused(ctx, x, c, xn, cn, okey, oval, n, k, d, grids[g])
        ctx.enqueue_copy(dst_ptr=hval.unsafe_ptr(), src_buf=oval)
        ctx.enqueue_copy(dst_ptr=hkey.unsafe_ptr(), src_buf=okey)
        ctx.synchronize()
        if g == 0:
            for i in range(n):
                base_val.append(hval.unsafe_ptr().unsafe_load(i))
                base_key.append(hkey.unsafe_ptr().unsafe_load(i))
                # A row whose winner is a duplicated centroid is a row the
                # tie-break decided.
                if Int(base_key[i]) % 4 == 0:
                    tied_rows += 1
        else:
            var moved = 0
            for i in range(n):
                if hval.unsafe_ptr().unsafe_load(i) != base_val[i]:
                    moved += 1
                elif hkey.unsafe_ptr().unsafe_load(i) != base_key[i]:
                    moved += 1
            if moved != 0:
                raise Error(
                    "check_assignment_geometry_invariance: grid_y "
                    + String(grids[g])
                    + " moved "
                    + String(moved)
                    + " of "
                    + String(n)
                    + " assignments against grid_y "
                    + String(grids[0])
                    + ". The answer depends on the launch shape, which is"
                    " the cross-vendor hazard reproduced on one box."
                )

    if tied_rows == 0:
        raise Error(
            "check_assignment_geometry_invariance: the planted duplicate"
            " centroids won no row, so no tie was ever resolved and the"
            " check did not exercise the total order."
        )
    print(
        "check_assignment_geometry_invariance OK (" + _mode_name() + "):",
        n,
        "assignments bit-identical at grid_y 1/2/5/8, with",
        tied_rows,
        "of them decided by the (value, key) tie-break",
    )


def check_float_fold_rows_are_pinned() raises:
    """IDENTITY_PATHS row 22 / DEVIATION 508, the STRUCTURAL half.

    `lib_block_size_for` is the comptime accessor every library kernel
    compiles its threadgroup against, and the matrix calls it SCHEDULING.
    For three rows that is wrong -- the block size is the WIDTH of a float
    `block.sum` -- and the table's own docstring invites a per-vendor number
    to land there without touching a kernel.

    This asserts what the gate guarantees: under IDENTICAL every column
    resolves those three rows to the SAME width, so a future measured value
    cannot silently split the vendors. It also pins the CLASSIFIER's
    membership, so adding a fourth float fold without adding it to
    `lib_block_bounds_a_float_fold` fails here rather than in E1.
    """
    if not lib_block_bounds_a_float_fold[K_LIB_ROW_NORM]():
        raise Error("K_LIB_ROW_NORM is a float fold and is not classified")
    if not lib_block_bounds_a_float_fold[K_LIB_REDUCE_BY_KEY]():
        raise Error("K_LIB_REDUCE_BY_KEY is a float fold and is not classified")
    if not lib_block_bounds_a_float_fold[K_LIB_PLUS_PLUS]():
        raise Error("K_LIB_PLUS_PLUS is a float fold and is not classified")
    # The integer folds must NOT be classified: pinning them would cost a
    # vendor its block size for no property, and claiming they need it would
    # misstate why the other three do.
    if lib_block_bounds_a_float_fold[K_LIB_EPS_NEIGHBORHOOD]():
        raise Error(
            "K_LIB_EPS_NEIGHBORHOOD folds Int32, which is associative; it"
            " must not be pinned as a float fold"
        )
    if lib_block_bounds_a_float_fold[K_LIB_ADJ_SCAN]():
        raise Error("K_LIB_ADJ_SCAN folds Int32; it must not be pinned")
    if lib_block_bounds_a_float_fold[K_LIB_WEAK_CC]():
        raise Error("K_LIB_WEAK_CC has no fold; it must not be pinned")

    var apple = lib_block_size_for[K_LIB_ROW_NORM, COLUMN_APPLE]()
    var nvidia = lib_block_size_for[K_LIB_ROW_NORM, COLUMN_NVIDIA]()
    var amd = lib_block_size_for[K_LIB_ROW_NORM, COLUMN_AMD]()
    var floor = lib_block_size_for[K_LIB_ROW_NORM, COLUMN_BIT_IDENTICAL]()

    comptime if IDENTICAL_BUILD:
        if apple != floor or nvidia != floor or amd != floor:
            raise Error(
                "check_float_fold_rows_are_pinned: under IDENTICAL the"
                " row-norm fold width resolves to "
                + String(apple)
                + "/"
                + String(nvidia)
                + "/"
                + String(amd)
                + " on Apple/NVIDIA/AMD against a floor of "
                + String(floor)
                + ". A fold width that follows the column is a summation"
                " order that follows the machine."
            )
        print(
            "check_float_fold_rows_are_pinned OK (IDENTICAL): the three"
            " float-fold rows resolve to",
            floor,
            "in every column, and the three integer rows are left free",
        )
    else:
        print(
            "check_float_fold_rows_are_pinned OK (FAST): classifier holds;"
            " columns are free and today all read",
            apple,
            "for row-norm",
        )


def check_kmeans_trace_localizes() raises:
    """The instrument, on the real fit: same fixture twice, then perturbed.

    Two traces of the same fit must agree record for record -- that is the
    run-to-run half of the claim, and it is not free: it fails the moment a
    float atomic or an unsorted scatter reaches the model.

    Then ONE cell of X is perturbed and the differ must name `fit.x_norm`
    as the FIRST divergence rather than reporting that the model changed. A
    hash that only disagrees at the end localizes nothing.

    THE PERTURBATION IS 1,024 ULPs, NOT ONE, AND THAT IS A MEASUREMENT.
    A single-ULP bump on one feature changed NO stage hash: the squared
    term moves by about 3e-7 while the eight-term norm it lands in has an
    ULP near 2e-6, so the perturbation is absorbed by the norm's own
    rounding and the fit is bit-identical. That is worth knowing about the
    instrument -- it sees what reaches a buffer, not what was nudged -- and
    1,024 ULPs is still a relative 1e-4, far below any tolerance-based
    check and exactly the size of drift a cross-vendor divergence produces.
    """
    var ctx = DeviceContext()
    var n = 384
    var d = 8
    var k = 5

    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * d)
    var hc = ctx.enqueue_create_host_buffer[DType.float32](k * d)
    ctx.synchronize()
    for i in range(n):
        for f in range(d):
            hx.unsafe_ptr().unsafe_store(i * d + f, _spread(i, f))
    for j in range(k):
        for f in range(d):
            hc.unsafe_ptr().unsafe_store(j * d + f, _spread(j + 777, f))

    var path_a = String("/tmp/mojolearn_kmeans_identity_a.trace")
    var path_b = String("/tmp/mojolearn_kmeans_identity_b.trace")
    var path_c = String("/tmp/mojolearn_kmeans_identity_c.trace")

    _traced_fit(ctx, hx, hc, n, d, k, path_a, -1)
    _traced_fit(ctx, hx, hc, n, d, k, path_b, -1)
    var same = first_divergence(path_a, path_b)
    if same != "":
        raise Error(
            "check_kmeans_trace_localizes: two runs of the SAME fit"
            " diverged at " + same
        )

    _traced_fit(ctx, hx, hc, n, d, k, path_c, 17 * d + 3)
    var moved = first_divergence(path_a, path_c)
    if moved == "":
        raise Error(
            "check_kmeans_trace_localizes: the perturbation of X"
            " changed NO stage hash. The trace is not reading the buffers"
            " the fit uses."
        )
    if moved.find("fit.x_norm") < 0:
        raise Error(
            "check_kmeans_trace_localizes: the first divergence was '"
            + moved
            + "', not fit.x_norm. A perturbation of X must show up in the"
            " first stage that reads X, or the instrument is naming the"
            " wrong place."
        )
    print(
        "check_kmeans_trace_localizes OK (" + _mode_name() + "): two runs",
        "agree on every stage hash;",
        _PERTURB_ULPS,
        "ULPs on one cell of X first show at fit.x_norm",
    )


def _traced_fit(
    ctx: DeviceContext,
    mut hx: HostBuffer[DType.float32],
    mut hc: HostBuffer[DType.float32],
    n: Int,
    d: Int,
    k: Int,
    path: String,
    perturb_cell: Int,
) raises:
    """One `kmeans_fit_main` with its trace pointed at `path`.

    `perturb_cell >= 0` bumps that cell of X by `_PERTURB_ULPS` ULPs before
    the fit, which is the sabotage the localization claim needs.

    The trace is taken through the ENVIRONMENT, because that is the only
    channel `kmeans_fit_main` reads and a check that reached past it would
    be testing a path no user has.
    """
    from std.os import setenv

    var x = ctx.enqueue_create_buffer[DType.float32](n * d)
    var w = ctx.enqueue_create_buffer[DType.float32](n)
    var cent = ctx.enqueue_create_buffer[DType.float32](k * d)
    var labels = ctx.enqueue_create_buffer[DType.uint32](n)
    var hw = ctx.enqueue_create_host_buffer[DType.float32](n)
    ctx.synchronize()

    var saved = Float32(0.0)
    if perturb_cell >= 0:
        saved = hx.unsafe_ptr().unsafe_load(perturb_cell)
        var bits = bitcast[DType.uint32](saved)
        hx.unsafe_ptr().unsafe_store(
            perturb_cell, bitcast[DType.float32](bits + UInt32(_PERTURB_ULPS))
        )
    for i in range(n):
        hw.unsafe_ptr().unsafe_store(i, Float32(1.0))
    ctx.enqueue_copy(dst_buf=x, src_ptr=hx.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=w, src_ptr=hw.unsafe_ptr())
    ctx.enqueue_copy(dst_buf=cent, src_ptr=hc.unsafe_ptr())
    ctx.synchronize()
    if perturb_cell >= 0:
        hx.unsafe_ptr().unsafe_store(perturb_cell, saved)

    # Truncate: `IdentityTrace`'s env constructor APPENDS deliberately, and
    # three fits into one path would produce a file the differ refuses.
    with open(path, "w") as fh:
        fh.write("")
    _ = setenv("MOJOLEARN_IDENTITY_TRACE", path, True)

    var params = KMeansParams.default()
    params.n_clusters = k
    params.init = INIT_ARRAY
    params.max_iter = 8
    params.n_init = 1
    params.seed = 7
    _ = kmeans_fit_main(
        ctx, x, w, cent, labels, params, n, d,
        Float32(4096.0), Float32(4096.0),
    )
    _ = setenv("MOJOLEARN_IDENTITY_TRACE", "", True)


def main() raises:
    check_float_fold_rows_are_pinned()
    check_pinned_fold_shape()
    check_fused_contraction_pin()
    check_assignment_geometry_invariance()
    check_kmeans_trace_localizes()
