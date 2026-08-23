"""Does the device eigensolver still answer above 32 features? Prove it.

NO RAFT COUNTERPART. This exists because of a shipped bug: the device Jacobi
held both the matrix and the accumulated basis in THREADGROUP memory, which
forced a `JACOBI_MAX_N = 32` cap, and a fit at 33 or more features silently
returned an answer that was not an eigendecomposition of anything. Nothing in
the section noticed, because every check in `pca_check.mojo` runs at
`PCA_COLS = 4`.

**A check that only ever runs below a cap cannot see the cap.** So this one
walks sizes ACROSS where the cap used to be: 16, 32, 33, 64, 128, 256. 33 is
the load-bearing one.

WHAT IS ASSERTED, AND WHY EACH IS NOT ENOUGH ALONE
--------------------------------------------------
1. `V^T V = I`. Orthonormality. Jacobi accumulates exact orthogonal
   rotations, so this holds no matter how badly it converged; it catches a
   basis update that skipped indices but NOT a solver that stopped early.
2. `A V = V diag(lambda)`, per cell, against a SAVED copy of the original
   `A`. This is the eigendecomposition property itself and it is the one that
   fails under the old cap: entries beyond 32 were never rotated, so the
   residual there is the size of the matrix.
3. The spectrum against the HOST Float64 reference in `jacobi_eigh.mojo`,
   sorted descending. Two independent implementations of the same rotation
   sequence, one in double on the host and one in single on the device.

The fixture is HASHED, not uniform. A matrix whose every entry is the same
number has the same expected value in every cell, and a check on it verifies
a total and nothing about placement; that exact mistake passed a
wrong-reduction bug twice in this repository. Every entry here is a distinct
value from splitmix64, so a rotation applied to the wrong index moves a
number that no other index would have produced.

FLOAT32 vs FLOAT64, AND THE TOLERANCE
-------------------------------------
The device solver is Float32 because Metal has no double. The host reference
is Float64. **They are not expected to agree bit for bit.** Everything is
compared RELATIVE to the matrix scale (`||A||_F`), with the tolerances stated
in `_TOL_*` below and printed with every result, so a reader never has to
guess what "passes" meant.
"""

from std.gpu import thread_idx
from std.math import sqrt
from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.primitives.block import sum as block_sum

from core.pinned_reduce import pinned_block_sum
from decomposition.mojo_only.jacobi_eigh import jacobi_eigh
from decomposition.mojo_only.jacobi_eigh_device import (
    JACOBI_TPB,
    jacobi_eigh_kernel,
)
from mojo_only.kernel_matrix import (
    COLUMN_AMD,
    COLUMN_APPLE,
    COLUMN_BIT_IDENTICAL,
    COLUMN_NVIDIA,
    K_LIB_JACOBI_EIGH,
    TARGET_COLUMN,
    lib_block_size_for,
    lib_lane_width_for,
)
from mojo_only.numerics import (
    GLOBAL_NUMERIC_MODE,
    NUMERIC_IDENTICAL,
    ftz,
    identical_mul_add,
    identical_sqrt,
)


comptime IDENTICAL_BUILD = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL


def _mode_name() -> String:
    comptime if IDENTICAL_BUILD:
        return String("IDENTICAL")
    return String("FAST")


# Relative to ||A||_F. Float32 Jacobi on an n x n matrix accumulates about
# sqrt(n) * eps32 per entry over its rotations, and eps32 is 1.2e-7, so at
# n = 256 the floor is near 2e-6. These sit an order of magnitude above it.
comptime _TOL_RESIDUAL = 5.0e-5  # || A V - V diag(lam) ||_max / ||A||_F
comptime _TOL_ORTHO = 2.0e-4  # || V^T V - I ||_max
comptime _TOL_SPECTRUM = 5.0e-4  # |lam_device - lam_host| / ||A||_F



def _hashed(i: Int, j: Int, seed: Int) -> Float64:
    """splitmix64 on the CELL, so every entry of the fixture is distinct.

    Symmetric by construction: the mixer is fed `min, max` of the pair, so
    `(i, j)` and `(j, i)` hash to the same value and `A` is exactly
    symmetric without a second pass.
    """
    var lo = i if i < j else j
    var hi = j if i < j else i
    var z = (
        UInt64(lo + 1) * 0x9E3779B97F4A7C15
        + UInt64(hi + 1) * 0xBF58476D1CE4E5B9
        + UInt64(seed) * 0x94D049BB133111EB
    )
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float64(z >> 11) * (1.0 / 9007199254740992.0) - 0.5


def _make_symmetric(n: Int, seed: Int, spike_at: Int, spike: Float64) -> List[Float64]:
    """An `n x n` symmetric matrix of distinct hashed entries.

    The diagonal is boosted so the spectrum is well separated and the
    eigenvalue comparison is not testing a degenerate subspace, which is
    defined only up to rotation and would fail for reasons that are not bugs.

    `spike_at` plants a large diagonal entry at one index. That is the
    CAP-SPECIFIC probe: under the old 32-wide threadgroup arrays an entry at
    index 63 was not in the matrix the kernel saw, so the spectrum could not
    respond to it.
    """
    var a = List[Float64]()
    for i in range(n):
        for j in range(n):
            a.append(_hashed(i, j, seed))
    for i in range(n):
        a[i * n + i] += Float64(i % 7) + 1.0
    if spike_at >= 0:
        a[spike_at * n + spike_at] += spike
    return a^


def _run_device(
    ctx: DeviceContext,
    a: List[Float64],
    n: Int,
    max_sweeps: Int,
    tol: Float64,
) raises -> Tuple[List[Float64], List[Float64], List[Float64]]:
    """Returns (eigenvalues, eigenvectors row major, the kernel info slots).

    The third element is `[converged, ||offdiag||_F / ||A||_F, sweeps]`, so a
    printed result says what the answer actually cost and whether the sweep
    limit was hit. A solver that quietly stopped early is a wrong answer with
    no error, which is the failure mode this whole file exists because of.

    The kernel DESTROYS the matrix it is given, so the caller keeps the
    original; this uploads a copy.
    """
    var a_buf = ctx.enqueue_create_buffer[DType.float32](n * n)
    var v_buf = ctx.enqueue_create_buffer[DType.float32](n * n)
    var i_buf = ctx.enqueue_create_buffer[DType.float32](3)
    var h = ctx.enqueue_create_host_buffer[DType.float32](n * n)
    ctx.synchronize()
    for i in range(n * n):
        h.unsafe_ptr().unsafe_store(i, Float32(a[i]))
    ctx.enqueue_copy(dst_buf=a_buf, src_ptr=h.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[jacobi_eigh_kernel](
        a_buf.unsafe_ptr(),
        v_buf.unsafe_ptr(),
        i_buf.unsafe_ptr(),
        Int32(n),
        Int32(max_sweeps),
        Float32(tol),
        grid_dim=(1, 1, 1),
        block_dim=(JACOBI_TPB, 1, 1),
    )
    ctx.synchronize()

    var ha = ctx.enqueue_create_host_buffer[DType.float32](n * n)
    var hv = ctx.enqueue_create_host_buffer[DType.float32](n * n)
    var hi = ctx.enqueue_create_host_buffer[DType.float32](3)
    ctx.enqueue_copy(dst_ptr=ha.unsafe_ptr(), src_buf=a_buf)
    ctx.enqueue_copy(dst_ptr=hv.unsafe_ptr(), src_buf=v_buf)
    ctx.enqueue_copy(dst_ptr=hi.unsafe_ptr(), src_buf=i_buf)
    ctx.synchronize()


    var lam = List[Float64]()
    for i in range(n):
        lam.append(Float64(ha.unsafe_ptr().unsafe_load(i * n + i)))
    var vecs = List[Float64]()
    for i in range(n * n):
        vecs.append(Float64(hv.unsafe_ptr().unsafe_load(i)))
    var info = List[Float64]()
    for i in range(3):
        info.append(Float64(hi.unsafe_ptr().unsafe_load(i)))
    return (lam^, vecs^, info^)


def _fro(a: List[Float64], n: Int) -> Float64:
    var s = 0.0
    for i in range(n * n):
        s += a[i] * a[i]
    return sqrt(s)


def _residual(
    a: List[Float64], lam: List[Float64], v: List[Float64], n: Int
) -> Float64:
    """max over CELLS of |(A V)[r][c] - lam[c] * V[r][c]|.

    Per cell, deliberately. A norm over the whole matrix would let one badly
    placed column hide inside a correct total.
    """
    var worst = 0.0
    for r in range(n):
        for c in range(n):
            var av = 0.0
            for k in range(n):
                av += a[r * n + k] * v[k * n + c]
            var d = abs(av - lam[c] * v[r * n + c])
            if d > worst:
                worst = d
    return worst


def _ortho_error(v: List[Float64], n: Int) -> Float64:
    var worst = 0.0
    for i in range(n):
        for j in range(n):
            var d = 0.0
            for k in range(n):
                d += v[k * n + i] * v[k * n + j]
            var want = 1.0 if i == j else 0.0
            if abs(d - want) > worst:
                worst = abs(d - want)
    return worst


def _sorted_desc(x: List[Float64]) -> List[Float64]:
    var out = List[Float64]()
    for i in range(len(x)):
        out.append(x[i])
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            if out[j] > out[i]:
                var t = out[i]
                out[i] = out[j]
                out[j] = t
    return out^


def _host_spectrum(a: List[Float64], n: Int, sweeps: Int) raises -> List[Float64]:
    var work = List[Float64]()
    for i in range(n * n):
        work.append(a[i])
    var vecs = List[Float64]()
    for _ in range(n * n):
        vecs.append(0.0)
    jacobi_eigh(work, vecs, n, sweeps, 1.0e-14)
    var lam = List[Float64]()
    for i in range(n):
        lam.append(work[i * n + i])
    return _sorted_desc(lam)


def check_jacobi_device_sizes() raises:
    """The whole point of the file: 16, 32, 33, 64, 128, 256.

    33 is where the old `JACOBI_MAX_N = 32` first produced a wrong answer
    with no error.
    """
    var ctx = DeviceContext()
    var sizes = List[Int]()
    sizes.append(16)
    sizes.append(32)
    sizes.append(33)
    sizes.append(64)
    sizes.append(128)
    sizes.append(256)

    for s in range(len(sizes)):
        var n = sizes[s]
        var a = _make_symmetric(n, 7, -1, 0.0)
        var scale = _fro(a, n)
        var got = _run_device(ctx, a, n, 40, 1.0e-10)
        var lam = got[0].copy()
        var v = got[1].copy()
        var info = got[2].copy()

        var ortho = _ortho_error(v, n)
        if ortho > _TOL_ORTHO:
            raise Error(
                "n = " + String(n) + ": V^T V is not the identity, worst"
                " entry off by " + String(ortho) + " against a tolerance of "
                + String(_TOL_ORTHO)
            )

        var res = _residual(a, lam, v, n) / scale
        if res > _TOL_RESIDUAL:
            raise Error(
                "n = " + String(n) + ": A V != V diag(lambda). Worst CELL"
                " residual " + String(res) + " relative to ||A||_F, against"
                " a tolerance of " + String(_TOL_RESIDUAL) + ". This is the"
                " shape the JACOBI_MAX_N = 32 cap produced above 32."
            )

        # Cross-check against the host Float64 reference. Capped at 128
        # because the host solver is O(n^3) per sweep in a single thread and
        # 256 costs minutes for a spectrum the residual check already pinned.
        var spec_err = -1.0
        if n <= 128:
            var host_lam = _host_spectrum(a, n, 60)
            var dev_lam = _sorted_desc(lam)
            spec_err = 0.0
            for i in range(n):
                var d = abs(dev_lam[i] - host_lam[i]) / scale
                if d > spec_err:
                    spec_err = d
            if spec_err > _TOL_SPECTRUM:
                raise Error(
                    "n = " + String(n) + ": device Float32 spectrum differs"
                    " from the host Float64 reference by " + String(spec_err)
                    + " relative to ||A||_F, against a tolerance of "
                    + String(_TOL_SPECTRUM)
                )

        print(
            "  n = " + String(n) + ": ||V^T V - I||_max = " + String(ortho)
            + ", worst cell of A V - V diag(lam) / ||A||_F = " + String(res)
            + ", vs host Float64 spectrum = " + String(spec_err)
            + ", sweeps = " + String(info[2])
            + ", final ||offdiag||/||A|| = " + String(info[1])
            + ("" if info[0] > 0.0 else " [DID NOT CONVERGE]")
            + (" (skipped, host too slow)" if spec_err < 0.0 else "")
        )

    print(
        "check_jacobi_device_sizes OK: 16, 32, 33, 64, 128, 256 all satisfy"
        " A V = V diag(lambda) and V^T V = I. The 32-feature cap is gone."
    )


def check_jacobi_reaches_past_32() raises:
    """SABOTAGE, in the shape the old cap would have failed.

    A cap at 32 does not raise and does not print. The only way to see it is
    to make an entry BEYOND it matter and check that the answer responds.

    Plant `+1000` on the diagonal at index 63 of a 64 x 64 matrix. The
    largest eigenvalue of a symmetric matrix is at least its largest diagonal
    entry, so it MUST move to roughly 1000. Under the old threadgroup arrays
    index 63 was outside the matrix the kernel saw, and the spectrum could
    not have moved at all.

    Run without the spike first, so the assertion is on the DIFFERENCE and
    not on an absolute number that some other bug could also produce.
    """
    var ctx = DeviceContext()
    var n = 64

    var plain = _make_symmetric(n, 11, -1, 0.0)
    var got0 = _run_device(ctx, plain, n, 40, 1.0e-10)
    var top0 = _sorted_desc(got0[0].copy())[0]

    var spiked = _make_symmetric(n, 11, 63, 1000.0)
    var got1 = _run_device(ctx, spiked, n, 40, 1.0e-10)
    var top1 = _sorted_desc(got1[0].copy())[0]

    if top1 - top0 < 900.0:
        raise Error(
            "SABOTAGE FAILED TO MOVE THE ANSWER: a +1000 spike at index 63 of"
            " a 64 x 64 matrix moved the largest eigenvalue only from "
            + String(top0) + " to " + String(top1) + ". The kernel is not"
            " reading index 63, which is exactly what the 32-wide"
            " threadgroup arrays did."
        )

    # And the same spike at an index the OLD cap DID cover must also move it,
    # so the test above is not passing for some reason unrelated to reach.
    var near = _make_symmetric(n, 11, 5, 1000.0)
    var got2 = _run_device(ctx, near, n, 40, 1.0e-10)
    var top2 = _sorted_desc(got2[0].copy())[0]
    if top2 - top0 < 900.0:
        raise Error(
            "a +1000 spike at index 5 did not move the largest eigenvalue"
            " either (" + String(top0) + " -> " + String(top2) + "), so the"
            " probe itself is broken, not the reach"
        )

    print(
        "check_jacobi_reaches_past_32 OK: top eigenvalue "
        + String(top0) + " -> " + String(top1) + " for a +1000 spike at"
        " index 63 (past the old cap), and -> " + String(top2)
        + " for the same spike at index 5 (inside it). Both move, so the"
        " kernel reads the whole matrix."
    )


def check_jacobi_scale_invariance() raises:
    """A tolerance that is not scale invariant is not a tolerance.

    Multiply the SAME matrix by 1000. Jacobi's rotations are unchanged by a
    uniform scale -- every angle depends only on ratios -- so the eigenvectors
    must be identical, the eigenvalues must scale by exactly 1000, and the
    solver must take the SAME NUMBER OF SWEEPS.

    That last one is the whole point. The convergence test used to be
    `off <= 1e-10` on an ABSOLUTE sum of squares, and `off` scales with the
    SQUARE of the data, so scaling by 1000 moved the target a million times
    further away. The same matrix in different units was a different problem.
    """
    var ctx = DeviceContext()
    var n = 64
    var a = _make_symmetric(n, 3, -1, 0.0)
    var big = List[Float64]()
    for i in range(n * n):
        big.append(a[i] * 1000.0)

    var g0 = _run_device(ctx, a, n, 40, 1.0e-7)
    var g1 = _run_device(ctx, big, n, 40, 1.0e-7)

    if g0[2][2] != g1[2][2]:
        raise Error(
            "scaling the matrix by 1000 changed the sweep count from "
            + String(g0[2][2]) + " to " + String(g1[2][2])
            + ". The convergence test is not scale invariant."
        )
    if g1[2][0] == 0.0:
        raise Error(
            "the scaled matrix did NOT converge in 40 sweeps (relative"
            " off-diagonal " + String(g1[2][1]) + "), while the unscaled one"
            " did. The convergence test is not scale invariant."
        )

    # And the answer really is the same problem: eigenvalues scale by 1000.
    var l0 = _sorted_desc(g0[0].copy())
    var l1 = _sorted_desc(g1[0].copy())
    var worst = 0.0
    var scale = _fro(a, n)
    for i in range(n):
        var d = abs(l1[i] / 1000.0 - l0[i]) / scale
        if d > worst:
            worst = d
    if worst > _TOL_SPECTRUM:
        raise Error(
            "scaling by 1000 did not scale the spectrum by 1000: worst"
            " relative difference " + String(worst)
        )

    print(
        "check_jacobi_scale_invariance OK: A and 1000 A both converge in "
        + String(g0[2][2]) + " sweeps to relative off-diagonal "
        + String(g0[2][1]) + " and " + String(g1[2][1])
        + ", and the spectra agree to " + String(worst) + " after dividing"
        " out the scale"
    )


# ========================================================================
# DEVIATION 524 -- WHAT `IDENTICAL` PROMISES ABOUT THE EIGENSOLVER
# ========================================================================
# IDENTITY_PATHS row 31. Everything above this line asks whether the answer
# is RIGHT. Everything below asks whether it is THE SAME ANSWER on every
# vendor, which is a different question and was not asked here at all until
# this round: the file had two `block.sum` folds whose cross-lane stage
# follows the hardware warp width (32 on Apple and NVIDIA, 64 on AMD) and a
# rotation written as plain `a*b + c` at every seam.
#
# THE ONE THAT IS WORSE THAN A DRIFT. The sweep loop compares the folded
# `off` against `limit` to decide whether to stop. So a last-bit
# disagreement between two vendors' folds does not perturb the answer, it
# changes the NUMBER OF SWEEPS, and a sweep is n(n-1)/2 rotations applied to
# both the matrix and the basis. `check_jacobi_sweep_count_is_a_knife_edge`
# measures that blast radius rather than asserting it is small, because the
# comment that used to sit in the kernel asserted exactly that and was
# wrong.
#
# WHAT EACH CHECK CAN PROVE ON ONE DEVICE, WHICH IS NOT THE CROSS-VENDOR
# CLAIM (only E1 is that):
#
#   fold width      a structural gate on the matrix row, no device needed
#   fold shape      the device's fold IS the halving tree, bit for bit,
#                   against a host fold written out in the check
#   pure function   the WHOLE solver replayed on the host in Float32, with
#                   the same partials, the same fold and the same pinned
#                   rotation spelling, must equal the device BIT FOR BIT
#   knife edge      one ulp of the exit quantity moves the sweep count and
#                   what that costs, in eigenvector cells
#   denormals       what this backend does with a matrix made of them
#
# The third is the one with teeth: it fails if ANY seam in the kernel is
# spelled differently from the model, which is what makes the sabotage in
# the lane report meaningful.


comptime _FOLD_PROBE_TPB = JACOBI_TPB

#: The sentinel the probe writes when the library fold cannot be instantiated
#: at `_FOLD_PROBE_TPB` on this column. A NaN would be tempting and is wrong:
#: it compares unequal to everything including itself, so a comparison that
#: forgot to check the sentinel would report a DIFFERENCE where the truth is
#: that no measurement was taken.
comptime _LIBRARY_ARM_ABSENT = Float32(-987654.0)


def _f32_bits(x: Float32) -> UInt32:
    return rebind[UInt32](x.to_bits())


def _bits_f32(b: UInt32) -> Float32:
    return bitcast[DType.float32](b)


def _run_device_f32(
    ctx: DeviceContext,
    a: List[Float32],
    n: Int,
    max_sweeps: Int,
    tol: Float32,
    mut a_out: List[Float32],
    mut v_out: List[Float32],
) raises -> List[Float32]:
    """`_run_device` without the Float64 round trip, for BIT comparisons.

    Returns the three info slots; fills `a_out` and `v_out` with the raw
    Float32 the kernel wrote. A Float32 -> Float64 -> Float32 round trip is
    exact, so this is not a correctness difference; it is here so that a
    check about bits never has a conversion in its evidence chain.
    """
    var a_buf = ctx.enqueue_create_buffer[DType.float32](n * n)
    var v_buf = ctx.enqueue_create_buffer[DType.float32](n * n)
    var i_buf = ctx.enqueue_create_buffer[DType.float32](3)
    var h = ctx.enqueue_create_host_buffer[DType.float32](n * n)
    ctx.synchronize()
    for i in range(n * n):
        h.unsafe_ptr().unsafe_store(i, a[i])
    ctx.enqueue_copy(dst_buf=a_buf, src_ptr=h.unsafe_ptr())
    ctx.synchronize()

    ctx.enqueue_function[jacobi_eigh_kernel](
        a_buf.unsafe_ptr(),
        v_buf.unsafe_ptr(),
        i_buf.unsafe_ptr(),
        Int32(n),
        Int32(max_sweeps),
        tol,
        grid_dim=(1, 1, 1),
        block_dim=(JACOBI_TPB, 1, 1),
    )
    ctx.synchronize()

    var ha = ctx.enqueue_create_host_buffer[DType.float32](n * n)
    var hv = ctx.enqueue_create_host_buffer[DType.float32](n * n)
    var hi = ctx.enqueue_create_host_buffer[DType.float32](3)
    ctx.enqueue_copy(dst_ptr=ha.unsafe_ptr(), src_buf=a_buf)
    ctx.enqueue_copy(dst_ptr=hv.unsafe_ptr(), src_buf=v_buf)
    ctx.enqueue_copy(dst_ptr=hi.unsafe_ptr(), src_buf=i_buf)
    ctx.synchronize()

    a_out.clear()
    v_out.clear()
    for i in range(n * n):
        a_out.append(ha.unsafe_ptr().unsafe_load(i))
        v_out.append(hv.unsafe_ptr().unsafe_load(i))
    var info = List[Float32]()
    for i in range(3):
        info.append(hi.unsafe_ptr().unsafe_load(i))
    return info^


def check_jacobi_fold_width_is_pinned() raises:
    """The STRUCTURAL half, and it needs no GPU.

    `JACOBI_TPB` is `lib_block_size_for[K_LIB_JACOBI_EIGH, TARGET_COLUMN]`,
    which the matrix calls SCHEDULING. In this kernel it is not: it is the
    WIDTH of the fold that decides the sweep count AND the stride that cuts
    the matrix into per-thread partials. Two columns carrying two numbers
    would be two summation orders, and `lib_block_bounds_a_float_fold` does
    NOT list this kernel -- deliberately, it says so, on the grounds that
    the fix belongs at the fold. The fold is fixed; this gates the residue,
    which is that the row must stay flat.

    It also pins the power-of-two requirement `pinned_block_sum`'s contract
    states, because a fold width of 48 would fold a slot nobody wrote.
    """
    var apple = lib_block_size_for[K_LIB_JACOBI_EIGH, COLUMN_APPLE]()
    var nvidia = lib_block_size_for[K_LIB_JACOBI_EIGH, COLUMN_NVIDIA]()
    var amd = lib_block_size_for[K_LIB_JACOBI_EIGH, COLUMN_AMD]()
    var floor = lib_block_size_for[K_LIB_JACOBI_EIGH, COLUMN_BIT_IDENTICAL]()

    if apple != floor or nvidia != floor or amd != floor:
        raise Error(
            "check_jacobi_fold_width_is_pinned: the eigensolver's fold width"
            " resolves to "
            + String(apple)
            + "/"
            + String(nvidia)
            + "/"
            + String(amd)
            + " on Apple/NVIDIA/AMD against "
            + String(floor)
            + " on the identity floor. That width is a SUMMATION ORDER and"
            " the stride of the per-thread partials, so a vendor number on"
            " this row splits the sweep count between two machines. Either"
            " put K_LIB_JACOBI_EIGH in lib_block_bounds_a_float_fold or"
            " leave the row flat."
        )
    if JACOBI_TPB != floor:
        raise Error(
            "check_jacobi_fold_width_is_pinned: JACOBI_TPB is "
            + String(JACOBI_TPB)
            + " but the matrix row reads "
            + String(floor)
            + " -- the kernel and the table disagree about the fold width"
        )
    if JACOBI_TPB <= 0 or (JACOBI_TPB & (JACOBI_TPB - 1)) != 0:
        raise Error(
            "check_jacobi_fold_width_is_pinned: JACOBI_TPB = "
            + String(JACOBI_TPB)
            + " is not a power of two, which pinned_block_sum's contract"
            " requires: the halving tree would fold a slot no thread wrote"
        )
    print(
        "check_jacobi_fold_width_is_pinned OK ("
        + _mode_name()
        + "): every column resolves K_LIB_JACOBI_EIGH to "
        + String(floor)
        + ", a power of two, and the kernel compiles against that same"
        " number. NOTE: "
        + String(floor)
        + " threads is HALF an AMD CDNA wavefront -- under IDENTICAL the"
        " halving tree touches only the "
        + String(floor)
        + " slots real threads wrote, so that is a scheduling question"
        " there and not a numeric one."
    )


def jacobi_fold_probe_kernel(
    out_pinned: MutPointer[Float32, MutAnyOrigin],
    out_library: MutPointer[Float32, MutAnyOrigin],
    a: MutPointer[Float32, MutAnyOrigin],
):
    """Both folds, side by side, ON THE DEVICE, at the solver's width.

    One value per thread and no grid-stride loop: this probe is about the
    fold's SHAPE, so nothing is summed before the fold gets it. The library
    arm is carried alongside because on a 32-lane machine the two shapes can
    COINCIDE, and a check that cannot say so would be claiming a difference
    it never measured.
    """
    var tid = Int(thread_idx.x)
    var v = a.unsafe_load(tid)
    var p = pinned_block_sum[_FOLD_PROBE_TPB](v)
    # THE LIBRARY ARM DOES NOT EXIST AT EVERY WIDTH, AND THAT IS ITSELF A
    # FINDING (DEVIATION 528). `max.gpu.primitives.block` carries
    # `constraint failed: Block size must be a greater than warp size`, so at
    # this solver's 32-thread block it REFUSES TO COMPILE on a 64-lane
    # column -- measured on an MI325X where `build_estimators.sh` failed for
    # exactly this reason. Compiling it unconditionally would make this check
    # unbuildable on the column it most needs to run on.
    #
    # So the arm is comptime-gated on the column's own lane width and the
    # sentinel says WHICH case ran. It is a REPORT, not a skip: the check
    # below prints "not instantiable at this width" rather than silently
    # comparing one number against itself.
    var l = _LIBRARY_ARM_ABSENT
    # `>=`, NOT `>`, AND THE DIFFERENCE IS MEASURED RATHER THAN DERIVED.
    # MAX's constraint text reads "Block size must be a greater than warp
    # size", which would elide this arm on Apple too (32 is not > 32) -- but
    # it demonstrably COMPILES AND RUNS at 32/32 on the Apple column and has
    # since this check was written, so the constraint is not the whole story
    # of when the primitive is instantiable. `>=` is the rule that matches
    # what both columns actually do: keep the arm where it is known to build
    # (Apple, NVIDIA, 32 lanes) and drop it where it is known not to (AMD,
    # 64). If a future column instantiates at 32 on a 64-lane part, this
    # under-reports rather than failing to build, which is the safe side.
    comptime if _FOLD_PROBE_TPB >= lib_lane_width_for[TARGET_COLUMN]():
        l = block_sum[block_size=_FOLD_PROBE_TPB, broadcast=True](v)
    if tid == 0:
        out_pinned.unsafe_store(0, p)
        out_library.unsafe_store(0, l)


def _host_halving_fold(values: List[Float32]) -> Float32:
    """`pinned_block_sum`'s IDENTICAL arm, on the host, in Float32.

    `red[t] += red[t + step]` for `step = n/2 ... 1`. Written out rather
    than imported, because the property under test is that the DEVICE takes
    this exact sequence of additions; importing the thing under test would
    make the check agree with itself.
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


def _fold_fixture() -> List[Float32]:
    """Values whose ADDITION ORDER matters, BUILT TO SEPARATE rather than
    hashed and hoped for.

    THIS IS THE SECOND FIXTURE. The first was 32 hashed full-mantissa values
    spread over three orders of magnitude -- the k-means lane's generator,
    which separates a halving fold from a sequential one at 128 values --
    and at 32 values it did not separate them at all: the check refused
    itself on its own first run. That is the fixture-cannot-fail hole
    (`[[uniform-test-data-hides-permutation]]`) caught by the guard that
    exists to catch it, and the lesson is that a fixture is a function of
    the WIDTH it is folded at.

    So this one is constructed. `v[0]` is 2^24, where the float spacing is
    2.0, and the other 31 are near 1.0 -- each individually LOST when added
    to `v[0]`, and jointly worth 30 when they are added to each other
    first. Sequential loses almost all of them; the halving tree pairs them
    up before they ever meet the big one. Every small value still carries a
    distinct hashed fraction, so a fold that dropped or double-counted one
    slot moves a number no other slot could have produced.
    """
    var out = List[Float32]()
    for i in range(_FOLD_PROBE_TPB):
        if i == 0:
            out.append(Float32(16777216.0))  # 2^24, spacing 2.0
        else:
            var z = UInt64(i + 1) * 0x9E3779B97F4A7C15
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
            z = z ^ (z >> 27)
            var frac = Float32(Int(z & UInt64(0xFFF))) / Float32(4096.0)
            out.append(Float32(1.0) + frac)
    return out^


def check_jacobi_fold_shape() raises:
    """IDENTITY_PATHS row 31, the fold. The device must take the halving
    tree, bit for bit, at the width the eigensolver actually folds at.

    `max.gpu.primitives.block.sum` folds its cross-lane stage at the
    HARDWARE width, so the same values reduce through two different
    association trees on a 32-lane and a 64-lane machine. Under IDENTICAL
    the fold is `pinned_block_sum`'s halving tree, which has no lane
    primitive in it and therefore cannot consult the hardware.

    THE HONEST PART: at 32 threads on a 32-lane machine the library's shape
    and the halving tree may be the SAME sequence of additions, and this
    check reports which it measured instead of implying a difference it
    cannot see. The property that matters is not "the bits moved on Apple",
    it is "the bits are a function of the value vector alone".
    """
    var ctx = DeviceContext()
    var values = _fold_fixture()
    var halving = _host_halving_fold(values)
    var sequential = _host_sequential_fold(values)
    if halving == sequential:
        raise Error(
            "check_jacobi_fold_shape: the fixture cannot tell two fold"
            " shapes apart (halving == sequential), so it could not detect"
            " a fold that ignored the pin. Fix the fixture, not the check."
        )

    var a = ctx.enqueue_create_buffer[DType.float32](_FOLD_PROBE_TPB)
    var op = ctx.enqueue_create_buffer[DType.float32](1)
    var ol = ctx.enqueue_create_buffer[DType.float32](1)
    var ha = ctx.enqueue_create_host_buffer[DType.float32](_FOLD_PROBE_TPB)
    var hp = ctx.enqueue_create_host_buffer[DType.float32](1)
    var hl = ctx.enqueue_create_host_buffer[DType.float32](1)
    ctx.synchronize()
    for i in range(_FOLD_PROBE_TPB):
        ha.unsafe_ptr().unsafe_store(i, values[i])
    ctx.enqueue_copy(dst_buf=a, src_ptr=ha.unsafe_ptr())
    ctx.synchronize()
    ctx.enqueue_function[jacobi_fold_probe_kernel](
        op.unsafe_ptr(),
        ol.unsafe_ptr(),
        a.unsafe_ptr(),
        grid_dim=(1, 1, 1),
        block_dim=(_FOLD_PROBE_TPB, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_copy(dst_ptr=hp.unsafe_ptr(), src_buf=op)
    ctx.enqueue_copy(dst_ptr=hl.unsafe_ptr(), src_buf=ol)
    ctx.synchronize()
    var pinned = hp.unsafe_ptr().unsafe_load(0)
    var library = hl.unsafe_ptr().unsafe_load(0)

    comptime if IDENTICAL_BUILD:
        if pinned != halving:
            raise Error(
                "check_jacobi_fold_shape: IDENTICAL did not take the"
                " halving tree at width "
                + String(_FOLD_PROBE_TPB)
                + ": device "
                + String(pinned)
                + " (bits "
                + String(_f32_bits(pinned))
                + ") vs host halving "
                + String(halving)
                + " (bits "
                + String(_f32_bits(halving))
                + "); a sequential fold would be "
                + String(sequential)
                + ". The pin is not reached."
            )
        print(
            "check_jacobi_fold_shape OK (IDENTICAL): device fold == host"
            " halving tree bit for bit at width",
            _FOLD_PROBE_TPB,
            "(bits",
            _f32_bits(pinned),
            "), and the fixture separates that shape from a sequential"
            " fold (",
            halving,
            "vs",
            sequential,
            "). The library fold on THIS device",
            (
                "was NOT INSTANTIABLE at this width (block size must exceed"
                " the lane width; DEVIATION 528), so no comparison was taken"
            ) if library == _LIBRARY_ARM_ABSENT else (
                "returned " + String(library) + (
                    " -- same bits" if library == pinned
                    else " -- DIFFERENT bits"
                )
            ),
        )
    else:
        if library == _LIBRARY_ARM_ABSENT:
            # A FAST build on a 64-lane column is a compile error by design
            # (E1_RUNBOOK's preconditions, the hist-2 LANE_WIDTH asserts), so
            # this branch should be unreachable there. Reported rather than
            # assumed, because "should be unreachable" is how the last three
            # findings in this file started.
            print(
                "check_jacobi_fold_shape OK (FAST): the library fold was NOT"
                " INSTANTIABLE at width",
                _FOLD_PROBE_TPB,
                "on this column (DEVIATION 528), so this run says nothing"
                " about whether pinned_block_sum's FAST arm equals it. Host"
                " halving",
                halving,
                "and sequential",
                sequential,
                "still differ, so the IDENTICAL arm's equality has teeth.",
            )
        else:
            if pinned != library:
                raise Error(
                    "check_jacobi_fold_shape (FAST): pinned_block_sum's FAST"
                    " arm is supposed to BE the library call, and it is not:"
                    " device pinned "
                    + String(pinned)
                    + " vs library "
                    + String(library)
                    + ". Under FAST the shipped bits must not move."
                )
            print(
                "check_jacobi_fold_shape OK (FAST): the library fold ran and"
                " pinned_block_sum is it, bit for bit (",
                pinned,
                "==",
                library,
                "). Host halving",
                halving,
                "and sequential",
                sequential,
                "differ, so the IDENTICAL arm's equality has teeth.",
            )


# ------------------------------------------------------------------------
# THE HOST REPLAY. Every line below mirrors `jacobi_eigh_device.mojo` and
# NONE of it imports from there, on purpose: a model that calls the thing it
# models proves only that the call happened.
# ------------------------------------------------------------------------


def _model_rot_sub(c: Float32, x: Float32, s: Float32, y: Float32) -> Float32:
    """The kernel's `_rot_sub`, transcribed: fuse the first product, round
    and flush the second, flush the result."""
    comptime if IDENTICAL_BUILD:
        return ftz(identical_mul_add(c, x, -ftz(s * y)))
    return c * x - s * y


def _model_rot_add(s: Float32, x: Float32, c: Float32, y: Float32) -> Float32:
    comptime if IDENTICAL_BUILD:
        return ftz(identical_mul_add(s, x, ftz(c * y)))
    return s * x + c * y


def _model_fold(values: List[Float32], sequential: Bool) -> Float32:
    """The fold the model uses. `sequential` is the OTHER shape -- not a
    vendor's, just a different association of the same multiset -- and it is
    here so that one function can answer "what would this solver have done
    if its fold had combined the partials in a different order", which is
    exactly the question a 64-lane wavefront asks."""
    if sequential:
        return _host_sequential_fold(values)
    return _host_halving_fold(values)


def _model_jacobi_folded(
    a_in: List[Float32],
    n: Int,
    max_sweeps: Int,
    tol: Float32,
    sequential_fold: Bool,
    mut a_out: List[Float32],
    mut v_out: List[Float32],
) -> Int:
    """The whole device kernel, on the host, in Float32. Returns the sweeps
    executed; `converged` is `executed < max_sweeps` by the same argument
    the kernel's counter makes.

    The PARTIALS are cut the way the kernel cuts them -- thread `t` walks
    `t, t + TPB, t + 2*TPB, ...` -- because the partition is part of the
    summation order, not an implementation detail of the launch.
    """
    a_out.clear()
    v_out.clear()
    for i in range(n * n):
        a_out.append(a_in[i])
    for r in range(n):
        for c in range(n):
            v_out.append(Float32(1.0) if r == c else Float32(0.0))

    var fpart = List[Float32]()
    for t in range(JACOBI_TPB):
        var lf = Float32(0.0)
        var fe = t
        while fe < n * n:
            var fv = ftz(a_out[fe])
            lf = ftz(identical_mul_add(fv, fv, lf))
            fe += JACOBI_TPB
        fpart.append(lf)
    var fro2 = _model_fold(fpart, sequential_fold)
    var limit = ftz(ftz(tol * tol) * fro2)

    var executed = 0
    for _sweep in range(max_sweeps):
        var opart = List[Float32]()
        for t in range(JACOBI_TPB):
            var lo = Float32(0.0)
            var e = t
            while e < n * n:
                var i = e // n
                var j = e - i * n
                if j > i:
                    var av = ftz(a_out[e])
                    lo = ftz(identical_mul_add(av, av, lo))
                e += JACOBI_TPB
            opart.append(lo)
        var off = _model_fold(opart, sequential_fold)
        if Float32(2.0) * off <= limit:
            break
        executed += 1

        for p in range(n):
            for q in range(p + 1, n):
                var c = Float32(1.0)
                var s = Float32(0.0)
                var apq = ftz(a_out[p * n + q])
                if apq != Float32(0.0):
                    var aqq = ftz(a_out[q * n + q])
                    var app = ftz(a_out[p * n + p])
                    var theta = ftz(
                        ftz(aqq - app) / ftz(Float32(2.0) * apq)
                    )
                    var root = ftz(
                        identical_sqrt(
                            ftz(identical_mul_add(theta, theta, Float32(1.0)))
                        )
                    )
                    var t = Float32(0.0)
                    if theta >= Float32(0.0):
                        t = ftz(Float32(1.0) / ftz(theta + root))
                    else:
                        t = ftz(Float32(-1.0) / ftz(root - theta))
                    var croot = ftz(
                        identical_sqrt(
                            ftz(identical_mul_add(t, t, Float32(1.0)))
                        )
                    )
                    c = ftz(Float32(1.0) / croot)
                    s = ftz(t * c)

                for k in range(n):
                    var akp = ftz(a_out[k * n + p])
                    var akq = ftz(a_out[k * n + q])
                    a_out[k * n + p] = _model_rot_sub(c, akp, s, akq)
                    a_out[k * n + q] = _model_rot_add(s, akp, c, akq)
                for k in range(n):
                    var apk = ftz(a_out[p * n + k])
                    var aqk = ftz(a_out[q * n + k])
                    a_out[p * n + k] = _model_rot_sub(c, apk, s, aqk)
                    a_out[q * n + k] = _model_rot_add(s, apk, c, aqk)
                for k in range(n):
                    var vkp = ftz(v_out[k * n + p])
                    var vkq = ftz(v_out[k * n + q])
                    v_out[k * n + p] = _model_rot_sub(c, vkp, s, vkq)
                    v_out[k * n + q] = _model_rot_add(s, vkp, c, vkq)
    return executed


def _model_jacobi(
    a_in: List[Float32],
    n: Int,
    max_sweeps: Int,
    tol: Float32,
    mut a_out: List[Float32],
    mut v_out: List[Float32],
) -> Int:
    """The model with the SHIPPED fold shape, which is the halving tree."""
    return _model_jacobi_folded(
        a_in, n, max_sweeps, tol, False, a_out, v_out
    )


def check_jacobi_is_a_pure_function_of_its_input() raises:
    """THE CHECK WITH TEETH. Under IDENTICAL the device answer must equal a
    host replay BIT FOR BIT, at every cell of both outputs.

    This is the only assertion in this file that can see a seam. Every
    other check compares against a Float64 oracle at a tolerance, which is
    exactly the instrument that cannot tell a pinned `fma(c, x, -fl(s*y))`
    from an unpinned `c*x - s*y`: the two differ in the last bit and the
    tolerance is 5e-5. So this one carries a model of the ARITHMETIC, not
    of the answer, and requires equality.

    What it therefore proves, on this one device: every fold, partial,
    rotation and update in the kernel takes the spelling written in the
    model, and no codegen rewrote one of them. What it CANNOT prove is that
    another vendor's codegen makes the same choice -- only E1 can, and the
    reason this check is worth running before E1 is that it turns "the pins
    are in the source" into "the pins are what ran".

    UNDER FAST IT ONLY REPORTS. FAST's arms are the naive chains and both
    the host and the device compiler are free to contract them, differently
    and per expression, so requiring equality there would be asserting a
    property nobody claims (`numerics.mojo`: FMA contraction is a codegen
    decision). The number is printed because it is the size of what the
    pins buy.

    n = 33, deliberately: 1089 elements over 32 threads is 34 strides plus
    one, so the partials are UNEQUAL in length and a model that assumed a
    tidy division would show up here.
    """
    var ctx = DeviceContext()
    var n = 33
    var a64 = _make_symmetric(n, 19, -1, 0.0)
    var a = List[Float32]()
    for i in range(n * n):
        a.append(Float32(a64[i]))

    var da = List[Float32]()
    var dv = List[Float32]()
    var info = _run_device_f32(ctx, a, n, 40, Float32(1.0e-7), da, dv)

    var ma = List[Float32]()
    var mv = List[Float32]()
    var executed = _model_jacobi(a, n, 40, Float32(1.0e-7), ma, mv)

    if Int(info[2]) != executed:
        raise Error(
            "check_jacobi_is_a_pure_function_of_its_input: the device"
            " executed "
            + String(Int(info[2]))
            + " sweeps and the host model "
            + String(executed)
            + ". A sweep count is decided by the folded `off`, so the two"
            " disagree about the FOLD before they disagree about anything"
            " else."
        )

    var worst_a = 0
    var worst_v = 0
    var worst_a_val = Float64(0.0)
    var worst_v_val = Float64(0.0)
    var first_bad = -1
    for i in range(n * n):
        if _f32_bits(da[i]) != _f32_bits(ma[i]):
            worst_a += 1
            if first_bad < 0:
                first_bad = i
            var d = abs(Float64(da[i]) - Float64(ma[i]))
            if d > worst_a_val:
                worst_a_val = d
        if _f32_bits(dv[i]) != _f32_bits(mv[i]):
            worst_v += 1
            var d2 = abs(Float64(dv[i]) - Float64(mv[i]))
            if d2 > worst_v_val:
                worst_v_val = d2

    comptime if IDENTICAL_BUILD:
        if worst_a != 0 or worst_v != 0:
            var where = String("")
            if first_bad >= 0:
                where = (
                    " First cell that differs is A["
                    + String(first_bad // n)
                    + "]["
                    + String(first_bad % n)
                    + "]: device bits "
                    + String(_f32_bits(da[first_bad]))
                    + " vs model bits "
                    + String(_f32_bits(ma[first_bad]))
                    + "."
                )
            raise Error(
                "check_jacobi_is_a_pure_function_of_its_input (IDENTICAL):"
                " the device does NOT take the pinned arithmetic. "
                + String(worst_a)
                + " of "
                + String(n * n)
                + " matrix cells and "
                + String(worst_v)
                + " of "
                + String(n * n)
                + " eigenvector cells differ from the host replay (worst"
                " |delta| "
                + String(worst_a_val)
                + " and "
                + String(worst_v_val)
                + ")."
                + where
                + " Some seam in the kernel is spelled differently from the"
                " model: a fold, a partial, the rotation, or one of the"
                " three update loops."
            )
        print(
            "check_jacobi_is_a_pure_function_of_its_input OK (IDENTICAL):"
            " n =",
            n,
            "-- all",
            2 * n * n,
            "output cells bit-identical to a host Float32 replay of the"
            " same partials, the same halving fold and the same pinned"
            " rotation, in",
            executed,
            "sweeps. Every pin in the kernel is REACHED.",
        )
    else:
        print(
            "check_jacobi_is_a_pure_function_of_its_input OK (FAST): n =",
            n,
            "--",
            worst_a,
            "matrix cells and",
            worst_v,
            "eigenvector cells differ from the host replay (worst |delta|",
            worst_a_val,
            "/",
            worst_v_val,
            ") in",
            executed,
            "sweeps. FAST does not claim equality here: both compilers may"
            " contract the naive chains, differently and per expression."
            " This is the size of what IDENTICAL buys.",
        )


def check_jacobi_sweep_count_is_a_knife_edge() raises:
    """WHAT A LAST BIT COSTS, measured, in both modes.

    The kernel comment used to say that a last-bit difference in the folded
    `off` "can only move which sweep a knife-edge matrix stops on, never the
    answer it stops at". This finds the knife edge and prices it.

    `off <= limit` is monotone in `tol` -- a larger tolerance never stops
    later -- so a BINARY SEARCH OVER THE BITS OF `tol` lands on two
    ADJACENT Float32 values that produce different sweep counts. One ulp of
    the exit quantity is exactly what an AMD 64-wide fold and an Apple
    32-wide fold of the same multiset differ by, and this is what it does to
    the answer.

    It asserts the CONSEQUENCE is not a last bit, because that is the claim
    the deleted comment made. If a future change made the two sides agree to
    1e-7 this check would fail and the comment would deserve to come back.
    """
    var ctx = DeviceContext()
    var n = 64
    var a64 = _make_symmetric(n, 23, -1, 0.0)
    var a = List[Float32]()
    for i in range(n * n):
        a.append(Float32(a64[i]))

    var scratch_a = List[Float32]()
    var scratch_v = List[Float32]()

    var lo = _f32_bits(Float32(1.0e-12))
    var hi = _f32_bits(Float32(1.0e-2))
    var i0 = _run_device_f32(ctx, a, n, 60, _bits_f32(lo), scratch_a, scratch_v)
    var s_lo = Int(i0[2])
    var i1 = _run_device_f32(ctx, a, n, 60, _bits_f32(hi), scratch_a, scratch_v)
    var s_hi = Int(i1[2])
    if s_lo == s_hi:
        raise Error(
            "check_jacobi_sweep_count_is_a_knife_edge: tol 1e-12 and 1e-2"
            " both take "
            + String(s_lo)
            + " sweeps, so there is no boundary to find between them and"
            " the fixture cannot show what a boundary costs"
        )

    var steps = 0
    while hi - lo > UInt32(1):
        steps += 1
        var mid = lo + (hi - lo) // UInt32(2)
        var im = _run_device_f32(
            ctx, a, n, 60, _bits_f32(mid), scratch_a, scratch_v
        )
        if Int(im[2]) == s_lo:
            lo = mid
        else:
            hi = mid

    var alo = List[Float32]()
    var vlo = List[Float32]()
    var ahi = List[Float32]()
    var vhi = List[Float32]()
    var ilo = _run_device_f32(ctx, a, n, 60, _bits_f32(lo), alo, vlo)
    var ihi = _run_device_f32(ctx, a, n, 60, _bits_f32(hi), ahi, vhi)
    if Int(ilo[2]) == Int(ihi[2]):
        raise Error(
            "check_jacobi_sweep_count_is_a_knife_edge: the search ended on"
            " two adjacent tolerances that take the same "
            + String(Int(ilo[2]))
            + " sweeps, so it did not find the boundary"
        )
    if hi - lo != UInt32(1):
        raise Error(
            "check_jacobi_sweep_count_is_a_knife_edge: the bracket is "
            + String(hi - lo)
            + " ulps wide, not 1"
        )

    var worst_v = Float64(0.0)
    var worst_l = Float64(0.0)
    var moved = 0
    for i in range(n * n):
        var d = abs(Float64(vlo[i]) - Float64(vhi[i]))
        if d > worst_v:
            worst_v = d
        if _f32_bits(vlo[i]) != _f32_bits(vhi[i]):
            moved += 1
    for i in range(n):
        var dl = abs(Float64(alo[i * n + i]) - Float64(ahi[i * n + i]))
        if dl > worst_l:
            worst_l = dl

    if worst_v < 1.0e-6:
        raise Error(
            "check_jacobi_sweep_count_is_a_knife_edge: one ulp of tol moved"
            " the sweep count from "
            + String(Int(ilo[2]))
            + " to "
            + String(Int(ihi[2]))
            + " but moved the eigenvectors by only "
            + String(worst_v)
            + ", which is last-bit noise. If that is really true the"
            " deleted comment was right and this check should be deleted"
            " with it -- but check the fixture first."
        )

    # And the OTHER half of the claim: at a fixed tolerance the answer is
    # the same bits every time. A knife edge that also moved run to run
    # would be a different defect wearing this one's clothes.
    var a2 = List[Float32]()
    var v2 = List[Float32]()
    var i2 = _run_device_f32(ctx, a, n, 60, _bits_f32(lo), a2, v2)
    for i in range(n * n):
        if _f32_bits(a2[i]) != _f32_bits(alo[i]) or _f32_bits(
            v2[i]
        ) != _f32_bits(vlo[i]):
            raise Error(
                "check_jacobi_sweep_count_is_a_knife_edge: two runs of the"
                " SAME input at the same tolerance disagree at cell "
                + String(i)
                + ". The solver is not a pure function of its input on one"
                " device, which is a race and not a rounding."
            )
    if Int(i2[2]) != Int(ilo[2]):
        raise Error(
            "check_jacobi_sweep_count_is_a_knife_edge: two runs of the same"
            " input executed "
            + String(Int(ilo[2]))
            + " and "
            + String(Int(i2[2]))
            + " sweeps"
        )

    print(
        "check_jacobi_sweep_count_is_a_knife_edge OK ("
        + _mode_name()
        + "): n = "
        + String(n)
        + ", tol bits "
        + String(lo)
        + " -> "
        + String(hi)
        + " (ONE ulp, "
        + String(_bits_f32(lo))
        + " -> "
        + String(_bits_f32(hi))
        + ", found in "
        + String(steps)
        + " probes) moves the executed sweeps "
        + String(Int(ilo[2]))
        + " -> "
        + String(Int(ihi[2]))
        + " and with them "
        + String(moved)
        + " of "
        + String(n * n)
        + " eigenvector cells, worst |delta| "
        + String(worst_v)
        + ", worst eigenvalue |delta| "
        + String(worst_l)
        + ". A last-bit difference in the exit quantity is NOT a last-bit"
        " difference in the answer. Repeat runs at a fixed tolerance are"
        " bit-identical."
    )


def check_jacobi_denormal_exit_test() raises:
    """The float compares that decide a BRANCH, and the one denormal that
    reaches a DISCRETE consequence.

    Three compares in the kernel branch on a float: `apq == 0` selects the
    identity rotation, `theta >= 0` picks which tan formula runs, and
    `2*off <= limit` ends the sweep loop. On normals all three are one
    answer everywhere -- IEEE compares are exact. On DENORMALS they are
    not, because Metal flushes and a CUDA default build does not.

    TWO OF THE THREE ARE INERT AND THIS CHECK SAYS SO RATHER THAN CLAIMING
    OTHERWISE. A denormal `apq` cannot move the answer: the else arm's
    rotation converges to the identity as `apq -> 0` (`theta` blows up,
    `t -> 1/(2*theta)`, `s -> 0`), so flushed or not the same rotation is
    applied to within 1e-38. `theta >= 0` likewise only splits two formulas
    that agree in the limit. The flush is applied at both anyway, because
    "bounded by 1e-38" is a bound on THIS expression and not a property of
    the next edit.

    THE THIRD IS NOT INERT, and this fixture is built to make it fire.
    `limit` is `tol^2 * ||A||_F^2` -- a product of normals that lands in the
    denormal band for a whole range of ordinary matrix scales -- and `off`
    is a sum of SQUARES, which underflow one at a time and can therefore be
    zero under a flush and normal after being summed. Scale the diagonal to
    1e-13 and the off-diagonal to 3e-20 and the two backends do not disagree
    in the last bit, they disagree about WHETHER THE MATRIX HAS ALREADY
    CONVERGED:

        flushing        every square flushes, `off` is 0, `limit` is 0,
                        `0 <= 0` holds, 0 sweeps, the matrix comes back
                        untouched
        denormal-honoring
                        each square is a denormal, their SUM is a normal
                        number bigger than `limit`, the loop runs its whole
                        budget and rotates

    On this Apple box only the first arm is observable, so the check
    ASSERTS the device took it and COMPUTES the second on the host, whose
    Float32 honors denormals -- and refuses to pass unless the host arm
    really does disagree, because a fixture on which both backends agree
    would prove nothing. Under IDENTICAL the explicit `ftz` at these seams
    is what makes a CUDA build take the device's arm; on that column this
    check stops being a report and becomes the gate.
    """
    var ctx = DeviceContext()
    var n = 8

    # Diagonal ~1e-13 so ||A||_F^2 is a NORMAL number small enough that
    # tol^2 * ||A||_F^2 lands in the denormal band; off-diagonal ~3e-20 so
    # each SQUARE is denormal but their sum is not.
    var a = List[Float32]()
    for i in range(n):
        for j in range(n):
            var lo = i if i < j else j
            var hi2 = j if i < j else i
            var z = UInt64(lo + 1) * 0x9E3779B97F4A7C15 + UInt64(
                hi2 + 1
            ) * 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 30)) * 0x94D049BB133111EB
            z = z ^ (z >> 31)
            var frac = Float32(Int(z & UInt64(0xFFF))) / Float32(4096.0)
            if i == j:
                a.append((Float32(1.0) + frac) * Float32(1.0e-13))
            else:
                a.append((Float32(1.0) + frac) * Float32(3.0e-20))
    var tol = Float32(1.0e-7)

    # The host's arithmetic on the same numbers, with NO flush anywhere:
    # what a denormal-honoring backend computes.
    var host_fro2 = Float32(0.0)
    for i in range(n * n):
        host_fro2 = host_fro2 + a[i] * a[i]
    var host_limit = tol * tol * host_fro2
    var host_off = Float32(0.0)
    var host_denormal_squares = 0
    for i in range(n):
        for j in range(i + 1, n):
            var sq = a[i * n + j] * a[i * n + j]
            host_off = host_off + sq
            if sq != Float32(0.0) and abs(sq) < Float32(
                1.1754943508222875e-38
            ):
                host_denormal_squares += 1

    if host_denormal_squares == 0:
        raise Error(
            "check_jacobi_denormal_exit_test: this HOST flushes denormals"
            " too, so it cannot stand in for a denormal-honoring backend"
            " and the fixture proves nothing here. The check needs a"
            " reference that keeps them."
        )
    if host_limit <= Float32(0.0) or host_limit >= Float32(
        1.1754943508222875e-38
    ):
        raise Error(
            "check_jacobi_denormal_exit_test: `limit` is "
            + String(host_limit)
            + ", which is not in the denormal band, so the fixture no"
            " longer exercises the case it was built for. Rescale the"
            " diagonal."
        )
    if not (Float32(2.0) * host_off > host_limit):
        raise Error(
            "check_jacobi_denormal_exit_test: unflushed, 2*off = "
            + String(Float32(2.0) * host_off)
            + " does NOT exceed limit = "
            + String(host_limit)
            + ", so a denormal-honoring backend would converge at sweep 0"
            " exactly like a flushing one and the fixture cannot separate"
            " them. Rescale the off-diagonal."
        )

    var da = List[Float32]()
    var dv = List[Float32]()
    var info = _run_device_f32(ctx, a, n, 15, tol, da, dv)

    if Int(info[2]) != 0 or info[0] != Float32(1.0):
        var msg = String(
            "check_jacobi_denormal_exit_test: this backend did NOT flush --"
            " it ran "
        ) + String(Int(info[2])) + " sweeps with converged = " + String(
            info[0]
        ) + (
            " where a flushing backend converges at sweep 0. Under"
            " IDENTICAL that is a REACHED-PIN failure: the `ftz` calls on"
            " the partial-sum loads and on `limit` are what make this"
            " column agree with Metal. Under FAST it is the honest"
            " hardware answer and the row-10 divergence, measured."
        )
        comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
            raise Error(msg)
        # FAST on a denormal-honoring column (NVIDIA, AMD): the message
        # above says this is the honest answer, and until 2026-08-23 the
        # check raised anyway, which aborted the FAST pass of
        # check-linalg-identity on every non-Metal leg and with it the
        # IDENTICAL pass the script runs after it. A FAST build does not
        # claim to flush; it is RECORDED here, and the judgement is the
        # IDENTICAL build's.
        print(
            "check_jacobi_denormal_exit_test RECORDED (" + _mode_name()
            + "): " + msg
        )
        return
    var untouched = True
    for i in range(n * n):
        if _f32_bits(da[i]) != _f32_bits(a[i]):
            untouched = False
    var identity = True
    for r in range(n):
        for c in range(n):
            var want = Float32(1.0) if r == c else Float32(0.0)
            if _f32_bits(dv[r * n + c]) != _f32_bits(want):
                identity = False
    if not untouched or not identity:
        raise Error(
            "check_jacobi_denormal_exit_test: the solver reported 0 sweeps"
            " but the matrix or the basis moved anyway (untouched = "
            + String(untouched)
            + ", basis is identity = "
            + String(identity)
            + ")"
        )
    print(
        "check_jacobi_denormal_exit_test OK ("
        + _mode_name()
        + "): the device converged at sweep 0 on an "
        + String(n)
        + " x "
        + String(n)
        + " matrix whose exit test is entirely denormal, and returned it"
        " untouched. The same numbers unflushed on this host give limit = "
        + String(host_limit)
        + " (denormal) and 2*off = "
        + String(Float32(2.0) * host_off)
        + " over "
        + String(host_denormal_squares)
        + " denormal squares, so a denormal-honoring backend would run its"
        " whole 15-sweep budget instead and return a DIFFERENT matrix."
        " That is the discrete consequence row 10's ftz aligns; it is one"
        " compare, not a last bit."
    )


def check_jacobi_reports_the_sweep_cap() raises:
    """The REFUSE move's evidence, and the cost of not making it.

    IDENTITY_PATHS row 25 is the precedent: a loop cut off at a cap returns
    a snapshot rather than a fixed point. Here the refusal is ALREADY WIRED
    and needs no mode gate, because it is at the CALLERS and it fires in
    both modes -- `eig_and_truncate` (`pca.mojo`) and `lstsq_eig`
    (`lstsq.mojo`) both raise on `info_out[0] == 0` rather than return an
    unconverged eigendecomposition. RAFT's own Jacobi arm fetches the sweep
    count and never reads it.

    What this asserts is the half that lives in this file: the kernel
    REPORTS the cap truthfully, and the answer at the cap is genuinely
    different from the converged one, so the refusal is worth making.
    """
    var ctx = DeviceContext()
    var n = 64
    var a64 = _make_symmetric(n, 29, -1, 0.0)
    var a = List[Float32]()
    for i in range(n * n):
        a.append(Float32(a64[i]))

    var ca = List[Float32]()
    var cv = List[Float32]()
    var conv = _run_device_f32(ctx, a, n, 40, Float32(1.0e-7), ca, cv)
    if conv[0] != Float32(1.0):
        raise Error(
            "check_jacobi_reports_the_sweep_cap: the fixture does not"
            " converge in 40 sweeps, so there is nothing to compare a"
            " truncated run against"
        )

    var ta = List[Float32]()
    var tv = List[Float32]()
    var trunc = _run_device_f32(ctx, a, n, 2, Float32(1.0e-7), ta, tv)
    if trunc[0] != Float32(0.0):
        raise Error(
            "check_jacobi_reports_the_sweep_cap: a 2-sweep budget reported"
            " CONVERGED on a matrix that needs "
            + String(Int(conv[2]))
            + " sweeps. The kernel's convergence flag is not the exit"
            " condition it claims to be."
        )
    if Int(trunc[2]) != 2:
        raise Error(
            "check_jacobi_reports_the_sweep_cap: a 2-sweep budget reported "
            + String(Int(trunc[2]))
            + " executed sweeps"
        )

    var worst = Float64(0.0)
    for i in range(n * n):
        var d = abs(Float64(tv[i]) - Float64(cv[i]))
        if d > worst:
            worst = d
    if worst < 1.0e-3:
        raise Error(
            "check_jacobi_reports_the_sweep_cap: the truncated answer is"
            " within "
            + String(worst)
            + " of the converged one, so this fixture cannot show what the"
            " refusal is worth. Use a matrix that needs more sweeps."
        )
    print(
        "check_jacobi_reports_the_sweep_cap OK ("
        + _mode_name()
        + "): the fixture converges in "
        + String(Int(conv[2]))
        + " sweeps; capped at 2 the kernel reports converged = 0, sweeps ="
        " 2 and a relative off-diagonal of "
        + String(trunc[1])
        + ", and its eigenvectors are "
        + String(worst)
        + " away from the converged ones at the worst cell. Both callers"
        " RAISE on that flag in both modes, which is the deviation from"
        " RAFT's Jacobi arm recorded in DEVIATION BLOCK 2."
    )


def _model_boundary_bits(
    a: List[Float32],
    n: Int,
    sequential: Bool,
    lo_in: UInt32,
    hi_in: UInt32,
) -> UInt32:
    """The tol BIT at which this fold shape changes its mind, by bisection.

    The exit test is monotone in `tol`, so the sweep count is a step
    function of the tolerance's bits and the step has one location. That
    location is what a fold shape OWNS: two shapes that agree on every
    partial sum land on the same bit, and two that do not, do not.
    """
    var lo = lo_in
    var hi = hi_in
    var sa = List[Float32]()
    var sv = List[Float32]()
    var s_lo = _model_jacobi_folded(
        a, n, 60, _bits_f32(lo), sequential, sa, sv
    )
    while hi - lo > UInt32(1):
        var mid = lo + (hi - lo) // UInt32(2)
        var sm = _model_jacobi_folded(
            a, n, 60, _bits_f32(mid), sequential, sa, sv
        )
        if sm == s_lo:
            lo = mid
        else:
            hi = mid
    return hi


def check_jacobi_fold_shape_decides_the_sweep_count() raises:
    """DEFECT 1, END TO END: the fold's SHAPE, not its inputs, decides how
    many sweeps run and therefore what comes back.

    The knife-edge check proves one ulp of the exit quantity moves the
    sweep count. This proves the fold shape moves the exit quantity by more
    than that ulp -- which is the step the argument needs and the step a
    single 32-lane box cannot take by running a kernel, because the only
    fold shape it can execute is its own.

    So it takes the step in the MODEL, and the model is not a stand-in: the
    same model is required to equal the device BIT FOR BIT in
    `check_jacobi_is_a_pure_function_of_its_input`, at the same width, with
    the same partials. A difference here is therefore a difference the
    device would have shown if its fold had the other shape, which is what
    a 64-wide CDNA wavefront is.

    THE FIXTURE HAS TO BE SEARCHED FOR AND THAT IS THE HONEST PART. On the
    first matrix tried (n = 64, seed 23) a sequential fold and a halving
    fold chose the SAME boundary bit and the same sweep count: the 32
    partials were sums of a hundred similar squares and every association
    of them rounded alike. That is worth knowing -- it means the hazard is
    not universal -- and it is exactly why this check scans seeds and
    reports which one separated rather than asserting on the first. If NO
    seed separates, the check says so instead of passing quietly.
    """
    var n = 32
    var lo0 = _f32_bits(Float32(1.0e-12))
    var hi0 = _f32_bits(Float32(1.0e-2))
    var found = -1
    var b_h = UInt32(0)
    var b_s = UInt32(0)
    for seed in range(1, 25):
        var a = _spread_fixture(n, seed)
        # `a_for_seed` scales by a function of (row + col), so it is still
        # symmetric -- assert that rather than trust it, because a
        # non-symmetric matrix does not converge and the failure would look
        # like a fold difference.
        for r in range(n):
            for c in range(n):
                if _f32_bits(a[r * n + c]) != _f32_bits(a[c * n + r]):
                    raise Error(
                        "check_jacobi_fold_shape_decides_the_sweep_count:"
                        " the fixture is not symmetric at ("
                        + String(r)
                        + ", "
                        + String(c)
                        + ")"
                    )
        var bh = _model_boundary_bits(a, n, False, lo0, hi0)
        var bs = _model_boundary_bits(a, n, True, lo0, hi0)
        if bh != bs:
            found = seed
            b_h = bh
            b_s = bs
            break

    if found < 0:
        raise Error(
            "check_jacobi_fold_shape_decides_the_sweep_count: no seed in"
            " 1..24 produced two different boundaries for the two fold"
            " shapes. Either the fold shapes agree on every partial this"
            " fixture generator can make -- in which case say so and keep"
            " the pin for the shapes it cannot make -- or the two shapes"
            " are no longer two shapes."
        )

    # At a tolerance BETWEEN the two boundaries the two shapes disagree
    # about the sweep count on the same input. Take the lower boundary,
    # which belongs to whichever shape decided later.
    var probe = b_h if b_h < b_s else b_s
    var ah = List[Float32]()
    var vh = List[Float32]()
    var as_ = List[Float32]()
    var vs = List[Float32]()
    var sh = _model_jacobi_folded(
        _spread_fixture(n, found), n, 60, _bits_f32(probe), False, ah, vh
    )
    var ss = _model_jacobi_folded(
        _spread_fixture(n, found), n, 60, _bits_f32(probe), True, as_, vs
    )
    if sh == ss:
        raise Error(
            "check_jacobi_fold_shape_decides_the_sweep_count: the"
            " boundaries differ ("
            + String(b_h)
            + " vs "
            + String(b_s)
            + ") but both shapes take "
            + String(sh)
            + " sweeps at the probe tolerance, so the probe is on the wrong"
            " side of one of them"
        )
    var worst = Float64(0.0)
    var moved = 0
    for i in range(n * n):
        var d = abs(Float64(vh[i]) - Float64(vs[i]))
        if d > worst:
            worst = d
        if _f32_bits(vh[i]) != _f32_bits(vs[i]):
            moved += 1

    # And the DEVICE agrees with the shipped shape, so the model is not
    # arguing with itself.
    var ctx = DeviceContext()
    var da = List[Float32]()
    var dv = List[Float32]()
    var info = _run_device_f32(
        ctx, _spread_fixture(n, found), n, 60, _bits_f32(probe), da, dv
    )
    comptime if IDENTICAL_BUILD:
        for i in range(n * n):
            if _f32_bits(da[i]) != _f32_bits(ah[i]) or _f32_bits(
                dv[i]
            ) != _f32_bits(vh[i]):
                raise Error(
                    "check_jacobi_fold_shape_decides_the_sweep_count: at"
                    " the probe tolerance the device ran "
                    + String(Int(info[2]))
                    + " sweeps and does not match the halving-fold model ("
                    + String(sh)
                    + " sweeps) at cell "
                    + String(i)
                    + ". The sequential-fold model runs "
                    + String(ss)
                    + " sweeps here, so a fold that is not the halving tree"
                    " is exactly what this looks like."
                )
    print(
        "check_jacobi_fold_shape_decides_the_sweep_count OK ("
        + _mode_name()
        + "): n = "
        + String(n)
        + ", seed "
        + String(found)
        + ". The halving fold changes its mind at tol bits "
        + String(b_h)
        + " and a sequential fold at "
        + String(b_s)
        + ". At tol = "
        + String(_bits_f32(probe))
        + " the two shapes execute "
        + String(sh)
        + " and "
        + String(ss)
        + " sweeps ON THE SAME INPUT, and their eigenvectors then differ in "
        + String(moved)
        + " of "
        + String(n * n)
        + " cells, worst |delta| "
        + String(worst)
        + ". The device (sweeps = "
        + String(Int(info[2]))
        + ") takes the halving shape. THAT is why the fold is not a library"
        " call any more."
    )


def _spread_fixture(n: Int, seed: Int) -> List[Float32]:
    """The spread fixture above, rebuilt. Kept as one function so the check
    cannot compare two different matrices by accident."""
    var a64 = _make_symmetric(n, seed, -1, 0.0)
    var a = List[Float32]()
    for i in range(n * n):
        var row = i // n
        var col = i % n
        var e = (row + col) % 7
        var scale = Float64(1.0)
        for _p in range(e):
            scale *= 16.0
        a.append(Float32(a64[i] * scale))
    return a^


def main() raises:
    print("jacobi_check -- build mode: " + _mode_name())
    check_jacobi_fold_width_is_pinned()
    check_jacobi_fold_shape()
    check_jacobi_device_sizes()
    check_jacobi_reaches_past_32()
    check_jacobi_scale_invariance()
    check_jacobi_is_a_pure_function_of_its_input()
    check_jacobi_sweep_count_is_a_knife_edge()
    check_jacobi_fold_shape_decides_the_sweep_count()
    check_jacobi_denormal_exit_test()
    check_jacobi_reports_the_sweep_cap()
