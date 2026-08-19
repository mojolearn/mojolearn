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

from std.math import sqrt
from max.gpu.host import DeviceBuffer, DeviceContext

from decomposition.mojo_only.jacobi_eigh import jacobi_eigh
from decomposition.mojo_only.jacobi_eigh_device import (
    JACOBI_TPB,
    jacobi_eigh_kernel,
)


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
