# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host-pointer surface for `mojolearn.identical.gemm.fp32.v1`.

DEVIATION 910. Written 2026-08-24 so the profile is reachable from Python.

WHY THIS FILE EXISTS
--------------------
`gemm/original/gemm_identical.mojo::identical_gemm` is the profile's
host-visible entry point, and it is a DEVICE-side API: it takes a
`DeviceContext` and three `DeviceBuffer`s. Every other family in this tree
has an `estimator.mojo` that takes RAW HOST POINTERS, owns its device work
and synchronizes before it returns (`decomposition/estimator.mojo`,
`cluster/estimator.mojo`), and that is the only shape the CPython bindings
know how to call. `gemm/` had no such file until this one. That is the whole
gap this closes, and it is the reason the flagship result of the repository
was not reachable from Python.

WHAT IS AND IS NOT HERE
-----------------------
**There is no arithmetic in this file, and there must never be any.** It
allocates three device buffers, copies host to device, calls the certified
`identical_gemm`, copies device to host, and waits. A multiply or an add
appearing below would be a second implementation of a bit-exact contract,
which is the one thing this lane cannot afford. The bits come from
`gemm/original/gemm_identical.mojo` and from nowhere else.

WHY `identical_gemm` AND NOT `identical_gemm_into`
--------------------------------------------------
`identical_gemm_into` is the ASYNCHRONOUS form and it takes a CALLER-OWNED
workspace. Its own docstring records what that costs a caller who guesses:
*"Sizing it for one plan and letting the dispatcher pick another is an
out-of-bounds write that a small shape will not show you"* -- a one-float
workspace passed to a SPLITK dispatch produced right answers at 64 x 4 and
regions of `+0.0` at 64 x 64. `identical_gemm` sizes its own workspace with
`identical_gemm_workspace_max_floats`, keeps it alive past the wait, and
synchronizes before it returns. It is the form Phase 3 and Phase 4 call. A
host-pointer surface has no reason to be asynchronous -- the caller's next
statement reads the output buffer -- so it takes the form that cannot get
the workspace wrong.

`[[mojo-buffer-freed-at-last-use]]`: a `DeviceBuffer` is dead at its last
use, so the `_ = a` lines at the bottom keep all three alive past the final
`ctx.synchronize()`. `identical_gemm` synchronizes internally as well, so
the operands are already safe by the time it returns; the explicit keeps are
there because the hazard is invisible at review time and free at run time.
"""

from max.gpu.host import DeviceContext

from gemm.original.gemm_identical import identical_gemm
from gemm.original.gemm_oracle import OP_NN, OP_NT, OP_TN


def identical_gemm_host(
    ctx: DeviceContext,
    c_ptr: MutPointer[Float32, MutUntrackedOrigin],
    a_ptr: MutPointer[Float32, MutUntrackedOrigin],
    b_ptr: MutPointer[Float32, MutUntrackedOrigin],
    m: Int,
    n: Int,
    k: Int,
    op: Int,
) raises:
    """`C[m x n] = op(A) . op(B)` from host memory, under the profile.

    Every pointer is the caller's, row-major and fully contiguous (contract
    section 2), float32 (contract section 1). The element COUNTS the three
    orientations require, which is what the buffers below are sized to:

        op       A elements   B elements   C elements
        OP_NN    m * k        k * n        m * n
        OP_NT    m * k        n * k        m * n
        OP_TN    k * m        k * n        m * n

    `m * k` and `n * k` in every row, which is why the sizing here does not
    branch on `op`. The contract's section 0.1 table is the authority on
    which SHAPE those counts are, and the caller is the one that has to get
    that right; a wrong orientation here is a plausible wrong number rather
    than a crash.

    **Degenerate shapes are refused rather than answered.** Contract section
    8 specifies all of them -- `k == 0` writes `+0.0` into every cell,
    `m == 0` or `n == 0` writes nothing, a negative extent is an error --
    and `identical_gemm_with_plan` already implements the first two. They
    are refused at this surface anyway, because no gate in this tree has run
    them THROUGH a host-pointer path, and an answer nothing has checked is
    not something to hand a Python caller. Lifting the refusal is a gate
    plus one edit here, in that order.
    """
    if m <= 0 or n <= 0 or k <= 0:
        raise Error(
            "identical_gemm_host: m, n and k must all be positive, got m="
            + String(m) + " n=" + String(n) + " k=" + String(k)
            + " (contract section 8 specifies the degenerate shapes; this"
            " surface has no gate on them and refuses rather than guesses)"
        )
    if op != OP_NN and op != OP_NT and op != OP_TN:
        raise Error(
            "identical_gemm_host: op must be 0 (OP_NN), 1 (OP_NT) or 2"
            " (OP_TN), got " + String(op)
        )

    var a = ctx.enqueue_create_buffer[DType.float32](m * k)
    var b = ctx.enqueue_create_buffer[DType.float32](n * k)
    var c = ctx.enqueue_create_buffer[DType.float32](m * n)
    ctx.enqueue_copy(dst_buf=a, src_ptr=a_ptr)
    ctx.enqueue_copy(dst_buf=b, src_ptr=b_ptr)
    ctx.synchronize()

    # THE ONE LINE THAT COMPUTES ANYTHING. Everything above is transport and
    # everything below is transport.
    identical_gemm(ctx, c, a, b, m, n, k, op)

    ctx.enqueue_copy(dst_ptr=c_ptr, src_buf=c)
    ctx.synchronize()
    _ = a
    _ = b
    _ = c
