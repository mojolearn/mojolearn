# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host-pointer surface for `mojolearn.identical.gemm.fp32.v1`.

DEVIATION 910. Written 2026-08-24 so the profile is reachable from Python.

WHY THIS FILE EXISTS
--------------------
`gemm/checks/gemm_identical.mojo::identical_gemm` is the profile's
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
`gemm/checks/gemm_identical.mojo` and from nowhere else.

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

from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from gemm.checks.gemm_identical import (
    identical_gemm,
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from gemm.host_transport import (
    GemmHostLease,
    ROLE_A,
    ROLE_B,
    ROLE_C,
    ROLE_WS,
    gemm_down_f32,
    gemm_up_f32,
)
from gemm.contract import OP_NN, OP_NT, OP_TN
from std.sys.compile import is_defined
from gemm.neural_dispatch import (
    identical_gemm_into as _neural_gemm_into,
    identical_gemm_workspace_max_floats as _neural_workspace_max_floats,
)
from gemm.experiments.neural_ozaki import NEURAL_OZAKI

comptime GEMM_OZAKI_LINALG = is_defined["MOJOLEARN_IDN_GEMM_OZAKI_LINALG"]()
"""lane/neural-small (2026-10-07), IDENTICAL grid control `gemm_ozaki_linalg`.
Absent = main: the linalg GEMM calls `gemm_identical` directly. Defined: it
calls `gemm.neural_dispatch.identical_gemm_into`, so a build that also sets
`-D MOJOLEARN_IDN_NEURAL_GEMM_OZAKI_SLICES=4|5|6` serves the product with
the Ozaki int8 profile (gemm/experiments/neural_ozaki.mojo) the neural GEMMs
already take. Cost reasoning: the incumbent fp32 SIMT body is bound by fp32
FMA throughput, the int8 matrix units (IMMA on NVIDIA, i8 MFMA on AMD) run
S(S+1)/2 int8 products per tile at several times that rate, which holds for
any k under `ozaki_max_k[S]` (a function of S only, not of a board shape);
k above the bound falls back to the incumbent inside the router.
BITS CHANGE: a new profile (integer diagonal sums, one RNE epilogue). Same
words on NVIDIA and AMD by construction (integer sums are order free); the
host column runs the same construction on its integer ALU (the PIECES
kernel), so it changes together with the device columns."""


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

    # lane/gap-neural-models (2026-10-02): pooled device buffers, the
    # dispatcher's workspace pooled with them, and staged transfers
    # (gemm/host_transport.mojo; MOJOLEARN_GEMM_POOL=0 and
    # MOJOLEARN_GEMM_STAGE_DOWN/UP=0 restore the old transport). The kernels
    # and their plan are the same; copies only, no bit moves.
    var lease = GemmHostLease()
    var a = lease.f32(ctx, ROLE_A, m * k)
    var b = lease.f32(ctx, ROLE_B, n * k)
    var c = lease.f32(ctx, ROLE_C, m * n)
    gemm_up_f32(ctx, lease, a, a_ptr, m * k)
    gemm_up_f32(ctx, lease, b, b_ptr, n * k)
    # NO WAIT AFTER THE UPLOADS (lane/apple-mlp-fused, 2026-09-30): the
    # uploads and the GEMM's launches sit on one in-order context.

    # THE ONE LINE THAT COMPUTES ANYTHING. Everything above is transport and
    # everything below is transport.
    comptime if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL:
        comptime if GEMM_OZAKI_LINALG:
            comptime assert NEURAL_OZAKI, (
                "MOJOLEARN_IDN_GEMM_OZAKI_LINALG routes the linalg GEMM to the"
                " Ozaki profile: define MOJOLEARN_IDN_NEURAL_GEMM_OZAKI_SLICES=4|5|6"
            )
            var ws = lease.f32(ctx, ROLE_WS, _neural_workspace_max_floats(m, n, k))
            _neural_gemm_into(ctx, c, a, b, ws, m, n, k, op)
            gemm_down_f32(ctx, lease, c, c_ptr, m * n)
            _ = ws
        else:
            var ws = lease.f32(ctx, ROLE_WS, identical_gemm_workspace_max_floats(m, n, k))
            identical_gemm_into(ctx, c, a, b, ws, m, n, k, op)
            gemm_down_f32(ctx, lease, c, c_ptr, m * n)
            _ = ws
    else:
        identical_gemm(ctx, c, a, b, m, n, k, op)
        gemm_down_f32(ctx, lease, c, c_ptr, m * n)
    lease.release()
    _ = a
    _ = b
    _ = c
