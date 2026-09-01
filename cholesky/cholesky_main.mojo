# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Cholesky driver: one factorization, the identity card, the mode it ran in.

    MOJOLEARN_IDENTITY_TRACE=/tmp/mac.chol.card \\
        tools/with_build_lock.sh pixi run mojo run -I . cholesky/cholesky_main.mojo
    MOJOLEARN_IDENTITY_TRACE=/tmp/mac.chol.identical.card \\
        tools/with_identical_mode.sh pixi run mojo run -I . cholesky/cholesky_main.mojo

    python3 tools/identity_trace_diff.py /tmp/mac.chol.identical.card /tmp/<other>.chol.identical.card

Environment knobs (all optional): `MOJOLEARN_CHOL_FIXTURE` (`planted`,
`rbf`, `ill`, `singular`, `signed_zero`, `denormal_pivot`; default `rbf`),
`MOJOLEARN_CHOL_JITTER` (`0` for none; default the profile's ridge),
`MOJOLEARN_CHOL_NRHS` (default 4).

THE CARD, and its ORDER is the point rather than its length. Stages, in the
sequence a divergence would first show up in:

    chol.input              the matrix as uploaded, before anything
    chol.jittered           after the ridge (DEVIATION 1637)
    chol.panel000.factored  the diagonal block of panel 0, factored
    chol.panel000.solved    after L21 = A21 . L11^{-T}
    chol.panel000.trailing  after A22 -= L21 L21^T
    chol.panel001.*         ... one triple per panel
    chol.factor             the finished L, upper triangle zeroed
    chol.nb                 (nb, n_panels, info) as Int32 -- the NUMERIC
                            parameter that produced everything above it
    chol.diag               diag(L), through the ported RAFT extractor
    chol.logdet             2 * sum log(diag)
    chol.solve.forward      L y = B
    chol.solve.back         L^T x = y

**A CARD THAT DIVERGES HAS AN ADDRESS AND THE ADDRESS IS THE DIAGNOSIS.**
`chol.jittered` moving means the fixture or the ridge moved, not the
factorization. A `panelNNN.factored` moving with its predecessor identical
means the panel's own arithmetic moved -- `identical_sqrt` (IDENTITY_PATHS
row 10, DEVIATION 258's NVIDIA sqrt), the `fma` pin (row 9) or the flush (row
10). A `panelNNN.trailing` moving with `solved` identical means the GEMM
moved, which is the gemm lane's certificate and not this one's. `chol.nb`
differing means two runs used two different NUMERIC parameters and nothing
below it is comparable at all. `chol.logdet` moving with `chol.diag`
identical is `identical_log` (row 12) and nothing else.

The fixture performs no unaccounted host floating-point operation; see
`cholesky/checks/cholesky_fixture.mojo`'s header for the two
constructions and their arguments. Prints scalars as decimal AND hex,
because `String(Float32)` does not round trip.
"""

from std.memory import bitcast
from std.os import getenv

from core.identity_trace import IdentityTrace
from cholesky.checks.cholesky_fixture import (
    FIX_DENORMAL_PIVOT,
    FIX_ILL,
    FIX_PLANTED,
    FIX_RBF,
    FIX_SIGNED_ZERO,
    FIX_SINGULAR,
    chol_fixture,
    chol_fixture_n,
    chol_fixture_name,
    chol_rhs_fixture,
)
from cholesky.checks.potrf import (
    CHOL_ELEM_TPB,
    CHOL_NB_PINNED,
    CHOL_PANEL_TPB,
    CHOL_PROFILE,
    add_jitter,
    chol_jitter_pinned,
    chol_logdet,
    chol_workspace_floats,
    potrf_lower,
)
from cholesky.checks.trsm import CHOL_SOLVE_TPB, cho_solve
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name

from max.gpu.host import DeviceContext


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29. This used to
    be a local two-way `IDENTICAL`-or-`FAST`, written when there were
    two tiers, and it answered "FAST" for a DETERMINISTIC build -- so
    a driver run under the middle tier printed the wrong arm onto
    every line it produced. A correctly-labelled measurement of the
    wrong arm is the failure this tree has been bitten by repeatedly,
    and forty-four copies of a mode label is how it happens.
    """
    return numeric_mode_name()


def _hex32(v: Float32) -> String:
    comptime DIGITS = "0123456789abcdef"
    var u = bitcast[DType.uint32](v)
    var out = String("0x")
    for i in range(8):
        var nib = Int((u >> UInt32(28 - 4 * i)) & UInt32(0xF))
        out += String(DIGITS[byte=nib])
    return out


def _fixture_from_env() -> Int:
    var s = String(getenv("MOJOLEARN_CHOL_FIXTURE"))
    if s == "planted":
        return FIX_PLANTED
    if s == "ill":
        return FIX_ILL
    if s == "singular":
        return FIX_SINGULAR
    if s == "signed_zero":
        return FIX_SIGNED_ZERO
    if s == "denormal_pivot":
        return FIX_DENORMAL_PIVOT
    return FIX_RBF


def main() raises:
    var which = _fixture_from_env()
    var n = chol_fixture_n(which)
    var jitter = chol_jitter_pinned()
    if String(getenv("MOJOLEARN_CHOL_JITTER")) == "0":
        jitter = Float32(0.0)
    var nrhs = 4
    var nrhs_env = String(getenv("MOJOLEARN_CHOL_NRHS"))
    if nrhs_env != "":
        nrhs = Int(atol(nrhs_env))
    print(
        "== cholesky/cholesky_main.mojo ["
        + _mode_name()
        + "] profile="
        + CHOL_PROFILE
        + " fixture="
        + chol_fixture_name(which)
        + " n="
        + String(n)
        + " nb="
        + String(CHOL_NB_PINNED)
        + " jitter="
        + _hex32(jitter)
        + " nrhs="
        + String(nrhs)
        + " =="
    )

    var a = chol_fixture(which, 0)
    var x_planted = chol_rhs_fixture(n, nrhs, 0)

    var ctx = DeviceContext()
    var trace = IdentityTrace()
    trace.header(
        "cholesky: profile="
        + CHOL_PROFILE
        + " fixture="
        + chol_fixture_name(which)
        + " n="
        + String(n)
        + " nb="
        + String(CHOL_NB_PINNED)
        + " jitter="
        + _hex32(jitter)
        + " mode="
        + _mode_name()
    )

    var da = ctx.enqueue_create_buffer[DType.float32](n * n)
    var ha = ctx.enqueue_create_host_buffer[DType.float32](n * n)
    for i in range(n * n):
        ha.unsafe_ptr().unsafe_store(i, a[i])
    ctx.enqueue_copy(dst_buf=da, src_ptr=ha.unsafe_ptr())
    ctx.synchronize()
    trace.record_device(ctx, "chol.input", da, n * n)

    var ws = ctx.enqueue_create_buffer[DType.float32](
        chol_workspace_floats(n, CHOL_NB_PINNED)
    )
    var dwork = ctx.enqueue_create_buffer[DType.float32](n + 1)
    ctx.synchronize()

    add_jitter(ctx, da, n, jitter, CHOL_ELEM_TPB)
    trace.record_device(ctx, "chol.jittered", da, n * n)
    var run = potrf_lower(
        ctx, da, ws, n, trace, CHOL_NB_PINNED, CHOL_PANEL_TPB, CHOL_ELEM_TPB
    )

    print(
        "  info="
        + String(run.info)
        + "  nb_that_ran="
        + String(run.nb)
        + "  panels="
        + String(run.n_panels)
    )
    if run.info != 0:
        print(
            "  the leading minor of order "
            + String(run.info)
            + " is not positive definite; the factorization stopped at"
            " column "
            + String(run.info - 1)
            + " and the card's last panel stage holds the PARTIAL factor."
            " That is LAPACK's contract (DEVIATION 1634) and not a failure"
            " of this run"
        )
        var trace_path_f = String(getenv("MOJOLEARN_IDENTITY_TRACE"))
        if trace_path_f == "":
            print("  no MOJOLEARN_IDENTITY_TRACE set: no card written")
        else:
            print("  card written to " + trace_path_f)
        _ = da^
        _ = ha^
        _ = ws^
        _ = dwork^
        return

    var logdet = chol_logdet(ctx, da, dwork, n, trace, CHOL_ELEM_TPB)
    print(
        "  logdet = " + String(logdet) + "  " + _hex32(logdet)
    )

    # B = A X for the planted X, formed on the host from the ORIGINAL matrix
    # plus the ridge, so the solve's answer is X back again wherever the
    # arithmetic is exact (the planted fixture) and close elsewhere.
    var aj = a.copy()
    for i in range(n):
        aj[i * n + i] = aj[i * n + i] + jitter
    var b = List[Float32]()
    for i in range(n):
        for j in range(nrhs):
            var acc = Float32(0.0)
            for k in range(n):
                acc = acc + aj[i * n + k] * x_planted[k * nrhs + j]
            b.append(acc)

    var db = ctx.enqueue_create_buffer[DType.float32](n * nrhs)
    var hb = ctx.enqueue_create_host_buffer[DType.float32](n * nrhs)
    for i in range(n * nrhs):
        hb.unsafe_ptr().unsafe_store(i, b[i])
    ctx.enqueue_copy(dst_buf=db, src_ptr=hb.unsafe_ptr())
    ctx.synchronize()
    cho_solve(ctx, da, db, n, nrhs, trace, CHOL_SOLVE_TPB)
    ctx.synchronize()

    var hx = ctx.enqueue_create_host_buffer[DType.float32](n * nrhs)
    ctx.enqueue_copy(dst_ptr=hx.unsafe_ptr(), src_buf=db)
    ctx.synchronize()

    var worst = Float32(0.0)
    var worst_at = 0
    for i in range(n * nrhs):
        var d = hx.unsafe_ptr().unsafe_load(i) - x_planted[i]
        if d < Float32(0.0):
            d = -d
        if d > worst:
            worst = d
            worst_at = i
    print(
        "  solve: worst |x_recovered - x_planted| = "
        + String(worst)
        + "  "
        + _hex32(worst)
        + " at flat index "
        + String(worst_at)
    )
    for i in range(4):
        print(
            "    x["
            + String(i)
            + "] = "
            + String(hx.unsafe_ptr().unsafe_load(i))
            + "  "
            + _hex32(hx.unsafe_ptr().unsafe_load(i))
            + "   planted "
            + _hex32(x_planted[i])
        )

    var trace_path = String(getenv("MOJOLEARN_IDENTITY_TRACE"))
    if trace_path == "":
        print(
            "  no MOJOLEARN_IDENTITY_TRACE set: everything computed, no card"
            " written"
        )
    else:
        print(
            "  card written to "
            + trace_path
            + " ("
            + String(3 * run.n_panels + 7)
            + " stages: input, jittered, "
            + String(run.n_panels)
            + " panels x 3, factor, nb, diag, logdet, solve x 2)"
        )
    _ = da^
    _ = ha^
    _ = db^
    _ = hb^
    _ = hx^
    _ = ws^
    _ = dwork^

    # DEVIATION 1946: the context dies LAST, after every value built on it.
    # Mojo frees at LAST USE, so without this the buffer releases above run
    # against a context that is already gone. On sm_89 the next GPU call in
    # the process then never returns (GPU idle, host threads in futex wait);
    # Apple and AMD do not show it, which is how it stayed latent here.
    _ = ctx^
