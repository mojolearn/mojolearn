# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""The Samba stack's host-pointer operations and the gradient accumulate on
the host (the training-primitives and optim-adam-clip lanes of
tools/identity_break.py; lane/cpu-training-misc, 2026-09-15):
`training/samba_ops.mojo`'s embedding forward and backward, RMSNorm forward
and backward, the LM head GEMM forward and backward and the balanced-tree
accumulate, spelled on the host.

HOST ONLY. Nothing here imports `max.gpu`, `std.gpu` or a `DeviceContext`.
`samba_ops.mojo` says it adds NO NEW ARITHMETIC: each operation is a device
entry of a profile whose host oracle already exists, so each is restated
here over that oracle, with the validation and refusal words of the device
entry in front of it:

  `host_samba_embedding_forward`   `samba_embedding_forward_host` (`:85`):
                                   `emb_forward_oracle`, THE NORMATIVE
                                   FORWARD of the embedding profile, at
                                   `EmbConfig.llama(vocab, width)`.
  `host_samba_embedding_backward`  `samba_embedding_backward_host` (`:114`):
                                   `emb_backward_oracle`, a fresh gradient.
  `host_samba_rms_norm_forward`    `samba_rms_norm_forward_host` (`:157`),
                                   `llama_rms_norm_kernel`
                                   (`transformer/impl/llama/modeling_llama.
                                   mojo:1374`): S1 `acc = ftz(fma(x, x, acc))`
                                   ascending from `+0.0` per row, S2 the mean
                                   through `identical_div` and
                                   `identical_rsqrt(ftz(mean + eps))`, S3 and
                                   S4 two pinned products. The kernel takes
                                   `eps` as an argument where
                                   `transformer_oracle.rms_norm_into` reads
                                   the profile's `RMS_EPS`, so the rows are
                                   spelled here with the caller's `eps`.
  `host_samba_rms_norm_backward`   `samba_rms_norm_backward_host` (`:190`):
                                   the forward's row sums recomputed, then
                                   `bwd_norm_dh_kernel`, `bwd_norm_dot_kernel`
                                   and `bwd_norm_dx_kernel`
                                   (`transformer/checks/transformer_backward.
                                   mojo:715-902`, the same statements as
                                   `transformer_backward_oracle.
                                   rms_norm_backward_into` with the caller's
                                   `eps`), and the weight gradient
                                   `gemm_oracle(ones[1 x m], dprod, OP_NN, 1,
                                   dm, m)` (`bwd_rms_norm`'s last call).
  `host_samba_linear_forward`      `samba_linear_forward_host` (`:263`):
                                   `gemm_oracle` at `OP_NT`.
  `host_samba_linear_backward`     `samba_linear_backward_host` (`:295`): the
                                   GEMM lane's backward call table
                                   (`gemm_backward_a_call`,
                                   `gemm_backward_b_call`) over `gemm_oracle`.
  `host_samba_accumulate`          `samba_accumulate_host` (`:395`) and
                                   `samba_tree_level_kernel` (`:348`): pairs
                                   of pieces in ascending microbatch index,
                                   `ftz(fma(1, ftz(left), ftz(right)))` per
                                   cell, until one piece remains.

THE NEGATIVE CONTROL. `gemm_oracle` flips a VALUE in every leaf under
`-D MOJOLEARN_HOST_SABOTAGE=1` (`GEMM_ORACLE_HOST_SABOTAGE`), which moves the
linear forward and backward and the RMSNorm weight gradient. The embedding
gather has no arm of its own. Accumulation corrupts its first output to zero:
pairwise addition has no fold-order fault (addition commutes), and the public
ordered-shard reduction reaches no GEMM. Its independent oracle must catch
the changed native result, including the cancellation fixture ending at three.
"""
from std.math import isfinite

from checks.numerics import ftz, identical_div, identical_mul, identical_mul_add, identical_rsqrt
from embedding.checks.embedding_oracle import (
    EmbConfig,
    emb_backward_oracle,
    emb_forward_oracle,
)
from gemm.checks.gemm_backward import (
    BWD_DC_LEFT,
    gemm_backward_a_call,
    gemm_backward_b_call,
)
from gemm.host.identical_gemm import OP_NN, OP_NT, gemm_oracle, GEMM_ORACLE_HOST_SABOTAGE
from training.checks.optimizer_oracle import microbatch_split_is_identical


#: `transformer/checks/transformer_backward.mojo:85-86`.
comptime HOST_BWD_NEG_HALF: Float32 = -0.5
comptime HOST_BWD_TWO: Float32 = 2.0


def host_samba_refuse_nonfinite(name: String, values: List[Float32]) raises:
    """`_refuse_nonfinite`, `samba_ops.mojo:53`, in its words."""
    for i in range(len(values)):
        if not isfinite(values[i]):
            raise Error(
                "mojolearn samba ops: non-finite " + name + " at flat index "
                + String(i)
            )


# ===========================================================================
# EMBEDDING
# ===========================================================================


def host_samba_embedding_forward(
    w: List[Float32], ids: List[Int32], n_positions: Int, vocab: Int, width: Int
) raises -> List[Float32]:
    if n_positions < 1 or vocab < 1 or width < 1:
        raise Error("mojolearn samba ops: embedding shape must be positive")
    host_samba_refuse_nonfinite("embedding weight", w)
    return emb_forward_oracle(w, ids, EmbConfig.llama(vocab, width))


def host_samba_embedding_backward(
    dy: List[Float32], ids: List[Int32], n_positions: Int, vocab: Int, width: Int
) raises -> List[Float32]:
    if n_positions < 1 or vocab < 1 or width < 1:
        raise Error("mojolearn samba ops: embedding shape must be positive")
    host_samba_refuse_nonfinite("embedding upstream gradient", dy)
    return emb_backward_oracle(dy, ids, EmbConfig.llama(vocab, width), List[Float32]())


# ===========================================================================
# RMSNORM
# ===========================================================================


def _refuse_rms(m: Int, dm: Int, eps: Float32) raises:
    if m < 1 or dm < 1:
        raise Error("mojolearn samba ops: rms_norm shape must be positive")
    if not isfinite(eps) or eps < Float32(0.0):
        raise Error("mojolearn samba ops: rms_norm eps must be finite and >= 0")


def host_rms_row_sumsq(x: List[Float32], m: Int, dm: Int) -> List[Float32]:
    """S1 per row: `acc = ftz(fma(x_j, x_j, acc))`, ascending from `+0.0`."""
    var sumsq = List[Float32](capacity=m)
    for t in range(m):
        var acc = Float32(0.0)
        for j in range(dm):
            var xj = ftz(x[t * dm + j])
            acc = ftz(identical_mul_add(xj, xj, acc))
        sumsq.append(acc)
    return sumsq^


def host_samba_rms_norm_forward(
    x: List[Float32], w: List[Float32], m: Int, dm: Int, eps: Float32
) raises -> List[Float32]:
    _refuse_rms(m, dm, eps)
    host_samba_refuse_nonfinite("rms_norm input", x)
    host_samba_refuse_nonfinite("rms_norm weight", w)
    var sumsq = host_rms_row_sumsq(x, m, dm)
    var y = List[Float32](capacity=m * dm)
    for t in range(m):
        var mean = ftz(identical_div(sumsq[t], Float32(dm)))
        var rstd = ftz(identical_rsqrt(ftz(mean + eps)))
        for j in range(dm):
            var inner = ftz(identical_mul(ftz(x[t * dm + j]), rstd))
            y.append(ftz(identical_mul(ftz(w[j]), inner)))
    return y^


def host_samba_rms_norm_backward(
    dy: List[Float32], x: List[Float32], w: List[Float32], m: Int, dm: Int, eps: Float32
) raises -> Tuple[List[Float32], List[Float32]]:
    """`(dx[m, dm], dw[dm])`."""
    _refuse_rms(m, dm, eps)
    host_samba_refuse_nonfinite("rms_norm input", x)
    host_samba_refuse_nonfinite("rms_norm weight", w)
    host_samba_refuse_nonfinite("rms_norm upstream gradient", dy)
    var sumsq = host_rms_row_sumsq(x, m, dm)
    var dx = List[Float32](capacity=m * dm)
    var dprod = List[Float32](capacity=m * dm)
    for t in range(m):
        # `bwd_norm_dh_kernel`, then `bwd_norm_dot_kernel`'s `c` fold.
        var dh = List[Float32](capacity=dm)
        var c = Float32(0.0)
        for j in range(dm):
            var dhj = ftz(identical_mul(ftz(dy[t * dm + j]), ftz(w[j])))
            dh.append(dhj)
            c = ftz(identical_mul_add(dhj, ftz(x[t * dm + j]), c))
        c = ftz(c)
        var mean = ftz(identical_div(ftz(sumsq[t]), Float32(dm)))
        var rstd = ftz(identical_rsqrt(ftz(mean + eps)))
        var r2 = ftz(identical_mul(rstd, rstd))
        var r3 = ftz(identical_mul(r2, rstd))
        var cr3 = ftz(identical_mul(c, r3))
        var da = ftz(identical_mul(HOST_BWD_NEG_HALF, cr3))
        var dv = ftz(identical_div(da, Float32(dm)))
        # `bwd_norm_dx_kernel`.
        for j in range(dm):
            var xj = ftz(x[t * dm + j])
            var dx1 = ftz(identical_mul(ftz(dh[j]), rstd))
            var tx = ftz(identical_mul(HOST_BWD_TWO, xj))
            var dx2 = ftz(identical_mul(dv, tx))
            dx.append(ftz(ftz(dx1) + ftz(dx2)))
            var inner = ftz(identical_mul(xj, rstd))
            dprod.append(ftz(identical_mul(ftz(dy[t * dm + j]), inner)))
    var ones = List[Float32](length=m, fill=Float32(1.0))
    var dw = gemm_oracle(ones, dprod, OP_NN, 1, dm, m)
    return (dx^, dw^)


# ===========================================================================
# THE LM HEAD
# ===========================================================================


def host_samba_linear_forward(
    a: List[Float32], w: List[Float32], m: Int, n: Int, k: Int
) raises -> List[Float32]:
    if m < 1 or n < 1 or k < 1:
        raise Error("mojolearn samba ops: linear shape must be positive")
    host_samba_refuse_nonfinite("linear input", a)
    host_samba_refuse_nonfinite("linear weight", w)
    return gemm_oracle(a, w, OP_NT, m, n, k)


def _gemm_by_call(
    dc: List[Float32], other: List[Float32], call: Tuple[Int, Int, Int, Int, Int]
) -> List[Float32]:
    if call[4] == BWD_DC_LEFT:
        return gemm_oracle(dc, other, call[0], call[1], call[2], call[3])
    return gemm_oracle(other, dc, call[0], call[1], call[2], call[3])


def host_samba_linear_backward(
    dc: List[Float32], a: List[Float32], w: List[Float32], m: Int, n: Int, k: Int
) raises -> Tuple[List[Float32], List[Float32]]:
    """`(da[m, k], dw[n, k])`: `identical_gemm_backward_a_into(da, dc, W)`
    and `identical_gemm_backward_b_into(dw, dc, A)` at the forward's
    `OP_NT`."""
    if m < 1 or n < 1 or k < 1:
        raise Error("mojolearn samba ops: linear shape must be positive")
    host_samba_refuse_nonfinite("linear input", a)
    host_samba_refuse_nonfinite("linear weight", w)
    host_samba_refuse_nonfinite("linear upstream gradient", dc)
    var da = _gemm_by_call(dc, w, gemm_backward_a_call(OP_NT, m, n, k))
    var dw = _gemm_by_call(dc, a, gemm_backward_b_call(OP_NT, m, n, k))
    return (da^, dw^)


# ===========================================================================
# THE BALANCED-TREE ACCUMULATE
# ===========================================================================


def host_samba_validate_accumulation(n: Int, a: Int, t_tokens: Int) raises:
    """`samba_validate_accumulation`, `samba_ops.mojo:368`, in its words."""
    if n < 1:
        raise Error("mojolearn samba ops: accumulate n must be at least 1")
    if a < 1:
        raise Error("mojolearn samba ops: accumulation_steps must be >= 1")
    var q = a
    while q > 1:
        if q % 2 != 0:
            raise Error(
                "mojolearn samba ops: accumulation_steps must be a POWER OF"
                " TWO (optimizer contract clause 9.2 condition 4), got "
                + String(a)
            )
        q = q // 2
    if t_tokens == 0:
        raise Error("mojolearn samba ops: t_tokens must be >= 1 or -1")
    if t_tokens > 0 and not microbatch_split_is_identical(t_tokens, a):
        raise Error(
            "mojolearn samba ops: MISALIGNED microbatch split, T = "
            + String(t_tokens) + " tokens at A = " + String(a)
            + " does not satisfy optimizer contract clause 9.2 (leaf size,"
            " T mod L, A divides P, A a power of two); this accumulation"
            " would be a different numerical experiment from the unsplit step"
        )


def host_samba_accumulate(
    parts: List[Float32], n: Int, a: Int, t_tokens: Int
) raises -> List[Float32]:
    """`out[n] = tree(parts[0], ..., parts[a - 1])`."""
    host_samba_validate_accumulation(n, a, t_tokens)
    host_samba_refuse_nonfinite("accumulate parts", parts)
    var cur = List[Float32](capacity=n * a)
    for i in range(n * a):
        cur.append(parts[i])
    if a == 1:
        var single = List[Float32](capacity=n)
        for i in range(n):
            single.append(ftz(cur[i]))
        return single^
    var pieces = a
    while pieces > 1:
        var pairs = pieces // 2
        var nxt = List[Float32](capacity=n * pairs)
        for j in range(pairs):
            for e in range(n):
                var left = ftz(cur[(2 * j) * n + e])
                var right = ftz(cur[(2 * j + 1) * n + e])
                nxt.append(ftz(identical_mul_add(Float32(1.0), left, right)))
        cur = nxt^
        pieces = pairs
    comptime if GEMM_ORACLE_HOST_SABOTAGE:
        # Pairwise addition has no fold-order fault. Corrupt the actual native
        # result so the ordered-shard reduction's independent oracle can prove
        # sensitivity even when no GEMM is involved.
        cur[0] = Float32(0.0)
    return cur^
