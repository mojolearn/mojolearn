# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The block's products under `numeric_profile="fixed15_v1"`, inference only
(lane/lowbit-blocks, 2026-09-29).

    GEMM profile  mojolearn.identical.gemm.int15i64.v1
                  (`gemm/IDENTICAL_LOWBIT_CONTRACT.md` section 6)

WHAT RUNS HERE. The seven projections of a Llama-shaped block
(`q, k, v, o, gate, up, down`) and the score product S11 (`Q.K^T`, one call
per (batch, head), `k = head_dim`). Nothing else: the attention product
P.V (S19, `attn_context_kernel`) STAYS on fp32.v1's pinned ascending chain
in version 1 (the brief's section on P.V: under the per-row rule it would
depend on the key span, and decode would not equal prefill), and every
elementwise seam is the block's own.

WHICH VALUES EACH PRODUCT QUANTIZES (clause W-9). Every product is OP_NT and
each operand is quantized along that product's own contracted extent from
its float32 values:
  projection  `Y = X W^T`   X one scale per TOKEN over the input features;
                            W one scale per OUTPUT FEATURE over the input
                            features. W's planes are made ONCE (at load, by
                            the caller) and kept; X is quantized per call by
                            the PARALLEL quantizer (clause W-10).
  S11         `Q K^T`       one scale per QUERY over head_dim, one per KEY
                            over head_dim, both per call.
Every row's scale is a function of that row's own values and the integer sum
is exact (W-4, W-5), so a cell is a function of its own two rows: which
other tokens share the call cannot move it (batch invariance), and a key's
codes are the same whether it arrived in a prefill or a decode step (decode
equals prefill). Every plan of `identical_gemm_int15_planes_into` is the
same bits (W-8), so the dispatcher's choice per column is scheduling.

THE ENTRY POINTS are Lane C's (`gemm/checks/gemm_int15.mojo`):
`quantize_planes_int15_parallel_device` and
`identical_gemm_int15_planes_into`. This file adds no arithmetic.
"""

from max.gpu.host import DeviceBuffer, DeviceContext

from core.step_phase import step_count_device_alloc, step_count_sync
from gemm.checks.gemm_int15 import (
    Int15QuantWorkspace,
    identical_gemm_int15_planes_into,
    quantize_planes_int15_parallel_device,
)
from gemm.host.gemm_int15_oracle import INT15_MAX_K

# The projections, in the binding's weight order (q, k, v, o, gate, up, down).
comptime LLAMA_PROJ_Q = 0
comptime LLAMA_PROJ_K = 1
comptime LLAMA_PROJ_V = 2
comptime LLAMA_PROJ_O = 3
comptime LLAMA_PROJ_GATE = 4
comptime LLAMA_PROJ_UP = 5
comptime LLAMA_PROJ_DOWN = 6
comptime LLAMA_PROJ_COUNT = 7


def llama_proj_name(which: Int) -> String:
    if which == LLAMA_PROJ_Q:
        return "q_proj"
    if which == LLAMA_PROJ_K:
        return "k_proj"
    if which == LLAMA_PROJ_V:
        return "v_proj"
    if which == LLAMA_PROJ_O:
        return "o_proj"
    if which == LLAMA_PROJ_GATE:
        return "gate_proj"
    if which == LLAMA_PROJ_UP:
        return "up_proj"
    return "down_proj"


def _refuse_k(k: Int, what: String) raises:
    if k < 1 or k > INT15_MAX_K:
        raise Error(
            "llama int15: " + what + " contracts k=" + String(k)
            + ", outside [1, " + String(INT15_MAX_K)
            + "] (INT15_MAX_K, contract W-4); refused by name"
        )


struct Int15Planes(Copyable, Movable):
    """One weight kept as the profile stores it: the two int8 planes of its
    codes (`code = hi * 128 + lo`) and one int32 exponent per row, `rows x
    cols` with rows the output features and cols the contracted extent. A
    copy shares the device allocations (a `DeviceBuffer` copy is a handle)."""

    var hi: DeviceBuffer[DType.int8]
    var lo: DeviceBuffer[DType.int8]
    var e: DeviceBuffer[DType.int32]
    var rows: Int
    var cols: Int

    def __init__(
        out self,
        var hi: DeviceBuffer[DType.int8],
        var lo: DeviceBuffer[DType.int8],
        var e: DeviceBuffer[DType.int32],
        rows: Int,
        cols: Int,
    ) raises:
        if rows < 1 or cols < 1:
            raise Error("llama int15: a weight's planes need rows and cols >= 1")
        if len(hi) < rows * cols or len(lo) < rows * cols or len(e) < rows:
            raise Error(
                "llama int15: planes shorter than " + String(rows) + " x "
                + String(cols)
            )
        self.hi = hi^
        self.lo = lo^
        self.e = e^
        self.rows = rows
        self.cols = cols


def int15_planes_from_f32(
    ctx: DeviceContext,
    mut w: DeviceBuffer[DType.float32],
    rows: Int,
    cols: Int,
    mut quant: Int15QuantWorkspace,
) raises -> Int15Planes:
    """A float32 weight on the device, quantized row by row to planes by the
    PARALLEL quantizer. Synchronizes (the planes are kept)."""
    step_count_device_alloc()
    var hi = ctx.enqueue_create_buffer[DType.int8](rows * cols)
    step_count_device_alloc()
    var lo = ctx.enqueue_create_buffer[DType.int8](rows * cols)
    step_count_device_alloc()
    var e = ctx.enqueue_create_buffer[DType.int32](rows)
    quantize_planes_int15_parallel_device(ctx, hi, lo, e, w, quant, rows, cols, False)
    step_count_sync()
    ctx.synchronize()
    return Int15Planes(hi^, lo^, e^, rows, cols)


struct LlamaInt15Weights(Copyable, Movable):
    """The seven projection weights of one block as planes. Under an ungated
    MLP `gate` is a one-row placeholder that no call reads."""

    var q: Int15Planes
    var k: Int15Planes
    var v: Int15Planes
    var o: Int15Planes
    var gate: Int15Planes
    var up: Int15Planes
    var down: Int15Planes
    var gated: Bool

    def __init__(
        out self,
        var q: Int15Planes,
        var k: Int15Planes,
        var v: Int15Planes,
        var o: Int15Planes,
        var gate: Int15Planes,
        var up: Int15Planes,
        var down: Int15Planes,
        gated: Bool,
    ):
        self.q = q^
        self.k = k^
        self.v = v^
        self.o = o^
        self.gate = gate^
        self.up = up^
        self.down = down^
        self.gated = gated

    def get(self, which: Int) -> Int15Planes:
        if which == LLAMA_PROJ_Q:
            return self.q.copy()
        if which == LLAMA_PROJ_K:
            return self.k.copy()
        if which == LLAMA_PROJ_V:
            return self.v.copy()
        if which == LLAMA_PROJ_O:
            return self.o.copy()
        if which == LLAMA_PROJ_GATE:
            return self.gate.copy()
        if which == LLAMA_PROJ_UP:
            return self.up.copy()
        return self.down.copy()

    def check_shapes(
        self, d_model: Int, q_width: Int, kv_width: Int, intermediate: Int
    ) raises:
        """Each weight's planes against the block's shapes, BY NAME: a
        transposed weight is a plausible buffer of the right size."""
        _want(self.q, "q_proj", q_width, d_model)
        _want(self.k, "k_proj", kv_width, d_model)
        _want(self.v, "v_proj", kv_width, d_model)
        _want(self.o, "o_proj", d_model, q_width)
        if self.gated:
            _want(self.gate, "gate_proj", intermediate, d_model)
        _want(self.up, "up_proj", intermediate, d_model)
        _want(self.down, "down_proj", d_model, intermediate)


def _want(p: Int15Planes, name: String, rows: Int, cols: Int) raises:
    if p.rows != rows or p.cols != cols:
        raise Error(
            "llama int15: " + name + " planes are " + String(p.rows) + " x "
            + String(p.cols) + ", the block wants " + String(rows) + " x "
            + String(cols)
        )
    _refuse_k(cols, name)


struct LlamaInt15Stage(Movable):
    """The per-call scratch of the profile on ONE in-order context: the left
    operand's planes and exponents, the right operand's (S11 only; a
    projection's right operand is the kept weight), and the parallel
    quantizer's chunk maxima. Grown on demand; growth drains the context
    first, as `Int15Workspace` does."""

    var ah: DeviceBuffer[DType.int8]
    var al: DeviceBuffer[DType.int8]
    var ea: DeviceBuffer[DType.int32]
    var bh: DeviceBuffer[DType.int8]
    var bl: DeviceBuffer[DType.int8]
    var eb: DeviceBuffer[DType.int32]
    var quant: Int15QuantWorkspace

    def __init__(out self, ctx: DeviceContext) raises:
        step_count_device_alloc()
        self.ah = ctx.enqueue_create_buffer[DType.int8](1)
        step_count_device_alloc()
        self.al = ctx.enqueue_create_buffer[DType.int8](1)
        step_count_device_alloc()
        self.ea = ctx.enqueue_create_buffer[DType.int32](1)
        step_count_device_alloc()
        self.bh = ctx.enqueue_create_buffer[DType.int8](1)
        step_count_device_alloc()
        self.bl = ctx.enqueue_create_buffer[DType.int8](1)
        step_count_device_alloc()
        self.eb = ctx.enqueue_create_buffer[DType.int32](1)
        self.quant = Int15QuantWorkspace(ctx)

    def ensure_a(mut self, ctx: DeviceContext, codes: Int, rows: Int) raises:
        if codes > len(self.ah) or rows > len(self.ea):
            step_count_sync()
            ctx.synchronize()
            if codes > len(self.ah):
                step_count_device_alloc()
                self.ah = ctx.enqueue_create_buffer[DType.int8](codes)
                step_count_device_alloc()
                self.al = ctx.enqueue_create_buffer[DType.int8](codes)
            if rows > len(self.ea):
                step_count_device_alloc()
                self.ea = ctx.enqueue_create_buffer[DType.int32](rows)

    def ensure_b(mut self, ctx: DeviceContext, codes: Int, rows: Int) raises:
        if codes > len(self.bh) or rows > len(self.eb):
            step_count_sync()
            ctx.synchronize()
            if codes > len(self.bh):
                step_count_device_alloc()
                self.bh = ctx.enqueue_create_buffer[DType.int8](codes)
                step_count_device_alloc()
                self.bl = ctx.enqueue_create_buffer[DType.int8](codes)
            if rows > len(self.eb):
                step_count_device_alloc()
                self.eb = ctx.enqueue_create_buffer[DType.int32](rows)


def llama_int15_proj(
    ctx: DeviceContext,
    mut st: LlamaInt15Stage,
    mut c: DeviceBuffer[DType.float32],
    mut a: DeviceBuffer[DType.float32],
    w: Int15Planes,
    m: Int,
    n: Int,
    k: Int,
) raises:
    """`C[m x n] = X[m x k] . W[n x k]^T` under the profile: X quantized per
    token by the parallel quantizer, W's kept planes. Asynchronous except
    where Lane C's dispatcher waits (clause W-14, Apple)."""
    if w.rows != n or w.cols != k:
        raise Error(
            "llama int15: a projection of " + String(n) + " x " + String(k)
            + " was handed planes of " + String(w.rows) + " x " + String(w.cols)
        )
    _refuse_k(k, "a projection")
    st.ensure_a(ctx, m * k, m)
    quantize_planes_int15_parallel_device(ctx, st.ah, st.al, st.ea, a, st.quant, m, k, False)
    var bh = w.hi.copy()
    var bl = w.lo.copy()
    var eb = w.e.copy()
    identical_gemm_int15_planes_into(ctx, c, st.ah, st.al, st.ea, bh, bl, eb, m, n, k)


def llama_int15_scores(
    ctx: DeviceContext,
    mut st: LlamaInt15Stage,
    mut c: DeviceBuffer[DType.float32],
    mut q: DeviceBuffer[DType.float32],
    mut kmat: DeviceBuffer[DType.float32],
    l: Int,
    s: Int,
    hd: Int,
) raises:
    """S11 under the profile, one (batch, head): `C[l x s] = Q[l x hd] .
    K[s x hd]^T`, one scale per query and one per key, each over head_dim,
    both quantized here by the parallel quantizer."""
    _refuse_k(hd, "the score product (head_dim)")
    st.ensure_a(ctx, l * hd, l)
    st.ensure_b(ctx, s * hd, s)
    quantize_planes_int15_parallel_device(ctx, st.ah, st.al, st.ea, q, st.quant, l, hd, False)
    quantize_planes_int15_parallel_device(ctx, st.bh, st.bl, st.eb, kmat, st.quant, s, hd, False)
    identical_gemm_int15_planes_into(ctx, c, st.ah, st.al, st.ea, st.bh, st.bl, st.eb, l, s, hd)
