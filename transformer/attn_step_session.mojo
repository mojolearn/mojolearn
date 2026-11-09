# SPDX-License-Identifier: Apache-2.0
"""The public `TransformerDecodeSession`'s GPU arm on the attention step op
(lane x4-transformer-on-executor, 2026-11-09; plan
docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md row X4, API
docs/plans/EXECUTOR_X1_API.md).

The session's weights are ONE packed executor buffer (`block_weight_sizes`
order, each piece at a `block_slot` boundary), its key and value caches ONE
packed state buffer (`block_state_sizes`: keys then values) and its position
ONE int32 word: the three operands of `OPS_ATTN_STEP` after x. `step`,
`forward` and `decode` run the op (`attn_step_run`, the body
`OpsSession._block_step` runs), so the session has no block structs of its
own any more: the op builds them once over these buffers (`_AttnBlock`, the
same weights constructor, rotary table, L = 1 stages and block forward entry
the session held) and keeps them in `sc`. Bits are the op's, which are the
session's (EXECUTOR_X1_API.md section 2: "The bodies are the resident
sessions' unchanged").

This replaces, for a session the public door opens, the session-owned
LlamaDeviceWeights / KV cache / rotary table / stages and the step, forward
and decode bodies over them (`_session_step_run_sync`,
`_session_step_run_deferred`, `_session_forward_run`, `_session_decode_run`
in transformer/transformer_binding_impl.mojo). The CausalLM resident read
still opens its layer sessions on those structs (`_lm_layers_run` reads
them); it moves to the executor with `Program.decode` (RUN OWED, see the
lane's reply).

No host compute: the host side computes layout offsets and copies whole
caller buffers once per call.
"""

from std.memory import bitcast
from max.gpu.host import DeviceBuffer, DeviceContext
from sequence.ops import FP
from sequence.exec_device import DeviceExec
from core.hostptr import f32_ptr
from core.decode_slab import decode_slab_check, decode_slab_put, decode_slab_take
from autograd.block_step_ops import (
    OPS_ATTN_STEP,
    BlockStepScratch,
    attn_step_run,
    block_guard,
    block_offsets,
    block_state_sizes,
    block_weight_sizes,
)
from transformer.block_options import BLOCK_OPTION_ADDRS, BLOCK_OPTION_PARAMS, BlockOptions


def _fp(addr: Int) -> FP:
    return FP(unsafe_from_address=addr)


struct AttnStepArm(Movable):
    """One block's step-op operands on one executor, and the host mirror of
    the position the op's state word holds."""

    var ex: DeviceExec
    var sc: BlockStepScratch
    #: the op's integer arguments: [b, d_model, n_heads, n_kv, head_dim,
    #: intermediate, s_max, window] and, for a non-default block, the 17
    #: option words
    var ints: List[Int]
    var w: FP
    var kv: FP
    var q: FP
    var nw: Int
    var nkv: Int
    #: one cache's floats and the value cache's offset in `kv`
    var cache_n: Int
    var v_at: Int
    var b: Int
    var dm: Int
    var smax: Int
    var cached: Int

    def __init__(
        out self, ctx: DeviceContext, a: List[Int], b: Int, dm: Int, nh: Int, nkv: Int, hd: Int, it: Int,
        smax: Int, s0: Int, window: Int, opts: BlockOptions, option_words: List[Int],
    ) raises:
        """Open on the caller's weights and caches. `a` is the
        `transformer_forward` 13 + 11 layout (slots 0 and 12 unused):
        weights at 1..9, k_cache 10, v_cache 11, the eleven optional
        weights after 13. `option_words` is empty for a default block, else
        the 17 option words. Everything is COPIED to the device here."""
        if s0 < 0 or s0 > smax:
            raise Error(
                String("transformer decode session: cached_tokens must be in [0, ") + String(smax)
                + "], got " + String(s0))
        if window < 0:
            raise Error("transformer decode session: window must be >= 0 (0 = full causal)")
        var ints: List[Int] = [b, dm, nh, nkv, hd, it, smax, window]
        if len(option_words) != 0:
            if len(option_words) != BLOCK_OPTION_PARAMS:
                raise Error("transformer decode session: the block options tail holds "
                            + String(BLOCK_OPTION_PARAMS) + " words")
            for i in range(len(option_words)):  # small-loop(option_words: the 17 block option words): argument words, no data
                ints.append(option_words[i])
        var sizes = block_weight_sizes(OPS_ATTN_STEP, ints)
        var at = block_offsets(sizes, block_guard(OPS_ATTN_STEP))
        var ssz = block_state_sizes(OPS_ATTN_STEP, ints)
        var sat = block_offsets(ssz, block_guard(OPS_ATTN_STEP))
        # the twenty pieces' caller addresses in packing order: the nine base
        # weights, then the eleven optional ones (zero where absent)
        var src = List[Int](capacity=len(sizes))
        for i in range(9):  # small-loop(9: the base weight slots): argument words, no data
            src.append(a[1 + i])
        for i in range(BLOCK_OPTION_ADDRS):  # small-loop(11: the optional weight slots): argument words, no data
            src.append(a[13 + i] if 13 + i < len(a) else 0)
        if opts.gated() and src[6] == 0:
            raise Error("transformer: gate_proj.weight address is null (a gated MLP needs it)")
        for i in range(len(sizes)):  # small-loop(sizes: the twenty weight pieces): address checks, no data
            if sizes[i] == 0 and src[i] != 0:
                raise Error("transformer options: weight piece " + String(i)
                            + " was passed but these block options have no such weight")
            if sizes[i] != 0 and src[i] == 0:
                raise Error("transformer options: weight piece " + String(i) + " address is null")
        var ex = DeviceExec(ctx)
        var nw = at[len(at) - 1]
        var nkv_floats = sat[len(sat) - 1]
        var w = ex.alloc(nw)
        var kv = ex.alloc(nkv_floats)
        var q = ex.alloc(1)
        for i in range(len(sizes)):  # driver-loop: over the block's twenty weight pieces, one copy each
            if sizes[i] != 0:
                ex.upload(w + at[i], f32_ptr(src[i]), sizes[i])
        self.ex = ex^
        self.sc = BlockStepScratch()
        self.ints = ints^
        self.w = w
        self.kv = kv
        self.q = q
        self.nw = nw
        self.nkv = nkv_floats
        self.cache_n = ssz[0]
        self.v_at = sat[1]
        self.b = b
        self.dm = dm
        self.smax = smax
        self.cached = s0
        self._load(a[10], a[11], s0)
        self.ex.sync()

    def _load(mut self, pk: Int, pv: Int, s0: Int) raises:
        """The caches and the position from the caller (queued; the
        executor stages the bytes at once). The op's resident struct over
        the old contents is dropped, so its next run reads the word back."""
        if s0 < 0 or s0 > self.smax:
            raise Error(
                String("transformer decode session: cached_tokens must be in [0, ") + String(self.smax)
                + "], got " + String(s0))
        self.sc.forget(Int(self.kv))
        self.ex.upload(self.kv, f32_ptr(pk), self.cache_n)
        self.ex.upload(self.kv + self.v_at, f32_ptr(pv), self.cache_n)
        var word = List[Float32](length=1, fill=Float32(0))
        word[0] = bitcast[DType.float32](Int32(s0))
        self.ex.upload(self.q, FP(unsafe_from_address=Int(word.unsafe_ptr())), 1)
        _ = len(word)  # keep-alive past the staging copy
        self.cached = s0

    def _operands(self, x: FP) -> List[FP]:
        var bufs: List[FP] = [x, self.w, self.kv, self.q]
        return bufs^

    def _counts(self, n: Int) -> List[Int]:
        var c: List[Int] = [n, self.nw, self.nkv, 1]
        return c^

    def run(mut self, px: Int, py: Int, l: Int) raises -> Int:
        """`l` tokens per row through the op (L = 1 a decode step, L > 1 a
        prefill or a chunked continuation): x up, one op, y down (the
        download is the call's one wait)."""
        if l < 1:
            raise Error("transformer decode session: L must be positive")
        var n = self.b * l * self.dm
        var x = self.ex.alloc(n)
        var y = self.ex.alloc(n)
        try:
            self.ex.upload(x, f32_ptr(px), n)
            attn_step_run(self.ex, self.sc, self._operands(x), self._counts(n), self.ints, y)
            self.ex.download(f32_ptr(py), y, n)
        except err:
            self.ex.free(x)
            self.ex.free(y)
            raise err
        self.ex.free(x)
        self.ex.free(y)
        self.cached += l
        return self.cached

    def decode(mut self, px: Int, py: Int, steps: Int) raises -> Int:
        """`steps` decode tokens per row in one call: the caller's
        (B, N, d_model) x up once, the op at L = 1 once per token with
        token t moved on the device, the (B, N, d_model) output down once.
        Bytes equal `steps` calls of `run(.., 1)`."""
        decode_slab_check(self.b, steps, self.dm, "transformer decode session decode")
        var row = self.b * self.dm
        var n = row * steps
        var xs = self.ex.alloc(n)
        var ys = self.ex.alloc(n)
        var xt = self.ex.alloc(row)
        var yt = self.ex.alloc(row)
        try:
            self.ex.upload(xs, f32_ptr(px), n)
            var ctx = self.ex.ctx.copy()
            var xs_v = _exec_view(self.ex, xs, n)
            var ys_v = _exec_view(self.ex, ys, n)
            var xt_v = _exec_view(self.ex, xt, row)
            var yt_v = _exec_view(self.ex, yt, row)
            var bufs = self._operands(xt)
            var counts = self._counts(row)
            for t in range(steps):  # driver-loop: over autoregressive decode steps (each reads the cache the previous step left), the step op once each
                decode_slab_take(ctx, xt_v, xs_v, self.b, steps, self.dm, t)
                attn_step_run(self.ex, self.sc, bufs, counts, self.ints, yt)
                decode_slab_put(ctx, ys_v, yt_v, self.b, steps, self.dm, t)
            self.ex.download(f32_ptr(py), ys, n)
        except err:
            self.ex.free(xs)
            self.ex.free(ys)
            self.ex.free(xt)
            self.ex.free(yt)
            raise err
        self.ex.free(xs)
        self.ex.free(ys)
        self.ex.free(xt)
        self.ex.free(yt)
        self.cached += steps
        return self.cached

    def export(mut self, pk: Int, pv: Int) raises -> Int:
        """Both caches into the caller's buffers (one wait each)."""
        self.ex.download(f32_ptr(pk), self.kv, self.cache_n)
        self.ex.download(f32_ptr(pv), self.kv + self.v_at, self.cache_n)
        return self.cached

    def load(mut self, pk: Int, pv: Int, s0: Int) raises -> Int:
        self._load(pk, pv, s0)
        self.ex.sync()
        return s0


def _exec_view(ex: DeviceExec, p: FP, n: Int) raises -> DeviceBuffer[DType.float32]:
    """The executor's storage [p, p + n) as a device buffer (no copy)."""
    var found = ex._find(p, n)
    return ex._sub(found[0], found[1], n)
