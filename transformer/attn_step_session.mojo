# SPDX-License-Identifier: Apache-2.0
"""Every GPU transformer block runtime on the attention step op (lanes
x4-transformer-on-executor and x4b-transformer-block-and-generate,
2026-11-09; plan docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md row X4,
API docs/plans/EXECUTOR_X1_API.md).

A block's weights are ONE packed executor buffer (`block_weight_sizes`
order, each piece at a `block_slot` boundary), its key and value caches ONE
packed state buffer (`block_state_sizes`: keys then values) and its position
ONE int32 word: the three operands of `OPS_ATTN_STEP` after x. Every run is
the op (`attn_step_run_kept`, the body `OpsSession._block_step` runs), so no
caller holds block structs of its own: the op builds them once over these
buffers (`_AttnBlock`, the same weights constructor, rotary table, L = 1
stages and block forward entry) and keeps them in `sc`. Bits are the op's.

`AttnStepArm` serves:
- the public `TransformerDecodeSession` (open, step, forward, decode,
  export, load), float32 and fixed15_v1 (the planes handed to the op as an
  adopted entry, `attn_step_adopt`);
- the CausalLM layer sessions (`run_device`, on the stack's device buffer
  and one prefill workspace shared by the layers);
- `TransformerBlock.forward` / `.step` and `CausalLM`'s per-layer forward
  (`block_forward`: the block session's stateless prefill with zero caches
  on the device, or a call over the caller's state arrays), and
  `TransformerBlock.backward` (`block_backward`: the zero-state prefill and
  the transformer stage rule over the op's kept stages).

No host compute: the host side computes layout offsets, keys the uploaded
weights by the caller's addresses and the process weight epoch
(core/weight_epoch.mojo; lane x45c: no pass over the weights) and copies
whole caller buffers.
"""

from std.memory import bitcast, memcpy
from max.gpu.host import DeviceBuffer, DeviceContext
from sequence.ops import FP
from sequence.exec_device import DeviceExec
from core.hostptr import f32_ptr
from core.weight_epoch import weight_epoch
from core.identity_trace import IdentityTrace
from core.decode_slab import decode_slab_check, decode_slab_put, decode_slab_take
from autograd.block_step_ops import (
    OPS_ATTN_STEP,
    AttnPrefill,
    BlockStepScratch,
    attn_step_adopt,
    attn_step_entry,
    attn_step_restart,
    attn_step_run_kept,
    block_guard,
    block_offsets,
    block_state_sizes,
    block_weight_sizes,
)
from autograd.kernels.transformer import LlamaBackwardStages
from autograd.rules_transformer import llama_decoder_layer_backward_autograd
from transformer.block_options import BLOCK_OPTION_ADDRS, BLOCK_OPTION_PARAMS, BlockOptions
from transformer.impl.llama.modeling_llama import LlamaDims, LlamaDeviceWeights


def _fp(addr: Int) -> FP:
    return FP(unsafe_from_address=addr)


def attn_step_ints(b: Int, dm: Int, nh: Int, nkv: Int, hd: Int, it: Int, smax: Int, window: Int,
                   opts: BlockOptions) -> List[Int]:
    """The op's integer arguments for a block: the eight shape words, then
    the 17 option words for a non-default record."""
    var ints: List[Int] = [b, dm, nh, nkv, hd, it, smax, window]
    if not opts.is_default():
        var tail = opts.to_params()
        for i in range(len(tail)):  # small-loop(tail: the 17 block option words): argument words, no data
            ints.append(tail[i])
    return ints^


def attn_step_same_layout(a: List[Int], b: List[Int]) -> Bool:
    """Whether two ints lists pack the weights alike (every word but b,
    s_max and window)."""
    if len(a) != len(b):
        return False
    for i in range(len(a)):  # small-loop(a: at most 25 op argument words): compares keys, no data
        if i == 0 or i == 6 or i == 7:
            continue
        if a[i] != b[i]:
            return False
    return True


def _weight_sources(a: List[Int], n: Int) -> List[Int]:
    """The twenty pieces' caller addresses in packing order: the nine base
    weights at 1..9 of the `transformer_forward` layout, then the eleven
    optional ones after 13 (zero where absent)."""
    var src = List[Int](capacity=n)
    for i in range(9):  # small-loop(9: the base weight slots): argument words, no data
        src.append(a[1 + i])
    for i in range(BLOCK_OPTION_ADDRS):  # small-loop(11: the optional weight slots): argument words, no data
        src.append(a[13 + i] if 13 + i < len(a) else 0)
    return src^


struct AttnStepArm(Movable):
    """One block's step-op operands on one executor, the host mirror of the
    position the op's state word holds, and the kept L > 1 stages."""

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
    #: the weight pieces' lengths and starts in `w` (the version check and
    #: the recopy read them)
    var sizes: List[Int]
    var at: List[Int]
    #: lane x45c: the caller addresses the weights were uploaded from and
    #: the weight epoch at that upload, kept only by an arm that serves a
    #: caller whose arrays may change between calls (`keyed`: the block
    #: session); empty otherwise. Equal addresses at the same epoch hold the
    #: same bytes (the weight door freezes them), so nothing is compared.
    var w_src: List[Int]
    var epoch: Int
    var adopted: Bool
    #: the kept L > 1 stages (block forward, backward, decode-session
    #: forward)
    var pre: AttnPrefill
    #: resident x / y buffers in `ex` (grow only)
    var xio: FP
    var yio: FP
    var nio: Int
    #: the block backward's stages and grad_output copy, kept per shape
    var bst: Optional[LlamaBackwardStages]
    var dout: Optional[DeviceBuffer[DType.float32]]
    var bkey: List[Int]

    def __init__(
        out self, ctx: DeviceContext, a: List[Int], b: Int, dm: Int, nh: Int, nkv: Int, hd: Int, it: Int,
        smax: Int, s0: Int, window: Int, opts: BlockOptions, option_words: List[Int], keyed: Bool = False,
    ) raises:
        """Open on the caller's float32 weights and caches. `a` is the
        `transformer_forward` 13 + 11 layout (slots 0 and 12 unused):
        weights at 1..9, k_cache 10, v_cache 11, the eleven optional
        weights after 13; a zero cache address opens on zero caches (on
        the device). `option_words` is empty for a default block, else the
        17 option words. Everything is COPIED to the device here."""
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
        var src = _weight_sources(a, len(sizes))
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
        var w = ex.alloc(nw)
        var q = ex.alloc(1)
        self.ex = ex^
        self.sc = BlockStepScratch()
        self.ints = List[Int]()
        self.w = w
        self.kv = _fp(0)
        self.q = q
        self.nw = nw
        self.nkv = 0
        self.cache_n = 0
        self.v_at = 0
        self.b = b
        self.dm = dm
        self.smax = smax
        self.cached = 0
        self.sizes = sizes^
        self.at = at^
        self.w_src = List[Int]()
        self.epoch = 0
        self.adopted = False
        self.pre = AttnPrefill()
        self.xio = _fp(0)
        self.yio = _fp(0)
        self.nio = 0
        self.bst = Optional[LlamaBackwardStages]()
        self.dout = Optional[DeviceBuffer[DType.float32]]()
        self.bkey = List[Int]()
        if keyed:
            self.epoch = weight_epoch()
            self.w_src = src.copy()
        self._upload_weights(src)
        self._shape(ints)
        self._load(a[10], a[11], s0)
        self.ex.sync()

    def __init__(
        out self, ctx: DeviceContext, var weights: LlamaDeviceWeights, b: Int, dm: Int, nh: Int, nkv: Int,
        hd: Int, it: Int, smax: Int, s0: Int, window: Int, option_words: List[Int], pk: Int, pv: Int,
        lean: Bool,
    ) raises:
        """Lane x4b: open on weights the caller built (the fixed15_v1 decode
        session's profile constructor). The op runs them as an adopted
        entry; `w` is a one-float placeholder."""
        if window < 0:
            raise Error("transformer decode session: window must be >= 0 (0 = full causal)")
        var ints: List[Int] = [b, dm, nh, nkv, hd, it, smax, window]
        if len(option_words) != 0:
            if len(option_words) != BLOCK_OPTION_PARAMS:
                raise Error("transformer decode session: the block options tail holds "
                            + String(BLOCK_OPTION_PARAMS) + " words")
            for i in range(len(option_words)):  # small-loop(option_words: the 17 block option words): argument words, no data
                ints.append(option_words[i])
        var ex = DeviceExec(ctx)
        var w = ex.alloc(1)
        var q = ex.alloc(1)
        self.ex = ex^
        self.sc = BlockStepScratch()
        self.ints = List[Int]()
        self.w = w
        self.kv = _fp(0)
        self.q = q
        self.nw = 1
        self.nkv = 0
        self.cache_n = 0
        self.v_at = 0
        self.b = b
        self.dm = dm
        self.smax = smax
        self.cached = 0
        self.sizes = List[Int]()
        self.at = List[Int]()
        self.w_src = List[Int]()
        self.epoch = 0
        self.adopted = True
        self.pre = AttnPrefill()
        self.xio = _fp(0)
        self.yio = _fp(0)
        self.nio = 0
        self.bst = Optional[LlamaBackwardStages]()
        self.dout = Optional[DeviceBuffer[DType.float32]]()
        self.bkey = List[Int]()
        self._shape(ints)
        if s0 < 0 or s0 > self.smax:
            raise Error(
                String("transformer decode session: cached_tokens must be in [0, ") + String(self.smax)
                + "], got " + String(s0))
        self._caches_in(pk, pv)
        attn_step_adopt(self.ex, self.sc, weights^, self.w, self.kv, self.q, self.ints, s0, lean)
        self.cached = s0
        self.ex.sync()

    # ------------------------------------------------------------ layout
    def _shape(mut self, ints: List[Int]) raises:
        """Take the op's integer arguments `ints`: a different batch, cache
        capacity or window reallocates the state buffer and drops the
        resident entries (they were built over the old shape). The weights
        buffer is kept (its packing does not depend on these words)."""
        if ints == self.ints:
            return
        if len(self.ints) != 0 and not attn_step_same_layout(self.ints, ints):
            raise Error("transformer: a block's step-op arm cannot change its weight layout")
        if self.adopted and len(self.ints) != 0:
            raise Error("transformer: a fixed15_v1 decode session keeps its shape")
        var ssz = block_state_sizes(OPS_ATTN_STEP, ints)
        var sat = block_offsets(ssz, block_guard(OPS_ATTN_STEP))
        var total = sat[len(sat) - 1]
        self.sc.clear()
        self.pre.clear()
        if total != self.nkv:
            if self.nkv != 0:
                self.ex.free(self.kv)
            self.kv = self.ex.alloc(total)
            self.nkv = total
        self.ints = ints.copy()
        self.cache_n = ssz[0]
        self.v_at = sat[1]
        self.b = ints[0]
        self.smax = ints[6]

    def _operands(self, x: FP) -> List[FP]:
        var bufs: List[FP] = [x, self.w, self.kv, self.q]
        return bufs^

    def _counts(self, n: Int) -> List[Int]:
        var c: List[Int] = [n, self.nw, self.nkv, 1]
        return c^

    def _io(mut self, n: Int) raises:
        """Resident x and y buffers of at least n floats (grow only)."""
        if n <= self.nio:
            return
        if self.nio != 0:
            self.ex.free(self.xio)
            self.ex.free(self.yio)
        self.xio = self.ex.alloc(n)
        self.yio = self.ex.alloc(n)
        self.nio = n

    def _view(self, p: FP, n: Int) raises -> DeviceBuffer[DType.float32]:
        return _exec_view(self.ex, p, n)

    # ------------------------------------------------------------ copies
    def _upload_weights(mut self, src: List[Int]) raises:
        var ctx = self.ex.ctx.copy()
        for i in range(len(self.sizes)):  # driver-loop: over the block's twenty weight pieces, one copy each
            if self.sizes[i] != 0:
                ctx.enqueue_copy(dst_buf=self._view(self.w + self.at[i], self.sizes[i]), src_ptr=f32_ptr(src[i]))

    def _caches_in(mut self, pk: Int, pv: Int) raises:
        """The caller's caches into the state buffer (queued), or zeros on
        the device for a zero address."""
        var ctx = self.ex.ctx.copy()
        if pk == 0 or pv == 0:
            if pk != 0 or pv != 0:
                raise Error("transformer: pass both caches or neither")
            var whole = self._view(self.kv, self.nkv)
            whole.enqueue_fill(Float32(0.0))
            return
        ctx.enqueue_copy(dst_buf=self._view(self.kv, self.cache_n), src_ptr=f32_ptr(pk))
        ctx.enqueue_copy(dst_buf=self._view(self.kv + self.v_at, self.cache_n), src_ptr=f32_ptr(pv))

    def _caches_out(mut self, pk: Int, pv: Int) raises:
        """Both caches into the caller's buffers (queued; the caller waits)."""
        var ctx = self.ex.ctx.copy()
        ctx.enqueue_copy(dst_ptr=f32_ptr(pk), src_buf=self._view(self.kv, self.cache_n))
        ctx.enqueue_copy(dst_ptr=f32_ptr(pv), src_buf=self._view(self.kv + self.v_at, self.cache_n))

    def _load(mut self, pk: Int, pv: Int, s0: Int) raises:
        """The caches and the position from the caller (queued). The op's
        resident entry keeps its structs: its cache struct and the state
        word move to s0 together (`attn_step_restart`)."""
        if s0 < 0 or s0 > self.smax:
            raise Error(
                String("transformer decode session: cached_tokens must be in [0, ") + String(self.smax)
                + "], got " + String(s0))
        self._caches_in(pk, pv)
        attn_step_restart(self.ex, self.sc, self._operands(self.w), self._counts(1), self.ints, s0, False)
        self.cached = s0

    def refresh_weights(mut self, a: List[Int]) raises:
        """The block session's version key (lane x45c): the same caller
        addresses at the same weight epoch keep the device copy with no pass
        over the weights; otherwise the caller's arrays are copied into the
        same buffer and the resident entry is rebuilt at the next run (so
        the weights constructor's refusals run on the new bytes)."""
        if self.adopted or len(self.w_src) == 0:
            raise Error("transformer: this step-op arm keeps no weight key")
        var src = _weight_sources(a, len(self.sizes))
        for i in range(len(self.sizes)):  # small-loop(sizes: the twenty weight pieces): address checks, no data
            if (self.sizes[i] != 0) != (src[i] != 0):
                raise Error("transformer options: weight piece " + String(i) + " presence differs from the session's")
        var now = weight_epoch()
        if now == self.epoch and src == self.w_src:
            return
        self.sc.forget(Int(self.w))
        self.epoch = now
        self.w_src = src.copy()
        self._upload_weights(src)

    # ------------------------------------------------------------ runs
    def _run(mut self, l: Int, forward_only: Bool) raises:
        """`l` tokens per row from `xio` to `yio` through the op, on the
        kept stages."""
        var n = self.b * l * self.dm
        attn_step_run_kept(self.ex, self.sc, self._operands(self.xio), self._counts(n), self.ints, self.yio,
                           self.pre, forward_only)
        self.cached += l

    def run(mut self, px: Int, py: Int, l: Int) raises -> Int:
        """`l` tokens per row through the op (L = 1 a decode step, L > 1 a
        prefill or a chunked continuation): x up, one op, y down, one wait."""
        if l < 1:
            raise Error("transformer decode session: L must be positive")
        var n = self.b * l * self.dm
        self._io(n)
        var ctx = self.ex.ctx.copy()
        ctx.enqueue_copy(dst_buf=self._view(self.xio, n), src_ptr=f32_ptr(px))
        self._run(l, False)
        ctx.enqueue_copy(dst_ptr=f32_ptr(py), src_buf=self._view(self.yio, n))
        self.ex.sync()
        return self.cached

    def run_device(mut self, x: DeviceBuffer[DType.float32], l: Int, mut shared: AttnPrefill) raises:
        """A CausalLM layer: `x` (b * l * d_model on this context, the
        stack's buffer) through the op in place; L > 1 on the stack's one
        prefill workspace `shared` (reused between compatible layers).
        Queued only; the stack waits."""
        if l < 1:
            raise Error("causal lm session: L must be positive")
        var n = self.b * l * self.dm
        self._io(n)
        var ctx = self.ex.ctx.copy()
        var xv = self._view(self.xio, n)
        var yv = self._view(self.yio, n)
        ctx.enqueue_copy(dst_buf=xv, src_buf=x.create_sub_buffer[DType.float32](0, n))
        attn_step_run_kept(self.ex, self.sc, self._operands(self.xio), self._counts(n), self.ints, self.yio,
                           shared, False)
        ctx.enqueue_copy(dst_buf=x.create_sub_buffer[DType.float32](0, n), src_buf=yv)
        self.cached += l

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
        self._io(row)
        try:
            var ctx = self.ex.ctx.copy()
            var xs_v = self._view(xs, n)
            var ys_v = self._view(ys, n)
            var xt_v = self._view(self.xio, row)
            var yt_v = self._view(self.yio, row)
            ctx.enqueue_copy(dst_buf=xs_v, src_ptr=f32_ptr(px))
            for t in range(steps):  # driver-loop: over autoregressive decode steps (each reads the cache the previous step left), the step op once each
                decode_slab_take(ctx, xt_v, xs_v, self.b, steps, self.dm, t)
                self._run(1, False)
                decode_slab_put(ctx, ys_v, yt_v, self.b, steps, self.dm, t)
            ctx.enqueue_copy(dst_ptr=f32_ptr(py), src_buf=ys_v)
            self.ex.sync()
        except err:
            self.ex.free(xs)
            self.ex.free(ys)
            raise err
        self.ex.free(xs)
        self.ex.free(ys)
        return self.cached

    def export(mut self, pk: Int, pv: Int) raises -> Int:
        """Both caches into the caller's buffers, one wait."""
        self._caches_out(pk, pv)
        self.ex.sync()
        return self.cached

    def load(mut self, pk: Int, pv: Int, s0: Int) raises -> Int:
        self._load(pk, pv, s0)
        self.ex.sync()
        return s0

    def k_view(self) raises -> DeviceBuffer[DType.float32]:
        """The key cache on the device (the CausalLM sampling checkpoint
        copies it)."""
        return self._view(self.kv, self.cache_n)

    def v_view(self) raises -> DeviceBuffer[DType.float32]:
        return self._view(self.kv + self.v_at, self.cache_n)

    # ------------------------------------------------------------ the block
    def block_forward(mut self, a: List[Int], ints: List[Int], l: Int, s0: Int, stateless: Bool) raises -> Int:
        """`TransformerBlock.forward` / `.step` and a CausalLM layer's
        per-call forward on the block session (`a` the `transformer_forward`
        13 + 11 layout). Stateless: zero caches at position 0 on the device,
        nothing of them crosses the host, `forward_only` (no backward reads
        these stages). Otherwise the caller's caches up at s0, the run, the
        caches back down (the host arrays are the state). One wait."""
        self._shape(ints)
        self.refresh_weights(a)
        var n = self.b * l * self.dm
        self._io(n)
        var ctx = self.ex.ctx.copy()
        if stateless:
            if s0 != 0:
                raise Error("transformer: a discarded cache starts at cached_tokens 0")
            self._caches_in(0, 0)
            attn_step_restart(self.ex, self.sc, self._operands(self.w), self._counts(1), self.ints, 0, False)
            self.cached = 0
        else:
            self._load(a[10], a[11], s0)
        ctx.enqueue_copy(dst_buf=self._view(self.xio, n), src_ptr=f32_ptr(a[0]))
        # forward-only either way: no backward reads a forward call's stages
        self._run(l, True)
        ctx.enqueue_copy(dst_ptr=f32_ptr(a[12]), src_buf=self._view(self.yio, n))
        if not stateless:
            self._caches_out(a[10], a[11])
        self.ex.sync()
        return self.cached

    def block_backward(mut self, a: List[Int], ints: List[Int], l: Int) raises:
        """`TransformerBlock.backward`: the zero-state prefill at positions
        [0, L) through the op (stages kept, not forward-only), then the
        transformer stage rule (`llama_decoder_layer_backward_autograd`,
        autograd/rules_transformer.mojo) over the op's kept stages, its
        weights, rotary table and input. `a` is the backward's 21 addresses:
        x, the nine weights, grad_output, d_x and the nine weight gradients.
        One wait."""
        self._shape(ints)
        var fa = List[Int](length=13 + BLOCK_OPTION_ADDRS, fill=0)
        fa[0] = a[0]
        for i in range(1, 10):  # small-loop(9: the base weight slots): argument words, no data
            fa[i] = a[i]
        self.refresh_weights(fa)
        var b = self.b
        var dm = self.dm
        var n = b * l * dm
        self._io(n)
        var ctx = self.ex.ctx.copy()
        self._caches_in(0, 0)
        attn_step_restart(self.ex, self.sc, self._operands(self.w), self._counts(1), self.ints, 0, False)
        self.cached = 0
        ctx.enqueue_copy(dst_buf=self._view(self.xio, n), src_ptr=f32_ptr(a[0]))
        self._run(l, False)
        var dims = LlamaDims(ints[1], ints[2], ints[3], ints[4], ints[5])
        var e = attn_step_entry(self.ex, self.sc, self._operands(self.xio), self._counts(n), self.ints)
        # the forward stages' own choice (the op's `_lean_for`; default
        # options, so `transformer_lean_stages(head_dim)` as before)
        var lean = self.sc.attn[e].lean
        var key: List[Int] = [b, l, ints[1], ints[2], ints[3], ints[4], ints[5], ints[7], Int(lean)]
        if not self.bst or not (key == self.bkey):
            self.bst = None
            self.dout = None
            self.bst = LlamaBackwardStages(ctx, b, l, l, dims, lean=lean)
            self.dout = ctx.enqueue_create_buffer[DType.float32](n)
            self.bkey = key^
        ctx.enqueue_copy(dst_buf=self.dout.value(), src_ptr=f32_ptr(a[10]))
        ref blk = self.sc.attn[e]
        ref bst = self.bst.value()
        var off = IdentityTrace.disabled()
        if l == 1:
            # L = 1 ran on the entry's resident stages and input buffer.
            llama_decoder_layer_backward_autograd(
                ctx, bst, blk.stages, blk.w, blk.rope.cos, blk.rope.sin, blk.dx,
                self.dout.value(), b, l, 0, off, String("pyb"))
        else:
            llama_decoder_layer_backward_autograd(
                ctx, bst, self.pre.stages.value(), blk.w, blk.rope.cos, blk.rope.sin, self.pre.dx.value(),
                self.dout.value(), b, l, 0, off, String("pyb"))
        var qw = dims.q_width()
        var kw = dims.kv_width()
        var it = ints[5]
        _down(ctx, bst.d_x, n, a[11])
        _down(ctx, bst.dw_norm1, dm, a[12])
        _down(ctx, bst.dw_norm2, dm, a[13])
        _down(ctx, bst.dw_q, qw * dm, a[14])
        _down(ctx, bst.dw_k, kw * dm, a[15])
        _down(ctx, bst.dw_v, kw * dm, a[16])
        _down(ctx, bst.dw_o, dm * qw, a[17])
        _down(ctx, bst.dw_gate, it * dm, a[18])
        _down(ctx, bst.dw_up, it * dm, a[19])
        _down(ctx, bst.dw_down, dm * it, a[20])
        self.ex.sync()


def _down(ctx: DeviceContext, buf: DeviceBuffer[DType.float32], n: Int, addr: Int) raises:
    """The first n floats of a device buffer into a caller buffer (queued)."""
    ctx.enqueue_copy(dst_ptr=f32_ptr(addr), src_buf=buf.create_sub_buffer[DType.float32](0, n))


# TOMBSTONE: _host_copy / _same_bits (REFACTORED: the per-call byte compare of the caller's weights against a host copy, ~7 MB per call) deleted 2026-11-09 by lane/x45c-causal-generate-program-decode; code recoverable at 5b5490960d.
# The arm keys its weights by caller address and the process weight epoch (core/weight_epoch.mojo).
# Restore: git apply experiments/removed/WEIGHT_BYTE_COMPARE.patch; record in docs/TOMBSTONES.md.


def _exec_view(ex: DeviceExec, p: FP, n: Int) raises -> DeviceBuffer[DType.float32]:
    """The executor's storage [p, p + n) as a device buffer (no copy)."""
    var found = ex._find(p, n)
    return ex._sub(found[0], found[1], n)
