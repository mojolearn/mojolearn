# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device-resident Samba forward and train step (lane S1 samba-resident,
2026-10-07; `-D MOJOLEARN_IDN_SAMBA_RESIDENT_STEP`, IDENTICAL only, default
off; `training/neural_identical_experiments.mojo::IDN_SAMBA_RESIDENT_STEP`).

WHAT ARM B PAYS. `python/mojolearn/_samba_impl.py` drives the stack layer by
layer from Python: the embedding, each Mamba-3 or attention block, the final
norm, the head and the loss are separate binding calls into three bindings,
and every one of them uploads its operands, computes, downloads its outputs
and waits (`training/samba_ops.mojo`, the block sessions). The gradients come
back as host arrays and the optimizer uploads them again. At the board shape
that is more than seven PCIe round trips and device drains per layer-op
family, times four layers, forward and backward.

WHAT THIS IS. One `SambaResidentSession` per stack holds the flat registry
(`param`), the registry-shaped gradient (`grad`), per-layer weight VIEWS into
`param` (no copy; the blocks read the registry directly), the per-layer
forward stages and backward stages, and the activation buffers between
blocks. `samba_resident_train_step` runs embedding -> blocks -> final norm ->
head -> loss -> head backward -> norm backward -> blocks backward ->
embedding backward -> AdamW on the device from ONE binding call; what crosses
the bus is the registry up (once), the token ids and targets up, the loss
word down and the registry down. The optimizer's moments are the Python
optimizer's own resident pair (`optimizer_resident_open`), passed in.

NO NEW ARITHMETIC, SAME BITS. Every launch below is the function arm B calls
for the same stage, on the same operands, in the same order:
`identical_embedding_forward_into`, `mamba3_block_forward` /
`mamba3_prefill_backward_on`, `llama_decoder_layer_forward` /
`llama_decoder_layer_backward_device`, `llama_rms_norm` / `bwd_rms_norm`,
the `OP_NT` head GEMM and its two backward GEMMs at `ROLE_HEAD`,
`identical_ce_loss_resident`, `identical_embedding_backward_into`, the
clause 9.2 pair add of the tied gradient, `identical_optimizer_step` through
`identical_optimizer_step_resident_io`. What changes is residency: the
operands never leave the device between those calls. Where arm B recomputes
a forward inside a backward (the final norm's row statistics, a block whose
session lost its stages) the recompute is the same kernel on the same input,
so reusing the forward's stages here is the same bits.

WHERE ONE-SYNC CANNOT HOLD (recorded, not hidden). Refusals read a device
word back: the token-range refusal inside `identical_embedding_forward_into`
(`emb_refuse_device_ids`), the whole-registry non-finite scan at the top of
each call (which stands in for arm B's per-weight scans and names a flat
index instead of a tensor), the per-call x/state scans inside
`mamba3_block_forward`, the logits non-finite refusal before the loss, and
`identical_ce_loss_resident`'s own loss download. `mamba3_prefill_backward_on`
and the optimizer entry also wait before they return (their own contracts).
None of those is a PCIe transfer of an activation or a gradient; every such
transfer is gone.

HOST WORK THAT REMAINS is arm B's own and is Mojo: the targets range walk and
the ignore-index count (`ce_refuse_targets`, `ce_count`) over the caller's
int32 targets before any device work, as `samba_head_loss_host` does.
"""

from std.os import getenv
from std.time import perf_counter_ns
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from core.device_scan import device_classify_nonfinite, device_first_nonfinite
from core.identity_trace import IdentityTrace
from embedding.checks.embedding_identical import (
    identical_embedding_backward_into,
    identical_embedding_forward_into,
)
from embedding.checks.embedding_oracle import EmbConfig
from gemm.contract import OP_NT
from gemm.experiments.neural_switches import ROLE_HEAD
from gemm.neural_backward import (
    identical_gemm_backward_a_into,
    identical_gemm_backward_a_workspace_max_floats,
    identical_gemm_backward_b_into,
    identical_gemm_backward_b_workspace_max_floats,
)
from gemm.neural_dispatch import (
    identical_gemm_into,
    identical_gemm_workspace_max_floats,
)
from mamba.checks.mamba3_fixture import M3_D_STATE, Mamba3Dims
from mamba.impl.modules.mamba3 import (
    Mamba3DeviceStages,
    Mamba3DeviceState,
    Mamba3DeviceWeights,
    mamba3_block_forward,
)
from mamba.impl.modules.mamba3_prefill_backward import mamba3_prefill_backward_on
from training.checks.loss_contract import (
    CeConfig,
    IGNORE_INDEX_DEFAULT,
    REDUCTION_SUM,
    ce_count,
    ce_nonfinite_message,
    ce_refuse_shape,
    ce_refuse_targets,
)
from training.estimator import (
    identical_ce_admit_call,
    identical_ce_loss_resident,
    identical_optimizer_step_resident_io,
)
from training.neural_ab_pointwise import NN58_ACCUMULATE_STATUS
from training.neural_ab_shards import NN63_CANONICAL_SHARD_MERGE, nn_shard_merge_into
from training.neural_identical_experiments import IDN_SAMBA_RESIDENT_STEP
from training.samba_ops import (
    SAMBA_TPB,
    _grid,
    samba_leaf_ids_kernel,
    samba_tree_level_kernel,
    samba_tree_status_kernel,
)
from transformer.block_options import BlockOptions
from transformer.checks.transformer_backward import (
    LlamaBackwardStages,
    bwd_rms_norm,
    llama_decoder_layer_backward_device,
)
from transformer.checks.transformer_fixture import RMS_EPS
from transformer.impl.llama.fused_attention import fused_forward_supported_head_dim
from transformer.impl.llama.modeling_llama import (
    ATTN_PATH_EAGER,
    PLANT_AT_NONE,
    LlamaDeviceStages,
    LlamaDeviceWeights,
    LlamaDims,
    LlamaKVCache,
    LlamaRopeTable,
    attention_path_choice,
    llama_decoder_layer_forward,
    llama_rms_norm,
)


#: The layer kinds, as `python/mojolearn/_samba_impl.py` codes them for
#: `samba_resident_open` (SambaConfig.layers: "mamba3" -> 0, "attention" -> 1).
comptime SAMBA_LAYER_MAMBA3 = 0
comptime SAMBA_LAYER_ATTENTION = 1
#: Every block kind has nine registry tensors (Mamba3Block._W_NAMES and
#: TransformerBlock._W_NAMES are both nine long); the registry is
#: embed.weight, then nine per layer, then norm_f.weight, then lm_head.weight
#: when the embedding is untied (`SambaConfig.registry`).
comptime SAMBA_BLOCK_TENSORS = 9


def _lean_stages(hd: Int) -> Bool:
    """`bindings/_mojolearn_transformer.mojo::transformer_lean_stages` for a
    default-options block: the attention stage buffers may be skipped when
    the fused forward will be attempted."""
    if not fused_forward_supported_head_dim(hd):
        return False
    return attention_path_choice(PLANT_AT_NONE) != ATTN_PATH_EAGER


def _d2d(
    ctx: DeviceContext,
    dst: DeviceBuffer[DType.float32],
    dst_at: Int,
    src: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """`dst[dst_at : dst_at + n] = src[0 : n]`, device to device, in stream
    order (no wait). A copy of bits, not arithmetic."""
    if n <= 0:
        return
    ctx.enqueue_copy(
        dst_buf=dst.create_sub_buffer[DType.float32](dst_at, n),
        src_buf=src.create_sub_buffer[DType.float32](0, n),
    )


def _pair_add_device(
    ctx: DeviceContext,
    mut out: DeviceBuffer[DType.float32],
    mut src: DeviceBuffer[DType.float32],
    n: Int,
) raises:
    """`out[n] = ftz(ftz(src[0:n]) + ftz(src[n:2n]))`: the clause 9.2 tree of
    TWO pieces, which is what `SambaStack.loss_and_grads` asks of
    `accumulate_grads([d_emb, dw_head], tokens=None)` for the tied embedding
    gradient. Launch for launch `training/samba_ops.mojo::samba_accumulate_buffer`
    at `a == 2` (one level, `pairs == 1`), including its NN63 and NN58 arms,
    with the result left on the device instead of copied to a host pointer."""
    comptime if NN63_CANONICAL_SHARD_MERGE:
        var ids = ctx.enqueue_create_buffer[DType.int32](2)
        ctx.enqueue_function[samba_leaf_ids_kernel](
            ids.unsafe_ptr(), Int32(2),
            grid_dim=(_grid(2), 1, 1), block_dim=(SAMBA_TPB, 1, 1),
        )
        var first = ctx.enqueue_create_buffer[DType.float32](2 * n)
        var second = ctx.enqueue_create_buffer[DType.float32](n)
        nn_shard_merge_into(ctx, out, src, ids, first, second, 2, n)
        comptime if NN58_ACCUMULATE_STATUS:
            var hit = device_first_nonfinite(ctx, out, n)
            if hit >= 0:
                raise Error("mojolearn samba ops: nonfinite accumulated gradient at " + String(hit))
        _ = ids^
        _ = first^
        _ = second^
        return
    comptime if NN58_ACCUMULATE_STATUS:
        var status = ctx.enqueue_create_buffer[DType.int32](1)
        status.enqueue_fill(Int32(n))
        ctx.enqueue_function[samba_tree_status_kernel](
            out.unsafe_ptr(), src.unsafe_ptr(), status.unsafe_ptr(), Int32(n), Int32(1),
            grid_dim=(_grid(n), 1, 1), block_dim=(SAMBA_TPB, 1, 1),
        )
        var host_status = ctx.enqueue_create_host_buffer[DType.int32](1)
        ctx.enqueue_copy(dst_ptr=host_status.unsafe_ptr(), src_buf=status)
        ctx.synchronize()
        if host_status[0] < Int32(n):
            raise Error("mojolearn samba ops: nonfinite accumulated gradient at " + String(host_status[0]))
        _ = status^
        _ = host_status^
    else:
        ctx.enqueue_function[samba_tree_level_kernel](
            out.unsafe_ptr(), src.unsafe_ptr(), Int32(n), Int32(1),
            grid_dim=(_grid(n), 1, 1), block_dim=(SAMBA_TPB, 1, 1),
        )


struct SambaResidentSession(Movable, Writable):
    """One stack's device residency: the registry and its gradient, the
    per-layer weight views, and the shape-keyed workspaces (rebuilt when
    `(b, l)` changes). Owned by the Python `SambaStack` through the
    binding's `_SambaResidentSession` type; dropped with it."""

    var ctx: DeviceContext
    var vocab: Int
    var dm: Int
    var tie: Bool
    var eps: Float32
    var has_attn: Bool
    var ldims: LlamaDims
    var lean: Bool
    var kinds: List[Int]
    #: per layer: its index in `m3_*` (a Mamba-3 layer) or `t_*` (attention)
    var slot: List[Int]
    #: the registry offsets, J + 1 entries (`SambaConfig.registry` order)
    var offsets: List[Int]
    var n_total: Int
    var param: DeviceBuffer[DType.float32]
    var grad: DeviceBuffer[DType.float32]
    var m3_w: List[Mamba3DeviceWeights]
    var t_w: List[LlamaDeviceWeights]
    # ---- shape-keyed workspaces (b, l); 0 x 0 before the first call
    var b: Int
    var l: Int
    var ids: DeviceBuffer[DType.int32]
    var targets: DeviceBuffer[DType.int32]
    #: n_layers + 1 block inputs: acts[0] is the embedding, acts[i + 1] block i's output
    var acts: List[DeviceBuffer[DType.float32]]
    var m3_state: List[Mamba3DeviceState]
    var m3_stages: List[Mamba3DeviceStages]
    var rope: Optional[LlamaRopeTable]
    var kv: Optional[LlamaKVCache]
    var t_stages: List[LlamaDeviceStages]
    var t_bst: List[LlamaBackwardStages]
    var sumsq: DeviceBuffer[DType.float32]
    var hn: DeviceBuffer[DType.float32]
    var logits: DeviceBuffer[DType.float32]
    var head_ws: DeviceBuffer[DType.float32]
    var dlogits: DeviceBuffer[DType.float32]
    var dhn: DeviceBuffer[DType.float32]
    var ws_a: DeviceBuffer[DType.float32]
    var ws_b: DeviceBuffer[DType.float32]
    var dot_out: DeviceBuffer[DType.float32]
    #: the running activation gradient (d loss / d block output), [M, dm]
    var dh: DeviceBuffer[DType.float32]
    var dh_tmp: DeviceBuffer[DType.float32]
    var dprod: DeviceBuffer[DType.float32]
    var rstd: DeviceBuffer[DType.float32]
    var dvcoef: DeviceBuffer[DType.float32]
    var ones: DeviceBuffer[DType.float32]
    #: the tied pair [d_emb | dw_head], 2 * vocab * dm floats (1 when untied)
    var pair: DeviceBuffer[DType.float32]
    var counts: DeviceBuffer[DType.int32]
    var run_begin: DeviceBuffer[DType.int32]
    var perm: DeviceBuffer[DType.int32]
    var busy: Bool

    def __init__(
        out self,
        ctx: DeviceContext,
        kinds: List[Int],
        vocab: Int,
        dm: Int,
        nh: Int,
        nkv: Int,
        hd: Int,
        it: Int,
        tie: Bool,
        eps: Float32,
        n_total: Int,
        param_ptr: MutPointer[Float32, MutUntrackedOrigin],
    ) raises:
        """The registry on the device and the weight views. `param_ptr` is the
        stack's flat float32 registry, uploaded once here so the attention
        weight views can run their constructor's own finiteness refusal
        (`LlamaDeviceWeights._validate_finite`); every later call uploads
        the registry again and scans it whole."""
        if vocab < 2 or dm < 1 or len(kinds) < 1:
            raise Error("mojolearn samba resident: vocab >= 2, d_model >= 1 and at least one layer are required")
        self.ctx = ctx.copy()
        self.vocab = vocab
        self.dm = dm
        self.tie = tie
        self.eps = eps
        self.kinds = kinds.copy()
        self.has_attn = False
        for i in range(len(kinds)):  # small-loop(kinds: one code per layer): layer kinds, not data
            if kinds[i] == SAMBA_LAYER_ATTENTION:
                self.has_attn = True
            elif kinds[i] != SAMBA_LAYER_MAMBA3:
                raise Error("mojolearn samba resident: layer kind code " + String(kinds[i]) + " is neither 0 (mamba3) nor 1 (attention)")
        self.ldims = LlamaDims(dm, max(nh, 1), max(nkv, 1), max(hd, 1), max(it, 1))
        if self.has_attn:
            self.ldims = LlamaDims(dm, nh, nkv, hd, it)
            self.ldims.validate()
        self.lean = _lean_stages(hd) if self.has_attn else False
        # ---- the registry offsets, SambaConfig.registry's order
        self.offsets = List[Int]()
        self.offsets.append(0)
        self.slot = List[Int]()
        var n_m3 = 0
        var n_t = 0
        var running = vocab * dm
        self.offsets.append(running)
        for i in range(len(kinds)):  # small-loop(kinds: one code per layer): registry bookkeeping, not data
            if kinds[i] == SAMBA_LAYER_MAMBA3:
                var d3 = Mamba3Dims.of(dm)
                var dip = d3.d_in_proj()
                var sizes: List[Int] = [
                    dm, dip * dm, d3.nheads, M3_D_STATE, M3_D_STATE,
                    d3.nheads * M3_D_STATE, d3.nheads * M3_D_STATE, d3.nheads,
                    dm * d3.d_inner,
                ]
                for k in range(SAMBA_BLOCK_TENSORS):  # small-loop(nine tensors of one block): sizes, not data
                    running += sizes[k]
                    self.offsets.append(running)
                self.slot.append(n_m3)
                n_m3 += 1
            else:
                var qw = self.ldims.q_width()
                var kw = self.ldims.kv_width()
                var sizes: List[Int] = [
                    dm, dm, qw * dm, kw * dm, kw * dm, dm * qw, it * dm, it * dm, dm * it,
                ]
                for k in range(SAMBA_BLOCK_TENSORS):  # small-loop(nine tensors of one block): sizes, not data
                    running += sizes[k]
                    self.offsets.append(running)
                self.slot.append(n_t)
                n_t += 1
        running += dm
        self.offsets.append(running)
        if not tie:
            running += vocab * dm
            self.offsets.append(running)
        self.n_total = running
        if n_total != running:
            raise Error(
                "mojolearn samba resident: the stack's registry holds " + String(n_total)
                + " floats, the configuration names " + String(running)
            )
        # ---- the registry and its gradient on the device
        self.param = ctx.enqueue_create_buffer[DType.float32](running)
        self.grad = ctx.enqueue_create_buffer[DType.float32](running)
        self.grad.enqueue_fill(Float32(0.0))
        ctx.enqueue_copy(dst_buf=self.param, src_ptr=param_ptr)
        ctx.synchronize()
        # ---- placeholders for the shape-keyed workspaces (rebuilt on first use)
        self.b = 0
        self.l = 0
        self.ids = ctx.enqueue_create_buffer[DType.int32](1)
        self.targets = ctx.enqueue_create_buffer[DType.int32](1)
        self.acts = List[DeviceBuffer[DType.float32]]()
        self.m3_state = List[Mamba3DeviceState]()
        self.m3_stages = List[Mamba3DeviceStages]()
        self.rope = None
        self.kv = None
        self.t_stages = List[LlamaDeviceStages]()
        self.t_bst = List[LlamaBackwardStages]()
        self.sumsq = ctx.enqueue_create_buffer[DType.float32](1)
        self.hn = ctx.enqueue_create_buffer[DType.float32](1)
        self.logits = ctx.enqueue_create_buffer[DType.float32](1)
        self.head_ws = ctx.enqueue_create_buffer[DType.float32](1)
        self.dlogits = ctx.enqueue_create_buffer[DType.float32](1)
        self.dhn = ctx.enqueue_create_buffer[DType.float32](1)
        self.ws_a = ctx.enqueue_create_buffer[DType.float32](1)
        self.ws_b = ctx.enqueue_create_buffer[DType.float32](1)
        self.dot_out = ctx.enqueue_create_buffer[DType.float32](1)
        self.dh = ctx.enqueue_create_buffer[DType.float32](1)
        self.dh_tmp = ctx.enqueue_create_buffer[DType.float32](1)
        self.dprod = ctx.enqueue_create_buffer[DType.float32](1)
        self.rstd = ctx.enqueue_create_buffer[DType.float32](1)
        self.dvcoef = ctx.enqueue_create_buffer[DType.float32](1)
        self.ones = ctx.enqueue_create_buffer[DType.float32](1)
        self.pair = ctx.enqueue_create_buffer[DType.float32](1)
        self.counts = ctx.enqueue_create_buffer[DType.int32](1)
        self.run_begin = ctx.enqueue_create_buffer[DType.int32](1)
        self.perm = ctx.enqueue_create_buffer[DType.int32](1)
        self.busy = False
        # ---- the weight views: the blocks read the registry in place
        self.m3_w = List[Mamba3DeviceWeights]()
        self.t_w = List[LlamaDeviceWeights]()
        for i in range(len(kinds)):  # small-loop(kinds: one code per layer): builds views, not data
            var base = 1 + SAMBA_BLOCK_TENSORS * i
            if kinds[i] == SAMBA_LAYER_MAMBA3:
                var d3 = Mamba3Dims.of(dm)
                self.m3_w.append(
                    Mamba3DeviceWeights(
                        d3,
                        self.view(base + 0), self.view(base + 1), self.view(base + 2),
                        self.view(base + 3), self.view(base + 4), self.view(base + 5),
                        self.view(base + 6), self.view(base + 7), self.view(base + 8),
                    )
                )
            else:
                # The default-options constructor (`_load_transformer_weights`
                # at the default record): RMS_EPS, the nine tensors in
                # TransformerBlock._W_NAMES order; it scans them once here.
                self.t_w.append(
                    LlamaDeviceWeights(
                        ctx, self.ldims, RMS_EPS,
                        self.view(base + 0), self.view(base + 1), self.view(base + 2),
                        self.view(base + 3), self.view(base + 4), self.view(base + 5),
                        self.view(base + 6), self.view(base + 7), self.view(base + 8),
                    )
                )

    def write_to(self, mut writer: Some[Writer]):
        writer.write("SambaResidentSession")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("SambaResidentSession")

    def view(self, j: Int) -> DeviceBuffer[DType.float32]:
        """Registry tensor `j` as a view of `param`."""
        return self.param.create_sub_buffer[DType.float32](
            self.offsets[j], self.offsets[j + 1] - self.offsets[j]
        )

    def grad_view(self, j: Int) -> DeviceBuffer[DType.float32]:
        """Registry tensor `j` as a view of `grad`."""
        return self.grad.create_sub_buffer[DType.float32](
            self.offsets[j], self.offsets[j + 1] - self.offsets[j]
        )

    def n_layers(self) -> Int:
        return len(self.kinds)

    def norm_f_index(self) -> Int:
        return 1 + SAMBA_BLOCK_TENSORS * len(self.kinds)

    def head_index(self) -> Int:
        """The head's registry tensor: `embed.weight` when tied, else `lm_head.weight`."""
        if self.tie:
            return 0
        return self.norm_f_index() + 1

    def tensor_size(self, j: Int) -> Int:
        return self.offsets[j + 1] - self.offsets[j]


def _ensure_shape(mut s: SambaResidentSession, b: Int, l: Int) raises:
    """(Re)build every `(b, l)`-shaped buffer when the call's shape differs
    from the held one. The blocks' stage and state buffers are the structs
    arm B builds per call (`Mamba3DeviceStages(ctx, b, l, 0, dims)`,
    `Mamba3DeviceState(ctx, b, dims)`, `LlamaDeviceStages(ctx, b, l, l, dims,
    0, lean)`, `LlamaBackwardStages(ctx, b, l, l, dims, lean)`), the rope
    table and the zero cache at `smax = l` as the fresh prefill session has
    them. One rope table and one cache scratch serve every attention layer
    (training is a full prefill from position 0; the backward reads each
    layer's own recorded `k_cache` / `v_cache` stages, not this scratch),
    as `training/byte_lm.mojo` does."""
    if s.b == b and s.l == l:
        return
    if b < 1 or l < 1:
        raise Error("mojolearn samba resident: B and L must be positive")
    ref ctx = s.ctx
    ctx.synchronize()
    var m = b * l
    var dm = s.dm
    var v = s.vocab
    s.ids = ctx.enqueue_create_buffer[DType.int32](m)
    s.targets = ctx.enqueue_create_buffer[DType.int32](m)
    s.acts = List[DeviceBuffer[DType.float32]]()
    for _ in range(s.n_layers() + 1):
        s.acts.append(ctx.enqueue_create_buffer[DType.float32](m * dm))
    s.m3_state = List[Mamba3DeviceState]()
    s.m3_stages = List[Mamba3DeviceStages]()
    s.t_stages = List[LlamaDeviceStages]()
    s.t_bst = List[LlamaBackwardStages]()
    s.rope = None
    s.kv = None
    var opts = BlockOptions()
    if s.has_attn:
        s.rope = LlamaRopeTable(ctx, s.ldims, opts, l)
        s.kv = LlamaKVCache(ctx, b, s.ldims, l, 0, opts.max_positions)
    for i in range(s.n_layers()):
        if s.kinds[i] == SAMBA_LAYER_MAMBA3:
            var d3 = Mamba3Dims.of(dm)
            s.m3_state.append(Mamba3DeviceState(ctx, b, d3))
            s.m3_stages.append(Mamba3DeviceStages(ctx, b, l, 0, d3))
        else:
            s.t_stages.append(LlamaDeviceStages(ctx, b, l, l, s.ldims, 0, lean=s.lean))
            s.t_bst.append(LlamaBackwardStages(ctx, b, l, l, s.ldims, lean=s.lean))
    s.sumsq = ctx.enqueue_create_buffer[DType.float32](m)
    s.hn = ctx.enqueue_create_buffer[DType.float32](m * dm)
    s.logits = ctx.enqueue_create_buffer[DType.float32](m * v)
    s.head_ws = ctx.enqueue_create_buffer[DType.float32](
        identical_gemm_workspace_max_floats(m, v, dm)
    )
    s.dlogits = ctx.enqueue_create_buffer[DType.float32](m * v)
    s.dhn = ctx.enqueue_create_buffer[DType.float32](m * dm)
    s.ws_a = ctx.enqueue_create_buffer[DType.float32](
        identical_gemm_backward_a_workspace_max_floats(OP_NT, m, v, dm)
    )
    s.ws_b = ctx.enqueue_create_buffer[DType.float32](
        identical_gemm_backward_b_workspace_max_floats(OP_NT, m, v, dm)
    )
    s.dot_out = ctx.enqueue_create_buffer[DType.float32](m)
    s.dh = ctx.enqueue_create_buffer[DType.float32](m * dm)
    s.dh_tmp = ctx.enqueue_create_buffer[DType.float32](m * dm)
    s.dprod = ctx.enqueue_create_buffer[DType.float32](m * dm)
    s.rstd = ctx.enqueue_create_buffer[DType.float32](m)
    s.dvcoef = ctx.enqueue_create_buffer[DType.float32](m)
    # The ones vector `bwd_rms_norm` folds with: exactly 1.0f in every cell,
    # the bits arm B's host loop wrote and uploaded.
    s.ones = ctx.enqueue_create_buffer[DType.float32](m)
    s.ones.enqueue_fill(Float32(1.0))
    s.pair = ctx.enqueue_create_buffer[DType.float32](2 * v * dm if s.tie else 1)
    s.counts = ctx.enqueue_create_buffer[DType.int32](v)
    s.run_begin = ctx.enqueue_create_buffer[DType.int32](v + 1)
    s.perm = ctx.enqueue_create_buffer[DType.int32](m)
    ctx.synchronize()
    s.b = b
    s.l = l


def _admit_registry(mut s: SambaResidentSession) raises:
    """Arm B refuses a non-finite weight by name at every op (`_refuse_host`
    / `_upload_checked`, `mamba3_refuse_bad_inputs`, `LlamaDeviceWeights`).
    Here the whole registry is scanned once per call after its upload (one
    device read, one word back) and a hit names the flat index; the
    Mamba-3 weight views are then marked checked for this call (their
    per-weight scans would re-read the same admitted bytes), while the
    attention views were scanned by their constructor at open."""
    var idx = device_first_nonfinite(s.ctx, s.param, s.n_total)
    if idx >= 0:
        raise Error(
            "mojolearn samba resident: non-finite parameter at flat registry index "
            + String(idx)
        )
    for j in range(len(s.m3_w)):
        s.m3_w[j].weights_checked = True


def _forward_blocks(mut s: SambaResidentSession, b: Int, l: Int, forward_only: Bool) raises:
    """Embedding output in `acts[0]` -> `acts[i + 1] = block_i(acts[i])` for
    every layer, in stack order, each from its certified zero state."""
    var m = b * l
    var dm = s.dm
    var trace = IdentityTrace.disabled()
    for i in range(s.n_layers()):
        var j = s.slot[i]
        if s.kinds[i] == SAMBA_LAYER_MAMBA3:
            # The certified zero state and zeroed stages per call, as the
            # fresh prefill builds them (`_m3_prefill_run`).
            s.m3_state[j].rezero()
            s.m3_stages[j].rezero()
            mamba3_block_forward(
                s.ctx, s.m3_stages[j], s.m3_state[j], s.m3_w[j], s.acts[i], b, l,
                trace, String("samba.resident.m3"),
            )
            _d2d(s.ctx, s.acts[i + 1], 0, s.m3_stages[j].residual_out, m * dm)
        else:
            # The bytes a fresh `LlamaKVCache` holds and constructor-zero
            # stages, as the session's stateless prefill restores them.
            s.kv.value().s = 0
            s.kv.value().k.enqueue_fill(Float32(0.0))
            s.kv.value().v.enqueue_fill(Float32(0.0))
            s.t_stages[j].reset(s.ctx)
            llama_decoder_layer_forward(
                s.ctx, s.t_stages[j], s.kv.value(), s.rope.value(), s.t_w[j], s.acts[i],
                b, l, 0, trace, String("samba.resident.attn"), forward_only=forward_only,
            )
            _d2d(s.ctx, s.acts[i + 1], 0, s.t_stages[j].residual2, m * dm)


def _backward_blocks(mut s: SambaResidentSession, b: Int, l: Int) raises:
    """`dh` holds d loss / d(last block output) on entry; every block's VJP
    runs on the stages its forward recorded this call (the input is
    `acts[i]`), its nine weight gradients are copied into `grad` at their
    registry offsets and `dh` becomes d loss / d(block input)."""
    var m = b * l
    var dm = s.dm
    var trace = IdentityTrace.disabled()
    var ton = String(getenv("MOJOLEARN_MAMBA_TIMING")) != ""
    for ii in range(s.n_layers()):
        var i = s.n_layers() - 1 - ii
        var j = s.slot[i]
        var base = 1 + SAMBA_BLOCK_TENSORS * i
        if s.kinds[i] == SAMBA_LAYER_MAMBA3:
            var d3 = s.m3_w[j].dims.copy()
            var tk = Int(perf_counter_ns())
            var g = mamba3_prefill_backward_on(
                s.ctx, s.m3_w[j], s.m3_stages[j], s.acts[i], s.dh, b, l, d3, ton, tk
            )
            _d2d(s.ctx, s.grad, s.offsets[base + 0], g.block_norm_weight, s.tensor_size(base + 0))
            _d2d(s.ctx, s.grad, s.offsets[base + 1], g.in_proj_weight, s.tensor_size(base + 1))
            _d2d(s.ctx, s.grad, s.offsets[base + 2], g.dt_bias, s.tensor_size(base + 2))
            _d2d(s.ctx, s.grad, s.offsets[base + 3], g.B_norm_weight, s.tensor_size(base + 3))
            _d2d(s.ctx, s.grad, s.offsets[base + 4], g.C_norm_weight, s.tensor_size(base + 4))
            _d2d(s.ctx, s.grad, s.offsets[base + 5], g.B_bias, s.tensor_size(base + 5))
            _d2d(s.ctx, s.grad, s.offsets[base + 6], g.C_bias, s.tensor_size(base + 6))
            _d2d(s.ctx, s.grad, s.offsets[base + 7], g.D, s.tensor_size(base + 7))
            _d2d(s.ctx, s.grad, s.offsets[base + 8], g.out_proj_weight, s.tensor_size(base + 8))
            _d2d(s.ctx, s.dh, 0, g.x, m * dm)
            # the gradient buffers stay alive past the copies' completion
            s.ctx.synchronize()
            _ = g^
        else:
            llama_decoder_layer_backward_device(
                s.ctx, s.t_bst[j], s.t_stages[j], s.t_w[j], s.rope.value().cos, s.rope.value().sin,
                s.acts[i], s.dh, b, l, 0, trace, String("samba.resident.attn.bwd"),
            )
            _d2d(s.ctx, s.grad, s.offsets[base + 0], s.t_bst[j].dw_norm1, s.tensor_size(base + 0))
            _d2d(s.ctx, s.grad, s.offsets[base + 1], s.t_bst[j].dw_norm2, s.tensor_size(base + 1))
            _d2d(s.ctx, s.grad, s.offsets[base + 2], s.t_bst[j].dw_q, s.tensor_size(base + 2))
            _d2d(s.ctx, s.grad, s.offsets[base + 3], s.t_bst[j].dw_k, s.tensor_size(base + 3))
            _d2d(s.ctx, s.grad, s.offsets[base + 4], s.t_bst[j].dw_v, s.tensor_size(base + 4))
            _d2d(s.ctx, s.grad, s.offsets[base + 5], s.t_bst[j].dw_o, s.tensor_size(base + 5))
            _d2d(s.ctx, s.grad, s.offsets[base + 6], s.t_bst[j].dw_gate, s.tensor_size(base + 6))
            _d2d(s.ctx, s.grad, s.offsets[base + 7], s.t_bst[j].dw_up, s.tensor_size(base + 7))
            _d2d(s.ctx, s.grad, s.offsets[base + 8], s.t_bst[j].dw_down, s.tensor_size(base + 8))
            _d2d(s.ctx, s.dh, 0, s.t_bst[j].d_x, m * dm)


def _upload_tokens(
    mut s: SambaResidentSession,
    ids_ptr: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    """The `B * L` input ids into the held device buffer (no wait: the
    embedding gather is the next thing queued on the in-order context)."""
    s.ctx.enqueue_copy(dst_buf=s.ids, src_ptr=ids_ptr)


def _norm_head_forward(mut s: SambaResidentSession, m: Int, train: Bool) raises:
    """`hn = norm_f(acts[L])`, `logits = hn . head^T`. The train step's head
    GEMM runs at `ROLE_HEAD` as `samba_head_loss_host` does; the forward's
    at the default role as `samba_linear_forward_host` does (an execution
    plan tag, the same bits)."""
    var nw = s.view(s.norm_f_index())
    var hw = s.view(s.head_index())
    llama_rms_norm(s.ctx, s.sumsq, s.hn, s.acts[s.n_layers()], nw, m, s.dm, s.eps)
    if train:
        identical_gemm_into[ROLE=ROLE_HEAD](
            s.ctx, s.logits, s.hn, hw, s.head_ws, m, s.vocab, s.dm, OP_NT
        )
    else:
        identical_gemm_into(s.ctx, s.logits, s.hn, hw, s.head_ws, m, s.vocab, s.dm, OP_NT)
    _ = nw^
    _ = hw^


def samba_resident_forward(
    mut s: SambaResidentSession,
    param_ptr: MutPointer[Float32, MutUntrackedOrigin],
    ids_ptr: MutPointer[Int32, MutUntrackedOrigin],
    out_ptr: MutPointer[Float32, MutUntrackedOrigin],
    b: Int,
    l: Int,
) raises -> Int:
    """`SambaStack.forward(inputs)`: `(B, L)` ids -> `(B * L, vocab)` logits
    written to `out_ptr`, every block from its zero state, no dropout. The
    registry goes up once, the ids once, the logits come down once (the
    board's forward span includes that download, as for lm-forward).
    Returns `B * L * vocab`."""
    comptime if not IDN_SAMBA_RESIDENT_STEP:
        raise Error("mojolearn samba resident: built without MOJOLEARN_IDN_SAMBA_RESIDENT_STEP")
    _ensure_shape(s, b, l)
    var m = b * l
    s.ctx.enqueue_copy(dst_buf=s.param, src_ptr=param_ptr)
    _upload_tokens(s, ids_ptr)
    _admit_registry(s)
    # The token-range refusal is the embedding entry's own
    # (`emb_refuse_device_ids`): the ids are checked on the device before
    # the gather reads them, so Python makes no min/max pass over them.
    var emb = EmbConfig.llama(s.vocab, s.dm)
    var ew = s.view(0)
    identical_embedding_forward_into(s.ctx, s.acts[0], ew, s.ids, m, emb)
    _forward_blocks(s, b, l, True)
    _norm_head_forward(s, m, False)
    s.ctx.enqueue_copy(dst_ptr=out_ptr, src_buf=s.logits)
    s.ctx.synchronize()
    _ = ew^
    return m * s.vocab


def samba_resident_train_step(
    mut s: SambaResidentSession,
    param_ptr: MutPointer[Float32, MutUntrackedOrigin],
    ids_ptr: MutPointer[Int32, MutUntrackedOrigin],
    targets_ptr: MutPointer[Int32, MutUntrackedOrigin],
    loss_ptr: MutPointer[Float32, MutUntrackedOrigin],
    mut m_state: DeviceBuffer[DType.float32],
    mut v_state: DeviceBuffer[DType.float32],
    mut p_stage: HostBuffer[DType.float32],
    mut g_stage: HostBuffer[DType.float32],
    offsets_ptr: MutPointer[Int32, MutUntrackedOrigin],
    init_ptr: MutPointer[Int32, MutUntrackedOrigin],
    info_ptr: MutPointer[Float32, MutUntrackedOrigin],
    b: Int,
    l: Int,
    n_tensors: Int,
    kind: Int,
    t: Int,
    nesterov: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    momentum: Float32,
    dampening: Float32,
    max_norm: Float32,
) raises -> Int:
    """`SambaStack.train_step(inputs, targets)` at `accumulation_steps == 1`
    and no dropout: one microbatch, the sum-reduced loss divided by the
    target count (`num_items = count`, as `loss_and_grads` passes it), the
    full backward, then `identical_optimizer_step` over the registry with
    the clip at `max_norm` (<= 0 off) and the Python optimizer's resident
    moments `m_state` / `v_state`. The optimizer's `params` list is the
    `optimizer_step` binding's, slot for slot. Writes the registry back to
    `param_ptr`, the loss to `loss_ptr`, the flags to `init_ptr` and the
    clip info to `info_ptr`. Returns `count`."""
    comptime if not IDN_SAMBA_RESIDENT_STEP:
        raise Error("mojolearn samba resident: built without MOJOLEARN_IDN_SAMBA_RESIDENT_STEP")
    if b < 1 or l < 1:
        raise Error("mojolearn samba resident: B and L must be positive")
    var m = b * l
    # ---- the optimizer registry must be the stack's (its offsets are the
    # view table above); refused before any device work
    if n_tensors + 1 != len(s.offsets):
        raise Error(
            "mojolearn samba resident: the optimizer registry has " + String(n_tensors)
            + " tensors, the stack's has " + String(len(s.offsets) - 1)
        )
    for j in range(n_tensors + 1):  # small-loop(n_tensors + 1: registry offsets): compares offsets, not data
        if Int(offsets_ptr.unsafe_load(j)) != s.offsets[j]:
            raise Error("mojolearn samba resident: optimizer offsets[" + String(j) + "] is not the stack's registry")
    # ---- the targets: arm B's own host walk (`ce_refuse_targets`,
    # `ce_count`), before any device work; `count` is the divisor
    var h_targets = List[Int32](capacity=m)
    for i in range(m):
        h_targets.append(targets_ptr.unsafe_load(i))
    var refuse_cfg = CeConfig(s.vocab, IGNORE_INDEX_DEFAULT, REDUCTION_SUM, Float32(0.0), 0)
    ce_refuse_targets(h_targets, refuse_cfg)
    var count = ce_count(h_targets, IGNORE_INDEX_DEFAULT)
    _ = h_targets^
    identical_ce_admit_call(REDUCTION_SUM, 1, m)
    var cfg = CeConfig(s.vocab, IGNORE_INDEX_DEFAULT, REDUCTION_SUM, Float32(0.0), count)
    ce_refuse_shape(m, m * s.vocab, cfg)

    _ensure_shape(s, b, l)
    var dm = s.dm
    var v = s.vocab
    # ---- transport in: the registry, the ids, the targets
    s.ctx.enqueue_copy(dst_buf=s.param, src_ptr=param_ptr)
    _upload_tokens(s, ids_ptr)
    s.ctx.enqueue_copy(dst_buf=s.targets, src_ptr=targets_ptr)
    _admit_registry(s)

    # ---- forward
    var emb = EmbConfig.llama(v, dm)
    var ew = s.view(0)
    identical_embedding_forward_into(s.ctx, s.acts[0], ew, s.ids, m, emb)
    _forward_blocks(s, b, l, False)
    _norm_head_forward(s, m, True)

    # ---- the loss: the logits refusal (the oracle's walk, on the device),
    # then `identical_ce_loss_resident` with the gradient left on the device
    var bad = device_first_nonfinite(s.ctx, s.logits, m * v)
    if bad >= 0:
        var is_nan = device_classify_nonfinite(s.ctx, s.logits, bad)
        raise Error(ce_nonfinite_message("logits", bad, is_nan))
    var rows = List[Float32](length=max(m, 1), fill=Float32(0.0))
    var row_ptr = rows.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    identical_ce_loss_resident(
        s.ctx, loss_ptr, row_ptr, s.dlogits, s.logits, s.targets, m, count, REDUCTION_SUM, 1, cfg,
    )
    _ = rows^

    # ---- head backward: dhn = dlogits . W, dW = dlogits^T . hn
    var hw = s.view(s.head_index())
    identical_gemm_backward_a_into[ROLE_HEAD](
        s.ctx, s.dhn, s.dlogits, hw, s.ws_a, m, v, dm, OP_NT
    )
    if s.tie:
        var dwh = s.pair.create_sub_buffer[DType.float32](v * dm, v * dm)
        identical_gemm_backward_b_into(s.ctx, dwh, s.dlogits, s.hn, s.ws_b, m, v, dm, OP_NT)
        _ = dwh^
    else:
        var dwh = s.grad_view(s.head_index())
        identical_gemm_backward_b_into(s.ctx, dwh, s.dlogits, s.hn, s.ws_b, m, v, dm, OP_NT)
        _ = dwh^

    # ---- final norm backward on the forward's own row statistics (arm B
    # recomputes them with the same kernel on the same input)
    var nw = s.view(s.norm_f_index())
    var dnw = s.grad_view(s.norm_f_index())
    bwd_rms_norm[0](
        s.ctx, s.dot_out, s.dh, dnw, s.dh_tmp, s.dprod, s.rstd, s.dvcoef, s.ones, s.dhn,
        s.acts[s.n_layers()], nw, s.sumsq,
        s.dh.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
        s.dh.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](), False, m, dm, s.eps,
    )

    # ---- the blocks, last to first
    _backward_blocks(s, b, l)

    # ---- the embedding gradient (fresh, run-sorted fold) and the tied pair
    if s.tie:
        var demb = s.pair.create_sub_buffer[DType.float32](0, v * dm)
        identical_embedding_backward_into(
            s.ctx, demb, s.dh, s.ids, s.counts, s.run_begin, s.perm, m, emb
        )
        # ONE pair add, embedding first, no alignment claim (tokens=None):
        # `accumulate_grads([d_emb, dw_head], tokens=None)` in arm B.
        var gemb = s.grad_view(0)
        _pair_add_device(s.ctx, gemb, s.pair, v * dm)
        _ = demb^
        _ = gemb^
    else:
        var gemb = s.grad_view(0)
        identical_embedding_backward_into(
            s.ctx, gemb, s.dh, s.ids, s.counts, s.run_begin, s.perm, m, emb
        )
        _ = gemb^

    # ---- the optimizer: `identical_optimizer_step` on the resident registry
    # and gradient (`io = 0`: no transfer; `param_ptr` stands in for the
    # untouched gradient host pointer, which an io of 0 never reads or
    # writes), the moments the Python optimizer keeps on the device. The
    # entry waits before it returns and writes the flags and the clip info.
    _ = identical_optimizer_step_resident_io(
        s.ctx, param_ptr, param_ptr, m_state, v_state, s.param, s.grad, p_stage, g_stage,
        offsets_ptr, init_ptr, info_ptr, n_tensors, kind, t, nesterov, lr, beta1, beta2,
        eps, weight_decay, momentum, dampening, max_norm, 0,
    )
    # ---- transport out: the updated registry
    s.ctx.enqueue_copy(dst_ptr=param_ptr, src_buf=s.param)
    s.ctx.synchronize()
    _ = ew^
    _ = hw^
    _ = nw^
    _ = dnw^
    return count
