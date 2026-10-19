# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The device-resident Samba forward and train step (lane S1 samba-resident,
2026-10-07; IDENTICAL only; promoted to the IDENTICAL default 2026-10-08 by
lane grid-act-4 (grid ge123e6f9: samba-forward 0.539x, samba-train-step 0.563x,
same bits); `-D MOJOLEARN_IDN_SAMBA_RESIDENT_STEP_OFF` restores arm B;
`training/neural_identical_experiments.mojo::IDN_SAMBA_RESIDENT_STEP`).

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

THE TARGETS ARE ADMITTED ON THE DEVICE. Arm B walks the host targets for the
range refusal and the ignore-index count (`ce_refuse_targets`, `ce_count` in
`samba_head_loss_host`). Here `samba_targets_scan_kernel` reads the uploaded
targets once, in parallel: the count of rows whose target is not the ignore
index is an integer sum (exact and order-free, so integer atomics are the
whole fold) and the first out-of-range row is an integer minimum; two words
come back, and a hit is spelled with `ce_refuse_targets`'s own message.
"""

from std.atomic import Atomic
from std.gpu import block_dim, block_idx, thread_idx
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
from mamba.impl.modules.mamba3_prefill_backward import Mamba3DeviceGradients, mamba3_prefill_backward_on
from training.checks.loss_contract import (
    CeConfig,
    IGNORE_INDEX_DEFAULT,
    REDUCTION_SUM,
    ce_nonfinite_message,
    ce_refuse_shape,
)
from training.estimator import (
    identical_ce_admit_call,
    identical_ce_loss_resident,
    identical_optimizer_step_resident_io,
)
from training.neural_ab_pointwise import NN58_ACCUMULATE_STATUS
from training.neural_ab_shards import NN63_CANONICAL_SHARD_MERGE, nn_shard_merge_into
from training.neural_identical_experiments import IDN_SAMBA_RESIDENT_STEP, IDN_SAMBA_ATTN_RECOMPUTE
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
from autograd.rules_lm import AUTOGRAD_LM
from autograd.rules_transformer import (
    AUTOGRAD_REFUSE_DEFER,
    llama_decoder_layer_backward_autograd,
    llama_decoder_layer_record,
)
from autograd.tape import Tape, Tensor
from autograd.refuse_scan import DeferredNonfiniteScan
from autograd.rules_mamba3 import M3_BLOCK, M3KeepRef, M3StagesRef, M3WeightsRef
from autograd.rules_samba import (
    AUTOGRAD_SAMBA,
    SAMBA_CE,
    SAMBA_EMBEDDING,
    SAMBA_HEAD,
    SAMBA_NORM_F,
    SambaBstRef,
    SambaF32Ref,
    SambaFwdRef,
    SambaI32Ref,
    SambaLayersRef,
    SambaRules,
    SambaScansRef,
    samba_f32_ref,
    samba_i32_ref,
    samba_tape_eligible,
)
from sequence.exec_device import DeviceExec
from sequence.ops import FP, Args
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


def samba_targets_scan_kernel(
    count: MutPointer[Int32, MutAnyOrigin],
    first_bad: MutPointer[Int32, MutAnyOrigin],
    targets: MutPointer[Int32, MutAnyOrigin],
    n_in: Int32,
    vocab: Int32,
    ignore: Int32,
):
    """One thread per target row: `count += 1` for a row whose target is not
    `ignore`, and `first_bad = min(first_bad, row)` for one that is also
    outside `[0, vocab)`. Both are integer folds, exact in any order."""
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= Int(n_in):
        return
    var t = targets.unsafe_load(i)
    if t == ignore:
        return
    _ = Atomic[DType.int32].fetch_add(count, Int32(1))
    if t < Int32(0) or t >= vocab:
        _ = Atomic[DType.int32].min(first_bad, Int32(i))


def _admit_targets_device(
    ctx: DeviceContext,
    mut targets: DeviceBuffer[DType.int32],
    m: Int,
    vocab: Int,
    ignore: Int,
) raises -> Int:
    """`ce_refuse_targets` and `ce_count` over the uploaded targets: one
    launch, two words back, one wait. Returns `count`; raises the oracle's
    message, with the offending value read back, on an out-of-range row."""
    var count = ctx.enqueue_create_buffer[DType.int32](1)
    var first_bad = ctx.enqueue_create_buffer[DType.int32](1)
    count.enqueue_fill(Int32(0))
    first_bad.enqueue_fill(Int32(m))
    ctx.enqueue_function[samba_targets_scan_kernel](
        count.unsafe_ptr(), first_bad.unsafe_ptr(), targets.unsafe_ptr(),
        Int32(m), Int32(vocab), Int32(ignore),
        grid_dim=(_grid(m), 1, 1), block_dim=(SAMBA_TPB, 1, 1),
    )
    var host = ctx.enqueue_create_host_buffer[DType.int32](2)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=count)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr() + 1, src_buf=first_bad)
    ctx.synchronize()
    var n_count = Int(host.unsafe_ptr().unsafe_load(0))
    var bad = Int(host.unsafe_ptr().unsafe_load(1))
    _ = count^
    _ = first_bad^
    _ = host^
    if bad < m:
        var one = ctx.enqueue_create_host_buffer[DType.int32](1)
        var cell = targets.create_sub_buffer[DType.int32](bad, 1)
        ctx.enqueue_copy(dst_ptr=one.unsafe_ptr(), src_buf=cell)
        ctx.synchronize()
        var t = Int(one.unsafe_ptr().unsafe_load(0))
        _ = cell^
        _ = one^
        raise Error(
            String("ce: target ") + String(t) + " at row " + String(bad)
            + " is neither ignore_index nor in [0, vocab) REFUSED"
        )
    return n_count


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
    # ---- the tape arm's own state (AUTOGRAD_SAMBA; rebuilt with the shape)
    #: per Mamba-3 layer: the storage of d loss / d(block input), [M, dm]
    var m3_dx: List[DeviceBuffer[DType.float32]]
    #: the attention layers' nine weight gradients are views of `grad`
    var t_grad_bound: Bool
    #: the loss node's value word (the tape reads no value of it)
    var tape_loss: DeviceBuffer[DType.float32]
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
        self.m3_dx = List[DeviceBuffer[DType.float32]]()
        self.t_grad_bound = False
        self.tape_loss = ctx.enqueue_create_buffer[DType.float32](1)
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

    def view(self, j: Int) raises -> DeviceBuffer[DType.float32]:
        """Registry tensor `j` as a view of `param`."""
        return self.param.create_sub_buffer[DType.float32](
            self.offsets[j], self.offsets[j + 1] - self.offsets[j]
        )

    def grad_view(self, j: Int) raises -> DeviceBuffer[DType.float32]:
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
    # the tape arm's per-shape state: rebuilt on its next use
    s.m3_dx = List[DeviceBuffer[DType.float32]]()
    s.t_grad_bound = False
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
            # attn_stash=recompute for the Samba train step only (grid
            # ge123e6f9: samba-train-step 0.905x; the LM step keeps its
            # stash, 1.24x slower there). A forward-only call keeps the
            # build's profile (samba-forward was not measured with it).
            comptime if IDN_SAMBA_ATTN_RECOMPUTE:
                s.t_stages[j].attn_recompute_backward = not forward_only
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
            # -D MOJOLEARN_AUTOGRAD_LM: a default attention block's backward
            # through the transformer tape entry (the same launches, the
            # same bst fields); the Mamba-3 blocks stay hand-chained.
            var on_tape = False
            comptime if AUTOGRAD_LM:
                on_tape = s.t_w[j].opts.is_default()
            if on_tape:
                llama_decoder_layer_backward_autograd(
                    s.ctx, s.t_bst[j], s.t_stages[j], s.t_w[j], s.rope.value().cos, s.rope.value().sin,
                    s.acts[i], s.dh, b, l, 0, trace, String("samba.resident.attn.bwd"),
                )
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


@always_inline
def _fpb(mut buf: DeviceBuffer[DType.float32]) -> FP:
    return FP(unsafe_from_address=Int(buf.unsafe_ptr()))


def _bind_attention_grad_views(mut s: SambaResidentSession) raises:
    """Tape arm: each attention layer's nine weight gradients become views
    of `grad` at the registry offsets `_backward_blocks` copies them to
    (`training/byte_lm.mojo::_afn_bind_grad_views`, in Samba's registry
    order: norm1, norm2, q, k, v, o, gate, up, down). Each is written whole,
    once, by the layer backward, so `grad` ends holding what the nine copies
    wrote and the copies go. Bound once per shape (`_ensure_shape` rebuilds
    the stages and clears the flag). A session runs one arm (the arm is
    chosen at build time): the hand arm's copies would then copy a view
    onto itself."""
    if s.t_grad_bound:
        return
    for i in range(s.n_layers()):  # driver-loop: over stack layers
        if s.kinds[i] != SAMBA_LAYER_ATTENTION:
            continue
        var j = s.slot[i]
        var base = 1 + SAMBA_BLOCK_TENSORS * i
        var g0 = s.grad_view(base + 0)
        var g1 = s.grad_view(base + 1)
        var g2 = s.grad_view(base + 2)
        var g3 = s.grad_view(base + 3)
        var g4 = s.grad_view(base + 4)
        var g5 = s.grad_view(base + 5)
        var g6 = s.grad_view(base + 6)
        var g7 = s.grad_view(base + 7)
        var g8 = s.grad_view(base + 8)
        s.t_bst[j].dw_norm1 = g0^
        s.t_bst[j].dw_norm2 = g1^
        s.t_bst[j].dw_q = g2^
        s.t_bst[j].dw_k = g3^
        s.t_bst[j].dw_v = g4^
        s.t_bst[j].dw_o = g5^
        s.t_bst[j].dw_gate = g6^
        s.t_bst[j].dw_up = g7^
        s.t_bst[j].dw_down = g8^
    s.t_grad_bound = True


def _ensure_m3_dx(mut s: SambaResidentSession, m: Int) raises:
    """Tape arm: one [M, dm] buffer per Mamba-3 layer for d loss / d(block
    input). The hand path copies every block's input gradient into `dh`;
    on the tape every activation gradient has storage of its own (the
    tape's aliasing rule), so a Mamba-3 block's lands here."""
    var n_m3 = len(s.m3_w)
    if len(s.m3_dx) == n_m3:
        return
    s.m3_dx = List[DeviceBuffer[DType.float32]]()
    for _ in range(n_m3):  # driver-loop: over Mamba-3 layers
        s.m3_dx.append(s.ctx.enqueue_create_buffer[DType.float32](m * s.dm))


def _backward_tape(
    mut s: SambaResidentSession,
    b: Int,
    l: Int,
    mut ew: DeviceBuffer[DType.float32],
    mut hw: DeviceBuffer[DType.float32],
    mut nw: DeviceBuffer[DType.float32],
    mut dnw: DeviceBuffer[DType.float32],
) raises:
    """-D MOJOLEARN_AUTOGRAD_SAMBA (default on): the train step's whole
    backward on ONE tape (`autograd/rules_samba.mojo`, header): the head,
    the final norm, every block last to first, the embedding, then the tied
    pair add, with the hand path's kernels on the hand path's buffers. The
    gradients land in `grad` (and `pair`) directly: no per-layer copy of an
    attention block's gradients, no per-block wait after a Mamba-3 block's
    copies, one wait at the end. `dlogits` holds the loss gradient on entry
    (`identical_ce_loss_resident`)."""
    var m = b * l
    var dm = s.dm
    var v = s.vocab
    var n = s.n_layers()
    _bind_attention_grad_views(s)
    _ensure_m3_dx(s, m)
    var ton = String(getenv("MOJOLEARN_MAMBA_TIMING")) != ""
    # ---- every activation gradient's storage: gin[i] = d loss / d acts[i]
    # (the gradient the consumer of acts[i] writes); gin[n] = dh, the final
    # norm's dx as in the hand path
    var gin = List[FP]()
    for i in range(n):  # driver-loop: over stack layers
        var j = s.slot[i]
        if s.kinds[i] == SAMBA_LAYER_MAMBA3:
            gin.append(_fpb(s.m3_dx[j]))
        else:
            gin.append(_fpb(s.t_bst[j].d_x))
    gin.append(_fpb(s.dh))
    # ---- the embedding's and the head's weight-gradient destinations: the
    # tied pair's halves [d_emb | dw_head], or their registry tensors
    var demb = s.grad_view(0)
    var dwh = s.grad_view(s.head_index())
    if s.tie:
        demb = s.pair.create_sub_buffer[DType.float32](0, v * dm)
        dwh = s.pair.create_sub_buffer[DType.float32](v * dm, v * dm)

    # ---- the forward as it ran, recorded (nothing is relaunched)
    var ex = DeviceExec(s.ctx)
    var tape = Tape()
    var g_emb = tape.variable_into(ex, _fpb(ew), _fpb(demb), v, dm)
    var h = tape.custom_into(ex, [g_emb], _fpb(s.acts[0]), gin[0], m, dm, SAMBA_EMBEDDING, Args())
    for i in range(n):  # driver-loop: over stack layers
        var j = s.slot[i]
        if s.kinds[i] == SAMBA_LAYER_MAMBA3:
            # [x, the nine weights] -> the block's own residual_out; the
            # weight gradients are the registry tensors themselves
            var base = 1 + SAMBA_BLOCK_TENSORS * i
            var ins = List[Tensor]()
            ins.append(h)
            for k in range(SAMBA_BLOCK_TENSORS):  # driver-loop: over the block's nine weight tensors
                var wv = s.view(base + k)
                var gv = s.grad_view(base + k)
                ins.append(tape.variable_into(ex, _fpb(wv), _fpb(gv), 1, s.tensor_size(base + k)))
            var saved = Args()
            saved.i11 = j
            h = tape.custom_into(
                ex, ins, _fpb(s.m3_stages[j].residual_out), gin[i + 1], m, dm, M3_BLOCK, saved
            )
        else:
            # the decoder layer's nodes; its output is its own residual2
            h = llama_decoder_layer_record(
                tape, ex, h, s.t_bst[j], s.t_stages[j], s.t_w[j], gin[i + 1], b, l, j
            )
    var g_nw = tape.variable_into(ex, _fpb(nw), _fpb(dnw), 1, dm)
    var hn = tape.custom_into(ex, [h, g_nw], _fpb(s.hn), _fpb(s.dhn), m, dm, SAMBA_NORM_F, Args())
    # tied: the same embedding data recorded as a second leaf, its gradient
    # the pair's second half (summed by the hand path's pair add below)
    var g_hw = tape.variable_into(ex, _fpb(hw), _fpb(dwh), v, dm)
    var lg = tape.custom_into(ex, [hn, g_hw], _fpb(s.logits), _fpb(s.dlogits), m, v, SAMBA_HEAD, Args())
    # the loss; its gradient is the unit seed s.ones[0] (1.0, never written)
    var one = _fpb(s.ones)
    var loss = tape.custom_into(ex, [lg], _fpb(s.tape_loss), one, 1, 1, SAMBA_CE, Args())
    var seed = tape.variable(ex, one, 1, 1, False)

    # ---- the rule set's pointers (all into the session, which outlives it)
    var scans = List[DeferredNonfiniteScan]()
    var scan_layers = List[Int]()
    var keep = List[Mamba3DeviceGradients]()
    var bst_refs = List[SambaBstRef]()
    var fwd_refs = List[SambaFwdRef]()
    var t_eps = List[Float32]()
    for j in range(len(s.t_bst)):  # driver-loop: over attention layers
        bst_refs.append(SambaBstRef(unsafe_from_address=Int(MutPointer(to=s.t_bst[j]))))
        fwd_refs.append(SambaFwdRef(unsafe_from_address=Int(MutPointer(to=s.t_stages[j]))))
        t_eps.append(s.t_w[j].eps)
    var w3 = List[M3WeightsRef]()
    var st3 = List[M3StagesRef]()
    for j in range(len(s.m3_w)):  # driver-loop: over Mamba-3 layers
        w3.append(M3WeightsRef(unsafe_from_address=Int(MutPointer(to=s.m3_w[j]))))
        st3.append(M3StagesRef(unsafe_from_address=Int(MutPointer(to=s.m3_stages[j]))))
    var cos_tab = _fpb(s.ones)
    var sin_tab = _fpb(s.ones)
    if s.has_attn:
        cos_tab = _fpb(s.rope.value().cos)
        sin_tab = _fpb(s.rope.value().sin)
    var r_dh = samba_f32_ref(s.dh)
    var r_dhn = samba_f32_ref(s.dhn)
    var r_dlogits = samba_f32_ref(s.dlogits)
    var r_ws_a = samba_f32_ref(s.ws_a)
    var r_ws_b = samba_f32_ref(s.ws_b)
    var r_dot = samba_f32_ref(s.dot_out)
    var r_dh_tmp = samba_f32_ref(s.dh_tmp)
    var r_dprod = samba_f32_ref(s.dprod)
    var r_rstd = samba_f32_ref(s.rstd)
    var r_dvcoef = samba_f32_ref(s.dvcoef)
    var r_ones = samba_f32_ref(s.ones)
    var r_sumsq = samba_f32_ref(s.sumsq)
    var r_ids = samba_i32_ref(s.ids)
    var r_counts = samba_i32_ref(s.counts)
    var r_run_begin = samba_i32_ref(s.run_begin)
    var r_perm = samba_i32_ref(s.perm)
    var r_keep = M3KeepRef(unsafe_from_address=Int(MutPointer(to=keep)))
    var r_scans = SambaScansRef(unsafe_from_address=Int(MutPointer(to=scans)))
    var r_layers = SambaLayersRef(unsafe_from_address=Int(MutPointer(to=scan_layers)))
    var rules = SambaRules(
        s.ctx, bst_refs^, fwd_refs^, t_eps^, cos_tab, sin_tab, w3^, st3^, r_keep, ton,
        b, l, dm, v, s.eps, one,
        r_dh, r_dhn, r_dlogits, r_ws_a, r_ws_b, r_dot, r_dh_tmp, r_dprod, r_rstd, r_dvcoef, r_ones, r_sumsq,
        r_ids, r_counts, r_run_begin, r_perm, r_scans, r_layers,
    )

    # ---- one backward, then the tied pair add (embedding first, as the hand
    # path's `accumulate_grads([d_emb, dw_head], tokens=None)`)
    comptime if AUTOGRAD_REFUSE_DEFER:
        # A NaN reaching a later rule may make it raise first; the refusal
        # is the error the hand path gives, so it is checked first.
        var raised = False
        var msg = String()
        try:
            tape.backward_with_rules(ex, loss, seed, rules)
        except e:
            raised = True
            msg = String(e)
        if not raised:
            tape.release(ex)
        for i in range(len(scans)):  # driver-loop: over attention layers (one scan each)
            scans[i].enqueue_readback(s.ctx)
        # Drain the queue before the keep list and the views go.
        ex.sync()
        for i in range(len(scans)):  # driver-loop: over attention layers (one scan each)
            var bad = scans[i].result_after_sync()
            if bad >= 0:
                raise Error(
                    String("llama backward: NaN or infinity in d_residual2 at flat index ") + String(bad)
                    + " REFUSED"
                )
        if raised:
            raise Error(msg)
    else:
        try:
            tape.backward_with_rules(ex, loss, seed, rules)
        except e:
            # the keep list's buffers may still be read by queued copies
            ex.sync()
            raise e
        tape.release(ex)
        # Drain the queue before the keep list and the views go.
        ex.sync()
    if s.tie:
        var gemb = s.grad_view(0)
        _pair_add_device(s.ctx, gemb, s.pair, v * dm)
        _ = gemb^
    _ = keep^
    _ = scans^
    _ = scan_layers^
    _ = demb^
    _ = dwh^
    _ = gin^


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
        raise Error("mojolearn samba resident: built with MOJOLEARN_IDN_SAMBA_RESIDENT_STEP_OFF or MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2")
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


def samba_resident_train_step[tape: Bool = AUTOGRAD_SAMBA](
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
        raise Error("mojolearn samba resident: built with MOJOLEARN_IDN_SAMBA_RESIDENT_STEP_OFF or MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2")
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
    identical_ce_admit_call(REDUCTION_SUM, 1, m)
    _ensure_shape(s, b, l)
    var dm = s.dm
    var v = s.vocab
    # ---- transport in: the registry, the ids, the targets
    s.ctx.enqueue_copy(dst_buf=s.param, src_ptr=param_ptr)
    _upload_tokens(s, ids_ptr)
    s.ctx.enqueue_copy(dst_buf=s.targets, src_ptr=targets_ptr)
    # ---- the targets' range refusal and the ignore-index count, on the
    # device (`ce_refuse_targets` / `ce_count` in arm B); `count` is the
    # loss divisor (`num_items = count`, as `loss_and_grads` passes it)
    var count = _admit_targets_device(s.ctx, s.targets, m, v, IGNORE_INDEX_DEFAULT)
    var cfg = CeConfig(v, IGNORE_INDEX_DEFAULT, REDUCTION_SUM, Float32(0.0), count)
    ce_refuse_shape(m, m * v, cfg)
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

    # ---- the backward: the tape (AUTOGRAD_SAMBA, default on) or the hand
    # walk (MOJOLEARN_AUTOGRAD_SAMBA_OFF); the same kernels on the same
    # buffers either way, so the same bits
    var hw = s.view(s.head_index())
    var nw = s.view(s.norm_f_index())
    var dnw = s.grad_view(s.norm_f_index())
    var on_tape = False
    comptime if tape:
        on_tape = samba_tape_eligible(s.t_w)
    if on_tape:
        _backward_tape(s, b, l, ew, hw, nw, dnw)
    else:
        # ---- head backward: dhn = dlogits . W, dW = dlogits^T . hn
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
