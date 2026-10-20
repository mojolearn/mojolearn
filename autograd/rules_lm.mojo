# SPDX-License-Identifier: Apache-2.0
# Language-model trainer backward on the tape (the byte LM's only backward
# path). docs/plans/AUTOGRAD_API.md section (c) item 4.
#
# ---------------------------------------------------------------- launch inventory
# The hand path, training/byte_lm.mojo::byte_gradient_device after the
# forward (_byte_forward_loss), for one step of an n-layer byte LM
# (IDENTICAL, trace off), in launch order. "tb." is ByteBuffers, "bst[k]" /
# "fwd[k]" layer k's LlamaBackwardStages / LlamaDeviceStages.
#
#   loss    v1 head: identical_ce_backward_into (L14-L16) -> tb.ce_dlogits
#           (skipped under AFN_LM_HEAD_FUSE: the forward's fused row kernel
#           already wrote dlogits)
#           chunked v2 head: chunked_lm_head_v2_gemm_backward_into, per vocab
#           chunk a GEMM and the dhidden / dweight kernels -> tb.d_h, tb.dw_lm
#           (loss, head dA and head dB in one fused kernel set)
#   head    v1 only: identical_gemm_backward_a_into[ROLE_HEAD] tb.d_h <-
#           ce_dlogits . lm_w; identical_gemm_backward_b_into tb.dw_lm <-
#           ce_dlogits, fwd[n-1].residual2 (OP_NT, tb.head_bwd_ws)
#   layers  for k = n-1 .. 0: llama_decoder_layer_backward_device (deleted) (bst[k],
#           fwd[k], d_out = tb.d_h (k = n-1) or bst[k+1].d_x, x = tb.x (k = 0)
#           or fwd[k-1].residual2): per layer the refusal scan R of d_out,
#           27 + A launches (rules_transformer.mojo's inventory; + 1 bit
#           copy without IDN_BWD_ENTRY_ALIAS) -> bst[k].d_x and the nine dw;
#           a host wait per layer when _byte_layer_sync() is on
#   embed   identical_embedding_backward[_prerefused]_into (E0-E4) or the
#           NN50 owned runs (match / rebuild, then backward_into): the
#           fixed-order scatter-add of bst[0].d_x into tb.dw_emb
#   pack    _pack_block per layer (none under BYTE_LM_BLOCK_VIEWS) and the
#           two copies of dw_emb and dw_lm into tb.grad
#
# Glue (moved to the tape): the order of the stages, the chaining of the
# layers (which buffer is d_out of layer k, which is x), the head and
# embedding backward assembly, and every residual join; the joins inside a
# layer are the transformer rules' (writes_total, the hand kernels). Fused
# kernels (kept, one rule each, launched with the hand path's arguments on
# the hand path's buffers): the CE backward, the head GEMM pair, the chunked
# head+loss backward, the embedding scatter-add, and each layer's seven
# TransformerRules.
#
# ---------------------------------------------------------------- the tape graph
#   emb_w (variable_into dw_emb)
#     -> LM_EMBEDDING     x = emb_w[ids]       grad storage bst[0].d_x
#     -> layer 0 .. n-1   llama_decoder_layer_record; layer k's output
#                         (residual2) gradient storage is bst[k+1].d_x, the
#                         top layer's tb.d_h: the storage the hand path hands
#                         each layer as d_out, read in place
#   lm_w (variable_into dw_lm)
#     -> LM_HEAD          logits = h lm_w^T    grad storage tb.ce_dlogits
#     -> LM_CE            the mean CE loss     grad storage = the seed
#   (chunked v2: one LM_HEAD_CE_CHUNKED node [h, lm_w] -> loss)
#   seed: element 0 of tb.ce_ones (1.0, never written), recorded as the
#   loss's gradient storage, so the tape copies nothing into the root.
#
# ACCUMULATION ORDER. No tensor of this graph outside a decoder layer has
# two consumers: the embedding output feeds only layer 0, each layer's
# output only the next layer (or the head), the logits only the loss, and
# the embedding and head weights are not tied. So every gradient outside
# the layers is ONE contribution, written in place by its rule (no fill, no
# scratch, no OP_AG_ACCUMULATE). The two consumers of a layer input (norm1
# and the residual1 add) are joined inside the layer as the hand path does:
# d_x = norm1_dx + d_residual1 (S36's bwd_add2_kernel, or the
# _AFN_NORM1_RESID fusion), by the norm1 rule (writes_total); the residual
# rule writes nothing. Inside a layer the order is the transformer lane's:
# d(norm2.out) = gate + up (S9), d(norm1.out) = (q + k) + v (S32),
# d_residual1 = norm2_dx + d(residual2) (S10-13). The embedding gradient is
# the embedding kernel's own fixed-order fold over positions. So the tape
# adds in no order of its own: the expected result is the hand path's bits.
#
# ---------------------------------------------------------------- launch count
# Tape: the hand launches above, one for one, in the same order, with the
# same arguments (the CE backward, head dA, head dB, then the layers top
# down, each with its refusal scan, then the embedding). Differences: no
# S0-1 bit copy per layer when IDN_BWD_ENTRY_ALIAS is off (the tape reads
# d_out in place; the copy was the identity), no per-layer host wait (the
# refusal scans still wait as in the hand path), and one wait at the end
# (ex.sync(), before the tape's views go away). The tape itself launches
# nothing: zeroed_destination is False and writes_total covers the joins.
#
# Refused (they keep the hand path, via byte_lm_tape_eligible): a layer with
# non-default block options.
from std.sys.compile import is_defined
from max.gpu.host import DeviceBuffer, DeviceContext

from autograd.tape import Tape, Tensor, BackwardRules
from sequence.exec_trait import Exec
from sequence.exec_device import DeviceExec
from sequence.ops import FP, Args
from core.device_scan import device_first_nonfinite
from autograd.refuse_scan import DeferredNonfiniteScan
from gemm.contract import OP_NT
from gemm.experiments.neural_switches import ROLE_HEAD
from gemm.neural_backward import identical_gemm_backward_a_into, identical_gemm_backward_b_into
from autograd.kernels.embedding import (
    identical_embedding_backward_into,
    identical_embedding_backward_prerefused_into,
)
from embedding.checks.embedding_oracle import EmbConfig
from autograd.kernels.embedding_runs import NN50_OWNED_RUNS, OwnedEmbeddingRuns
from autograd.kernels.lm import identical_ce_backward_into
from training.checks.loss_contract import CeConfig
from autograd.kernels.lm import chunked_lm_head_v2_gemm_backward_into
from transformer.impl.llama.modeling_llama import (
    LlamaDeviceStages,
    LlamaDeviceWeights,
    llama_key_span,
)
from autograd.kernels.transformer import LlamaBackwardStages
from autograd.rules_transformer import (
    TF_RMS_NORM_INPUT,
    TF_RESIDUAL,
    TransformerRules,
    llama_decoder_layer_record,
)

# The language-model backward runs on the tape, the only path (promoted 2026-10-19). One run per arm, tools/bench_board_neural.py
# lm-train-step, ours, full shape: NVIDIA L40S 34.611 -> 35.351 ms (1.02), AMD MI325X 42.939 -> 42.953 ms (1.00); average 1.01;
# same loss_first_step 9.0155868. Every gradient and updated parameter equalled the hand walk word for word on Metal, NVIDIA
# and AMD (44/44); training/checks/lm_tape_backward_check.mojo now compares against those stored digests.
# TOMBSTONE: MOJOLEARN_AUTOGRAD_LM_OFF (hand LM backward walk; superseded by the tape, NV 1.02 / AMD 1.00, same bits) deleted 2026-10-19 by a54a5fb2e; code recoverable at a54a5fb2e^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_LM_OFF.patch; record in docs/TOMBSTONES.md.


def _check_removed_defines() -> Bool:
    """Refusals for retired byte LM backward defines (lane del-lm, docs/TOMBSTONES.md). They live here, next to the code they named, not
    in core/six_lane_experiment_guards.mojo: a line there marks every binding that imports core as changed (a full
    rebuild and identity re-record for one refusal)."""
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_LM_OFF"](), "removed 2026-10-19 (lane/del-lm, a54a5fb2e): AUTOGRAD_LM_OFF restored the hand byte LM backward walk; the tape replaced it (lm-train-step NV 34.611 -> 35.351 ms 1.02, AMD 42.939 -> 42.953 ms 1.00, avg 1.01; 44/44 words equal on Metal, NVIDIA, AMD). Restore with experiments/removed/MOJOLEARN_AUTOGRAD_LM_OFF.patch"
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_LM"](), "removed 2026-10-19 (lane/del-lm, a54a5fb2e): AUTOGRAD_LM was the opt-in spelling of the byte LM tape backward, now the only path. Restore with experiments/removed/MOJOLEARN_AUTOGRAD_LM_OFF.patch"
    return True


comptime LM_REMOVED_DEFINES_OK = _check_removed_defines()

# Lane ag-pass6-host (2026-10-19, PASS6_IDEAS idea J), default on: each
# layer's refusal scan of its d_out (_refuse_nonfinite) queues its verdict
# on the device (autograd/refuse_scan.mojo DeferredNonfiniteScan) at the
# same queue position, reading the same words, and the step reads every
# verdict at its one closing wait: n_layers drains a step become one. The
# first refused layer in backward order raises the same error as before
# (checked before any other error the backward raised, so a NaN still
# reports as the refusal); the step's gradients are written by then.
# Accepted steps: no kernel or order moves, the same bits.
# -D MOJOLEARN_AUTOGRAD_REFUSE_DEFER_OFF keeps one wait per layer (arm A).
comptime LM_REFUSE_DEFER = not is_defined["MOJOLEARN_AUTOGRAD_REFUSE_DEFER_OFF"]()

# Tags 64-71 (LayerNorm 1, recurrent 16-18, transformer 32-47, Mamba 48-63).
comptime LM_EMBEDDING = 64
comptime LM_HEAD = 65
comptime LM_CE = 66
comptime LM_HEAD_CE_CHUNKED = 67

comptime _F32Ref = MutPointer[DeviceBuffer[DType.float32], MutUntrackedOrigin]
comptime _I32Ref = MutPointer[DeviceBuffer[DType.int32], MutUntrackedOrigin]
comptime _BstRef = MutPointer[LlamaBackwardStages, MutUntrackedOrigin]
comptime _FwdRef = MutPointer[LlamaDeviceStages, MutUntrackedOrigin]
comptime _RunsRef = MutPointer[List[OwnedEmbeddingRuns], MutUntrackedOrigin]
comptime _ScansRef = MutPointer[List[DeferredNonfiniteScan], MutUntrackedOrigin]
comptime _LayersRef = MutPointer[List[Int], MutUntrackedOrigin]


@always_inline
def _fp(mut buf: DeviceBuffer[DType.float32]) -> FP:
    return FP(unsafe_from_address=Int(buf.unsafe_ptr()))


@always_inline
def _view(ctx: DeviceContext, p: FP, n: Int) -> DeviceBuffer[DType.float32]:
    """A non-owning view of n floats at p."""
    return DeviceBuffer[DType.float32](ctx, p.unsafe_origin_cast[MutAnyOrigin](), n if n > 0 else 1, owning=False)


@always_inline
def _f32(mut buf: DeviceBuffer[DType.float32]) -> _F32Ref:
    return _F32Ref(unsafe_from_address=Int(MutPointer(to=buf)))


@always_inline
def _i32(mut buf: DeviceBuffer[DType.int32]) -> _I32Ref:
    return _I32Ref(unsafe_from_address=Int(MutPointer(to=buf)))


def _same(p: FP, mut buf: DeviceBuffer[DType.float32], what: String) raises:
    # The rules launch on the trainer's own buffers; the tape must hand them
    # exactly those (the driver records them so). Anything else is a driver bug.
    if Int(p) != Int(buf.unsafe_ptr()):
        raise Error(String("byte LM backward (autograd): ") + what + " is not the trainer's buffer")


struct LmRules(BackwardRules):
    """Backward adapters for the byte LM's fused native ops (embedding,
    head, loss) and, by layer index, each decoder layer's TransformerRules.

    Holds pointers to the trainer's buffers and stages; all must outlive
    the backward."""

    var ctx: DeviceContext
    var bsts: List[_BstRef]
    var fwds: List[_FwdRef]
    var eps: List[Float32]
    var cos_tab: FP
    var sin_tab: FP
    var b: Int
    var l: Int
    var dm: Int
    var vocab: Int
    var seed: FP
    var chunked: Bool
    var dlogits_ready: Bool
    var emb_prerefused: Bool
    var emb_plan: Int
    var x: _F32Ref
    var emb_w: _F32Ref
    var dw_emb: _F32Ref
    var lm_w: _F32Ref
    var dw_lm: _F32Ref
    var d_h: _F32Ref
    var logits: _F32Ref
    var ce_dlogits: _F32Ref
    var ce_weights: _F32Ref
    var ce_expo: _F32Ref
    var ce_denom: _F32Ref
    var ce_logp: _F32Ref
    var ce_max: _F32Ref
    var head_ws: _F32Ref
    var head_bwd_ws: _F32Ref
    var targets: _I32Ref
    var ids: _I32Ref
    var emb_counts: _I32Ref
    var emb_run_begin: _I32Ref
    var emb_perm: _I32Ref
    var emb_runs: _RunsRef
    # idea J: the deferred refusal scans and their layers, in backward
    # order (the trainer owns both lists; null when LM_REFUSE_DEFER is off)
    var scans: _ScansRef
    var scan_layers: _LayersRef

    def __init__(
        out self, ctx: DeviceContext, var bsts: List[_BstRef], var fwds: List[_FwdRef], var eps: List[Float32],
        cos_tab: FP, sin_tab: FP, b: Int, l: Int, dm: Int, vocab: Int, seed: FP,
        chunked: Bool, dlogits_ready: Bool, emb_prerefused: Bool, emb_plan: Int,
        x: _F32Ref, emb_w: _F32Ref, dw_emb: _F32Ref, lm_w: _F32Ref, dw_lm: _F32Ref, d_h: _F32Ref,
        logits: _F32Ref, ce_dlogits: _F32Ref, ce_weights: _F32Ref, ce_expo: _F32Ref, ce_denom: _F32Ref,
        ce_logp: _F32Ref, ce_max: _F32Ref, head_ws: _F32Ref, head_bwd_ws: _F32Ref,
        targets: _I32Ref, ids: _I32Ref, emb_counts: _I32Ref, emb_run_begin: _I32Ref, emb_perm: _I32Ref,
        emb_runs: _RunsRef, scans: _ScansRef, scan_layers: _LayersRef,
    ):
        self.ctx = ctx.copy()
        self.bsts = bsts^
        self.fwds = fwds^
        self.eps = eps^
        self.cos_tab = cos_tab
        self.sin_tab = sin_tab
        self.b = b
        self.l = l
        self.dm = dm
        self.vocab = vocab
        self.seed = seed
        self.chunked = chunked
        self.dlogits_ready = dlogits_ready
        self.emb_prerefused = emb_prerefused
        self.emb_plan = emb_plan
        self.x = x
        self.emb_w = emb_w
        self.dw_emb = dw_emb
        self.lm_w = lm_w
        self.dw_lm = dw_lm
        self.d_h = d_h
        self.logits = logits
        self.ce_dlogits = ce_dlogits
        self.ce_weights = ce_weights
        self.ce_expo = ce_expo
        self.ce_denom = ce_denom
        self.ce_logp = ce_logp
        self.ce_max = ce_max
        self.head_ws = head_ws
        self.head_bwd_ws = head_bwd_ws
        self.targets = targets
        self.ids = ids
        self.emb_counts = emb_counts
        self.emb_run_begin = emb_run_begin
        self.emb_perm = emb_perm
        self.emb_runs = emb_runs
        self.scans = scans
        self.scan_layers = scan_layers

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        # Never: every kernel overwrites every element of the destination it
        # is handed (the hand path reuses dw_emb, dw_lm, d_h, ce_dlogits and
        # the stage buffers across steps with no fill), and the transformer
        # rules say the same of theirs.
        return False

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        if tag >= TF_RMS_NORM_INPUT and tag <= TF_RESIDUAL:
            var tf = TransformerRules(
                self.ctx, self.bsts[0], self.fwds[0], self.cos_tab, self.sin_tab, self.eps[0], self.b, self.l, 0,
            )
            return tf.writes_total(tag, saved, edge)
        # embedding, head, loss: each input has one consumer, the plain
        # first-contribution write in place (accumulation order, header)
        return False

    def supports(self, tag: Int) -> Bool:
        return (tag >= TF_RMS_NORM_INPUT and tag <= TF_RESIDUAL) or (tag >= LM_EMBEDDING and tag <= LM_HEAD_CE_CHUNKED)

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        # Every launch goes on self.ctx, the context `ex` wraps.
        if tag >= TF_RMS_NORM_INPUT and tag <= TF_RESIDUAL:
            var layer = saved.i11
            if layer < 0 or layer >= len(self.bsts):
                raise Error("byte LM backward (autograd): node carries no valid layer index")
            if tag == TF_RESIDUAL and saved.i10 == 1:
                # the layer's output: the hand path's entry refusal of d_out
                self._refuse_nonfinite(upstream, rows[1] * cols[1], layer)
            var tf = TransformerRules(
                self.ctx, self.bsts[layer], self.fwds[layer], self.cos_tab, self.sin_tab,
                self.eps[layer], self.b, self.l, 0,
            )
            tf.apply(ex, tag, saved, output, upstream, inputs, gradients, rows, cols)
        elif tag == LM_EMBEDDING:
            self._embedding(upstream, gradients)
        elif tag == LM_HEAD:
            self._head(upstream, gradients)
        elif tag == LM_CE:
            self._ce(upstream, gradients)
        elif tag == LM_HEAD_CE_CHUNKED:
            self._head_ce_chunked(upstream, gradients)
        else:
            raise Error("unknown language-model backward rule")

    def _refuse_nonfinite(self, upstream: FP, n: Int, layer: Int) raises:
        # the deleted hand layer walk's first device read: the
        # incoming gradient is refused by bits before any of its stages.
        var d = _view(self.ctx, upstream, n)
        comptime if LM_REFUSE_DEFER:
            # the verdict is read at the step's closing wait (byte_backward_autograd)
            self.scans[].append(DeferredNonfiniteScan(self.ctx, d, n))
            self.scan_layers[].append(layer)
            return
        var bad = device_first_nonfinite(self.ctx, d, n)
        if bad >= 0:
            raise Error(
                String("llama backward: NaN or infinity in d_residual2 of block ") + String(layer)
                + " at flat index " + String(bad) + " REFUSED"
            )

    def _embedding(self, upstream: FP, gradients: List[FP]) raises:
        # byte_gradient_device's embedding backward, verbatim: dw_emb from
        # block 0's d_x, the trainer's id buffers and plan.
        ref bst0 = self.bsts[0][]
        _same(upstream, bst0.d_x, "the embedding output gradient")
        _same(gradients[0], self.dw_emb[], "the embedding weight gradient")
        var m = self.b * self.l
        var emb = EmbConfig.llama(self.vocab, self.dm)
        comptime if NN50_OWNED_RUNS:
            ref runs = self.emb_runs[]
            var reuse = False
            if len(runs) == 1:
                reuse = runs[0].matches(self.ctx, self.ids[], m, emb)
            if not reuse:
                if len(runs) == 1:
                    _ = runs.pop()
                runs.append(OwnedEmbeddingRuns(self.ctx, self.ids[], m, emb))
            runs[0].backward_into(self.ctx, self.dw_emb[], bst0.d_x, emb)
        else:
            if self.emb_prerefused:
                identical_embedding_backward_prerefused_into(
                    self.ctx, self.dw_emb[], bst0.d_x, self.ids[], self.emb_counts[],
                    self.emb_run_begin[], self.emb_perm[], m, emb, self.emb_plan,
                )
            else:
                identical_embedding_backward_into(
                    self.ctx, self.dw_emb[], bst0.d_x, self.ids[], self.emb_counts[],
                    self.emb_run_begin[], self.emb_perm[], m, emb, self.emb_plan,
                )

    def _head(self, upstream: FP, gradients: List[FP]) raises:
        # forward logits [M, V] = h [M, DM] . lm_w [V, DM]^T (OP_NT): the hand
        # path's dA (ROLE_HEAD) then dB, on its workspace
        ref top = self.fwds[len(self.fwds) - 1][]
        _same(upstream, self.ce_dlogits[], "the logits gradient")
        _same(gradients[0], self.d_h[], "the head input gradient")
        _same(gradients[1], self.dw_lm[], "the head weight gradient")
        var m = self.b * self.l
        identical_gemm_backward_a_into[ROLE_HEAD](
            self.ctx, self.d_h[], self.ce_dlogits[], self.lm_w[], self.head_bwd_ws[], m, self.vocab, self.dm, OP_NT,
        )
        identical_gemm_backward_b_into(
            self.ctx, self.dw_lm[], self.ce_dlogits[], top.residual2, self.head_bwd_ws[], m, self.vocab, self.dm, OP_NT,
        )

    def _ce(self, upstream: FP, gradients: List[FP]) raises:
        # The mean loss's VJP at the unit seed: the hand kernel folds 1/M
        # itself, so the rule reads no upstream and refuses any other seed.
        if Int(upstream) != Int(self.seed):
            raise Error("byte LM backward (autograd): the loss rule takes only the unit seed")
        _same(gradients[0], self.ce_dlogits[], "the logits gradient")
        if self.dlogits_ready:
            return  # AFN_LM_HEAD_FUSE: the forward's row kernel wrote dlogits
        var m = self.b * self.l
        identical_ce_backward_into(
            self.ctx, self.ce_weights[], self.ce_dlogits[], self.ce_expo[], self.ce_denom[],
            self.ce_logp[], self.targets[], m, m, CeConfig.causal_lm(self.vocab),
        )

    def _head_ce_chunked(self, upstream: FP, gradients: List[FP]) raises:
        # The chunked v2 head: loss, dA and dB in the hand path's one call
        if Int(upstream) != Int(self.seed):
            raise Error("byte LM backward (autograd): the loss rule takes only the unit seed")
        ref top = self.fwds[len(self.fwds) - 1][]
        _same(gradients[0], self.d_h[], "the head input gradient")
        _same(gradients[1], self.dw_lm[], "the head weight gradient")
        var m = self.b * self.l
        chunked_lm_head_v2_gemm_backward_into(
            self.ctx, self.d_h[], self.dw_lm[], self.logits[], self.head_ws[], top.residual2,
            self.lm_w[], self.targets[], self.ce_max[], self.ce_denom[], m, self.vocab, self.dm,
        )


def byte_lm_tape_eligible(ws: List[LlamaDeviceWeights]) -> Bool:
    """Whether every layer is one the tape records (default block options)."""
    if len(ws) < 1:
        return False
    for k in range(len(ws)):  # driver-loop: over decoder layers
        if not ws[k].opts.is_default():
            return False
    return True


def byte_backward_autograd(
    ctx: DeviceContext,
    mut bsts: List[LlamaBackwardStages],
    mut fwds: List[LlamaDeviceStages],
    mut ws: List[LlamaDeviceWeights],
    mut cos_tab: DeviceBuffer[DType.float32],
    mut sin_tab: DeviceBuffer[DType.float32],
    mut x: DeviceBuffer[DType.float32],
    mut emb_w: DeviceBuffer[DType.float32],
    mut dw_emb: DeviceBuffer[DType.float32],
    mut lm_w: DeviceBuffer[DType.float32],
    mut dw_lm: DeviceBuffer[DType.float32],
    mut d_h: DeviceBuffer[DType.float32],
    mut logits: DeviceBuffer[DType.float32],
    mut ce_dlogits: DeviceBuffer[DType.float32],
    mut ce_loss: DeviceBuffer[DType.float32],
    mut ce_ones: DeviceBuffer[DType.float32],
    mut ce_weights: DeviceBuffer[DType.float32],
    mut ce_expo: DeviceBuffer[DType.float32],
    mut ce_denom: DeviceBuffer[DType.float32],
    mut ce_logp: DeviceBuffer[DType.float32],
    mut ce_max: DeviceBuffer[DType.float32],
    mut head_ws: DeviceBuffer[DType.float32],
    mut head_bwd_ws: DeviceBuffer[DType.float32],
    mut targets: DeviceBuffer[DType.int32],
    mut ids: DeviceBuffer[DType.int32],
    mut emb_counts: DeviceBuffer[DType.int32],
    mut emb_run_begin: DeviceBuffer[DType.int32],
    mut emb_perm: DeviceBuffer[DType.int32],
    mut emb_runs: List[OwnedEmbeddingRuns],
    b: Int,
    l: Int,
    vocab: Int,
    chunked: Bool,
    dlogits_ready: Bool,
    emb_prerefused: Bool,
    emb_plan: Int,
) raises:
    """The byte LM's whole backward on one tape: embedding gather, every
    decoder layer, the LM head and the mean cross-entropy, recorded over
    the forward that already ran, then one backward.

    Writes exactly what the hand walk writes: d_h, dw_lm (and ce_dlogits),
    every layer's `bsts[k]` gradients (d_x and the nine weight gradients)
    and dw_emb, with the same kernels on the same buffers. The caller packs
    them into its flat gradient as before.

    A NaN or infinity in a layer's incoming gradient raises an error; the
    step's gradients may already have been written when it does."""
    comptime assert LM_REMOVED_DEFINES_OK, "retired byte LM backward define"
    var n_layers = len(fwds)
    if n_layers < 1 or len(bsts) != n_layers or len(ws) != n_layers:
        raise Error("byte LM backward (autograd): one forward stage, backward stage and weight set per layer")
    if not byte_lm_tape_eligible(ws):
        raise Error("byte LM backward (autograd): only default Llama blocks are recorded")
    if b <= 0 or l <= 0:
        raise Error("byte LM backward (autograd): B and L must be positive")
    var dims = fwds[0].dims.copy()
    dims.validate()
    var dm = dims.d_model
    var m = b * l
    for k in range(n_layers):  # driver-loop: over decoder layers
        # the hand path's per-layer refusals that need no device read
        if bsts[k].b != b or bsts[k].l != l:
            raise Error("llama backward: the stage buffers do not match (B, L)")
        var s = llama_key_span(0, l, fwds[k].window)
        if s > bsts[k].s_max:
            raise Error(String("llama backward: pos0 + L = ") + String(s) + " exceeds the allocated s_max " + String(bsts[k].s_max))
    if len(d_h) != m * dm or len(x) < m * dm:
        raise Error("byte LM backward (autograd): d_h and the block input must hold [B, L, d_model]")
    if len(ce_ones) < 1:
        raise Error("byte LM backward (autograd): the unit seed needs one element of ce_ones")

    var ex = DeviceExec(ctx)
    var tape = Tape()
    # ---- embedding: x = emb_w[ids], its gradient block 0's d_x
    var g_emb = tape.variable_into(ex, _fp(emb_w), _fp(dw_emb), vocab, dm)
    var h = tape.custom_into(ex, [g_emb], _fp(x), _fp(bsts[0].d_x), m, dm, LM_EMBEDDING, Args())
    # ---- the layers; layer k's output gradient is the storage the hand
    # path passes layer k as d_out: bst[k+1].d_x, the top layer's d_h
    for k in range(n_layers):  # driver-loop: over decoder layers
        var din = _fp(d_h) if k == n_layers - 1 else _fp(bsts[k + 1].d_x)
        h = llama_decoder_layer_record(tape, ex, h, bsts[k], fwds[k], ws[k], din, b, l, k)
    # ---- head and loss; the loss's gradient is the unit seed (ce_ones[0])
    var g_lm = tape.variable_into(ex, _fp(lm_w), _fp(dw_lm), vocab, dm)
    var one = _fp(ce_ones)
    var loss: Tensor
    if chunked:
        loss = tape.custom_into(ex, [h, g_lm], _fp(ce_loss), one, 1, 1, LM_HEAD_CE_CHUNKED, Args())
    else:
        var lg = tape.custom_into(ex, [h, g_lm], _fp(logits), _fp(ce_dlogits), m, vocab, LM_HEAD, Args())
        loss = tape.custom_into(ex, [lg], _fp(ce_loss), one, 1, 1, LM_CE, Args())
    var seed = tape.variable(ex, one, 1, 1, False)

    var scans = List[DeferredNonfiniteScan]()
    var scan_layers = List[Int]()
    var bst_refs = List[_BstRef]()
    var fwd_refs = List[_FwdRef]()
    var eps = List[Float32]()
    for k in range(n_layers):  # driver-loop: over decoder layers
        bst_refs.append(_BstRef(unsafe_from_address=Int(MutPointer(to=bsts[k]))))
        fwd_refs.append(_FwdRef(unsafe_from_address=Int(MutPointer(to=fwds[k]))))
        eps.append(ws[k].eps)
    var rules = LmRules(
        ctx, bst_refs^, fwd_refs^, eps^, _fp(cos_tab), _fp(sin_tab), b, l, dm, vocab, one,
        chunked, dlogits_ready, emb_prerefused, emb_plan,
        _f32(x), _f32(emb_w), _f32(dw_emb), _f32(lm_w), _f32(dw_lm), _f32(d_h),
        _f32(logits), _f32(ce_dlogits), _f32(ce_weights), _f32(ce_expo), _f32(ce_denom),
        _f32(ce_logp), _f32(ce_max), _f32(head_ws), _f32(head_bwd_ws),
        _i32(targets), _i32(ids), _i32(emb_counts), _i32(emb_run_begin), _i32(emb_perm),
        _RunsRef(unsafe_from_address=Int(MutPointer(to=emb_runs))),
        _ScansRef(unsafe_from_address=Int(MutPointer(to=scans))),
        _LayersRef(unsafe_from_address=Int(MutPointer(to=scan_layers))),
    )
    comptime if LM_REFUSE_DEFER:
        # idea J: a NaN reaching a later rule may make it raise first; the
        # refusal is the error the hand path gives, so it is checked first.
        var raised = False
        var msg = String()
        try:
            tape.backward_with_rules(ex, loss, seed, rules)
        except e:
            raised = True
            msg = String(e)
        if not raised:
            tape.release(ex)
        for i in range(len(scans)):  # driver-loop: over decoder layers (one scan each)
            scans[i].enqueue_readback(ctx)
        # Drain the queue before the stages and views go back to the caller.
        ex.sync()
        for i in range(len(scans)):  # driver-loop: over decoder layers (one scan each)
            var bad = scans[i].result_after_sync()
            if bad >= 0:
                raise Error(
                    String("llama backward: NaN or infinity in d_residual2 of block ") + String(scan_layers[i])
                    + " at flat index " + String(bad) + " REFUSED"
                )
        if raised:
            raise Error(msg)
    else:
        tape.backward_with_rules(ex, loss, seed, rules)
        tape.release(ex)
        # Drain the queue before the stages and views go back to the caller.
        ex.sync()
