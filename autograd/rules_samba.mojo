# SPDX-License-Identifier: Apache-2.0
# The Samba whole-stack train-step backward on ONE tape
# (TAPE_EVERYWHERE_PLAN.md path 2; the record loop is
# training/samba_resident.mojo::_backward_tape, which imports this module;
# this module never imports samba_resident).
#
# ---------------------------------------------------------------- launch inventory
# The hand path (deleted 2026-10-19, MOJOLEARN_AUTOGRAD_SAMBA_OFF, tombstone
# below; the launches the rules reproduce), samba_resident_train_step after
# the loss (identical_ce_loss_resident already left dlogits on the device),
# for an n-layer stack, in launch order ("s." the session, "bst[j]" / "fwd[j]"
# attention slot j's LlamaBackwardStages / LlamaDeviceStages, "st[j]" Mamba-3
# slot j's Mamba3DeviceStages):
#
#   head    identical_gemm_backward_a_into[ROLE_HEAD] s.dhn <- dlogits . W
#           (ws_a); identical_gemm_backward_b_into dW_head <- dlogits, s.hn
#           (ws_b) into pair[vd:2vd] (tied) or grad[lm_head] (untied)
#   norm_f  bwd_rms_norm[0] -> s.dh, grad[norm_f] (its kernels + the
#           weight-gradient GEMM)
#   blocks  for i = n-1 .. 0:
#           attention: the layer tape entry (refusal scan of s.dh, 27 + A
#             launches, one wait), then 9 _d2d of bst.dw_* into grad and 1
#             _d2d of bst.d_x into s.dh
#           Mamba-3: mamba3_prefill_backward_on (~30 launches, its own
#             closing synchronize), 9 _d2d into grad + 1 into s.dh, then ONE
#             MORE ctx.synchronize() to keep its gradients alive past the
#             copies
#   embed   identical_embedding_backward_into s.dh -> pair[0:vd] or grad[0]
#   tied    _pair_add_device grad[0] = pair[0:vd] + pair[vd:2vd]
#
# ---------------------------------------------------------------- the tape graph
#   embed.weight (variable_into pair[0:vd] | grad[0])
#     -> SAMBA_EMBEDDING   acts[0] = W[ids]        grad storage G(0)
#     -> layer 0 .. n-1    attention: llama_decoder_layer_record (TF_* 32-38,
#                          the transformer rules; the nine weight gradients
#                          land in bst.dw_*, bound to views of grad), input
#                          gradient bst.d_x;
#                          Mamba-3: one M3_BLOCK node (autograd/rules_mamba3)
#                          [x, nine weights (variable_into grad views)] ->
#                          st.residual_out, input gradient storage a
#                          per-layer buffer of the session (m3_dx)
#                          layer i's output gradient storage is G(i+1), the
#                          storage of the next layer's input; G(n) = s.dh
#   norm_f.weight (variable_into grad[norm_f])
#     -> SAMBA_NORM_F      hn = rms_norm(h)       grad storage s.dhn
#   head weight (variable_into pair[vd:2vd] | grad[lm_head]; tied: the same
#   embed.weight data recorded as a second leaf)
#     -> SAMBA_HEAD        logits = hn W^T        grad storage s.dlogits
#     -> SAMBA_CE          the loss               grad storage = the seed
#   seed: s.ones[0] (exactly 1.0, never written), so the tape copies nothing
#   into the root.
#
# ACCUMULATION ORDER. Outside the decoder layers no tensor has two consumers
# (the embedding output feeds layer 0, each layer's output the next layer or
# the final norm, hn the head, the logits the loss), and the tied pair is two
# leaves with two destinations whose sum is the hand path's own pair-add
# launch, run after the backward with the hand path's operand order
# (embedding first). So the tape adds nothing of its own; inside each
# attention layer the joins are the transformer rules' (writes_total, the
# hand kernels). Expected result: the hand path's bits, every gradient word.
#
# ---------------------------------------------------------------- launch count
# Tape: the hand launches above, one for one, in the same order, with the
# same arguments, except
#   - attention: no 9 + 1 _d2d per layer (the weight gradients are written
#     in place into grad views, the input gradient is read in place by the
#     next rule), no per-layer wait (the refusal scans are deferred to the
#     step's closing wait, AUTOGRAD_REFUSE_DEFER);
#   - Mamba-3: no extra synchronize per block (the gradients stay alive in
#     the keep list); the 10 copies stay (the VJP allocates its outputs) and
#     so does the VJP's own closing synchronize (its contract);
#   - the embedding reads G(0) in place (no copy into s.dh first).
# One wait at the end (ex.sync(), before the keep list and the views go).
#
# Refused (samba_resident_train_step raises, samba_tape_eligible): an attention
# layer with non-default block options (the resident session builds default ones).
from max.gpu.host import DeviceBuffer, DeviceContext

from autograd.tape import BackwardRules
from sequence.exec_trait import Exec
from sequence.ops import FP, Args
from core.device_scan import device_first_nonfinite
from autograd.refuse_scan import DeferredNonfiniteScan
from gemm.contract import OP_NT
from gemm.experiments.neural_switches import ROLE_HEAD
from gemm.neural_backward import identical_gemm_backward_a_into, identical_gemm_backward_b_into
from embedding.checks.embedding_identical import identical_embedding_backward_into
from embedding.checks.embedding_oracle import EmbConfig
from transformer.impl.llama.modeling_llama import LlamaDeviceStages, LlamaDeviceWeights
from transformer.checks.transformer_backward import LlamaBackwardStages, bwd_rms_norm
from autograd.rules_transformer import (
    AUTOGRAD_REFUSE_DEFER,
    TF_RMS_NORM_INPUT,
    TF_RESIDUAL,
    TransformerRules,
)
from autograd.rules_mamba3 import (
    M3_BLOCK,
    M3KeepRef,
    M3StagesRef,
    M3WeightsRef,
    Mamba3BlockRules,
)

# The Samba train step's backward runs on the tape only (training/checks/samba_tape_step_check.mojo
# checks it against stored digests).
# TOMBSTONE: MOJOLEARN_AUTOGRAD_SAMBA_OFF (DELETED: the tape is the only Samba train-step backward) deleted 2026-10-19 by SHA_X; code recoverable at SHA_X^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_SAMBA_OFF.patch
# (the AUTOGRAD_SAMBA switch; MOJOLEARN_AUTOGRAD_SAMBA and MOJOLEARN_AUTOGRAD_SAMBA_OFF are refused in core/six_lane_experiment_guards.mojo)

# Tags 88-95 (LayerNorm 1, recurrent 16-18, transformer 32-47, Mamba 48-63,
# LM 64-71, Mamba-3 72-87).
comptime SAMBA_EMBEDDING = 88
comptime SAMBA_NORM_F = 89
comptime SAMBA_HEAD = 90
comptime SAMBA_CE = 91
comptime SAMBA_TAG_FIRST = 88
comptime SAMBA_TAG_LAST = 95

comptime SambaF32Ref = MutPointer[DeviceBuffer[DType.float32], MutUntrackedOrigin]
comptime SambaI32Ref = MutPointer[DeviceBuffer[DType.int32], MutUntrackedOrigin]
comptime SambaBstRef = MutPointer[LlamaBackwardStages, MutUntrackedOrigin]
comptime SambaFwdRef = MutPointer[LlamaDeviceStages, MutUntrackedOrigin]
comptime SambaScansRef = MutPointer[List[DeferredNonfiniteScan], MutUntrackedOrigin]
comptime SambaLayersRef = MutPointer[List[Int], MutUntrackedOrigin]


@always_inline
def _view(ctx: DeviceContext, p: FP, n: Int) -> DeviceBuffer[DType.float32]:
    """A non-owning view of n floats at p."""
    return DeviceBuffer[DType.float32](ctx, p.unsafe_origin_cast[MutAnyOrigin](), n if n > 0 else 1, owning=False)


@always_inline
def samba_f32_ref(mut buf: DeviceBuffer[DType.float32]) -> SambaF32Ref:
    return SambaF32Ref(unsafe_from_address=Int(MutPointer(to=buf)))


@always_inline
def samba_i32_ref(mut buf: DeviceBuffer[DType.int32]) -> SambaI32Ref:
    return SambaI32Ref(unsafe_from_address=Int(MutPointer(to=buf)))


def _same(p: FP, mut buf: DeviceBuffer[DType.float32], what: String) raises:
    # The rules launch on the session's own buffers; the tape must hand them
    # exactly those (the driver records them so). Anything else is a driver bug.
    if Int(p) != Int(buf.unsafe_ptr()):
        raise Error(String("samba backward (autograd): ") + what + " is not the session's buffer")


def samba_tape_eligible(ws: List[LlamaDeviceWeights]) -> Bool:
    """Whether every attention layer is one the tape records (default block
    options). A stack with no attention layer is eligible."""
    for k in range(len(ws)):  # driver-loop: over attention layers
        if not ws[k].opts.is_default():
            return False
    return True


struct SambaRules(BackwardRules):
    """Backward adapters for the Samba stack's fused native ops (embedding,
    final norm, head, loss), each attention layer's TransformerRules and each
    Mamba-3 layer's M3_BLOCK rule, picked by the slot in Args i11.

    Holds pointers to the session's buffers and stages; all must outlive
    the backward."""

    var ctx: DeviceContext
    var bsts: List[SambaBstRef]
    var fwds: List[SambaFwdRef]
    var t_eps: List[Float32]
    var cos_tab: FP
    var sin_tab: FP
    var m3_w: List[M3WeightsRef]
    var m3_st: List[M3StagesRef]
    var keep: M3KeepRef
    var ton: Bool
    var b: Int
    var l: Int
    var dm: Int
    var vocab: Int
    var eps: Float32
    var seed: FP
    var dh: SambaF32Ref
    var dhn: SambaF32Ref
    var dlogits: SambaF32Ref
    var ws_a: SambaF32Ref
    var ws_b: SambaF32Ref
    var dot_out: SambaF32Ref
    var dh_tmp: SambaF32Ref
    var dprod: SambaF32Ref
    var rstd: SambaF32Ref
    var dvcoef: SambaF32Ref
    var ones: SambaF32Ref
    var sumsq: SambaF32Ref
    var ids: SambaI32Ref
    var counts: SambaI32Ref
    var run_begin: SambaI32Ref
    var perm: SambaI32Ref
    # the deferred refusal scans and their attention slots, in backward
    # order (the driver owns both lists)
    var scans: SambaScansRef
    var scan_layers: SambaLayersRef

    def __init__(
        out self, ctx: DeviceContext,
        var bsts: List[SambaBstRef], var fwds: List[SambaFwdRef], var t_eps: List[Float32],
        cos_tab: FP, sin_tab: FP,
        var m3_w: List[M3WeightsRef], var m3_st: List[M3StagesRef], keep: M3KeepRef, ton: Bool,
        b: Int, l: Int, dm: Int, vocab: Int, eps: Float32, seed: FP,
        dh: SambaF32Ref, dhn: SambaF32Ref, dlogits: SambaF32Ref, ws_a: SambaF32Ref, ws_b: SambaF32Ref,
        dot_out: SambaF32Ref, dh_tmp: SambaF32Ref, dprod: SambaF32Ref, rstd: SambaF32Ref,
        dvcoef: SambaF32Ref, ones: SambaF32Ref, sumsq: SambaF32Ref,
        ids: SambaI32Ref, counts: SambaI32Ref, run_begin: SambaI32Ref, perm: SambaI32Ref,
        scans: SambaScansRef, scan_layers: SambaLayersRef,
    ):
        self.ctx = ctx.copy()
        self.bsts = bsts^
        self.fwds = fwds^
        self.t_eps = t_eps^
        self.cos_tab = cos_tab
        self.sin_tab = sin_tab
        self.m3_w = m3_w^
        self.m3_st = m3_st^
        self.keep = keep
        self.ton = ton
        self.b = b
        self.l = l
        self.dm = dm
        self.vocab = vocab
        self.eps = eps
        self.seed = seed
        self.dh = dh
        self.dhn = dhn
        self.dlogits = dlogits
        self.ws_a = ws_a
        self.ws_b = ws_b
        self.dot_out = dot_out
        self.dh_tmp = dh_tmp
        self.dprod = dprod
        self.rstd = rstd
        self.dvcoef = dvcoef
        self.ones = ones
        self.sumsq = sumsq
        self.ids = ids
        self.counts = counts
        self.run_begin = run_begin
        self.perm = perm
        self.scans = scans
        self.scan_layers = scan_layers

    def _slot_check(self, slot: Int, n: Int, what: String) raises:
        if slot < 0 or slot >= n:
            raise Error(String("samba backward (autograd): node carries no valid ") + what + " slot")

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        # Never: every kernel overwrites every element of the destination it
        # is handed (the hand path reuses dhn, dh, dlogits, the stage buffers
        # and the gradient registry across steps with no fill), and the
        # transformer and Mamba-3 rules say the same of theirs.
        return False

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        if tag >= TF_RMS_NORM_INPUT and tag <= TF_RESIDUAL:
            # TransformerRules.writes_total reads only the tag, the edge and
            # saved.i0; any slot's instance answers for every layer
            if len(self.bsts) < 1:
                return False
            var tf = TransformerRules(
                self.ctx, self.bsts[0], self.fwds[0], self.cos_tab, self.sin_tab, self.t_eps[0], self.b, self.l, 0,
            )
            return tf.writes_total(tag, saved, edge)
        if tag == M3_BLOCK:
            return True
        # embedding, final norm, head, loss: each input has one consumer, the
        # plain first-contribution write in place (accumulation order, header)
        return False

    def supports(self, tag: Int) -> Bool:
        return (
            (tag >= TF_RMS_NORM_INPUT and tag <= TF_RESIDUAL)
            or tag == M3_BLOCK
            or (tag >= SAMBA_EMBEDDING and tag <= SAMBA_CE)
        )

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        # Every launch goes on self.ctx, the context `ex` wraps.
        if tag >= TF_RMS_NORM_INPUT and tag <= TF_RESIDUAL:
            var layer = saved.i11
            self._slot_check(layer, len(self.bsts), "attention")
            var tf = TransformerRules(
                self.ctx, self.bsts[layer], self.fwds[layer], self.cos_tab, self.sin_tab,
                self.t_eps[layer], self.b, self.l, 0,
            )
            if tag == TF_RESIDUAL and saved.i10 == 1:
                # the layer's output: the hand path's entry refusal of d_out
                self._refuse_nonfinite(upstream, rows[1] * cols[1], layer)
            tf.apply(ex, tag, saved, output, upstream, inputs, gradients, rows, cols)
        elif tag == M3_BLOCK:
            var slot = saved.i11
            self._slot_check(slot, len(self.m3_w), "Mamba-3")
            var m3 = Mamba3BlockRules(self.ctx, self.m3_w[slot], self.m3_st[slot], self.keep, self.b, self.l, self.ton)
            m3.apply(ex, tag, saved, output, upstream, inputs, gradients, rows, cols)
        elif tag == SAMBA_EMBEDDING:
            self._embedding(upstream, gradients)
        elif tag == SAMBA_NORM_F:
            self._norm_f(upstream, inputs, gradients)
        elif tag == SAMBA_HEAD:
            self._head(upstream, inputs, gradients)
        elif tag == SAMBA_CE:
            self._ce(upstream, gradients)
        else:
            raise Error("unknown Samba backward rule")

    def _refuse_nonfinite(self, upstream: FP, n: Int, layer: Int) raises:
        # llama_decoder_layer_backward_autograd's first device read: the
        # incoming gradient is refused by bits before any of its stages.
        var d = _view(self.ctx, upstream, n)
        comptime if AUTOGRAD_REFUSE_DEFER:
            # the verdict is read at the step's closing wait (the driver)
            self.scans[].append(DeferredNonfiniteScan(self.ctx, d, n))
            self.scan_layers[].append(layer)
            return
        var bad = device_first_nonfinite(self.ctx, d, n)
        if bad >= 0:
            raise Error(String("llama backward: NaN or infinity in d_residual2 at flat index ") + String(bad) + " REFUSED")

    def _embedding(self, upstream: FP, gradients: List[FP]) raises:
        # the hand path's embedding backward: the fixed-order scatter-add of
        # layer 0's input gradient (read in place) into the embedding
        # weight's destination (pair[0:vd] tied, grad[0] untied)
        var m = self.b * self.l
        var emb = EmbConfig.llama(self.vocab, self.dm)
        var dw = _view(self.ctx, gradients[0], self.vocab * self.dm)
        var dy = _view(self.ctx, upstream, m * self.dm)
        identical_embedding_backward_into(
            self.ctx, dw, dy, self.ids[], self.counts[], self.run_begin[], self.perm[], m, emb
        )

    def _norm_f(self, upstream: FP, inputs: List[FP], gradients: List[FP]) raises:
        # the hand path's final-norm backward on the forward's own row
        # statistics: bwd_rms_norm[0], dx into s.dh (no residual join)
        _same(upstream, self.dhn[], "the final norm output gradient")
        _same(gradients[0], self.dh[], "the final norm input gradient")
        var m = self.b * self.l
        var x = _view(self.ctx, inputs[0], m * self.dm)
        var nw = _view(self.ctx, inputs[1], self.dm)
        var dnw = _view(self.ctx, gradients[1], self.dm)
        var dh_p = self.dh[].unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
        bwd_rms_norm[0](
            self.ctx, self.dot_out[], self.dh[], dnw, self.dh_tmp[], self.dprod[], self.rstd[], self.dvcoef[],
            self.ones[], self.dhn[], x, nw, self.sumsq[],
            dh_p, dh_p, False, m, self.dm, self.eps,
        )

    def _head(self, upstream: FP, inputs: List[FP], gradients: List[FP]) raises:
        # forward logits [M, V] = hn [M, DM] . W [V, DM]^T (OP_NT): the hand
        # path's dA (ROLE_HEAD, ws_a) then dB (ws_b)
        _same(upstream, self.dlogits[], "the logits gradient")
        _same(gradients[0], self.dhn[], "the head input gradient")
        var m = self.b * self.l
        var hn = _view(self.ctx, inputs[0], m * self.dm)
        var hw = _view(self.ctx, inputs[1], self.vocab * self.dm)
        var dwh = _view(self.ctx, gradients[1], self.vocab * self.dm)
        identical_gemm_backward_a_into[ROLE_HEAD](
            self.ctx, self.dhn[], self.dlogits[], hw, self.ws_a[], m, self.vocab, self.dm, OP_NT
        )
        identical_gemm_backward_b_into(
            self.ctx, dwh, self.dlogits[], hn, self.ws_b[], m, self.vocab, self.dm, OP_NT
        )

    def _ce(self, upstream: FP, gradients: List[FP]) raises:
        # identical_ce_loss_resident left the sum-reduced loss's gradient
        # (divided by the target count) in dlogits during the forward: the
        # rule launches nothing, reads no upstream and takes only the unit seed.
        if Int(upstream) != Int(self.seed):
            raise Error("samba backward (autograd): the loss rule takes only the unit seed")
        _same(gradients[0], self.dlogits[], "the logits gradient")
