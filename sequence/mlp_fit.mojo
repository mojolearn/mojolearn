# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The multi-layer perceptron's epoch loop and prediction over `Exec`
(`sequence/mlp.mojo` holds the element bodies and the reference citations)."""
from sequence.exec_trait import Exec
from core.fit_loop_clock import fit_loop_start, fit_loop_stop
from sequence.fit_loop_words import fit_loop_publish
from sequence.mlp import (
    ACT_SOFTMAX,
    EPI_ACT_BWD,
    EPI_BIAS_ACT,
    EPI_L2GRAD,
    MLP_EPOCH_DEV,
    MLP_DEVICE_EPOCH_ORDER,
    MLP_L2_BLOCK,
    MLP_LOSSBLOCK,
    MLP_LB_ROWS,
    mlp_lb_block,
    mlp_lb_blocks,
    SplitMix,
    fisher_yates,
    mlp_epoch_key,
    mlp_perm_args,
)
from sequence.ops import (
    FP,
    OP_ONE_HOT,
    OP_PROBA2,
    OP_ACT,
    OP_ACT_BWD,
    OP_COLSUM_DIV,
    OP_COPY,
    OP_GATHER_ROWS,
    OP_GEMM_EPI,
    OP_AF_BLK_SUMSQ,
    OP_MLP_BLOSS,
    OP_MLP_L2FOLD,
    OP_MLP_L2PART,
    OP_MLP_PERM,
    OP_MLP_ROWLOSS,
    OP_MLP_LB_PART,
    OP_MLP_LB_FOLD,
    OP_MLP_LB_EPOCH_PART,
    OP_MLP_LB_EPOCH_FOLD,
    OP_OPT,
    OP_SOFTMAX,
    OP_SUMSQ,
    OPT_SK_ADAM,
    OPT_SK_SGD,
    Args,
    dummy_ptr,
)
from sequence.recurrent import bias_rows, gather_rows, gemm
from autograd.kernels.recurrent import colsum
from checks.numerics import ftz, identical_div, identical_mul, identical_pow64, identical_sqrt
from std.sys.compile import is_defined
from autograd.tape import Tape, Tensor
from autograd.ops import RELU as AG_RELU
from autograd.ops import SIGMOID as AG_SIGMOID
from autograd.ops import SOFTMAX as AG_SOFTMAX
from autograd.ops import TANH as AG_TANH
from autograd.data import AUTOGRAD_GATHER_EPOCH, gather_epoch, gather_epoch_fits
from autograd.fold import fold_partials_bound
from sequence.ops import dummy_ptr
from core.nan_canon import NAN_CANON_ON, nan_canon_f32

# The tape-assembled backward is the default (promoted 2026-10-18). A/B
# fad4bec3e, tools/sequence_speed.py case_mlp, HIGGS 1M x 28, (256,) b512
# 1 epoch adam, identical, one run per arm: NVIDIA L40S 0.334 -> 0.388 s
# (1.16x, slower), AMD MI325X 1.107 -> 0.602 s (0.54x); average of the two
# vendors 0.85x (the switch rule; vendor split flagged). Quality on 200k
# held-out rows: acc 0.68186 -> 0.68230, logloss 0.59475 -> 0.59462, AUC
# 0.74755 -> 0.74781. Same digest on NVIDIA and AMD for each arm
# (~/mojolearn-evidence/native-autograd-20261018/AB_MLP_RESULTS_2026-10-18.md).
# TOMBSTONE: MOJOLEARN_AUTOGRAD_MLP_OFF (DELETED-superseded) deleted 2026-10-19 by 6784245b2; code recoverable at 6784245b2^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_MLP_OFF.patch
# It was the hand-assembled backward (arm A above): per layer gemm_epi EPI_L2GRAD
# (dW), OP_COLSUM_DIV (db), gemm_epi EPI_ACT_BWD (dX); the tape trains every MLP
# with the same 48 check digests (autograd/checks/mlp_cases_digests.tsv).
# TOMBSTONE: MOJOLEARN_AUTOGRAD_PREDICT (folded in: the tape forward is the one predict path) deleted 2026-11-08 by c3b7dc42e6; code recoverable at c3b7dc42e6^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_PREDICT.patch
# Lane x3-mlp-rnn-predict-on-tape (plan plan EXECUTOR_CONSOLIDATION_AND_CLEANUP (git history) X3): mlp_predict runs
# mlp_forward_autograd on a tape in inference mode (what -D MOJOLEARN_AUTOGRAD_PREDICT selected); the hand
# mlp_forward (gemm_epi EPI_BIAS_ACT a layer on the one-thread OP_GEMM_EPI, + OP_SOFTMAX) and the hand
# mlp_predict over it are deleted. The fused tape node issues EPI_BIAS_ACT on the tiled GEMM, op_gemm's chain
# term for term, so the expected digest is the hand forward's; the one pre-release identity run is the proof.
# A/B owed: arm A f7d10d2b09 (hand), arm B c3b7dc42e6 (tape), mlp-predict and sequential-predict, NVIDIA and AMD.
# The fused tape node is the tape default (lane ag-pass3-impl, 2026-10-19,
# PASS3_IDEAS idea 1; lane ag-fusion2 wrote it as the opt-in
# MOJOLEARN_AUTOGRAD_MLP_FUSED). Each layer is one tape.linear_into node and
# the backward seeds p - y with backward_from_preactivation. Per layer that is
# the hand path's launches with the hand path's arguments, on the tiled GEMM
# (OP_AGK_GEMM_EPI) where the hand path runs the one-thread OP_GEMM_EPI:
# forward 1 (gemm_epi EPI_BIAS_ACT, + the softmax launch for a softmax
# output), backward dW gemm_epi EPI_L2GRAD with db from the same launch
# (AUTOGRAD_PASS3_DB: the tile's column-sum epilogue, or OP_COLSUM_DIV after
# a GEMM off the tile) and dX gemm_epi EPI_ACT_BWD; no seed copy, no
# L2GRAD/DIVS export.
# BIT VERSION against the composed tape default it replaces (digest
# aee3b17ef66dc060 on the HIGGS recipe: tools/sequence_speed.py case_mlp,
# HIGGS 1M x 28, (256,) b512 1 epoch adam, identical): act' is op_act_bwd's
# body instead of op_ag_unary_vjp's and db is OP_COLSUM_DIV's chain instead
# of the blocked fold. Expected digest: the deleted hand path's (2c2662ecb00cceed,
# -D MOJOLEARN_AUTOGRAD_MLP_OFF) on every column (host, Metal, NVIDIA, AMD),
# since every launch is a hand-path body with the hand path's fold order
# (the tiled GEMM is op_gemm's chain term for term).
# TOMBSTONE: MOJOLEARN_AUTOGRAD_MLP_UNFUSED (opt-in arm, the composed matmul + add + activation graph; no A/B recorded) deleted 2026-11-15 by lane/expdel2-29-neural; code recoverable at a03ced9b31.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_MLP_UNFUSED.patch
# With it went MLP_COMPOSED_OPT_FUSE (the composed graph's export in the optimizer launch), so this file no longer reads
# autograd/optim.mojo AUTOGRAD_OPT_FUSE.
# TOMBSTONE: MOJOLEARN_AUTOGRAD_OPT_FUSE_OFF (old path) read deleted here 2026-11-15 by lane/expdel2-29-neural; code recoverable at a03ced9b31.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_OPT_FUSE_OFF.expdel2-29-neural.patch
# TOMBSTONE: MOJOLEARN_AUTOGRAD_MLP_SEED_SCALE (DROPPED-noise) deleted 2026-10-18 on
# feature/native-autograd-20261018; code recoverable at 91ebd1cb4 (never on main, no patch).
# It folded the 1/B into the backward seed (one divide of the B x O seed, then L2GRAD with
# alpha/B and n = 1, no bias divide), saving L - 1 launches a batch. A/B on HIGGS 1M
# (256,) b512: NV 0.388 -> 0.387 s, AMD 0.602 -> 0.599 s, same digest as the default
# (B = 512 is a power of two, so the scaling rounds identically); noise.
# Restore: git checkout 91ebd1cb4 -- sequence/mlp_fit.mojo (then merge by hand).
# TOMBSTONE: MOJOLEARN_AUTOGRAD_LOSS (opt-in arm, no A/B; the default reported-loss folds kept) deleted 2026-11-08 by dbab55bea7; code recoverable at c3b7dc42e6^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_LOSS.patch
# Lane x3-mlp-rnn-predict-on-tape (plan EXECUTOR_CONSOLIDATION_AND_CLEANUP 5.6): the opt-in arm computed each
# batch's reported data term through autograd/losses.mojo (loss_value_into, _mlp_loss_kind) and handed that one sum
# to the loss block / op_mlp_bloss; new bits in the reported loss and possibly n_iter, parameters untouched. No A/B
# ever ran, so today's bits-defining default (the row losses' blocked folds) is the one arm; the A/B is owed.

# TOMBSTONE: MOJOLEARN_AUTOGRAD_OPTIM (promoted: the executor optimizer is the trainers' one update; the hand arm deleted) deleted 2026-11-08 by 75825f479c; code recoverable at c3b7dc42e6^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_OPTIM.patch
# Lane x3-mlp-rnn-predict-on-tape (plan EXECUTOR_CONSOLIDATION_AND_CLEANUP 5.6; Andrew 2026-11-08: keep the
# simpler path): the parameters update through autograd/optim.mojo (OptimizerState + step_lr, step_lr_finish for
# the composed graph), the optimizer Sequential and ops.Session already use, instead of the hand-built OP_OPT
# launch (its Adam running products pw1 / pw2 and word assembly, a copy of optimizer_scalars) that is deleted.
# Bits: the same. step_lr issues the same one OP_OPT launch over Gr with the same words the body reads (sklearn
# Adam: f1 b1, f2 b2, f3 eps, f5 lr sqrt(1 - b2^t) / (1 - b1^t) from running products advanced by one identical_mul
# a step; sklearn SGD: f0 lr, f1 momentum, i2 Nesterov); the words it sets that the hand path left 0 are not read
# by those bodies. A/B owed (ms per fit, mlp-fit rows).
# Restored 2026-11-09 by lane r2-restore-unmeasured-defaults (AGENTS.md "A measured winner is never deleted on
# reasoning": the promotion had no measurement): the hand-built OP_OPT launch is the default again, as it was before
# X3 (the define's default at f7d10d2b09 was off).
# TOMBSTONE: MOJOLEARN_AUTOGRAD_OPTIM (opt-in arm restored 2026-11-09, no A/B recorded) deleted 2026-11-15 by lane/expdel2-29-neural; code recoverable at ff6c789744.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_OPTIM.2026-11-15.patch
# TOMBSTONE: MOJOLEARN_AUTOGRAD_OPTIM (decided B arm, the executor optimizer step_lr; L40S ba08dac006 gru/lstm/mlp/rnn fit B 0.99-1.01x, same speed) deleted 2026-11-17 by lane/fb12-loser-arms; code recoverable at 2203db7094.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_OPTIM.fb12-loser-arms.patch; record in docs/TOMBSTONES.md.

comptime SOLVER_ADAM = 0
comptime SOLVER_SGD = 1
comptime LR_CONSTANT = 0
comptime LR_INVSCALING = 1
comptime LR_ADAPTIVE = 2


struct MLPNet(Copyable, Movable):
    var sizes: List[Int]   # D, h1, ..., O
    var act: Int
    var out_act: Int

    def __init__(out self, var sizes: List[Int], act: Int, out_act: Int):
        self.sizes = sizes^
        self.act = act
        self.out_act = out_act

    def n_layers(self) -> Int:
        return len(self.sizes) - 1

    def w_off(self, i: Int) -> Int:
        """Offset of layer i's coefs (i = 0 .. n_layers - 1); its intercepts
        follow them."""
        var off = 0
        for k in range(i):
            off += self.sizes[k] * self.sizes[k + 1] + self.sizes[k + 1]
        return off

    def b_off(self, i: Int) -> Int:
        return self.w_off(i) + self.sizes[i] * self.sizes[i + 1]

    def n_params(self) -> Int:
        return self.w_off(self.n_layers())


# TOMBSTONE: MOJOLEARN_AUTOGRAD_PREDICT (folded in: the tape forward is the one predict path) deleted 2026-11-08 by c3b7dc42e6; code recoverable at c3b7dc42e6^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_PREDICT.patch
# gemm_epi (one OP_GEMM_EPI launch) and _act_launch (OP_SOFTMAX / OP_ACT in place) had one caller, the hand
# mlp_forward; both went with it. The tape issues the same kernels through linear_into.


# TOMBSTONE: MOJOLEARN_AUTOGRAD_PREDICT (folded in: the tape forward is the one predict path) deleted 2026-11-08 by c3b7dc42e6; code recoverable at c3b7dc42e6^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_PREDICT.patch
# mlp_forward (the hand forward, its one caller the hand mlp_predict) is gone: mlp_forward_autograd below is the
# only MLP forward, for training and for prediction (inference mode).
#
# RESTORED 2026-11-17 (lane fb-neural; AGENTS.md "A measured winner is never deleted on reasoning"): the X3
# fold-in had no measurement ("Numbers: none. RUN OWED"), and the hand predict was 0.8.55's measured default.
# The full board on one L40S (wheel d95f805cce against 0.8.55, ~/mojolearn-evidence/measure-candidate/
# build-d95f805cce/full/extra-vs-0855.tsv) shows the tape predict slower with the same digest on every MLP
# infer row: mlp-clf istella 42.6 -> 130.7 ms (Xq 100000 x 220, (256, 256), 5 classes), mlp-reg istella
# 26.4 -> 64.7, mlp-clf taxi 15.1 -> 25.3, mlp-reg taxi 6.8 -> 10.1. So gemm_epi, _act_launch, mlp_forward and
# the hand mlp_predict come back as the default (code from f7d10d2b09, unchanged), and
# -D MOJOLEARN_AUTOGRAD_PREDICT (default off) took the tape predict (mlp_predict_autograd) for the A/B.
# Same bits on both arms (the tape node is op_gemm's chain term for term; the two wheels' board digests match).
# VERDICT 2026-11-18 (A/B ab1, owed rows autograd-predict-mlp / -mlp-predict, NVIDIA L40S host 37f71a6102af,
# driver 595.112.07, wheel d652a4b0a1, base and arm back to back in one job, one run per arm): the tape predict
# is SLOWER on all four infer cells, mlp-clf istella 148.3 -> 154.7 ms (1.04), mlp-clf taxi 21.48 -> 22.46
# (1.05), mlp-reg taxi 10.49 -> 11.08 (1.06), mlp-reg istella 74.72 -> 79.21 (1.06); fits equal (0.99-1.00),
# neural/mlp-predict 0.96 (0.089 ms, a sub-millisecond cell); digests equal on all 9 cells. LOSER: the arm
# (mlp_predict_autograd and the switch) is deleted; the hand mlp_predict is the one MLP predict. Limits: NVIDIA
# only (AGENTS.md part 8), one run per arm.
# TOMBSTONE: MOJOLEARN_AUTOGRAD_PREDICT (DROPPED-slower: tape predict 1.04-1.06x on the four MLP infer cells, NV L40S 2026-11-18) deleted 2026-11-18 by lane/fb15-ab-verdicts; code recoverable at c989eb9222.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_PREDICT-2026-11-18.patch; record in docs/TOMBSTONES.md.


def gemm_epi[E: Exec](
    mut ex: E, A: FP, Bm: FP, C: FP, M: Int, N: Int, K: Int,
    sam: Int, sak: Int, sbk: Int, sbn: Int, epi: Int, p3: FP, i10: Int, f0: Float32, f1: Float32,
) raises:
    """`gemm` (dense C, no accumulate) with its follower in the same launch
    (mlp.mojo::op_gemm_epi)."""
    var a = Args()
    a.p0 = A
    a.p1 = Bm
    a.p2 = C
    a.p3 = p3
    a.i0 = M
    a.i1 = N
    a.i2 = K
    a.i3 = sam
    a.i4 = sak
    a.i5 = sbk
    a.i6 = sbn
    a.i8 = N
    a.i9 = epi
    a.i10 = i10
    a.f0 = f0
    a.f1 = f1
    ex.launch[OP_GEMM_EPI](a, M * N)


def _act_launch[E: Exec](mut ex: E, z: FP, rows: Int, cols: Int, act: Int) raises:
    var a = Args()
    a.p0 = z
    a.p1 = z
    a.i0 = act
    a.i1 = cols
    if act == ACT_SOFTMAX:
        ex.launch[OP_SOFTMAX](a, rows)
    elif act != 0:
        ex.launch[OP_ACT](a, rows * cols)


def mlp_forward[E: Exec](mut ex: E, net: MLPNet, P: FP, acts: List[FP], B: Int) raises:
    """acts[0] holds the batch [B, D]; acts[i] <- act(acts[i-1] @ W_i + b_i)."""
    var L = net.n_layers()
    for i in range(L):  # driver-loop: over layers
        var fi = net.sizes[i]
        var fo = net.sizes[i + 1]
        # one launch: the GEMM, its bias and (not softmax) its activation
        var act = net.out_act if i == L - 1 else net.act
        gemm_epi(ex, acts[i], P + net.w_off(i), acts[i + 1], B, fo, fi, fi, 1, fo, 1,
                 EPI_BIAS_ACT, P + net.b_off(i), 0 if act == ACT_SOFTMAX else act, Float32(0.0), Float32(0.0))
        if act == ACT_SOFTMAX:
            _act_launch(ex, acts[i + 1], B, fo, act)


def _mlp_autograd_unary(act: Int) -> Int:
    """The tape opcode of an MLP activation code, 0 for identity."""
    if act == 1:
        return AG_SIGMOID
    if act == 2:
        return AG_TANH
    if act == 3:
        return AG_RELU
    if act == ACT_SOFTMAX:
        return AG_SOFTMAX
    return 0


def mlp_forward_autograd[E: Exec](
    mut tape: Tape, mut ex: E, net: MLPNet, P: FP, Gr: FP, acts: List[FP], B: Int,
    alpha: Float32 = 0,
) raises -> Tensor:
    # The graph is recorded from the forward computation. No reverse-layer
    # loop or manually connected activation gradients remains in this path.
    # Each layer's output is recorded straight into acts[i + 1] (no copy) and
    # each parameter's gradient lands in its slot of Gr (no export copy).
    tape.reset()
    var h = tape.variable(ex, acts[0], B, net.sizes[0], False)
    # One fused node a layer; its weight gradient comes out of the GEMM
    # as (dW + alpha W) / B and its bias gradient as colsum / B, the
    # hand path's EPI_L2GRAD and OP_COLSUM_DIV. Returns the output
    # layer's node; its seed is the pre-activation gradient p - y.
    for i in range(net.n_layers()):  # driver-loop: over layers
        var fw = tape.variable_into(ex, P + net.w_off(i), Gr + net.w_off(i), net.sizes[i], net.sizes[i + 1])
        var fb = tape.variable_into(ex, P + net.b_off(i), Gr + net.b_off(i), 1, net.sizes[i + 1])
        var fact = net.out_act if i == net.n_layers() - 1 else net.act
        h = tape.linear_into(ex, h, fw, Optional[Tensor](fb), _mlp_autograd_unary(fact), acts[i + 1],
                             alpha, Float32(B))
    return h


def mlp_backward_autograd[E: Exec](
    mut tape: Tape, mut ex: E, net: MLPNet, logits: Tensor, delta: FP, P: FP, Gr: FP,
    B: Int, alpha: Float32,
) raises:
    # LAUNCHES PER LAYER, read from the code (lane ag-fusion). Before: the
    # forward recorded matmul, add and activation (3) plus 5 record-time
    # gradient fills (W, b, product, logits, activation) plus the copy into
    # acts = 9; the backward refilled the same 5 gradients, ran each VJP into
    # scratch and copied it into the parent (activation 2, add 2 + 2, matmul
    # 2 + 2, the first layer's matmul 2), then exported with copy + L2GRAD +
    # copy + DIVS (4): 19 for a hidden layer above the first, 17 for the
    # first and last. About 28 a layer; the default trainer runs 4 (1 + 3).
    # After (with lane ag-metal-folds' per-opcode bodies): forward 3 (2 for
    # an identity layer), no fills, no copy; backward activation VJP 1, the
    # product's add VJP 1, the bias's add VJP 3 (elementwise VJP, then the
    # two-launch blocked fold over the batch rows), matmul GEMMs 2 (1 for the
    # first layer), all straight into the parents' gradients and Gr, plus
    # L2GRAD + DIVS 2: 9 above the first layer, 8 for the first and last.
    # Per batch on top: the seed copy into the root gradient (1). (The base
    # tape's bias VJP was one launch with a serial
    # per-column fold; the blocked fold replaces it.)
    # Lane ag-pass5-composed (read from the code; HIGGS (256,), O 1): 25 a
    # batch -> 15 (+1 an epoch): the per-batch gather -> one an epoch
    # (AUTOGRAD_GATHER_EPOCH), the seed copy and the product's add VJP a
    # layer borrowed (AUTOGRAD_GRAD_ALIAS), the output layer's matmul dX + dW
    # one launch (AUTOGRAD_PAIR), and L2GRAD + DIVS (4) inside the optimizer
    # launch (MLP_COMPOSED_OPT_FUSE). Every one keeps the same words.
    #
    # Existing row-loss produces the combined loss/link derivative p-y with
    # respect to LOGITS, not probabilities. Seed that precise graph boundary.
    var O = net.sizes[net.n_layers()]
    var seed = tape.variable(ex, delta, B, O, False)
    # LAUNCHES PER BATCH (lane ag-pass3-impl, read from the code; HIGGS
    # MLP D 28, (256,), O 1, b512, adam, NVIDIA/AMD identical):
    #   tape default before this pass (composed graph, below): 28 =
    #     gather 1, forward 6, loss 5, seed copy 1, backward 11, export 4.
    #   now: 13 = gather 1; forward 2 (gemm_epi EPI_BIAS_ACT a layer);
    #     loss 5 (the trainer's block, unchanged); backward 4: layer 2 dW (N = 1, off
    #     the tile: one-thread gemm_epi + OP_COLSUM_DIV) 2, dH (gemm_epi
    #     EPI_ACT_BWD, tiled) 1, layer 1 dW + db (tiled, column-sum
    #     epilogue) 1, no dX; optimizer 1.
    #   deleted hand path (AUTOGRAD_MLP_OFF): 14 (gather 1, forward 2, loss 5,
    #     backward 5, optimizer 1); the tape now issues one launch fewer
    #     (db rides in the tiled dW1 launch).
    # Per layer: backward 1 on the tile (dW with db) + 1 (dX with the
    # layer below's act'); dX absent for the first layer; +1 OP_COLSUM_DIV
    # when dW falls off the tile (an edge under GT_MIN_EDGE, Apple, the
    # host). No seed copy (the seed is the output node's dZ, read in
    # place) and no export.
    tape.backward_from_preactivation(ex, logits, seed)


def _l2_blocked[E: Exec](mut ex: E, net: MLPNet, P: FP, l2: FP, parts: FP) raises:
    """MLP_BLOCKED_FOLDS (sequence/mlp.mojo): l2[i] = ||W_i||^2 in the blocked
    order. Up to four layers: one launch of every block, one of the folds;
    more layers: one block launch and one fold per layer."""
    var L = net.n_layers()
    if L <= 4:
        var q = Args()
        q.p0 = P
        q.p1 = parts
        q.i1 = MLP_L2_BLOCK
        q.i2 = L
        var nb = 0
        for i in range(L):
            var o_ = net.w_off(i)
            var c_ = net.sizes[i] * net.sizes[i + 1]
            nb += (c_ + MLP_L2_BLOCK - 1) // MLP_L2_BLOCK
            if i == 0:
                q.i4 = o_
                q.i5 = c_
            elif i == 1:
                q.i6 = o_
                q.i7 = c_
            elif i == 2:
                q.i8 = o_
                q.i9 = c_
            else:
                q.i10 = o_
                q.i11 = c_
        ex.launch[OP_MLP_L2PART](q, nb)
        var f = q
        f.p0 = parts
        f.p1 = l2
        ex.launch[OP_MLP_L2FOLD](f, L)
        return
    var poff = 0
    for i in range(L):
        var c_ = net.sizes[i] * net.sizes[i + 1]
        var nb = (c_ + MLP_L2_BLOCK - 1) // MLP_L2_BLOCK
        var b = Args()
        b.p0 = P + net.w_off(i)
        b.p1 = parts + poff
        b.i0 = c_
        b.i1 = MLP_L2_BLOCK
        ex.launch[OP_AF_BLK_SUMSQ](b, nb)
        var f = Args()
        f.p0 = parts + poff
        f.p1 = l2
        f.i0 = i
        f.i1 = nb
        ex.launch[OP_MLP_L2FOLD](f, 1)
        poff += nb


# ------------------------------------------------------------------ pass-3 loss block (MLP_LOSSBLOCK)
@always_inline
def _lb_set_layer(mut a: Args, j: Int, off: Int, cnt: Int):
    """op_mlp_lb_part's layer j (offset, count) slot, op_sumsq's packing."""
    if j == 0:
        a.i4 = off
        a.i5 = cnt
    elif j == 1:
        a.i6 = off
        a.i7 = cnt
    elif j == 2:
        a.i8 = off
        a.i9 = cnt
    else:
        a.i10 = off
        a.i11 = cnt


@always_inline
def _lb_set_count(mut a: Args, j: Int, cnt: Int):
    """op_mlp_lb_fold's layer j count slot (i6..i9)."""
    if j == 0:
        a.i6 = cnt
    elif j == 1:
        a.i7 = cnt
    elif j == 2:
        a.i8 = cnt
    else:
        a.i9 = cnt


def mlp_lb_l2_parts(net: MLPNet) -> Int:
    """Floats of L2 block-partial storage the loss block needs for `net`."""
    var n = 0
    for i in range(net.n_layers()):  # driver-loop: over layers
        n += mlp_lb_blocks(net.sizes[i] * net.sizes[i + 1])
    return n


def mlp_lb_row_parts(max_rows: Int) -> Int:
    """Floats of row block-partial storage covering any batch of up to
    `max_rows` rows (autograd/fold.mojo fold_partials_bound: R(max_rows)
    blocks always suffice)."""
    return fold_partials_bound(max_rows)


def mlp_loss_block[E: Exec, //, ROWS: Bool = MLP_LB_ROWS](
    mut ex: E, net: MLPNet, P: FP, pred: FP, y: FP, seed: FP, rowparts: FP, l2parts: FP, l2: FP,
    dloss: FP, bi: Int, B: Int, loss_kind: Int, half_alpha: Float32, rowloss: FP,
) raises:
    """MLP_LOSSBLOCK (sequence/mlp.mojo): one batch's loss block. seed[0:B O]
    = pred - y, l2[i] = ||W_i||^2 and dloss[bi] = the batch's reported loss
    (data term + half_alpha sum(l2) / B), every sum a blocked fold with the
    tape's block rule. The data term is the B row losses' blocked sum (row
    partials written to `rowparts`; the handed-in data sum of the deleted
    -D MOJOLEARN_AUTOGRAD_LOSS arm went with it, lane X3). Launches: one op_mlp_lb_part and one op_mlp_lb_fold per
    group of four layers (2 for L <= 4).
    ROWS (MLP_LB_ROWS, the default; False is the check's reference):
    the row losses one thread per row into `rowloss` (B floats), the row
    blocks and the per-layer norms folded by many threads, then a one-thread
    batch loss: one more launch (3 for L <= 4), the same bits."""
    comptime if ROWS:
        _mlp_loss_block_rows(
            ex, net, P, pred, y, seed, rowparts, l2parts, l2, dloss, bi, B, loss_kind, half_alpha, rowloss,
        )
        return
    var L = net.n_layers()
    var O = net.sizes[L]
    var groups = (L + 3) // 4
    var pstart = 0
    for g in range(groups):  # driver-loop: over groups of four layers
        var p = Args()
        p.p0 = pred
        p.p1 = y
        p.p2 = seed
        p.p3 = rowparts
        p.p4 = P
        p.p5 = l2parts + pstart
        p.i0 = loss_kind
        p.i1 = O
        p.i2 = B if g == 0 else 0
        var threads = (B * O + mlp_lb_blocks(B)) if g == 0 else 0
        for j in range(4):  # driver-loop: over the group's layers
            var l = 4 * g + j
            if l < L:
                var c = net.sizes[l] * net.sizes[l + 1]
                _lb_set_layer(p, j, net.w_off(l), c)
                threads += mlp_lb_blocks(c)
                pstart += mlp_lb_blocks(c)
        ex.launch[OP_MLP_LB_PART](p, threads)
    pstart = 0
    for g in range(groups):  # driver-loop: over groups of four layers
        var f = Args()
        f.p0 = rowparts
        f.p1 = l2parts + pstart
        f.p2 = dloss
        f.p3 = l2
        f.i0 = bi
        f.i1 = B
        f.i2 = O
        f.i3 = L
        f.i4 = loss_kind
        f.i5 = mlp_lb_blocks(B)
        f.i10 = 4 * g
        f.i11 = 1 if g == groups - 1 else 0
        f.f0 = half_alpha
        for j in range(4):  # driver-loop: over the group's layers
            var l = 4 * g + j
            if l < L:
                var c = net.sizes[l] * net.sizes[l + 1]
                _lb_set_count(f, j, c)
                pstart += mlp_lb_blocks(c)
        ex.launch[OP_MLP_LB_FOLD](f, 1)


def _mlp_loss_block_rows[E: Exec](
    mut ex: E, net: MLPNet, P: FP, pred: FP, y: FP, seed: FP, rowparts: FP, l2parts: FP, l2: FP,
    dloss: FP, bi: Int, B: Int, loss_kind: Int, half_alpha: Float32, rowloss: FP,
) raises:
    """mlp_loss_block under MLP_LB_ROWS (sequence/mlp.mojo's loss-section
    comment): per group of four layers one op_mlp_lb_part (group 0: B O seed
    threads and B row threads) and one
    op_mlp_lb_fold with i11 = 2 (group 0: the row blocks; every group: one
    thread per layer), then one op_mlp_lb_fold with i11 = 3 (1 thread)."""
    var L = net.n_layers()
    var O = net.sizes[L]
    var groups = (L + 3) // 4
    var rows = B
    var pstart = 0
    for g in range(groups):  # driver-loop: over groups of four layers
        var p = Args()
        p.p0 = pred
        p.p1 = y
        p.p2 = seed
        p.p3 = rowloss
        p.p4 = P
        p.p5 = l2parts + pstart
        p.i0 = loss_kind
        p.i1 = O
        p.i2 = B if g == 0 else 0
        p.i3 = 1
        var threads = (B * O + rows) if g == 0 else 0
        for j in range(4):  # driver-loop: over the group's layers
            var l = 4 * g + j
            if l < L:
                var c = net.sizes[l] * net.sizes[l + 1]
                _lb_set_layer(p, j, net.w_off(l), c)
                threads += mlp_lb_blocks(c)
                pstart += mlp_lb_blocks(c)
        ex.launch[OP_MLP_LB_PART](p, threads)
    pstart = 0
    for g in range(groups):  # driver-loop: over groups of four layers
        var f = Args()
        f.p0 = rowloss
        f.p1 = l2parts + pstart
        f.p3 = l2
        f.p4 = rowparts
        f.i1 = B
        f.i5 = mlp_lb_blocks(rows) if g == 0 else 0
        f.i10 = 4 * g
        f.i11 = 2
        var nl = 0
        for j in range(4):  # driver-loop: over the group's layers
            var l = 4 * g + j
            if l < L:
                var c = net.sizes[l] * net.sizes[l + 1]
                _lb_set_count(f, j, c)
                pstart += mlp_lb_blocks(c)
                nl += 1
        ex.launch[OP_MLP_LB_FOLD](f, f.i5 + nl)
    var b = Args()
    b.p0 = rowparts
    b.p2 = dloss
    b.p3 = l2
    b.i0 = bi
    b.i1 = B
    b.i2 = O
    b.i3 = L
    b.i4 = loss_kind
    b.i5 = mlp_lb_blocks(B)
    b.i11 = 3
    b.f0 = half_alpha
    ex.launch[OP_MLP_LB_FOLD](b, 1)


def mlp_epoch_loss_block[E: Exec](
    mut ex: E, dloss: FP, parts: FP, dcurve: FP, n_batches: Int, bs: Int, N: Int, it: Int,
) raises:
    """MLP_LOSSBLOCK: dcurve[it] = (sum over batches of dloss[bi] * B_bi) / N
    as a blocked fold (blocks of mlp_lb_block(n_batches) batches, one fma
    each ascending from +0.0, then the partials ascending): two launches an
    epoch. `parts` holds mlp_lb_row_parts(n_batches) floats."""
    var R = mlp_lb_block(n_batches)
    var e = Args()
    e.p0 = dloss
    e.p1 = parts
    e.i0 = n_batches
    e.i1 = bs
    e.i2 = N
    e.i4 = R
    ex.launch[OP_MLP_LB_EPOCH_PART](e, mlp_lb_blocks(n_batches))
    var f = Args()
    f.p0 = parts
    f.p2 = dcurve
    f.i0 = mlp_lb_blocks(n_batches)
    f.i2 = N
    f.i3 = it
    ex.launch[OP_MLP_LB_EPOCH_FOLD](f, 1)


def mlp_fit[E: Exec](
    mut ex: E, net: MLPNet, X: FP, Y: FP, N: Int, Pio: FP, curve: FP,
    loss_kind: Int, solver: Int, lr_sched: Int, nesterov: Bool, batch: Int, max_iter: Int,
    shuffle: Bool, seed: UInt64, n_iter_no_change: Int,
    lr_init: Float32, b1: Float32, b2: Float32, eps: Float32, momentum: Float32,
    power_t: Float64, alpha: Float32, tol: Float64, y_codes: Int = 0, loop_words: Int = 0,
) raises -> Int:
    """Returns n_iter; curve[0:n_iter] is the epoch loss (float32 of the
    float64 accumulation; under MLP_EPOCH_DEV the float32 fold of
    `op_mlp_epoch_loss`). `y_codes` (cpu2-l11-neural): 0, `Y` is the
    dense (N, O) target; 1, `Y` holds N int32 class codes and the (N, O)
    one-hot target is built on the executor (`OP_ONE_HOT`); 2 (O = 1), the
    codes themselves as the float target. Not in host NumPy either way.
    `loop_words` (lane alm-neural, 2026-11-24): the address of the
    `FitLoop` words (sequence/fit_loop_words.mojo), or 0. The loop region
    is the epoch loop below: the clock is read once before epoch 0's first
    launch (after the uploads and allocations) and once when the loop ends;
    every epoch already ends on `ex.sync()` and its loss readback, so the
    last epoch's work is done there (no synchronization added, nothing
    inside the loop); its iterations are the epochs run (`n_iter`)."""
    # TOMBSTONE: MOJOLEARN_AUTOGRAD_MLP_OFF (DELETED-superseded) deleted 2026-10-19 by 6784245b2; code recoverable at 6784245b2^.
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_MLP_OFF.patch
    # The removed refusal lives here, not in core/six_lane_experiment_guards.mojo: a core line rebuilds 76 bindings (rebuild fence).
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_MLP_OFF"](), "removed 2026-10-19 (lane/del-mlp, 6784245b2): AUTOGRAD_MLP_OFF (the hand-assembled MLP backward) was superseded by the tape: same 48 check digests on host, Metal, NVIDIA and AMD; HIGGS 1M (256,) b512 hand -> fused tape NV 0.476 -> 0.526 s, AMD 0.408 -> 0.366 s, digest 8f410a4ae2d620af on both arms; recoverable at 6784245b2^"
    var L = net.n_layers()
    var D = net.sizes[0]
    var O = net.sizes[L]
    var np_ = net.n_params()
    var bs = batch if batch < N else N
    var dX = ex.alloc(N * D)
    ex.upload(dX, X, N * D)
    var dY = ex.alloc(N * O)
    if y_codes == 1 or (y_codes == 2 and O == 1):
        var dC = ex.alloc(N)
        ex.upload(dC, Y, N)
        var oh = Args()
        oh.p0 = dC
        oh.p1 = dY
        oh.i0 = O
        oh.i1 = 1 if y_codes == 2 else 0
        ex.launch[OP_ONE_HOT](oh, N * O)
    else:
        ex.upload(dY, Y, N * O)
    var P = ex.alloc(np_)
    ex.upload(P, Pio, np_)
    var Gr = ex.alloc(np_)
    # TOMBSTONE: MOJOLEARN_AUTOGRAD_OPTIM (promoted: the executor optimizer is the trainers' one update; the hand arm deleted) deleted 2026-11-08 by 75825f479c; code recoverable at c3b7dc42e6^.
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_OPTIM.patch
    # Restored 2026-11-09 (lane r2-restore-unmeasured-defaults): the hand OP_OPT state s1, s2, s3 is the default.
    # TOMBSTONE: MOJOLEARN_AUTOGRAD_OPTIM (decided B arm, the executor optimizer step_lr; L40S ba08dac006 gru/lstm/mlp/rnn fit B 0.99-1.01x, same speed) deleted 2026-11-17 by lane/fb12-loser-arms; code recoverable at 2203db7094.
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_OPTIM.fb12-loser-arms.patch; record in docs/TOMBSTONES.md.
    var s1 = ex.alloc(np_)
    var s2 = ex.alloc(np_)
    var s3 = ex.alloc(np_)
    var acts = List[FP]()
    var tape = Tape()
    var tape_logits = Tensor(0, 0, -1)
    for i in range(L + 1):
        acts.append(ex.alloc(bs * net.sizes[i]))
    # the gradient seed p - y at the logits (OP_MLP_ROWLOSS / the loss block)
    var dlogits = ex.alloc(bs * O)
    var yb = ex.alloc(bs * O)
    # Lane ag-pass5-composed (PASS5_IDEAS idea 6), AUTOGRAD_GATHER_EPOCH: when
    # the permuted copy of X and y fits (autograd/data.mojo
    # gather_epoch_fits, a memory rule), one gather per epoch writes it and
    # each batch reads its rows as a contiguous slice: acts[0] and yb point
    # into the copy. The same words (op_gather_rows either way).
    var epoch_gather = False
    var Xp = dummy_ptr()
    var Yp = dummy_ptr()
    comptime if AUTOGRAD_GATHER_EPOCH:
        epoch_gather = gather_epoch_fits(N, D + O)
        if epoch_gather:
            Xp = ex.alloc(N * D)
            Yp = ex.alloc(N * O)
    var rowloss = ex.alloc(bs)
    var l2 = ex.alloc(L)
    var n_batches = (N + bs - 1) // bs
    var dloss = ex.alloc(n_batches)
    var didx = ex.alloc(N)
    # MLP_LOSSBLOCK: the loss block's L2, row and epoch block partials
    var lb_l2parts = ex.alloc(mlp_lb_l2_parts(net) if MLP_LOSSBLOCK else 1)
    var lb_rowparts = ex.alloc(mlp_lb_row_parts(bs) if MLP_LOSSBLOCK else 1)
    var lb_epochparts = ex.alloc(mlp_lb_row_parts(n_batches) if MLP_LOSSBLOCK else 1)
    # TOMBSTONE: MOJOLEARN_AUTOGRAD_LOSS (opt-in arm, no A/B; the default reported-loss folds kept) deleted 2026-11-08 by dbab55bea7; code recoverable at c3b7dc42e6^.
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_LOSS.patch
    # MLP_EPOCH_DEV: the epoch losses stay on the executor, one word read back
    var dcurve = ex.alloc(max_iter if MLP_EPOCH_DEV else 1)
    var loss_word = List[Float32](length=1, fill=Float32(0.0))
    var perm = List[Float32]()
    comptime if not MLP_DEVICE_EPOCH_ORDER:
        for i in range(N):
            perm.append(Float32(i))
    var rng = SplitMix(seed)
    var host_loss = List[Float32](length=n_batches, fill=Float32(0.0))
    var best = Float64(1.0e308)
    var no_imp = 0
    var lr = lr_init
    var pw1 = Float32(1.0)
    var pw2 = Float32(1.0)
    var t_samples = 0
    var n_iter = 0
    # lane alm-left: the queued setup (uploads, fills) drained ONCE before the
    # clock, outside the loop, so the loop region holds loop work only
    ex.sync()
    var loop_start = fit_loop_start()
    for it in range(max_iter):
        comptime if MLP_EPOCH_DEV:
            if shuffle:
                ex.launch[OP_MLP_PERM](mlp_perm_args(N, mlp_epoch_key(seed, it), didx), N)
            elif it == 0:
                comptime if MLP_DEVICE_EPOCH_ORDER:
                    var identity_order = mlp_perm_args(N, UInt64(0), didx)
                    identity_order.i6 = 1
                    ex.launch[OP_MLP_PERM](identity_order, N)
                else:
                    ex.upload(didx, FP(unsafe_from_address=Int(perm.unsafe_ptr())), N)
        else:
            if shuffle:
                fisher_yates(rng, perm)
            ex.upload(didx, FP(unsafe_from_address=Int(perm.unsafe_ptr())), N)
        comptime if AUTOGRAD_GATHER_EPOCH:
            # the order changes only when shuffling: an unshuffled fit keeps epoch 0's copy
            if epoch_gather and (shuffle or it == 0):
                gather_epoch(ex, dX, dY, didx, N, D, O, Xp, Yp)
        for bi in range(n_batches):
            var off = bi * bs
            var B = bs if off + bs <= N else N - off
            var gathered = False
            comptime if AUTOGRAD_GATHER_EPOCH:
                if epoch_gather:
                    acts[0] = Xp + off * D
                    yb = Yp + off * O
                    gathered = True
            if not gathered:
                # X and y rows in one launch (apple2; the same copies)
                var gx = Args()
                gx.p0 = dX
                gx.p1 = didx
                gx.p2 = acts[0]
                gx.p3 = dY
                gx.p4 = yb
                gx.i0 = B
                gx.i1 = D
                gx.i2 = off
                gx.i3 = O
                ex.launch[OP_GATHER_ROWS](gx, B * (D + O))
            # TOMBSTONE: MOJOLEARN_AUTOGRAD_MLP_OFF (DELETED-superseded) deleted 2026-10-19 by 6784245b2; code recoverable at 6784245b2^.
            # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_MLP_OFF.patch
            tape_logits = mlp_forward_autograd(tape, ex, net, P, Gr, acts, B, alpha)
            comptime if MLP_LOSSBLOCK:
                # lane ag-pass3-lossblock: seed, row-loss and L2 sums and the
                # batch loss in 2 launches (was 5; sequence/mlp.mojo's
                # loss-section comment). FAST: below.
                # lane ag-pass5-loss (MLP_LB_ROWS): row losses one thread per
                # row into rowloss, 3 launches, the same bits.
                mlp_loss_block(
                    ex, net, P, acts[L], yb, dlogits, lb_rowparts, lb_l2parts, l2, dloss, bi, B, loss_kind,
                    ftz(identical_mul(Float32(0.5), alpha)), rowloss,
                )
            else:
                var a = Args()
                a.p0 = acts[L]
                a.p1 = yb
                a.p2 = dlogits
                a.p3 = rowloss
                a.i0 = loss_kind
                a.i1 = O
                ex.launch[OP_MLP_ROWLOSS](a, B)
                if L <= 4:
                    # every layer's ||W||^2 in one launch, a thread each (apple2)
                    var q = Args()
                    q.p0 = P
                    q.p1 = l2
                    q.i2 = L
                    for i in range(L):
                        var o_ = net.w_off(i)
                        var c_ = net.sizes[i] * net.sizes[i + 1]
                        if i == 0:
                            q.i4 = o_
                            q.i5 = c_
                        elif i == 1:
                            q.i6 = o_
                            q.i7 = c_
                        elif i == 2:
                            q.i8 = o_
                            q.i9 = c_
                        else:
                            q.i10 = o_
                            q.i11 = c_
                    ex.launch[OP_SUMSQ](q, L)
                else:
                    for i in range(L):
                        var q = Args()
                        q.p0 = P + net.w_off(i)
                        q.p1 = l2
                        q.i0 = i
                        q.i1 = net.sizes[i] * net.sizes[i + 1]
                        ex.launch[OP_SUMSQ](q, 1)
                var bl = Args()
                bl.p0 = rowloss
                bl.p1 = l2
                bl.p2 = dloss
                bl.i0 = bi
                bl.i1 = B
                bl.i2 = O
                bl.i3 = L
                bl.i4 = loss_kind
                bl.f0 = ftz(identical_mul(Float32(0.5), alpha))
                ex.launch[OP_MLP_BLOSS](bl, 1)
            # TOMBSTONE: MOJOLEARN_AUTOGRAD_MLP_OFF (DELETED-superseded) deleted 2026-10-19 by 6784245b2; code recoverable at 6784245b2^.
            # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_MLP_OFF.patch
            mlp_backward_autograd(tape, ex, net, tape_logits, dlogits, P, Gr, B, alpha)
            # TOMBSTONE: MOJOLEARN_AUTOGRAD_OPTIM (promoted: the executor optimizer is the trainers' one update; the hand arm deleted) deleted 2026-11-08 by 75825f479c; code recoverable at c3b7dc42e6^.
            # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_OPTIM.patch
            # Restored 2026-11-09 (lane r2-restore-unmeasured-defaults): the hand OP_OPT launch is the default.
            # TOMBSTONE: MOJOLEARN_AUTOGRAD_OPTIM (decided B arm, the executor optimizer step_lr; L40S ba08dac006 gru/lstm/mlp/rnn fit B 0.99-1.01x, same speed) deleted 2026-11-17 by lane/fb12-loser-arms; code recoverable at 2203db7094.
            # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_OPTIM.fb12-loser-arms.patch; record in docs/TOMBSTONES.md.
            var o = Args()
            o.p0 = P
            o.p1 = Gr
            o.p2 = s1
            o.p3 = s2
            o.p4 = s3
            if solver == SOLVER_ADAM:
                pw1 = ftz(identical_mul(pw1, b1))
                pw2 = ftz(identical_mul(pw2, b2))
                o.i0 = OPT_SK_ADAM
                o.f1 = b1
                o.f2 = b2
                o.f3 = eps
                o.f5 = ftz(identical_div(ftz(identical_mul(lr, ftz(identical_sqrt(Float32(1.0) - pw2)))),
                                         Float32(1.0) - pw1))
            else:
                o.i0 = OPT_SK_SGD
                o.i2 = 1 if nesterov else 0
                o.f0 = lr
                o.f1 = momentum
            ex.launch[OP_OPT](o, np_)
        var loss: Float64
        comptime if MLP_EPOCH_DEV:
            # the epoch loss as a blocked fold (2 launches, chains ~sqrt(n_batches));
            # MLP_EPOCH_DEV and MLP_LOSSBLOCK are both IDENTICAL
            mlp_epoch_loss_block(ex, dloss, lb_epochparts, dcurve, n_batches, bs, N, it)
            ex.sync()
            ex.download(FP(unsafe_from_address=Int(loss_word.unsafe_ptr())), dcurve + it, 1)
            curve.unsafe_store(it, loss_word[0])
            loss = Float64(loss_word[0])
        else:
            ex.sync()
            ex.download(FP(unsafe_from_address=Int(host_loss.unsafe_ptr())), dloss, n_batches)
            var acc = Float64(0.0)
            for bi in range(n_batches):
                var off = bi * bs
                var B = bs if off + bs <= N else N - off
                acc += Float64(host_loss[bi]) * Float64(B)
            loss = acc / Float64(N)
            curve.unsafe_store(it, Float32(loss))
        n_iter = it + 1
        t_samples += N
        if loss > best - tol:
            no_imp += 1
        else:
            no_imp = 0
        if loss < best:
            best = loss
        if solver == SOLVER_SGD and lr_sched == LR_INVSCALING:
            lr = Float32(Float64(lr_init) / identical_pow64(Float64(t_samples + 1), power_t))
        if no_imp > n_iter_no_change:
            if solver == SOLVER_SGD and lr_sched == LR_ADAPTIVE and lr > Float32(1e-6):
                lr = ftz(identical_div(lr, Float32(5.0)))
                no_imp = 0
            else:
                break
    # the last epoch ended on its ex.sync() and loss readback: the loop's work is done
    fit_loop_publish(loop_words, fit_loop_stop(loop_start, n_iter))
    # lane ag-fusion: return the tape's storage now, not at the end of ex
    tape.release(ex)
    ex.sync()
    ex.download(Pio, P, np_)
    _ = perm^
    _ = host_loss^
    _ = loss_word^
    return n_iter


# TOMBSTONE: MOJOLEARN_AUTOGRAD_PREDICT (folded in: the tape forward is the one predict path) deleted 2026-11-08 by c3b7dc42e6; code recoverable at c3b7dc42e6^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_PREDICT.patch
# The hand mlp_predict (the same chunking, copies and proba2 over the hand mlp_forward) is gone; mlp_predict below
# is the former mlp_predict_autograd under the public name.


def mlp_predict[E: Exec](mut ex: E, net: MLPNet, X: FP, N: Int, Pin: FP, dst: FP, chunk: Int,
                         proba2: Bool = False, resident: Bool = False) raises:
    """The hand predict (RESTORED 2026-11-17, see the note above
    AUTOGRAD_PREDICT): each chunk of `chunk` rows runs mlp_forward, one
    OP_GEMM_EPI launch a layer (+ OP_SOFTMAX for a softmax output), reading
    its rows of the uploaded input and writing its rows of the output in
    place (no per-chunk copy). `proba2`
    (cpu2-l11-neural): a one-output logistic net writes the (N, 2)
    probability `[1 - p, p]` (`OP_PROBA2` on the executor). `resident`
    (perf3-seq-neural): `Pin` is already this executor's memory (the model's
    resident fitted block, sequence/perf3_seq_resident.mojo), read in place:
    no per-call allocation and upload of the parameters."""
    var L = net.n_layers()
    var D = net.sizes[0]
    var O = net.sizes[L]
    var bs = chunk if chunk < N else N
    var dX = ex.alloc(N * D)
    ex.upload(dX, X, N * D)
    var P: FP
    if resident:
        P = Pin
    else:
        P = ex.alloc(net.n_params())
        ex.upload(P, Pin, net.n_params())
    var dout = ex.alloc(N * O)
    # Lane fx-percall-copies (2026-11-20): each chunk's first layer reads its rows of dX in place and its last
    # layer writes its rows of dout in place. Before, every chunk ran two device-to-device OP_COPY launches
    # (B x D floats in, B x O floats out) around the same forward, and held bs x (D + O) floats of staging.
    # Same products on the same values in the same order: no bits change. Only the hidden activations are
    # allocated; slots 0 and L of `acts` are re-pointed per chunk (two pointer stores, no allocation).
    var acts = List[FP]()
    acts.append(dX)
    for i in range(1, L):  # driver-loop: over hidden layers
        acts.append(ex.alloc(bs * net.sizes[i]))
    acts.append(dout)
    var b0 = 0
    while b0 < N:  # driver-loop: over prediction chunks
        var B = bs if b0 + bs <= N else N - b0
        acts[0] = dX + b0 * D
        acts[L] = dout + b0 * O
        mlp_forward(ex, net, P, acts, B)
        b0 += B
    if proba2 and O == 1:
        var two = ex.alloc(N * 2)
        var q = Args()
        q.p0 = dout
        q.p1 = two
        ex.launch[OP_PROBA2](q, N * 2)
        ex.sync()
        ex.download(dst, two, N * 2)
        # NAN_CANON_ON (default on; core/nan_canon.mojo):
        # the returned predictions' NaN words become the one canonical word,
        # on the device executor and on the host one alike (this function is
        # generic over both, so every column takes the same pass).
        comptime if NAN_CANON_ON:
            _ = nan_canon_f32(dst, N * 2)
        return
    ex.sync()
    ex.download(dst, dout, N * O)
    comptime if NAN_CANON_ON:
        _ = nan_canon_f32(dst, N * O)


# TOMBSTONE: MOJOLEARN_AUTOGRAD_PREDICT (DROPPED-slower: tape predict 1.04-1.06x on the four MLP infer cells, NV L40S 2026-11-18) deleted 2026-11-18 by lane/fb15-ab-verdicts; code recoverable at c989eb9222.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_PREDICT-2026-11-18.patch; record in docs/TOMBSTONES.md.
# mlp_predict_autograd (the tape forward in inference mode, chunk by chunk) is gone: see the verdict above
# AUTOGRAD_PREDICT's former place. The tape forward still trains every MLP (mlp_forward_autograd).
