# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RECURRENT NETWORKS: RNN (tanh, relu), LSTM and GRU layers under a linear
head, trained by backpropagation through time with one of the lane's
optimizers, written once against `sequence/exec.mojo::Exec`.

References: PyTorch `torch/nn/modules/rnn.py` (`nn.RNN`, `nn.LSTM`, `nn.GRU`:
the parameter layout `weight_ih_l{k}` [G*H, D_in], `weight_hh_l{k}` [G*H, H],
`bias_ih_l{k}`, `bias_hh_l{k}`, the gate orders i,f,g,o and r,z,n, and the GRU
update h' = n + z (h - n) of `aten/src/ATen/native/cuda/RNN.cu`).
`sequence/NOT_IMPLEMENTED.tsv` names what is not carried.

The arithmetic order is ours and fixed: the input projection of every step is
one GEMM over the whole sequence plus b_ih; each step adds h @ W_hh^T + b_hh
(GH) to it as GX + GH; every weight gradient is ONE GEMM whose reduction runs
over (time, batch) rows in ascending order; bias gradients are column sums in
the same row order.
"""
from sequence.exec_trait import Exec
from core.fit_loop_clock import fit_loop_start, fit_loop_stop
from sequence.fit_loop_words import fit_loop_publish
from sequence.ops import (
    FP,
    Args,
    OP_BIAS,
    OP_CE,
    OP_CELL_BWD,
    OP_CELL_FWD,
    OP_CELL_BWD_H,
    OP_COLSUM,
    OP_COPY,
    OP_SEQ_IOTA,
    OP_FILL,
    OP_GATHER_ROWS,
    OP_GATHER_SEQ,
    OP_GEMM,
    OP_GEMM_SPLITK,
    OP_MSE,
    OP_OPT,
    OPT_ADAGRAD,
    OPT_ADAM,
    OPT_ADAMW,
    OPT_ADAMAX,
    OPT_NADAM,
    dummy_ptr,
    gates_of,
)
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, identical_div, identical_mul, identical_pow64, identical_sqrt, ftz


# TOMBSTONE: MOJOLEARN_AUTOGRAD_OPTIM (promoted: the executor optimizer is the trainers' one update; the hand arm deleted) deleted 2026-11-08 by 75825f479c; code recoverable at c3b7dc42e6^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_OPTIM.patch
# Lane x3-mlp-rnn-predict-on-tape (plan EXECUTOR_CONSOLIDATION_AND_CLEANUP 5.6): the opt-in arm ran rnn_fit's
# update through autograd/optim.mojo (OptimizerState + step_lr), the same one OP_OPT launch with the same words
# (optimizer_config_packed maps OptConfig's words to named fields that pack back to the same words,
# optimizer_scalars is opt_scalars word for word). Andrew 2026-11-08 (bits may change; keep the simpler path):
# that shared module is now rnn_fit's only update, the one Sequential and ops.Session already use; rnn_fit's hand
# opt_step call is gone (opt_step stays for its other callers). Same bits by construction. A/B owed (ms per fit).
# Restored 2026-11-09 by lane r2-restore-unmeasured-defaults (AGENTS.md "A measured winner is never deleted on
# reasoning": the promotion had no measurement): rnn_fit's hand opt_step is the default again, as before X3 (the
# define's default at f7d10d2b09 was off).
# TOMBSTONE: MOJOLEARN_AUTOGRAD_OPTIM (the restored opt-in executor-optimizer arm, no A/B recorded) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_OPTIM.expdel2-30-neural.patch
# TOMBSTONE: MOJOLEARN_AUTOGRAD_OPTIM (decided B arm, the executor optimizer step_lr; L40S ba08dac006 gru/lstm/mlp/rnn fit B 0.99-1.01x, same speed) deleted 2026-11-17 by lane/fb12-loser-arms; code recoverable at 2203db7094.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_OPTIM.fb12-loser-arms.patch; record in docs/TOMBSTONES.md.

# TOMBSTONE: MOJOLEARN_AUTOGRAD_LOSS (rnn_fit: promoted, the blocked fold replaces the serial OP_SUM) deleted 2026-11-08 by dbab55bea7; code recoverable at c3b7dc42e6^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_LOSS.patch
# Lane x3-mlp-rnn-predict-on-tape (plan EXECUTOR_CONSOLIDATION_AND_CLEANUP 5.6; Andrew 2026-11-08: bits may
# change between versions, never keep a serial fold for old bits): each step's reported loss (losses[k]) comes
# from autograd/losses.mojo (loss_value_into: the squared-error terms or op_ce's row losses, term for term, then
# the tape's blocked fold with factor 1 / count) instead of sum_into's one-thread ascending OP_SUM (deleted).
# The gradient seed w.dy still comes from OP_MSE / OP_CE (gradient only: their unread loss stores are deleted). NEW BITS in losses[] only (every column together:
# host, Metal, NVIDIA, AMD run the same fold); parameters unchanged. A/B owed: arm A f7d10d2b09 (OP_SUM), arm B dbab55bea7.
from autograd.losses import LossScratch, loss_scratch, release_loss_scratch, loss_value_into, REDUCTION_MEAN
from autograd.loss_ops import LS_MSE, LS_CROSS_ENTROPY
from sequence.rnn_options_ops import OP_RNO_ROW_WEIGHTS
from autograd.epoch_monitor import EpochMonitor
from autograd.optim import OP_AGO_SUMSQ_PART, OP_AGO_NORM_ACC, OP_AGH_OPT_SCALED
from autograd.ops import ag_fold_block
from std.sys.compile import is_defined
# lane stream-fits (2026-11-25): the recurrent estimators' fit(source=) / partial_fit(source=)
from autograd.batch_source import (
    BatchSource, BS_FIELD_STEPS, BS_FIELD_LENGTHS, BS_FIELD_TARGETS, BATCH_SOURCE_STAGE_WORDS,
    batch_source_upload, batch_source_upload_ids, batch_source_fit_saved, source_part_store,
    RNN_STREAM_MONITOR_WORDS, RNN_STREAM_POSITION_WORDS,
)

#: nr-small D3 (2026-10-04): under IDENTICAL the recurrent weight and bias
#: gradients (C = A B over the K = T B time x batch rows, M N cells of one
#: K-long chain each, and the bias column sums as ones^T dG, one fma by 1.0
#: per term = the colsum add) are THE BLOCKED ORDER: K is cut into
#: consecutive blocks of wgrad_block(K) rows (a function of K alone), each
#: block folded from +0.0 by `gemm_dot`'s chain (`OP_GEMM_SPLITK` stage 0),
#: then each cell adds its block partials ascending from +0.0 (stage 1).
#: Device and host run the same ops, so the bits move on every column
#: together; the oracle (`sequence/checks/oracle.mojo::o_bptt_dw`) states
#: the same blocks. One block (K <= WGRAD_MIN_BLOCK) is the old chain.
# TOMBSTONE: MOJOLEARN_IDN_SEQ_WGRAD_BLOCKED_OFF (old path, no A/B recorded) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_SEQ_WGRAD_BLOCKED_OFF.expdel2-30-neural.patch
# TOMBSTONE: MOJOLEARN_IDN_ALL_OFF (umbrella old-path switch, EXP:2438) read folded here (non-owner) 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_ALL_OFF.expdel2-30-neural.patch
comptime SEQ_WGRAD_BLOCKED = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
# NN43 fp32 fixed-leaf-v2: K is partitioned at absolute multiples of 128;
# each leaf starts at +0, uses the incumbent pinned ascending FMA chain,
# then leaves are added ascending from +0. Odd tail has only actual rows.
# No launch/vendor/board shape chooses the graph. New bits across versions
# are permitted. Host Exec and GPU Exec run the same OP_GEMM_SPLITK body.
# Unmeasured, disabled; no quality/identity evidence is claimed.
# L11 (2026-10-07): NN43 and NI51's leaf arm were arms 2 and 1 of
# -D MOJOLEARN_IDN_SEQ_WGRAD; arm 0 (sqrt(K) blocks from 512 rows) is the one form.
# TOMBSTONE: MOJOLEARN_IDN_SEQ_WGRAD (opt-in arms 1 and 2, no A/B recorded) deleted 2026-11-15 by lane/expdel2-30-neural; code recoverable at e39daf79ad.
# Restore: git apply experiments/removed/MOJOLEARN_IDN_SEQ_WGRAD.expdel2-30-neural.patch
# (Sweep 2026-11-15: NN43_WGRAD_FIXED128 = False removed with its reader's dead arm in autograd/kernels/recurrent.mojo.)

# NI51 V subarm: a smaller minimum leaf trades more independent gradient
# work for partial storage. Above that minimum the existing sqrt(K) growth
# bounds partial count; no dataset or benchmark dimension selects this graph.
# The generic executor makes host and all GPU columns share the new bits.
# Default OFF; multi-step recurrent quality and identity are unmeasured.
comptime WGRAD_MIN_BLOCK = 512
#: partial floats one launch pair may hold (rows of cells are chunked to
#: fit; chunking moves no bit, every cell's blocks are the same)
comptime WGRAD_IDN_SCRATCH = 1 << 22


comptime TASK_MSE = 0
comptime TASK_CE = 1


struct Net(ImplicitlyCopyable, Movable):
    """The recurrent network's shape and its flat parameter layout, PyTorch's
    order: per layer, per direction (lane rnn-parity: `dirs` 2 is
    `bidirectional=True`, the reverse direction's block after the forward
    one, torch's `_reverse` names), weight_ih, weight_hh, bias_ih, bias_hh;
    then the head [O, dirs H] and its bias. Layer l > 0 reads the previous
    layer's dirs H features (the directions concatenated, forward first)."""

    var cell: Int
    var D: Int
    var H: Int
    var L: Int
    var O: Int
    var dirs: Int

    def __init__(out self, cell: Int, D: Int, H: Int, L: Int, O: Int):
        self.cell = cell
        self.D = D
        self.H = H
        self.L = L
        self.O = O
        self.dirs = 1

    def __init__(out self, cell: Int, D: Int, H: Int, L: Int, O: Int, dirs: Int):
        self.cell = cell
        self.D = D
        self.H = H
        self.L = L
        self.O = O
        self.dirs = dirs

    def G(self) -> Int:
        return gates_of(self.cell)

    def din(self, l: Int) -> Int:
        return self.D if l == 0 else self.dirs * self.H

    def layer_size(self, l: Int) -> Int:
        """One direction's block of layer l."""
        var gh = self.G() * self.H
        return gh * self.din(l) + gh * self.H + 2 * gh

    def w_ih(self, l: Int, d: Int = 0) -> Int:
        var off = 0
        for k in range(l):
            off += self.dirs * self.layer_size(k)
        if d != 0:
            off += self.layer_size(l)
        return off

    def w_hh(self, l: Int, d: Int = 0) -> Int:
        return self.w_ih(l, d) + self.G() * self.H * self.din(l)

    def b_ih(self, l: Int, d: Int = 0) -> Int:
        return self.w_hh(l, d) + self.G() * self.H * self.H

    def b_hh(self, l: Int, d: Int = 0) -> Int:
        return self.b_ih(l, d) + self.G() * self.H

    def slots(self) -> Int:
        """Layer-direction blocks (the Work's buffer slots), l dirs + d."""
        return self.L * self.dirs

    def head_in(self) -> Int:
        """The head's input width: the top layer's directions concatenated."""
        return self.dirs * self.H

    def head(self) -> Int:
        return self.w_ih(self.L)

    def n_params(self) -> Int:
        return self.head() + self.O * self.head_in() + self.O


#: Lane rnn-parity (Andrew, 2026-11-25): the options of the estimators' recurrent fit and predict beyond the
#: last-step, unweighted, zero-state model. `RnnOptions()` is that model: rnn_fit and rnn_predict then run their
#: launches of before, one for one (`plain`). Host words only; resolved once by the caller (sequence/ops_impl.mojo).
struct RnnOptions(ImplicitlyCopyable, Movable):
    #: output="sequence": the head reads every step's state and the loss covers every valid step (time-major rows
    #: t B + b); else the head reads the final state (PyTorch's h_n: the forward direction at each row's last valid
    #: step, the reverse direction at time 0)
    var seq_out: Bool
    #: bias=False: the recurrent layers' biases stay 0 (their gradient is zeroed before every update); the head
    #: keeps its bias (nn.Linear)
    var no_bias: Bool
    #: dropout between stacked layers (nn.GRU's `dropout`), fit only; 0 is none
    var dropout: Float32
    #: clip_grad_norm: the total 2-norm of all gradients is scaled to at most this before every update
    #: (torch.nn.utils.clip_grad_norm_); 0 is none
    var clip: Float32
    #: sample weights: `sw` holds N (sw_steps False) or N T (True, output="sequence" only) host float32 words
    var has_sw: Bool
    var sw_steps: Bool
    var sw: FP
    #: class weights: `cw` holds O host float32 words, one per class code (classifiers only)
    var has_cw: Bool
    var cw: FP
    #: initial state: `h0` holds L dirs N H host float32 words (PyTorch's h_0 with the batch axis over all N
    #: sequences), `c0` the LSTM's c_0 likewise (absent: zeros)
    var has_h0: Bool
    var h0: FP
    var has_c0: Bool
    var c0: FP
    #: per-epoch validation (the generic EpochMonitor, autograd/epoch_monitor.mojo): `nv` sequences (0: none) in
    #: `vx` [nv, T, D], their targets `vy` (as Y), their lengths `vl` when `has_vl`; `patience` n_iter_no_change
    #: (0: every epoch), `tol`, `restore` (the best epoch's parameters and optimizer state at the end). `vcurve`
    #: receives epochs + 2 host words: each epoch's validation loss, then the epochs run and the best epoch.
    var nv: Int
    var vx: FP
    var vy: FP
    var has_vl: Bool
    var vl: FP
    var patience: Int
    var tol: Float32
    var restore: Bool
    var vcurve: FP
    #: resume (warm start of the optimizer, partial_fit): `state` holds 3 n_params + 4 host words in and out (the
    #: optimizer's three state buffers, beta1^t, beta2^t, NAdam's mu product, and the steps taken before, `t0`)
    var has_state: Bool
    var state: FP
    #: the fit's epochs (the vcurve's length less its two result words)
    var epochs: Int

    def vcurve_words(self) -> Int:
        return self.epochs if self.nv > 0 else 0

    def __init__(out self):
        self.seq_out = False
        self.no_bias = False
        self.dropout = Float32(0.0)
        self.clip = Float32(0.0)
        self.has_sw = False
        self.sw_steps = False
        self.sw = dummy_ptr()
        self.has_cw = False
        self.cw = dummy_ptr()
        self.has_h0 = False
        self.h0 = dummy_ptr()
        self.has_c0 = False
        self.c0 = dummy_ptr()
        self.nv = 0
        self.vx = dummy_ptr()
        self.vy = dummy_ptr()
        self.has_vl = False
        self.vl = dummy_ptr()
        self.patience = 0
        self.tol = Float32(0.0)
        self.restore = False
        self.vcurve = dummy_ptr()
        self.has_state = False
        self.state = dummy_ptr()
        self.epochs = 0

    def plain(self, net: Net) -> Bool:
        """No option on: the launches of the last-step model of before."""
        return (net.dirs == 1 and not self.seq_out and not self.no_bias and self.dropout == Float32(0.0)
                and self.clip == Float32(0.0) and not self.has_sw and not self.has_cw and not self.has_h0
                and self.nv == 0 and not self.has_state)

    def weighted(self, has_lens: Bool) -> Bool:
        """The loss is a weighted mean (sum w l / sum w): sample or class weights, or every step's loss masked by
        the lengths."""
        return self.has_sw or self.has_cw or (self.seq_out and has_lens)


# ------------------------------------------------------------------ launch helpers
def gemm[E: Exec](
    mut ex: E, A: FP, B: FP, C: FP, M: Int, N: Int, K: Int,
    sam: Int, sak: Int, sbk: Int, sbn: Int, accumulate: Bool, ldc: Int,
) raises:
    var a = Args()
    a.p0 = A
    a.p1 = B
    a.p2 = C
    a.i0 = M
    a.i1 = N
    a.i2 = K
    a.i3 = sam
    a.i4 = sak
    a.i5 = sbk
    a.i6 = sbn
    a.i7 = 1 if accumulate else 0
    a.i8 = ldc
    # FAST (apple2): a product of few cells over a long K (VAR's Z^T Z) ran
    # on M N threads, each one K-long chain; split K so about 8192 threads
    # fold blocks of >= 4096, then one ordered sum per cell. IDENTICAL keeps
    # the one chain (its bits); so does every product outside that shape.
    comptime if GLOBAL_NUMERIC_MODE != NUMERIC_IDENTICAL:
        var MN = M * N
        if MN <= 1024 and K >= 32768:
            var S = min(8192 // MN, K // 4096)
            if S > 1:
                var KS = (K + S - 1) // S
                S = (K + KS - 1) // KS
                a.p3 = ex.alloc(S * MN)
                a.i9 = S
                a.i10 = KS
                a.i11 = 0
                ex.launch[OP_GEMM_SPLITK](a, S * MN)
                a.i11 = 1
                ex.launch[OP_GEMM_SPLITK](a, MN)
                return
    ex.launch[OP_GEMM](a, M * N)


# TOMBSTONE: MOJOLEARN_SEQ_FAST_LSTM_WGRAD (DROPPED-quality with the SCAN bundle, 2026-10-04: lstm-clf accuracy 0.9608 ->
# 0.5002, lstm-reg r2 0.9804 -> -0.1043; never A/B-ed alone) deleted 2026-10-09 on lane/owed-deletions-D2: the FAST split-K
# weight-gradient folds (wgrad_gemm, WGRAD_SCRATCH); code recoverable at b639a2bd2.
# Restore: git apply experiments/removed/MOJOLEARN_SEQ_FAST_LSTM_WGRAD.patch


def bias_rows[E: Exec](mut ex: E, X: FP, b: FP, Y: FP, R: Int, C: Int) raises:
    var a = Args()
    a.p0 = X
    a.p1 = b
    a.p2 = Y
    a.i0 = R
    a.i1 = C
    a.i2 = C
    a.i3 = C
    ex.launch[OP_BIAS](a, R * C)


def fill[E: Exec](mut ex: E, p: FP, n: Int, v: Float32) raises:
    var a = Args()
    a.p0 = p
    a.f0 = v
    ex.launch[OP_FILL](a, n)


def gather_seq[E: Exec](
    mut ex: E, X: FP, idx: FP, dst: FP, T: Int, B: Int, D: Int, off: Int, lengths: Optional[FP] = None,
) raises:
    """With `lengths` (every row's length, the rows' own order) the steps at
    or past a row's length are stored as +0.0 (op_gather_seq)."""
    var a = Args()
    a.p0 = X
    a.p1 = idx
    a.p2 = dst
    a.i0 = T
    a.i1 = B
    a.i2 = D
    a.i3 = off
    if lengths:
        a.p3 = lengths.value()
        a.i4 = 1
    ex.launch[OP_GATHER_SEQ](a, T * B * D)


def gather_rows[E: Exec](mut ex: E, Y: FP, idx: FP, dst: FP, B: Int, O: Int, off: Int) raises:
    var a = Args()
    a.p0 = Y
    a.p1 = idx
    a.p2 = dst
    a.i0 = B
    a.i1 = O
    a.i2 = off
    ex.launch[OP_GATHER_ROWS](a, B * O)


# TOMBSTONE: MOJOLEARN_AUTOGRAD_LOSS (rnn_fit: promoted, the blocked fold replaces the serial OP_SUM) deleted 2026-11-08 by dbab55bea7; code recoverable at c3b7dc42e6^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_LOSS.patch
# sum_into (dst[at] = scale * the n values of X added ascending by ONE thread, OP_SUM) was rnn_fit's reported
# loss; loss_value_into's blocked fold replaces it.


# ------------------------------------------------------------------ work buffers
struct Work(Movable):
    var gx: List[FP]
    var gh: List[FP]
    var act: List[FP]
    var hall: List[FP]
    var call: List[FP]
    var dgh: FP
    var dh: FP
    var dhn: FP
    var dc: FP
    var xb: FP
    var yb: FP
    var yhat: FP
    var dy: FP
    var one: FP
    var wsplit: FP
    #: sequence lengths (lane seq-blocks-2): `lens` holds the current batch's
    #: row lengths (float32, B words) when `has_lens`; the layers then run the
    #: per-step chain with the length mask (sequence/ops.mojo op_cell_fwd /
    #: op_cell_bwd). Without lengths `lens` is a placeholder nothing reads.
    var lens: FP
    var has_lens: Bool
    #: lane rnn-parity (RnnOptions): `wrow` the batch's loss weights (R words, R = B or T B with output="sequence")
    #: and `tiny` the weighted mean's denominator guard (2^-126), when the loss is weighted; `h0b` / `c0b` the
    #: batch's initial states (one B H block per layer-direction slot), when given. Placeholders (`one`) else.
    var wrow: FP
    var tiny: FP
    var h0b: FP
    var c0b: FP
    var has_h0: Bool
    var has_c0: Bool

    def __init__(out self, gx: FP, gh: FP, act: FP, hall: FP, call: FP, dgh: FP, dh: FP, dhn: FP, dc: FP,
                 one: FP, wsplit: FP):
        """ONE layer's buffers on storage the caller owns (lane
        fx-ops-session-blocks, 2026-11-19: the session's recurrent scan op,
        autograd/rules_recurrent.mojo `rnn_scan_op`, hands tape storage of
        the sizes the constructor below allocates for T, B and one layer).
        `one` holds 1.0. The batch, target and head buffers are placeholders
        nothing on that path reads."""
        self.gx = List[FP]()
        self.gh = List[FP]()
        self.act = List[FP]()
        self.hall = List[FP]()
        self.call = List[FP]()
        self.gx.append(gx)
        self.gh.append(gh)
        self.act.append(act)
        self.hall.append(hall)
        self.call.append(call)
        self.dgh = dgh
        self.dh = dh
        self.dhn = dhn
        self.dc = dc
        self.xb = one
        self.yb = one
        self.yhat = one
        self.dy = one
        self.one = one
        self.wsplit = wsplit
        self.lens = one
        self.has_lens = False
        self.wrow = one
        self.tiny = one
        self.h0b = one
        self.c0b = one
        self.has_h0 = False
        self.has_c0 = False

    def __init__[E: Exec](
        out self, mut ex: E, net: Net, T: Int, B: Int, train: Bool, has_lens: Bool = False,
    ) raises:
        self = Self(ex, net, T, B, train, has_lens, RnnOptions())

    def __init__[E: Exec](
        out self, mut ex: E, net: Net, T: Int, B: Int, train: Bool, has_lens: Bool, opts: RnnOptions,
    ) raises:
        """The trainer's and the predictor's buffers for batches of up to B rows. With RnnOptions() (the plain
        model) the allocations of before, one for one; lane rnn-parity: one buffer slot per layer and direction,
        the head's rows R = T B with output="sequence", and the weight / initial-state buffers when used."""
        var GH = net.G() * net.H
        var H = net.H
        var S = T if opts.seq_out else 1
        self.gx = List[FP]()
        self.gh = List[FP]()
        self.act = List[FP]()
        self.hall = List[FP]()
        self.call = List[FP]()
        for _ in range(net.slots()):
            self.gx.append(ex.alloc(T * B * GH))
            self.gh.append(ex.alloc(T * B * GH))
            self.act.append(ex.alloc(T * B * GH))
            self.hall.append(ex.alloc((T + 1) * B * H))
            self.call.append(ex.alloc((T + 1) * B * H))
        var tb = T * B if train else 1
        # TOMBSTONE: MOJOLEARN_AUTOGRAD_RNN_OFF (hand glue, superseded by the tape) deleted 2026-10-19 by e88ae4e38; code recoverable at e88ae4e38^.
        # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_RNN_OFF.patch
        # w.dgx, w.dhout0 and w.dhout1 were the hand backward's dGX and dH_out buffers; the tape owns those gradients.
        self.dgh = ex.alloc(tb * GH)
        self.dh = ex.alloc(B * H if train else 1)
        self.dhn = ex.alloc(B * H if train else 1)
        self.dc = ex.alloc(B * H if train else 1)
        self.xb = ex.alloc(T * B * net.D)
        self.yb = ex.alloc(S * B * net.O)
        self.yhat = ex.alloc(S * B * net.O)
        self.dy = ex.alloc(S * B * net.O)
        # TOMBSTONE: w.sq (the squared-error / row-loss words OP_MSE and OP_CE stored and nothing read; the reported loss is
        # loss_value_into's) deleted 2026-11-09 by lane/m1-followups-misc; code recoverable at 5b5490960d.
        # Restore: git apply experiments/removed/SEQ_LOSS_SEED_UNREAD_STORES.patch
        comptime if SEQ_WGRAD_BLOCKED:
            # the bias sums as ones^T dG (one fma by 1.0 per term, exact)
            self.one = ex.alloc(1)
            fill(ex, self.one, 1, Float32(1.0))
            self.wsplit = ex.alloc(WGRAD_IDN_SCRATCH if train else 1)
        else:
            # unread placeholders (the unblocked weight gradient takes no ones or split scratch)
            self.one = ex.alloc(1)
            self.wsplit = self.one
        self.has_lens = has_lens
        self.lens = ex.alloc(B) if has_lens else self.one
        self.wrow = self.one
        self.tiny = self.one
        if train and (opts.weighted(has_lens) or opts.nv > 0):
            self.wrow = ex.alloc(S * B)
            self.tiny = ex.alloc(1)
            # the smallest normal float32: sum w + 2^-126 is sum w for any sum above 2^-100, and 0 / 2^-126 is 0
            fill(ex, self.tiny, 1, Float32(1.1754943508222875e-38))
        self.has_h0 = opts.has_h0
        self.has_c0 = opts.has_c0
        self.h0b = ex.alloc(net.slots() * B * H) if opts.has_h0 else self.one
        self.c0b = ex.alloc(net.slots() * B * H) if opts.has_c0 else self.one


# TOMBSTONE: MOJOLEARN_AUTOGRAD_PREDICT (folded in: the tape forward is the one predict path) deleted 2026-11-08 by 9a9a8e3148; code recoverable at c3b7dc42e6^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_PREDICT.patch
# The hand forward (every layer over the time-major input: input projection, scan or per-step OP_CELL_FWD_H
# chain) and head (h_T @ W_o^T + b_o) are gone with their one caller, the hand rnn_predict: prediction records
# the trainer's forward (autograd/rules_recurrent.mojo rnn_forward_autograd) in inference mode, the same launches.


# TOMBSTONE: MOJOLEARN_AUTOGRAD_RNN_OFF (hand glue, superseded by the tape) deleted 2026-10-19 by e88ae4e38; code recoverable at e88ae4e38^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_RNN_OFF.patch
# The hand-chained backward (`backward`: head dW/db, the zeroed h_T delta, per layer the cell or scan
# backward, dW/db of both projections and the input-projection delta) is gone: rnn_fit's backward is the tape's
# (autograd/rules_recurrent.mojo AutogradRnnStep, RecurrentRules), which runs the same launches with the same
# arguments. The cell, scan and weight-gradient kernels it called stay. This function also carried the
# MOJOLEARN_SEQ_FAST_LSTM_WGRAD tombstone (deleted 2026-10-09, lane/owed-deletions-D2, recoverable at b639a2bd2).


# ------------------------------------------------------------------ optimizer
@fieldwise_init
struct OptConfig(ImplicitlyCopyable, Movable):
    """kind (sequence/ops.mojo OPT_*), flags, beta1 / momentum / lr_decay,
    beta2 / alpha, eps, weight_decay, f7 (dampening / RMSprop momentum)."""

    var kind: Int
    var flags: Int
    var f1: Float32
    var f2: Float32
    var eps: Float32
    var wd: Float32
    var f7: Float32


struct OptState(Movable):
    """The host scalars' running state: beta1^t and beta2^t (one product per
    step, identical_mul, so a step's bias correction is a function of t) and
    NAdam's mu product."""

    var pw1: Float32
    var pw2: Float32
    var mu_prod: Float32

    def __init__(out self):
        self.pw1 = Float32(1.0)
        self.pw2 = Float32(1.0)
        self.mu_prod = Float32(1.0)


def opt_scalars(cfg: OptConfig, mut st: OptState, t: Int, lr: Float32) -> Tuple[Float32, Float32, Float32]:
    """(f5, f6, f7) of step t (one-based), advancing the running state. The
    caller runs every step in order, so the state is a function of t."""
    var f5 = Float32(0.0)
    var f6 = Float32(0.0)
    var f7 = cfg.f7
    if cfg.kind == OPT_ADAM or cfg.kind == OPT_ADAMW or cfg.kind == OPT_ADAMAX or cfg.kind == OPT_NADAM:
        st.pw1 = ftz(identical_mul(st.pw1, cfg.f1))
        st.pw2 = ftz(identical_mul(st.pw2, cfg.f2))
    if cfg.kind == OPT_ADAM or cfg.kind == OPT_ADAMW:
        f5 = ftz(identical_div(lr, Float32(1.0) - st.pw1))
        f6 = ftz(identical_sqrt(Float32(1.0) - st.pw2))
    elif cfg.kind == OPT_ADAMAX:
        f5 = ftz(identical_div(lr, Float32(1.0) - st.pw1))
    elif cfg.kind == OPT_NADAM:
        # torch _single_tensor_nadam: mu_t = b1 (1 - 0.5 0.96^(t md)),
        # mu_{t+1} likewise, mu_product *= mu_t (float32 state);
        # c1 = -lr (1 - mu_t) / (1 - mu_product),
        # c2 = -lr mu_{t+1} / (1 - mu_product mu_{t+1}) (host float64)
        var md = Float64(cfg.f7)
        var b1 = Float64(cfg.f1)
        var mu = b1 * (1.0 - 0.5 * identical_pow64(0.96, Float64(t) * md))
        var mu_next = b1 * (1.0 - 0.5 * identical_pow64(0.96, Float64(t + 1) * md))
        st.mu_prod = ftz(identical_mul(st.mu_prod, Float32(mu)))
        var mp = Float64(st.mu_prod)
        var lr64 = Float64(lr)
        f5 = Float32(1.0) - st.pw2
        f6 = Float32(-lr64 * (1.0 - mu) / (1.0 - mp))
        f7 = Float32(-lr64 * mu_next / (1.0 - mp * mu_next))
    elif cfg.kind == OPT_ADAGRAD:
        f5 = ftz(identical_div(lr, Float32(1.0) + ftz(identical_mul(Float32(t - 1), cfg.f1))))
    return (f5, f6, f7)


def opt_step[E: Exec](
    mut ex: E, cfg: OptConfig, mut st: OptState, t: Int, lr: Float32,
    P: FP, Gr: FP, s1: FP, s2: FP, s3: FP, n: Int,
) raises:
    var a = Args()
    a.p0 = P
    a.p1 = Gr
    a.p2 = s1
    a.p3 = s2
    a.p4 = s3
    a.i0 = cfg.kind
    a.i1 = t
    a.i2 = cfg.flags
    a.f0 = lr
    a.f1 = cfg.f1
    a.f2 = cfg.f2
    a.f3 = cfg.eps
    a.f4 = cfg.wd
    var sc = opt_scalars(cfg, st, t, lr)
    a.f5 = sc[0]
    a.f6 = sc[1]
    a.f7 = sc[2]
    ex.launch[OP_OPT](a, n)


def opt_step_clipped[E: Exec](
    mut ex: E, cfg: OptConfig, mut st: OptState, t: Int, lr: Float32,
    P: FP, Gr: FP, s1: FP, s2: FP, s3: FP, n: Int, cell: FP,
) raises:
    """Lane rnn-parity: opt_step with the gradient scaled by the clip coefficient in cell[1] first, as ONE
    launch of the generic training loop's clipped update (autograd/optim.mojo op_agh_opt_scaled: the scale,
    then op_opt on the stored word). op_opt's arguments are opt_step's, word for word."""
    var a = Args()
    a.p0 = P
    a.p1 = Gr
    a.p2 = s1
    a.p3 = s2
    a.p4 = s3
    a.i0 = cfg.kind
    a.i1 = t
    a.i2 = cfg.flags
    a.f0 = lr
    a.f1 = cfg.f1
    a.f2 = cfg.f2
    a.f3 = cfg.eps
    a.f4 = cfg.wd
    var sc = opt_scalars(cfg, st, t, lr)
    a.f5 = sc[0]
    a.f6 = sc[1]
    a.f7 = sc[2]
    a.p6 = cell
    a.i3 = 1
    ex.launch[OP_AGH_OPT_SCALED](a, n)


# TOMBSTONE: MOJOLEARN_AUTOGRAD_DATA (promoted: autograd/data.mojo epoch_order is the one epoch order; the hand launches deleted) deleted 2026-11-08 by 75825f479c; code recoverable at c3b7dc42e6^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_DATA.patch
# The opt-in arm built rnn_fit's epoch order through autograd/data.mojo (epoch_order), the same OP_MLP_PERM /
# OP_SEQ_IOTA launches with the same key (same bits by construction). It is now the only path (the data layer
# distributed_program and partition_runtime already use); rnn_fit's inline launches are gone. A/B owed.
from autograd.data import epoch_order as ag_epoch_order


# ------------------------------------------------------------------ training step seam
trait RnnStep(Defaultable & Movable & Deinitable):
    """One optimizer step's differentiation: forward_head leaves the head's
    output in w.yhat; backward reads dy [B, O] from w.dy and writes every
    parameter gradient into Gr (the `Net` layout). rnn_fit keeps the loss,
    shuffle, batching and optimizer. AutogradRnnStep
    (autograd/rules_recurrent.mojo) is the one step; the trait stays because
    that module imports this one.
    forward_head also receives Gr so a step can name it as the gradient
    destination when it records the forward (the tape step does)."""

    def forward_head[E: Exec](
        mut self, mut ex: E, net: Net, P: FP, Gr: FP, x: FP, T: Int, B: Int, w: Work,
    ) raises:
        ...

    def backward[E: Exec](mut self, mut ex: E, net: Net, P: FP, Gr: FP, x: FP, T: Int, B: Int, w: Work) raises:
        ...

    def forward_options[E: Exec](
        mut self, mut ex: E, net: Net, opts: RnnOptions, P: FP, Gr: FP, x: FP, T: Int, B: Int, w: Work,
        seed: UInt64, step: Int,
    ) raises:
        """Lane rnn-parity: forward_head with RnnOptions (directions, dropout between layers with the mask of
        (seed, step), the per-step head, the initial states of w.h0b / w.c0b). The head's output is w.yhat:
        [B, O], or [T B, O] time-major with output="sequence"; its gradient storage is w.dy."""
        ...

    def weighted_backward[E: Exec](mut self, mut ex: E, net: Net, task: Int, R: Int, w: Work, loss_dst: FP) raises:
        """Lane rnn-parity: the weighted mean loss sum_r w_r l_r / (sum_r w_r + 2^-126) of the R head rows of
        the last forward_options (l_r the row's mean squared error over the O outputs, or its softmax
        cross-entropy against the code in w.yb; w_r = w.wrow[r]) recorded on the step's tape, its value stored
        at loss_dst[0], and every parameter gradient into Gr through it."""
        ...

    def validation_loss[E: Exec](
        mut self, mut ex: E, net: Net, task: Int, opts: RnnOptions, P: FP, T: Int, w: Work, setup: RnnFitSetup,
        dst: FP,
    ) raises:
        """Lane rnn-parity: the loss of the parameters P on the opts.nv validation sequences into dst[0] (an
        executor word): the forward in inference mode in chunks of the Work's rows, each row (each valid step
        with output="sequence") weighted 1, the plain mean over all of them (sum l / count, the chunks' sums
        folded in order). Sample and class weights do not enter it."""
        ...


# Lane rnn-parity fault injection (verify's sabotage arms of the weighted fit, never in a product build): the
# loss weights drop their sample-weight factor on ONE side, the device executor's fit or the host executor's.
# The verify lane sequence-rnn-weighted-steps (tools/identity_lanes/rnn_parity.py) must FAIL under either.
comptime RNN_PARITY_SABOTAGE_WEIGHT_DEVICE = is_defined["MOJOLEARN_RNN_PARITY_SABOTAGE_WEIGHT_DEVICE"]()
comptime RNN_PARITY_SABOTAGE_WEIGHT_HOST = is_defined["MOJOLEARN_RNN_PARITY_SABOTAGE_WEIGHT_HOST"]()


# TOMBSTONE: MOJOLEARN_AUTOGRAD_RNN_OFF (hand glue, superseded by the tape) deleted 2026-10-19 by e88ae4e38; code recoverable at e88ae4e38^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_RNN_OFF.patch
# HandStep (the hand-assembled step over forward, head and backward) is gone: AutogradRnnStep is the only step.


# ------------------------------------------------------------------ fit / predict
def rnn_fit[E: Exec, //, S: RnnStep](
    mut ex: E, net: Net, task: Int,
    X: FP, Y: FP, N: Int, T: Int,
    epochs: Int, bs: Int, shuffle: Bool, seed: UInt64,
    Pio: FP, losses: FP, lrs: FP, cfg: OptConfig, init_acc: Float32,
    lengths: Optional[FP], loop_words: Int, x_device: Bool, opts: RnnOptions,
) raises:
    """`epochs * ceil(N / bs)` optimizer steps; step k of epoch e trains on
    the samples order_e[j bs : j bs + B] (j = k mod per, B = min(bs, N - j
    bs)), order_e the epoch's row order. Lane cpu4-python: the order is built
    on the executor each epoch (`op_mlp_perm`, the MLP's Feistel permutation
    keyed by `mlp_epoch_key(seed, e)`, integers only; or `op_seq_iota`
    without shuffle) instead of a host Fisher-Yates schedule uploaded whole:
    a new shuffle order on every column together (host column: the same
    ops on the host executor). Pio holds the starting parameters and
    receives the trained ones; losses[k] is the mean loss of step k before
    its update.
    `lengths` (lane seq-blocks-2): N row lengths (float32 words, each in
    [1, T], admitted by the caller), carried through the batching: every
    step gathers its rows' lengths with their sequences; a row's padded
    steps enter neither its state, the loss nor the gradient, and the head
    reads its state at its own last valid step. Without it nothing changes.
    `loop_words` (lane alm-neural, 2026-11-24): the address of the
    `FitLoop` words (sequence/fit_loop_words.mojo), or 0. The loop region
    is the step loop below: the clock is read once before step 0's first
    launch (after the uploads and allocations) and once after the existing
    `ex.sync()` that follows the last step (no synchronization added,
    nothing inside the loop); its iterations are the epochs run.
    `opts` (lane rnn-parity, Andrew 2026-11-25): RnnOptions. RnnOptions()
    (`plain`) runs the loop of before, launch for launch. Otherwise the
    options loop: Y holds N S rows of targets (S = T with output="sequence":
    row i T + s the target of step s of sequence i, batch-first; else 1),
    the head reads every step (time-major [T B, O]) or the final state, the
    loss is the plain mean or, weighted (RnnOptions.weighted), sum w l /
    (sum w + 2^-126) over the batch's rows with w the sample weight times the
    class weight, 0 at a padded step; bias=False zeroes the recurrent biases
    and their gradients; clip_grad_norm scales the gradients as
    torch.nn.utils.clip_grad_norm_ before the update."""
    var np_ = net.n_params()
    var ycols = net.O if task == TASK_MSE else 1
    var St = T if opts.seq_out else 1
    var plain = opts.plain(net)
    var per = (N + bs - 1) // bs
    var n_steps = epochs * per
    var bmax = min(bs, N)
    # Lane vdf-scope-neural: `x_device` (see autograd/rules_recurrent.mojo
    # `rnn_predict`): X read in place from the member store, no upload.
    var dX: FP
    if x_device:
        dX = X
    else:
        dX = ex.alloc(N * T * net.D)
        ex.upload(dX, X, N * T * net.D)
    var dY = ex.alloc(N * St * ycols)
    ex.upload(dY, Y, N * St * ycols)
    var has_lens = Bool(lengths)
    var dL = ex.alloc(N if has_lens else 1)
    var dL_opt = Optional[FP](None)
    if has_lens:
        ex.upload(dL, lengths.value(), N)
        dL_opt = Optional[FP](dL)
    var dord = ex.alloc(N)
    var P = ex.alloc(np_)
    ex.upload(P, Pio, np_)
    var Gr = ex.alloc(np_)
    # TOMBSTONE: MOJOLEARN_AUTOGRAD_OPTIM (promoted: the executor optimizer is the trainers' one update; the hand arm deleted) deleted 2026-11-08 by 75825f479c; code recoverable at c3b7dc42e6^.
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_OPTIM.patch
    # Restored 2026-11-09 (lane r2-restore-unmeasured-defaults): opt_step's state s1, s2, s3 is the default (s2
    # filled with init_acc, Adagrad's initial accumulator).
    # TOMBSTONE: MOJOLEARN_AUTOGRAD_OPTIM (decided B arm, the executor optimizer step_lr; L40S ba08dac006 gru/lstm/mlp/rnn fit B 0.99-1.01x, same speed) deleted 2026-11-17 by lane/fb12-loser-arms; code recoverable at 2203db7094.
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_OPTIM.fb12-loser-arms.patch; record in docs/TOMBSTONES.md.
    var s1 = ex.alloc(np_)
    var s2 = ex.alloc(np_)
    var s3 = ex.alloc(np_)
    if init_acc != Float32(0.0):
        fill(ex, s2, np_, init_acc)
    var dloss = ex.alloc(n_steps)
    var w = Work(ex, net, T, bmax, True, has_lens, opts)
    # TOMBSTONE: MOJOLEARN_AUTOGRAD_LOSS (rnn_fit: promoted, the blocked fold replaces the serial OP_SUM) deleted 2026-11-08 by dbab55bea7; code recoverable at c3b7dc42e6^.
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_LOSS.patch
    var loss_ws = loss_scratch(ex, St * bmax * net.O)
    var st = OptState()
    var step = S()
    # lane rnn-parity: the options' setup, once (nothing when plain)
    var setup = RnnFitSetup(ex, net, task, opts, N, St, np_, P, dY, has_lens, w, T, ycols, bmax)
    # lane alm-left: the queued setup (uploads, fills) drained ONCE before the
    # clock, outside the loop, so the loop region holds loop work only
    ex.sync()
    var loop_start = fit_loop_start()
    var epochs_run = epochs
    if not plain:
        # resume (partial_fit, lane rnn-parity): the optimizer state and step count of the previous call
        var t0 = 0
        if opts.has_state:
            ex.upload(s1, opts.state, np_)
            ex.upload(s2, opts.state + np_, np_)
            ex.upload(s3, opts.state + 2 * np_, np_)
            st.pw1 = opts.state.unsafe_load(3 * np_)
            st.pw2 = opts.state.unsafe_load(3 * np_ + 1)
            st.mu_prod = opts.state.unsafe_load(3 * np_ + 2)
            t0 = Int(opts.state.unsafe_load(3 * np_ + 3))
        var monitor = EpochMonitor(Float64(opts.tol), opts.patience)
        var best = RnnBest(ex, opts, np_)
        var vword = List[Float32](length=1, fill=0.0)
        for k in range(n_steps):  # driver-loop: over optimizer steps (the options loop, lane rnn-parity)
            var j = k % per
            if j == 0:
                ag_epoch_order(ex, shuffle, seed, k // per, N, dord)
            var off = j * bs
            var B = min(bs, N - off)
            _rnn_options_step(ex, step, net, task, opts, setup, w, P, Gr, dX, dY, dL, dL_opt, dord, has_lens,
                              N, T, St, B, off, ycols, np_, seed, t0 + k, dloss + k, loss_ws, dord, off)
            if opts.clip > Float32(0.0):
                opt_step_clipped(ex, cfg, st, t0 + k + 1, ftz(lrs.unsafe_load(k)), P, Gr, s1, s2, s3, np_,
                                 setup.cell)
            else:
                opt_step(ex, cfg, st, t0 + k + 1, ftz(lrs.unsafe_load(k)), P, Gr, s1, s2, s3, np_)
            if opts.nv > 0 and j == per - 1:
                # the epoch's end: the validation loss, one word read back, the generic monitor's decision
                var e = k // per
                step.validation_loss(ex, net, task, opts, P, T, w, setup, best.vloss + e)
                ex.sync()
                ex.download(FP(unsafe_from_address=Int(vword.unsafe_ptr())), best.vloss + e, 1)
                epochs_run = e + 1
                if monitor.observe(Float64(vword[0]), e + 1) and opts.restore:
                    best.keep(ex, P, s1, s2, s3, np_, st)
                if monitor.stop():
                    break
        if opts.nv > 0:
            ex.sync()
            ex.download(opts.vcurve, best.vloss, epochs_run)
            opts.vcurve.unsafe_store(epochs, Float32(epochs_run))
            opts.vcurve.unsafe_store(epochs + 1, Float32(monitor.best_epoch))
            if opts.restore and monitor.best_epoch > 0:
                best.restore(ex, P, s1, s2, s3, np_, st)
                t0 -= (epochs_run - monitor.best_epoch) * per
        if opts.has_state:
            ex.sync()
            ex.download(opts.state, s1, np_)
            ex.download(opts.state + np_, s2, np_)
            ex.download(opts.state + 2 * np_, s3, np_)
            opts.state.unsafe_store(3 * np_, st.pw1)
            opts.state.unsafe_store(3 * np_ + 1, st.pw2)
            opts.state.unsafe_store(3 * np_ + 2, st.mu_prod)
            opts.state.unsafe_store(3 * np_ + 3, Float32(t0 + epochs_run * per))
        _ = vword^
        _ = best^
    for k in range(n_steps if plain else 0):
        var j = k % per
        if j == 0:
            # TOMBSTONE: MOJOLEARN_AUTOGRAD_DATA (promoted: autograd/data.mojo epoch_order is the one epoch order; the hand launches deleted) deleted 2026-11-08 by 75825f479c; code recoverable at c3b7dc42e6^.
            # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_DATA.patch
            ag_epoch_order(ex, shuffle, seed, k // per, N, dord)
        var off = j * bs
        var B = min(bs, N - off)
        gather_seq(ex, dX, dord, w.xb, T, B, net.D, off, dL_opt)
        if has_lens:
            gather_rows(ex, dL, dord, w.lens, B, 1, off)
        step.forward_head(ex, net, P, Gr, w.xb, T, B, w)
        if task == TASK_MSE:
            gather_rows(ex, dY, dord, w.yb, B, net.O, off)
            var a = Args()
            a.p0 = w.yhat
            a.p1 = w.yb
            a.p2 = w.dy
            a.f0 = Float32(2.0) / Float32(B * net.O)
            ex.launch[OP_MSE](a, B * net.O)
            loss_value_into(ex, LS_MSE, w.yhat, w.yb, B, net.O, REDUCTION_MEAN, dloss + k, loss_ws)
        else:
            gather_rows(ex, dY, dord, w.yb, B, 1, off)
            var a = Args()
            a.p0 = w.yhat
            a.p1 = w.yb
            a.p2 = w.dy
            a.i0 = B
            a.i1 = net.O
            a.f0 = Float32(1.0) / Float32(B)
            ex.launch[OP_CE](a, B)
            loss_value_into(ex, LS_CROSS_ENTROPY, w.yhat, w.yb, B, net.O, REDUCTION_MEAN, dloss + k, loss_ws)
        step.backward(ex, net, P, Gr, w.xb, T, B, w)
        # TOMBSTONE: MOJOLEARN_AUTOGRAD_OPTIM (decided B arm, the executor optimizer step_lr; L40S ba08dac006 gru/lstm/mlp/rnn fit B 0.99-1.01x, same speed) deleted 2026-11-17 by lane/fb12-loser-arms; code recoverable at 2203db7094.
        # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_OPTIM.fb12-loser-arms.patch; record in docs/TOMBSTONES.md.
        opt_step(ex, cfg, st, k + 1, ftz(lrs.unsafe_load(k)), P, Gr, s1, s2, s3, np_)
    release_loss_scratch(ex, loss_ws)
    ex.sync()
    fit_loop_publish(loop_words, fit_loop_stop(loop_start, epochs_run))
    ex.download(Pio, P, np_)
    ex.download(losses, dloss, n_steps)
    _ = setup^  # the options' device buffers live through the loop's queued launches


#: Lane stream-fits-2 fault injection (verify's sabotage arms of the streamed recurrent fit, never in a product
#: build), each on ONE side (the plain name the device executor, _HOST the host executor); lane ops-composed-cases
#: (case gru-stream-cw-val-resume) must REFUSE under either:
#:   CW_BATCH: each batch's loss weights are made BEFORE its targets are uploaded, so from the previous batch's labels
#:   (and record ids): class and sample weights from the wrong batch.
#:   VAL_AHEAD: the validation loss of a pass is taken one step late, after the read-ahead batch (the next pass's
#:   first) has trained.
comptime STREAM_FIT_SABOTAGE_CW_BATCH = is_defined["MOJOLEARN_STREAM_FIT_SABOTAGE_CW_BATCH"]()
comptime STREAM_FIT_SABOTAGE_CW_BATCH_HOST = is_defined["MOJOLEARN_STREAM_FIT_SABOTAGE_CW_BATCH_HOST"]()
comptime STREAM_FIT_SABOTAGE_VAL_AHEAD = is_defined["MOJOLEARN_STREAM_FIT_SABOTAGE_VAL_AHEAD"]()
comptime STREAM_FIT_SABOTAGE_VAL_AHEAD_HOST = is_defined["MOJOLEARN_STREAM_FIT_SABOTAGE_VAL_AHEAD_HOST"]()

def rnn_fit_stream[E: Exec, //, S: RnnStep](
    mut ex: E, net: Net, task: Int, mut src: BatchSource, total: Int, T: Int, seed: UInt64,
    Pio: FP, losses: FP, lrs: FP, cfg: OptConfig, init_acc: Float32, loop_words: Int, opts: RnnOptions,
    saved_out: Int, position_out: Int,
) raises:
    """The recurrent estimators' fit(source=) and partial_fit(source=) (lane stream-fits, 2026-11-25): at most
    `total` optimizer steps, each on the next batch of `src` (autograd/batch_source.mojo, kept by the door and set
    at the position to train from), crossing epochs of the source as they end. Every batch is padded to T steps
    (the source's longest
    record) and masked by its records' lengths; it is uploaded batch-first into staging buffers and laid out by the
    options loop's own gathers through an identity order (one launch each), so a step is `_rnn_options_step`'s, word
    for word, with the batch's rows where the array fit gathers its rows. The next batch is read from disk right
    after the step is queued (the device trains while the host reads); the waits bound the host staging
    (BATCH_SOURCE_STAGE_WORDS). The targets are the source's: one row of St ycols words a record (St = T with
    output="sequence": a target a step, padded rows past a record's length weigh 0; ycols = n_outputs for a
    regressor, one class index for a classifier). losses[k] is step k's loss; lrs[k] its rate. With
    opts.has_state the optimizer state (3 n_params + 4 words) comes in and goes out as in rnn_fit's options loop.
    saved_out receives the source part the estimator's state keeps (SOURCE_PART_WORDS int64 words: the consumed
    position and the source's binding words), position_out RNN_STREAM_POSITION_WORDS words: the consumed (epoch,
    batch) for the handle and the steps run.

    WEIGHTS (opts.has_cw / has_sw; lanes fable-finalize, stream-fits-2): each batch's loss weights are made from
    that batch's own labels and record ids, one OP_RNO_ROW_WEIGHTS launch over its B St rows after its targets are
    uploaded (the array fit's kernel and words: class weight of the row's label times the record's sample weight,
    one a step with sw_steps; the steps past a record's length then gather +0.0). The sample weights are per RECORD
    (opts.sw holds src.n or src.n T words in record order: the records numbered over the shards in index order),
    gathered by the batch's record ids. INITIAL STATES (opts.has_h0 / has_c0): h_0 / c_0 hold one state per record
    (L dirs src.n H words), gathered by record id as the array fit gathers them by row. 'balanced' class weights
    arrive in opts.cw, counted by the caller in the one admission read of the targets.

    VALIDATION (opts.nv > 0): at the end of every pass over the source (the step that trains its last batch) the
    validation loss of the parameters (the array fit's `validation_loss`) into the next word of the curve; the
    generic EpochMonitor decides over the passes, carried across calls by the RNN_STREAM_MONITOR_WORDS after the
    curve's two result words (opts.vcurve[epochs + 2 ...]: best loss, stall, best pass, passes before), so a
    partial_fit resumed in a fresh process monitors as the uninterrupted run. EARLY STOPPING (opts.patience > 0):
    the pass's word is read back (one download a pass, as the array fit's epoch) and the call stops after the pass
    the monitor stops at. RESTORE (opts.restore, fit only): the best pass's parameters and optimizer state are kept
    on the executor and restored at the end; the position and the optimizer's step count become the best pass's
    end, so a later call continues from the restored model. opts.vcurve receives the passes' losses, then the passes
    completed and the best pass (counted over the calls), then the monitor words."""
    var np_ = net.n_params()
    var ycols = net.O if task == TASK_MSE else 1
    var St = T if opts.seq_out else 1
    var bmax = min(src.spec.batch, src.n)
    if src.width != net.D:
        raise Error("ValueError: fit(source=): the source's records have " + String(src.width) + " features; the "
                    + "network reads " + String(net.D))
    if T < src.max_len:
        raise Error("ValueError: fit(source=): the sequences are padded to " + String(T) + " steps; the source's "
                    + "longest record has " + String(src.max_len))
    if src.twidth != St * ycols:
        raise Error("ValueError: fit(source=): each record's target row must hold " + String(St * ycols)
                    + " words (" + (String("a target a step, padded to the longest record") if opts.seq_out
                    else String("one target")) + "); the source's targets have width " + String(src.twidth))
    if total < 1:
        raise Error("ValueError: fit(source=): no step to train")
    # the per-record arrays are gathered by float32 record ids: exact below 2^24
    var by_id = opts.has_sw or opts.has_h0
    if by_id and src.n >= 16777216:
        raise Error("ValueError: fit(source=): per-record sample weights and initial states need fewer than 2^24 "
                    + "records")
    var sx = ex.alloc(bmax * T * net.D)
    var sy = ex.alloc(bmax * St * ycols)
    var sl = ex.alloc(bmax)
    var sl_opt = Optional[FP](sl)
    var dord = ex.alloc(bmax)
    var sid = ex.alloc(bmax if by_id else 1)
    var ia = Args()
    ia.p0 = dord
    ex.launch[OP_SEQ_IOTA](ia, bmax)
    var P = ex.alloc(np_)
    ex.upload(P, Pio, np_)
    var Gr = ex.alloc(np_)
    var s1 = ex.alloc(np_)
    var s2 = ex.alloc(np_)
    var s3 = ex.alloc(np_)
    if init_acc != Float32(0.0):
        fill(ex, s2, np_, init_acc)
    var dloss = ex.alloc(total)
    var w = Work(ex, net, T, bmax, True, True, opts)
    var loss_ws = loss_scratch(ex, St * bmax * net.O)
    var st = OptState()
    var step = S()
    # the options' setup over one batch's rows (the weights of every padded step 0 through the lengths); the
    # per-record weights and states over the source's records
    var setup = RnnFitSetup(ex, net, task, opts, bmax, St, np_, P, sy, True, w, T, ycols, bmax, src.n)
    var weigh = setup.weighted and (opts.has_cw or opts.has_sw)
    var best = RnnBest(ex, opts, np_)
    var passes = 0
    var t0 = 0
    if opts.has_state:
        ex.upload(s1, opts.state, np_)
        ex.upload(s2, opts.state + np_, np_)
        ex.upload(s3, opts.state + 2 * np_, np_)
        st.pw1 = opts.state.unsafe_load(3 * np_)
        st.pw2 = opts.state.unsafe_load(3 * np_ + 1)
        st.mu_prod = opts.state.unsafe_load(3 * np_ + 2)
        t0 = Int(opts.state.unsafe_load(3 * np_ + 3))
    # the monitor over the passes, as an earlier call (partial_fit) left it
    var monitor = EpochMonitor(Float64(opts.tol), opts.patience)
    var done0 = 0
    if opts.nv > 0:
        var carry = opts.vcurve + opts.epochs + 2
        done0 = Int(carry.unsafe_load(3))
        if done0 > 0:
            monitor = EpochMonitor.resumed(Float64(opts.tol), opts.patience, Float64(carry.unsafe_load(0)),
                                           Int(carry.unsafe_load(1)))
            monitor.best_epoch = Int(carry.unsafe_load(2))
    var watch = opts.nv > 0 and (opts.patience > 0 or opts.restore)  # a decision a pass, inside the loop
    var vword = List[Float32](length=1, fill=0.0)
    var best_epoch_src = -1  # the source epoch whose end the kept (best) pass is, when restoring
    var best_steps = 0
    var ran = total
    var late = False  # sabotage VAL_AHEAD only: a pass end whose validation is owed
    var cw_early = False
    comptime if (STREAM_FIT_SABOTAGE_CW_BATCH and E.IS_DEVICE) or (STREAM_FIT_SABOTAGE_CW_BATCH_HOST
                                                                  and not E.IS_DEVICE):
        cw_early = True  # SABOTAGE: the weights of the previous batch's labels and records
    var val_ahead = False
    comptime if (STREAM_FIT_SABOTAGE_VAL_AHEAD and E.IS_DEVICE) or (STREAM_FIT_SABOTAGE_VAL_AHEAD_HOST
                                                                   and not E.IS_DEVICE):
        val_ahead = True  # SABOTAGE: the validation after the read-ahead batch has trained
    ex.sync()
    var loop_start = fit_loop_start()
    var staged = 0
    for k in range(total):  # driver-loop: over optimizer steps (one streamed batch each)
        var at = src.pending()
        src.epoch = at[0]
        src.consumed = at[1]
        if not src.holds(at[0], at[1], T):
            src.read(at[0], at[1], T)
        if staged >= BATCH_SOURCE_STAGE_WORDS:
            ex.sync()
            staged = 0
        var B = src.held_rows
        if weigh and cw_early:
            _stream_weights(ex, opts, setup, w, sy, sid, B, St)
        staged += batch_source_upload(ex, src, BS_FIELD_STEPS, sx).count()
        staged += batch_source_upload(ex, src, BS_FIELD_TARGETS, sy).count()
        staged += batch_source_upload(ex, src, BS_FIELD_LENGTHS, sl).count()
        if by_id:
            staged += batch_source_upload_ids(ex, src, sid)
        if weigh and not cw_early:
            # this batch's loss weights from its own labels and record ids (rows past B keep an earlier batch's
            # and are not read)
            _stream_weights(ex, opts, setup, w, sy, sid, B, St)
        _rnn_options_step(ex, step, net, task, opts, setup, w, P, Gr, sx, sy, sl, sl_opt, dord, True,
                          bmax, T, St, B, 0, ycols, np_, seed, t0 + k, dloss + k, loss_ws, sid, 0)
        if opts.clip > Float32(0.0):
            opt_step_clipped(ex, cfg, st, t0 + k + 1, ftz(lrs.unsafe_load(k)), P, Gr, s1, s2, s3, np_,
                             setup.cell)
        else:
            opt_step(ex, cfg, st, t0 + k + 1, ftz(lrs.unsafe_load(k)), P, Gr, s1, s2, s3, np_)
        # this batch is consumed (its step is queued); the next one is read while the device trains this one
        src.consumed = at[1] + 1
        var pass_end = at[1] + 1 == src.n_batches
        var due = pass_end
        if val_ahead:
            due = late
            late = pass_end
        if opts.nv > 0 and due and passes < opts.epochs:
            # a pass over the source ends with this step: the validation loss of its parameters
            step.validation_loss(ex, net, task, opts, P, T, w, setup, best.vloss + passes)
            passes += 1
            if watch:
                # the monitor's decision, one word read back a pass (as the array fit's epoch)
                ex.sync()
                ex.download(FP(unsafe_from_address=Int(vword.unsafe_ptr())), best.vloss + passes - 1, 1)
                if monitor.observe(Float64(vword[0]), done0 + passes) and opts.restore:
                    best.keep(ex, P, s1, s2, s3, np_, st)
                    best_epoch_src = at[0]
                    best_steps = k + 1
                if monitor.stop():
                    ran = k + 1
                    break
        var ahead = src.next_position(at[0], at[1])
        src.read(ahead[0], ahead[1], T)
    _ = vword^
    release_loss_scratch(ex, loss_ws)
    ex.sync()
    fit_loop_publish(loop_words, fit_loop_stop(loop_start, ran))
    var kept = t0 + ran
    if best_epoch_src >= 0:
        # restore_best: the best pass's parameters and optimizer state, the source at that pass's end
        best.restore(ex, P, s1, s2, s3, np_, st)
        kept = t0 + best_steps
        src.epoch = best_epoch_src
        src.consumed = src.n_batches
        ex.sync()
    ex.download(Pio, P, np_)
    ex.download(losses, dloss, ran)
    if opts.nv > 0:
        if passes > 0:
            ex.download(opts.vcurve, best.vloss, passes)
        if not watch:
            for e in range(passes):  # driver-loop: over the completed passes (the monitor on the host words)
                _ = monitor.observe(Float64(opts.vcurve.unsafe_load(e)), done0 + e + 1)
        opts.vcurve.unsafe_store(opts.epochs, Float32(passes))
        opts.vcurve.unsafe_store(opts.epochs + 1, Float32(monitor.best_epoch))
        var carry = opts.vcurve + opts.epochs + 2
        carry.unsafe_store(0, Float32(monitor.best))
        carry.unsafe_store(1, Float32(monitor.stall))
        carry.unsafe_store(2, Float32(monitor.best_epoch))
        carry.unsafe_store(3, Float32(done0 + passes))
    if opts.has_state:
        ex.download(opts.state, s1, np_)
        ex.download(opts.state + np_, s2, np_)
        ex.download(opts.state + 2 * np_, s3, np_)
        opts.state.unsafe_store(3 * np_, st.pw1)
        opts.state.unsafe_store(3 * np_ + 1, st.pw2)
        opts.state.unsafe_store(3 * np_ + 2, st.mu_prod)
        opts.state.unsafe_store(3 * np_ + 3, Float32(kept))
    source_part_store(batch_source_fit_saved[E.IS_DEVICE](src),
                      UnsafePointer[Int64, MutUntrackedOrigin](unsafe_from_address=saved_out))
    var position = UnsafePointer[Int64, MutUntrackedOrigin](unsafe_from_address=position_out)
    position[0] = Int64(src.epoch)
    position[1] = Int64(src.consumed)
    position[2] = Int64(ran)
    _ = setup^  # the options' device buffers live through the loop's queued launches
    _ = sl_opt^
    _ = best^


def _stream_weights[E: Exec](
    mut ex: E, opts: RnnOptions, setup: RnnFitSetup, w: Work, labels: FP, sid: FP, B: Int, St: Int,
) raises:
    """One streamed batch's loss weights into setup.wt (B St rows, batch-first): the array fit's OP_RNO_ROW_WEIGHTS
    over this batch's labels (`labels`, its uploaded targets) with the class weights (setup.cw_dev) and its records'
    sample weights (gathered from setup.sw_dev by the record ids `sid`, one launch)."""
    var a = Args()
    a.p0 = w.one
    a.p1 = labels
    a.p2 = w.one
    a.p3 = setup.wt
    a.i0 = St
    if opts.has_sw:
        gather_rows(ex, setup.sw_dev, sid, setup.swb, B, St if opts.sw_steps else 1, 0)
        a.p0 = setup.swb
        a.i1 = 1
        a.i2 = 1 if opts.sw_steps else 0
    if opts.has_cw:
        a.p2 = setup.cw_dev
        a.i3 = 1
    comptime if RNN_PARITY_SABOTAGE_WEIGHT_DEVICE and E.IS_DEVICE:
        a.i1 = 0
    comptime if RNN_PARITY_SABOTAGE_WEIGHT_HOST and not E.IS_DEVICE:
        a.i1 = 0
    ex.launch[OP_RNO_ROW_WEIGHTS](a, B * St)


struct RnnBest(Movable):
    """Lane rnn-parity: the per-epoch validation losses (`vloss`, one executor word an epoch) and the best
    epoch's parameters and optimizer state (kept on the executor, copied when the generic EpochMonitor says an
    epoch improved and opts.restore asks for it; restored at the end). Placeholders without validation."""

    var vloss: FP
    var p: FP
    var s1: FP
    var s2: FP
    var s3: FP
    var pw1: Float32
    var pw2: Float32
    var mu: Float32

    def __init__[E: Exec](out self, mut ex: E, opts: RnnOptions, np_: Int) raises:
        var keep = opts.nv > 0 and opts.restore
        self.vloss = ex.alloc(max(1, opts.vcurve_words()))
        self.p = ex.alloc(np_ if keep else 1)
        self.s1 = ex.alloc(np_ if keep else 1)
        self.s2 = ex.alloc(np_ if keep else 1)
        self.s3 = ex.alloc(np_ if keep else 1)
        self.pw1 = Float32(1.0)
        self.pw2 = Float32(1.0)
        self.mu = Float32(1.0)

    def keep[E: Exec](mut self, mut ex: E, P: FP, s1: FP, s2: FP, s3: FP, n: Int, st: OptState) raises:
        _copy_words(ex, self.p, P, n)
        _copy_words(ex, self.s1, s1, n)
        _copy_words(ex, self.s2, s2, n)
        _copy_words(ex, self.s3, s3, n)
        self.pw1 = st.pw1
        self.pw2 = st.pw2
        self.mu = st.mu_prod

    def restore[E: Exec](self, mut ex: E, P: FP, s1: FP, s2: FP, s3: FP, n: Int, mut st: OptState) raises:
        _copy_words(ex, P, self.p, n)
        _copy_words(ex, s1, self.s1, n)
        _copy_words(ex, s2, self.s2, n)
        _copy_words(ex, s3, self.s3, n)
        st.pw1 = self.pw1
        st.pw2 = self.pw2
        st.mu_prod = self.mu


def _copy_words[E: Exec](mut ex: E, dst: FP, src: FP, n: Int) raises:
    var a = Args()
    a.p0 = src
    a.p1 = dst
    ex.launch[OP_COPY](a, n)


struct RnnFitSetup(Movable):
    """Lane rnn-parity: the executor buffers of rnn_fit's options, made once before the loop. `wt` the fit's
    N S loss weights (OP_RNO_ROW_WEIGHTS: sample weight times class weight), when weighted; `h0` / `c0` the
    uploaded initial states (L dirs N H words); `parts` and `cell` the gradient clip's block partials and its
    [total norm, coefficient] words (the generic loop's clip, autograd/optim.mojo). Placeholders (w.one) for what is unused, so a
    plain fit allocates nothing here."""

    var weighted: Bool
    var wt: FP
    var h0: FP
    var c0: FP
    var parts: FP
    var cell: FP
    var block: Int
    var nparts: Int
    #: validation (opts.nv > 0): the sequences, targets and lengths on the executor, their identity order, a
    #: weight of 1.0 per row (or step), and two words per chunk (the chunk's loss sum and its row count)
    var vx: FP
    var vy: FP
    var vl: FP
    var vidx: FP
    var vones: FP
    var vparts: FP
    var vchunk: Int
    #: a streamed fit (`stream_rows` > 0, the source's record count; lane fable-finalize, stream-fits-2): the class
    #: weights (`cw_dev`, O words) and the per-record sample weights (`sw_dev`, stream_rows or stream_rows T words,
    #: record order) stay on the executor; each batch's loss weights are made from its own labels and record ids
    #: (`stream_weights`) into `wt`, through `swb` (the batch's sample weights, gathered by record id). An array
    #: fit weighs all N rows once here and frees them.
    var cw_dev: FP
    var sw_dev: FP
    var swb: FP
    #: the rows of the initial states h0 / c0 (N for an array fit, the source's records for a streamed one)
    var state_rows: Int

    def __init__[E: Exec](
        out self, mut ex: E, net: Net, task: Int, opts: RnnOptions, N: Int, St: Int, np_: Int, P: FP, dY: FP,
        has_lens: Bool, w: Work, T: Int, ycols: Int, bmax: Int, stream_rows: Int = 0,
    ) raises:
        self.cw_dev = w.one
        self.sw_dev = w.one
        self.swb = w.one
        self.state_rows = stream_rows if stream_rows > 0 else N
        self.weighted = opts.weighted(has_lens)
        self.wt = w.one
        self.h0 = w.one
        self.c0 = w.one
        self.parts = w.one
        self.cell = w.one
        self.block = 1
        self.nparts = 0
        self.vx = w.one
        self.vy = w.one
        self.vl = w.one
        self.vidx = w.one
        self.vones = w.one
        self.vparts = w.one
        self.vchunk = bmax
        if opts.nv > 0:
            var nv = opts.nv
            self.vx = ex.alloc(nv * T * net.D)
            ex.upload(self.vx, opts.vx, nv * T * net.D)
            self.vy = ex.alloc(nv * St * ycols)
            ex.upload(self.vy, opts.vy, nv * St * ycols)
            if opts.has_vl:
                self.vl = ex.alloc(nv)
                ex.upload(self.vl, opts.vl, nv)
            self.vidx = ex.alloc(nv)
            var ia = Args()
            ia.p0 = self.vidx
            ex.launch[OP_SEQ_IOTA](ia, nv)
            self.vones = ex.alloc(nv * St)
            fill(ex, self.vones, nv * St, Float32(1.0))
            self.vparts = ex.alloc(2 * ((nv + bmax - 1) // bmax))
        if self.weighted:
            self.wt = ex.alloc(N * St)
            if opts.has_cw and task != TASK_CE:
                raise Error("rnn_fit: class weights need the cross-entropy task")
            if stream_rows > 0 and (opts.has_sw or opts.has_cw):
                # a streamed fit: no batch is on the executor yet; the weights stay for `stream_weights`
                if opts.has_sw:
                    var n_sw = stream_rows * St if opts.sw_steps else stream_rows
                    self.sw_dev = ex.alloc(n_sw)
                    ex.upload(self.sw_dev, opts.sw, n_sw)
                    self.swb = ex.alloc(N * St if opts.sw_steps else N)
                if opts.has_cw:
                    self.cw_dev = ex.alloc(net.O)
                    ex.upload(self.cw_dev, opts.cw, net.O)
            else:
                var a = Args()
                a.p0 = w.one
                a.p1 = dY
                a.p2 = w.one
                a.p3 = self.wt
                a.i0 = St
                if opts.has_sw:
                    var n_sw = N * St if opts.sw_steps else N
                    a.p0 = ex.alloc(n_sw)
                    ex.upload(a.p0, opts.sw, n_sw)
                    a.i1 = 1
                    a.i2 = 1 if opts.sw_steps else 0
                if opts.has_cw:
                    a.p2 = ex.alloc(net.O)
                    ex.upload(a.p2, opts.cw, net.O)
                    a.i3 = 1
                comptime if RNN_PARITY_SABOTAGE_WEIGHT_DEVICE and E.IS_DEVICE:
                    a.i1 = 0
                comptime if RNN_PARITY_SABOTAGE_WEIGHT_HOST and not E.IS_DEVICE:
                    a.i1 = 0
                ex.launch[OP_RNO_ROW_WEIGHTS](a, N * St)
                # the uploaded weights are read by the launch above only; freed after it
                ex.sync()
                if opts.has_sw:
                    ex.free(a.p0)
                if opts.has_cw:
                    ex.free(a.p2)
        var nh = net.slots() * self.state_rows * net.H
        if opts.has_h0:
            self.h0 = ex.alloc(nh)
            ex.upload(self.h0, opts.h0, nh)
            if opts.has_c0:
                self.c0 = ex.alloc(nh)
                ex.upload(self.c0, opts.c0, nh)
        if opts.no_bias:
            # bias=False: the recurrent layers' biases start (and stay) at 0; b_ih and b_hh are adjacent
            var GH = net.G() * net.H
            for l in range(net.L):  # driver-loop: over recurrent layers
                for d in range(net.dirs):  # driver-loop: over directions
                    fill(ex, P + net.b_ih(l, d), 2 * GH, Float32(0.0))
        if opts.clip > Float32(0.0):
            # the generic clip's blocks (autograd/optim.mojo OptimizerState): ag_fold_block(n) gradient words each
            self.block = ag_fold_block(np_)
            self.nparts = (np_ + self.block - 1) // self.block
            self.parts = ex.alloc(self.nparts)
            self.cell = ex.alloc(2)


def _rnn_options_step[E: Exec, S: RnnStep](
    mut ex: E, mut step: S, net: Net, task: Int, opts: RnnOptions, setup: RnnFitSetup, w: Work,
    P: FP, Gr: FP, dX: FP, dY: FP, dL: FP, dL_opt: Optional[FP], dord: FP, has_lens: Bool,
    N: Int, T: Int, St: Int, B: Int, off: Int, ycols: Int, np_: Int, seed: UInt64, k: Int, loss_dst: FP,
    loss_ws: LossScratch, sid: FP, soff: Int,
) raises:
    """One optimizer step of rnn_fit's options loop up to the gradient in Gr (lane rnn-parity): the batch's
    sequences, lengths and initial states, the forward, the targets and weights, the loss and the backward, then
    bias=False's zeroed bias gradients and the gradient clip's norm. The update is the caller's. `sid` / `soff`
    (lane stream-fits-2): the order the initial states are gathered in (setup.state_rows rows): the batch order
    `dord` / `off` of an array fit, the batch's record ids at 0 for a streamed one."""
    var H = net.H
    gather_seq(ex, dX, dord, w.xb, T, B, net.D, off, dL_opt)
    if has_lens:
        gather_rows(ex, dL, dord, w.lens, B, 1, off)
    if opts.has_h0:
        var SR = setup.state_rows
        for kk in range(net.slots()):  # driver-loop: over layer-direction slots
            gather_rows(ex, setup.h0 + kk * SR * H, sid, w.h0b + kk * B * H, B, H, soff)
            if opts.has_c0:
                gather_rows(ex, setup.c0 + kk * SR * H, sid, w.c0b + kk * B * H, B, H, soff)
    step.forward_options(ex, net, opts, P, Gr, w.xb, T, B, w, seed, k)
    var R = B * St
    if opts.seq_out:
        # every step's target, time-major like the head's rows (a padded step keeps its target; its weight is 0)
        gather_seq(ex, dY, dord, w.yb, T, B, ycols, off)
    else:
        gather_rows(ex, dY, dord, w.yb, B, ycols, off)
    if setup.weighted:
        if opts.seq_out:
            # the steps at or past a row's length gather +0.0: their loss weight is 0
            gather_seq(ex, setup.wt, dord, w.wrow, T, B, 1, off, dL_opt)
        else:
            gather_rows(ex, setup.wt, dord, w.wrow, B, 1, off)
        step.weighted_backward(ex, net, task, R, w, loss_dst)
    else:
        var a = Args()
        a.p0 = w.yhat
        a.p1 = w.yb
        a.p2 = w.dy
        if task == TASK_MSE:
            a.f0 = Float32(2.0) / Float32(R * net.O)
            ex.launch[OP_MSE](a, R * net.O)
            loss_value_into(ex, LS_MSE, w.yhat, w.yb, R, net.O, REDUCTION_MEAN, loss_dst, loss_ws)
        else:
            a.i0 = R
            a.i1 = net.O
            a.f0 = Float32(1.0) / Float32(R)
            ex.launch[OP_CE](a, R)
            loss_value_into(ex, LS_CROSS_ENTROPY, w.yhat, w.yb, R, net.O, REDUCTION_MEAN, loss_dst, loss_ws)
        step.backward(ex, net, P, Gr, w.xb, T, R, w)
    if opts.no_bias:
        var GH = net.G() * H
        for l in range(net.L):  # driver-loop: over recurrent layers
            for d in range(net.dirs):  # driver-loop: over directions
                fill(ex, Gr + net.b_ih(l, d), 2 * GH, Float32(0.0))
    if opts.clip > Float32(0.0):
        # torch.nn.utils.clip_grad_norm_ as the generic training loop's clip block (autograd/optim.mojo
        # clip_norm_accumulate over one tensor, the whole gradient): block partials of g^2, then [norm, coef] in
        # setup.cell; the caller's update scales by the coefficient in the same launch (opt_step_clipped)
        var p = Args()
        p.p0 = Gr
        p.p1 = setup.parts
        p.i0 = np_
        p.i1 = setup.block
        ex.launch[OP_AGO_SUMSQ_PART](p, setup.nparts)
        var q = Args()
        q.p0 = setup.parts
        q.p1 = setup.cell
        q.i0 = setup.nparts
        q.i1 = 3
        q.f0 = opts.clip
        ex.launch[OP_AGO_NORM_ACC](q, 1)


# TOMBSTONE: MOJOLEARN_AUTOGRAD_PREDICT (folded in: the tape forward is the one predict path) deleted 2026-11-08 by 9a9a8e3148; code recoverable at c3b7dc42e6^.
# Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_PREDICT.patch
# rnn_predict moved to autograd/rules_recurrent.mojo (it runs the tape forward, and that module imports this one).
