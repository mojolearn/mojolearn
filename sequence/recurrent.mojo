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
from autograd.optim import OP_AGO_SUMSQ_PART, OP_AGO_NORM_ACC, OP_AGH_OPT_SCALED
from autograd.ops import ag_fold_block
from std.sys.compile import is_defined

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

    def plain(self, net: Net) -> Bool:
        """No option on: the launches of the last-step model of before."""
        return (net.dirs == 1 and not self.seq_out and not self.no_bias and self.dropout == Float32(0.0)
                and self.clip == Float32(0.0) and not self.has_sw and not self.has_cw and not self.has_h0)

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
        if train and opts.weighted(has_lens):
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


# Lane rnn-parity fault injection (verify's sabotage arms of the weighted fit, never in a product build): the
# loss weights drop their sample-weight factor on ONE side, the device executor's fit or the host executor's.
# The verify lane sequence-rnn-weighted-steps (tools/identity_lanes/sequence.py) must FAIL under either.
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
    var setup = RnnFitSetup(ex, net, task, opts, N, St, np_, P, dY, has_lens, w)
    # lane alm-left: the queued setup (uploads, fills) drained ONCE before the
    # clock, outside the loop, so the loop region holds loop work only
    ex.sync()
    var loop_start = fit_loop_start()
    if not plain:
        for k in range(n_steps):  # driver-loop: over optimizer steps (the options loop, lane rnn-parity)
            var j = k % per
            if j == 0:
                ag_epoch_order(ex, shuffle, seed, k // per, N, dord)
            var off = j * bs
            var B = min(bs, N - off)
            _rnn_options_step(ex, step, net, task, opts, setup, w, P, Gr, dX, dY, dL, dL_opt, dord, has_lens,
                              N, T, St, B, off, ycols, np_, seed, k, dloss + k, loss_ws)
            if opts.clip > Float32(0.0):
                opt_step_clipped(ex, cfg, st, k + 1, ftz(lrs.unsafe_load(k)), P, Gr, s1, s2, s3, np_, setup.cell)
            else:
                opt_step(ex, cfg, st, k + 1, ftz(lrs.unsafe_load(k)), P, Gr, s1, s2, s3, np_)
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
    fit_loop_publish(loop_words, fit_loop_stop(loop_start, epochs))
    ex.download(Pio, P, np_)
    ex.download(losses, dloss, n_steps)
    _ = setup^  # the options' device buffers live through the loop's queued launches


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

    def __init__[E: Exec](
        out self, mut ex: E, net: Net, task: Int, opts: RnnOptions, N: Int, St: Int, np_: Int, P: FP, dY: FP,
        has_lens: Bool, w: Work,
    ) raises:
        self.weighted = opts.weighted(has_lens)
        self.wt = w.one
        self.h0 = w.one
        self.c0 = w.one
        self.parts = w.one
        self.cell = w.one
        self.block = 1
        self.nparts = 0
        if self.weighted:
            self.wt = ex.alloc(N * St)
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
                if task != TASK_CE:
                    raise Error("rnn_fit: class weights need the cross-entropy task")
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
        var nh = net.slots() * N * net.H
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
    loss_ws: LossScratch,
) raises:
    """One optimizer step of rnn_fit's options loop up to the gradient in Gr (lane rnn-parity): the batch's
    sequences, lengths and initial states, the forward, the targets and weights, the loss and the backward, then
    bias=False's zeroed bias gradients and the gradient clip's norm. The update is the caller's."""
    var H = net.H
    gather_seq(ex, dX, dord, w.xb, T, B, net.D, off, dL_opt)
    if has_lens:
        gather_rows(ex, dL, dord, w.lens, B, 1, off)
    if opts.has_h0:
        for kk in range(net.slots()):  # driver-loop: over layer-direction slots
            gather_rows(ex, setup.h0 + kk * N * H, dord, w.h0b + kk * B * H, B, H, off)
            if opts.has_c0:
                gather_rows(ex, setup.c0 + kk * N * H, dord, w.c0b + kk * B * H, B, H, off)
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
