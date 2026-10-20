# SPDX-License-Identifier: Apache-2.0
#
# PASS 2 (lane ag-pass2-kernels, 2026-10-19): LAUNCHES AND MEMORY PASSES PER
# TAPE OPERATION, before -> after, read from the code (r x c the node's
# output; "pass" = one read or write of an r x c matrix). Measure with
# MOJOLEARN_AG_PROFILE=1 (counts) or 2 (synced host-clock times, relative).
#   binary fwd (add/sub/mul/div): 1 -> 1 launch; 3 passes -> 3; per element
#     two integer divisions per operand -> none when the operand has the
#     output's shape (_x_at / _y_at), kept for a broadcast operand.
#   binary VJP, operand of the output's shape: 1 -> 1; no index divisions.
#   binary VJP, broadcast operand (bias, row, column, scalar): 3 -> 2
#     launches (OP_AG_BIN_VJP into scratch + REDUCE_PART + REDUCE_FOLD ->
#     OP_AGK_BIN_VJP_PART + REDUCE_FOLD); passes 4-5 -> 2-3 (the scratch
#     write and its re-read are gone); same terms, same order, same bits.
#     The default tape MLP: one launch fewer per layer per batch.
#   unary fwd / VJP, reduce VJP, accumulate: unchanged (1 launch, 2-3 passes;
#     the hand bodies are scalar, one element a thread, so are these).
#   softmax fwd: 1 -> 1 launch per row block; per row 3 reads of x + 1 write
#     and 2C exponentials -> 2 reads of x, 2 writes + 1 read of y and C
#     exponentials (each exponential parked in y); same words. Log-softmax
#     unchanged. Softmax VJP unchanged (1 launch, one dot per row).
#   sum / mean: 2 -> 2 launches (blocked fold, unchanged order).
#   matmul fwd and its two VJP GEMMs: 1 launch each, unchanged; already
#     OP_GEMM, which DeviceExec runs on the tiled kernel (sequence/
#     gemm_tiled.mojo) on NVIDIA and AMD in identical, op_gemm's chain; the
#     transposed operands are GEMM strides, so 0 transpose launches before
#     and after (OP_AG_TRANSPOSE runs only for an explicit transpose node).
#   fused linear fwd, dW, dX (gemm_epi): 1 launch each -> 1, now
#     OP_AGK_GEMM_EPI: on NVIDIA/AMD identical the tiled GEMM with the
#     epilogue on the stored word (each A row and B column staged once a
#     tile instead of re-read by every cell's thread); same words. Apple and
#     the host run op_gemm_epi as before.
#   fused linear db: 1 launch (OP_COLSUM_DIV, one thread a column, a serial
#     chain over all batch rows) -> unchanged for batches of at most 64 rows;
#     above, 2 launches (REDUCE_PART over ceil(B / R) x O threads with chains
#     of R = ag_fold_block(B), then REDUCE_FOLD) under AUTOGRAD_PASS2_FOLD:
#     THE ONLY BIT CHANGE of this pass (new bits above 64 rows).
# The blocked fold itself (REDUCE_PART / REDUCE_FOLD) was already two-level
# for the B x O column sums (ceil(B / R) x O threads, chains <= R), not one
# thread per column: its block rule and order are unchanged.
#
# Native element bodies shared by HostExec and DeviceExec. Graph traversal
# and gradient accumulation order live in tape.mojo, not inside GPU atomics.
#
# One operation code per body (the library's design, sequence/dispatch.mojo).
# Each body reads and writes only the pointer slots listed in its comment;
# the tape binds every other slot to the Args() placeholder, as every other
# sequence operation does, and never binds one buffer to two slots of a
# launch. No body reads back a slot it writes, except OP_AG_ACCUMULATE's
# element-wise read-modify-write of its own destination word,
# OP_AG_SOFTMAX_FWD's read-back of its own row's exponentials and
# OP_AGF_LEVEL's in-place fold of its own group of partials.
from sequence.ops import Args, FP, ld, st, add, sub, mul
from checks.numerics import (
    ftz, identical_div, identical_exp, identical_log, identical_sigmoid, identical_sqrt,
    identical_tanh,
)
from std.sys.compile import is_defined

# Sequence codes 120/121 are recurrent scan kernels, intercepted before the
# generic dispatcher. Codes OP_AG_FIRST..OP_AG_LAST are this family, on both
# executors (sequence/dispatch.mojo asserts the range is free).
comptime OP_AG_BIN_FWD = 1000
comptime OP_AG_BIN_VJP = 1001
comptime OP_AG_UNARY_FWD = 1002
comptime OP_AG_UNARY_VJP = 1003
comptime OP_AG_TRANSPOSE = 1004
comptime OP_AG_REDUCE_PART = 1005
comptime OP_AG_REDUCE_FOLD = 1006
comptime OP_AG_REDUCE_VJP = 1007
comptime OP_AG_SOFTMAX_FWD = 1008
comptime OP_AG_SOFTMAX_VJP = 1009
comptime OP_AG_ACCUMULATE = 1010
comptime OP_AG_FIRST = OP_AG_BIN_FWD
comptime OP_AG_LAST = OP_AG_ACCUMULATE
# Lane ag-pass2-kernels (2026-10-19): the tape's kernel family, a range of
# its own (1400..), registered in sequence/dispatch.mojo.
#   OP_AGK_BIN_VJP_PART: a broadcast operand's binary VJP terms folded
#     straight into the blocked fold's partials (op_agk_bin_vjp_part), so
#     the elementwise VJP matrix is never written: the same terms in the
#     same order as OP_AG_BIN_VJP into scratch then OP_AG_REDUCE_PART.
#   OP_AGK_GEMM_EPI: OP_GEMM_EPI's body (sequence/mlp.mojo op_gemm_epi) on
#     the tiled GEMM where OP_GEMM is tiled (NVIDIA and AMD in identical,
#     sequence/gemm_tiled.mojo seq_gemm_epi_tiled_kernel); the same chain
#     and the same epilogue, so the same words as OP_GEMM_EPI everywhere.
comptime OP_AGK_BIN_VJP_PART = 1400
comptime OP_AGK_GEMM_EPI = 1401
comptime OP_AGK_FIRST = OP_AGK_BIN_VJP_PART
comptime OP_AGK_LAST = OP_AGK_GEMM_EPI


# Lane ag-pass3-impl (PASS3_IDEAS idea 2): OP_AGK_GEMM_EPI with i11 = 1 also
# writes p4[n] = (sum over k ascending of B(k, n)) / f1, B dense (i5 = N,
# i6 = 1): the fused linear node's bias gradient from its dW launch. The
# tiled kernel folds it from the staged B slab (sequence/gemm_tiled.mojo);
# where the GEMM runs elsewhere (the one-thread launch, Apple, the host) the
# executor follows it with this OP_COLSUM_DIV launch. Both are op_colsum
# then op_divs, the hand MLP's words.
@always_inline
def agk_colsum_args(a: Args) -> Args:
    """OP_COLSUM_DIV's arguments for OP_AGK_GEMM_EPI's column sums."""
    var c = Args()
    c.p0 = a.p1
    c.p1 = a.p4
    c.i0 = a.i2
    c.i2 = a.i5
    c.f0 = a.f1
    return c

# BIT VERSION (lane ag-pass2-kernels, default on): the fused linear node's
# bias gradient (tape.linear / linear_into, edge 2) is the tape's blocked
# fold over the batch rows (two launches, chains of ag_fold_block(rows))
# instead of the hand MLP's OP_COLSUM / OP_COLSUM_DIV (one thread per
# output column, one serial chain over every row). A batch of at most
# ag_fold_block(rows) = 64 rows is one block: the fold is then the column
# sum's chain term for term and keeps the one launch, so its bits are the
# old bits; longer batches get new bits on every column and the host
# together. -D MOJOLEARN_AUTOGRAD_PASS2_FOLD_OFF restores the column sum.
comptime AUTOGRAD_PASS2_FOLD = not is_defined["MOJOLEARN_AUTOGRAD_PASS2_FOLD_OFF"]()

# BIT VERSION (lane ag-pass3-impl, 2026-10-19, default on; PASS3_IDEAS idea
# 2): a fused linear node with a bias, a divisor and both a weight and a bias
# gradient takes db from its dW launch (OP_AGK_GEMM_EPI i11 = 1: the column
# sums of dz folded from the tile's staged slab on NVIDIA/AMD, the hand
# path's OP_COLSUM_DIV launch after the GEMM on Apple, the host and the
# one-thread GEMM). db is then OP_COLSUM_DIV's word on every column: the
# hand MLP's bits, i.e. AUTOGRAD_PASS2_FOLD's change is undone for these
# nodes while its launch saving stays (2 db launches a layer -> 0 on the
# tile, 1 off it). Bits move against pass 2 only for batches of more than
# ag_fold_block(rows) = 64 rows (pass 2's blocked fold -> the column sum);
# every column and the host together. Other linear nodes keep pass 2's db.
# The column sum is one chain over the batch rows a column, run by a thread
# that also folds its own K = rows GEMM chain in the same tile, so the tile
# keeps the GEMM's shape of parallelism. -D MOJOLEARN_AUTOGRAD_PASS3_DB_OFF.
comptime AUTOGRAD_PASS3_DB = not is_defined["MOJOLEARN_AUTOGRAD_PASS3_DB_OFF"]()

# Lane ag-pass6-fold (2026-10-19, PASS6_IDEAS ideas D(2), F(2), H(1)): tape-
# side changes, identical (no word moves), each default on with an _OFF arm.
#   AUTOGRAD_LOSS_REDUCED (D(2)): losses.mse / mae / huber /
#     binary_cross_entropy_with_logits and loss_value_into fold the loss
#     terms inside the leaf chain (one node, no terms buffer): forward 3 -> 2
#     launches above 64 terms (1 at most 64), backward 2 -> 1. The terms are
#     ftz(loss_term(ld p, ld y)), the word a stored term reads back as; the
#     VJP's upstream factor mul(ld(g, 0), factor) is REDUCE_VJP's stored
#     word, kept in a register. -D MOJOLEARN_AUTOGRAD_LOSS_REDUCED_OFF.
#   AUTOGRAD_TRANSPOSE_VIEW (F(2)): tape.transpose records the node without
#     its copy; matmul reads the parent's buffer through GEMM strides (the
#     same words in the same k order); any other reader materializes the
#     copy once, when it first reads the value (Tape._use / materialize).
#     -D MOJOLEARN_AUTOGRAD_TRANSPOSE_VIEW_OFF.
#   AUTOGRAD_CURVE_WORD (H(1)): model_fit's per-batch curve word (the loss
#     times the batch rows) is stored by the loss's own fold launch
#     (mul(loss, B) on the loss's stored word, as tape.scale computes it)
#     instead of a scale launch and a copy launch: 2 launches a batch fewer.
#     -D MOJOLEARN_AUTOGRAD_CURVE_WORD_OFF.
comptime AUTOGRAD_TRANSPOSE_VIEW = not is_defined["MOJOLEARN_AUTOGRAD_TRANSPOSE_VIEW_OFF"]()
comptime AUTOGRAD_CURVE_WORD = not is_defined["MOJOLEARN_AUTOGRAD_CURVE_WORD_OFF"]()

# Node kinds recorded by the tape. Binary and unary kinds also travel as a
# runtime integer (a.i1 / a.i0) inside their operation's body.
comptime ADD = 1
comptime SUBTRACT = 2
comptime MULTIPLY = 3
comptime DIVIDE = 4
comptime RELU = 10
comptime SIGMOID = 11
comptime TANH = 12
comptime EXP = 13
comptime LOG = 14
comptime SQRT = 15
comptime SCALE = 16
comptime TRANSPOSE = 20
comptime RESHAPE = 21
comptime REDUCE = 22
comptime SOFTMAX = 23
comptime LOG_SOFTMAX = 24
# Loss nodes (autograd/losses.mojo): per-element terms of two same-shape
# inputs, and per-row softmax cross-entropy of logits against class codes.
# Their bodies are the loss family in autograd/loss_ops.mojo.
comptime LOSS_ELEM = 25
comptime LOSS_CE = 26
# Mean softmax cross-entropy over rows as one node (tape.softmax_cross_entropy):
# its backward writes (softmax - onehot) / rows into the logits' gradient.
comptime LOSS_SCE = 27
# Lane ag-pass6-fold (PASS6 idea D(2)): per-element loss terms and their
# mean / sum as ONE node (tape.loss_reduced, autograd/losses.mojo): the terms
# are computed inside the fold's leaf chain (autograd/fold_ops.mojo
# OP_AGS_LOSS_PART), never stored; the backward is one launch
# (OP_AGS_LOSS_VJP). Same words as LOSS_ELEM followed by REDUCE.
comptime LOSS_REDUCED = 28
comptime MATMUL = 30
# act(x @ W + b) as one node (tape.linear / linear_into): the forward and
# backward launches of the hand-assembled MLP, no new element bodies.
comptime LINEAR = 31
comptime CUSTOM = 40
comptime LEAF = 0

# Reduction axes (a.i0 of the reduce bodies).
comptime AXIS_ALL = -1
comptime AXIS_ROWS = 0
comptime AXIS_COLS = 1

# ---------------------------------------------------------------------------
# THE BLOCKED FOLD (OP_AG_REDUCE_PART / OP_AG_REDUCE_FOLD, OP_AGK_BIN_VJP_PART,
# OP_AGF_LEVEL; launched by autograd/fold.mojo). DeviceExec runs every body
# below on the generic element kernel (TPB = 128 threads a block, one thread
# an element t), on Metal, NVIDIA and AMD alike; HostExec runs the same
# bodies in an ascending loop, so every column takes one path.
#
# GEOMETRY BEFORE (v1, -D MOJOLEARN_AUTOGRAD_FOLD_V2_OFF; lane ag-pass3-fold
# read it from the code 2026-10-19). Block length R = ag_fold_block(n): the
# smallest power of two >= 64 with R * R >= n, n the folded extent.
#   AXIS_ROWS (rows x cols, one output a column): PART ceil(rows / R) * cols
#     threads, t -> (block t // cols, column t % cols): neighbouring threads
#     read neighbouring columns (coalesced), the chain walks the column at
#     stride cols, R terms. FOLD: cols threads, a chain of ceil(rows / R)
#     partials (coalesced across threads).
#   AXIS_COLS (one output a row): PART rows * ceil(cols / R) threads, t ->
#     (row t // nb, block t % nb): each chain is R contiguous words, so
#     neighbouring threads are R words apart (not coalesced). FOLD: rows
#     threads, nb contiguous partials each.
#   AXIS_ALL: PART ceil(n / R) threads, R contiguous words each; FOLD one
#     thread, ceil(n / R) partials.
#   Every chain is load, add, load, add: one dependent load latency a term.
#   The shapes the tape folds (PART threads x chain; FOLD threads x chain):
#     512 x 256 rows (MLP hidden bias): 2048 x 64; 256 x 8.
#     512 x 1 rows (MLP output bias):   8 x 64;    1 x 8.
#     4096 x 4096 rows:                 262144 x 64; 4096 x 64.
#     100000 x 1 rows, 1 x 100000 cols: R = 512: 196 x 512; 1 x 196.
#     4096 x 4096 all:                  R = 4096: 4096 x 4096; 1 x 4096.
#   The AMD MI325X profile (MLP 28-256-1, B = 512) put REDUCE_PART at
#   ~24 us a launch against ~14 us for an elementwise body over 65k words:
#   the excess is the 64-term dependent load chain, not the word count.
#
# GEOMETRY AFTER (v2, the default). Three changes, all fixed-order:
#   (1) Leaf blocks of a fixed AG_FOLD_LEAF = 64 terms (PART, BIN_VJP_PART),
#       then a tree of fan-in 64: while an output has more than 64 partials,
#       OP_AGF_LEVEL folds each run of 64 consecutive partials (ascending
#       from +0.0) in place into the run's first slot; the last level is
#       OP_AG_REDUCE_FOLD's one ascending chain. The order is a function of
#       the extent alone: blocks in index order, each folded serially in
#       index order, groups of 64 partials in group order. For n <= 4096
#       (one level) it is v1's order term for term (v1's R is 64 there), so
#       those folds keep their bits; above 4096 terms the bits are new, on
#       every column together (HostExec runs the same bodies). Chains never
#       exceed 64 terms, and the partial threads are n / 64 per output.
#   (2) Each chain issues its loads eight at a time ahead of the adds
#       (_fold_chain, _vjp_chain): the adds are the same adds in the same
#       order, so the same bits, but eight loads are in flight per thread
#       instead of one: a 64-term chain waits 8 load latencies, not 64.
#   (3) A fold of at most 64 terms an output is one launch: PART adds the
#       chain and applies the factor (the FOLD's +0.0 + s, then * or /
#       factor), so the same bits in one launch instead of two.
#   Thread mapping and coalescing per axis are v1's: AXIS_ROWS keeps
#   consecutive threads on consecutive columns (coalesced loads, the chain
#   at stride cols); AXIS_COLS and AXIS_ALL keep contiguous blocks and accept
#   the 64-word stride between neighbouring threads, because folding each
#   block serially in index order is what makes the order fixed (a strided
#   per-thread walk would interleave blocks and change the order).
#   The same shapes after (PART; LEVEL...; FOLD as threads x chain):
#     512 x 256 rows:   2048 x 64; 256 x 8                    (same bits)
#     512 x 1 rows:     8 x 64; 1 x 8                         (same bits)
#     4096 x 4096 rows: 262144 x 64; 4096 x 64                (same bits)
#     100000 x 1 rows / 1 x 100000 cols: 1563 x 64; 25 x 64; 1 x 25 (new)
#     4096 x 4096 all:  262144 x 64; 4096 x 64; 64 x 64; 1 x 64     (new)
#     any extent <= 64: one launch, outputs x extent        (same bits)
# ---------------------------------------------------------------------------
# A/B define (lane ag-pass3-fold, default on = v2): -D
# MOJOLEARN_AUTOGRAD_FOLD_V2_OFF restores v1 (the sqrt block rule, two
# launches, one load a term) for the orchestrator's digest comparison.
comptime AUTOGRAD_FOLD_V2 = not is_defined["MOJOLEARN_AUTOGRAD_FOLD_V2_OFF"]()
# AUTOGRAD_LOSS_REDUCED (lane ag-pass6-fold, described with the tape flags
# above) needs v2's leaf (ag_part_finish): a FOLD_V2_OFF build keeps
# loss_terms + mean / sum.
comptime AUTOGRAD_LOSS_REDUCED = AUTOGRAD_FOLD_V2 and not is_defined["MOJOLEARN_AUTOGRAD_LOSS_REDUCED_OFF"]()
# v2's leaf block length and tree fan-in. A fixed 64 (v1's smallest block):
# from size reasoning, not a shape: long enough that a leaf thread's chain
# amortises its launch slot, short enough that 8-wide load groups cover a
# chain in 8 latencies, and a fold of n terms needs ceil(log64 n) launches
# (2 up to 4096 terms, 3 up to 262144, 4 up to 16.7M).
comptime AG_FOLD_LEAF = 64
# Loads issued ahead of the adds in a fold chain (v2).
comptime AG_FOLD_LOADS = 8

# Lane ag-pass3-fold (2026-10-19): the fold tree's level body, a range of
# its own (1700..), registered in sequence/dispatch.mojo.
#   OP_AGF_LEVEL: one level of v2's fold tree, in place over the partials.
comptime OP_AGF_LEVEL = 1700
comptime OP_AGF_FIRST = OP_AGF_LEVEL
comptime OP_AGF_LAST = OP_AGF_LEVEL


@always_inline
def ag_fold_block(extent: Int) -> Int:
    """Elements per block of a blocked fold over `extent` terms: the smallest
    power of two R >= 64 with R * R >= extent (the ln_fold_rows rule,
    sequence/layernorm.mojo). It depends on the folded extent alone, never on
    a vendor or a benchmark shape: the block count and the block length both
    grow as sqrt(extent), so the partial launch has ~sqrt(extent) threads per
    output and the fold chain ~sqrt(extent) terms. An extent of at most 64
    is one block (one ascending chain)."""
    var r = 64
    while r * r < extent:
        r *= 2
    return r


@always_inline
def div(a: Float32, b: Float32) -> Float32:
    return ftz(identical_div(a, b))


@always_inline
def _index(row: Int, col: Int, rows: Int, cols: Int) -> Int:
    return (0 if rows == 1 else row) * cols + (0 if cols == 1 else col)


@always_inline
def _ld_right(a: Args, i: Int) -> Float32:
    # i10 = 1: the right operand is the left operand's buffer (x * x). The
    # tape then leaves p1 unbound instead of binding one buffer twice.
    # Load through a branch, never through a pointer select: on Metal
    # (Mojo 1.0.0 / MAX 26.5.0) `a.p0 if flag else a.p1` compiles to a
    # pointer whose loads read zeros with no error (orchestrator bisect
    # 2026-10-18, ~/mojolearn-evidence/native-autograd-20261018/
    # orch-metal-folds-01/control-run.log vs control-v2-run.log).
    if a.i10 == 1:
        return ld(a.p0, i)
    return ld(a.p1, i)


@always_inline
def _unary(op: Int, x: Float32, scale: Float32) -> Float32:
    if op == RELU:
        return x if x > 0 else Float32(0)
    if op == SIGMOID:
        return ftz(identical_sigmoid(x))
    if op == TANH:
        return ftz(identical_tanh(x))
    if op == EXP:
        return ftz(identical_exp(x))
    if op == LOG:
        return ftz(identical_log(x))
    if op == SQRT:
        return ftz(identical_sqrt(x))
    return mul(x, scale)


@always_inline
def unary_vjp_reads_input(op: Int) -> Bool:
    return op == RELU or op == LOG


@always_inline
def unary_vjp_reads_output(op: Int) -> Bool:
    return op == SIGMOID or op == TANH or op == EXP or op == SQRT


@always_inline
def _x_at(a: Args, t: Int) -> Float32:
    """The left operand at output element t (output shape i2 x i3, left
    shape i4 x i5). An operand of the output's shape is read at t with no
    index arithmetic (a uniform branch: the shapes are launch arguments)."""
    if a.i4 == a.i2 and a.i5 == a.i3:
        return ld(a.p0, t)
    return ld(a.p0, _index(t // a.i3, t % a.i3, a.i4, a.i5))


@always_inline
def _y_at(a: Args, t: Int) -> Float32:
    """The right operand at output element t (right shape i6 x i7), through
    _ld_right (p0 when i10 = 1)."""
    if a.i6 == a.i2 and a.i7 == a.i3:
        return _ld_right(a, t)
    return _ld_right(a, _index(t // a.i3, t % a.i3, a.i6, a.i7))


@always_inline
def _y1_at(a: Args, t: Int) -> Float32:
    """The right operand at output element t from p1 (DIVIDE's VJP binds p1)."""
    if a.i6 == a.i2 and a.i7 == a.i3:
        return ld(a.p1, t)
    return ld(a.p1, _index(t // a.i3, t % a.i3, a.i6, a.i7))


@always_inline
def op_ag_bin_fwd(t: Int, a: Args):
    """p0 x, p1 y (unbound when i10 = 1), p2 z. i1 kind, i2/i3 output
    shape, i4/i5 x shape, i6/i7 y shape. Thread t = output element t."""
    st(a.p2, t, _bin_value(a.i1, _x_at(a, t), _y_at(a, t)))


@always_inline
def _bin_value(op: Int, x: Float32, y: Float32) -> Float32:
    """op_ag_bin_fwd's value of kind `op` (ADD .. DIVIDE)."""
    if op == ADD:
        return add(x, y)
    if op == SUBTRACT:
        return sub(x, y)
    if op == MULTIPLY:
        return mul(x, y)
    return div(x, y)


@always_inline
def _bin_vjp_at(t: Int, a: Args) -> Float32:
    """The elementwise VJP for operand i8 at output element t: the value
    op_ag_bin_vjp stores and op_agk_bin_vjp_part folds."""
    var op = a.i1
    var g = ld(a.p3, t)
    if op == SUBTRACT and a.i8 == 1:
        g = -g
    elif op == MULTIPLY:
        if a.i8 == 0:
            g = mul(g, _y_at(a, t))
        else:
            g = mul(g, _x_at(a, t))
    elif op == DIVIDE:
        var y = _y1_at(a, t)
        if a.i8 == 0:
            g = div(g, y)
        else:
            g = -div(mul(g, ld(a.p2, t)), y)
    return g


@always_inline
def op_ag_bin_vjp(t: Int, a: Args):
    """The elementwise VJP at output element t, for operand i8, written at
    output index t of p4 (output shape i2 x i3). Slots read by kind:
    ADD/SUBTRACT p3 only; MULTIPLY p0 and p1 (the other operand); DIVIDE p1
    and, for operand 1, p2 (the forward output z). p0 x, p1 y (unbound when
    i10 = 1), p2 z, p3 upstream, p4 destination. A broadcast operand's
    gradient is op_agk_bin_vjp_part instead (tape.mojo _binary_vjp)."""
    st(a.p4, t, _bin_vjp_at(t, a))


@always_inline
def _fold_chain(p: FP, base: Int, stride: Int, count: Int) -> Float32:
    """p[base + k * stride] for k = 0 .. count - 1 added ascending from +0.0
    (v2 fold chain). The same adds in the same order as one load an add;
    each group of AG_FOLD_LOADS = 8 loads is issued before its adds so the
    loads are in flight together. count <= 0 gives +0.0."""
    var s = Float32(0)
    var k = 0
    while k + AG_FOLD_LOADS <= count:
        var i = base + k * stride
        var v0 = ld(p, i)
        var v1 = ld(p, i + stride)
        var v2 = ld(p, i + 2 * stride)
        var v3 = ld(p, i + 3 * stride)
        var v4 = ld(p, i + 4 * stride)
        var v5 = ld(p, i + 5 * stride)
        var v6 = ld(p, i + 6 * stride)
        var v7 = ld(p, i + 7 * stride)
        s = add(s, v0)
        s = add(s, v1)
        s = add(s, v2)
        s = add(s, v3)
        s = add(s, v4)
        s = add(s, v5)
        s = add(s, v6)
        s = add(s, v7)
        k += AG_FOLD_LOADS
    while k < count:
        s = add(s, ld(p, base + k * stride))
        k += 1
    return s


@always_inline
def _vjp_chain(a: Args, base: Int, stride: Int, count: Int) -> Float32:
    """_fold_chain over the binary VJP terms ftz(_bin_vjp_at(i)) at
    i = base + k * stride: each group of eight terms is computed (its loads
    issued) before its adds, which keep the ascending order."""
    var s = Float32(0)
    var k = 0
    while k + AG_FOLD_LOADS <= count:
        var i = base + k * stride
        var v0 = ftz(_bin_vjp_at(i, a))
        var v1 = ftz(_bin_vjp_at(i + stride, a))
        var v2 = ftz(_bin_vjp_at(i + 2 * stride, a))
        var v3 = ftz(_bin_vjp_at(i + 3 * stride, a))
        var v4 = ftz(_bin_vjp_at(i + 4 * stride, a))
        var v5 = ftz(_bin_vjp_at(i + 5 * stride, a))
        var v6 = ftz(_bin_vjp_at(i + 6 * stride, a))
        var v7 = ftz(_bin_vjp_at(i + 7 * stride, a))
        s = add(s, v0)
        s = add(s, v1)
        s = add(s, v2)
        s = add(s, v3)
        s = add(s, v4)
        s = add(s, v5)
        s = add(s, v6)
        s = add(s, v7)
        k += AG_FOLD_LOADS
    while k < count:
        s = add(s, ftz(_bin_vjp_at(base + k * stride, a)))
        k += 1
    return s


@always_inline
def _fold_part_index(axis: Int, o: Int, b: Int, cols: Int, nb: Int) -> Int:
    """Where leaf partial b of output o sits (op_ag_reduce_part's layout):
    AXIS_ROWS block-major b * cols + o, AXIS_COLS row-major o * nb + b,
    AXIS_ALL b. nb = leaf blocks per output."""
    if axis == AXIS_ROWS:
        return b * cols + o
    if axis == AXIS_COLS:
        return o * nb + b
    return b


@always_inline
def _agk_bin_vjp_part_v2(t: Int, a: Args):
    """v2 of op_agk_bin_vjp_part: the same thread mapping and partial layout,
    chains through _vjp_chain. i11 = 1 (one block an output, at most i9
    terms): the finished output, +0.0 + the chain (OP_AG_REDUCE_FOLD's one
    partial at factor 1), is written to p5[t] instead of a partial."""
    var rows = a.i2
    var cols = a.i3
    var B = a.i9
    var s: Float32
    if a.i0 == AXIS_ALL:
        var lo = t * B
        s = _vjp_chain(a, lo, 1, min(B, rows * cols - lo))
    elif a.i0 == AXIS_ROWS:
        var c = t % cols
        var lo = (t // cols) * B
        s = _vjp_chain(a, lo * cols + c, cols, min(B, rows - lo))
    else:
        var nb = max((cols + B - 1) // B, 1)
        var r = t // nb
        var lo = (t % nb) * B
        s = _vjp_chain(a, r * cols + lo, 1, min(B, cols - lo))
    if a.i11 == 1:
        st(a.p5, t, add(Float32(0), s))
    else:
        st(a.p4, t, s)


@always_inline
def op_agk_bin_vjp_part(t: Int, a: Args):
    """Partial sums of the blocked fold of a broadcast operand's binary VJP
    (see _agk_bin_vjp_part_v1; v2 by default, AUTOGRAD_FOLD_V2)."""
    comptime if AUTOGRAD_FOLD_V2:
        _agk_bin_vjp_part_v2(t, a)
    else:
        _agk_bin_vjp_part_v1(t, a)


@always_inline
def _agk_bin_vjp_part_v1(t: Int, a: Args):
    """Partial sums of the blocked fold of a broadcast operand's binary VJP:
    op_ag_reduce_part over the matrix op_ag_bin_vjp would write, with each
    term computed in place (ftz, as the stored word reads back) instead of
    loaded. op_ag_bin_vjp's slots (p0..p3, i1..i8, i10; the folded source
    is the i2 x i3 output), plus p4 partials, i0 axis, i9 block length B
    (ag_fold_block of the folded extent). Thread mapping and partial layout
    are op_ag_reduce_part's; the second launch is OP_AG_REDUCE_FOLD."""
    var rows = a.i2
    var cols = a.i3
    var B = a.i9
    var s = Float32(0)
    if a.i0 == AXIS_ALL:
        var lo = t * B
        var hi = min(lo + B, rows * cols)
        for k in range(lo, hi):
            s = add(s, ftz(_bin_vjp_at(k, a)))
    elif a.i0 == AXIS_ROWS:
        var c = t % cols
        var lo = (t // cols) * B
        var hi = min(lo + B, rows)
        for r in range(lo, hi):
            s = add(s, ftz(_bin_vjp_at(r * cols + c, a)))
    else:
        var nb = (cols + B - 1) // B
        var r = t // nb
        var lo = (t % nb) * B
        var hi = min(lo + B, cols)
        for k in range(lo, hi):
            s = add(s, ftz(_bin_vjp_at(r * cols + k, a)))
    st(a.p4, t, s)


@always_inline
def op_ag_unary_fwd(t: Int, a: Args):
    """p0 x, p1 y. i0 kind, f0 scale (SCALE). Thread t = element t."""
    st(a.p1, t, _unary(a.i0, ld(a.p0, t), a.f0))


@always_inline
def op_ag_unary_vjp(t: Int, a: Args):
    """p0 forward input (RELU, LOG only), p1 forward output (SIGMOID, TANH,
    EXP, SQRT only), p2 upstream, p3 destination. i0 kind, f0 scale."""
    var op = a.i0
    var g = ld(a.p2, t)
    var x = Float32(0)
    var y = Float32(0)
    # the forward input / output only where the kind reads it (the other
    # slot may be unbound): a branch, never a pointer select
    if unary_vjp_reads_input(op):
        x = ld(a.p0, t)
    if unary_vjp_reads_output(op):
        y = ld(a.p1, t)
    st(a.p3, t, _unary_vjp_value(op, g, x, y, a.f0))


@always_inline
def _unary_vjp_value(op: Int, g: Float32, x: Float32, y: Float32, scale: Float32) -> Float32:
    """op_ag_unary_vjp's value: g the upstream word, x the forward input
    (RELU, LOG), y the forward output (SIGMOID, TANH, EXP, SQRT)."""
    if op == RELU:
        return g if x > 0 else Float32(0)
    if op == SIGMOID:
        return mul(mul(g, y), sub(Float32(1), y))
    if op == TANH:
        return mul(g, sub(Float32(1), mul(y, y)))
    if op == EXP:
        return mul(g, y)
    if op == LOG:
        return div(g, x)
    if op == SQRT:
        return div(g, mul(Float32(2), y))
    return mul(g, scale)


@always_inline
def op_ag_transpose(t: Int, a: Args):
    """p0 source (i0 rows x i1 cols), p1 destination (i1 x i0). Forward and
    VJP share indexing. i2 selects a forward word copy; zero retains the
    established VJP FTZ behavior. Thread t = destination element t."""
    var source = (t % a.i0) * a.i1 + t // a.i0
    if a.i2 == 1:
        # Correctness repair (2026-10-20): layout materialization previously
        # used ld/st and flushed subnormals even though identity views kept
        # the original words. Forward movement now agrees with OP_COPY on
        # every executor. Arithmetic/reduction and VJP FTZ stay unchanged.
        a.p1.unsafe_store(t, a.p0.unsafe_load(source))
    else:
        st(a.p1, t, ld(a.p0, source))


@always_inline
def _ag_reduce_part_v2(t: Int, a: Args):
    """v2 of op_ag_reduce_part: the same slots, thread mapping and partial
    layout, chains through _fold_chain (i3 B = AG_FOLD_LEAF). i4 = 1 (one
    block an output): the finished output, (+0.0 + the chain) * f0, or / f0
    when i5 = 1, is written to p2[t] (OP_AG_REDUCE_FOLD's one-partial chain
    and factor, so its bits in one launch)."""
    var rows = a.i1
    var cols = a.i2
    var B = a.i3
    var s: Float32
    if a.i0 == AXIS_ALL:
        var lo = t * B
        s = _fold_chain(a.p0, lo, 1, min(B, rows * cols - lo))
    elif a.i0 == AXIS_ROWS:
        var c = t % cols
        var lo = (t // cols) * B
        s = _fold_chain(a.p0, lo * cols + c, cols, min(B, rows - lo))
    else:
        var nb = max((cols + B - 1) // B, 1)
        var r = t // nb
        var lo = (t % nb) * B
        s = _fold_chain(a.p0, r * cols + lo, 1, min(B, cols - lo))
    ag_part_finish(t, a, s)


@always_inline
def ag_part_finish(t: Int, a: Args, s: Float32):
    """The end of a leaf thread of OP_AG_REDUCE_PART (and of the loss leaf,
    autograd/fold_ops.mojo OP_AGS_LOSS_PART, which shares these slots): i4 =
    1 (one block an output) writes the finished output (+0.0 + s) * f0, or /
    f0 when i5 = 1, to p2[t]; else the partial s to p1[t]. i7 = 1 (lane
    ag-pass6-fold, AUTOGRAD_CURVE_WORD) also writes mul(output, f1) to p3[t]:
    the word tape.scale(output, f1) stores, from the same stored word."""
    if a.i4 == 1:
        var v = add(Float32(0), s)
        var out: Float32
        if a.i5 == 1:
            out = div(v, a.f0)
        else:
            out = mul(v, a.f0)
        st(a.p2, t, out)
        if a.i7 == 1:
            st(a.p3, t, mul(ftz(out), a.f1))
    else:
        st(a.p1, t, s)


@always_inline
def ag_leaf_span(t: Int, rows: Int, cols: Int, axis: Int, B: Int) -> Tuple[Int, Int]:
    """(first index, term count) of leaf t of a fold over AXIS_ALL or
    AXIS_COLS (v2's mapping in _ag_reduce_part_v2 / _agk_bin_vjp_part_v2):
    the leaf's terms are contiguous (stride 1). Lane ag-pass6-fold: the
    staged leaf kernel (autograd/fold_kernels.mojo) loads them through
    threadgroup memory with this span."""
    if axis == AXIS_ALL:
        var lo = t * B
        return (lo, min(B, rows * cols - lo))
    var nb = max((cols + B - 1) // B, 1)
    var r = t // nb
    var lo = (t % nb) * B
    return (r * cols + lo, min(B, cols - lo))


@always_inline
def op_ag_reduce_part(t: Int, a: Args):
    """The blocked fold's first launch (v2 by default, AUTOGRAD_FOLD_V2;
    _ag_reduce_part_v1 under MOJOLEARN_AUTOGRAD_FOLD_V2_OFF)."""
    comptime if AUTOGRAD_FOLD_V2:
        _ag_reduce_part_v2(t, a)
    else:
        _ag_reduce_part_v1(t, a)


@always_inline
def _ag_reduce_part_v1(t: Int, a: Args):
    """Partial sums of a blocked fold. p0 source (i1 rows x i2 cols), p1
    partials, i0 axis, i3 block length B (ag_fold_block of the folded
    extent). Each partial is its block's terms added ascending from +0.0.
      AXIS_ALL:  thread t = block t of the i1*i2 elements; p1[t].
      AXIS_ROWS: thread t = (block t // i2, column t % i2), folding rows;
                 p1[t] (block-major).
      AXIS_COLS: thread t = (row t // nb, block t % nb), folding columns;
                 p1[t] (row-major), nb = ceil(i2 / B)."""
    var rows = a.i1
    var cols = a.i2
    var B = a.i3
    var s = Float32(0)
    if a.i0 == AXIS_ALL:
        var lo = t * B
        var hi = min(lo + B, rows * cols)
        for k in range(lo, hi):
            s = add(s, ld(a.p0, k))
    elif a.i0 == AXIS_ROWS:
        var c = t % cols
        var lo = (t // cols) * B
        var hi = min(lo + B, rows)
        for r in range(lo, hi):
            s = add(s, ld(a.p0, r * cols + c))
    else:
        var nb = (cols + B - 1) // B
        var r = t // nb
        var lo = (t % nb) * B
        var hi = min(lo + B, cols)
        for k in range(lo, hi):
            s = add(s, ld(a.p0, r * cols + k))
    st(a.p1, t, s)


@always_inline
def _ag_reduce_fold_v2(t: Int, a: Args):
    """v2 of op_ag_reduce_fold, the last level of the fold tree: p1[t] = (the
    i6 live partials of output t, p0, at stride i5 in partial index, added
    ascending from +0.0) * f0, or / f0 when i4 = 1. i0 axis, i1 rows, i2
    cols, i3 leaf block length (AG_FOLD_LEAF). i5 = 0 (a launch that did not
    run OP_AGF_LEVEL): stride 1 over every leaf partial, v1's chain."""
    var rows = a.i1
    var cols = a.i2
    var B = a.i3
    var extent = rows * cols if a.i0 == AXIS_ALL else (rows if a.i0 == AXIS_ROWS else cols)
    var nb = (extent + B - 1) // B
    var S = a.i5
    var P = a.i6
    if S == 0:
        S = 1
        P = nb
    var base = _fold_part_index(a.i0, t, 0, cols, nb)
    var stride = S * cols if a.i0 == AXIS_ROWS else S
    var s = _fold_chain(a.p0, base, stride, P)
    _fold_finish(t, a, s)


@always_inline
def _fold_finish(t: Int, a: Args, s: Float32):
    """OP_AG_REDUCE_FOLD's store: p1[t] = s * f0, or s / f0 when i4 = 1;
    i7 = 1 (AUTOGRAD_CURVE_WORD) also p3[t] = mul(p1[t], f1), the word
    tape.scale stores from the stored output."""
    var out: Float32
    if a.i4 == 1:
        out = div(s, a.f0)
    else:
        out = mul(s, a.f0)
    st(a.p1, t, out)
    if a.i7 == 1:
        st(a.p3, t, mul(ftz(out), a.f1))


@always_inline
def op_agf_level(t: Int, a: Args):
    """One level of v2's fold tree, in place. p0 partials (op_ag_reduce_part's
    layout; read and written: thread t reads only its own group and writes
    only the group's first slot, so no two threads touch one word). i0 axis,
    i1 rows, i2 cols, i3 L (AG_FOLD_LEAF, the leaf length and the fan-in),
    i5 S (stride of the live partials in partial index: L ** level), i6 P
    (live partials an output). Thread t = (group j, output o), G = ceil(P /
    L) groups: AXIS_ROWS j = t // cols, o = t % cols (neighbouring threads on
    neighbouring columns); AXIS_COLS o = t // G, j = t % G; AXIS_ALL j = t.
    Group j's live partials j * L .. min(j * L + L, P) - 1 are added
    ascending from +0.0 into live partial j * L."""
    var rows = a.i1
    var cols = a.i2
    var L = a.i3
    var S = a.i5
    var P = a.i6
    var extent = rows * cols if a.i0 == AXIS_ALL else (rows if a.i0 == AXIS_ROWS else cols)
    var nb = (extent + L - 1) // L
    var groups = (P + L - 1) // L
    var o: Int
    var j: Int
    if a.i0 == AXIS_ROWS:
        j = t // cols
        o = t % cols
    elif a.i0 == AXIS_COLS:
        o = t // groups
        j = t % groups
    else:
        o = 0
        j = t
    var lo = j * L
    var base = _fold_part_index(a.i0, o, lo * S, cols, nb)
    var stride = S * cols if a.i0 == AXIS_ROWS else S
    st(a.p0, base, _fold_chain(a.p0, base, stride, min(L, P - lo)))


@always_inline
def apply_autograd_fold[OP: Int](t: Int, a: Args):
    """The OP_AGF_FIRST..OP_AGF_LAST family (sequence/dispatch.mojo)."""
    comptime if OP == OP_AGF_LEVEL:
        op_agf_level(t, a)


@always_inline
def op_ag_reduce_fold(t: Int, a: Args):
    """The blocked fold's last launch (v2 by default, AUTOGRAD_FOLD_V2;
    _ag_reduce_fold_v1 under MOJOLEARN_AUTOGRAD_FOLD_V2_OFF)."""
    comptime if AUTOGRAD_FOLD_V2:
        _ag_reduce_fold_v2(t, a)
    else:
        _ag_reduce_fold_v1(t, a)


@always_inline
def _ag_reduce_fold_v1(t: Int, a: Args):
    """Second launch of the blocked fold: p1[t] = (the partials of output t,
    p0, added ascending from +0.0) * f0, or / f0 when i4 = 1. i0 axis, i1
    rows, i2 cols, i3 B as in op_ag_reduce_part. AXIS_ALL: one output; AXIS_ROWS: thread t =
    column t; AXIS_COLS: thread t = row t."""
    var rows = a.i1
    var cols = a.i2
    var B = a.i3
    var s = Float32(0)
    if a.i0 == AXIS_ALL:
        var nb = (rows * cols + B - 1) // B
        for k in range(nb):
            s = add(s, ld(a.p0, k))
    elif a.i0 == AXIS_ROWS:
        var nb = (rows + B - 1) // B
        for k in range(nb):
            s = add(s, ld(a.p0, k * cols + t))
    else:
        var nb = (cols + B - 1) // B
        for k in range(nb):
            s = add(s, ld(a.p0, t * nb + k))
    # i4 = 1: the fused linear node's bias gradient divided by its divisor,
    # as op_colsum_div divides its column sum (AUTOGRAD_PASS2_FOLD)
    _fold_finish(t, a, s)


@always_inline
def op_ag_reduce_vjp(t: Int, a: Args):
    """p0 upstream (reduced shape), p1 destination (input shape, i2 cols).
    i0 axis, f0 factor. Thread t = input element t."""
    var i = 0
    if a.i0 == AXIS_ROWS:
        i = t % a.i2
    elif a.i0 == AXIS_COLS:
        i = t // a.i2
    st(a.p1, t, mul(ld(a.p0, i), a.f0))


@always_inline
def op_ag_softmax_fwd(t: Int, a: Args):
    """p0 x, p1 y, i0 = 1 for log-softmax, i1 cols. Thread t = row t: the
    row max, the exponential sum and the normalisation in one launch.
    Softmax parks each exponential in p1 and reads it back for the
    division (its own row); p1 must not overlap p0."""
    var cols = a.i1
    var base = t * cols
    var maximum = ld(a.p0, base)
    for k in range(1, cols):
        maximum = max(maximum, ld(a.p0, base + k))
    var total = Float32(0)
    if a.i0 == 0:
        # each exponential once (lane ag-pass2-kernels): parked in the
        # output row, read back for the division (this thread's own words;
        # p1 never overlaps p0). The same words as computing it twice.
        for k in range(cols):
            var e = ftz(identical_exp(sub(ld(a.p0, base + k), maximum)))
            total = add(total, e)
            st(a.p1, base + k, e)
        for k in range(cols):
            st(a.p1, base + k, div(ld(a.p1, base + k), total))
    else:
        for k in range(cols):
            total = add(total, ftz(identical_exp(sub(ld(a.p0, base + k), maximum))))
        var log_total = ftz(identical_log(total))
        for k in range(cols):
            st(a.p1, base + k, sub(sub(ld(a.p0, base + k), maximum), log_total))


@always_inline
def op_ag_softmax_vjp(t: Int, a: Args):
    """p0 forward output y, p1 upstream, p2 destination, i0 = 1 for
    log-softmax, i1 cols. Thread t = row t: the row's dot product is
    computed once, O(rows * cols)."""
    var cols = a.i1
    var base = t * cols
    var total = Float32(0)
    for k in range(cols):
        var g = ld(a.p1, base + k)
        if a.i0 == 0:
            g = mul(g, ld(a.p0, base + k))
        total = add(total, g)
    for k in range(cols):
        var i = base + k
        if a.i0 == 0:
            st(a.p2, i, mul(ld(a.p0, i), sub(ld(a.p1, i), total)))
        else:
            st(a.p2, i, sub(ld(a.p1, i), mul(ftz(identical_exp(ld(a.p0, i))), total)))


@always_inline
def op_ag_accumulate(t: Int, a: Args):
    """p1[t] += p0[t]. p0 contribution, p1 gradient."""
    st(a.p1, t, add(ld(a.p1, t), ld(a.p0, t)))


# ---------------------------------------------------------------------------
# FOUR ELEMENTS A THREAD (lane ag-pass6-fold, 2026-10-19, PASS6_IDEAS idea E;
# identical). The elementwise bodies above take one float a thread with
# scalar loads and stores: the L40S streams them at 88-93% of its bandwidth,
# the MI325X at 32-48% (opbench add same 65536x4096 4021 / 1132 us, 801 /
# 2845 GB/s), because a one-word-a-lane stream saturates the MI300 family
# near 2.8 TB/s and 16-byte accesses are what reach 4.5-5 TB/s. The bodies
# below run the SAME per-element operations on the same words, four
# consecutive elements a thread, with 16-byte loads of every operand read at
# the output index and a 16-byte store: each lane of the vector is flushed
# exactly as ld / st flush a word, and each output is the scalar body's
# expression on those words, so every output word is the scalar body's.
# Operands at another shape (a broadcast row, column or scalar) keep their
# scalar loads. The device executor launches them (autograd/fold_kernels.mojo
# ag_vec4_kernel) only where every bound buffer is 16-byte aligned and the
# launch has at least 4 x TPB elements; the last n % 4 elements run the
# scalar body. HostExec never calls them.
# ---------------------------------------------------------------------------
comptime AG_VEC = 4
comptime F4 = SIMD[DType.float32, AG_VEC]


@always_inline
def ld4(p: FP, i: Int) -> F4:
    """ld of the four words p[i .. i + 3] (i a multiple of 4, p 16-byte
    aligned): one 16-byte load, each lane flushed as ld flushes."""
    var v = p.unsafe_load[width=AG_VEC, alignment=16](i)
    var out = F4(0)
    comptime for k in range(AG_VEC):
        out[k] = ftz(v[k])
    return out


@always_inline
def st4(p: FP, i: Int, v: F4):
    """st of four words at p[i .. i + 3]: each lane flushed as st flushes,
    one 16-byte store."""
    var w = F4(0)
    comptime for k in range(AG_VEC):
        w[k] = ftz(v[k])
    p.unsafe_store[alignment=16](i, w)


@always_inline
def _x4(a: Args, t0: Int) -> F4:
    """_x_at for outputs t0 .. t0 + 3."""
    if a.i4 == a.i2 and a.i5 == a.i3:
        return ld4(a.p0, t0)
    var v = F4(0)
    comptime for k in range(AG_VEC):
        v[k] = _x_at(a, t0 + k)
    return v


@always_inline
def _y4(a: Args, t0: Int) -> F4:
    """_y_at for outputs t0 .. t0 + 3 (p0 when i10 = 1: a branch, never a
    pointer select)."""
    if a.i6 == a.i2 and a.i7 == a.i3:
        if a.i10 == 1:
            return ld4(a.p0, t0)
        return ld4(a.p1, t0)
    var v = F4(0)
    comptime for k in range(AG_VEC):
        v[k] = _y_at(a, t0 + k)
    return v


@always_inline
def _y14(a: Args, t0: Int) -> F4:
    """_y1_at for outputs t0 .. t0 + 3."""
    if a.i6 == a.i2 and a.i7 == a.i3:
        return ld4(a.p1, t0)
    var v = F4(0)
    comptime for k in range(AG_VEC):
        v[k] = _y1_at(a, t0 + k)
    return v


@always_inline
def op_ag_bin_fwd4(t0: Int, a: Args):
    """op_ag_bin_fwd for outputs t0 .. t0 + 3."""
    var x = _x4(a, t0)
    var y = _y4(a, t0)
    var z = F4(0)
    comptime for k in range(AG_VEC):
        z[k] = _bin_value(a.i1, x[k], y[k])
    st4(a.p2, t0, z)


@always_inline
def op_ag_bin_vjp4(t0: Int, a: Args):
    """op_ag_bin_vjp (_bin_vjp_at) for outputs t0 .. t0 + 3."""
    var op = a.i1
    var g = ld4(a.p3, t0)
    var r = g
    if op == SUBTRACT and a.i8 == 1:
        comptime for k in range(AG_VEC):
            r[k] = -g[k]
    elif op == MULTIPLY:
        var o: F4
        if a.i8 == 0:
            o = _y4(a, t0)
        else:
            o = _x4(a, t0)
        comptime for k in range(AG_VEC):
            r[k] = mul(g[k], o[k])
    elif op == DIVIDE:
        var y = _y14(a, t0)
        if a.i8 == 0:
            comptime for k in range(AG_VEC):
                r[k] = div(g[k], y[k])
        else:
            var z = ld4(a.p2, t0)
            comptime for k in range(AG_VEC):
                r[k] = -div(mul(g[k], z[k]), y[k])
    st4(a.p4, t0, r)


@always_inline
def op_ag_unary_fwd4(t0: Int, a: Args):
    """op_ag_unary_fwd for elements t0 .. t0 + 3."""
    var x = ld4(a.p0, t0)
    var y = F4(0)
    comptime for k in range(AG_VEC):
        y[k] = _unary(a.i0, x[k], a.f0)
    st4(a.p1, t0, y)


@always_inline
def op_ag_unary_vjp4(t0: Int, a: Args):
    """op_ag_unary_vjp for elements t0 .. t0 + 3 (the forward input / output
    loaded only where the kind reads it, as the scalar body)."""
    var op = a.i0
    var g = ld4(a.p2, t0)
    var x = F4(0)
    var y = F4(0)
    if unary_vjp_reads_input(op):
        x = ld4(a.p0, t0)
    if unary_vjp_reads_output(op):
        y = ld4(a.p1, t0)
    var r = F4(0)
    comptime for k in range(AG_VEC):
        r[k] = _unary_vjp_value(op, g[k], x[k], y[k], a.f0)
    st4(a.p3, t0, r)


@always_inline
def op_ag_reduce_vjp4(t0: Int, a: Args):
    """op_ag_reduce_vjp for input elements t0 .. t0 + 3 (the upstream word
    of each by the scalar body's index; one 16-byte store)."""
    var r = F4(0)
    comptime for k in range(AG_VEC):
        var t = t0 + k
        var i = 0
        if a.i0 == AXIS_ROWS:
            i = t % a.i2
        elif a.i0 == AXIS_COLS:
            i = t // a.i2
        r[k] = mul(ld(a.p0, i), a.f0)
    st4(a.p1, t0, r)


@always_inline
def op_ag_accumulate4(t0: Int, a: Args):
    """op_ag_accumulate for elements t0 .. t0 + 3 (its own destination words
    read, then written, by this thread only)."""
    var d = ld4(a.p1, t0)
    var c = ld4(a.p0, t0)
    var r = F4(0)
    comptime for k in range(AG_VEC):
        r[k] = add(d[k], c[k])
    st4(a.p1, t0, r)


@always_inline
def ag_vec4_op[OP: Int]() -> Bool:
    """The OP_AG_* codes with a four-element body."""
    return (OP == OP_AG_BIN_FWD or OP == OP_AG_BIN_VJP or OP == OP_AG_UNARY_FWD or OP == OP_AG_UNARY_VJP
            or OP == OP_AG_REDUCE_VJP or OP == OP_AG_ACCUMULATE)


@always_inline
def apply_autograd_vec4[OP: Int](t0: Int, a: Args):
    """Outputs t0 .. t0 + 3 of OP (ag_vec4_op[OP]()), the scalar body's words."""
    comptime if OP == OP_AG_BIN_FWD:
        op_ag_bin_fwd4(t0, a)
    elif OP == OP_AG_BIN_VJP:
        op_ag_bin_vjp4(t0, a)
    elif OP == OP_AG_UNARY_FWD:
        op_ag_unary_fwd4(t0, a)
    elif OP == OP_AG_UNARY_VJP:
        op_ag_unary_vjp4(t0, a)
    elif OP == OP_AG_REDUCE_VJP:
        op_ag_reduce_vjp4(t0, a)
    elif OP == OP_AG_ACCUMULATE:
        op_ag_accumulate4(t0, a)


@always_inline
def apply_autograd[OP: Int](t: Int, a: Args):
    comptime if OP == OP_AG_BIN_FWD:
        op_ag_bin_fwd(t, a)
    elif OP == OP_AG_BIN_VJP:
        op_ag_bin_vjp(t, a)
    elif OP == OP_AG_UNARY_FWD:
        op_ag_unary_fwd(t, a)
    elif OP == OP_AG_UNARY_VJP:
        op_ag_unary_vjp(t, a)
    elif OP == OP_AG_TRANSPOSE:
        op_ag_transpose(t, a)
    elif OP == OP_AG_REDUCE_PART:
        op_ag_reduce_part(t, a)
    elif OP == OP_AG_REDUCE_FOLD:
        op_ag_reduce_fold(t, a)
    elif OP == OP_AG_REDUCE_VJP:
        op_ag_reduce_vjp(t, a)
    elif OP == OP_AG_SOFTMAX_FWD:
        op_ag_softmax_fwd(t, a)
    elif OP == OP_AG_SOFTMAX_VJP:
        op_ag_softmax_vjp(t, a)
    elif OP == OP_AG_ACCUMULATE:
        op_ag_accumulate(t, a)


@always_inline
def apply_autograd_kernels[OP: Int](t: Int, a: Args):
    """The OP_AGK_FIRST..OP_AGK_LAST family except OP_AGK_GEMM_EPI, whose
    body (op_gemm_epi) sequence/dispatch.mojo calls itself."""
    comptime if OP == OP_AGK_BIN_VJP_PART:
        op_agk_bin_vjp_part(t, a)
