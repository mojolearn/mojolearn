# SPDX-License-Identifier: Apache-2.0
from sequence.exec_trait import Exec
from sequence.ops import FP, Args, dummy_ptr, OP_COPY, OP_FILL, OP_GEMM
from sequence.ops import OP_ACT, OP_ACT_BWD, OP_COLSUM, OP_COLSUM_DIV, OP_GEMM_EPI, OP_SOFTMAX
from sequence.mlp import (
    ACT_IDENTITY, ACT_LOGISTIC, ACT_TANH, ACT_RELU, ACT_SOFTMAX,
    EPI_BIAS_ACT, EPI_L2GRAD, EPI_ACT_BWD,
)
from checks.numerics import identical_div
from autograd.fold import blocked_fold, blocked_fold_finish, fold_block, fold_partials, fold_single_launch, loss_fold
from autograd.fold_ops import OP_AGS_LOSS_VJP
from autograd.profile import ag_profile_report
from sequence.launch_profile import (
    launch_profile_level, launch_profile_mark, PROFILE_FORWARD, PROFILE_BACKWARD, PROFILE_OTHER,
    PROFILE_KIND_ALL,
)
from autograd.ops import (
    OP_AG_BIN_FWD, OP_AG_BIN_VJP, OP_AG_UNARY_FWD, OP_AG_UNARY_VJP, OP_AG_TRANSPOSE,
    OP_AG_REDUCE_PART, OP_AG_REDUCE_FOLD, OP_AG_REDUCE_VJP, OP_AG_SOFTMAX_FWD,
    OP_AG_SOFTMAX_VJP, OP_AG_ACCUMULATE,
    LEAF, ADD, SUBTRACT, MULTIPLY,
    DIVIDE, RELU, SIGMOID, TANH, EXP, LOG, SQRT, SCALE, TRANSPOSE, RESHAPE,
    REDUCE, SOFTMAX, LOG_SOFTMAX, MATMUL, LINEAR, CUSTOM, AXIS_ALL, AXIS_ROWS, AXIS_COLS,
    ag_fold_block, unary_vjp_reads_input, unary_vjp_reads_output, LOSS_ELEM, LOSS_CE, LOSS_SCE,
    OP_AGK_BIN_VJP_PART, OP_AGK_GEMM_EPI, AUTOGRAD_PASS2_FOLD, AUTOGRAD_PASS3_DB,
    LOSS_REDUCED, AUTOGRAD_LOSS_REDUCED, AUTOGRAD_TRANSPOSE_VIEW, AUTOGRAD_CURVE_WORD,
)
from autograd.loss_ops import (
    OP_LS_ELEM_FWD, OP_LS_ELEM_VJP, OP_LS_CE_FWD, OP_LS_CE_VJP, LS_MSE, LS_MAE, LS_HUBER,
    LS_BCE_LOGITS,
)
from autograd.data_ops import OP_AG_DROPOUT_FWD, OP_AG_DROPOUT_VJP, DROPOUT, dropout_args
from std.memory import memcmp
from autograd.pass1_ops import OP_P1_SCE_VJP
from autograd.composed_ops import OP_AGC_LINEAR_PAIR
from sequence.gemm_tiled import GR_TILE, seq_gemm_route
from autograd.embedding_ops import (
    EMBEDDING, AG_EMB_PLAN_AUTO, AG_EMB_PLAN_SORT, embedding_refuse_shape, embedding_check_words,
    embedding_refuse_ids, embedding_forward_into, embedding_plan, embedding_perm_words, embedding_backward_into,
)
from std.sys.compile import is_defined
from std.memory import bitcast
# lane ag-rankn-core: rank-N shapes and views (docs/plans/AUTOGRAD_RANKN.md)
from autograd.rankn import (
    ShapeN, RN_MAX_RANK, PERMUTE_N, RN_TAG_FIRST, RN_TAG_END, RN_VIEW_MARK, rn_check, rn_collapse,
    rn_desc_view, rn_normalize_axis,
)
from autograd.rankn_ops import rn_copy_strided

comptime _INFERRED_REFUSAL = "tensor was computed in inference mode (no_grad) and has no graph: call record() and run the forward again to differentiate it"
comptime _STALE_REFUSAL = "output depends on a tensor marked mutated after it was used: record the graph again"

# Lane ag-pass5-composed (2026-10-19, PASS5_IDEAS idea 5), default on:
# PASS-THROUGH GRADIENTS ARE BORROWED, NOT COPIED. backward_from's seed
# becomes the root's gradient buffer (no OP_COPY) when the root is a
# tape-owned, non-leaf, non-linear, non-custom node, and the first
# contribution of a same-shape add (either operand) or a subtraction's left
# operand to a once-consumed, tape-owned, non-leaf, non-linear, non-custom
# operand is the add's own gradient buffer (no copy VJP), as the reshape's
# already is. Bits: identical by construction: OP_COPY copies bits, and the
# copy VJP stores ftz(g) where every consumer loads through ld (ftz), so
# every kernel reads the same word; only a host download of an
# intermediate's gradient can differ, where the caller's seed holds a
# subnormal (it then shows the seed's word). Invariant: a borrowed buffer is
# never written during the backward that borrows it (one consumer, no
# in-place backward: linear and custom operands never borrow), and each
# backward first returns every node to its own buffer (_Node.own_grad), so
# a later backward never writes the caller's old seed. gradient() of a
# borrowing node is the buffer it borrowed (for the root, the seed's
# storage). Launches: the composed MLP's seed copy and one copy VJP a layer.
# -D MOJOLEARN_AUTOGRAD_GRAD_ALIAS_OFF keeps the copies (arm A).
comptime AUTOGRAD_GRAD_ALIAS = not is_defined["MOJOLEARN_AUTOGRAD_GRAD_ALIAS_OFF"]()

# Lane ag-pass5-composed (2026-10-19, PASS5_IDEAS idea 4), default on: ONE
# LAUNCH FOR A LINEAR (OR MATMUL) NODE'S BACKWARD GEMMS. When both its
# input and its weight want a gradient and neither product takes the
# 16 x 16 tile (sequence/gemm_tiled.mojo seq_gemm_route: a thin edge or a
# short K, where every output word is one thread's chain with no operand
# reuse), dX, dW and (with AUTOGRAD_PASS3_DB) db run as three index ranges
# of one OP_AGC_LINEAR_PAIR launch (autograd/composed_ops.mojo), each the
# body its own launch runs one thread a cell. A tiled product keeps its own
# launch: the tile's staged reuse costs less than the launch floor saved.
# A few-cell, long-K weight product (the coop GEMM's shape in
# sequence/exec_device.mojo: at most 1024 cells, K >= 32768) is not paired:
# one thread a cell would serialize its K. Bits: identical by construction
# (the same chain per word as the tiled, thin and cell routes, the same
# epilogues; sequence/gemm_tiled.mojo states that contract). Launches: the
# fused MLP's dH + dW2 (+ db2) 2 -> 1 a batch; the composed MLP's matmul
# dX + dW 2 -> 1 a layer above the first. -D MOJOLEARN_AUTOGRAD_PAIR_OFF
# keeps the separate launches (arm A).
comptime AUTOGRAD_PAIR = not is_defined["MOJOLEARN_AUTOGRAD_PAIR_OFF"]()


def _ranges_overlap(a: Int, a_count: Int, b: Int, b_count: Int) -> Bool:
    """Whether the float32 ranges [a, a + a_count) and [b, b + b_count)
    (byte addresses, element counts) share a byte."""
    return a < b + b_count * 4 and b < a + a_count * 4


@fieldwise_init
struct Tensor(ImplicitlyCopyable, Movable):
    """A matrix or scalar recorded by a Tape; scalars have shape (1, 1)."""
    var _owner: Int
    var _generation: Int
    var _index: Int


@fieldwise_init
struct IdsHandle(ImplicitlyCopyable, Movable):
    """Integer row ids recorded by a Tape (Tape.ids): a non-differentiable
    input of Tape.embedding. Not a Tensor: value() and gradient() never
    take it."""
    var _owner: Int
    var _generation: Int
    var _index: Int


@fieldwise_init
struct _Ids(ImplicitlyCopyable, Movable):
    # n int32 ids, their bit patterns in n executor words (caller storage)
    var data: FP
    var n: Int
    # every id is known to lie in [0, checked) (0: not yet checked)
    var checked: Int


@fieldwise_init
struct _Node(ImplicitlyCopyable, Movable):
    var value: FP
    var grad: FP
    var rows: Int
    var cols: Int
    var requires_grad: Bool
    var op: Int
    var start: Int
    var count: Int
    var saved: Args
    # The gradient buffer holds a value: written by a backward (or zeroed for
    # an unreached leaf), or caller storage from variable_into.
    var grad_ready: Bool
    # Produced in inference mode (no_grad): a value with no parents, no
    # gradient storage and no backward.
    var inferred: Bool
    # A LINEAR node's gradient buffer holds dL/d(pre-activation), not
    # dL/d(output): its activation derivative was already applied, by its
    # one consumer's dX epilogue or in place by its own backward.
    var grad_preact: Bool
    # The gradient buffer is caller storage (variable_into, custom_into,
    # binary_recorded with caller_grad): never re-pointed by a backward.
    var caller_grad: Bool
    # -1, or the first node of the alias group whose one gradient buffer
    # this node shares (custom_into_shared: a pass-through edge, the node's
    # gradient IS that input's gradient). Every member holds the same range.
    var alias_of: Int
    # The gradient buffer given at record time. `grad` differs from it only
    # while the node borrows a pass-through buffer (a reshape's or an add's
    # gradient, or backward_from's seed: AUTOGRAD_GRAD_ALIAS); each backward
    # starts by returning to it.
    var own_grad: FP


trait BackwardRules:
    # Native extensions receive one distinct destination per input, zeroed
    # unless zeroed_destination says the rule needs no zeros there. They
    # write VJPs there; the tape owns graph traversal and accumulation.
    # Return False for unknown tags so a backward refuses BEFORE touching grads.
    def supports(self, tag: Int) -> Bool:
        ...

    # True: zero-fill destination `edge` before apply (the safe default).
    # False: the rule overwrites every element of that destination, or never
    # touches it and the input needs no gradient, so the fill is skipped.
    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        ...

    # True: destination `edge` is the input's gradient buffer itself, never
    # scratch, and the tape adds nothing to it afterwards. The rule writes
    # that input's whole gradient there (or, for a graph whose rules share
    # the join, leaves it to the rule that does), including whatever other
    # consumers contributed. It is never zero-filled. A driver that records
    # a fixed graph uses it to run a fan-in or residual join inside the
    # kernel that already reads the operands, as a hand-written backward
    # does. False everywhere is the plain contract above.
    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        ...

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        ...


@fieldwise_init
struct NoCustomRules(BackwardRules):
    def supports(self, tag: Int) -> Bool:
        return False

    def zeroed_destination(self, tag: Int, saved: Args, edge: Int) -> Bool:
        return True

    def writes_total(self, tag: Int, saved: Args, edge: Int) -> Bool:
        return False

    def apply[E: Exec](
        self, mut ex: E, tag: Int, saved: Args, output: FP, upstream: FP,
        inputs: List[FP], gradients: List[FP], rows: List[Int], cols: List[Int],
    ) raises:
        raise Error("custom backward rule not registered")


def _fill[E: Exec](mut ex: E, p: FP, n: Int, value: Float32 = 0) raises:
    var a = Args()
    a.p0 = p
    a.f0 = value
    ex.launch[OP_FILL](a, n)


def _copy[E: Exec](mut ex: E, dst: FP, src: FP, n: Int) raises:
    var a = Args()
    a.p0 = src
    a.p1 = dst
    ex.launch[OP_COPY](a, n)


def _accumulate[E: Exec](mut ex: E, dst: FP, src: FP, n: Int) raises:
    var a = Args()
    a.p0 = src
    a.p1 = dst
    ex.launch[OP_AG_ACCUMULATE](a, n)


def _transpose[E: Exec](mut ex: E, dst: FP, src: FP, src_rows: Int, src_cols: Int) raises:
    var a = Args()
    a.p0 = src
    a.p1 = dst
    a.i0 = src_rows
    a.i1 = src_cols
    ex.launch[OP_AG_TRANSPOSE](a, src_rows * src_cols)


def _gemm_epi[E: Exec](
    mut ex: E, A: FP, Bm: FP, C: FP, M: Int, N: Int, K: Int,
    sam: Int, sak: Int, sbk: Int, sbn: Int, epi: Int, p3: FP, i10: Int, f0: Float32, f1: Float32,
    colsum_dst: FP = dummy_ptr(), colsum: Bool = False,
) raises:
    """A GEMM and its elementwise follower in one launch, with the argument
    layout of the MLP trainer's gemm_epi (sequence/mlp_fit.mojo). It
    launches OP_AGK_GEMM_EPI, op_gemm_epi's body, which DeviceExec runs on
    the tiled GEMM where OP_GEMM is tiled (lane ag-pass2-kernels): the hand
    path's words, the tiled kernel's loads. `colsum` (lane ag-pass3-impl,
    B dense) also writes colsum_dst[n] = column n of B summed / f1, the
    hand path's OP_COLSUM_DIV word (autograd/ops.mojo agk_colsum_args)."""
    var a = Args()
    a.p0 = A
    a.p1 = Bm
    a.p2 = C
    a.p3 = p3
    if colsum:
        if sbn != 1 or sbk != N:
            raise Error("autograd: the GEMM epilogue's column sums need a dense B")
        a.p4 = colsum_dst
        a.i11 = 1
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
    ex.launch[OP_AGK_GEMM_EPI](a, M * N)


def _linear_act(activation: Int) raises -> Int:
    """The MLP activation code of a linear node's activation."""
    if activation == 0:
        return ACT_IDENTITY
    if activation == SIGMOID:
        return ACT_LOGISTIC
    if activation == TANH:
        return ACT_TANH
    if activation == RELU:
        return ACT_RELU
    if activation == SOFTMAX:
        return ACT_SOFTMAX
    raise Error("linear takes activation 0 (none), RELU, SIGMOID, TANH or SOFTMAX")


def _gemm[E: Exec](
    mut ex: E, dst: FP, a: FP, b: FP, m: Int, n: Int, k: Int,
    ar: Int, ac: Int, br: Int, bc: Int,
) raises:
    var args = Args()
    args.p0 = a
    args.p1 = b
    args.p2 = dst
    args.i0 = m
    args.i1 = n
    args.i2 = k
    args.i3 = ar
    args.i4 = ac
    args.i5 = br
    args.i6 = bc
    args.i8 = n
    ex.launch[OP_GEMM](args, m * n)


@always_inline
def _passes_upstream(n: _Node, edge: Int, p: _Node) -> Bool:
    """True when binary node n's VJP into operand `edge` (node p) is n's
    upstream gradient itself: either operand of an add, or the left operand
    of a subtraction, at the output's shape (no broadcast)."""
    return (n.op == ADD or (n.op == SUBTRACT and edge == 0)) and p.rows == n.rows and p.cols == n.cols


struct Tape(Movable):
    """Record native FP32 tensor operations and compute first derivatives.

    Use the same executor for every operation on a tape. It owns tensor
    storage and must outlive the tape. reset() invalidates tensor handles
    and reuses storage for the next graph; release() also returns that
    storage to the executor. Operations are out of place.

    Example::

        var ex = HostExec()
        var tape = Tape()
        var x = tape.scalar(ex, 2.0, True)
        var loss = tape.multiply(ex, x, x)
        tape.backward(ex, loss)
        var dx = tape.gradients(ex, x)

    Inference: after no_grad() every operation computes its value exactly
    as when recording (the same kernels, so the same bits) but keeps no
    graph and no gradient storage; record() switches recording back on.
    Tensors made in inference mode are constants: backward from one, or
    gradient() of one, raises. Example::

        tape.no_grad()
        var y = tape.relu(ex, tape.matmul(ex, x, w))
        var prediction = tape.values(ex, y)
        tape.record()
    """
    var _nodes: List[_Node]
    var _parents: List[Int]
    var _buffers: List[FP]
    var _capacities: List[Int]
    var _cursor: Int
    var _identity: Optional[MutPointer[Int, MutUntrackedOrigin]]
    var _generation: Int
    var _recording: Bool
    # mark_mutated: one flag per node once any tensor of this generation is
    # marked (empty until then), True for a node that read a mutated
    # tensor's old value directly or through its inputs.
    var _stale: List[Bool]
    # watch / assert_unchanged: node indices and their host value copies.
    var _watched: List[Int]
    var _snapshots: List[List[Float32]]
    # MOJOLEARN_AG_PROFILE set at construction: mark launches with the node
    # kind that issued them (sequence/launch_profile.mojo).
    var _profile: Bool
    # AUTOGRAD_CURVE_WORD (lane ag-pass6-fold): arm_fold_word's destination
    # and factor, armed until the next AXIS_ALL reduction takes them;
    # _word_node is the node whose fold wrote the word (-1: none yet).
    var _word_dst: FP
    var _word_factor: Float32
    var _word_armed: Bool
    var _word_node: Int
    # lane ag-embedding-op: the ids of this graph (Tape.ids), cleared by reset
    var _ids: List[_Ids]
    # lane ag-rankn-core: one rank-N shape per node, parallel to _nodes
    # (appended by _record, cleared by reset). The node's rows x cols is its
    # matrix view: rows = the product of the leading dims, cols = the last.
    var _shapes: List[ShapeN]

    def __init__(out self):
        self._nodes = List[_Node]()
        self._parents = List[Int]()
        self._buffers = List[FP]()
        self._capacities = List[Int]()
        self._cursor = 0
        self._identity = None
        self._generation = 0
        self._recording = True
        self._stale = List[Bool]()
        self._watched = List[Int]()
        self._snapshots = List[List[Float32]]()
        self._profile = launch_profile_level() > 0
        self._word_dst = dummy_ptr()
        self._word_factor = Float32(0)
        self._word_armed = False
        self._word_node = -1
        self._ids = List[_Ids]()
        self._shapes = List[ShapeN]()

    def no_grad(mut self):
        """Enter inference mode: later operations keep no graph or gradients.

        Values are computed exactly as when recording. The mode lasts until
        record() and is kept across reset() and release().
        """
        self._recording = False

    def record(mut self):
        """Leave inference mode: later operations are recorded for backward."""
        self._recording = True

    def is_recording(self) -> Bool:
        """True unless no_grad() is in effect."""
        return self._recording

    def profile_mark(self, kind: Int, backward: Bool = True):
        """Attribute the launches since the last mark to node kind `kind`.

        Only has an effect when MOJOLEARN_AG_PROFILE is set (1 counts
        launches, 2 also times them). Forward operations mark themselves;
        a backward marks its launches as one row unless its node loop
        marks each node.
        """
        if not self._profile:
            return
        try:
            launch_profile_mark(PROFILE_BACKWARD if backward else PROFILE_FORWARD, kind)
        except:
            pass

    def _profile_phase(self, phase: Int):
        if not self._profile:
            return
        try:
            launch_profile_mark(phase, PROFILE_KIND_ALL)
        except:
            pass

    def profile_report(self) raises -> String:
        """The launch profile as text, when MOJOLEARN_AG_PROFILE is set.

        One line per operation code (launches, elements and, with
        MOJOLEARN_AG_PROFILE=2, milliseconds measured between
        synchronisations, a relative profile) and one line per forward or
        backward node kind. The table covers every executor and tape in
        the process since it was last cleared.
        """
        return ag_profile_report()

    def __deinit__(deinit self):
        # The tape keeps no executor reference, so it cannot free executor
        # storage here: call release(ex) when done. Anything not released is
        # reclaimed when the executor ends (it must outlive the tape).
        if self._identity:
            self._identity.value().free()

    def release[E: Exec](mut self, mut ex: E) raises:
        """Return every buffer this tape allocated to the executor.

        Invalidates all tensor handles, like reset(). Caller-owned storage
        (variable, variable_into and the *_into outputs) is left alone. The
        tape can record again afterwards; it allocates fresh storage.
        """
        for i in range(len(self._buffers)):  # driver-loop: over tape storage slots
            ex.free(self._buffers[i])
        self._buffers.clear()
        self._capacities.clear()
        self.reset()

    def reset(mut self):
        """Discard the recorded graph and reuse its executor storage."""
        # launches since the last node (optimizer steps, losses) are "other"
        self._profile_phase(PROFILE_OTHER)
        self._nodes.clear()
        self._parents.clear()
        self._generation += 1
        self._cursor = 0
        self._stale.clear()
        self._watched.clear()
        self._snapshots.clear()
        self._word_armed = False
        self._word_node = -1
        self._ids.clear()
        self._shapes.clear()

    def arm_fold_word(mut self, dst: FP, factor: Float32):
        """AUTOGRAD_CURVE_WORD: the next AXIS_ALL sum or mean recorded on
        this tape also writes dst[0] = its output times `factor` (the word
        scale(output, factor) stores), from its own fold launch. Disarmed
        by that reduction, by disarm_fold_word or by reset."""
        self._word_dst = dst
        self._word_factor = factor
        self._word_armed = True
        self._word_node = -1

    def disarm_fold_word(mut self, x: Tensor) raises -> Bool:
        """Disarm arm_fold_word; True when x's own fold wrote the word (so
        the caller's scale and copy are not needed)."""
        var i = self._check(x)
        self._word_armed = False
        var wrote = self._word_node == i
        self._word_node = -1
        return wrote

    # ---- AUTOGRAD_TRANSPOSE_VIEW (lane ag-pass6-fold, PASS6 idea F(2)) ----
    # A transpose node records its value buffer without filling it
    # (saved.i9 = 1, saved.p0 the parent's buffer, saved.i0 / i1 the
    # parent's rows / cols). matmul reads the parent through GEMM strides
    # (the same words in the same k order); every other reader of the value
    # goes through _use or materialize, which launch the copy once. Values
    # are immutable while a graph lives (operations are out of place, and
    # a leaf must stay unchanged until backward finishes), so the copy made
    # later is the copy made at record time. value() refuses a pending view
    # (call materialize or value_of): no reader can see the unfilled buffer.
    #
    # Lane ag-rankn-core generalises the pending view (AUTOGRAD_RANKN.md 1.3):
    # a PERMUTE_N node, and a custom node of a rank-N tag (RN_TAG_FIRST ..
    # RN_TAG_END) recorded with Tape.strided_view_args (saved.i10 =
    # RN_VIEW_MARK), are pending strided views: saved.p0 the source, saved.i0
    # the base offset, saved.i1..i8 the strides, saved.p11 the descriptor.
    # A strided reader (operand_nd; matmul through GEMM strides when the
    # view's matrix view has two strides) reads them in place; a dense reader
    # fills the value buffer once with OP_RN_COPY_STRIDED (_fill_view). These
    # views are pending regardless of AUTOGRAD_TRANSPOSE_VIEW, which governs
    # the transpose node only; the TRANSPOSE branch is the code above.
    @always_inline
    def _pending(self, i: Int) -> Bool:
        if self._nodes[i].saved.i9 != 1:
            return False
        var op = self._nodes[i].op
        if op == PERMUTE_N:
            return True
        if op >= CUSTOM + RN_TAG_FIRST and op < CUSTOM + RN_TAG_END and self._nodes[i].saved.i10 == RN_VIEW_MARK:
            return True
        comptime if AUTOGRAD_TRANSPOSE_VIEW:
            return op == TRANSPOSE
        else:
            return False

    def _fill_view[E: Exec](self, mut ex: E, i: Int) raises:
        """Write pending view i's words into its value buffer (one copy
        launch); the caller clears or keeps the pending flag."""
        var n = self._nodes[i]
        if n.op == TRANSPOSE:
            _transpose(ex, n.value, n.saved.p0, n.saved.i0, n.saved.i1)
        else:
            rn_copy_strided(ex, n.value, n.saved.p0, n.saved.i0, n.saved.p11, 0, n.rows * n.cols)

    def _materialize_at[E: Exec](mut self, mut ex: E, i: Int) raises:
        if self._pending(i):
            self._fill_view(ex, i)
            self._nodes[i].saved.i9 = 0

    def _use[E: Exec](mut self, mut ex: E, x: Tensor) raises -> Int:
        """_check, then materialize a pending view (its value is about to
        be read as a dense matrix)."""
        var i = self._check(x)
        self._materialize_at(ex, i)
        return i

    def materialize[E: Exec](mut self, mut ex: E, x: Tensor) raises:
        """Make x's value buffer hold its values (a transpose or permute
        recorded as a view is copied once; anything else is unchanged)."""
        _ = self._use(ex, x)

    def value_of[E: Exec](mut self, mut ex: E, x: Tensor) raises -> FP:
        """value(x) after materialize(ex, x)."""
        var i = self._use(ex, x)
        return self._nodes[i].value

    def _check(self, x: Tensor) raises -> Int:
        if not self._identity:
            raise Error("tensor belongs to an empty tape")
        if x._owner != Int(self._identity.value()) or x._generation != self._generation:
            raise Error("tensor belongs to another tape or an expired graph")
        if x._index < 0 or x._index >= len(self._nodes):
            raise Error("invalid tensor handle")
        return x._index

    def _alias_root(self, i: Int) -> Int:
        """The alias group of node i: its first member (i itself when it
        shares no gradient)."""
        var a = self._nodes[i].alias_of
        return i if a < 0 else a

    def _refuse_overlap(self, p: FP, n: Int, values: Bool, what: String, shared: Int = -1) raises:
        """Refuse caller storage [p, p + n) that overlaps a recorded
        tensor's gradient storage, or (values=True) its value storage.

        The ALIASING RULE (docs/plans/NATIVE_AUTOGRAD.md): between record
        and backward, storage the tape has seen is read-only except through
        tape operations. A host scan over the recorded nodes (graph
        structure, never data); inference mode records no gradients and
        skips it. `shared` >= 0 names an alias group (custom_into_shared):
        a member's gradient that is exactly [p, p + n) is the declared
        sharing and accepted; any other overlap is refused."""
        if not self._recording:
            return
        var a = Int(p)
        for i in range(len(self._nodes)):  # driver-loop: over graph nodes
            var node = self._nodes[i]
            var count = node.rows * node.cols
            # AUTOGRAD_GRAD_ALIAS: a node's own buffer, the one its next
            # backward writes; a borrowed range is its owner's (checked as
            # that node) or the caller's old seed (written by no backward)
            var g = Int(node.own_grad) if AUTOGRAD_GRAD_ALIAS else Int(node.grad)
            if node.requires_grad and _ranges_overlap(a, n, g, count):
                if not (shared >= 0 and self._alias_root(i) == shared and g == a and count == n):
                    raise Error(what + " overlaps the gradient storage of a recorded tensor")
            if values and _ranges_overlap(a, n, Int(node.value), count):
                raise Error(what + " overlaps the value storage of a recorded tensor")

    def _refuse_seed_overlap(self, root: Int, seed: FP, accumulate: Bool) raises:
        """A backward seed may not overlap any recorded gradient, except
        the seed-in-place case: exactly the root's own gradient range
        (custom_into's grad_destination passed as the seed). An in-place
        seed cannot accumulate into a leaf root (it would add the buffer to
        itself)."""
        var a = Int(seed)
        var count = self._nodes[root].rows * self._nodes[root].cols
        for i in range(len(self._nodes)):  # driver-loop: over graph nodes
            var node = self._nodes[i]
            if not node.requires_grad:
                continue
            # the buffer this backward writes (AUTOGRAD_GRAD_ALIAS: the
            # node's own; a range borrowed by an earlier backward is
            # returned before anything is written)
            var g = Int(node.own_grad) if AUTOGRAD_GRAD_ALIAS else Int(node.grad)
            if not _ranges_overlap(a, count, g, node.rows * node.cols):
                continue
            if i == root and g == a:
                if accumulate and node.op == LEAF:
                    raise Error("a seed held in the root's own gradient cannot accumulate into it")
                continue
            # a member of the root's alias group holds the root's gradient
            # range itself (custom_into_shared): the same seed in place
            if (g == a and node.rows * node.cols == count
                    and self._alias_root(i) == self._alias_root(root)):
                continue
            raise Error("backward seed overlaps the gradient storage of a recorded tensor")

    def _refuse_stale(self, root: Int) raises:
        if len(self._stale) > 0 and self._stale[root]:
            raise Error(_STALE_REFUSAL)

    def _stale_of(self, parents: List[Tensor]) -> Bool:
        """A new node is stale when an input is (its parents are checked)."""
        if len(self._stale) == 0:
            return False
        for p in parents:  # driver-loop: over node inputs
            if self._stale[p._index]:
                return True
        return False

    def mark_mutated(mut self, x: Tensor) raises:
        """Declare that x's storage was changed outside the tape.

        Every tensor recorded before this call that depends on x through
        x's consumers becomes stale, and a backward from a stale output
        raises. Tensors recorded afterwards read x's new value and stay
        valid. reset() clears the marks.
        """
        var i = self._check(x)
        while len(self._stale) < len(self._nodes):  # driver-loop: over graph nodes
            self._stale.append(False)
        for j in range(i + 1, len(self._nodes)):  # driver-loop: over graph nodes, creation order
            var n = self._nodes[j]
            for edge in range(n.count):  # driver-loop: over node inputs
                var p = self._parents[n.start + edge]
                if p == i or self._stale[p]:
                    self._stale[j] = True

    def watch[E: Exec](mut self, mut ex: E, x: Tensor) raises:
        """Keep a host copy of x's value for a later assert_unchanged.

        A debugging aid: it downloads the value. Watching x again replaces
        the copy.
        """
        var i = self._use(ex, x)
        var copy = self.values(ex, x)
        for k in range(len(self._watched)):  # driver-loop: over watched tensors
            if self._watched[k] == i:
                self._snapshots[k] = copy^
                return
        self._watched.append(i)
        self._snapshots.append(copy^)

    def assert_unchanged[E: Exec](mut self, mut ex: E, x: Tensor) raises:
        """Raise if x's value differs, bit for bit, from its watch() copy.

        On a difference x is marked mutated (see mark_mutated) before the
        error is raised, so a later backward through its old consumers
        refuses. A debugging aid: it downloads the value.

        Example::

            var w = tape.variable(ex, weights, rows, cols)
            tape.watch(ex, w)
            # ... record, then before backward:
            tape.assert_unchanged(ex, w)
        """
        var i = self._use(ex, x)
        for k in range(len(self._watched)):  # driver-loop: over watched tensors
            if self._watched[k] != i:
                continue
            var before = self._snapshots[k].copy()
            var now = self.values(ex, x)
            if len(now) != len(before) or memcmp(now.unsafe_ptr(), before.unsafe_ptr(), len(now)) != 0:
                self.mark_mutated(x)
                raise Error("tensor storage changed between record and backward")
            return
        raise Error("assert_unchanged needs an earlier watch() of this tensor")

    def _alloc[E: Exec](mut self, mut ex: E, n: Int) raises -> FP:
        var slot = self._cursor
        if slot == len(self._buffers):
            self._buffers.append(ex.alloc(n))
            self._capacities.append(n)
        elif self._capacities[slot] < n:
            # A slot is reused only by a later graph (after reset) or by the
            # next node's backward scratch, so nothing still names the old
            # buffer: return it (a device keeps it until its queue drains).
            # Doubling capacity bounds reallocations for growing shapes.
            var capacity = max(n, self._capacities[slot] * 2)
            ex.free(self._buffers[slot])
            self._buffers[slot] = ex.alloc(capacity)
            self._capacities[slot] = capacity
        self._cursor += 1
        return self._buffers[slot]

    def _shape(self, rows: Int, cols: Int) raises:
        if rows <= 0 or cols <= 0 or rows > ((1 << 60) // 4) // cols:
            raise Error("tensor dimensions must be positive and fit in addressable storage")

    def reserve[E: Exec](mut self, mut ex: E, elements: Int) raises -> FP:
        """Reserve executor storage for a custom operation until the next reset."""
        self._shape(1, elements)
        return self._alloc(ex, elements)

    def _record[E: Exec](
        mut self, mut ex: E, value: FP, rows: Int, cols: Int,
        op: Int, parents: List[Tensor], saved: Args, leaf_grad: Bool = False,
        grad_storage: FP = dummy_ptr(), caller_grad: Bool = False,
    ) raises -> Tensor:
        # ONE VALUE PATH (lane ag-forward). Every operation computes its value
        # (allocation through _alloc, or caller storage for *_into, then the
        # same launch helper) BEFORE calling _record, and _record never
        # launches a value kernel. Inference mode differs from recording only
        # here: the node keeps no parents, gets no gradient storage and is
        # marked inferred. So a recorded and a non-recorded forward run the
        # same kernels on the same inputs and produce the same bits by
        # construction. (Recording also takes gradient slots from _alloc, so
        # the two modes may place values in different slots; slot placement
        # never changes a value.)
        if self._profile:
            # the launches that computed this value (lane ag-pass2-kernels)
            self.profile_mark(op, False)
        self._shape(rows, cols)
        if not self._identity:
            self._identity = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(alloc[Int](1)))
        if not self._recording:
            for p in parents:  # driver-loop: over node inputs
                _ = self._check(p)
            if len(self._stale) > 0:
                self._stale.append(self._stale_of(parents))
            var at = len(self._nodes)
            self._nodes.append(_Node(value, dummy_ptr(), rows, cols, False, op, len(self._parents), 0,
                                     saved, False, True, False, False, -1, dummy_ptr()))
            self._shapes.append(ShapeN.matrix(rows, cols))
            return Tensor(Int(self._identity.value()), self._generation, at)
        var requires = leaf_grad
        var start = len(self._parents)
        for p in parents:
            var i = self._check(p)
            requires = requires or self._nodes[i].requires_grad
        for p in parents:
            self._parents.append(p._index)
        # No fill here: backward writes each gradient's first contribution
        # straight into it (see the invariant in _backward).
        var grad = dummy_ptr()
        var ready = False
        if requires:
            if caller_grad:
                grad = grad_storage
                ready = True
            else:
                grad = self._alloc(ex, rows * cols)
        if len(self._stale) > 0:
            self._stale.append(self._stale_of(parents))
        var index = len(self._nodes)
        self._nodes.append(_Node(value, grad, rows, cols, requires, op, start, len(parents), saved, ready, False, False,
                                 requires and caller_grad, -1, grad))
        self._shapes.append(ShapeN.matrix(rows, cols))
        return Tensor(Int(self._identity.value()), self._generation, index)

    def variable[E: Exec](
        mut self, mut ex: E, data: FP, rows: Int, cols: Int, requires_grad: Bool = True,
    ) raises -> Tensor:
        """Record borrowed executor memory as a leaf; keep it unchanged until backward finishes."""
        return self._record(ex, data, rows, cols, LEAF, List[Tensor](), Args(), requires_grad)

    def variable_into[E: Exec](
        mut self, mut ex: E, data: FP, grad_destination: FP, rows: Int, cols: Int,
    ) raises -> Tensor:
        """Record a differentiable leaf whose gradient lands in caller storage.

        `grad_destination` holds rows * cols floats of this executor. A
        backward writes the gradient there directly (accumulate=True adds to
        what it holds). Keep `data` unchanged until backward finishes, and
        do not let `grad_destination` overlap any tensor of the graph.
        """
        self._shape(rows, cols)
        var count = rows * cols
        self._refuse_overlap(grad_destination, count, True, "variable_into: grad_destination")
        self._refuse_overlap(data, count, False, "variable_into: data")
        if _ranges_overlap(Int(data), count, Int(grad_destination), count):
            raise Error("variable_into: grad_destination overlaps data")
        return self._record(ex, data, rows, cols, LEAF, List[Tensor](), Args(), True,
                            grad_destination, True)

    def tensor[E: Exec](
        mut self, mut ex: E, values: List[Float32], rows: Int, cols: Int,
        requires_grad: Bool = False,
    ) raises -> Tensor:
        """Copy a row-major host list onto the executor and record a leaf."""
        self._shape(rows, cols)
        if len(values) != rows * cols:
            raise Error("tensor data length does not match shape")
        var data = self._alloc(ex, rows * cols)
        ex.upload(data, FP(unsafe_from_address=Int(values.unsafe_ptr())), rows * cols)
        return self.variable(ex, data, rows, cols, requires_grad)

    def scalar[E: Exec](mut self, mut ex: E, value: Float32, requires_grad: Bool = False) raises -> Tensor:
        """Create a scalar on the executor."""
        var p = self._alloc(ex, 1)
        _fill(ex, p, 1, value)
        return self.variable(ex, p, 1, 1, requires_grad)

    # ---- the embedding op (lane ag-embedding-op, TAPE_EVERYWHERE path 4) ----
    # Tape.embedding gathers table rows by integer ids (an IdsHandle, the
    # tape's one non-float input) and records an EMBEDDING node whose VJP is
    # the embedding profile's run-sorted scatter-add into the table's
    # gradient (autograd/embedding_ops.mojo states the words). The VJP
    # writes the table's whole gradient itself: when that gradient already
    # holds a value (an accumulate=True leaf that holds one, or another
    # consumer's earlier contribution) the fold CONTINUES from it, as
    # Embedding.backward(ids, dy, grad=carry) does, instead of adding a
    # separately folded term; otherwise it starts from +0.0. So a tape
    # backward reproduces Embedding.backward's bits with and without a
    # carry, and microbatches in ascending position order reproduce the
    # unsplit gradient.
    def ids[E: Exec](mut self, mut ex: E, data: FP, n: Int) raises -> IdsHandle:
        """Record n integer row ids held in executor storage for embedding().

        `data` holds n int32 values, each as its bit pattern in one 4-byte
        word of this executor. The ids take no gradient. Keep the storage
        unchanged until backward finishes. The handle expires at reset().
        """
        if n < 1:
            raise Error("ids: at least one id is needed")
        self._refuse_overlap(data, n, False, "ids: data")
        if not self._identity:
            self._identity = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(alloc[Int](1)))
        self._ids.append(_Ids(data, n, 0))
        return IdsHandle(Int(self._identity.value()), self._generation, len(self._ids) - 1)

    def _check_ids(self, h: IdsHandle) raises -> Int:
        if not self._identity:
            raise Error("ids belong to an empty tape")
        if h._owner != Int(self._identity.value()) or h._generation != self._generation:
            raise Error("ids belong to another tape or an expired graph")
        if h._index < 0 or h._index >= len(self._ids):
            raise Error("invalid ids handle")
        return h._index

    def ids_count(self, h: IdsHandle) raises -> Int:
        """The number of ids behind handle h."""
        return self._ids[self._check_ids(h)].n

    def embedding[E: Exec](
        mut self, mut ex: E, table: Tensor, ids: IdsHandle, padding_idx: Int = -1,
        plan: Int = AG_EMB_PLAN_AUTO,
    ) raises -> Tensor:
        """Rows of `table` (vocab, width) picked by `ids`: an (n, width) tensor.

        Row t is table[ids[t]]. An id outside [0, vocab) is refused before
        any row is read. The table's gradient sums, for each row, the
        output gradient of every position that picked it, in ascending
        position order; row `padding_idx` (-1: none) gets a zero gradient.
        `plan` 0 (scan), 1 (sort) or 2 (auto) chooses how the backward
        groups positions by row; every plan gives the same bits.

        Example::

            var w = tape.variable(ex, table, vocab, width)
            var x = tape.embedding(ex, w, tape.ids(ex, ids, n))
            tape.backward(ex, tape.sum(ex, x))
        """
        return self._embedding_to(ex, table, ids, padding_idx, plan, dummy_ptr(), False)

    def embedding_into[E: Exec](
        mut self, mut ex: E, table: Tensor, ids: IdsHandle, output: FP, padding_idx: Int = -1,
        plan: Int = AG_EMB_PLAN_AUTO,
    ) raises -> Tensor:
        """embedding() written to caller storage of n * width floats.

        Keep `output` unchanged until backward finishes; it must not
        overlap a recorded tensor.
        """
        return self._embedding_to(ex, table, ids, padding_idx, plan, output, True)

    def _embedding_to[E: Exec](
        mut self, mut ex: E, table: Tensor, ids: IdsHandle, padding_idx: Int, plan: Int,
        into: FP, use_into: Bool,
    ) raises -> Tensor:
        var w_at = self._use(ex, table)
        var w = self._nodes[w_at]
        var k = self._check_ids(ids)
        var rec = self._ids[k]
        var vocab = w.rows
        var width = w.cols
        var n = rec.n
        embedding_refuse_shape(n, vocab, width, padding_idx)
        _ = embedding_plan(plan, vocab, n)
        self._shape(n, width)
        if rec.checked == 0 or vocab < rec.checked:
            # every id against this table's rows, on the executor, before
            # the gather reads any (one read-back; a handle checked against
            # at most this many rows is not checked again)
            var cw = embedding_check_words(n)
            var scratch_a = self._alloc(ex, cw)
            var scratch_b = self._alloc(ex, cw)
            embedding_refuse_ids(ex, rec.data, n, vocab, scratch_a, scratch_b)
            self._ids[k].checked = vocab
        if use_into:
            self._refuse_overlap(into, n * width, True, "embedding_into: output")
        var output = into if use_into else self._alloc(ex, n * width)
        embedding_forward_into(ex, w.value, rec.data, output, n, width)
        var saved = Args()
        saved.p0 = rec.data
        saved.i0 = n
        saved.i1 = vocab
        saved.i2 = width
        saved.i3 = padding_idx
        saved.i4 = plan
        return self._record(ex, output, n, width, EMBEDDING, [table], saved)

    def value(self, x: Tensor) raises -> FP:
        """Return the executor pointer for a tensor's values."""
        var i = self._check(x)
        if self._pending(i):
            raise Error("autograd: the tensor is a view whose values are not copied yet: call materialize(ex, x) or value_of(ex, x)")
        return self._nodes[i].value

    def gradient(self, x: Tensor) raises -> FP:
        """Return the executor pointer for a differentiable tensor's gradient."""
        var n = self._nodes[self._check(x)]
        if n.inferred:
            raise Error(_INFERRED_REFUSAL)
        if not n.requires_grad:
            raise Error("tensor does not require a gradient")
        if not n.grad_ready:
            raise Error("tensor has no gradient: run backward from an output that depends on it")
        if n.grad_preact:
            raise Error("this linear output holds its pre-activation gradient: use preactivation_gradient")
        return n.grad

    def clear_gradient(mut self, x: Tensor) raises:
        """Forget a differentiable leaf's gradient: `gradient` refuses it
        until the next backward, and a backward with accumulate=True starts
        it from zero (its first contribution is written in place, the
        schedule of a leaf never written before). No launch."""
        var i = self._check(x)
        if self._nodes[i].op != LEAF:
            raise Error("clear_gradient takes a leaf tensor")
        if self._nodes[i].inferred or not self._nodes[i].requires_grad:
            return
        self._nodes[i].grad_ready = False

    def preactivation_gradient(self, x: Tensor) raises -> FP:
        """Return the gradient with respect to a linear output's pre-activation.

        Available after a backward for an output recorded by linear or
        linear_into whose activation derivative was applied in place, and
        for one without an activation.
        """
        var n = self._nodes[self._check(x)]
        if n.op != LINEAR:
            raise Error("preactivation_gradient takes an output of linear or linear_into")
        if not n.requires_grad or not n.grad_ready:
            raise Error("tensor has no gradient: run backward from an output that depends on it")
        if not n.grad_preact and n.saved.i0 != ACT_IDENTITY:
            raise Error("the pre-activation gradient of this output was not kept")
        return n.grad

    def requires_grad(self, x: Tensor) raises -> Bool:
        """Return whether a backward computes a gradient for this tensor."""
        return self._nodes[self._check(x)].requires_grad

    def rows(self, x: Tensor) raises -> Int:
        return self._nodes[self._check(x)].rows

    def cols(self, x: Tensor) raises -> Int:
        return self._nodes[self._check(x)].cols

    # ---- rank-N shapes and views (lane ag-rankn-core, docs/plans/AUTOGRAD_RANKN.md 1.1-1.4) ----
    # Storage is one flat row-major buffer per tensor; the shape is metadata
    # (_shapes). rows(x) and cols(x) stay the matrix view. Nothing here
    # changes a word an existing operation writes: the existing methods only
    # gain metadata writes (_with_shape), and the new ones are new nodes.
    def shape_of(self, x: Tensor) raises -> ShapeN:
        """The shape of x (rank 2, (rows, cols), unless x was recorded with
        a rank-N shape or inherits one)."""
        return self._shapes[self._check(x)]

    def ndim(self, x: Tensor) raises -> Int:
        """The rank of x."""
        return self._shapes[self._check(x)].ndim

    def dim(self, x: Tensor, k: Int) raises -> Int:
        """The extent of axis k of x (negative k counts from the end)."""
        var s = self._shapes[self._check(x)]
        return s.d[rn_normalize_axis(k, s.ndim)]

    @always_inline
    def _with_shape(mut self, t: Tensor, shape: ShapeN) -> Tensor:
        """Set the rank-N shape of node t (its count equals rows * cols)."""
        self._shapes[t._index] = shape
        return t

    def _lead_shape(self, i: Int, cols: Int) -> ShapeN:
        """The shape of a row-wise result of node i with `cols` columns: i's
        leading dims and cols when i has rank 3 or more, else the matrix
        (rows of i, cols)."""
        var s = self._shapes[i]
        if s.ndim >= 3:
            return s.with_last(cols)
        return ShapeN.matrix(self._nodes[i].rows, cols)

    def _binary_shape(self, a_at: Int, b_at: Int, rows: Int, cols: Int) -> ShapeN:
        """The shape of an elementwise binary result over the matrix views:
        the operands' shape when they agree; the full operand's shape when
        the other is a (1, 1), (1, cols) or (rows, 1) broadcast of it (the
        trailing-dims broadcast); otherwise the matrix (rows, cols)."""
        var sa = self._shapes[a_at]
        var sb = self._shapes[b_at]
        if sa.same(sb):
            return sa
        var a_full = self._nodes[a_at].rows == rows and self._nodes[a_at].cols == cols
        var b_full = self._nodes[b_at].rows == rows and self._nodes[b_at].cols == cols
        if a_full and not b_full:
            return sa
        if b_full and not a_full:
            return sb
        return ShapeN.matrix(rows, cols)

    def _record_nd[E: Exec](
        mut self, mut ex: E, value: FP, shape: ShapeN,
        op: Int, parents: List[Tensor], saved: Args, leaf_grad: Bool = False,
        grad_storage: FP = dummy_ptr(), caller_grad: Bool = False,
    ) raises -> Tensor:
        """_record with a rank-N shape: the node's matrix view is
        (shape.rows(), shape.cols())."""
        rn_check(shape)
        var t = self._record(ex, value, shape.rows(), shape.cols(), op, parents, saved, leaf_grad,
                             grad_storage, caller_grad)
        return self._with_shape(t, shape)

    def variable_nd[E: Exec](
        mut self, mut ex: E, data: FP, shape: ShapeN, requires_grad: Bool = True,
    ) raises -> Tensor:
        """variable() with a shape of any rank up to 8: `data` holds
        shape.count() floats, row-major."""
        rn_check(shape)
        var t = self.variable(ex, data, shape.rows(), shape.cols(), requires_grad)
        return self._with_shape(t, shape)

    def variable_into_nd[E: Exec](
        mut self, mut ex: E, data: FP, grad_destination: FP, shape: ShapeN,
    ) raises -> Tensor:
        """variable_into() with a shape of any rank up to 8."""
        rn_check(shape)
        var t = self.variable_into(ex, data, grad_destination, shape.rows(), shape.cols())
        return self._with_shape(t, shape)

    def tensor_nd[E: Exec](
        mut self, mut ex: E, values: List[Float32], shape: ShapeN, requires_grad: Bool = False,
    ) raises -> Tensor:
        """tensor() with a shape of any rank up to 8: `values` holds
        shape.count() floats, row-major."""
        rn_check(shape)
        var t = self.tensor(ex, values, shape.rows(), shape.cols(), requires_grad)
        return self._with_shape(t, shape)

    def custom_nd[E: Exec](
        mut self, mut ex: E, inputs: List[Tensor], output: FP, shape: ShapeN, tag: Int, saved: Args,
    ) raises -> Tensor:
        """custom() with a shape of any rank up to 8 (the rule sees the
        matrix view rows x cols)."""
        rn_check(shape)
        var t = self.custom(ex, inputs, output, shape.rows(), shape.cols(), tag, saved)
        return self._with_shape(t, shape)

    def custom_into_nd[E: Exec](
        mut self, mut ex: E, inputs: List[Tensor], output: FP, grad_destination: FP, shape: ShapeN,
        tag: Int, saved: Args,
    ) raises -> Tensor:
        """custom_into() with a shape of any rank up to 8."""
        rn_check(shape)
        var t = self.custom_into(ex, inputs, output, grad_destination, shape.rows(), shape.cols(), tag, saved)
        return self._with_shape(t, shape)

    def descriptor[E: Exec](mut self, mut ex: E, words: List[Int]) raises -> FP:
        """Tape storage holding `words` as int32 bit patterns, one a word,
        until the next reset: the shapes and strides a rank-N kernel reads
        (bind it to Args.p11; read it with autograd.rankn.rn_word). One
        upload, never per element."""
        if len(words) < 1:
            raise Error("descriptor: at least one word is needed")
        var host = List[Float32](length=len(words), fill=Float32(0))
        for i in range(len(words)):  # small-loop(descriptor words: shape metadata, a few 17-word records, no data)
            var w = words[i]
            if w < -(1 << 31) or w >= (1 << 31):
                raise Error("descriptor: a shape or stride does not fit in 32 bits")
            host[i] = bitcast[DType.float32](Int32(w))
        var p = self._alloc(ex, len(words))
        ex.upload(p, FP(unsafe_from_address=Int(host.unsafe_ptr())), len(words))
        return p

    def strided_view_args[E: Exec](
        mut self, mut ex: E, source: FP, base: Int, shape: ShapeN, strides: InlineArray[Int, RN_MAX_RANK],
    ) raises -> Args:
        """The saved Args of a pending strided view (autograd.rankn
        RN_VIEW_MARK): element c of the view is source[base + sum_k c_k *
        strides[k]]. Record it with PERMUTE_N, or with custom_nd and a tag in
        RN_TAG_FIRST .. RN_TAG_END, and an output buffer of shape.count()
        floats that the tape fills at the first dense read."""
        rn_check(shape)
        var words = List[Int]()
        rn_desc_view(words, shape, strides)
        var a = Args()
        a.p0 = source
        a.p11 = self.descriptor(ex, words)
        a.i0 = base
        a.i1 = strides[0]
        a.i2 = strides[1]
        a.i3 = strides[2]
        a.i4 = strides[3]
        a.i5 = strides[4]
        a.i6 = strides[5]
        a.i7 = strides[6]
        a.i8 = strides[7]
        a.i9 = 1
        a.i10 = RN_VIEW_MARK
        return a

    def _view_strides(self, i: Int) -> InlineArray[Int, RN_MAX_RANK]:
        """A pending strided view's strides (saved.i1 .. i8)."""
        var a = self._nodes[i].saved
        var st = InlineArray[Int, RN_MAX_RANK](fill=0)
        st[0] = a.i1
        st[1] = a.i2
        st[2] = a.i3
        st[3] = a.i4
        st[4] = a.i5
        st[5] = a.i6
        st[6] = a.i7
        st[7] = a.i8
        return st^

    def operand_nd[E: Exec](
        self, mut ex: E, x: Tensor,
    ) raises -> Tuple[FP, Int, ShapeN, InlineArray[Int, RN_MAX_RANK]]:
        """x as a strided read, without a copy: (buffer, base offset, shape,
        element strides). Element c of x is buffer[base + sum_k c_k *
        strides[k]]. A dense tensor is (its values, 0, its shape, row-major
        strides); a pending view is its source read through the view's
        strides."""
        var i = self._check(x)
        var s = self._shapes[i]
        var zero: Int = 0
        if self._pending(i):
            if self._nodes[i].op == TRANSPOSE:
                # element (r, c) is the source's (c, r): strides (1, rows)
                var st = InlineArray[Int, RN_MAX_RANK](fill=0)
                st[0] = 1
                st[1] = self._nodes[i].rows
                return (self._nodes[i].saved.p0, zero, s, st^)
            return (self._nodes[i].saved.p0, self._nodes[i].saved.i0, s, self._view_strides(i))
        return (self._nodes[i].value, zero, s, s.strides())

    def _gemm_view(self, i: Int) -> Tuple[Bool, Int, Int]:
        """Whether node i's matrix view reads as a GEMM operand in place:
        (True, row stride, column stride) for a dense node, a pending
        transpose, or a pending strided view at base 0 whose leading axes
        collapse to one stride; else (False, 0, 0) and the caller
        materializes it first."""
        var zero: Int = 0
        var one: Int = 1
        if not self._pending(i):
            return (True, self._nodes[i].cols, one)
        if self._nodes[i].op == TRANSPOSE:
            return (True, one, self._nodes[i].rows)
        if self._nodes[i].saved.i0 != 0:
            return (False, zero, zero)
        var s = self._shapes[i]
        var st = self._view_strides(i)
        if s.ndim < 1:
            return (True, zero, zero)
        var lead_d = InlineArray[Int, RN_MAX_RANK](fill=1)
        var lead_s = InlineArray[Int, RN_MAX_RANK](fill=0)
        for k in range(s.ndim - 1):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
            lead_d[k] = s.d[k]
            lead_s[k] = st[k]
        var lead = ShapeN(s.ndim - 1, lead_d^)
        var ls = List[InlineArray[Int, RN_MAX_RANK]]()
        ls.append(lead_s^)
        rn_collapse(lead, ls)
        if lead.ndim != 1:
            return (False, zero, zero)
        return (True, ls[0][0], st[s.ndim - 1])

    def view_nd[E: Exec](mut self, mut ex: E, x: Tensor, shape: ShapeN) raises -> Tensor:
        """x's elements, row-major, in a new shape with the same element
        count (reshape, flatten, squeeze, unsqueeze). No copy: the result
        shares x's storage, and its gradient passes through unchanged. A
        pending view is copied once first."""
        rn_check(shape)
        var n_at = self._use(ex, x)
        var n = self._nodes[n_at]
        if shape.count() != n.rows * n.cols:
            raise Error("view_nd must preserve the element count")
        return self._record_nd(ex, n.value, shape, RESHAPE, [x], Args())

    def permute[E: Exec](mut self, mut ex: E, x: Tensor, axes: List[Int]) raises -> Tensor:
        """x with its axes reordered: axis j of the result is axis axes[j]
        of x (negative axes count from the end; every axis once).

        No copy at record time: the result is a view read in place by
        strided readers (matmul when its matrix view has two strides, and
        the rank-N operations); any other reader copies it once. Its
        gradient is the upstream gradient permuted back.

        Example::

            var x = tape.tensor_nd(ex, data, ShapeN.of([2, 3, 4]), True)
            var y = tape.permute(ex, x, [0, 2, 1])   # (2, 4, 3)
        """
        var xi = self._check(x)
        var s = self._shapes[xi]
        if len(axes) != s.ndim:
            raise Error("permute: axes must name every axis of the tensor once")
        var perm = InlineArray[Int, RN_MAX_RANK](fill=0)
        var seen = InlineArray[Bool, RN_MAX_RANK](fill=False)
        var identity = True
        for j in range(s.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
            var k = rn_normalize_axis(axes[j], s.ndim)
            if seen[k]:
                raise Error("permute: axes must name every axis of the tensor once")
            seen[k] = True
            perm[j] = k
            if k != j:
                identity = False
        if identity:
            return self.view_nd(ex, x, s)
        # the source as x is read (a view of a view composes its strides)
        var src = self.operand_nd(ex, x)
        var src_strides = src[3].copy()
        var od = InlineArray[Int, RN_MAX_RANK](fill=1)
        var fs = InlineArray[Int, RN_MAX_RANK](fill=0)
        for j in range(s.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
            od[j] = s.d[perm[j]]
            fs[j] = src_strides[perm[j]]
        var o = ShapeN(s.ndim, od^)
        var saved = self.strided_view_args(ex, src[0], src[1], o, fs)
        # VJP record (saved.p1): x's gradient (dense, x's shape) element c
        # reads the upstream (dense, o's shape) at sum_k c_k * so[inv[k]],
        # inv[perm[j]] = j
        var so = o.strides()
        var vs = InlineArray[Int, RN_MAX_RANK](fill=0)
        for j in range(s.ndim):  # small-loop(ndim: tensor rank, at most 8): shape metadata, no data
            vs[perm[j]] = so[j]
        var vjp_words = List[Int]()
        rn_desc_view(vjp_words, s, vs)
        saved.p1 = self.descriptor(ex, vjp_words)
        var output = self._alloc(ex, o.count())
        return self._record_nd(ex, output, o, PERMUTE_N, [x], saved)

    def values[E: Exec](self, mut ex: E, x: Tensor) raises -> List[Float32]:
        """Download a tensor into a row-major host list."""
        var vi = self._check(x)
        var n = self._nodes[vi]
        if self._pending(vi):
            # a pending view: fill its buffer (the flag stays; a later
            # materialize copies the same words again)
            self._fill_view(ex, vi)
        var result = List[Float32](length=n.rows * n.cols, fill=0)
        ex.download(FP(unsafe_from_address=Int(result.unsafe_ptr())), n.value, len(result))
        return result^

    def gradients[E: Exec](self, mut ex: E, x: Tensor) raises -> List[Float32]:
        """Download a differentiable tensor's gradient into a host list."""
        var p = self.gradient(x)
        var n = self._nodes[self._check(x)]
        var result = List[Float32](length=n.rows * n.cols, fill=0)
        ex.download(FP(unsafe_from_address=Int(result.unsafe_ptr())), p, len(result))
        return result^

    def _binary[E: Exec](mut self, mut ex: E, x: Tensor, y: Tensor, op: Int) raises -> Tensor:
        return self._binary_to(ex, x, y, op, dummy_ptr(), False)

    def binary_into[E: Exec](mut self, mut ex: E, x: Tensor, y: Tensor, op: Int, output: FP) raises -> Tensor:
        """add/subtract/multiply/divide (op ADD, SUBTRACT, MULTIPLY or DIVIDE
        from autograd.ops) written to caller storage of the broadcast shape.

        Values equal the matching method's. Keep `output` unchanged until
        backward finishes; it must not overlap an input.
        """
        if op != ADD and op != SUBTRACT and op != MULTIPLY and op != DIVIDE:
            raise Error("binary_into takes ADD, SUBTRACT, MULTIPLY or DIVIDE")
        return self._binary_to(ex, x, y, op, output, True)

    def _binary_to[E: Exec](
        mut self, mut ex: E, x: Tensor, y: Tensor, op: Int, into: FP, use_into: Bool,
    ) raises -> Tensor:
        var a_at = self._use(ex, x)
        var a = self._nodes[a_at]
        var b_at = self._use(ex, y)
        var b = self._nodes[b_at]
        if (a.rows != b.rows and a.rows != 1 and b.rows != 1) or (a.cols != b.cols and a.cols != 1 and b.cols != 1):
            raise Error("incompatible tensor shapes for broadcasting")
        var rows = max(a.rows, b.rows)
        var cols = max(a.cols, b.cols)
        self._shape(rows, cols)
        if use_into:
            self._refuse_overlap(into, rows * cols, True, "binary_into: output")
        var output = into if use_into else self._alloc(ex, rows * cols)
        # Every other slot keeps the Args() placeholder; the body touches
        # p0, p1 and p2 only. x * x binds its one buffer once (i10 = 1).
        var args = Args()
        args.i1 = op
        args.i2 = rows
        args.i3 = cols
        args.i4 = a.rows
        args.i5 = a.cols
        args.i6 = b.rows
        args.i7 = b.cols
        args.p0 = a.value
        if Int(b.value) == Int(a.value):
            args.i10 = 1
        else:
            args.p1 = b.value
        args.p2 = output
        ex.launch[OP_AG_BIN_FWD](args, rows * cols)
        var shape = self._binary_shape(a_at, b_at, rows, cols)  # rank-N metadata only
        var t = self._record(ex, output, rows, cols, op, [x, y], args)
        return self._with_shape(t, shape)

    def add[E: Exec](mut self, mut ex: E, x: Tensor, y: Tensor) raises -> Tensor:
        """Add tensors with scalar, row, or column broadcasting."""
        return self._binary(ex, x, y, ADD)

    def subtract[E: Exec](mut self, mut ex: E, x: Tensor, y: Tensor) raises -> Tensor:
        return self._binary(ex, x, y, SUBTRACT)

    def multiply[E: Exec](mut self, mut ex: E, x: Tensor, y: Tensor) raises -> Tensor:
        return self._binary(ex, x, y, MULTIPLY)

    def divide[E: Exec](mut self, mut ex: E, x: Tensor, y: Tensor) raises -> Tensor:
        return self._binary(ex, x, y, DIVIDE)

    def _unary[E: Exec](mut self, mut ex: E, x: Tensor, op: Int, scale: Float32 = 1) raises -> Tensor:
        return self._unary_to(ex, x, op, scale, dummy_ptr(), False)

    def unary_into[E: Exec](
        mut self, mut ex: E, x: Tensor, op: Int, output: FP, scale: Float32 = 1,
    ) raises -> Tensor:
        """An elementwise or row operation written to caller storage.

        op is RELU, SIGMOID, TANH, EXP, LOG, SQRT, SCALE (factor `scale`),
        SOFTMAX or LOG_SOFTMAX from autograd.ops. `output` holds x's
        rows * cols floats; values equal the matching method's. Keep it
        unchanged until backward finishes; it must not overlap x.
        """
        if (op < RELU or op > SCALE) and op != SOFTMAX and op != LOG_SOFTMAX:
            raise Error("unary_into takes an elementwise or softmax operation")
        return self._unary_to(ex, x, op, scale, output, True)

    def _unary_to[E: Exec](
        mut self, mut ex: E, x: Tensor, op: Int, scale: Float32, into: FP, use_into: Bool,
    ) raises -> Tensor:
        var n_at = self._use(ex, x)
        var n = self._nodes[n_at]
        if use_into:
            self._refuse_overlap(into, n.rows * n.cols, True, "unary_into: output")
        var output = into if use_into else self._alloc(ex, n.rows * n.cols)
        var a = Args()
        a.p0 = n.value
        a.p1 = output
        if op == SOFTMAX or op == LOG_SOFTMAX:
            a.i0 = 1 if op == LOG_SOFTMAX else 0
            a.i1 = n.cols
            ex.launch[OP_AG_SOFTMAX_FWD](a, n.rows)
        else:
            a.i0 = op
            a.f0 = scale
            ex.launch[OP_AG_UNARY_FWD](a, n.rows * n.cols)
        var shape = self._shapes[n_at]  # rank-N metadata only: the input's shape
        var t = self._record(ex, output, n.rows, n.cols, op, [x], a)
        return self._with_shape(t, shape)

    def relu[E: Exec](mut self, mut ex: E, x: Tensor) raises -> Tensor:
        return self._unary(ex, x, RELU)

    def sigmoid[E: Exec](mut self, mut ex: E, x: Tensor) raises -> Tensor:
        return self._unary(ex, x, SIGMOID)

    def tanh[E: Exec](mut self, mut ex: E, x: Tensor) raises -> Tensor:
        return self._unary(ex, x, TANH)

    def exp[E: Exec](mut self, mut ex: E, x: Tensor) raises -> Tensor:
        return self._unary(ex, x, EXP)

    def log[E: Exec](mut self, mut ex: E, x: Tensor) raises -> Tensor:
        return self._unary(ex, x, LOG)

    def sqrt[E: Exec](mut self, mut ex: E, x: Tensor) raises -> Tensor:
        return self._unary(ex, x, SQRT)

    def scale[E: Exec](mut self, mut ex: E, x: Tensor, factor: Float32) raises -> Tensor:
        return self._unary(ex, x, SCALE, factor)

    def softmax[E: Exec](mut self, mut ex: E, x: Tensor) raises -> Tensor:
        """Normalize each row with a numerically stable softmax."""
        return self._unary(ex, x, SOFTMAX)

    def log_softmax[E: Exec](mut self, mut ex: E, x: Tensor) raises -> Tensor:
        return self._unary(ex, x, LOG_SOFTMAX)

    def dropout[E: Exec](
        mut self, mut ex: E, x: Tensor, p: Float32, seed: UInt64, counter: Int,
    ) raises -> Tensor:
        """Zero each element with probability p and scale the rest by 1 / (1 - p).

        The mask is a pure function of (seed, counter, element index), so the
        same seed and counter give the same mask on every executor and every
        run. Use a new counter for every training step (for example the step
        number) and the same one when a step is replayed. 0 <= p < 1 and
        0 <= counter < 2^32.

        Example::

            var h = tape.relu(ex, tape.matmul(ex, x, w))
            var d = tape.dropout(ex, h, 0.1, seed, step)
        """
        if not (p >= Float32(0) and p < Float32(1)):
            raise Error("dropout probability must be in [0, 1)")
        if counter < 0 or counter > 0xFFFFFFFF:
            raise Error("dropout counter must be in [0, 2^32)")
        var n_at = self._use(ex, x)
        var n = self._nodes[n_at]
        var output = self._alloc(ex, n.rows * n.cols)
        # s = 1 / (1 - p) once here in float32; every element then takes one
        # rounded multiply by s (autograd/data_ops.mojo).
        var scale = identical_div(Float32(1), Float32(1) - p)
        var a = dropout_args(seed, counter, p, scale)
        a.p0 = n.value
        a.p1 = output
        ex.launch[OP_AG_DROPOUT_FWD](a, n.rows * n.cols)
        var shape = self._shapes[n_at]  # rank-N metadata only: the input's shape
        var t = self._record(ex, output, n.rows, n.cols, DROPOUT, [x], dropout_args(seed, counter, p, scale))
        return self._with_shape(t, shape)

    def matmul[E: Exec](mut self, mut ex: E, x: Tensor, y: Tensor) raises -> Tensor:
        """Multiply two matrices using the existing native GEMM executor.

        x may have any rank: its leading axes are rows of the product, and
        the result keeps them."""
        var xi = self._check(x)
        var yi = self._check(y)
        # lane ag-rankn-core: a pending rank-N view whose matrix view needs
        # more than two strides (or a base offset) is copied once here; a
        # dense operand or a pending transpose is untouched (no new launch on
        # any existing call)
        comptime if AUTOGRAD_TRANSPOSE_VIEW:
            if not self._gemm_view(xi)[0]:
                self._materialize_at(ex, xi)
            if not self._gemm_view(yi)[0]:
                self._materialize_at(ex, yi)
        else:
            self._materialize_at(ex, xi)
            self._materialize_at(ex, yi)
        var shape = self._lead_shape(xi, self._nodes[yi].cols)  # rank-N metadata only
        var a = self._nodes[xi]
        var b = self._nodes[yi]
        if a.cols != b.rows:
            raise Error("matmul inner dimensions differ")
        self._shape(a.rows, b.cols)
        var output = self._alloc(ex, a.rows * b.cols)
        comptime if AUTOGRAD_TRANSPOSE_VIEW:
            # each operand dense or a transpose view read through its strides
            # (the GEMM folds the same words in the same k order either
            # way); the backward reads the same buffers and strides (saved)
            var l = self._operand(xi)
            var r = self._operand(yi)
            _gemm(ex, output, l[0], r[0], a.rows, b.cols, a.cols, l[1], l[2], r[1], r[2])
            var saved = Args()
            saved.p0 = l[0]
            saved.p1 = r[0]
            saved.i0 = l[3]
            saved.i1 = r[3]
            saved.i2 = l[1]
            saved.i3 = l[2]
            saved.i4 = r[1]
            saved.i5 = r[2]
            saved.i6 = 1
            var t = self._record(ex, output, a.rows, b.cols, MATMUL, [x, y], saved)
            return self._with_shape(t, shape)
        else:
            _gemm(ex, output, a.value, b.value, a.rows, b.cols, a.cols, a.cols, 1, b.cols, 1)
            var t = self._record(ex, output, a.rows, b.cols, MATMUL, [x, y], Args())
            return self._with_shape(t, shape)

    def linear[E: Exec](
        mut self, mut ex: E, x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
        activation: Int = 0, l2: Float32 = 0, divisor: Float32 = 0,
    ) raises -> Tensor:
        """A dense layer, activation(x @ weight + bias), recorded as one node.

        Use it for dense layers instead of matmul, add and an activation:
        it runs one forward launch and three backward launches per layer.

        Args:
            x: The (rows, inputs) input.
            weight: The (inputs, outputs) weights.
            bias: An optional (1, outputs) bias.
            activation: 0 (none), RELU, SIGMOID, TANH or SOFTMAX from
                autograd.ops.
            l2: L2 penalty on the weights; needs a positive `divisor`.
            divisor: When positive, this layer's weight gradient is
                (x^T dZ + l2 * weight) / divisor and its bias gradient is
                the column sum of dZ divided by `divisor` (dZ the gradient
                with respect to the pre-activation). 0 leaves both unscaled.

        Returns:
            The (rows, outputs) activated output.

        Example::

            var h = tape.linear(ex, x, w1, b1, RELU)
            var logits = tape.linear(ex, h, w2, b2)
        """
        return self._linear_to(ex, x, weight, bias, activation, l2, divisor, dummy_ptr(), False)

    def linear_into[E: Exec](
        mut self, mut ex: E, x: Tensor, weight: Tensor, bias: Optional[Tensor], activation: Int,
        output: FP, l2: Float32 = 0, divisor: Float32 = 0,
    ) raises -> Tensor:
        """linear() with its output written to caller storage.

        `output` holds rows * outputs floats of this executor and must not
        overlap x. Keep it unchanged until backward finishes. Record
        `weight` and `bias` with variable_into to have their gradients land
        in caller storage as well.
        """
        return self._linear_to(ex, x, weight, bias, activation, l2, divisor, output, True)

    def _linear_to[E: Exec](
        mut self, mut ex: E, x: Tensor, weight: Tensor, bias: Optional[Tensor], activation: Int,
        l2: Float32, divisor: Float32, into: FP, use_into: Bool,
    ) raises -> Tensor:
        # LAUNCHES PER DENSE LAYER (lane ag-fusion2, read from the code).
        # Hand-assembled MLP (sequence/mlp_fit.mojo, AUTOGRAD_MLP_OFF, deleted 2026-10-19):
        #   forward 1 (gemm_epi EPI_BIAS_ACT; + OP_SOFTMAX for a softmax
        #   output); backward 3: gemm_epi EPI_L2GRAD (dW), OP_COLSUM_DIV
        #   (db), gemm_epi EPI_ACT_BWD (dX times the layer below's act');
        #   2 for the first layer (no dX).
        # Unfused tape (matmul + add + activation): forward 3 (2 without an
        #   activation); backward activation VJP 1, the product's add VJP 1,
        #   the bias's add VJP 3 (elementwise VJP + two-launch blocked fold),
        #   GEMMs 2 (1 for the first layer), the MLP's L2GRAD + DIVS export
        #   2: 9 (8 for the first layer); plus 1 seed copy a backward.
        # This node: forward 1 (gemm_epi EPI_BIAS_ACT, the hand path's call;
        #   + OP_SOFTMAX for softmax; without a bias OP_GEMM + OP_ACT).
        #   Backward 3: dW = gemm_epi EPI_L2GRAD (OP_GEMM when unscaled),
        #   db = OP_COLSUM_DIV (OP_COLSUM when unscaled), dX = gemm_epi
        #   EPI_ACT_BWD with the input layer's activation derivative in the
        #   epilogue when the input is a linear node consumed only here
        #   (else OP_GEMM). 2 for a layer whose input needs no gradient.
        #   The dZ materialisation launch (OP_ACT_BWD in place, or the
        #   softmax VJP into scratch) runs only when no consumer fused the
        #   activation derivative and the seed is not the pre-activation
        #   gradient (backward_from_preactivation): 0 in the MLP. With
        #   backward_from_preactivation the root needs no seed copy either.
        #   So the MLP runs exactly the hand path's launches with the hand
        #   path's arguments.
        # Lane ag-pass3-impl (AUTOGRAD_PASS3_DB, default on): with a bias, a
        #   divisor and both parameter gradients wanted, db comes out of the
        #   dW launch (the tile's column-sum epilogue on NVIDIA/AMD; one
        #   OP_COLSUM_DIV after the GEMM elsewhere): backward 2 a layer on the
        #   tile (1 for the first layer), the hand path's db words.
        var a_at = self._use(ex, x)
        var a = self._nodes[a_at]
        var w_at = self._use(ex, weight)
        var w = self._nodes[w_at]
        if a.cols != w.rows:
            raise Error("linear: input columns must equal weight rows")
        var act = _linear_act(activation)
        if divisor < 0 or (l2 != 0 and not (divisor > 0)):
            raise Error("linear: divisor must be positive when l2 is set, and never negative")
        var rows = a.rows
        var cols = w.cols
        var inner = a.cols
        self._shape(rows, cols)
        var parents = List[Tensor]()
        parents.append(x)
        parents.append(weight)
        if use_into:
            self._refuse_overlap(into, rows * cols, True, "linear_into: output")
        var output = into if use_into else self._alloc(ex, rows * cols)
        if bias:
            var bt = bias.value()
            var b_at = self._use(ex, bt)
            var b = self._nodes[b_at]
            if b.rows != 1 or b.cols != cols:
                raise Error("linear: bias must have shape (1, outputs)")
            parents.append(bt)
            # the GEMM, its bias and (not softmax) its activation: one launch
            _gemm_epi(ex, a.value, w.value, output, rows, cols, inner, inner, 1, cols, 1,
                      EPI_BIAS_ACT, b.value, 0 if act == ACT_SOFTMAX else act, Float32(0.0), Float32(0.0))
        else:
            _gemm(ex, output, a.value, w.value, rows, cols, inner, inner, 1, cols, 1)
            if act != ACT_IDENTITY and act != ACT_SOFTMAX:
                var f = Args()
                f.p0 = output
                f.i0 = act
                ex.launch[OP_ACT](f, rows * cols)
        if act == ACT_SOFTMAX:
            # in place, as the MLP trainer's _act_launch
            var s = Args()
            s.p0 = output
            s.p1 = output
            s.i0 = act
            s.i1 = cols
            ex.launch[OP_SOFTMAX](s, rows)
        var saved = Args()
        saved.i0 = act
        if bias:
            saved.i1 = 1
        saved.f0 = l2
        saved.f1 = divisor
        var shape = self._lead_shape(a_at, cols)  # rank-N metadata only
        var t = self._record(ex, output, rows, cols, LINEAR, parents, saved)
        return self._with_shape(t, shape)

    def transpose[E: Exec](mut self, mut ex: E, x: Tensor) raises -> Tensor:
        var n_at = self._use(ex, x)
        var n = self._nodes[n_at]
        var output = self._alloc(ex, n.rows * n.cols)
        comptime if AUTOGRAD_TRANSPOSE_VIEW:
            # a view: the copy waits for its first dense reader (_use);
            # matmul reads n's buffer through strides and never copies
            var saved = Args()
            saved.p0 = n.value
            saved.i0 = n.rows
            saved.i1 = n.cols
            saved.i9 = 1
            return self._record(ex, output, n.cols, n.rows, TRANSPOSE, [x], saved)
        else:
            _transpose(ex, output, n.value, n.rows, n.cols)
            return self._record(ex, output, n.cols, n.rows, TRANSPOSE, [x], Args())

    def _operand(self, i: Int) -> Tuple[FP, Int, Int, Int]:
        """matmul's view of node i: (buffer, row stride, column stride,
        1 when the node is a pending view read through strides).
        A dense node is (value, cols, 1, 0); a view of a rows x cols node
        P is (P's value, 1, rows, 1): element (r, c) is P's (c, r) word.
        A pending rank-N view (lane ag-rankn-core) is (its source, the
        strides _gemm_view gives, 1); matmul copies one _gemm_view refuses
        before it gets here."""
        var n = self._nodes[i]
        var one: Int = 1
        var zero: Int = 0
        if self._pending(i):
            if n.op != TRANSPOSE:
                var g = self._gemm_view(i)
                return (n.saved.p0, g[1], g[2], one)
            return (n.saved.p0, one, n.rows, one)
        return (n.value, n.cols, one, zero)

    def reshape[E: Exec](mut self, mut ex: E, x: Tensor, rows: Int, cols: Int) raises -> Tensor:
        self._shape(rows, cols)
        var n_at = self._use(ex, x)
        var n = self._nodes[n_at]
        if rows * cols != n.rows * n.cols:
            raise Error("reshape must preserve element count")
        return self._record(ex, n.value, rows, cols, RESHAPE, [x], Args())

    def detach[E: Exec](mut self, mut ex: E, x: Tensor) raises -> Tensor:
        """Create a constant view of x that stops gradient propagation."""
        var n_at = self._use(ex, x)
        var n = self._nodes[n_at]
        var shape = self._shapes[n_at]  # rank-N metadata only: x's shape
        var t = self.variable(ex, n.value, n.rows, n.cols, False)
        return self._with_shape(t, shape)

    def _reduce[E: Exec](mut self, mut ex: E, x: Tensor, axis: Int, mean: Bool) raises -> Tensor:
        if axis < -1 or axis > 1:
            raise Error("reduction axis must be -1 (all), 0, or 1")
        var n_at = self._use(ex, x)
        var n = self._nodes[n_at]
        var rows = n.rows if axis == 1 else 1
        var cols = n.cols if axis == 0 else 1
        var count = n.rows * n.cols if axis == -1 else (n.rows if axis == 0 else n.cols)
        var output = self._alloc(ex, rows * cols)
        var factor = identical_div(Float32(1), Float32(count)) if mean else Float32(1)
        var word = False
        comptime if AUTOGRAD_CURVE_WORD:
            # arm_fold_word: this fold's launch also stores the curve word
            word = self._word_armed and axis == AXIS_ALL
        if word:
            var partials = self._alloc(ex, fold_partials(n.rows, n.cols, axis))
            blocked_fold(ex, output, n.value, n.rows, n.cols, axis, factor, partials,
                         False, self._word_dst, self._word_factor, True)
            self._word_armed = False
        else:
            self._fold(ex, output, n.value, n.rows, n.cols, axis, factor)
        var a = Args()
        a.i0 = axis
        a.f0 = factor
        var out = self._record(ex, output, rows, cols, REDUCE, [x], a)
        if axis == AXIS_COLS:
            # rank-N metadata only: the leading dims and (1,) for rank >= 3
            _ = self._with_shape(out, self._lead_shape(n_at, 1))
        if word:
            self._word_node = out._index
        return out

    def _fold[E: Exec](
        mut self, mut ex: E, dst: FP, src: FP, rows: Int, cols: Int, axis: Int, factor: Float32,
    ) raises:
        """dst = factor * (src summed over `axis`) as a blocked fold: blocks of
        ag_fold_block(extent) consecutive terms, each added ascending from
        +0.0 (OP_AG_REDUCE_PART, parallel over blocks x outputs), then each
        output's block partials added ascending from +0.0 (OP_AG_REDUCE_FOLD,
        parallel over outputs). One order on every executor. The partials
        are tape scratch (released with the graph's storage at reset)."""
        # The launches live in autograd/fold.mojo (blocked_fold), shared with
        # the losses and metrics so every sum in autograd has one order.
        var partials = self._alloc(ex, fold_partials(rows, cols, axis))
        blocked_fold(ex, dst, src, rows, cols, axis, factor, partials)

    def sum[E: Exec](mut self, mut ex: E, x: Tensor, axis: Int = -1) raises -> Tensor:
        """Sum all elements or one matrix axis, retaining reduced dimensions."""
        return self._reduce(ex, x, axis, False)

    def mean[E: Exec](mut self, mut ex: E, x: Tensor, axis: Int = -1) raises -> Tensor:
        return self._reduce(ex, x, axis, True)

    def mse_loss[E: Exec](mut self, mut ex: E, prediction: Tensor, target: Tensor) raises -> Tensor:
        var difference = self.subtract(ex, prediction, target)
        var square = self.multiply(ex, difference, difference)
        return self.mean(ex, square)

    def cross_entropy[E: Exec](mut self, mut ex: E, logits: Tensor, targets: Tensor) raises -> Tensor:
        """Mean cross-entropy with dense class weights, one row per example."""
        var a_at = self._use(ex, logits)
        var a = self._nodes[a_at]
        var b_at = self._use(ex, targets)
        var b = self._nodes[b_at]
        if a.rows != b.rows or a.cols != b.cols:
            raise Error("cross-entropy targets must have the logits' shape")
        var logp = self.log_softmax(ex, logits)
        var weighted = self.multiply(ex, logp, targets)
        var rows = self.sum(ex, weighted, 1)
        var loss = self.mean(ex, rows)
        return self.scale(ex, loss, -1)

    def loss_terms[E: Exec](
        mut self, mut ex: E, prediction: Tensor, target: Tensor, kind: Int, delta: Float32 = 1,
    ) raises -> Tensor:
        """Per-element loss terms of two same-shape tensors, differentiable in
        both. kind is LS_MSE, LS_MAE, LS_HUBER (threshold `delta`) or
        LS_BCE_LOGITS (prediction holds logits) from autograd.loss_ops.
        autograd/losses.mojo reduces the terms with mean or sum."""
        if kind != LS_MSE and kind != LS_MAE and kind != LS_HUBER and kind != LS_BCE_LOGITS:
            raise Error("loss_terms takes LS_MSE, LS_MAE, LS_HUBER or LS_BCE_LOGITS")
        if kind == LS_HUBER and not (delta > 0):
            raise Error("Huber delta must be positive")
        var a_at = self._use(ex, prediction)
        var a = self._nodes[a_at]
        var b_at = self._use(ex, target)
        var b = self._nodes[b_at]
        if a.rows != b.rows or a.cols != b.cols:
            raise Error("loss target must have the prediction's shape")
        if Int(a.value) == Int(b.value):
            raise Error("loss prediction and target must be distinct tensors")
        var output = self._alloc(ex, a.rows * a.cols)
        var args = Args()
        args.p0 = a.value
        args.p1 = b.value
        args.p2 = output
        args.i0 = kind
        args.f0 = delta
        ex.launch[OP_LS_ELEM_FWD](args, a.rows * a.cols)
        var saved = Args()
        saved.i0 = kind
        saved.f0 = delta
        return self._record(ex, output, a.rows, a.cols, LOSS_ELEM, [prediction, target], saved)

    def loss_reduced[E: Exec](
        mut self, mut ex: E, prediction: Tensor, target: Tensor, kind: Int, delta: Float32 = 1,
        mean: Bool = True,
    ) raises -> Tensor:
        """The mean (or, mean=False, the sum) of loss_terms(prediction,
        target, kind, delta) as one (1, 1) node: the terms are folded in
        the leaf chain and never stored (autograd/fold_ops.mojo; lane
        ag-pass6-fold, AUTOGRAD_LOSS_REDUCED). Value and gradients equal
        mean / sum of loss_terms bit for bit."""
        if kind != LS_MSE and kind != LS_MAE and kind != LS_HUBER and kind != LS_BCE_LOGITS:
            raise Error("loss_reduced takes LS_MSE, LS_MAE, LS_HUBER or LS_BCE_LOGITS")
        if kind == LS_HUBER and not (delta > 0):
            raise Error("Huber delta must be positive")
        var a_at = self._use(ex, prediction)
        var a = self._nodes[a_at]
        var b_at = self._use(ex, target)
        var b = self._nodes[b_at]
        if a.rows != b.rows or a.cols != b.cols:
            raise Error("loss target must have the prediction's shape")
        if Int(a.value) == Int(b.value):
            raise Error("loss prediction and target must be distinct tensors")
        var count = a.rows * a.cols
        # Tape.mean's factor (and losses.reduction_factor's)
        var factor = identical_div(Float32(1), Float32(count)) if mean else Float32(1)
        var output = self._alloc(ex, 1)
        var partials = self._alloc(ex, fold_partials(a.rows, a.cols, AXIS_ALL))
        var word = False
        comptime if AUTOGRAD_CURVE_WORD:
            word = self._word_armed
        if word:
            loss_fold(ex, output, a.value, b.value, a.rows, a.cols, kind, delta, factor, partials,
                      self._word_dst, self._word_factor, True)
            self._word_armed = False
        else:
            loss_fold(ex, output, a.value, b.value, a.rows, a.cols, kind, delta, factor, partials)
        var saved = Args()
        saved.i0 = kind
        saved.f0 = delta
        saved.f1 = factor
        var out = self._record(ex, output, 1, 1, LOSS_REDUCED, [prediction, target], saved)
        if word:
            self._word_node = out._index
        return out

    def cross_entropy_rows[E: Exec](mut self, mut ex: E, logits: Tensor, codes: Tensor) raises -> Tensor:
        """Per-row softmax cross-entropy (rows, 1) of logits (rows, classes)
        against class codes (rows, 1) held as float values. The codes take
        no gradient; a code outside [0, classes) gives NaN."""
        var a_at = self._use(ex, logits)
        var a = self._nodes[a_at]
        var b_at = self._use(ex, codes)
        var b = self._nodes[b_at]
        if b.rows != a.rows or b.cols != 1:
            raise Error("cross-entropy codes must have shape (rows, 1)")
        if b.requires_grad:
            raise Error("cross-entropy codes cannot require a gradient")
        if Int(a.value) == Int(b.value):
            raise Error("cross-entropy logits and codes must be distinct tensors")
        var output = self._alloc(ex, a.rows)
        var args = Args()
        args.p0 = a.value
        args.p1 = b.value
        args.p2 = output
        args.i1 = a.cols
        ex.launch[OP_LS_CE_FWD](args, a.rows)
        return self._record(ex, output, a.rows, 1, LOSS_CE, [logits, codes], Args())

    def softmax_cross_entropy[E: Exec](mut self, mut ex: E, logits: Tensor, codes: Tensor) raises -> Tensor:
        """Mean softmax cross-entropy of logits (rows, classes) against class
        codes (rows, 1, the class index as a float value), as one node.

        The value and the logits' gradient equal those of
        cross_entropy_rows followed by mean, bit for bit; the backward is a
        single launch that writes (softmax - onehot) / rows into the
        logits' gradient. The codes take no gradient; a code outside
        [0, classes) gives NaN.

        Example::

            var logits = tape.linear(ex, h, w, b)
            var loss = tape.softmax_cross_entropy(ex, logits, codes)
            tape.backward(ex, loss)
        """
        # LAUNCHES (read from the code). cross_entropy_rows + mean: forward
        # 3 (OP_LS_CE_FWD, the two-launch blocked fold), backward 2
        # (OP_AG_REDUCE_VJP, OP_LS_CE_VJP). Tape.cross_entropy (dense
        # targets): forward 7, backward 5. This node: forward 3 (the same
        # row losses and the same fold; a mean in one launch would need a
        # serial or an unpinned reduction), backward 1 (OP_P1_SCE_VJP).
        var a_at = self._use(ex, logits)
        var a = self._nodes[a_at]
        var b_at = self._use(ex, codes)
        var b = self._nodes[b_at]
        if b.rows != a.rows or b.cols != 1:
            raise Error("cross-entropy codes must have shape (rows, 1)")
        if b.requires_grad:
            raise Error("cross-entropy codes cannot require a gradient")
        if Int(a.value) == Int(b.value):
            raise Error("cross-entropy logits and codes must be distinct tensors")
        var terms = self._alloc(ex, a.rows)
        var args = Args()
        args.p0 = a.value
        args.p1 = b.value
        args.p2 = terms
        args.i1 = a.cols
        ex.launch[OP_LS_CE_FWD](args, a.rows)
        var output = self._alloc(ex, 1)
        # Tape.mean's factor and fold over the (rows, 1) row losses
        var factor = identical_div(Float32(1), Float32(a.rows))
        self._fold(ex, output, terms, a.rows, 1, AXIS_ALL, factor)
        var saved = Args()
        saved.f0 = factor
        return self._record(ex, output, 1, 1, LOSS_SCE, [logits, codes], saved)

    def custom[E: Exec](
        mut self, mut ex: E, inputs: List[Tensor], output: FP,
        rows: Int, cols: Int, tag: Int, saved: Args,
    ) raises -> Tensor:
        """Record an already executed native operation with a custom backward tag.

        Keep output and saved buffers alive and unchanged through backward.
        Pass its BackwardRules implementation to backward_with_rules().
        """
        if tag < 0 or tag > (1 << 30):
            raise Error("custom operation tag is out of range")
        var args = saved
        # The tag is metadata; all Args slots remain available to the rule.
        var result = self._record(ex, output, rows, cols, CUSTOM, inputs, args)
        self._nodes[result._index].op = CUSTOM + tag
        return result

    def custom_into[E: Exec](
        mut self, mut ex: E, inputs: List[Tensor], output: FP, grad_destination: FP,
        rows: Int, cols: Int, tag: Int, saved: Args,
    ) raises -> Tensor:
        """Like custom(), with the operation's gradient held in caller storage.

        `grad_destination` holds rows * cols floats of this executor. Pass
        the same storage as the seed of backward_with_rules() and the seed
        is used in place, without a copy. Keep it unchanged through
        backward, and do not let it overlap any tensor of the graph other
        than that seed.
        """
        if tag < 0 or tag > (1 << 30):
            raise Error("custom operation tag is out of range")
        self._shape(rows, cols)
        self._refuse_overlap(output, rows * cols, True, "custom_into: output")
        self._refuse_overlap(grad_destination, rows * cols, True, "custom_into: grad_destination")
        if _ranges_overlap(Int(output), rows * cols, Int(grad_destination), rows * cols):
            raise Error("custom_into: grad_destination overlaps output")
        var args = saved
        var result = self._record(ex, output, rows, cols, CUSTOM, inputs, args, False, grad_destination, True)
        self._nodes[result._index].op = CUSTOM + tag
        return result

    def custom_into_shared[E: Exec](
        mut self, mut ex: E, inputs: List[Tensor], output: FP, shares: Tensor,
        rows: Int, cols: Int, tag: Int, saved: Args,
    ) raises -> Tensor:
        """Like custom_into(), with the operation's gradient held in the
        gradient storage of `shares`, one of its inputs.

        Use it for an operation whose vector-Jacobian product into `shares`
        is the identity, as a residual add's is into its branch: the two
        gradients are then one buffer and the backward copies nothing.
        `shares` must have this output's element count and hold its
        gradient in caller storage (custom_into, custom_into_shared or
        variable_into). Its rule must leave that destination as it is.
        Every other overlap is refused as in custom_into.
        """
        if tag < 0 or tag > (1 << 30):
            raise Error("custom operation tag is out of range")
        self._shape(rows, cols)
        var si = self._check(shares)
        var found = False
        for t in inputs:  # driver-loop: over node inputs
            if t._index == si:
                found = True
        if not found:
            raise Error("custom_into_shared: shares must be one of the inputs")
        var s = self._nodes[si]
        if self._recording and (not s.requires_grad or not s.caller_grad):
            raise Error("custom_into_shared: shares must hold its gradient in caller storage")
        if s.rows * s.cols != rows * cols:
            raise Error("custom_into_shared: shares must have the output's element count")
        var group = self._alias_root(si)
        self._refuse_overlap(output, rows * cols, True, "custom_into_shared: output")
        self._refuse_overlap(s.grad, rows * cols, True, "custom_into_shared: shared gradient", group)
        if _ranges_overlap(Int(output), rows * cols, Int(s.grad), rows * cols):
            raise Error("custom_into_shared: the shared gradient overlaps output")
        var args = saved
        var result = self._record(ex, output, rows, cols, CUSTOM, inputs, args, False, s.grad, True)
        self._nodes[result._index].op = CUSTOM + tag
        if self._recording:
            self._nodes[result._index].alias_of = group
        return result

    def binary_recorded[E: Exec](
        mut self, mut ex: E, x: Tensor, y: Tensor, op: Int, output: FP,
        grad_destination: FP = dummy_ptr(), caller_grad: Bool = False,
    ) raises -> Tensor:
        """Record an already executed add/subtract/multiply/divide.

        `output` already holds the broadcast result of x op y (computed by
        the caller's own kernel); nothing is launched. The backward is the
        tape's own binary VJP, which reads the operands' values (multiply,
        divide) and `output` (divide). With caller_grad=True the gradient
        lands in `grad_destination`, as in custom_into.
        """
        if op != ADD and op != SUBTRACT and op != MULTIPLY and op != DIVIDE:
            raise Error("binary_recorded takes ADD, SUBTRACT, MULTIPLY or DIVIDE")
        var a_at = self._use(ex, x)
        var a = self._nodes[a_at]
        var b_at = self._use(ex, y)
        var b = self._nodes[b_at]
        if (a.rows != b.rows and a.rows != 1 and b.rows != 1) or (a.cols != b.cols and a.cols != 1 and b.cols != 1):
            raise Error("incompatible tensor shapes for broadcasting")
        var rows = max(a.rows, b.rows)
        var cols = max(a.cols, b.cols)
        self._shape(rows, cols)
        self._refuse_overlap(output, rows * cols, True, "binary_recorded: output")
        if caller_grad:
            self._refuse_overlap(grad_destination, rows * cols, True, "binary_recorded: grad_destination")
            if _ranges_overlap(Int(output), rows * cols, Int(grad_destination), rows * cols):
                raise Error("binary_recorded: grad_destination overlaps output")
        return self._record(ex, output, rows, cols, op, [x, y], Args(), False,
                            grad_destination, caller_grad)

    def _binary_vjp[E: Exec](
        mut self, mut ex: E, n: _Node, edge: Int, contribution: FP, rows: Int, cols: Int,
    ) raises:
        """Operand `edge`'s VJP of binary node n into `contribution` (the
        operand's shape rows x cols). The elementwise VJP is computed at the
        output's shape; a broadcast operand's gradient is that matrix summed
        over the broadcast axis (or all of it, for a scalar operand) by the
        blocked fold (_fold), never one thread per gradient element."""
        var left = self._nodes[self._parents[n.start]]
        var right = self._nodes[self._parents[n.start + 1]]
        var a = Args()
        a.i1 = n.op
        a.i2 = n.rows
        a.i3 = n.cols
        a.i4 = left.rows
        a.i5 = left.cols
        a.i6 = right.rows
        a.i7 = right.cols
        a.i8 = edge
        a.p3 = n.grad
        # Bind only the slots the kind reads (op_ag_bin_vjp), each buffer once.
        if n.op == MULTIPLY:
            a.p0 = left.value
            if Int(right.value) == Int(left.value):
                a.i10 = 1
            else:
                a.p1 = right.value
        elif n.op == DIVIDE:
            a.p1 = right.value
            if edge == 1:
                a.p2 = n.value
        var reduce_rows = rows < n.rows
        var reduce_cols = cols < n.cols
        if not reduce_rows and not reduce_cols:
            a.p4 = contribution
            ex.launch[OP_AG_BIN_VJP](a, n.rows * n.cols)
            return
        var axis = AXIS_ALL if reduce_rows and reduce_cols else (AXIS_ROWS if reduce_rows else AXIS_COLS)
        if n.op == ADD or (n.op == SUBTRACT and edge == 0):
            # The elementwise VJP is the upstream itself (ftz(g), stored
            # flushed), and the fold flushes each term on load, so folding
            # n.grad directly gives the same bits without the elementwise
            # launch or its scratch. (Not the subtraction's right operand:
            # -1 * fold(g) and fold(-g) differ in the sign of a zero sum.)
            self._fold(ex, contribution, n.grad, n.rows, n.cols, axis, Float32(1))
            return
        # Lane ag-pass2-kernels: the VJP terms go straight into the blocked
        # fold's partials (OP_AGK_BIN_VJP_PART), the same terms in the same
        # order as OP_AG_BIN_VJP into an output-sized scratch then
        # OP_AG_REDUCE_PART: one launch and one write + read of the
        # rows x cols matrix fewer, the same bits.
        var block = fold_block(n.rows, n.cols, axis)
        a.i0 = axis
        a.i9 = block
        var extent = n.rows * n.cols if axis == AXIS_ALL else (n.rows if axis == AXIS_ROWS else n.cols)
        var outputs = 1 if axis == AXIS_ALL else (n.cols if axis == AXIS_ROWS else n.rows)
        if fold_single_launch(n.rows, n.cols, axis):
            # Lane ag-pass3-fold (AUTOGRAD_FOLD_V2): one block an output, so
            # the one launch writes the contribution (+0.0 + the chain, the
            # fold's one-partial chain at factor 1): the same bits, no
            # partials and no second launch. p4 is unused: left unbound.
            a.p5 = contribution
            a.i11 = 1
            ex.launch[OP_AGK_BIN_VJP_PART](a, outputs)
            return
        var partials = self._alloc(ex, fold_partials(n.rows, n.cols, axis))
        a.p4 = partials
        ex.launch[OP_AGK_BIN_VJP_PART](a, ((extent + block - 1) // block) * outputs)
        blocked_fold_finish(ex, contribution, partials, n.rows, n.cols, axis, Float32(1))

    def backward[E: Exec](mut self, mut ex: E, output: Tensor, accumulate: Bool = False) raises:
        """Differentiate a scalar, clearing gradients unless accumulate=True."""
        self.backward_scalar_with_rules(ex, output, NoCustomRules(), accumulate)

    def backward_scalar_with_rules[E: Exec, R: BackwardRules](
        mut self, mut ex: E, output: Tensor, rules: R, accumulate: Bool = False,
    ) raises:
        """Differentiate a scalar using native custom backward rules."""
        var n = self._nodes[self._check(output)]
        if n.inferred:
            raise Error(_INFERRED_REFUSAL)
        if n.rows * n.cols != 1:
            raise Error("non-scalar backward requires an explicit seed tensor")
        self._refuse_stale(output._index)
        self._profile_phase(PROFILE_OTHER)
        if not n.requires_grad:
            raise Error("output has no differentiable ancestors")
        var cursor = self._cursor
        # The seed 1 is written straight into the root's own gradient (one
        # launch, no seed copy) unless the root is a leaf whose gradient
        # accumulates, which must add the 1 to what it holds.
        var seed: FP
        if accumulate and n.op == LEAF:
            seed = self._alloc(ex, 1)
        else:
            # AUTOGRAD_GRAD_ALIAS: the root's own buffer, never one it
            # borrowed in an earlier backward (a caller's seed)
            seed = n.own_grad if AUTOGRAD_GRAD_ALIAS else n.grad
        _fill(ex, seed, 1, 1)
        try:
            self._backward(ex, output, seed, rules, accumulate)
        except e:
            self._cursor = cursor
            raise e
        self._cursor = cursor
        self._profile_phase(PROFILE_BACKWARD)

    def backward_from[E: Exec](
        mut self, mut ex: E, output: Tensor, seed: Tensor, accumulate: Bool = False,
    ) raises:
        """Compute a vector-Jacobian product using a seed of the output's shape.

        The seed may be read in place rather than copied: keep its storage
        unchanged until you have read the gradients of this backward. The
        output's gradient is then the seed's storage itself.
        """
        self.backward_with_rules(ex, output, seed, NoCustomRules(), accumulate)

    def backward_from_preactivation[E: Exec](
        mut self, mut ex: E, output: Tensor, seed: Tensor, accumulate: Bool = False,
    ) raises:
        """Backpropagate a seed that is the gradient with respect to a linear
        output's pre-activation (x @ W + b), not its activated output.

        Use it when the loss derivative already includes the output
        activation, as the softmax or sigmoid with log loss derivative
        p - y does. `output` comes from linear or linear_into and `seed` has
        its shape. The seed is read in place (no copy); keep it unchanged
        until backward finishes. The output's own gradient is not stored.
        """
        self.backward_from_preactivation_with_rules(ex, output, seed, NoCustomRules(), accumulate)

    def backward_from_preactivation_with_rules[E: Exec, R: BackwardRules](
        mut self, mut ex: E, output: Tensor, seed: Tensor, rules: R, accumulate: Bool = False,
    ) raises:
        var n = self._nodes[self._check(output)]
        var s_at = self._use(ex, seed)
        var s = self._nodes[s_at]
        if n.op != LINEAR:
            raise Error("backward_from_preactivation takes an output of linear or linear_into")
        if n.rows != s.rows or n.cols != s.cols:
            raise Error("backward seed shape must equal output shape")
        self._refuse_stale(output._index)
        if n.requires_grad:
            self._refuse_seed_overlap(output._index, s.value, accumulate)
        self._profile_phase(PROFILE_OTHER)
        var cursor = self._cursor
        try:
            self._backward(ex, output, s.value, rules, accumulate, True)
        except e:
            self._cursor = cursor
            raise e
        self._cursor = cursor
        self._profile_phase(PROFILE_BACKWARD)

    def _linear_dz[E: Exec](mut self, mut ex: E, i: Int) raises -> FP:
        """dL/dZ of linear node i from its output gradient dY: dY itself
        without an activation; act'(Y) applied in place (OP_ACT_BWD, the
        hand path's body) for an elementwise one; the softmax VJP into
        backward scratch for softmax."""
        var n = self._nodes[i]
        var act = n.saved.i0
        if act == ACT_IDENTITY:
            return n.grad
        if act == ACT_SOFTMAX:
            var d = self._alloc(ex, n.rows * n.cols)
            var s = Args()
            s.p0 = n.value
            s.p1 = n.grad
            s.p2 = d
            s.i0 = 0
            s.i1 = n.cols
            ex.launch[OP_AG_SOFTMAX_VJP](s, n.rows)
            return d
        var a = Args()
        a.p0 = n.grad
        a.p1 = n.value
        a.i0 = act
        ex.launch[OP_ACT_BWD](a, n.rows * n.cols)
        self._nodes[i].grad_preact = True
        return n.grad

    def _linear_vjp[E: Exec](
        mut self, mut ex: E, n: _Node, edge: Int, dst: FP, first: Bool, dz: FP, consumers: List[Int],
        db_dst: FP, fuse_db: Bool,
    ) raises:
        """Edge `edge` of linear node n into `dst` (the parent's gradient
        for a first contribution, else scratch), from dz = dL/dZ. The calls
        and arguments are the hand MLP backward's (sequence/mlp_fit.mojo).
        fuse_db (AUTOGRAD_PASS3_DB): edge 1's launch also writes the bias
        gradient into db_dst (edge 2's destination), and edge 2 launches
        nothing."""
        var xi = self._parents[n.start]
        var x = self._nodes[xi]
        var w = self._nodes[self._parents[n.start + 1]]
        var rows = n.rows
        var fi = x.cols
        var fo = n.cols
        var divisor = n.saved.f1
        if edge == 1:
            # dW = x^T dz (+ l2 W) / divisor
            if divisor > 0:
                # (fuse_db: + db = colsum(dz) / divisor from the same launch)
                _gemm_epi(ex, x.value, dz, dst, fi, fo, rows, 1, fi, fo, 1,
                          EPI_L2GRAD, w.value, 0, n.saved.f0, divisor, db_dst, fuse_db)
            else:
                _gemm(ex, dst, x.value, dz, fi, fo, rows, 1, fi, fo, 1)
        elif edge == 2:
            if fuse_db:
                return  # written by edge 1's launch
            # db = column sum of dz over the rows (/ divisor)
            var blocked = False
            comptime if AUTOGRAD_PASS2_FOLD:
                # More rows than one fold block: the blocked fold (chains of
                # ag_fold_block(rows), parallel over blocks x columns), not
                # one serial chain over every row a column. One block is the
                # column sum's chain, so the one-launch column sum stays.
                blocked = rows > ag_fold_block(rows)
            if blocked:
                var partials = self._alloc(ex, fold_partials(rows, fo, AXIS_ROWS))
                blocked_fold(ex, dst, dz, rows, fo, AXIS_ROWS, divisor if divisor > 0 else Float32(1),
                             partials, divisor > 0)
            else:
                var c = Args()
                c.p0 = dz
                c.p1 = dst
                c.i0 = rows
                c.i2 = fo
                if divisor > 0:
                    c.f0 = divisor
                    ex.launch[OP_COLSUM_DIV](c, fo)
                else:
                    ex.launch[OP_COLSUM](c, fo)
        else:
            # dX = dz W^T. When x is a linear node whose only consumer is
            # this node, its activation derivative rides in the epilogue
            # and x's gradient holds its pre-activation gradient: the hand
            # path's EPI_ACT_BWD (act' read from x's output values).
            var x_act = x.saved.i0
            if first and consumers[xi] == 1 and x.op == LINEAR and x_act != ACT_SOFTMAX:
                _gemm_epi(ex, dz, w.value, dst, rows, fi, fo, fo, 1, 1, fo,
                          EPI_ACT_BWD, x.value, x_act, Float32(0.0), Float32(0.0))
                if x_act != ACT_IDENTITY:
                    self._nodes[xi].grad_preact = True
            else:
                _gemm(ex, dst, dz, w.value, rows, fi, fo, fo, 1, 1, fo)

    def _borrows(self, p: _Node) -> Bool:
        """AUTOGRAD_GRAD_ALIAS: whether node p may take a pass-through
        gradient buffer (a consumer's, or the seed) as its own: a recorded,
        tape-owned intermediate whose backward never writes its own gradient
        (not a leaf, not linear, which applies act' in place, not custom,
        not in an alias group)."""
        return (p.requires_grad and not p.inferred and p.op != LEAF and p.op < CUSTOM and p.op != LINEAR
                and not p.caller_grad and p.alias_of < 0)

    def _pair_backward[E: Exec](
        mut self, mut ex: E, n: _Node, dz: FP, contributions: List[FP], first: List[Bool],
        consumers: List[Int], mut paired: List[Bool],
    ) raises:
        """AUTOGRAD_PAIR: linear or matmul node n's dX, dW and (fused db)
        bias gradient in one OP_AGC_LINEAR_PAIR launch into the edges'
        destinations, with the arguments _linear_vjp (or the matmul VJP)
        gives each launch; marks the edges written in `paired`. Launches
        nothing where the rule in the AUTOGRAD_PAIR comment says no."""
        var xi = self._parents[n.start]
        var x = self._nodes[xi]
        var w = self._nodes[self._parents[n.start + 1]]
        if not x.requires_grad or not w.requires_grad:
            return
        if Int(x.value) == Int(w.value):
            return  # x @ x: one buffer would fill two slots of the launch
        var rows = n.rows
        var fi = x.cols
        var fo = n.cols
        var c0 = contributions[0]
        var c1 = contributions[1]
        if Int(c0) == Int(c1) or Int(c0) == Int(dz) or Int(c1) == Int(dz):
            return
        var dx_epi = 0
        var x_act = 0
        var dw_epi = 0
        var db = False
        var alpha = Float32(0.0)
        var divisor = Float32(0.0)
        if n.op == LINEAR:
            var xa = x.saved.i0
            # _linear_vjp edge 0's epilogue rule
            if first[0] and consumers[xi] == 1 and x.op == LINEAR and xa != ACT_SOFTMAX:
                dx_epi = EPI_ACT_BWD
                x_act = xa
            if n.saved.f1 > 0:
                divisor = n.saved.f1
                alpha = n.saved.f0
                dw_epi = EPI_L2GRAD
            # _backward's fuse_db rule (W's gradient is wanted, checked above)
            comptime if AUTOGRAD_PASS3_DB:
                db = (n.count == 3 and n.saved.i1 == 1 and n.saved.f1 > 0
                      and self._nodes[self._parents[n.start + 2]].requires_grad)
        # The route rule (AUTOGRAD_PAIR comment): neither product on the tile,
        # and no few-cell long-K weight product.
        if seq_gemm_route(rows, fi, fo, dx_epi != 0) == GR_TILE:
            return
        if seq_gemm_route(fi, fo, rows, dw_epi != 0 or db) == GR_TILE:
            return
        if fi * fo <= 1024 and rows >= 32768:
            return
        var a = Args()
        a.p0 = x.value
        a.p1 = w.value
        a.p2 = dz
        a.p3 = c0
        a.p4 = c1
        a.i0 = rows
        a.i1 = fi
        a.i2 = fo
        a.i3 = dx_epi
        a.i4 = x_act
        a.i5 = dw_epi
        a.f0 = alpha
        a.f1 = divisor
        var threads = rows * fi + fi * fo
        if db:
            a.p5 = contributions[2]
            a.i6 = 1
            threads += fo
        ex.launch[OP_AGC_LINEAR_PAIR](a, threads)
        if dx_epi != 0 and x_act != ACT_IDENTITY:
            self._nodes[xi].grad_preact = True
        paired[0] = True
        paired[1] = True
        if db:
            paired[2] = True

    def backward_with_rules[E: Exec, R: BackwardRules](
        mut self, mut ex: E, output: Tensor, seed: Tensor, rules: R, accumulate: Bool = False,
    ) raises:
        var n = self._nodes[self._check(output)]
        var s_at = self._use(ex, seed)
        var s = self._nodes[s_at]
        if n.inferred:
            raise Error(_INFERRED_REFUSAL)
        if n.rows != s.rows or n.cols != s.cols:
            raise Error("backward seed shape must equal output shape")
        self._refuse_stale(output._index)
        if n.requires_grad:
            self._refuse_seed_overlap(output._index, s.value, accumulate)
        self._profile_phase(PROFILE_OTHER)
        var cursor = self._cursor
        try:
            self._backward(ex, output, s.value, rules, accumulate)
        except e:
            self._cursor = cursor
            raise e
        self._cursor = cursor
        self._profile_phase(PROFILE_BACKWARD)

    def _backward[E: Exec, R: BackwardRules](
        mut self, mut ex: E, output: Tensor, seed: FP, rules: R, accumulate: Bool,
        preact: Bool = False,
    ) raises:
        var root = self._check(output)
        if self._nodes[root].inferred:
            raise Error(_INFERRED_REFUSAL)
        if not self._nodes[root].requires_grad:
            raise Error("output has no differentiable ancestors")
        var active = List[Bool](length=len(self._nodes), fill=False)
        # Edges from differentiated nodes into each node (a linear node's
        # dX epilogue may apply its input's act' only for a sole consumer).
        var consumers = List[Int](length=len(self._nodes), fill=0)
        active[root] = True
        # Creation order is topological. Mark only ancestors of this output;
        # disconnected invalid-domain operations are never differentiated.
        for reverse in range(root + 1):
            var i = root - reverse
            if not active[i]:
                continue
            var n = self._nodes[i]
            if n.op >= CUSTOM and not rules.supports(n.op - CUSTOM):
                raise Error("custom backward rule not registered")
            for edge in range(n.count):
                var p = self._parents[n.start + edge]
                if self._nodes[p].requires_grad:
                    active[p] = True
                    consumers[p] += 1
        # GRADIENT WRITE INVARIANT (lane ag-fusion). Gradients are never
        # zero-filled before use. touched[i] says node i's gradient already
        # holds a contribution. The FIRST contribution to a parent is written
        # by the VJP (or matmul GEMM) directly into the parent's grad; only
        # later contributions go to scratch and are added with ACCUMULATE in
        # the same order as before. That is bit for bit the old zero,
        # copy-first, add-the-rest schedule. A VJP reads parent VALUES and
        # the node's own grad, never a parent's grad, so writing the parent's
        # grad in place is safe. Leaves start touched only when
        # accumulate=True and they already hold a gradient (a leaf never
        # written before takes its first contribution in place, so where
        # that contribution is -0.0 it keeps the sign the old 0 + c fill
        # dropped). After the traversal an unreached differentiable leaf is
        # zeroed (gradient 0); an unreached intermediate has no gradient
        # (gradient() refuses).
        #
        # LAUNCHES THE TRAVERSAL ADDS (lane ag-pass1-launches, from the code;
        # the operations' own VJP launches are listed at each operation).
        # Before this pass, per backward and per node:
        #   seed: scalar backward fill + copy (2); explicit seed copy 1
        #   custom node: 1 zero fill per input (zeroed_destination True in
        #     every rule set but the recurrent one)
        #   later contribution to an input: scratch VJP + 1 ACCUMULATE
        #     (a same-shape add: its 1 copy VJP + 1 ACCUMULATE)
        #   broadcast add operand: elementwise VJP 1 + fold 2
        #   reshape, first contribution: 1 copy
        #   accumulate=True leaf never written before: 1 fill
        # After:
        #   seed: scalar backward 1 fill into the root's gradient (none
        #     copied); explicit seed copy 1, 0 when the root's gradient is
        #     the seed's storage (custom_into / variable_into)
        #   custom node: a fill only where the rule asks for one (the
        #     transformer, Mamba-1 and Mamba-2 rules ask for none); 0
        #     scratch and 0 ACCUMULATE for a writes_total input
        #   later contribution: same-shape add 1 ACCUMULATE straight from
        #     the upstream (no VJP launch); others unchanged
        #   broadcast add operand: fold 2 (the upstream folded directly)
        #   reshape, first contribution: 0 when the operand is consumed only
        #     there and its gradient is tape storage (it shares the buffer)
        #   accumulate=True leaf never written before: 0
        var touched = List[Bool](length=len(self._nodes), fill=False)
        for i in range(len(self._nodes)):  # driver-loop: over graph nodes
            self._nodes[i].grad_preact = False
            comptime if AUTOGRAD_GRAD_ALIAS:
                # every node back on its own buffer: a buffer borrowed by an
                # earlier backward (a reshape's, an add's, a caller's seed)
                # is never written by this one
                self._nodes[i].grad = self._nodes[i].own_grad
            var n = self._nodes[i]
            if not n.requires_grad:
                continue
            if accumulate and n.op == LEAF:
                # A leaf never written before holds no value to add to: it
                # stays untouched, so its first contribution is written in
                # place like any other (no fill before use) and, if no path
                # reaches it, the zero fill after the traversal gives it 0.
                if n.grad_ready:
                    touched[i] = True
            else:
                self._nodes[i].grad_ready = False
        # Preserve the seed before overwriting a leaf root's gradient. A
        # pre-activation seed (backward_from_preactivation) is read in place
        # as the root linear node's dZ: no copy.
        if preact:
            pass
        elif touched[root]:
            _accumulate(ex, self._nodes[root].grad, seed, self._nodes[root].rows * self._nodes[root].cols)
        elif Int(self._nodes[root].grad) != Int(seed):
            if AUTOGRAD_GRAD_ALIAS and self._borrows(self._nodes[root]):
                # the root borrows the seed: no copy (the seed is read in
                # place; nothing in this backward writes the root's gradient)
                self._nodes[root].grad = seed
            else:
                _copy(ex, self._nodes[root].grad, seed, self._nodes[root].rows * self._nodes[root].cols)
        touched[root] = True
        # ACCUMULATION ORDER (docs/plans/NATIVE_AUTOGRAD.md, "Accumulation
        # order"): contributions to a gradient are applied in decreasing
        # consumer index, the inputs of one consumer in increasing edge
        # order; the first is written, later ones added. last_node/last_edge
        # hold each gradient's previous contribution; a violation refuses.
        var last_node = List[Int](length=len(self._nodes), fill=-1)
        var last_edge = List[Int](length=len(self._nodes), fill=-1)
        var scratch_start = self._cursor
        for reverse in range(root + 1):  # driver-loop: over graph nodes, reverse creation order
            var i = root - reverse
            if not active[i]:
                continue
            var n = self._nodes[i]
            if n.op == LEAF:
                continue
            self._cursor = scratch_start
            # Destinations: the parent's own grad for a first contribution,
            # scratch for a later one (and, for a custom rule, for an input
            # that needs no gradient: the rule writes every destination).
            var first = List[Bool](length=n.count, fill=False)
            var total = List[Bool](length=n.count, fill=False)
            # AUTOGRAD_GRAD_ALIAS: the operand borrowed n's gradient (no VJP launch)
            var borrowed = List[Bool](length=n.count, fill=False)
            # EMBEDDING: the table's gradient already holds a value (the fold continues from it)
            var emb_carry = False
            var contributions = List[FP]()
            for edge in range(n.count):  # driver-loop: over node inputs
                var parent = self._parents[n.start + edge]
                var p = self._nodes[parent]
                if p.requires_grad and n.op == EMBEDDING:
                    # the scatter-add writes the table's whole gradient in
                    # place: from +0.0 for a first contribution, else
                    # continuing from the held value (Tape.embedding)
                    emb_carry = touched[parent]
                    first[edge] = True
                    total[edge] = True
                    touched[parent] = True
                    contributions.append(p.grad)
                elif p.requires_grad and n.op >= CUSTOM and rules.writes_total(n.op - CUSTOM, n.saved, edge):
                    # the rule writes this input's whole gradient in place
                    first[edge] = True
                    total[edge] = True
                    touched[parent] = True
                    contributions.append(p.grad)
                elif p.requires_grad and not touched[parent]:
                    first[edge] = True
                    touched[parent] = True
                    if (n.op == RESHAPE and consumers[parent] == 1 and p.op != LEAF and p.op < CUSTOM
                            and p.op != LINEAR and not p.caller_grad):
                        # A reshape's only-consumed, tape-owned operand takes
                        # the reshape's gradient buffer as its own: the copy
                        # it replaces is a copy of bits, and no later write
                        # reaches either (one consumer; its backward never
                        # writes its own gradient, unlike LINEAR's act' in
                        # place). No launch.
                        self._nodes[parent].grad = n.grad
                    elif (AUTOGRAD_GRAD_ALIAS and _passes_upstream(n, edge, p) and consumers[parent] == 1
                            and not n.caller_grad and self._borrows(p)):
                        # a same-shape add's operand (or a subtraction's
                        # left one) consumed only here takes the add's
                        # gradient buffer: its VJP is that word, flushed,
                        # and every reader loads through ld (ftz). No launch.
                        self._nodes[parent].grad = n.grad
                        borrowed[edge] = True
                    contributions.append(self._nodes[parent].grad)
                elif p.requires_grad and (n.op == RESHAPE or _passes_upstream(n, edge, p)):
                    # a reshape's VJP, and a same-shape add's, is its upstream
                    # gradient itself: add it straight from n.grad, no scratch
                    # and no VJP launch (ACCUMULATE flushes on load exactly as
                    # the add's VJP would have before storing)
                    contributions.append(n.grad)
                elif p.requires_grad or n.op >= CUSTOM:
                    contributions.append(self._alloc(ex, p.rows * p.cols))
                else:
                    contributions.append(dummy_ptr())
            # A linear node's dL/dZ: the pre-activation seed at the root, its
            # gradient when a consumer already applied act', else formed here.
            var dz = n.grad
            if n.op == LINEAR:
                if i == root and preact:
                    dz = seed
                elif not n.grad_preact:
                    dz = self._linear_dz(ex, i)
            if n.op >= CUSTOM:
                var inputs = List[FP]()
                var rows = List[Int]()
                var cols = List[Int]()
                for edge in range(n.count):  # driver-loop: over node inputs
                    var p = self._nodes[self._parents[n.start + edge]]
                    inputs.append(p.value)
                    rows.append(p.rows)
                    cols.append(p.cols)
                    # BackwardRules receive distinct destinations, zeroed
                    # unless the rule overwrites (or ignores) that one, or
                    # writes the input's whole gradient there (writes_total:
                    # it may already hold other consumers' terms).
                    if not total[edge] and rules.zeroed_destination(n.op - CUSTOM, n.saved, edge):
                        _fill(ex, contributions[edge], p.rows * p.cols)
                rules.apply(ex, n.op - CUSTOM, n.saved, n.value, n.grad, inputs, contributions, rows, cols)
            # AUTOGRAD_PAIR: the edges a paired launch already wrote
            var paired = List[Bool](length=n.count, fill=False)
            comptime if AUTOGRAD_PAIR:
                # a matmul that read a transpose view through strides keeps
                # its two GEMMs (the paired launch reads dense operands)
                if n.op == LINEAR or (n.op == MATMUL and n.saved.i0 == 0 and n.saved.i1 == 0):
                    self._pair_backward(ex, n, dz, contributions, first, consumers, paired)
            for edge in range(n.count):  # driver-loop: over node inputs
                var parent = self._parents[n.start + edge]
                var p = self._nodes[parent]
                if not p.requires_grad:
                    continue
                # contribution is p.grad itself for a first contribution
                # (every body below, including _binary_vjp's blocked fold,
                # then writes the parent's gradient directly), else scratch.
                var contribution = contributions[edge]
                if n.op == RESHAPE:
                    # (no copy when the operand took this buffer above)
                    if first[edge] and Int(contribution) != Int(n.grad):
                        _copy(ex, contribution, n.grad, p.rows * p.cols)
                elif not first[edge] and _passes_upstream(n, edge, p):
                    pass  # the accumulate below adds n.grad itself
                elif borrowed[edge] or paired[edge]:
                    pass  # the operand holds n's gradient itself / the paired launch wrote it
                elif n.op < CUSTOM:
                    if n.op == LINEAR:
                        # AUTOGRAD_PASS3_DB: with a bias, a divisor and both
                        # parameter gradients wanted, db rides in dW's launch
                        var fuse_db = False
                        comptime if AUTOGRAD_PASS3_DB:
                            fuse_db = (n.count == 3 and n.saved.i1 == 1 and n.saved.f1 > 0
                                       and self._nodes[self._parents[n.start + 1]].requires_grad
                                       and self._nodes[self._parents[n.start + 2]].requires_grad)
                        var db_dst = contributions[2] if fuse_db else dummy_ptr()
                        self._linear_vjp(ex, n, edge, contribution, first[edge], dz, consumers, db_dst, fuse_db)
                    elif n.op == MATMUL:
                        var left = self._nodes[self._parents[n.start]]
                        var right = self._nodes[self._parents[n.start + 1]]
                        if n.saved.i6 == 1:
                            # AUTOGRAD_TRANSPOSE_VIEW: the forward's buffers and
                            # strides (saved by matmul); dense operands give the
                            # arguments below exactly
                            if edge == 0:
                                # dX = G @ Y^T: Y^T(k, n) = Y(n, k)
                                _gemm(ex, contribution, n.grad, n.saved.p1, left.rows, left.cols, right.cols,
                                      right.cols, 1, n.saved.i5, n.saved.i4)
                            else:
                                # dY = X^T @ G: X^T(m, k) = X(k, m)
                                _gemm(ex, contribution, n.saved.p0, n.grad, right.rows, right.cols, left.rows,
                                      n.saved.i3, n.saved.i2, right.cols, 1)
                        elif edge == 0:
                            _gemm(ex, contribution, n.grad, right.value, left.rows, left.cols, right.cols,
                                  right.cols, 1, 1, right.cols)
                        else:
                            _gemm(ex, contribution, left.value, n.grad, right.rows, right.cols, left.rows,
                                  1, left.cols, right.cols, 1)
                    elif n.op <= DIVIDE:
                        self._binary_vjp(ex, n, edge, contribution, p.rows, p.cols)
                    elif n.op == TRANSPOSE:
                        # n is (p.cols, p.rows); its upstream transposes back.
                        _transpose(ex, contribution, n.grad, n.rows, n.cols)
                    elif n.op == PERMUTE_N:
                        # lane ag-rankn-core: the upstream read through the
                        # inverse permutation (saved.p1, a view record over
                        # p's shape) into p's gradient; every word written once
                        rn_copy_strided(ex, contribution, n.grad, 0, n.saved.p1, 0, p.rows * p.cols)
                    elif n.op == REDUCE:
                        var a = Args()
                        a.p0 = n.grad
                        a.p1 = contribution
                        a.i0 = n.saved.i0
                        a.i2 = p.cols
                        a.f0 = n.saved.f0
                        ex.launch[OP_AG_REDUCE_VJP](a, p.rows * p.cols)
                    elif n.op == DROPOUT:
                        # The mask is re-derived from the saved (seed,
                        # counter) slots, never stored (autograd/data_ops.mojo).
                        var a = n.saved
                        a.p0 = n.grad
                        a.p1 = contribution
                        ex.launch[OP_AG_DROPOUT_VJP](a, p.rows * p.cols)
                    elif n.op == LOSS_ELEM:
                        var a = Args()
                        a.p0 = self._nodes[self._parents[n.start]].value
                        a.p1 = self._nodes[self._parents[n.start + 1]].value
                        a.p2 = n.grad
                        a.p3 = contribution
                        a.i0 = n.saved.i0
                        a.i8 = edge
                        a.f0 = n.saved.f0
                        ex.launch[OP_LS_ELEM_VJP](a, p.rows * p.cols)
                    elif n.op == LOSS_REDUCED:
                        # loss_reduced: REDUCE_VJP's word in a register, then
                        # OP_LS_ELEM_VJP's body (autograd/fold_ops.mojo)
                        var a = Args()
                        a.p0 = self._nodes[self._parents[n.start]].value
                        a.p1 = self._nodes[self._parents[n.start + 1]].value
                        a.p2 = n.grad
                        a.p3 = contribution
                        a.i0 = n.saved.i0
                        a.i8 = edge
                        a.f0 = n.saved.f0
                        a.f1 = n.saved.f1
                        ex.launch[OP_AGS_LOSS_VJP](a, p.rows * p.cols)
                    elif n.op == LOSS_CE:
                        # only the logits (edge 0): the codes never require a gradient
                        var a = Args()
                        a.p0 = p.value
                        a.p1 = self._nodes[self._parents[n.start + 1]].value
                        a.p2 = n.grad
                        a.p3 = contribution
                        a.i1 = p.cols
                        ex.launch[OP_LS_CE_VJP](a, p.rows)
                    elif n.op == LOSS_SCE:
                        # the logits only (edge 0), in one launch: the mean's
                        # factor rides in the row body (autograd/pass1_ops.mojo)
                        var a = Args()
                        a.p0 = p.value
                        a.p1 = self._nodes[self._parents[n.start + 1]].value
                        a.p2 = n.grad
                        a.p3 = contribution
                        a.i1 = p.cols
                        a.f0 = n.saved.f0
                        ex.launch[OP_P1_SCE_VJP](a, p.rows)
                    elif n.op == EMBEDDING:
                        # the run-sorted scatter-add (autograd/embedding_ops.mojo);
                        # its run scratch is this node's backward scratch
                        var e_n = n.saved.i0
                        var e_vocab = n.saved.i1
                        var e_plan = embedding_plan(n.saved.i4, e_vocab, e_n)
                        var e_begin = self._alloc(ex, e_vocab + 1)
                        var e_perm = self._alloc(ex, embedding_perm_words(e_plan, e_n))
                        var e_keys = dummy_ptr()
                        if e_plan == AG_EMB_PLAN_SORT:
                            e_keys = self._alloc(ex, embedding_perm_words(e_plan, e_n))
                        embedding_backward_into(
                            ex, n.saved.p0, n.grad, contribution, e_n, e_vocab, n.saved.i2, n.saved.i3,
                            emb_carry, e_plan, e_begin, e_perm, e_keys,
                        )
                    elif n.op == SOFTMAX or n.op == LOG_SOFTMAX:
                        var a = Args()
                        a.p0 = n.value
                        a.p1 = n.grad
                        a.p2 = contribution
                        a.i0 = 1 if n.op == LOG_SOFTMAX else 0
                        a.i1 = p.cols
                        ex.launch[OP_AG_SOFTMAX_VJP](a, p.rows)
                    else:
                        var a = Args()
                        if unary_vjp_reads_input(n.op):
                            a.p0 = p.value
                        if unary_vjp_reads_output(n.op):
                            a.p1 = n.value
                        a.p2 = n.grad
                        a.p3 = contribution
                        a.i0 = n.op
                        a.f0 = n.saved.f0
                        ex.launch[OP_AG_UNARY_VJP](a, p.rows * p.cols)
                # a writes_total edge owns the input's whole gradient: it may
                # follow other consumers' (in the same decreasing order)
                if last_node[parent] >= 0 and ((first[edge] and not total[edge]) or not (
                    i < last_node[parent] or (i == last_node[parent] and edge > last_edge[parent])
                )):
                    raise Error("autograd: gradient contributions applied out of the canonical order")
                last_node[parent] = i
                last_edge[parent] = edge
                if not first[edge]:
                    _accumulate(ex, p.grad, contribution, p.rows * p.cols)
        self._cursor = scratch_start
        for i in range(len(self._nodes)):  # driver-loop: over graph nodes
            var n = self._nodes[i]
            if not n.requires_grad:
                continue
            if touched[i]:
                self._nodes[i].grad_ready = True
            elif n.op == LEAF:
                # unreached differentiable leaf: its gradient is zero
                _fill(ex, n.grad, n.rows * n.cols)
                self._nodes[i].grad_ready = True
        if preact:
            # the root's gradient buffer was never written: the seed was used
            self._nodes[root].grad_ready = False
