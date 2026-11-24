# SPDX-License-Identifier: Apache-2.0
"""RUN of VotingClassifier / VotingRegressor predict, predict_proba and
transform (lane bm-meta, 2026-11-22; docs/plans/PYTHON_AT_THE_BOUNDARY.md
section 2.2). No Python: typed member outputs in, the fold written into the
caller's buffers out.

The UNPACK (bindings/trees_public_meta_door.mojo `_vote_unpack`) reads the
estimator once: the weights, the class count, and per member either its held
native member state (`native`: plan + state, served by its family's member
entry) or, for a member its family does not serve (a tree, forest or
extra-trees member, whose door predicts from the one staged device copy of X;
a class without a member mark), the host block its own door returned. RUN
calls the native members (`member_call`, one call per member, the member
splitting its own rows over the cores: the members run one after the other
because each call already uses every core) and folds every member's block
with the x_trees glue ops in member order, the same ops in the same order as
the door did before this split: no bits change.

A native member whose call refuses (a member that tests its rows refusing a
non-finite one) is reported by index (`-1`: none) and nothing is written to
the outputs; the boundary then takes that member's door, which raises the
estimator's own words (the fallback rule).
"""
from core.entry_table import EntryRef
from core.member_abi import (
    MemberPlan, MemberMatrix, member_call, member_call_of, MEMBER_F32, MEMBER_F64, MEMBER_I32, MEMBER_I64,
    MEMBER_OP_PREDICT, MemberCall, MEMBER_SCOPE_NONE, member_pin_settle,
)
from core.array_helpers_impl import fsum_f64
from xtrees.trees_meta_run import CalibratedProgram
from xtrees.glue_calls import (
    xt_accumulate, xt_accumulate_cols, xt_accumulate_cols_many, xt_accumulate_onehot, xt_argmax_rows, xt_scale,
    xt_scatter, xt_ovr_assemble, xt_accumulate_rows, xt_count_rows, xt_normalize_rows, xt_count_equal, xt_oob_r2,
    xt_calibrated_proba, xt_margin2, xt_softmax_rows, xt_weighted_median, xt_put_f32,
)

comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _I64P = MutPointer[Int64, MutUntrackedOrigin]
comptime _PlanP = MutPointer[MemberPlan, MutUntrackedOrigin]

#: `vote_*` returns this when every member answered.
comptime VOTE_ALL_ANSWERED = -1


struct VoteMember(ImplicitlyCopyable, Movable):
    """One fitted member of a vote, for one operation.

    `native`: RUN calls the MemberPlan at address `plan` (the caller's, held
    alive and unmoved through RUN: no plan is copied per call) over `state`
    (`operation`, `cols` columns per row of element type `dtype`, as the
    member's OUTPUTS answered). Else
    `address` holds the host block the member's own door returned: float32
    `rows x cols` (probabilities, predictions, a transform block) or int32
    codes (a hard vote). `class_cols`: the int32 ensemble code of each of the
    member's probability columns (soft vote), 0 otherwise. `weight`: the
    member's vote weight."""
    var native: Bool
    var plan: Int
    var state: Int
    var operation: Int
    var cols: Int
    var dtype: Int
    var address: Int
    var class_cols: Int
    var weight: Float64

    def __init__(out self, address: Int, cols: Int, class_cols: Int, weight: Float64):
        """A member whose door already answered: its block at `address`."""
        self.native = False
        self.plan = 0
        self.state = 0
        self.operation = MEMBER_OP_PREDICT
        self.cols = cols
        self.dtype = MEMBER_F32
        self.address = address
        self.class_cols = class_cols
        self.weight = weight

    def __init__(out self, plan: Int, state: Int, operation: Int, cols: Int, dtype: Int, class_cols: Int,
                 weight: Float64):
        """A member RUN calls through its held native state (`plan`: the
        address of its MemberPlan)."""
        self.native = True
        self.plan = plan
        self.state = state
        self.operation = operation
        self.cols = cols
        self.dtype = dtype
        self.address = 0
        self.class_cols = class_cols
        self.weight = weight


struct VoteRows(Movable):
    """The rows a vote reads: `n` x `d` float32 C-order host words at `x`
    (the caller keeps them alive through RUN).

    THE INDEXED ROWS (lane mf-cal7, 2026-11-23; `member_f32_block` only):
    `index` nonzero: the rows are the `n` int32 row numbers at `index` of the
    WHOLE base at `x` (`base_rows` rows), handed to a member whose kind reads
    a row index (MEMBER_CAN_ROW_INDEX) so its device gathers them out of the
    one resident base (CalibratedClassifierCV's DeviceRows route); `scope`
    the caller's resident scope and `pin` X's pin token offered on the call
    (0: none), `took` the address of the byte written 1 when the member's
    cache took the pin's reference (THE PIN RULE, core/member_abi.mojo)."""
    var x: Int
    var n: Int
    var d: Int
    var base_rows: Int
    var index: Int
    var scope: Int
    var pin: Int
    var took: Int

    def __init__(out self, x: Int, n: Int, d: Int):
        self.x = x
        self.n = n
        self.d = d
        self.base_rows = n
        self.index = 0
        self.scope = MEMBER_SCOPE_NONE
        self.pin = 0
        self.took = 0

    def __init__(out self, base: Int, base_rows: Int, d: Int, index: Int, count: Int, scope: Int, pin: Int,
                 took: Int):
        """THE INDEXED ROWS: `count` int32 row numbers at `index` of the base
        `base` (`base_rows` x `d`)."""
        self.x = base
        self.n = count
        self.d = d
        self.base_rows = base_rows
        self.index = index
        self.scope = scope
        self.pin = pin
        self.took = took


def vote_rows_point(mut call: MemberCall, rows: VoteRows):
    """Point `call` at `rows` (THE INDEXED ROWS: the base, the row index, the
    scope and the offered pin)."""
    if rows.index == 0:
        call.x = MemberMatrix(rows.x, rows.n, rows.d)
        return
    call.x = MemberMatrix(rows.x, rows.base_rows, rows.d)
    call.row_index = MemberMatrix(rows.index, rows.n, 1, MEMBER_I32)
    call.scope = rows.scope
    call.pin = rows.pin


def vote_rows_call(plan: MemberPlan, operation: Int, mut call: MemberCall, rows: VoteRows) raises:
    """One member call over `rows`; an offered pin's answer is read whether
    the call returns or raises (THE PIN RULE)."""
    if rows.index == 0 or call.pin == 0:
        _ = member_call(plan.member, operation, call)
        return
    try:
        _ = member_call(plan.member, operation, call)
    finally:
        if member_pin_settle(call) != 0 and rows.took != 0:
            MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=rows.took)[] = UInt8(1)


def _member_f32(m: VoteMember, rows: VoteRows, mut keep: List[List[Float32]]) -> Int:
    """The member's float32 `n x cols` block: its door's block, or one native
    call written into storage `keep` owns (a float64 answer is rounded to
    float32 word by word, as the door's float32 admission of it rounded).
    0 when the native call refused."""
    try:
        return member_f32_block(m, rows, keep)
    except:
        return 0


def member_f32_block(m: VoteMember, rows: VoteRows, mut keep: List[List[Float32]]) raises -> Int:
    """`_member_f32`, raising the member's own words when its call refuses
    (lane mf-cal3, 2026-11-23: CalibratedClassifierCV.fit raises them; the
    vote RUNs keep `_member_f32`, whose 0 sends a member to its door). One
    body: `_member_f32` is this call under a catch."""
    if not m.native:
        return m.address
    var count = rows.n * m.cols
    var plan = _PlanP(unsafe_from_address=m.plan)
    var call = member_call_of(plan[], m.state)
    vote_rows_point(call, rows)
    if m.dtype == MEMBER_F64:
        var wide = List[Float64](unsafe_uninit_length=count)
        call.dst = MemberMatrix(Int(wide.unsafe_ptr()), rows.n, m.cols, MEMBER_F64)
        vote_rows_call(plan[], m.operation, call, rows)
        var words = List[Float32](unsafe_uninit_length=count)
        for i in range(count):  # small-loop(one rounding pass over one member's block, as the door's admission made)
            words[i] = wide[i].cast[DType.float32]()
        _ = wide^
        keep.append(words^)
    else:
        var words = List[Float32](unsafe_uninit_length=count)
        call.dst = MemberMatrix(Int(words.unsafe_ptr()), rows.n, m.cols, MEMBER_F32)
        vote_rows_call(plan[], m.operation, call, rows)
        keep.append(words^)
    return Int(keep[len(keep) - 1].unsafe_ptr())


def _member_codes_into(m: VoteMember, rows: VoteRows, dst: Int) -> Bool:
    """One native member's int32 class codes written at `dst` (`rows.n`
    words; int64 codes narrowed once, as the door's int32 cast narrowed
    them). False when the native call refused. THE INDEXED ROWS (lane
    vdf-meta2): an indexed `rows` hands the member the base, the row index
    and the offered pin (`vote_rows_point`), as `member_f32_block` does; it
    read the first `rows.n` rows of the base before."""
    try:
        var plan = _PlanP(unsafe_from_address=m.plan)
        var call = member_call_of(plan[], m.state)
        vote_rows_point(call, rows)
        if m.dtype == MEMBER_I64:
            var wide = List[Int64](unsafe_uninit_length=rows.n)
            call.dst = MemberMatrix(Int(wide.unsafe_ptr()), rows.n, 1, MEMBER_I64)
            vote_rows_call(plan[], m.operation, call, rows)
            var codes = _I32P(unsafe_from_address=dst)
            for i in range(rows.n):  # small-loop(one narrowing pass over one member's codes, as the door's cast made)
                codes[i] = Int32(wide[i])
            _ = wide^
        else:
            call.dst = MemberMatrix(dst, rows.n, 1, MEMBER_I32)
            vote_rows_call(plan[], m.operation, call, rows)
    except:
        return False
    return True


def _member_codes(m: VoteMember, rows: VoteRows, mut keep: List[List[Int32]]) -> Int:
    """The member's int32 class codes: its door's codes, or one native call
    into storage `keep` owns (`_member_codes_into`). 0 when the native call
    refused."""
    if not m.native:
        return m.address
    var codes = List[Int32](unsafe_uninit_length=rows.n)
    if not _member_codes_into(m, rows, Int(codes.unsafe_ptr())):
        return 0
    keep.append(codes^)
    return Int(keep[len(keep) - 1].unsafe_ptr())


def vote_weight_sum(weights: List[Float64]) raises -> Float64:
    """`math.fsum(weights)` (core/array_helpers_impl.mojo `fsum_f64`); an
    overflow of finite terms raises Python's words."""
    var terms = weights.copy()
    var total: List[Float64] = [0.0]
    var status = fsum_f64(_F64P(unsafe_from_address=Int(terms.unsafe_ptr())),
                          _F64P(unsafe_from_address=Int(total.unsafe_ptr())), len(terms))
    _ = terms^
    if status != 0 and total[0] != 0:
        raise Error("OverflowError: math range error")
    return total[0]


def vote_soft_run(glue: EntryRef, rows: VoteRows, members: List[VoteMember], k: Int, acc: Int) raises -> Int:
    """`VotingClassifier._soft`: acc (float64 n x k, zero) = the weighted sum
    of the members' probabilities placed at their class columns, divided by
    the sum of the weights. One fused glue call over every member, in member
    order."""
    var keep = List[List[Float32]]()
    var addrs: List[Int] = [acc]
    var ints: List[Int] = [rows.n, k]
    var floats: List[Float64] = [Float64(rows.n), Float64(k)]
    var weights = List[Float64](capacity=len(members))
    for j in range(len(members)):  # driver-loop: over members
        var block = _member_f32(members[j], rows, keep)
        if block == 0:
            return j
        addrs.append(block)
        addrs.append(members[j].class_cols)
        ints.append(members[j].cols)
        ints.append(0)
        floats.append(Float64(members[j].cols))
        floats.append(members[j].weight)
        weights.append(members[j].weight)
    _ = xt_accumulate_cols_many(glue, addrs, ints, floats)
    _ = keep^  # the native blocks are read through the fused call above
    _ = xt_scale(glue, acc, rows.n * k, vote_weight_sum(weights))
    return VOTE_ALL_ANSWERED


def vote_codes_run(glue: EntryRef, rows: VoteRows, members: List[VoteMember], k: Int, soft: Bool, acc: Int,
                   codes: Int) raises -> Int:
    """`VotingClassifier.predict` before its decode: int32 codes into
    `classes_` at `codes` (n). Soft: the argmax of `vote_soft_run`; hard: the
    argmax of the weighted one-hot votes of the members' codes, member by
    member. `acc`: float64 n x k zero scratch."""
    if soft:
        var refused = vote_soft_run(glue, rows, members, k, acc)
        if refused != VOTE_ALL_ANSWERED:
            return refused
    else:
        var keep = List[List[Int32]]()
        var blocks = List[Int](capacity=len(members))
        for j in range(len(members)):  # driver-loop: over members (every member answers before any vote is cast)
            var block = _member_codes(members[j], rows, keep)
            if block == 0:
                return j
            blocks.append(block)
        for j in range(len(members)):  # driver-loop: over members
            _ = xt_accumulate_onehot(glue, acc, blocks[j], rows.n, k, members[j].weight, 0.0)
        _ = keep^  # the native codes are read through the votes above
    _ = xt_argmax_rows(glue, acc, codes, rows.n, k)
    return VOTE_ALL_ANSWERED


def vote_regress_run(glue: EntryRef, rows: VoteRows, members: List[VoteMember], acc: Int) raises -> Int:
    """`VotingRegressor.predict`: acc (float64 n, zero) = the weighted sum of
    the members' predictions in member order, divided by the sum of the
    weights."""
    var keep = List[List[Float32]]()
    var blocks = List[Int](capacity=len(members))
    var weights = List[Float64](capacity=len(members))
    for j in range(len(members)):  # driver-loop: over members (every member answers before the sum)
        var block = _member_f32(members[j], rows, keep)
        if block == 0:
            return j
        blocks.append(block)
        weights.append(members[j].weight)
    for j in range(len(members)):  # driver-loop: over members
        _ = xt_accumulate(glue, acc, blocks[j], rows.n, members[j].weight)
    _ = keep^  # the native blocks are read through the sums above
    _ = xt_scale(glue, acc, rows.n, vote_weight_sum(weights))
    return VOTE_ALL_ANSWERED


def vote_transform_run(glue: EntryRef, rows: VoteRows, members: List[VoteMember], k: Int, soft: Bool,
                       width: Int, out_words: Int) raises -> Int:
    """`Voting*.transform`: out (float64 n x width, zero) holds each member's
    block side by side: soft, member j's probabilities placed at their class
    columns (float64, then rounded to float32 as the door's admission of
    them rounded) at columns j*k; else member j's float32 output at column
    j (its `cols` columns)."""
    var n = rows.n
    var order = List[Int32](unsafe_uninit_length=n)
    for i in range(n):  # small-loop(the row numbers 0 .. n-1 the scatter places, once per call)
        order[i] = Int32(i)
    var order_at = Int(order.unsafe_ptr())
    var keep = List[List[Float32]]()
    var blocks = List[Int](capacity=len(members))
    for j in range(len(members)):  # driver-loop: over members (every member answers before anything is placed)
        var block = _member_f32(members[j], rows, keep)
        if block == 0:
            return j
        blocks.append(block)
    for j in range(len(members)):  # driver-loop: over members
        if soft:
            var p = List[Float64](length=n * k, fill=0.0)
            _ = xt_accumulate_cols(glue, Int(p.unsafe_ptr()), blocks[j], members[j].class_cols, n, k,
                                   members[j].cols, 1.0)
            var words = List[Float32](unsafe_uninit_length=n * k)
            for i in range(n * k):  # small-loop(one rounding pass over one member's placed block)
                words[i] = p[i].cast[DType.float32]()
            _ = p^
            _ = xt_scatter(glue, out_words, Int(words.unsafe_ptr()), order_at, n, width, n, k, j * k)
            _ = words^
        else:
            _ = xt_scatter(glue, out_words, blocks[j], order_at, n, width, n, members[j].cols, j)
    _ = keep^  # the native blocks are read through the scatters above
    _ = order^
    return VOTE_ALL_ANSWERED


def ovr_assemble_run(glue: EntryRef, rows: VoteRows, members: List[VoteMember], at: List[Int], k: Int,
                     finish: Int, out_words: Int) raises -> Int:
    """`OneVsRestClassifier` predict (`finish` 2: int32 row-argmax codes into
    `out_words`) and predict_proba (`finish` 1: float64 n x k normalized
    rows) for k > 2 (lane bm-meta-2, 2026-11-22): member j's positive column
    `at[j]` of its float32 block (its decision column, or its probability
    column of code 1; -1: a member that never saw code 1) widened into the
    scores and finished by the glue's one `x_trees_ovr_assemble` program,
    the op the door called before this split over the same blocks in the
    same order: no bits change. Every member answers before anything is
    written; a native member that refuses is returned by index."""
    var n = rows.n
    var keep = List[List[Float32]]()
    var addrs = List[Int](capacity=len(members) + 1)
    addrs.append(out_words)
    var ints: List[Int] = [n, k, finish]
    var floats: List[Float64] = [Float64(n), Float64(k), Float64(finish)]
    for j in range(len(members)):  # driver-loop: over members (every member answers before anything is written)
        var block = _member_f32(members[j], rows, keep)
        if block == 0:
            return j
        addrs.append(block)
        ints.append(members[j].cols)
        floats.append(Float64(members[j].cols))
        ints.append(at[j])
        floats.append(Float64(at[j]))
    _ = xt_ovr_assemble(glue, addrs, ints, floats)
    _ = keep^  # the native blocks are read through the program above
    return VOTE_ALL_ANSWERED


def multioutput_codes_run(rows: VoteRows, members: List[VoteMember], dst: List[Int], start: Int) -> Int:
    """`MultiOutputClassifier` predict and score before the decode (lane
    bm-meta-3, 2026-11-23): member j's int32 codes into output j's fitted
    classes, written at `dst[j]` (`rows.n` words) for members `start` ..
    m-1. A door member's codes are already its column (`dst[j]` is its
    block's address): nothing is written. The members run one after the
    other, each call splitting its own rows over the cores, as before. A
    native member that refuses is returned by index (outputs before it are
    final; the caller asks its door and runs again from it)."""
    for j in range(start, len(members)):  # driver-loop: over members (outputs)
        if members[j].native and not _member_codes_into(members[j], rows, dst[j]):
            return j
    return VOTE_ALL_ANSWERED


def multioutput_proba_run(glue: EntryRef, rows: VoteRows, members: List[VoteMember], ks: List[Int],
                          accs: List[Int], start: Int) raises -> Int:
    """`MultiOutputClassifier.predict_proba` (lane bm-meta-3, 2026-11-23):
    member j's probabilities placed at their class columns of output j's
    float64 `rows.n x ks[j]` zero block `accs[j]` (one `xt_accumulate_cols`,
    weight 1), member after member from `start`, the same op over the same
    blocks in the same order as the door before this split: no bits change.
    A native member that refuses is returned by index before its block is
    touched (blocks before it are final)."""
    var keep = List[List[Float32]]()
    for j in range(start, len(members)):  # driver-loop: over members (outputs)
        var block = _member_f32(members[j], rows, keep)
        if block == 0:
            return j
        _ = xt_accumulate_cols(glue, accs[j], block, members[j].class_cols, rows.n, ks[j], members[j].cols, 1.0)
    _ = keep^  # the native blocks are read through the accumulations above
    return VOTE_ALL_ANSWERED


# ----------------------------------------------------------------- Bagging
# Lane bm-meta-3 (2026-11-23). Each Bagging member reads its own rows: the
# feature subset of X (or X itself) for predict, its out-of-bag rows for the
# OOB score, gathered by the UNPACK (bindings/trees_public_boost_door.mojo)
# into float32 C-order blocks whose addresses `xs[j]` (`ns[j]` x `ds[j]`)
# the caller keeps alive through RUN.


def bag_predict_run(glue: EntryRef, members: List[VoteMember], xs: List[Int], ds: List[Int], n: Int, k: Int,
                    classifier: Bool, acc: Int, start: Int) raises -> Int:
    """`BaggingClassifier.predict_proba` (classifier: acc float64 n x k,
    zero: each member's probabilities placed at its class columns, weight 1)
    and `BaggingRegressor.predict` (acc float64 n, zero: the members'
    float32 predictions added), member after member from `start`, then
    divided by the member count once every member has answered: the ops of
    the door before this split, over the same blocks in the same order (no
    bits change). A native member that refuses is returned by index before
    its block is added (the sums before it are final)."""
    var keep = List[List[Float32]]()
    for j in range(start, len(members)):  # driver-loop: over the member estimators
        var block = _member_f32(members[j], VoteRows(xs[j], n, ds[j]), keep)
        if block == 0:
            return j
        if classifier:
            _ = xt_accumulate_cols(glue, acc, block, members[j].class_cols, n, k, members[j].cols, 1.0)
        else:
            _ = xt_accumulate(glue, acc, block, n, 1.0)
    _ = keep^  # the native blocks are read through the sums above
    _ = xt_scale(glue, acc, n * (k if classifier else 1), Float64(len(members)))
    return VOTE_ALL_ANSWERED


def bag_oob_run(glue: EntryRef, members: List[VoteMember], xs: List[Int], ns: List[Int], ds: List[Int],
                oobs: List[Int], n: Int, k: Int, classifier: Bool, acc: Int, counts: Int, start: Int) raises -> Int:
    """`_BaggingBase._oob_outputs`: each member's output on its `ns[j]`
    out-of-bag rows (int32 row numbers at `oobs[j]`; a member with none is
    skipped), added into `acc` (float64 n x k, zero; k 1 for a regressor)
    at those rows, and `counts` (int32 n) counting them, in member order. A
    classifier member's probabilities are placed at its class columns of a
    float64 m x k zero block first; a regressor's float32 predictions are
    widened to float64 (exact). The same glue ops in the same order as the
    door before this split: no bits change. A native member that refuses is
    returned by index before anything of it is added."""
    var keep = List[List[Float32]]()
    for j in range(start, len(members)):  # driver-loop: over the member estimators
        var m = ns[j]
        if m == 0:
            continue
        var block = _member_f32(members[j], VoteRows(xs[j], m, ds[j]), keep)
        if block == 0:
            return j
        var width = k if classifier else 1
        var out = List[Float64](length=m * width, fill=0.0)
        if classifier:
            _ = xt_accumulate_cols(glue, Int(out.unsafe_ptr()), block, members[j].class_cols, m, k, members[j].cols,
                                   1.0)
        else:
            var p = _F32P(unsafe_from_address=block)
            for i in range(m):  # small-loop(one exact widening pass over one member's out-of-bag predictions)
                out[i] = Float64(p[i])
        _ = xt_accumulate_rows(glue, acc, Int(out.unsafe_ptr()), oobs[j], n, width, m)
        _ = out^  # read through the accumulation above
        _ = xt_count_rows(glue, counts, oobs[j], n, m)
    _ = keep^  # the native blocks are read through the sums above
    return VOTE_ALL_ANSWERED


def bag_oob_classifier_finish(glue: EntryRef, acc: Int, codes: Int, pred: Int, n: Int, k: Int) raises -> Int:
    """`BaggingClassifier._set_oob_score` after the sums: each row of `acc`
    normalized in place (`oob_decision_function_`), its argmax into `pred`
    (int32 n), and the count of rows whose argmax is the true code
    (`codes`, int32 n): `oob_score_` = hits / n."""
    _ = xt_normalize_rows(glue, acc, n, k)
    _ = xt_argmax_rows(glue, acc, pred, n, k)
    return xt_count_equal(glue, pred, codes, n)


@fieldwise_init
struct BagR2(ImplicitlyCopyable, Movable):
    """`bag_oob_r2_run`'s answer: `finite` False when a target or
    out-of-bag prediction is not finite or a sum overflows float64 (the
    door raises its words); else the R2 sums."""

    var finite: Bool
    var ss_tot: Float64
    var ss_res: Float64


def bag_oob_r2_run(glue: EntryRef, acc: Int, counts: Int, y: Int, pred: Int, n: Int) raises -> BagR2:
    """`BaggingRegressor._set_oob_score` after the sums: `oob_prediction_`
    (float64 n at `pred`) and the R2 sums of the float32 targets at `y`
    (one `x_trees_oob_r2` call, as the door made)."""
    var words = List[Float64](length=4, fill=0.0)
    var flags = List[Int32](length=4, fill=0)
    _ = xt_oob_r2(glue, acc, counts, y, pred, Int(words.unsafe_ptr()), Int(flags.unsafe_ptr()), n)
    var got = BagR2(flags[0] == 0 and flags[1] == 0, words[1], words[2])
    _ = words^
    _ = flags^
    return got


# ---------------------------------------------------------------- Stacking
# Lane bm-meta-3 (2026-11-23): the members' stacked outputs (`transform`,
# the predict path, and each cv fold's test outputs inside fit), typed.

#: What a Stacking member's `stack_method_` asks it for.
comptime STACK_PREDICT = 0
comptime STACK_PROBA = 1
comptime STACK_DECISION = 2


def stack_place_run(glue: EntryRef, rows: VoteRows, members: List[VoteMember], kinds: List[Int],
                    class_counts: List[Int], k: Int, binary: Bool, meta: Int, meta_rows: Int, width: Int,
                    place: Int, col0: Int, mut widths: List[Int], start: Int) raises -> Int:
    """`_StackingBase` meta features: each member's output over `rows`
    placed into `meta` (float64 `meta_rows` x `width`) at the rows `place`
    (int32 `rows.n` row numbers) and the member's columns, side by side from
    `col0`, member after member from `start` (`widths[j]` is written for
    each member done: the columns it took).

    `kinds[j]` / `class_counts[j]` (the member's own fitted class count; 0
    for STACK_PREDICT): a predict block is placed as it is (its columns).
    A probability or decision block whose member saw every class (`cols`
    == k) is placed as it is, except that a binary predict_proba keeps its
    second column (`_StackingBase._out`): the old door's block, the same
    words (no bits change). THE FOLD THAT LACKS A CLASS (a cv fold whose
    training rows miss a class, scikit-learn `_enforce_prediction_order`):
    the member's columns go to its class columns (`class_cols`) of a k-wide
    block, a missing class's column 0.0 for predict_proba and the lowest
    float32 for decision_function; a decision_function member that saw two
    classes or fewer is refused in scikit-learn's words. A native member
    that refuses is returned by index before anything of it is placed.
    `meta` 0: nothing is placed, only `widths` is written (the fit's
    width probe over the first rows)."""
    var keep = List[List[Float32]]()
    var at = col0
    for j in range(start):  # driver-loop: over the members already placed
        at += widths[j]
    for j in range(start, len(members)):  # driver-loop: over members
        var m = members[j]
        var block = _member_f32(m, rows, keep)
        if block == 0:
            return j
        var n = rows.n
        var src = block
        var c = m.cols
        var placed = List[Float32]()
        if kinds[j] != STACK_PREDICT and c != k and not (kinds[j] == STACK_DECISION and k == 2 and c == 1):
            if kinds[j] == STACK_DECISION and class_counts[j] <= 2:
                raise Error("ValueError: Only " + String(class_counts[j]) + " class/es in training fold, but "
                            + String(k) + " in overall dataset. This is not supported for decision_function with "
                            + "imbalanced folds. To fix this, use a cross-validation technique resulting in "
                            + "properly stratified folds")
            var fill = Float32(0.0) if kinds[j] == STACK_PROBA else Float32.MIN_FINITE
            placed = List[Float32](length=n * k, fill=fill)
            var b = _F32P(unsafe_from_address=block)
            var codes = _I32P(unsafe_from_address=m.class_cols)
            for i in range(n):  # small-loop(one placing pass over one member's fold block: a copy, no arithmetic)
                for q in range(c):  # small-loop(the member's class columns)
                    placed[i * k + Int(codes[q])] = b[i * c + q]
            src = Int(placed.unsafe_ptr())
            c = k
        if kinds[j] == STACK_PROBA and binary:
            # `_one_col(p, 1)`: the second column alone
            if meta != 0:
                var col = List[Float32](unsafe_uninit_length=n)
                var b = _F32P(unsafe_from_address=src)
                for i in range(n):  # small-loop(one column copy of one member's block)
                    col[i] = b[i * c + 1]
                _ = xt_scatter(glue, meta, Int(col.unsafe_ptr()), place, meta_rows, width, n, 1, at)
                _ = col^
            c = 1
        elif meta != 0:
            _ = xt_scatter(glue, meta, src, place, meta_rows, width, n, c, at)
        _ = placed^  # read through the scatter above
        if j < len(widths):
            widths[j] = c
        else:
            widths.append(c)
        at += c
    _ = keep^  # the native blocks are read through the scatters above
    return VOTE_ALL_ANSWERED


# ------------------------------------------------- CalibratedClassifierCV
# Lane bm-meta-3 (2026-11-23).


@fieldwise_init
struct CalPair(ImplicitlyCopyable, Movable):
    """One fitted (member, calibrators) pair of a calibrated predict: the
    member's score block's positive column `at` (binary, a two-column
    block), its class codes `class_cols` and class count `ks` (k > 2), and
    its calibrators `cals[first : first + count]`."""

    var at: Int
    var class_cols: Int
    var ks: Int
    var first: Int
    var count: Int


@fieldwise_init
struct CalFit(ImplicitlyCopyable, Movable):
    """One fitted calibrator: Platt (`isotonic` False: `a`, `b`) or isotonic
    (`m` knots at `knots_x` / `knots_y`, float64, the caller's)."""

    var isotonic: Bool
    var a: Float64
    var b: Float64
    var knots_x: Int
    var knots_y: Int
    var m: Int


def calibrated_proba_run(glue: EntryRef, rows: VoteRows, members: List[VoteMember], pairs: List[CalPair],
                         cals: List[CalFit], k: Int, acc: Int) raises -> Int:
    """`CalibratedClassifierCV.predict_proba` (k >= 2): each pair's member
    scores (its decision_function, else its predict_proba: `members[i]`,
    native or its door's float32 block), then the whole epilogue as one
    `x_trees_calibrated_proba` program (`CalibratedProgram`: the same words
    in the same order as the door's `_calibrated_fused` before this split,
    no bits change) into `acc` (float64 n x k). Every member answers before
    the program runs; a native member that refuses is returned by index."""
    var keep = List[List[Float32]]()
    var blocks = List[Int](capacity=len(members))
    for i in range(len(members)):  # driver-loop: over the (member, calibrators) pairs (every member answers first)
        var block = _member_f32(members[i], rows, keep)
        if block == 0:
            return i
        blocks.append(block)
    var program = CalibratedProgram(acc, rows.n, k, len(pairs))
    for i in range(len(pairs)):  # driver-loop: over the (member, calibrators) pairs
        var p = pairs[i]
        program.pair(blocks[i], p.class_cols, members[i].cols, p.at, p.ks, p.count)
        for j in range(p.first, p.first + p.count):  # driver-loop: over the pair's calibrators
            if cals[j].isotonic:
                program.isotonic(cals[j].knots_x, cals[j].knots_y, cals[j].m)
            else:
                program.sigmoid(cals[j].a, cals[j].b)
    _ = xt_calibrated_proba(glue, program.addrs, program.ints, program.floats)
    _ = keep^  # the native blocks are read through the program above
    _ = blocks^
    return VOTE_ALL_ANSWERED


# ---------------------------------------------------------------- AdaBoost
# Lane bm-meta-3 (2026-11-23). An AdaBoost member is a one-tree forest: its
# door predicts from the one staged device copy of X (not served by a member
# call, by rule), so the UNPACK hands RUN each member's door block.

#: What `ada_classifier_run` finishes the decision into.
comptime ADA_DECISION = 0
comptime ADA_PROBA = 1
comptime ADA_CODES = 2


def ada_classifier_run(glue: EntryRef, codes: List[Int], alphas: List[Float64], n: Int, k: Int, acc: Int,
                       finish: Int, dst: Int) raises:
    """`AdaBoostClassifier._decision` and its finishes: acc (float64 n x k,
    zero) = each member's one-hot vote of its int32 codes (`codes[i]`),
    weighted +a / -a/(k-1) with its weight `alphas[i]`, over the fsum of
    every weight (`vote_weight_sum`, the core fsum the door's `_fsum` called).
    Then `finish`: ADA_DECISION (k 2: the margin into `dst`, float64 n),
    ADA_PROBA (k 2: the margin pairs into `dst`, float64 n x 2; else acc /
    (k-1) in place; then the row softmax of that block), ADA_CODES (int32 n
    into `dst`: k 2 the margin's sign, else the row argmax). The glue ops of
    the door before this split in the same order: no bits change."""
    for i in range(len(codes)):  # driver-loop: over the member estimators
        var a = alphas[i]
        _ = xt_accumulate_onehot(glue, acc, codes[i], n, k, a, -a / Float64(k - 1))
    _ = xt_scale(glue, acc, n * k, vote_weight_sum(alphas))
    if finish == ADA_DECISION:
        if k == 2:
            _ = xt_margin2(glue, acc, dst, n, 0)
    elif finish == ADA_PROBA:
        var block = acc
        if k == 2:
            _ = xt_margin2(glue, acc, dst, n, 2)
            block = dst
        else:
            _ = xt_scale(glue, acc, n * k, Float64(k - 1))
        _ = xt_softmax_rows(glue, block, n, k)
    else:
        if k == 2:
            _ = xt_margin2(glue, acc, dst, n, 1)
        else:
            _ = xt_argmax_rows(glue, acc, dst, n, k)


def ada_regressor_run(glue: EntryRef, blocks: List[Int], weights: List[Float64], n: Int, dst: Int) raises:
    """`AdaBoostRegressor.predict`: the weighted median (float32 n into
    `dst`) of the members' float32 predictions (`blocks[j]`, n each) under
    their weights, one `x_trees_weighted_median` over the m x n block the
    door filled member by member before this split (same words)."""
    var m = len(blocks)
    var preds = List[Float32](unsafe_uninit_length=max(m * n, 1))
    for j in range(m):  # driver-loop: over the member estimators
        _ = xt_put_f32(glue, Int(preds.unsafe_ptr()), blocks[j], j * n, n)
    var w = weights.copy()
    _ = xt_weighted_median(glue, Int(preds.unsafe_ptr()), Int(w.unsafe_ptr()), dst, n, m)
    _ = preds^
    _ = w^
