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
    MEMBER_OP_PREDICT,
)
from core.array_helpers_impl import fsum_f64
from xtrees.glue_calls import (
    xt_accumulate, xt_accumulate_cols, xt_accumulate_cols_many, xt_accumulate_onehot, xt_argmax_rows, xt_scale,
    xt_scatter, xt_ovr_assemble,
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
    (the caller keeps them alive through RUN)."""
    var x: Int
    var n: Int
    var d: Int

    def __init__(out self, x: Int, n: Int, d: Int):
        self.x = x
        self.n = n
        self.d = d


def _member_f32(m: VoteMember, rows: VoteRows, mut keep: List[List[Float32]]) -> Int:
    """The member's float32 `n x cols` block: its door's block, or one native
    call written into storage `keep` owns (a float64 answer is rounded to
    float32 word by word, as the door's float32 admission of it rounded).
    0 when the native call refused."""
    if not m.native:
        return m.address
    var count = rows.n * m.cols
    try:
        var plan = _PlanP(unsafe_from_address=m.plan)
        var call = member_call_of(plan[], m.state)
        call.x = MemberMatrix(rows.x, rows.n, rows.d)
        if m.dtype == MEMBER_F64:
            var wide = List[Float64](unsafe_uninit_length=count)
            call.dst = MemberMatrix(Int(wide.unsafe_ptr()), rows.n, m.cols, MEMBER_F64)
            _ = member_call(plan[].member, m.operation, call)
            var words = List[Float32](unsafe_uninit_length=count)
            for i in range(count):  # small-loop(one rounding pass over one member's block, as the door's admission made)
                words[i] = wide[i].cast[DType.float32]()
            _ = wide^
            keep.append(words^)
        else:
            var words = List[Float32](unsafe_uninit_length=count)
            call.dst = MemberMatrix(Int(words.unsafe_ptr()), rows.n, m.cols, MEMBER_F32)
            _ = member_call(plan[].member, m.operation, call)
            keep.append(words^)
    except:
        return 0
    return Int(keep[len(keep) - 1].unsafe_ptr())


def _member_codes_into(m: VoteMember, rows: VoteRows, dst: Int) -> Bool:
    """One native member's int32 class codes written at `dst` (`rows.n`
    words; int64 codes narrowed once, as the door's int32 cast narrowed
    them). False when the native call refused."""
    try:
        var plan = _PlanP(unsafe_from_address=m.plan)
        var call = member_call_of(plan[], m.state)
        call.x = MemberMatrix(rows.x, rows.n, rows.d)
        if m.dtype == MEMBER_I64:
            var wide = List[Int64](unsafe_uninit_length=rows.n)
            call.dst = MemberMatrix(Int(wide.unsafe_ptr()), rows.n, 1, MEMBER_I64)
            _ = member_call(plan[].member, m.operation, call)
            var codes = _I32P(unsafe_from_address=dst)
            for i in range(rows.n):  # small-loop(one narrowing pass over one member's codes, as the door's cast made)
                codes[i] = Int32(wide[i])
            _ = wide^
        else:
            call.dst = MemberMatrix(dst, rows.n, 1, MEMBER_I32)
            _ = member_call(plan[].member, m.operation, call)
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
