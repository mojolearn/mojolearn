# SPDX-License-Identifier: Apache-2.0
"""RUN of the DART fit (bindings/trees_public_boost_door.mojo `_dart_boost`):
the whole boosting round loop on typed values. The door UNPACKs the
estimator's fields, the X / y addresses, the class starts, the entry
references and the first member's validated fit request into a
`DartRoundPlan`; `dart_rounds_run` opens the x_trees DART session, runs every
round (drop thresholds, the device step, the rescale, the round's row bag)
and every member of it (column sample, the rf image's one-tree DART member
fit, export, column remap, the device add), closes the session and returns
the members' model words and the tree coefficients and weights. The door
PACKs them into the public `trees_` members (`dart_member_into`) after the
loop. No Python import (tools/python_free_dirs.sh, `*_run.mojo`).

Lane py-trees (2026-11-14): moved from the door, call for call, in the same
order and with the same words, so every vendor's bits are unchanged. Per
member, the door's loop built a Python RandomForestRegressor (`_member_like`:
a dict copy and the `_cfg` copy), a 16-item Python params list that
`forest_fit_request_py` read back slot by slot, a Python Array per sampled
column set and per round bag (`_indices`, `_copy_i32`), and a colid copy
through memoryview/tobytes/frombytes; it resolved the glue entry again in
every `_indices` / `_dart_bag` call. Here the request is built once and three
slots are rewritten per member, the bag and column buffers are allocated once
per fit, the glue entry once per fit, and the colid is remapped in place.
Added per member: one copy of the member's model words (at most
5 * (2 * num_leaves - 1) words plus two offsets) from the arena into the
member's Arrays at PACK; the arena grows by doubling (O(log trees)
allocations per fit). The rounds are serial by nature (round t's drop set
and residuals read round t-1's trees); the per-row work of each round runs
on the device inside the image's step, fit and add.
"""
from std.memory import unsafe_memcpy
from core.entry_table import EntryRef
from core.trees_entry_table import TreesEntryKey
from core.forest_calls import xf_fit_dart, xf_export_sizes, xf_export_into, xf_export_release
from core.forest_fit_request import ForestFitArgs
from xtrees.dart_abi import XD_CAP_ROUND
from xtrees.dart_calls import xd_caps, xd_open, xd_step, xd_add, xd_close
from xtrees.glue_calls import (
    xt_ref, xt_bag_rows, xt_dart_thresholds, xt_dart_rescale, xt_iota_i32, xt_remap_cols, xt_sample_indices,
    xt_exact_sum, xt_log64, xt_class_counts, xt_class_priors,
)
from xtrees.trees_boost_run import boost_dart_cap_check, _sub_seed
from xtrees.trees_forest_run import forest_export_counts
from bindings.trees_public_inputs_run import TREES_NONFINITE

# The x_trees glue entry is resolved once per fit (lane loops-trees-ab, 2026-11-11).
# TOMBSTONE: MOJOLEARN_DART_ROUND_REF_OFF (old path, no A/B recorded) deleted 2026-11-15 by lane/expdel2-02-base; code recoverable at 650388311e.
# Restore: git apply experiments/removed/MOJOLEARN_DART_ROUND_REF_OFF.patch; record in docs/TOMBSTONES.md.

# Params slots of the member's fit request (`_rf_params`' 16-slot list).
comptime _P_ROWS = 0
comptime _P_COLS = 1
comptime _P_SEED = 13

# Words per member in `DartRounds.members`.
comptime DART_M_OFF_START = 0
comptime DART_M_OFF_N = 1
comptime DART_M_NODE_START = 2   # colid, quesval and left_child share start and count
comptime DART_M_NODES = 3
comptime DART_M_LEAF_START = 4
comptime DART_M_LEAF_N = 5
comptime DART_M_VAL_START = 6
comptime DART_M_VAL_N = 7
comptime DART_M_COLS = 8         # the member's n_features_in_
comptime DART_M_TREES = 9        # export meta[0]: `_n_trees`
comptime DART_M_OUTPUTS = 10     # export meta[1]: `_num_outputs`
comptime DART_M_WORDS = 11

comptime _WALK_REFUSAL = "RuntimeError: x_trees dart: a tree walk left its tree (child or column out of range)"


@fieldwise_init
struct DartRoundPlan(Movable):
    """Everything the round loop reads, as plain values (UNPACK, once per fit)."""
    var dart: EntryRef          # the x_trees DART entry (xd_ref)
    var glue_key: TreesEntryKey  # the x_trees image (its glue entry, xt_ref)
    var forest: EntryRef        # the rf image of the members (forest_key)
    var x: Int                  # C-order float32 X, n x d
    var y: Int                  # float32 targets / class codes, n
    var inits: List[Float64]    # the K class starts (`init_score_`)
    var n: Int
    var d: Int
    var k: Int
    var kind: Int               # 0 regression, 1 binary, 2 multiclass
    var n_iters: Int
    var node_cap: Int
    var drop_seed: Int
    var skip_thr: Int
    var uniform: Int
    var xgb: Int
    var max_drop: Int
    var drop_rate: Float64
    var lr: Float64
    var lam: Float64
    var l1: Float64
    var mds: Float64
    var bag: Bool               # subsample_freq > 0 and subsample < 1
    var bag_freq: Int
    var bag_frac: Float64
    var bag_seed: Int
    var col_count: Int          # sampled columns per tree; 0 = every column
    var col_seed: Int
    var seed: Int               # the members' seed base (`_sub_seed(seed, j)`)


struct DartRounds(Movable):
    """The fitted members' model words (one arena per array) and the tree
    coefficients and weights (RUN result; the door PACKs it)."""
    var offsets: List[Int32]
    var colid: List[Int32]
    var quesval: List[Float32]
    var left: List[Int32]
    var leaves: List[Float32]
    var values: List[Float32]
    var members: List[Int]
    var coefs: List[Float64]
    var weights: List[Float64]

    def __init__(out self):
        self.offsets = List[Int32]()
        self.colid = List[Int32]()
        self.quesval = List[Float32]()
        self.left = List[Int32]()
        self.leaves = List[Float32]()
        self.values = List[Float32]()
        self.members = List[Int]()
        self.coefs = List[Float64]()
        self.weights = List[Float64]()

    def count(self) -> Int:
        return len(self.members) // DART_M_WORDS

    def word(self, member: Int, field: Int) -> Int:
        return self.members[member * DART_M_WORDS + field]


def dart_round_admit(dart: EntryRef, n_iters: Int, k: Int, node_cap: Int) raises:
    """The fit's refusals before any member is built: a binary without the
    device round, then the leaf-value store bound."""
    if (xd_caps(dart) & XD_CAP_ROUND) == 0:
        raise Error("NotImplementedError: mojolearn DART: this binary has no x_trees_dart_* round (a FAST CPU-only install or a "
                    + "MOJOLEARN_*DART_DEVICE_OFF build); the host loop is removed, so the fit is refused by name")
    boost_dart_cap_check(n_iters, k, node_cap)


def _i32(addr: Int, index: Int) -> Int:
    return Int(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr)[index])


def _grow_i32(mut arena: List[Int32], count: Int) -> Int:
    var start = len(arena)
    arena.resize(start + count, Int32(0))
    return start


def _grow_f32(mut arena: List[Float32], count: Int) -> Int:
    var start = len(arena)
    arena.resize(start + count, Float32(0))
    return start


def _member_fit(plan: DartRoundPlan, mut args: ForestFitArgs, handle: Int, c: Int, j: Int, rows: Int, m: Int,
                cols: Int, dc: Int, mut got: DartRounds) raises:
    """`_dart_member` + `_dart_member_fit` + `forest_fit_dart_member`: the
    member's one-tree fit on the round's device X and class-c target, its
    export appended to the arenas (the rf image's XF_FIT_DART, 'into')."""
    args.ints[_P_ROWS] = m
    args.floats[_P_ROWS] = Float64(m)
    args.ints[_P_COLS] = dc
    args.floats[_P_COLS] = Float64(dc)
    var member_seed = _sub_seed(plan.seed, j)
    args.ints[_P_SEED] = member_seed
    args.floats[_P_SEED] = Float64(member_seed)
    args.x = rows
    args.y = cols
    args.session = handle
    args.cls = c
    var fitted = xf_fit_dart(plan.forest, args)
    try:
        var sizes = xf_export_sizes(plan.forest, fitted)
        var counts = forest_export_counts(sizes)
        var off = _grow_i32(got.offsets, counts[0])
        var node = _grow_i32(got.colid, counts[1])
        _ = _grow_f32(got.quesval, counts[2])
        _ = _grow_i32(got.left, counts[3])
        var leaf = _grow_f32(got.leaves, counts[4])
        var dst: List[Int] = [
            Int(got.offsets.unsafe_ptr()) + off * 4, Int(got.colid.unsafe_ptr()) + node * 4,
            Int(got.quesval.unsafe_ptr()) + node * 4, Int(got.left.unsafe_ptr()) + node * 4,
            Int(got.leaves.unsafe_ptr()) + leaf * 4,
        ]
        xf_export_into(plan.forest, sizes, dst)
        xf_export_release(plan.forest, fitted)
        got.members.append(off)
        got.members.append(counts[0])
        got.members.append(node)
        got.members.append(counts[1])
        got.members.append(leaf)
        got.members.append(counts[4])
        got.members.append(0)   # values: set by the add
        got.members.append(0)
        got.members.append(dc)
        got.members.append(sizes.meta[0])
        got.members.append(sizes.meta[1])
    except error:
        try:
            xf_export_release(plan.forest, fitted)
        except:
            pass
        raise error^


def dart_rounds_run(plan: DartRoundPlan, var member: ForestFitArgs) raises -> DartRounds:
    """`_DARTBase._boost`'s device round (`_boost_loop_device`) on typed
    values: open, every round and member, close. Call `dart_round_admit`
    first. `member` is the first member's validated fit request (the door's
    `_rf_params` + `forest_fit_request_py`, once); its rows, columns and seed
    slots are rewritten per member. Returns the members' words and the
    coefficients and weights."""
    var n = plan.n
    var d = plan.d
    var K = plan.k
    var n_iters = plan.n_iters
    var glue = xt_ref(plan.glue_key)
    var all_cols = List[Int32](length=max(d, 1), fill=0)
    if d > 0:
        _ = xt_iota_i32(glue, Int(all_cols.unsafe_ptr()), d)
    var inits32 = List[Float32](capacity=len(plan.inits))
    for c in range(len(plan.inits)):  # small-loop(the K class starts)
        inits32.append(Float32(plan.inits[c]))
    var bad = List[Int32](length=1, fill=0)
    var weights64 = List[Float64](length=max(n_iters, 1), fill=0.0)
    var coefs64 = List[Float64](length=max(n_iters * K, 1), fill=0.0)
    var coef32 = List[Float32](length=max(n_iters * K, 1), fill=0.0)
    var thr64 = List[Int64](length=max(n_iters, 1), fill=0)
    var flags = List[Int32](length=max(n_iters, 1), fill=0)
    var state = List[Float64](length=1, fill=0.0)
    var rescaled: List[Float64] = [0.0, 0.0, 0.0]
    var bag_rows = List[Int32](length=max(1, n), fill=0)
    var bag_start = -1
    var bag_count = 0
    var col_ids = List[Int32](length=max(plan.col_count, 1), fill=0)
    var got = DartRounds()
    var args = member^
    var handle = 0
    try:
        handle = xd_open(plan.dart, plan.x, plan.y, Int(inits32.unsafe_ptr()), n, d, K, plan.kind, n_iters,
                         plan.node_cap)
    except error:
        # the open's device test of X (lane a8c), in the forest's one wording
        if TREES_NONFINITE in String(error):
            raise Error(TREES_NONFINITE)
        raise error^
    try:
        for it in range(n_iters):  # driver-loop: over the boosting rounds (serial: round t reads round t-1's trees)
            _ = xt_dart_thresholds(glue, Int(weights64.unsafe_ptr()), Int(coefs64.unsafe_ptr()),
                                   Int(coef32.unsafe_ptr()), Int(thr64.unsafe_ptr()), Int(state.unsafe_ptr()), it, K,
                                   plan.uniform, plan.max_drop, plan.drop_rate)
            _ = xd_step(plan.dart, handle, Int(coef32.unsafe_ptr()), Int(thr64.unsafe_ptr()), Int(flags.unsafe_ptr()),
                        Int(bad.unsafe_ptr()), it, plan.drop_seed, it, plan.skip_thr)
            if bad[0] != 0:
                raise Error(_WALK_REFUSAL)
            _ = xt_dart_rescale(glue, Int(weights64.unsafe_ptr()), Int(coefs64.unsafe_ptr()),
                                Int(flags.unsafe_ptr()), Int(state.unsafe_ptr()), Int(rescaled.unsafe_ptr()), it, K,
                                plan.uniform, plan.xgb, plan.lr)
            var shrink = rescaled[1]
            var factor = rescaled[2]
            # `_dart_bag`: the round's bagged rows, redrawn once per window
            var m = 0
            if plan.bag:
                var start = it - it % plan.bag_freq
                if bag_start != start:
                    bag_count = xt_bag_rows(glue, Int(bag_rows.unsafe_ptr()), n, plan.bag_seed,
                                            start, plan.bag_frac)
                    bag_start = start
                m = bag_count
            var rows = Int(bag_rows.unsafe_ptr()) if plan.bag else 0
            for c in range(K):  # driver-loop: over the K member trees of one round
                var j = it * K + c
                # `_dart_cols`: the tree's sorted feature subset (none: all)
                var cols = 0
                var dc = d
                if plan.col_count > 0:
                    _ = xt_sample_indices(glue, Int(col_ids.unsafe_ptr()), d, plan.col_count, 0,
                                          plan.col_seed, j)
                    cols = Int(col_ids.unsafe_ptr())
                    dc = plan.col_count
                # The member fits every row when the round is not bagged (rows 0):
                # its row count is n, as `_dart_member_fit` passed (`m = n if rows is
                # None`). `m` stays 0 for xd_add (0 = every row). 71efbd8c7c passed
                # m = 0 here: the host entry refused 'invalid shape', the device one
                # fit zero rows.
                _member_fit(plan, args, handle, c, j, rows, m if plan.bag else n, cols, dc, got)
                var w = got.count() - 1
                var node = got.word(w, DART_M_NODE_START)
                var nodes = got.word(w, DART_M_NODES)
                var colid = Int(got.colid.unsafe_ptr()) + node * 4
                if cols != 0:
                    _ = xt_remap_cols(glue, colid, cols, nodes, dc)
                var off = Int(got.offsets.unsafe_ptr()) + got.word(w, DART_M_OFF_START) * 4
                var lo = _i32(off, 0)
                var n_nodes = _i32(off, 1) - lo
                var val = _grow_f32(got.values, n_nodes)
                got.members[w * DART_M_WORDS + DART_M_VAL_START] = val
                got.members[w * DART_M_WORDS + DART_M_VAL_N] = n_nodes
                var add_rows = rows if plan.bag else Int(all_cols.unsafe_ptr())
                _ = xd_add(plan.dart, handle, colid, Int(got.quesval.unsafe_ptr()) + node * 4,
                           Int(got.left.unsafe_ptr()) + node * 4, Int(got.values.unsafe_ptr()) + val * 4, add_rows,
                           j, c, lo, n_nodes, shrink, factor, plan.lam, plan.l1, plan.mds, m)
    finally:
        _ = xd_close(plan.dart, handle, Int(bad.unsafe_ptr()))
        # `bad` is written by the close itself (dart_close's bad word) and must
        # be named after it: on the refusal path (a round or member raising)
        # nothing below reads it, so Mojo freed it at its last use inside the
        # loop and the close wrote into freed storage (heap corruption; the
        # next free in dart_close segfaulted: trees-dart-clf/-reg/-shap-tree,
        # gdb trace of the 91d5 smoke, 2026-11-15).
        _ = len(bad)
        # the session and the step read these words through their addresses
        _ = len(inits32)
        _ = len(all_cols)
        _ = len(bag_rows)
        _ = len(col_ids)
        _ = len(thr64)
        _ = len(coef32)
        _ = len(flags)
        _ = len(state)
        _ = len(rescaled)
    if bad[0] != 0:
        raise Error(_WALK_REFUSAL)
    got.coefs = List[Float64](capacity=n_iters * K)
    for i in range(n_iters * K):  # small-loop(the tree coefficients, trees-sized): RUN result
        got.coefs.append(coefs64[i])
    got.weights = List[Float64](capacity=n_iters)
    for i in range(n_iters):  # small-loop(the tree weights, rounds-sized): RUN result
        got.weights.append(weights64[i])
    _ = len(args)
    return got^


def _copy_words(dst: Int, src: Int, count: Int):
    if count > 0:
        unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=dst),
                      src=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=src), count=count * 4)


def dart_member_into(rounds: DartRounds, member: Int, dst: List[Int]):
    """PACK helper: member `member`'s six arrays (offsets, colid, quesval,
    left_child, leaves, leaf values) into the caller's Arrays at `dst`,
    sized by the member's words."""
    var node = rounds.word(member, DART_M_NODE_START)
    var nodes = rounds.word(member, DART_M_NODES)
    _copy_words(dst[0], Int(rounds.offsets.unsafe_ptr()) + rounds.word(member, DART_M_OFF_START) * 4,
                rounds.word(member, DART_M_OFF_N))
    _copy_words(dst[1], Int(rounds.colid.unsafe_ptr()) + node * 4, nodes)
    _copy_words(dst[2], Int(rounds.quesval.unsafe_ptr()) + node * 4, nodes)
    _copy_words(dst[3], Int(rounds.left.unsafe_ptr()) + node * 4, nodes)
    _copy_words(dst[4], Int(rounds.leaves.unsafe_ptr()) + rounds.word(member, DART_M_LEAF_START) * 4,
                rounds.word(member, DART_M_LEAF_N))
    _copy_words(dst[5], Int(rounds.values.unsafe_ptr()) + rounds.word(member, DART_M_VAL_START) * 4,
                rounds.word(member, DART_M_VAL_N))


@fieldwise_init
struct DartWords(Copyable, Movable):
    """The estimator's DART settings as plain values (UNPACK, once per
    estimator: the door's `_dart_boost` and the Dart member's admission read
    the same fields with the same checks). `col_seed_refusal` holds the
    `random_state` refusal text of an invalid `feature_fraction_seed`, raised
    only when a fit samples columns (where the door reads that seed)."""
    var n_iters: Int
    var node_cap: Int
    var skip_thr: Int
    var uniform: Int
    var xgb: Int
    var max_drop: Int
    var drop_rate: Float64
    var lr: Float64
    var lam: Float64
    var l1: Float64
    var mds: Float64
    var bag: Bool
    var bag_freq: Int
    var bag_frac: Float64
    var bag_seed: Int
    var col_frac: Float64
    var col_seed: Int
    var col_seed_refusal: String
    var seed: Int
    var drop_seed: Int


def dart_starts(glue: EntryRef, y: Int, n: Int, K: Int, kind: Int) raises -> List[Float64]:
    """`_DARTBase._boost`'s class starts (`init_score_`): the exact mean of
    y (regression), its log-odds (binary), or the class priors (multiclass),
    through the image's glue (`glue` = xt_ref). Refuses a non-finite y and a
    binary y without both classes, in the estimator's words."""
    var starts = List[Float64]()
    if kind == 0 or kind == 1:
        var sum_out = List[Float64](length=1, fill=0.0)
        var sum_flags = List[Int32](length=8, fill=0)
        _ = xt_exact_sum(glue, y, Int(sum_out.unsafe_ptr()), Int(sum_flags.unsafe_ptr()), n, 0)
        if sum_flags[0] != 0 or sum_flags[1] != 0 or sum_flags[2] != 0:
            raise Error("Input y contains NaN or infinity.")
        var mean = sum_out[0] / Float64(n)
        if kind == 0:
            starts.append(mean)
        else:
            if not (0.0 < mean and mean < 1.0):
                raise Error("y must hold both classes")
            var log_word: List[Float64] = [0.0]
            _ = xt_log64(glue, Int(log_word.unsafe_ptr()), mean / (1.0 - mean))
            starts.append(log_word[0])
            _ = len(log_word)
        _ = len(sum_out)
        _ = len(sum_flags)
        return starts^
    var counts = List[Int32](length=K, fill=0)
    _ = xt_class_counts(glue, y, Int(counts.unsafe_ptr()), n, K)
    var priors = List[Float64](length=K, fill=0.0)
    _ = xt_class_priors(glue, Int(counts.unsafe_ptr()), Int(priors.unsafe_ptr()), n, K)
    _ = len(counts)
    return priors^


def dart_round_plan(words: DartWords, dart: EntryRef, glue_key: TreesEntryKey, forest: EntryRef, x: Int, y: Int,
                    var inits: List[Float64], n: Int, d: Int, K: Int, kind: Int) raises -> DartRoundPlan:
    """The round loop's plan from the settings and the fit's dimensions:
    the sampled column count `max(int(d * colsample_bytree + 0.5),
    min(1, d))` when below d (else every column), as the door computed it."""
    var col_count = 0
    var col_seed = 0
    if words.col_frac < 1.0:
        var count = max(Int(Float64(d) * words.col_frac + 0.5), min(1, d))
        if count < d:
            if words.col_seed_refusal != "":
                raise Error(words.col_seed_refusal)
            col_count = count
            col_seed = words.col_seed
    return DartRoundPlan(
        dart=dart, glue_key=glue_key.copy(), forest=forest, x=x, y=y, inits=inits^, n=n, d=d, k=K, kind=kind,
        n_iters=words.n_iters, node_cap=words.node_cap, drop_seed=words.drop_seed, skip_thr=words.skip_thr,
        uniform=words.uniform, xgb=words.xgb, max_drop=words.max_drop, drop_rate=words.drop_rate, lr=words.lr,
        lam=words.lam, l1=words.l1, mds=words.mds, bag=words.bag, bag_freq=words.bag_freq,
        bag_frac=words.bag_frac, bag_seed=words.bag_seed, col_count=col_count, col_seed=col_seed, seed=words.seed)
