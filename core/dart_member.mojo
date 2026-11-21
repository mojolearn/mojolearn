# SPDX-License-Identifier: Apache-2.0
"""DARTClassifier and DARTRegressor as MEMBERS (core/member_abi.mojo), family
17 (`MEMBER_FAMILY_DART`).

One entry, `dart_member_v2`, published by the x_trees images (the GPU
`_mojolearn_x_trees` and the host `_mojolearn_x_trees_host`, beside their DART
entry: `dart_member_publish_image`). It runs the estimator's EXISTING typed
DART RUN (bindings/trees_dart_rounds_run.mojo): `dart_starts` (the class
starts), `dart_round_plan`, `dart_rounds_run` (the whole round loop: the
x_trees DART session, the rf image's one-tree member fits), and the native
DART predict the estimator's own predict launches (XD_PREDICT, or the kept
device forest XD_FOREST_PREPARE / XD_FOREST_PREDICT where the image has
XD_CAP_RESIDENT), then the same glue steps (positive codes, transpose,
argmax, stack, softmax). No kernel is touched. Kind 0 is the classifier,
kind 1 the regressor.

Route slots: `route.entry` the x_trees DART entry (xd_ref), `route.second`
its glue entry (xt_ref), `route.third` the rf image's forest entry the
members fit in; `route.mode` / `route.vendor` the x_trees image's key.

MEMBER_OP_NEW parameters (Float64 words; "int" words travel as the Float64
bit pattern of their Int64, std.memory.bitcast, so seeds stay exact):
  [0] kind: 0 classifier, 1 regressor
  [1] n_estimators, [2] the leaf-value node cap (2 * num_leaves - 1)
  [3] int: skip_drop's 53-bit threshold (`_dart_thr`), [4] uniform_drop 0/1,
  [5] xgboost_dart_mode 0/1, [6] max_drop, [7] drop_rate
  [8] learning_rate, [9] reg_lambda, [10] reg_alpha, [11] max_delta_step
  [12] 1: row bagging (subsample_freq > 0 and subsample < 1),
  [13] subsample_freq, [14] subsample, [15] int: bagging_seed
  [16] colsample_bytree, [17] int: feature_fraction_seed, [18] 1: that seed
      is invalid (a fit that samples columns refuses it, as the estimator)
  [19] int: random_state (the members' seed base), [20] int: drop_seed
  [21] the member criterion word, [22] its max_features form, [23] count,
  [24] fraction, [25] 1: the member reads the image's bin count
  [26] S, the member's parameter slot count (0 when n_estimators is 0);
      then S triples [27 + 3k] int reading, [28 + 3k] float reading,
      [29 + 3k] 1 when the slot has no reading (core/forest_member_run.mojo)
Views: X dense row-major host float32 (finite: the x_trees open tests it);
y host int32 class codes 0..classes-1 (classifier, `call.classes` >= 2) or
a float32 target column (regressor). Refused in words: sample weights (the
estimator's fit takes none), a row or column index, a device-home view, a
nonzero `fit_output`, a multi-column y, a weighted score.
Outputs: predict writes int32 class codes (classifier) or the raw values
(regressor: float64, the estimator's dtype, or float32); predict_proba
float64 rows x classes; decision float64 rows x 1 (regressor, binary) or
rows x classes (multiclass); score (classifier only) the accuracy.

Fitted-state blocks (MEMBER_OP_EXPORT / MEMBER_OP_IMPORT), in index order:
  0..5 `MEMBER_BLOCK_FAMILY + 0..5` the members' arenas: offsets int32,
      colid int32, quesval float32, left_child int32, leaves float32, leaf
      values float32 (each 1 x words)
  6 `MEMBER_BLOCK_FAMILY + 6` float64 members x 11: each member's words
      (bindings/trees_dart_rounds_run.mojo DART_M_*)
  7 `MEMBER_BLOCK_FAMILY + 7` float64 1 x trees: the tree coefficients
  8 `MEMBER_BLOCK_FAMILY + 8` float64 1 x rounds: the tree weights
  9 `MEMBER_BLOCK_SCALARS` float64 [features, K, kind, classes, starts (K)]
An import needs all ten; the last one carries MEMBER_IMPORT_DONE.

The state is OWNED by its slot in a process-wide store of fixed chunks
(lock for mint, free and lookup only). The predict words (per-tree
addresses into the slot's arenas) and, on a resident image, the kept device
forest are built once: the words at the end of a fit or import, the device
forest at the first predict / predict_proba / decision / score of the state
(as the door), so a state only fitted and packed uploads nothing; both are
released at the next fit, import or free. SERIAL BY NATURE: the rounds (round t reads round
t-1's trees); the per-row work runs inside the images. Parallel: the class
code cast and the accuracy count split rows over the host workers.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy, bitcast
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.entry_table import EntryRef, entry_ref_unresolved
from core.trees_entry_table import TreesEntryKey
from core.forest_fit_request import ForestFitArgs
from bindings.forest_config_run import ForestConfig, forest_request
from bindings.trees_dart_rounds_run import (
    DartWords, DartRounds, dart_starts, dart_round_plan, dart_round_admit, dart_rounds_run,
    DART_M_OFF_START, DART_M_NODE_START, DART_M_VAL_START, DART_M_VAL_N, DART_M_WORDS,
)
from xtrees.trees_forest_run import ForestFit
from xtrees.dart_abi import XD_CAP_RESIDENT
from xtrees.dart_calls import xd_caps, xd_predict, xd_forest_prepare, xd_forest_predict, xd_forest_release
from xtrees.glue_calls import xt_positive_codes, xt_transpose_f64, xt_argmax_rows, xt_softmax_rows, xt_stack_w64
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_PREDICT_PROBA, MEMBER_OP_DECISION, MEMBER_OP_SCORE, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT,
    MEMBER_OP_IMPORT, MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_DECISION,
    MEMBER_CAN_SCORE, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT, MEMBER_CAN_CLASSIFIER,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_DART
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock
from std.atomic import Atomic, Ordering

comptime _STORE_NAME: StaticString = "MojolearnDartMemberStoreV2"
comptime _ESTIMATOR = "DART"
comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _HEAD = 27
comptime _BLOCKS = 10
comptime _SEED_REFUSAL = ("random_state must be None or a nonnegative int (a RandomState object is refused:"
    + " draws come from the lane's own counter RNG, xtrees/ops.mojo)")

comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]


def _int_word(words: _F64P, index: Int) -> Int:
    return Int(bitcast[DType.int64](words[index]))


struct DartMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var classifier: Bool
    var words: DartWords
    var criterion: Int
    var form: Int
    var count: Int
    var fraction: Float64
    var bins_image: Bool
    var ints: List[Int]
    var floats: List[Float64]
    var bad: List[UInt8]
    var fitted: Bool
    var features: Int
    var k: Int
    var kind: Int
    var classes: Int
    # The fitted members (DartRounds' arenas, owned here) and the starts.
    var offsets: List[Int32]
    var colid: List[Int32]
    var quesval: List[Float32]
    var left: List[Int32]
    var leaves: List[Float32]
    var values: List[Float32]
    var members: List[Int]
    var coefs: List[Float64]
    var weights: List[Float64]
    var starts: List[Float64]
    # The predict words (`_dart_forest`'s flat arrays) and the kept forest.
    var p_cols: List[Int64]
    var p_ques: List[Int64]
    var p_lefts: List[Int64]
    var p_vals: List[Int64]
    var p_sizes: List[Int64]
    var p_coefs: List[Float64]
    # The export forms of the member words and the scalars (float64).
    var x_members: List[Float64]
    var x_scalars: List[Float64]
    var trees: Int
    var dart: EntryRef
    var glue: EntryRef
    var snapshot: Int
    # The kept device forest is uploaded at the first predict of the fitted
    # state (lane fx-audit 27), as the door's `_dart_resident_predict`:
    # `prepared` (acquire / release) says it was decided, `prepare_lock`
    # serializes concurrent first predicts.
    var prepared: Int64
    var prepare_lock: Int64

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.classifier = False
        self.words = DartWords(0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, False, 0, 1.0, 0, 1.0, 0, String(""),
            0, 0)
        self.criterion = -1
        self.form = 0
        self.count = 0
        self.fraction = 0.0
        self.bins_image = False
        self.ints = List[Int]()
        self.floats = List[Float64]()
        self.bad = List[UInt8]()
        self.fitted = False
        self.features = 0
        self.k = 0
        self.kind = 0
        self.classes = 0
        self.offsets = List[Int32]()
        self.colid = List[Int32]()
        self.quesval = List[Float32]()
        self.left = List[Int32]()
        self.leaves = List[Float32]()
        self.values = List[Float32]()
        self.members = List[Int]()
        self.coefs = List[Float64]()
        self.weights = List[Float64]()
        self.starts = List[Float64]()
        self.p_cols = List[Int64]()
        self.p_ques = List[Int64]()
        self.p_lefts = List[Int64]()
        self.p_vals = List[Int64]()
        self.p_sizes = List[Int64]()
        self.p_coefs = List[Float64]()
        self.x_members = List[Float64]()
        self.x_scalars = List[Float64]()
        self.trees = 0
        self.dart = entry_ref_unresolved()
        self.glue = entry_ref_unresolved()
        self.snapshot = 0
        self.prepared = 0
        self.prepare_lock = 0


struct DartMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[DartMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        # Side-effect-free even when the process slot's initializers race.
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[DartMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _SlotPtr = MutPointer[DartMemberSlot, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock; a chunk's buffer never grows.
    var store = SharedSlot[DartMemberStore, _STORE_NAME, 2].get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[DartMemberSlot]()


def _clear(slot: _SlotPtr):
    slot[].fitted = False
    slot[].offsets = List[Int32]()
    slot[].colid = List[Int32]()
    slot[].quesval = List[Float32]()
    slot[].left = List[Int32]()
    slot[].leaves = List[Float32]()
    slot[].values = List[Float32]()
    slot[].members = List[Int]()
    slot[].coefs = List[Float64]()
    slot[].weights = List[Float64]()
    slot[].starts = List[Float64]()
    slot[].p_cols = List[Int64]()
    slot[].p_ques = List[Int64]()
    slot[].p_lefts = List[Int64]()
    slot[].p_vals = List[Int64]()
    slot[].p_sizes = List[Int64]()
    slot[].p_coefs = List[Float64]()
    slot[].x_members = List[Float64]()
    slot[].x_scalars = List[Float64]()
    slot[].trees = 0


def _mint(words: _F64P, count: Int) raises -> Int:
    if count < _HEAD:
        raise Error("member ABI: a DART member takes at least " + String(_HEAD) + " words")
    var slots = Int(words[26])
    if count != _HEAD + 3 * slots or (slots != 0 and slots < 3):
        raise Error("member ABI: a DART member takes " + String(_HEAD) + " words and three per member slot")
    if words[0] != 0.0 and words[0] != 1.0:
        raise Error("member ABI: unknown DART member kind")
    var store = SharedSlot[DartMemberStore, _STORE_NAME, 2].get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = -1
        if len(store[].free) > 0:
            index = store[].free.pop()
        else:
            if store[].count >= _INDEX_SPAN - 1:
                raise Error("RuntimeError: mojolearn: too many live DART member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[DartMemberSlot](length=_CHUNK, fill=DartMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _SlotPtr(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].classifier = words[0] == 0.0
        slot[].words = DartWords(
            n_iters=Int(words[1]), node_cap=Int(words[2]), skip_thr=_int_word(words, 3), uniform=Int(words[4]),
            xgb=Int(words[5]), max_drop=Int(words[6]), drop_rate=words[7], lr=words[8], lam=words[9],
            l1=words[10], mds=words[11], bag=words[12] != 0.0, bag_freq=Int(words[13]), bag_frac=words[14],
            bag_seed=_int_word(words, 15), col_frac=words[16], col_seed=_int_word(words, 17),
            col_seed_refusal=String(_SEED_REFUSAL) if words[18] != 0.0 else String(""),
            seed=_int_word(words, 19), drop_seed=_int_word(words, 20))
        slot[].criterion = Int(words[21])
        slot[].form = Int(words[22])
        slot[].count = Int(words[23])
        slot[].fraction = words[24]
        slot[].bins_image = words[25] != 0.0
        slot[].ints = List[Int](length=slots, fill=0)
        slot[].floats = List[Float64](length=slots, fill=0.0)
        slot[].bad = List[UInt8](length=slots, fill=0)
        for k in range(slots):  # small-loop(slots): about twenty parameter slots, once per state
            slot[].ints[k] = _int_word(words, _HEAD + 3 * k)
            slot[].floats[k] = words[_HEAD + 3 * k + 1]
            slot[].bad[k] = UInt8(1) if words[_HEAD + 3 * k + 2] != 0.0 else UInt8(0)
        _clear(slot)
        slot[].snapshot = 0
        return slot[].generation * _INDEX_SPAN + index + 1
    finally:
        shared_slot_unlock(lock)


def _lookup(handle: Int, release: Bool) raises -> Int:
    """The slot address of a live handle; a released, foreign or made-up
    handle is refused."""
    var store = SharedSlot[DartMemberStore, _STORE_NAME, 2].get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = handle % _INDEX_SPAN - 1
        if handle <= 0 or index < 0 or index >= store[].count:
            raise Error("NotFittedError: this member state was released or never created")
        var address = _slot_address(index)
        var slot = _SlotPtr(unsafe_from_address=address)
        if not slot[].live or slot[].generation != handle // _INDEX_SPAN:
            raise Error("NotFittedError: this member state was released or never created")
        if release:
            slot[].live = False
            _clear(slot)
            slot[].ints = List[Int]()
            slot[].floats = List[Float64]()
            slot[].bad = List[UInt8]()
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def _release_snapshot(slot: _SlotPtr):
    """The kept device forest back to its image (non-raising); the next
    predict decides it again."""
    slot[].prepared = 0
    if slot[].snapshot == 0:
        return
    var handle = slot[].snapshot
    slot[].snapshot = 0
    try:
        _ = xd_forest_release(slot[].dart, handle)
    except:
        pass


def _whole(call: MutPointer[MemberCall, MutUntrackedOrigin]) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a DART member reads whole host views (no row or column index)")


def _dense(view: MemberMatrix, dtype: Int, what: String) raises:
    if not view.dense_host(dtype):
        var words = String("float32") if dtype == MEMBER_F32 else (
            String("float64") if dtype == MEMBER_F64 else String("int32"))
        raise Error("TypeError: a DART member reads " + what + " as a dense host " + words + " view")


def _cast_codes(src: Int, dst: Int, rows: Int):
    """int32 class codes to the float32 codes the DART fit reads
    (`codes.astype('<f4')`), split over the host workers."""
    var tasks = host_predict_task_count_work(rows, rows)
    var chunk = host_predict_chunk(rows, tasks)
    var source = _I32P(unsafe_from_address=src)
    var target = _F32P(unsafe_from_address=dst)

    def task(unit: Int) {imm source, imm target, imm chunk, imm rows}:
        for i in range(unit * chunk, min((unit + 1) * chunk, rows)):  # native data loop (one host task's rows)
            target[i] = Float32(source[i])

    host_parallelize(task, tasks)


def _narrow(src: Int, dst: Int, rows: Int):
    """float64 values to float32, split over the host workers."""
    var tasks = host_predict_task_count_work(rows, rows)
    var chunk = host_predict_chunk(rows, tasks)
    var source = _F64P(unsafe_from_address=src)
    var target = _F32P(unsafe_from_address=dst)

    def task(unit: Int) {imm source, imm target, imm chunk, imm rows}:
        for i in range(unit * chunk, min((unit + 1) * chunk, rows)):  # native data loop (one host task's rows)
            target[i] = Float32(source[i])

    host_parallelize(task, tasks)


def _count_equal(left: Int, right: Int, rows: Int) -> Int:
    """Rows where two int32 code vectors agree (per-task counts summed)."""
    var tasks = host_predict_task_count_work(rows, rows)
    var chunk = host_predict_chunk(rows, tasks)
    var counts = List[Int](length=max(tasks, 1), fill=0)
    var a = _I32P(unsafe_from_address=left)
    var b = _I32P(unsafe_from_address=right)
    var sums = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(counts.unsafe_ptr()))

    def task(unit: Int) {imm a, imm b, imm sums, imm chunk, imm rows}:
        var hits = 0
        for i in range(unit * chunk, min((unit + 1) * chunk, rows)):  # native data loop (one host task's rows)
            if a[i] == b[i]:
                hits += 1
        sums[unit] = hits

    host_parallelize(task, tasks)
    var total = 0
    for t in range(len(counts)):  # small-loop(tasks): one count per host task
        total += counts[t]
    return total


def _prepare(slot: _SlotPtr, dart: EntryRef, glue: EntryRef) raises:
    """The predict words of the fitted members (`_dart_forest`: per tree the
    address of its colid / quesval / left_child from its first node, of its
    leaf values, its node count and its coefficient). Once per fit or
    import; the kept device forest waits for the first predict
    (`_ensure_snapshot`)."""
    slot[].dart = dart
    slot[].glue = glue
    slot[].x_members = List[Float64](length=len(slot[].members), fill=0.0)
    for w in range(len(slot[].members)):  # driver-loop: over the members' words (11 per tree), once per state
        slot[].x_members[w] = Float64(slot[].members[w])
    slot[].x_scalars = List[Float64](length=4 + len(slot[].starts), fill=0.0)
    slot[].x_scalars[0] = Float64(slot[].features)
    slot[].x_scalars[1] = Float64(slot[].k)
    slot[].x_scalars[2] = Float64(slot[].kind)
    slot[].x_scalars[3] = Float64(slot[].classes)
    for c in range(len(slot[].starts)):  # small-loop(K): the class starts
        slot[].x_scalars[4 + c] = slot[].starts[c]
    var members = len(slot[].members) // DART_M_WORDS
    var trees = min(members, len(slot[].coefs))
    slot[].trees = trees
    slot[].p_cols = List[Int64](length=max(trees, 1), fill=0)
    slot[].p_ques = List[Int64](length=max(trees, 1), fill=0)
    slot[].p_lefts = List[Int64](length=max(trees, 1), fill=0)
    slot[].p_vals = List[Int64](length=max(trees, 1), fill=0)
    slot[].p_sizes = List[Int64](length=max(trees, 1), fill=0)
    slot[].p_coefs = List[Float64](length=max(trees, 1), fill=0.0)
    var colid = Int(slot[].colid.unsafe_ptr())
    var quesval = Int(slot[].quesval.unsafe_ptr())
    var left = Int(slot[].left.unsafe_ptr())
    var values = Int(slot[].values.unsafe_ptr())
    for i in range(trees):  # driver-loop: over the fitted trees, once per fitted state
        var base = i * DART_M_WORDS
        var off = slot[].members[base + DART_M_OFF_START]
        var lo = Int(slot[].offsets[off])
        var node = slot[].members[base + DART_M_NODE_START] + lo
        slot[].p_cols[i] = Int64(colid + 4 * node)
        slot[].p_ques[i] = Int64(quesval + 4 * node)
        slot[].p_lefts[i] = Int64(left + 4 * node)
        slot[].p_vals[i] = Int64(values + 4 * slot[].members[base + DART_M_VAL_START])
        slot[].p_sizes[i] = Int64(Int(slot[].offsets[off + 1]) - lo)
        slot[].p_coefs[i] = slot[].coefs[i]
    # Called after `_release_snapshot`, the state owned by its fitting or
    # importing caller (no predict runs concurrently).
    slot[].prepared = 0


def _ensure_snapshot(slot: _SlotPtr) raises:
    """On a resident image, the kept device forest of the fitted state,
    uploaded at its first predict (the door's `_dart_resident_predict`).
    After the first call: one acquire load."""
    var flag = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(UnsafePointer(to=slot[].prepared)))
    if Atomic.load[ordering=Ordering.ACQUIRE](flag) != 0:
        return
    var lock = MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=Int(UnsafePointer(to=slot[].prepare_lock)))
    shared_slot_lock(lock)
    try:
        if Atomic.load[ordering=Ordering.ACQUIRE](flag) == 0:
            if (xd_caps(slot[].dart) & XD_CAP_RESIDENT) != 0:
                var made: List[Int] = [0, 0]
                slot[].snapshot = xd_forest_prepare(slot[].dart, Int(slot[].p_cols.unsafe_ptr()),
                    Int(slot[].p_ques.unsafe_ptr()), Int(slot[].p_lefts.unsafe_ptr()), Int(slot[].p_vals.unsafe_ptr()),
                    Int(slot[].p_sizes.unsafe_ptr()), Int(slot[].p_coefs.unsafe_ptr()),
                    Int(slot[].starts.unsafe_ptr()), Int(made.unsafe_ptr()), slot[].features, slot[].k, slot[].trees)
                _ = len(made)
            Atomic.store[ordering=Ordering.RELEASE](flag, Int64(1))
    finally:
        shared_slot_unlock(lock)


def _fit(slot: _SlotPtr, call: MutPointer[MemberCall, MutUntrackedOrigin]) raises:
    var rows = call[].x.rows
    var cols = call[].x.cols
    _whole(call)
    if call[].fit_output != 0:
        member_unsupported(_ESTIMATOR, call[].fit_output)
    if call[].weights.rows > 0:
        raise Error("TypeError: a DART fit takes no sample_weight")
    _dense(call[].x, MEMBER_F32, "X")
    if call[].y.cols != 1:
        raise Error("ValueError: a DART member fits one target column")
    if call[].y.rows != rows:
        raise Error("ValueError: Found input variables with inconsistent numbers of samples")
    var classes = 0
    var K = 1
    var kind = 0
    if slot[].classifier:
        _dense(call[].y, MEMBER_I32, "y (class codes)")
        classes = call[].classes
        if classes < 2:
            raise Error("y must hold at least two classes")
        kind = 1 if classes == 2 else 2
        K = 1 if classes == 2 else classes
    else:
        _dense(call[].y, MEMBER_F32, "y")
    _release_snapshot(slot)
    _clear(slot)
    var dart = call[].route.entry
    var glue = call[].route.second
    var forest = call[].route.third
    var glue_key = TreesEntryKey(call[].route.mode, call[].route.vendor)
    # The DART fit reads float32 targets / class codes; the cast lives until the run returns.
    var codes = List[Float32]()
    var y_addr = call[].y.data
    if slot[].classifier:
        codes = List[Float32](length=rows, fill=0.0)
        _cast_codes(call[].y.data, Int(codes.unsafe_ptr()), rows)
        y_addr = Int(codes.unsafe_ptr())
    var starts = dart_starts(glue, y_addr, rows, K, kind)
    dart_round_admit(dart, slot[].words.n_iters, K, slot[].words.node_cap)
    var request = ForestFitArgs(0, 0, List[Int](), List[Float64](), List[UInt8](), -1)
    var member_entry = dart
    if slot[].words.n_iters * K > 0:
        if len(slot[].ints) < 3:
            raise Error("member ABI: a DART member with trees needs its member slots")
        var config = ForestConfig(False, slot[].ints.copy(), slot[].floats.copy(), slot[].bad.copy())
        config.criterion = slot[].criterion
        config.features.form = slot[].form
        config.features.count = slot[].count
        config.features.fraction = slot[].fraction
        config.bins_image = slot[].bins_image
        request = forest_request(config, forest, rows, cols, 0, 0, 0, ForestFit(False), False)
        member_entry = forest
    var plan = dart_round_plan(slot[].words, dart, glue_key, member_entry, call[].x.data, y_addr, starts.copy(),
        rows, cols, K, kind)
    var rounds = dart_rounds_run(plan, request^)
    _ = len(codes)
    swap(slot[].offsets, rounds.offsets)
    swap(slot[].colid, rounds.colid)
    swap(slot[].quesval, rounds.quesval)
    swap(slot[].left, rounds.left)
    swap(slot[].leaves, rounds.leaves)
    swap(slot[].values, rounds.values)
    swap(slot[].members, rounds.members)
    swap(slot[].coefs, rounds.coefs)
    swap(slot[].weights, rounds.weights)
    slot[].starts = starts^
    slot[].features = cols
    slot[].k = K
    slot[].kind = kind
    slot[].classes = classes
    _prepare(slot, dart, glue)
    slot[].fitted = True


def _require_fitted(slot: _SlotPtr) raises:
    if not slot[].fitted:
        raise Error("NotFittedError: this DART member is not fitted yet")


def _width(slot: _SlotPtr, call: MutPointer[MemberCall, MutUntrackedOrigin]) raises:
    _whole(call)
    _dense(call[].x, MEMBER_F32, "X")
    if call[].x.cols != slot[].features:
        raise Error("ValueError: X has " + String(call[].x.cols) + " features, but DART is expecting "
            + String(slot[].features) + " features as input.")


def _raw(slot: _SlotPtr, x: MemberMatrix, dst: Int) raises:
    """The class-major raw scores (K * rows float64 at `dst`): the kept
    forest or the per-call walk, as the estimator's `_dart_raw`."""
    var n = x.rows
    if n == 0:
        return
    _ensure_snapshot(slot)
    if slot[].snapshot != 0:
        _ = xd_forest_predict(slot[].dart, slot[].snapshot, x.data, dst, n, x.cols, slot[].k)
        return
    _ = xd_predict(slot[].dart, x.data, dst, Int(slot[].p_cols.unsafe_ptr()), Int(slot[].p_ques.unsafe_ptr()),
        Int(slot[].p_lefts.unsafe_ptr()), Int(slot[].p_vals.unsafe_ptr()), Int(slot[].p_sizes.unsafe_ptr()),
        Int(slot[].p_coefs.unsafe_ptr()), Int(slot[].starts.unsafe_ptr()), n, x.cols, slot[].k, slot[].trees)


def _codes(slot: _SlotPtr, x: MemberMatrix, dst: Int) raises:
    """A classifier's int32 class codes (the estimator's `dart_codes`)."""
    var n = x.rows
    var K = slot[].k
    var raw = List[Float64](length=max(K * n, 1), fill=0.0)
    _raw(slot, x, Int(raw.unsafe_ptr()))
    if n == 0:
        return
    if K == 1:
        _ = xt_positive_codes(slot[].glue, Int(raw.unsafe_ptr()), dst, n)
        _ = len(raw)
        return
    var acc = List[Float64](length=K * n, fill=0.0)
    _ = xt_transpose_f64(slot[].glue, Int(raw.unsafe_ptr()), Int(acc.unsafe_ptr()), K, n)
    _ = xt_argmax_rows(slot[].glue, Int(acc.unsafe_ptr()), dst, n, K)
    _ = len(raw)
    _ = len(acc)


def _predict(slot: _SlotPtr, call: MutPointer[MemberCall, MutUntrackedOrigin]) raises:
    _require_fitted(slot)
    _width(slot, call)
    var rows = call[].x.rows
    if call[].dst.rows != rows or call[].dst.cols != 1:
        raise Error("ValueError: the prediction buffer holds one value per row of X")
    if slot[].classifier:
        _dense(call[].dst, MEMBER_I32, "the prediction buffer (class codes)")
        _codes(slot, call[].x, call[].dst.data)
        return
    if call[].dst.dtype == MEMBER_F64:
        _dense(call[].dst, MEMBER_F64, "the prediction buffer")
        _raw(slot, call[].x, call[].dst.data)
        return
    _dense(call[].dst, MEMBER_F32, "the prediction buffer")
    var raw = List[Float64](length=max(rows, 1), fill=0.0)
    _raw(slot, call[].x, Int(raw.unsafe_ptr()))
    _narrow(Int(raw.unsafe_ptr()), call[].dst.data, rows)
    _ = len(raw)


def _decision(slot: _SlotPtr, call: MutPointer[MemberCall, MutUntrackedOrigin], proba: Bool) raises:
    """decision_function (raw scores, rows x K row-major) or predict_proba
    (the estimator's stack + softmax), float64."""
    _require_fitted(slot)
    _width(slot, call)
    if proba and not slot[].classifier:
        member_unsupported(_ESTIMATOR, MEMBER_OP_PREDICT_PROBA)
    var rows = call[].x.rows
    var K = slot[].k
    var width = (2 if K == 1 else K) if proba else K
    _dense(call[].dst, MEMBER_F64, "the probability buffer" if proba else "the decision buffer")
    if call[].dst.rows != rows or call[].dst.cols != width:
        raise Error("ValueError: the output buffer holds " + String(width) + " values per row of X")
    if rows == 0:
        return
    if K == 1 and not proba:
        _raw(slot, call[].x, call[].dst.data)
        return
    var raw = List[Float64](length=K * rows, fill=0.0)
    _raw(slot, call[].x, Int(raw.unsafe_ptr()))
    if K == 1:
        var zero = List[Float64](length=rows, fill=0.0)
        var columns: List[Int] = [call[].dst.data, Int(zero.unsafe_ptr()), Int(raw.unsafe_ptr())]
        _ = xt_stack_w64(slot[].glue, columns, rows)
        _ = xt_softmax_rows(slot[].glue, call[].dst.data, rows, 2)
        _ = len(zero)
        _ = len(raw)
        return
    _ = xt_transpose_f64(slot[].glue, Int(raw.unsafe_ptr()), call[].dst.data, K, rows)
    _ = len(raw)
    if proba:
        _ = xt_softmax_rows(slot[].glue, call[].dst.data, rows, K)


def _score(slot: _SlotPtr, call: MutPointer[MemberCall, MutUntrackedOrigin]) raises:
    """A classifier's accuracy; a regressor's score is the caller's metric."""
    _require_fitted(slot)
    if not slot[].classifier:
        member_unsupported(_ESTIMATOR, MEMBER_OP_SCORE)
    if call[].weights.rows > 0:
        raise Error("NotImplementedError: mojolearn: a weighted score of a DART member is not published; "
            + "the caller predicts and runs its own weighted metric")
    _width(slot, call)
    _dense(call[].y, MEMBER_I32, "y (class codes)")
    var rows = call[].x.rows
    if call[].y.rows != rows or call[].y.cols != 1:
        raise Error("ValueError: Found input variables with inconsistent numbers of samples")
    var codes = List[Int32](length=max(rows, 1), fill=0)
    _codes(slot, call[].x, Int(codes.unsafe_ptr()))
    var hits = _count_equal(Int(codes.unsafe_ptr()), call[].y.data, rows)
    _ = len(codes)
    call[].value = Float64(hits) / Float64(rows) if rows > 0 else 0.0


def _block_of(index: Int) raises -> Int:
    if index < 0 or index >= _BLOCKS:
        raise Error("member ABI: a DART member has ten fitted blocks")
    return MEMBER_BLOCK_SCALARS if index == 9 else MEMBER_BLOCK_FAMILY + index


def _export(slot: _SlotPtr, call: MutPointer[MemberCall, MutUntrackedOrigin]) raises:
    """A view of one fitted block (the slot owns the words; valid until the
    state's next fit, import or free). Member words and the scalars are
    exported from their float64 forms (`_prepare`)."""
    _require_fitted(slot)
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        block = _block_of(block)
    if block == MEMBER_BLOCK_FAMILY:
        call[].dst = MemberMatrix(Int(slot[].offsets.unsafe_ptr()), 1, len(slot[].offsets), MEMBER_I32)
    elif block == MEMBER_BLOCK_FAMILY + 1:
        call[].dst = MemberMatrix(Int(slot[].colid.unsafe_ptr()), 1, len(slot[].colid), MEMBER_I32)
    elif block == MEMBER_BLOCK_FAMILY + 2:
        call[].dst = MemberMatrix(Int(slot[].quesval.unsafe_ptr()), 1, len(slot[].quesval), MEMBER_F32)
    elif block == MEMBER_BLOCK_FAMILY + 3:
        call[].dst = MemberMatrix(Int(slot[].left.unsafe_ptr()), 1, len(slot[].left), MEMBER_I32)
    elif block == MEMBER_BLOCK_FAMILY + 4:
        call[].dst = MemberMatrix(Int(slot[].leaves.unsafe_ptr()), 1, len(slot[].leaves), MEMBER_F32)
    elif block == MEMBER_BLOCK_FAMILY + 5:
        call[].dst = MemberMatrix(Int(slot[].values.unsafe_ptr()), 1, len(slot[].values), MEMBER_F32)
    elif block == MEMBER_BLOCK_FAMILY + 6:
        call[].dst = MemberMatrix(Int(slot[].x_members.unsafe_ptr()), len(slot[].x_members) // DART_M_WORDS,
            DART_M_WORDS, MEMBER_F64)
    elif block == MEMBER_BLOCK_FAMILY + 7:
        call[].dst = MemberMatrix(Int(slot[].coefs.unsafe_ptr()), 1, len(slot[].coefs), MEMBER_F64)
    elif block == MEMBER_BLOCK_FAMILY + 8:
        call[].dst = MemberMatrix(Int(slot[].weights.unsafe_ptr()), 1, len(slot[].weights), MEMBER_F64)
    elif block == MEMBER_BLOCK_SCALARS:
        call[].dst = MemberMatrix(Int(slot[].x_scalars.unsafe_ptr()), 1, len(slot[].x_scalars), MEMBER_F64)
    else:
        raise Error("member ABI: a DART member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = _BLOCKS


def _copy_i32(view: MemberMatrix) raises -> List[Int32]:
    _dense(view, MEMBER_I32, "a fitted block")
    var count = view.rows * view.cols
    var words = List[Int32](length=count, fill=0)
    unsafe_memcpy(dest=words.unsafe_ptr(), src=_I32P(unsafe_from_address=view.data), count=count)
    return words^


def _copy_f32(view: MemberMatrix) raises -> List[Float32]:
    _dense(view, MEMBER_F32, "a fitted block")
    var count = view.rows * view.cols
    var words = List[Float32](length=count, fill=0.0)
    unsafe_memcpy(dest=words.unsafe_ptr(), src=_F32P(unsafe_from_address=view.data), count=count)
    return words^


def _copy_f64(view: MemberMatrix) raises -> List[Float64]:
    _dense(view, MEMBER_F64, "a fitted block")
    var count = view.rows * view.cols
    var words = List[Float64](length=count, fill=0.0)
    unsafe_memcpy(dest=words.unsafe_ptr(), src=_F64P(unsafe_from_address=view.data), count=count)
    return words^


def _import(slot: _SlotPtr, call: MutPointer[MemberCall, MutUntrackedOrigin]) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE checks the members'
    words against the arenas, prepares the predict words and marks the
    state fitted."""
    var block = call[].block
    if slot[].fitted:
        _release_snapshot(slot)
        _clear(slot)
    if block == MEMBER_BLOCK_FAMILY:
        slot[].offsets = _copy_i32(call[].x)
    elif block == MEMBER_BLOCK_FAMILY + 1:
        slot[].colid = _copy_i32(call[].x)
    elif block == MEMBER_BLOCK_FAMILY + 2:
        slot[].quesval = _copy_f32(call[].x)
    elif block == MEMBER_BLOCK_FAMILY + 3:
        slot[].left = _copy_i32(call[].x)
    elif block == MEMBER_BLOCK_FAMILY + 4:
        slot[].leaves = _copy_f32(call[].x)
    elif block == MEMBER_BLOCK_FAMILY + 5:
        slot[].values = _copy_f32(call[].x)
    elif block == MEMBER_BLOCK_FAMILY + 6:
        if call[].x.cols != DART_M_WORDS:
            raise Error("member ABI: the DART member words block has " + String(DART_M_WORDS) + " words per member")
        var words = _copy_f64(call[].x)
        slot[].members = List[Int](length=len(words), fill=0)
        for w in range(len(words)):  # driver-loop: over the members' words (11 per tree), once per import
            slot[].members[w] = Int(words[w])
    elif block == MEMBER_BLOCK_FAMILY + 7:
        slot[].coefs = _copy_f64(call[].x)
    elif block == MEMBER_BLOCK_FAMILY + 8:
        slot[].weights = _copy_f64(call[].x)
    elif block == MEMBER_BLOCK_SCALARS:
        var scalars = _copy_f64(call[].x)
        if len(scalars) < 5 or len(scalars) != 4 + Int(scalars[1]):
            raise Error("member ABI: the DART scalars block is [features, K, kind, classes, starts (K)]")
        slot[].features = Int(scalars[0])
        slot[].k = Int(scalars[1])
        slot[].kind = Int(scalars[2])
        slot[].classes = Int(scalars[3])
        slot[].starts = List[Float64](length=slot[].k, fill=0.0)
        for c in range(slot[].k):  # small-loop(K): the class starts
            slot[].starts[c] = scalars[4 + c]
    else:
        raise Error("member ABI: a DART member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    var members = len(slot[].members) // DART_M_WORDS
    if slot[].features < 1 or slot[].k < 1 or len(slot[].starts) != slot[].k:
        raise Error("member ABI: a fitted DART state needs its ten blocks")
    if slot[].classifier != (slot[].kind != 0):
        raise Error("member ABI: the DART state's kind differs from the member's")
    for i in range(members):  # driver-loop: over the imported trees (bounds of each, once per import)
        var base = i * DART_M_WORDS
        var off = slot[].members[base + DART_M_OFF_START]
        if off < 0 or off + 1 >= len(slot[].offsets):
            raise Error("member ABI: a DART member's offsets lie outside the offsets block")
        var lo = Int(slot[].offsets[off])
        var hi = Int(slot[].offsets[off + 1])
        var node = slot[].members[base + DART_M_NODE_START]
        var value = slot[].members[base + DART_M_VAL_START]
        if (lo < 0 or hi < lo or node < 0 or node + hi > len(slot[].colid) or node + hi > len(slot[].quesval)
                or node + hi > len(slot[].left) or value < 0
                or value + slot[].members[base + DART_M_VAL_N] > len(slot[].values)):
            raise Error("member ABI: a DART member's nodes lie outside the arenas")
    _prepare(slot, call[].route.entry, call[].route.second)
    slot[].fitted = True


def _capabilities(classifier: Bool) -> Int:
    """No `MEMBER_CAN_TEST_ROWS`: the fit tests X at the x_trees open
    (xtrees/dart_host.mojo:125, xtrees/dart_device.mojo:429) and y at
    `dart_starts`, but the predict ops (XD_PREDICT, XD_FOREST_PREDICT) do
    not test X, and a fit of zero trees opens no session. No sample weight:
    the estimator's fit takes none."""
    var mask = MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_DECISION | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
    if classifier:
        mask |= MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_SCORE | MEMBER_CAN_CLASSIFIER
    return mask


def _member(operation: Int, address: Int) raises -> Int:
    var call = MutPointer[MemberCall, MutUntrackedOrigin](unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        if call[].state == 0:
            return _capabilities(True)
        return _capabilities(_SlotPtr(unsafe_from_address=_lookup(call[].state, False))[].classifier)
    if operation == MEMBER_OP_NEW:
        if not call[].params:
            raise Error("member ABI: a DART member needs its parameter words")
        call[].state = _mint(_F64P(unsafe_from_address=call[].params), call[].param_count)
        return 0
    if operation == MEMBER_OP_FREE:
        var held = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
        _release_snapshot(held)
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    if operation == MEMBER_OP_OUTPUTS:
        # With a state the state answers; with none, the kind word of the NEW
        # parameters (word 0) and the caller's `call.classes` (item 10).
        var classes = call[].classes
        var classifier: Bool
        if call[].state != 0:
            var known = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
            classifier = known[].classifier
            if known[].fitted or known[].classes > 0:
                classes = known[].classes
        else:
            if not call[].params or call[].param_count < 1:
                raise Error("member ABI: a DART member answers OUTPUTS without a state from its parameter words")
            classifier = _F64P(unsafe_from_address=call[].params)[0] == 0.0
        call[].classes = classes
        var multi = classifier and classes > 2
        if call[].flags == MEMBER_OP_PREDICT_PROBA:
            call[].out_cols = classes
            if classifier:
                call[].out_dtype = MEMBER_F64
        elif call[].flags == MEMBER_OP_DECISION:
            call[].out_cols = classes if multi else 1
            call[].out_dtype = MEMBER_F64
        else:
            call[].out_cols = 1
            # predict: int32 codes; a regressor writes float64 or float32
            # (`_predict`): float32 is said, the type callers place.
            if call[].flags == MEMBER_OP_PREDICT:
                call[].out_dtype = MEMBER_I32 if classifier else MEMBER_F32
        return 0
    var slot = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
    if operation == MEMBER_OP_FIT:
        _fit(slot, call)
    elif operation == MEMBER_OP_PREDICT:
        _predict(slot, call)
    elif operation == MEMBER_OP_PREDICT_PROBA:
        _decision(slot, call, True)
    elif operation == MEMBER_OP_DECISION:
        _decision(slot, call, False)
    elif operation == MEMBER_OP_SCORE:
        _score(slot, call)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import(slot, call)
    else:
        member_unsupported(_ESTIMATOR, operation)
    return 0


def dart_member_v2(operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of DARTClassifier and DARTRegressor (schema `MEMBER_ABI`)."""
    try:
        var word = _member(operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def dart_member_publish(mode: Int, vendor: Int, publisher: Int) raises:
    """Publish this image's Dart member entry."""
    member_publish[MEMBER_FAMILY_DART](mode, vendor, publisher, dart_member_v2)


def dart_member_publish_image(vendor_word: Int) raises:
    """The one publication line of an x_trees image's module initialization,
    after its DART entry: the image's tier and the vendor word of its glue
    table key (`x_trees_glue_vendor`; -1 publishes nothing, as the DART
    entry)."""
    if vendor_word < 0:
        return
    var store = SharedSlot[DartMemberStore, _STORE_NAME, 2].get_or_create_ptr()
    dart_member_publish(Int(GLOBAL_NUMERIC_MODE), vendor_word, Int(store))
