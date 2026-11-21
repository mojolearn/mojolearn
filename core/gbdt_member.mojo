# SPDX-License-Identifier: Apache-2.0
"""GradientBoosting as a MEMBER (core/member_abi.mojo), family 16
(`MEMBER_FAMILY_GBDT`).

One entry, `gbdt_member_v2[I]`, published by the gbdt images (the GPU
`_mojolearn_gbdt` and the host `_mojolearn_gbdt_host`, beside their gbdt
entry: `gbdt_member_publish_image[I]`). It runs the estimator's EXISTING
typed fit RUN: the typed config words (bindings/gbdt_config_run.mojo, the
same `gbdt_fit_words` / `gbdt_fit_options` the fit door calls), then
`xg_fit` and `xg_model_dim` through `call.route.entry` (the gbdt entry of
the image; gbdt/fit_calls.mojo), and the estimator's own predict route:
the image's resident model (XG_RESIDENT_PREPARE once per fitted state, on
its first predict, XG_RESIDENT_PREDICT per call) where the image holds one
(XG_CAP_RESIDENT), else the per-call parse (XG_PREDICT / XG_PREDICT_MULTI)
of a column-major X, as `GradientBoosting.predict` / `predict_proba`. No
kernel is touched. Three kinds, one fitted form (the booster's model
text):
  0 `GradientBoostingClassifier` (binary Logloss; y int32 class codes 0 / 1
    with `call.classes` 2; predict int32 codes and predict_proba float32
    rows x 2 through the image's XG_BINARY transform of the margins, as the
    adapter's `_binary_output`; decision float32 rows x 1, the raw margin;
    score the unweighted accuracy, float32 count / rows as the adapter's
    metric; MEMBER_CAN_CLASSIFIER),
  1 `GradientBoosting` (the booster: its own predict returns raw scores),
  2 `GradientBoostingRegressor` (RMSE; predict float32 rows x 1; no score:
    the caller predicts and runs its own R^2).
The adapters' NEW words are their inner learner's config (the learner the
adapter's fit builds), so their fit is the learner's fit.

Route slots: `route.entry` the gbdt entry; `route.mode` / `route.vendor`
the image's key. `I.release(handle)` returns a resident model to the
image's registry (under the image's registry lock).

MEMBER_OP_NEW parameters (Float64 words): [0] kind (0, 1, 2), then the
`GbdtConfig` words in `gbdt_config_words` order (bindings/gbdt_config_run.mojo:
ints as the bit pattern of their Int64, booleans 0/1, strings as a length
and one word per byte, lists as a length and their words).
Views: X dense row-major host float32 (rows x features; finite or NaN:
nan_mode decides, as the estimator); y a host float32 column (the target,
or the class codes of a classification loss); optional `weights` a host
float32 column (MEMBER_CAN_SAMPLE_WEIGHT). Refused in words: a row or
column index, a device-home view, a nonzero `fit_output`, a multi-column y.
Outputs: predict float32 rows x approx_dim (raw scores); predict_proba
float64 rows x 2 (Logloss, CrossEntropy) or float32 rows x classes
(MultiClass softmax, MultiClassOneVsAll sigmoid); other losses refuse it in
the estimator's words.

Fitted-state blocks (MEMBER_OP_EXPORT / MEMBER_OP_IMPORT), in index order:
  0 `MEMBER_BLOCK_FAMILY + 0` int32 1 x ceil(bytes / 4): the model text's
    UTF-8 bytes, four per word, little-endian, zero padded
  1 `MEMBER_BLOCK_FAMILY + 1` float64 1 x iterations: the learn loss curve
  2 `MEMBER_BLOCK_SCALARS` float64 [features, approx_dim, n_classes (0 =
    None), bias, best_iteration, stopped_early, learning_rate, text bytes,
    boosting (0 Plain, 1 Ordered)]
An import needs all three; the last one carries MEMBER_IMPORT_DONE.

LIFETIME: the state owns everything it reads after FIT (the model text and
the curves); it keeps no pointer into a fit buffer (no MEMBER_BORROWS_FIT).
The state is OWNED by its slot in a process-wide store of fixed chunks
(lock for mint, free and lookup only). SERIAL BY NATURE: the boosting rounds
(round t reads round t-1's trees) run inside the image; the member's own
per-row work (the code scan, the NaN scan, the column-major copy, the
widening cast) splits rows or columns over the host workers.
"""
from std.sys.info import size_of
from std.memory import unsafe_memcpy, bitcast
from std.math import isfinite
from checks.numerics import GLOBAL_NUMERIC_MODE
from core.entry_table import EntryRef, entry_ref_gil_free
from core.gbdt_entry_table import gbdt_vendor_word
from gbdt.fit_request import GbdtFitArgs, gbdt_string_at
from gbdt.fit_calls import (
    xg_row_major, xg_model_dim, xg_fit, xg_resident, xg_resident_prepare, xg_resident_predict, xg_predict,
    xg_predict_multi, xg_sigmoid_pair, xg_binary,
)
from gbdt.trees_gbdt_run import gbdt_text_has_ctr
from gbdt.trees_gbdt_fit_run import gbdt_bias_of
from bindings.gbdt_config_run import GbdtConfig, gbdt_config_of_words, gbdt_fit_words, gbdt_fit_options
from bindings.trees_public_inputs_run import trees_finite_f32
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk
from core.member_abi import (
    MemberCall, MemberMatrix, MEMBER_F32, MEMBER_F64, MEMBER_I32,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_NEW, MEMBER_OP_FREE, MEMBER_OP_FIT, MEMBER_OP_PREDICT,
    MEMBER_OP_PREDICT_PROBA, MEMBER_OP_OUTPUTS, MEMBER_OP_EXPORT, MEMBER_OP_IMPORT, MEMBER_OP_DECISION,
    MEMBER_OP_SCORE,
    MEMBER_CAN_FIT, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_PROBA, MEMBER_CAN_EXPORT, MEMBER_CAN_IMPORT,
    MEMBER_CAN_SAMPLE_WEIGHT, MEMBER_HOME_DEVICE, MEMBER_GIL_FREE, MEMBER_CAN_DECISION, MEMBER_CAN_SCORE,
    MEMBER_CAN_CLASSIFIER,
    MEMBER_BLOCK_SCALARS, MEMBER_BLOCK_FAMILY, MEMBER_BLOCK_BY_INDEX, MEMBER_IMPORT_DONE,
    member_publish, member_call_from_spans, member_unsupported,
)
from core.member_families import MEMBER_FAMILY_GBDT
from core.native_descriptor_abi import NativeDescriptorError
from core.native_descriptor_call import native_descriptor_report
from core.shared_slot import SharedSlot, shared_slot_lock, shared_slot_unlock

comptime _STORE_NAME: StaticString = "MojolearnGbdtMemberStoreV2"
comptime _ESTIMATOR = "GradientBoosting"
#: the member kinds (the classes' `_mojolearn_member` words)
comptime _KIND_CLASSIFIER = 0
comptime _KIND_BOOSTER = 1
comptime _KIND_REGRESSOR = 2
comptime _CHUNK = 256
comptime _INDEX_SPAN = 16777216
comptime _BLOCKS = 3
comptime _SCALARS = 9
#: the image's predict modes (bindings/trees_public_gbdt_door.mojo `_PREDICT_*`)
comptime _PREDICT_RAW = 0
comptime _PREDICT_SOFTMAX = 1
comptime _PREDICT_SIGMOID = 2
comptime _PREDICT_SIGMOID_PAIR = 3

comptime _I32P = MutPointer[Int32, MutUntrackedOrigin]
comptime _F32P = MutPointer[Float32, MutUntrackedOrigin]
comptime _F64P = MutPointer[Float64, MutUntrackedOrigin]
comptime _CallPtr = MutPointer[MemberCall, MutUntrackedOrigin]


trait GbdtMemberImage:
    """What an image adds to the member: the return of a resident model to
    its registry (the host image holds none and never gets a handle)."""

    @staticmethod
    def release(handle: Int) raises:
        ...


struct GbdtMemberSlot(Copyable, Movable, Defaultable):
    var generation: Int
    var live: Bool
    var kind: Int
    var config: GbdtConfig
    var fitted: Bool
    var text: String
    var learn: List[Float64]
    var features: Int
    var dim: Int
    var classes: Int
    var bias: Float64
    var best_iteration: Int
    var stopped_early: Bool
    var rate: Float64
    var ordered: Bool
    var resident: Int
    # Export forms, built at export.
    var x_text: List[Int32]
    var x_scalars: List[Float64]

    def __init__(out self):
        self.generation = 0
        self.live = False
        self.kind = _KIND_BOOSTER
        self.config = GbdtConfig()
        self.fitted = False
        self.text = String("")
        self.learn = List[Float64]()
        self.features = 0
        self.dim = 0
        self.classes = 0
        self.bias = 0.0
        self.best_iteration = 0
        self.stopped_early = False
        self.rate = 0.0
        self.ordered = False
        self.resident = -1
        self.x_text = List[Int32]()
        self.x_scalars = List[Float64]()


struct GbdtMemberStore(Defaultable, Movable):
    var synchronization: List[Int64]
    var chunks: List[List[GbdtMemberSlot]]
    var free: List[Int]
    var count: Int

    def __init__(out self):
        self.synchronization = List[Int64](length=2, fill=0)
        self.chunks = List[List[GbdtMemberSlot]]()
        self.free = List[Int]()
        self.count = 0


comptime _SlotPtr = MutPointer[GbdtMemberSlot, MutUntrackedOrigin]


def _slot_address(index: Int) raises -> Int:
    # Under the store's lock; a chunk's buffer never grows.
    var store = SharedSlot[GbdtMemberStore, _STORE_NAME, 2].get_or_create_ptr()
    return Int(store[].chunks[index // _CHUNK].unsafe_ptr()) + (index % _CHUNK) * size_of[GbdtMemberSlot]()


def _clear[I: GbdtMemberImage](slot: _SlotPtr):
    """The fitted state dropped; a resident model back to its image
    (non-raising)."""
    if slot[].resident >= 0:
        var handle = slot[].resident
        slot[].resident = -1
        try:
            I.release(handle)
        except:
            pass
    slot[].fitted = False
    slot[].text = String("")
    slot[].learn = List[Float64]()
    slot[].x_text = List[Int32]()
    slot[].x_scalars = List[Float64]()
    slot[].features = 0
    slot[].dim = 0
    slot[].classes = 0


def _name(kind: Int) -> String:
    if kind == _KIND_CLASSIFIER:
        return String("GradientBoostingClassifier")
    if kind == _KIND_REGRESSOR:
        return String("GradientBoostingRegressor")
    return String(_ESTIMATOR)


def _kind_word(word: Float64) raises -> Int:
    if word != 0.0 and word != 1.0 and word != 2.0:
        raise Error("member ABI: a Gbdt member's kind word is 0, 1 or 2")
    return Int(word)


def _mint(words: _F64P, count: Int) raises -> Int:
    if count < 2:
        raise Error("member ABI: a Gbdt member takes its kind word (0, 1, 2) and its config words")
    var kind = _kind_word(words[0])
    var copied = List[Float64](length=count, fill=0.0)
    unsafe_memcpy(dest=copied.unsafe_ptr(), src=words, count=count)
    var config = gbdt_config_of_words(copied^, 1)
    # The adapters fix their learner's loss (`_GBDTAdapter._loss`).
    if kind == _KIND_CLASSIFIER and config.loss != "Logloss":
        raise Error("member ABI: a GradientBoostingClassifier member fits loss='Logloss'")
    if kind == _KIND_REGRESSOR and config.loss != "RMSE":
        raise Error("member ABI: a GradientBoostingRegressor member fits loss='RMSE'")
    var store = SharedSlot[GbdtMemberStore, _STORE_NAME, 2].get_or_create_ptr()
    var lock = rebind[MutPointer[Int64, MutUntrackedOrigin]](store[].synchronization.unsafe_ptr())
    shared_slot_lock(lock)
    try:
        var index = -1
        if len(store[].free) > 0:
            index = store[].free.pop()
        else:
            if store[].count >= _INDEX_SPAN - 1:
                raise Error("RuntimeError: mojolearn: too many live Gbdt member states")
            if store[].count == len(store[].chunks) * _CHUNK:
                store[].chunks.append(List[GbdtMemberSlot](length=_CHUNK, fill=GbdtMemberSlot()))
            index = store[].count
            store[].count += 1
        var slot = _SlotPtr(unsafe_from_address=_slot_address(index))
        slot[].generation += 1
        slot[].live = True
        slot[].kind = kind
        slot[].config = config^
        slot[].fitted = False
        slot[].resident = -1
        return slot[].generation * _INDEX_SPAN + index + 1
    finally:
        shared_slot_unlock(lock)


def _lookup(handle: Int, release: Bool) raises -> Int:
    """The slot address of a live handle; a released, foreign or made-up
    handle is refused. A released slot's fitted state is dropped by the
    caller (`_clear`) before the release."""
    var store = SharedSlot[GbdtMemberStore, _STORE_NAME, 2].get_or_create_ptr()
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
            slot[].config = GbdtConfig()
            store[].free.append(index)
        return address
    finally:
        shared_slot_unlock(lock)


def _whole(call: _CallPtr) raises:
    if call[].row_index.rows > 0 or call[].column_index.rows > 0:
        raise Error("TypeError: a Gbdt member reads whole host views (no row or column index)")


def _dense(view: MemberMatrix, dtype: Int, what: String) raises:
    if view.home == MEMBER_HOME_DEVICE or not view.dense_host(dtype):
        var words = String("float32") if dtype == MEMBER_F32 else (
            String("float64") if dtype == MEMBER_F64 else String("int32"))
        raise Error("TypeError: a Gbdt member reads " + what + " as a dense host " + words + " view")


def _column_major(src: Int, rows: Int, cols: Int) -> List[Float32]:
    """A row-major float32 block as column-major words (the layout the
    image's column-major fit and per-call parse read; the fit door admits X
    in that order with the same one copy), split by columns over the host
    workers."""
    var out = List[Float32](length=max(rows * cols, 1), fill=0.0)
    var source = _F32P(unsafe_from_address=src)
    var target = _F32P(unsafe_from_address=Int(out.unsafe_ptr()))
    var tasks = host_predict_task_count_work(cols, rows * cols)
    var chunk = host_predict_chunk(cols, tasks)

    def task(unit: Int) {imm source, imm target, imm chunk, imm rows, imm cols}:
        for j in range(unit * chunk, min((unit + 1) * chunk, cols)):  # native data loop (one host task's columns)
            for i in range(rows):  # native data loop (the column's rows)
                target[j * rows + i] = source[i * cols + j]

    host_parallelize(task, tasks)
    return out^


def _has_nan(src: Int, count: Int) -> Bool:
    """Whether a float32 block holds a NaN (per-task flags)."""
    var tasks = host_predict_task_count_work(count, count)
    var chunk = host_predict_chunk(count, tasks)
    var flags = List[UInt8](length=max(tasks, 1), fill=0)
    var source = _F32P(unsafe_from_address=src)
    var marks = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(flags.unsafe_ptr()))

    def task(unit: Int) {imm source, imm marks, imm chunk, imm count}:
        for i in range(unit * chunk, min((unit + 1) * chunk, count)):  # native data loop (one host task's cells)
            if source[i] != source[i]:
                marks[unit] = 1
                return

    host_parallelize(task, tasks)
    for t in range(len(flags)):  # small-loop(tasks): one flag per host task
        if flags[t] != 0:
            return True
    return False


def _code_classes(src: Int, rows: Int, loss: String) raises -> Int:
    """`_codes` of the fit door on the member's float32 class codes: the
    number of distinct codes, refused in the door's words when a label is not
    a finite nonnegative integer, when fewer than two classes are present or
    when the codes are not 0..k-1 (two passes split over the host workers:
    the range, then the presence of each code)."""
    var tasks = host_predict_task_count_work(rows, rows)
    var chunk = host_predict_chunk(rows, tasks)
    var source = _F32P(unsafe_from_address=src)
    var lows = List[Float32](length=max(tasks, 1), fill=Float32(3.0e38))
    var highs = List[Float32](length=max(tasks, 1), fill=Float32(-1.0))
    var bads = List[UInt8](length=max(tasks, 1), fill=0)
    var lo_p = _F32P(unsafe_from_address=Int(lows.unsafe_ptr()))
    var hi_p = _F32P(unsafe_from_address=Int(highs.unsafe_ptr()))
    var bad_p = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(bads.unsafe_ptr()))

    def scan(unit: Int) {imm source, imm lo_p, imm hi_p, imm bad_p, imm chunk, imm rows}:
        for i in range(unit * chunk, min((unit + 1) * chunk, rows)):  # native data loop (one host task's rows)
            var v = source[i]
            # (a float32 at or above 2**24 is an integer; Int() of it is exact
            # below 1e15, and such a code is past the rows: not dense, below.)
            if not isfinite(v) or v < 0 or (v < Float32(1.0e15) and Float32(Int(v)) != v):
                bad_p[unit] = 1
                return
            lo_p[unit] = min(lo_p[unit], v)
            hi_p[unit] = max(hi_p[unit], v)

    host_parallelize(scan, tasks)
    var low = Float32(3.0e38)
    var high = Float32(-1.0)
    for t in range(len(bads)):  # small-loop(tasks): one range per host task
        if bads[t] != 0:
            raise Error("mojolearn: " + loss + " labels must be finite nonnegative integer class codes")
        low = min(low, lows[t])
        high = max(high, highs[t])
    if rows == 0 or low == high:
        raise Error("mojolearn: " + loss + " needs at least two classes")
    if high >= Float32(rows):
        raise Error("mojolearn: " + loss + " labels must be dense class codes 0..k-1; encode labels before fitting")
    var top = Int(high)
    if top >= rows:
        raise Error("mojolearn: " + loss + " labels must be dense class codes 0..k-1; encode labels before fitting")
    var width = top + 1
    var seen = List[UInt8](length=max(tasks, 1) * width, fill=0)
    var seen_p = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(seen.unsafe_ptr()))

    def mark(unit: Int) {imm source, imm seen_p, imm chunk, imm rows, imm width}:
        for i in range(unit * chunk, min((unit + 1) * chunk, rows)):  # native data loop (one host task's rows)
            seen_p[unit * width + Int(source[i])] = 1

    host_parallelize(mark, tasks)
    var k = 0
    for c in range(width):  # native data loop (the codes; tasks per code are a small-loop)
        var any = False
        for t in range(max(tasks, 1)):  # small-loop(tasks)
            if seen[t * width + c] != 0:
                any = True
                break
        if any:
            k += 1
    if k != width:
        raise Error("mojolearn: " + loss + " labels must be dense class codes 0..k-1; encode labels before fitting")
    return k


def _binary_codes(src: Int, rows: Int, classes: Int) raises -> List[Float32]:
    """GradientBoostingClassifier's targets: the caller's int32 codes of its
    two classes as float32 0 / 1 (the adapter's `_codes_f32`), refused in the
    adapter's words for one class or more than two; the cast split over the
    host workers."""
    if classes < 2:
        raise Error("Classifier can't train when only one class is present.")
    if classes > 2:
        raise Error("Only binary classification is supported. The type of the target is multiclass.")
    var out = List[Float32](length=max(rows, 1), fill=0.0)
    var tasks = host_predict_task_count_work(rows, rows)
    var chunk = host_predict_chunk(rows, tasks)
    var bads = List[UInt8](length=max(tasks, 1), fill=0)
    var source = _I32P(unsafe_from_address=src)
    var target = _F32P(unsafe_from_address=Int(out.unsafe_ptr()))
    var bad_p = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(bads.unsafe_ptr()))

    def task(unit: Int) {imm source, imm target, imm bad_p, imm chunk, imm rows}:
        for i in range(unit * chunk, min((unit + 1) * chunk, rows)):  # native data loop (one host task's rows)
            var c = source[i]
            if c < 0 or c > 1:
                bad_p[unit] = 1
                return
            target[i] = Float32(c)

    host_parallelize(task, tasks)
    for t in range(len(bads)):  # small-loop(tasks): one flag per host task
        if bads[t] != 0:
            raise Error("ValueError: a GradientBoostingClassifier member reads the class codes 0 and 1")
    return out^


def _multi_output(loss: String) -> Bool:
    return loss == "MultiClass" or loss == "MultiClassOneVsAll"


def _binary(loss: String) -> Bool:
    return loss == "Logloss" or loss == "CrossEntropy"


def _fit[I: GbdtMemberImage](slot: _SlotPtr, call: _CallPtr) raises:
    """`gbdt_fit` of the door on the member's views: its checks in its
    order and words (the ones the member's views can reach), the typed
    request from the config, `xg_fit` and the model dimension."""
    _whole(call)
    if call[].fit_output != 0:
        raise Error("TypeError: a Gbdt member has no fit output")
    _dense(call[].x, MEMBER_F32, "X")
    var rows = call[].x.rows
    var cols = call[].x.cols
    if rows == 0 or cols == 0:
        raise Error("mojolearn: fit requires at least one row and one feature")
    var kind = slot[].kind
    if kind == _KIND_CLASSIFIER:
        _dense(call[].y, MEMBER_I32, "y (class codes)")
    else:
        _dense(call[].y, MEMBER_F32, "y")
    if call[].y.cols != 1:
        raise Error("TypeError: a Gbdt member reads one target column")
    if call[].y.rows != rows:
        raise Error("mojolearn: y has " + String(call[].y.rows) + " values for " + String(rows) + " rows")
    ref config = slot[].config
    var multi = _multi_output(config.loss)
    # The classifier's codes become the learner's float32 targets (the
    # adapter's `_codes_f32`); finite by construction.
    var codes = List[Float32]()
    var y_addr = call[].y.data
    if kind == _KIND_CLASSIFIER:
        codes = _binary_codes(call[].y.data, rows, call[].classes)
        y_addr = Int(codes.unsafe_ptr())
    elif not trees_finite_f32(y_addr, rows):
        if kind == _KIND_REGRESSOR:
            raise Error("y must be finite")
        raise Error("Input y contains NaN or infinity" + (" (the class labels)" if multi else "")
                    + " (GradientBoosting).")
    if config.nan_mode == "Forbidden" and not trees_finite_f32(call[].x.data, rows * cols):
        if _has_nan(call[].x.data, rows * cols):
            raise Error("mojolearn: nan_mode='Forbidden' but X contains NaN. "
                        + "CatBoost refuses this pair; this implementation would otherwise "
                        + "bin the NaNs silently with no NaN bin. Use "
                        + "nan_mode='Min' or 'Max', or clean the column.")
    var n_classes = 0
    if multi:
        n_classes = _code_classes(y_addr, rows, config.loss)
        if len(config.class_weights) != 0 and len(config.class_weights) != n_classes:
            raise Error("mojolearn: class_weights needs " + String(n_classes) + " entries for " + config.loss)
    var n_weights = 0
    var w_addr = call[].x.data
    if call[].weights.rows > 0:
        _dense(call[].weights, MEMBER_F32, "sample weights")
        if call[].weights.rows * call[].weights.cols != rows:
            raise Error("ValueError: sample_weight has " + String(call[].weights.rows * call[].weights.cols)
                        + " values for " + String(rows) + " rows")
        w_addr = call[].weights.data
        n_weights = rows
    var entry = entry_ref_gil_free(call[].route.entry)
    var row_major = False
    try:
        row_major = xg_row_major(entry)
    except:
        row_major = False
    var scratch = List[Float32]()
    var x_addr = call[].x.data
    if not row_major:
        scratch = _column_major(call[].x.data, rows, cols)
        x_addr = Int(scratch.unsafe_ptr())
    var words = gbdt_fit_words(config, rows, cols, 0, n_weights, 0, False)
    var options = gbdt_fit_options(config, rows)
    var flags = List[UInt32](length=1, fill=0)
    var rate = words.rate
    # The word lists move into the request by swap: moving the fields would leave
    # `words` and `options` partly destroyed.
    var fit_ints = List[Int]()
    swap(fit_ints, words.ints)
    var fit_floats = List[Float64]()
    swap(fit_floats, words.floats)
    var fit_bad = List[UInt8]()
    swap(fit_bad, words.bad)
    var fit_strs = List[String]()
    swap(fit_strs, options.strs)
    var args = GbdtFitArgs(x_addr, y_addr, w_addr, Int(flags.unsafe_ptr()), x_addr, y_addr,
                           fit_ints^, fit_floats^, fit_bad^, fit_strs^)
    var outcome = xg_fit(entry, row_major, args)
    _ = len(scratch)  # the column-major copy lives until the fit returned
    _ = len(codes)  # the classifier's float32 targets too
    _ = len(flags)
    if gbdt_text_has_ctr(outcome.text):
        raise Error("RuntimeError: mojolearn: a Gbdt member fit returned a CTR model (no cat_features were given)")
    var dim = xg_model_dim(entry, outcome.text)
    _clear[I](slot)
    slot[].bias = gbdt_bias_of(outcome.text)
    swap(slot[].text, outcome.text)  # by swap: `outcome` is read after (moving a field out of it is refused)
    swap(slot[].learn, outcome.learn)
    slot[].best_iteration = outcome.best_iteration
    slot[].stopped_early = outcome.stopped_early
    slot[].features = cols
    slot[].dim = dim
    slot[].classes = dim + 1 if config.loss == "MultiClass" else (dim if config.loss == "MultiClassOneVsAll" else 0)
    slot[].rate = rate
    slot[].ordered = options.boosting == "Ordered"
    slot[].fitted = True


def _require_fitted(slot: _SlotPtr, before: String) raises:
    if not slot[].fitted:
        raise Error("RuntimeError: mojolearn: " + before + " before fit()")


def _x(slot: _SlotPtr, call: _CallPtr) raises:
    _whole(call)
    _dense(call[].x, MEMBER_F32, "X")
    if call[].x.cols != slot[].features:
        raise Error("mojolearn: model was fitted on " + String(slot[].features) + " features, got "
                    + String(call[].x.cols))


def _resident(slot: _SlotPtr, entry: EntryRef) raises -> Int:
    """The image's resident model of this state (prepared on the first
    predict, as `GradientBoosting._resident_handle`), or -1 on an image
    without one."""
    if slot[].resident >= 0:
        return slot[].resident
    if not xg_resident(entry):
        return -1
    slot[].resident = xg_resident_prepare(entry, Int(slot[].text.unsafe_ptr()), slot[].text.byte_length())
    return slot[].resident


def _apply(slot: _SlotPtr, call: _CallPtr, mode: Int, dst: Int) raises -> Int:
    """One predict in `mode` into `dst`: the resident model (row-major X as
    it is), else the per-call parse of a column-major copy. The width per
    row written."""
    var entry = entry_ref_gil_free(call[].route.entry)
    var rows = call[].x.rows
    var handle = _resident(slot, entry)
    if handle >= 0:
        return xg_resident_predict(entry, handle, call[].x.data, rows, dst, mode, True)
    var cm = _column_major(call[].x.data, rows, call[].x.cols)
    var text = Int(slot[].text.unsafe_ptr())
    var nbytes = slot[].text.byte_length()
    var width: Int
    if mode == _PREDICT_RAW and slot[].dim == 1:
        var wrote = xg_predict(entry, text, nbytes, Int(cm.unsafe_ptr()), rows, dst)
        if wrote != rows:
            raise Error("RuntimeError: mojolearn: predict wrote " + String(wrote) + " of " + String(rows) + " rows")
        width = 1
    else:
        width = xg_predict_multi(entry, text, nbytes, Int(cm.unsafe_ptr()), rows, dst, mode)
    _ = len(cm)
    return width


def _binary_out(slot: _SlotPtr, call: _CallPtr, probabilities: Bool, dst: Int) raises:
    """GradientBoostingClassifier's `_binary_output` into `dst` (int32 codes,
    rows, or float32 `[1 - p, p]`, rows x 2): the learner's raw margins (the
    route `_apply` takes), the adapter's finiteness refusal, then the image's
    XG_BINARY transform. X is checked by the caller."""
    var rows = call[].x.rows
    if rows == 0:
        return
    var raw = List[Float32](length=rows, fill=0.0)
    var width = _apply(slot, call, _PREDICT_RAW, Int(raw.unsafe_ptr()))
    if width != 1:
        raise Error("RuntimeError: mojolearn: a binary GBDT model wrote width " + String(width))
    if not trees_finite_f32(Int(raw.unsafe_ptr()), rows):
        raise Error("Classifier margins must be finite scalar Float32 values")
    var entry = entry_ref_gil_free(call[].route.entry)
    var wrote = xg_binary(entry, probabilities, Int(raw.unsafe_ptr()), dst, rows)
    _ = len(raw)
    if wrote != (2 * rows if probabilities else rows):
        raise Error("RuntimeError: GBDT binary output returned an unexpected row count")


def _classify(slot: _SlotPtr, call: _CallPtr, probabilities: Bool) raises:
    """GradientBoostingClassifier's `predict` (int32 codes, rows x 1) and
    `predict_proba` (float32 rows x 2)."""
    _require_fitted(slot, String("predict_proba()") if probabilities else String("predict()"))
    _x(slot, call)
    if probabilities:
        _dense(call[].dst, MEMBER_F32, "the predict_proba output")
        if call[].dst.rows != call[].x.rows or call[].dst.cols != 2:
            raise Error("TypeError: a GradientBoostingClassifier member predict_proba writes rows x 2 float32")
    else:
        _dense(call[].dst, MEMBER_I32, "the predict output (class codes)")
        if call[].dst.rows != call[].x.rows or call[].dst.cols != 1:
            raise Error("TypeError: a GradientBoostingClassifier member predict writes rows x 1 int32 codes")
    _binary_out(slot, call, probabilities, call[].dst.data)


def _decision(slot: _SlotPtr, call: _CallPtr) raises:
    """GradientBoostingClassifier's `decision_function`: the learner's raw
    float32 margin, rows x 1 (one column: a binary decision)."""
    if slot[].kind != _KIND_CLASSIFIER:
        member_unsupported(_name(slot[].kind), MEMBER_OP_DECISION)
    _require_fitted(slot, "decision_function()")
    _x(slot, call)
    _dense(call[].dst, MEMBER_F32, "the decision output")
    if call[].dst.rows != call[].x.rows or call[].dst.cols != 1:
        raise Error("TypeError: a GradientBoostingClassifier member decision_function writes rows x 1 float32")
    if call[].x.rows == 0:
        return
    var width = _apply(slot, call, _PREDICT_RAW, call[].dst.data)
    if width != 1:
        raise Error("RuntimeError: mojolearn: a binary GBDT model wrote width " + String(width))


def _score(slot: _SlotPtr, call: _CallPtr) raises:
    """GradientBoostingClassifier's `score` without weights: the accuracy of
    the predicted codes against the caller's int32 codes, float32 count /
    rows widened (the adapter's `accuracy_score`; integer counts merged
    exactly over the host workers). A weighted score is the caller's metric
    (no MEMBER_CAN_WEIGHTED_SCORE)."""
    var name = _name(slot[].kind)
    if slot[].kind != _KIND_CLASSIFIER:
        member_unsupported(name, MEMBER_OP_SCORE)
    if call[].weights.rows > 0:
        raise Error("NotImplementedError: mojolearn: a weighted score of a " + name
                    + " member is not published; the caller predicts and runs its own weighted metric")
    _require_fitted(slot, "score()")
    _x(slot, call)
    _dense(call[].y, MEMBER_I32, "y (class codes)")
    var rows = call[].x.rows
    if call[].y.cols != 1 or call[].y.rows != rows:
        raise Error("mojolearn metrics: y_true has " + String(call[].y.rows * call[].y.cols) + " entries and "
                    + "y_pred has " + String(rows) + "; they must be the same length")
    if rows <= 0:
        raise Error("accuracy_score: n must be positive, got " + String(rows) + " (0 / 0 is refused by name)")
    var codes = List[Int32](length=rows, fill=0)
    _binary_out(slot, call, False, Int(codes.unsafe_ptr()))
    var tasks = host_predict_task_count_work(rows, rows)
    var chunk = host_predict_chunk(rows, tasks)
    var partials = List[Int](length=max(tasks, 1), fill=0)
    var truth = _I32P(unsafe_from_address=call[].y.data)
    var guess = _I32P(unsafe_from_address=Int(codes.unsafe_ptr()))
    var part_p = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(partials.unsafe_ptr()))

    def count(unit: Int) {imm truth, imm guess, imm part_p, imm chunk, imm rows}:
        var hits = 0
        for i in range(unit * chunk, min((unit + 1) * chunk, rows)):  # native data loop (one host task's rows)
            if truth[i] == guess[i]:
                hits += 1
        part_p[unit] = hits

    host_parallelize(count, tasks)
    var hits = 0
    for t in range(len(partials)):  # small-loop(tasks): one count per host task
        hits += partials[t]
    _ = len(codes)
    call[].value = Float64(Float32(hits) / Float32(rows))


def _predict(slot: _SlotPtr, call: _CallPtr) raises:
    """`GradientBoosting.predict`: raw float32 scores, rows x approx_dim
    (the regressor's: rows x 1); the classifier's: `_classify`."""
    if slot[].kind == _KIND_CLASSIFIER:
        _classify(slot, call, False)
        return
    _require_fitted(slot, "predict()")
    _x(slot, call)
    _dense(call[].dst, MEMBER_F32, "the predict output")
    if call[].dst.rows != call[].x.rows or call[].dst.cols != slot[].dim:
        raise Error("TypeError: a Gbdt member predict writes rows x " + String(slot[].dim) + " float32")
    var width = _apply(slot, call, _PREDICT_RAW, call[].dst.data)
    if width != slot[].dim:
        raise Error("RuntimeError: mojolearn: predict wrote width " + String(width) + ", expected " + String(slot[].dim))


def _widen(src: Int, dst: Int, count: Int):
    """float32 raw scores to float64, split over the host workers."""
    var tasks = host_predict_task_count_work(count, count)
    var chunk = host_predict_chunk(count, tasks)
    var source = _F32P(unsafe_from_address=src)
    var target = _F64P(unsafe_from_address=dst)

    def task(unit: Int) {imm source, imm target, imm chunk, imm count}:
        for i in range(unit * chunk, min((unit + 1) * chunk, count)):  # native data loop (one host task's rows)
            target[i] = Float64(source[i])

    host_parallelize(task, tasks)


def _proba(slot: _SlotPtr, call: _CallPtr) raises:
    """`GradientBoosting.predict_proba`: the same routes and dtypes (the
    classifier's: `_classify`; the regressor has none)."""
    if slot[].kind == _KIND_CLASSIFIER:
        _classify(slot, call, True)
        return
    if slot[].kind == _KIND_REGRESSOR:
        member_unsupported(_name(slot[].kind), MEMBER_OP_PREDICT_PROBA)
    ref loss = slot[].config.loss
    var rows = call[].x.rows
    if _multi_output(loss):
        _require_fitted(slot, "predict_proba()")
        _x(slot, call)
        _dense(call[].dst, MEMBER_F32, "the predict_proba output")
        if call[].dst.rows != rows or call[].dst.cols != slot[].classes:
            raise Error("TypeError: a Gbdt member predict_proba writes rows x " + String(slot[].classes) + " float32")
        var mode = _PREDICT_SIGMOID if loss == "MultiClassOneVsAll" else _PREDICT_SOFTMAX
        var width = _apply(slot, call, mode, call[].dst.data)
        if width != slot[].classes:
            raise Error("RuntimeError: mojolearn: predict_proba wrote width " + String(width) + ", expected "
                        + String(slot[].classes))
        return
    if not _binary(loss):
        raise Error("mojolearn: predict_proba is defined for Logloss, "
                    + "CrossEntropy, MultiClass and MultiClassOneVsAll; this model was fitted with '"
                    + loss + "'. Use predict() and apply the link yourself.")
    _require_fitted(slot, "predict_proba()")
    _x(slot, call)
    _dense(call[].dst, MEMBER_F64, "the predict_proba output")
    if call[].dst.rows != rows or call[].dst.cols != 2:
        raise Error("TypeError: a Gbdt member predict_proba writes rows x 2 float64")
    var entry = entry_ref_gil_free(call[].route.entry)
    if _resident(slot, entry) >= 0:
        var width = _apply(slot, call, _PREDICT_SIGMOID_PAIR, call[].dst.data)
        if width != 2:
            raise Error("RuntimeError: mojolearn: predict_proba wrote width " + String(width) + ", expected 2")
        return
    var raw32 = List[Float32](length=max(rows, 1), fill=0.0)
    _ = _apply(slot, call, _PREDICT_RAW, Int(raw32.unsafe_ptr()))
    var raw = List[Float64](length=max(rows, 1), fill=0.0)
    _widen(Int(raw32.unsafe_ptr()), Int(raw.unsafe_ptr()), rows)
    var wrote = xg_sigmoid_pair(entry, Int(raw.unsafe_ptr()), call[].dst.data, rows)
    _ = len(raw)
    _ = len(raw32)
    if wrote != rows:
        raise Error("RuntimeError: mojolearn: gbdt_sigmoid_pair wrote " + String(wrote) + " of " + String(rows) + " rows")


def _block_of(index: Int) raises -> Int:
    if index < 0 or index >= _BLOCKS:
        raise Error("member ABI: a Gbdt member has three fitted blocks")
    return MEMBER_BLOCK_SCALARS if index == 2 else MEMBER_BLOCK_FAMILY + index


def _export(slot: _SlotPtr, call: _CallPtr) raises:
    """A view of one fitted block (the slot owns the words; valid until the
    state's next fit, import or free)."""
    _require_fitted(slot, "export")
    var block = call[].block
    if (call[].flags & MEMBER_BLOCK_BY_INDEX) != 0:
        block = _block_of(block)
    if block == MEMBER_BLOCK_FAMILY:
        var nbytes = slot[].text.byte_length()
        if len(slot[].x_text) * 4 < nbytes or len(slot[].x_text) == 0:
            slot[].x_text = List[Int32](length=max((nbytes + 3) // 4, 1), fill=0)
            var into = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(slot[].x_text.unsafe_ptr()))
            var from_text = MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(slot[].text.unsafe_ptr()))
            unsafe_memcpy(dest=into, src=from_text, count=nbytes)
        call[].dst = MemberMatrix(Int(slot[].x_text.unsafe_ptr()), 1, len(slot[].x_text), MEMBER_I32)
    elif block == MEMBER_BLOCK_FAMILY + 1:
        call[].dst = MemberMatrix(Int(slot[].learn.unsafe_ptr()), 1, len(slot[].learn), MEMBER_F64)
    elif block == MEMBER_BLOCK_SCALARS:
        slot[].x_scalars = [Float64(slot[].features), Float64(slot[].dim), Float64(slot[].classes), slot[].bias,
                            Float64(slot[].best_iteration), 1.0 if slot[].stopped_early else 0.0, slot[].rate,
                            Float64(slot[].text.byte_length()), 1.0 if slot[].ordered else 0.0]
        call[].dst = MemberMatrix(Int(slot[].x_scalars.unsafe_ptr()), 1, _SCALARS, MEMBER_F64)
    else:
        raise Error("member ABI: a Gbdt member has no fitted block " + String(block))
    call[].block = block
    call[].out_cols = _BLOCKS


def _import[I: GbdtMemberImage](slot: _SlotPtr, call: _CallPtr) raises:
    """Copy one fitted block in; MEMBER_IMPORT_DONE builds the model text
    and marks the state fitted."""
    var block = call[].block
    if slot[].fitted:
        _clear[I](slot)
    var view = call[].x
    if block == MEMBER_BLOCK_FAMILY:
        _dense(view, MEMBER_I32, "a fitted block")
        slot[].x_text = List[Int32](length=max(view.rows * view.cols, 1), fill=0)
        unsafe_memcpy(dest=slot[].x_text.unsafe_ptr(), src=_I32P(unsafe_from_address=view.data),
                      count=view.rows * view.cols)
    elif block == MEMBER_BLOCK_FAMILY + 1:
        _dense(view, MEMBER_F64, "a fitted block")
        slot[].learn = List[Float64](length=view.rows * view.cols, fill=0.0)
        unsafe_memcpy(dest=slot[].learn.unsafe_ptr(), src=_F64P(unsafe_from_address=view.data),
                      count=view.rows * view.cols)
    elif block == MEMBER_BLOCK_SCALARS:
        _dense(view, MEMBER_F64, "a fitted block")
        if view.rows * view.cols != _SCALARS:
            raise Error("member ABI: the Gbdt scalars block is [features, approx_dim, n_classes, bias, "
                        + "best_iteration, stopped_early, learning_rate, text bytes, boosting]")
        var s = _F64P(unsafe_from_address=view.data)
        slot[].x_scalars = List[Float64](length=_SCALARS, fill=0.0)
        for k in range(_SCALARS):  # small-loop(scalars)
            slot[].x_scalars[k] = s[k]
    else:
        raise Error("member ABI: a Gbdt member has no fitted block " + String(block))
    if (call[].flags & MEMBER_IMPORT_DONE) == 0:
        return
    if len(slot[].x_scalars) != _SCALARS:
        raise Error("member ABI: a fitted Gbdt state needs its three blocks")
    var nbytes = Int(slot[].x_scalars[7])
    if nbytes < 1 or nbytes > len(slot[].x_text) * 4:
        raise Error("member ABI: the Gbdt model text block is shorter than its byte count")
    slot[].text = gbdt_string_at(Int(slot[].x_text.unsafe_ptr()), nbytes)
    slot[].features = Int(slot[].x_scalars[0])
    slot[].dim = Int(slot[].x_scalars[1])
    slot[].classes = Int(slot[].x_scalars[2])
    slot[].bias = slot[].x_scalars[3]
    slot[].best_iteration = Int(slot[].x_scalars[4])
    slot[].stopped_early = slot[].x_scalars[5] != 0.0
    slot[].rate = slot[].x_scalars[6]
    slot[].ordered = slot[].x_scalars[8] != 0.0
    if slot[].features < 1 or slot[].dim < 1:
        raise Error("member ABI: a fitted Gbdt state needs its features and model dimension")
    if slot[].kind != _KIND_BOOSTER and slot[].dim != 1:
        raise Error("member ABI: a " + _name(slot[].kind) + " state has one model dimension")
    if gbdt_text_has_ctr(slot[].text):
        raise Error("member ABI: a Gbdt member does not hold a CTR model (cat_features fits keep their doors)")
    slot[].fitted = True


def _capabilities(kind: Int, loss: String) -> Int:
    """No `MEMBER_CAN_TEST_ROWS`: the fit tests y here and X only for
    nan_mode='Forbidden' (NaN; infinities reach the kernel), and predict
    does not test X (the classifier tests its margins, not X). No
    MEMBER_BORROWS_FIT: the state owns its text (the classifier's float32
    targets are freed when its fit returns). No MEMBER_CAN_NOTICES: the
    fits raise no warnings (the adapters' doors raise none either)."""
    # MEMBER_GIL_FREE (THE GIL RULE): every operation of both kinds is traced
    # (fx-gil-RESULT.md, "gbdt"): the route entry is marked `entry_ref_gil_free`
    # at every operation, the image's operations never touch the GIL, and the
    # store (SharedSlot) and the resident registry have their own lock words.
    # The adapter kinds (0, 2) reach nothing the booster's trace did not:
    # the same entry operations (XG_FIT, XG_CAPS, XG_MODEL_DIM, the predicts
    # and the resident prepare / predict) plus XG_BINARY, an operation of the
    # same traced entry (`gbdt_entry_dispatch` runs it bare under the bit;
    # `gbdt_binary_native` touches no CPython; its device context is
    # `process_ctx`, own lock, as XG_SIGMOID_PAIR's), the same store and the
    # same resident registry; their own work is host loops over plain words.
    var mask = (MEMBER_CAN_FIT | MEMBER_CAN_PREDICT | MEMBER_CAN_EXPORT | MEMBER_CAN_IMPORT
                | MEMBER_CAN_SAMPLE_WEIGHT | MEMBER_GIL_FREE)
    if kind == _KIND_CLASSIFIER:
        return mask | MEMBER_CAN_PREDICT_PROBA | MEMBER_CAN_DECISION | MEMBER_CAN_SCORE | MEMBER_CAN_CLASSIFIER
    if kind == _KIND_REGRESSOR:
        return mask
    if _multi_output(loss) or _binary(loss):
        mask |= MEMBER_CAN_PREDICT_PROBA
    return mask


def _loss_of_words(call: _CallPtr) raises -> String:
    if not call[].params or call[].param_count < 2:
        raise Error("member ABI: a Gbdt member answers without a state from its parameter words")
    var words = _F64P(unsafe_from_address=call[].params)
    var n = Int(words[1])
    if n < 0 or 2 + n > call[].param_count:
        raise Error("member ABI: a Gbdt member's parameter words end early")
    var out = String(capacity_bytes=n)
    for k in range(n):  # small-loop(loss name bytes)
        out += chr(Int(words[2 + k]))
    return out^


def _member[I: GbdtMemberImage](operation: Int, address: Int) raises -> Int:
    var call = _CallPtr(unsafe_from_address=address)
    if operation == MEMBER_OP_CAPABILITIES:
        if call[].state == 0:
            if call[].params and call[].param_count >= 2:
                return _capabilities(_kind_word(_F64P(unsafe_from_address=call[].params)[0]), _loss_of_words(call))
            # The family's union: the booster's widest kind and the classifier.
            return _capabilities(_KIND_BOOSTER, String("MultiClass")) | _capabilities(_KIND_CLASSIFIER, String("Logloss"))
        var asked = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
        return _capabilities(asked[].kind, asked[].config.loss)
    if operation == MEMBER_OP_NEW:
        if not call[].params:
            raise Error("member ABI: a Gbdt member needs its parameter words")
        call[].state = _mint(_F64P(unsafe_from_address=call[].params), call[].param_count)
        return 0
    if operation == MEMBER_OP_FREE:
        var held = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
        _clear[I](held)
        _ = _lookup(call[].state, True)
        call[].state = 0
        return 0
    if operation == MEMBER_OP_OUTPUTS:
        # With a fitted state the state answers; without one the loss word:
        # a one-dimensional loss predicts one column, a binary proba two; the
        # width of a multi-class model is chosen by the fit (refused).
        var loss: String
        var kind: Int
        var dim = 0
        var classes = 0
        if call[].state != 0:
            var known = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
            loss = known[].config.loss
            kind = known[].kind
            if known[].fitted:
                dim = known[].dim
                classes = known[].classes
        else:
            loss = _loss_of_words(call)
            kind = _kind_word(_F64P(unsafe_from_address=call[].params)[0])
        if kind == _KIND_CLASSIFIER:
            # Two classes, one model dimension, before and after a fit.
            if call[].flags == MEMBER_OP_PREDICT:
                call[].out_cols = 1
                call[].out_dtype = MEMBER_I32
            elif call[].flags == MEMBER_OP_PREDICT_PROBA:
                call[].out_cols = 2
                call[].out_dtype = MEMBER_F32
            elif call[].flags == MEMBER_OP_DECISION:
                call[].out_cols = 1
                call[].out_dtype = MEMBER_F32
            else:
                member_unsupported(_name(kind), call[].flags)
            call[].classes = 2
            return 0
        if kind == _KIND_REGRESSOR:
            if call[].flags != MEMBER_OP_PREDICT:
                member_unsupported(_name(kind), call[].flags)
            call[].out_cols = 1
            call[].out_dtype = MEMBER_F32
            call[].classes = 0
            return 0
        if dim == 0:
            if _multi_output(loss):
                raise Error("member ABI: the output width of a " + loss + " Gbdt member is chosen by its fit")
            dim = 1
        if call[].flags == MEMBER_OP_PREDICT_PROBA:
            if _binary(loss):
                call[].out_cols = 2
                call[].out_dtype = MEMBER_F64
            elif _multi_output(loss):
                call[].out_cols = classes
                call[].out_dtype = MEMBER_F32
            else:
                member_unsupported(_ESTIMATOR, MEMBER_OP_PREDICT_PROBA)
        elif call[].flags == MEMBER_OP_PREDICT:
            call[].out_cols = dim
            call[].out_dtype = MEMBER_F32
        else:
            member_unsupported(_ESTIMATOR, call[].flags)
        call[].classes = classes
        return 0
    var slot = _SlotPtr(unsafe_from_address=_lookup(call[].state, False))
    if operation == MEMBER_OP_FIT:
        _fit[I](slot, call)
    elif operation == MEMBER_OP_PREDICT:
        _predict(slot, call)
    elif operation == MEMBER_OP_PREDICT_PROBA:
        _proba(slot, call)
    elif operation == MEMBER_OP_EXPORT:
        _export(slot, call)
    elif operation == MEMBER_OP_IMPORT:
        _import[I](slot, call)
    elif operation == MEMBER_OP_DECISION:
        _decision(slot, call)
    elif operation == MEMBER_OP_SCORE:
        _score(slot, call)
    else:
        member_unsupported(_name(slot[].kind), operation)
    return 0


def gbdt_member_v2[I: GbdtMemberImage](operation: Int, aa: Int, na: Int, pp: Int, np: Int, ff: Int, nf: Int,
    result: Int, context: Int, sink: NativeDescriptorError) abi("C") -> Int:
    """The member entry of GradientBoosting and its two adapters (schema
    `MEMBER_ABI`)."""
    try:
        var word = _member[I](operation, member_call_from_spans(operation, aa, na, np, nf))
        if result:
            MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=result)[] = word
        return 0
    except error:
        native_descriptor_report(error, context, sink)
        return 1


def gbdt_member_publish_image[I: GbdtMemberImage](vendor: StaticString) raises:
    """The one publication line of a gbdt image's module initialization,
    after its gbdt entry: the image's tier and the vendor word of the
    string its entry is published under (-1 publishes nothing)."""
    var word = gbdt_vendor_word(vendor)
    if word < 0:
        return
    var store = SharedSlot[GbdtMemberStore, _STORE_NAME, 2].get_or_create_ptr()
    member_publish[MEMBER_FAMILY_GBDT](Int(GLOBAL_NUMERIC_MODE), word, Int(store), gbdt_member_v2[I])
