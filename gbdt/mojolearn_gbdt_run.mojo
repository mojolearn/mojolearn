# SPDX-License-Identifier: Apache-2.0
"""RUN of the GBDT GPU binding (bindings/_mojolearn_gbdt.mojo): the fit's
slot checks and device fit, the resident registry lock, the borrowed-buffer
pointer helpers and the typed bodies of the two-level FeatureFreq and
ordered RMSE fits. Typed Mojo values and addresses only, no Python import
(tools/python_free_dirs.sh). The door UNPACKs the Python arguments,
releases the GIL around each RUN call and PACKs the result (lane
rs-boosting-trees, 2026-11-12: moved word for word; the fit's GIL release
is hoisted to the door around the whole RUN call, so the GIL is released
as before or longer). Same operands in the same order, so the bits are
unchanged."""
from std.memory import bitcast
from core.device_plan import DevicePlan, device_plan_current
from core.hostptr import list_f32, list_u32
from core.pair_split_device import device_split_pairs
from core.neural_context import process_ctx
from core.shared_slot import SharedSlot
from std.atomic import Atomic, Ordering
from checks.numerics import GLOBAL_NUMERIC_MODE as _DEVCTX_MODE, NUMERIC_IDENTICAL as _DEVCTX_IDENTICAL
from gbdt.fit_request import GbdtFitArgs, GbdtFitOutcome
from gbdt.estimator import (
    GbdtFitParams,
    GbdtFitResult,
    gbdt_fit,
    gbdt_fit_two_level_feature_freq,
)
from gbdt.resident_model import (
    GBDT_MODEL_REGISTRY,
    ResidentGbdtModel,
    gbdt_resident_predict,
)


#: This binding's ONE process-lifetime DeviceContext (core/neural_context.mojo,
#: lane/devctx-lifetime): a context per call exhausts Metal command queues.
comptime _DEVCTX_SLOT = "MojolearnGbdtContextIdenticalV3" if _DEVCTX_MODE == _DEVCTX_IDENTICAL else "MojolearnGbdtContextFastV3"


# Hazard B (p3-trees): GBDT_MODEL_REGISTRY is one process-wide Dict, and
# gbdt_resident_predict holds an element of it while the GIL is released. A
# prepare on another thread (Dict insert may rehash and move every entry) or
# a release (pop) would move or destroy the entry under the running predict.
# Every registry step of these bindings now takes this lock word (the
# cluster/resident_centers.mojo pattern): prepare builds and uploads the
# entry outside the lock and only inserts under it; predict holds the lock
# for its whole call, so two threads never run predict_into (which mutates
# the entry's workspace) on the same registry at once. The spin runs with the
# GIL released, so a waiting thread never blocks the holder or other Python
# threads.
struct _GbdtRegistryLock(Defaultable, Movable):
    var word: List[Int64]

    def __init__(out self):
        self.word = List[Int64](length=1, fill=Int64(0))


comptime _GBDT_REGISTRY_LOCK = SharedSlot[_GbdtRegistryLock, ( "MojolearnGbdtResidentLockIdenticalV1" if _DEVCTX_MODE == _DEVCTX_IDENTICAL else "MojolearnGbdtResidentLockFastV1" )]


def _gbdt_lock_word() raises -> MutPointer[Int64, MutUntrackedOrigin]:
    var state = _GBDT_REGISTRY_LOCK.get_or_create_ptr()
    return rebind[MutPointer[Int64, MutUntrackedOrigin]](state[].word.unsafe_ptr())


def _gbdt_lock(word: MutPointer[Int64, MutUntrackedOrigin]):
    while True:  # driver-loop: spin on the GBDT registry lock word (GIL released)
        var expected = Int64(0)
        if Atomic.compare_exchange[
            success_ordering=Ordering.SEQUENTIAL,
            failure_ordering=Ordering.RELAXED,
        ](word, expected, Int64(1)):
            return


def _gbdt_unlock(word: MutPointer[Int64, MutUntrackedOrigin]):
    Atomic.store[ordering=Ordering.RELEASE](word, Int64(0))


def _gbdt_locked_insert_run(word: MutPointer[Int64, MutUntrackedOrigin], var entry: ResidentGbdtModel) raises -> Int:
    """`gbdt_resident_prepare`'s registry step once the door holds the lock
    word (`_gbdt_locked_insert` takes it with the GIL released)."""
    try:
        var state = GBDT_MODEL_REGISTRY.get_or_create_ptr()
        if state[].next_id == 9223372036854775807:
            raise Error("resident GBDT model handle space exhausted")
        var handle = state[].next_id
        state[].next_id += 1
        state[].entries[handle] = entry^
        _gbdt_unlock(word)
        return handle
    except error:
        _gbdt_unlock(word)
        raise error^


def _gbdt_locked_predict(
    handle: Int,
    x: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    out_f32: MutPointer[Float32, MutUntrackedOrigin],
    out_f64: MutPointer[Float64, MutUntrackedOrigin],
    out_i64: MutPointer[Int64, MutUntrackedOrigin],
    mode: Int,
    row_major: Bool,
) raises -> Int:
    """`gbdt_resident_predict` under the lock word. Called with the GIL
    released."""
    var word = _gbdt_lock_word()
    _gbdt_lock(word)
    try:
        var width = gbdt_resident_predict(handle, x, n_rows, out_f32, out_f64, out_i64, mode, row_major)
        _gbdt_unlock(word)
        return width
    except error:
        _gbdt_unlock(word)
        raise error^


def _f64_ptr(addr: Int) raises -> MutPointer[Float64, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null buffer address")
    return MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=addr)


def _i64_ptr(addr: Int) raises -> MutPointer[Int64, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null buffer address")
    return MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=addr)


def _f32_ptr(addr: Int) raises -> MutPointer[Float32, MutUntrackedOrigin]:
    """A caller's float32 buffer, borrowed, never owned.

    The origin is untracked because the owner is a NumPy array on the other
    side of the boundary and Mojo cannot see it. The wrapper holds that array
    through the length of the call, which is the whole contract.
    """
    if addr == 0:
        raise Error("mojolearn: null buffer address")
    return MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)


def _u32_ptr(addr: Int) raises -> MutPointer[UInt32, MutUntrackedOrigin]:
    if addr == 0:
        raise Error("mojolearn: null buffer address")
    return MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=addr)


def _gbdt_device_plan() raises -> DevicePlan:
    """THE GBDT FAMILY'S DEVICE PLAN, read ONCE here at the binding entry
    (docs/plans/MULTI_GPU_ENCAPSULATION_20261103.md). Nothing below the
    entry reads the thread plan; the value travels down to the dispatchers in
    `gbdt/methods/pointwise_multi_gpu.mojo`, which parse `plan.setting` as
    the old in-kernel reads did and open `plan.device(rank)` per shard."""
    return device_plan_current("MOJOLEARN_GBDT_DEVICE_COUNT")


def gbdt_fit_native_run(row_major: Bool, params: GbdtFitArgs) raises -> GbdtFitOutcome:
    """Fit a gradient-boosted ensemble. The door `gbdt_fit_native` calls
    this with the GIL released.

    Returns `[model_text, best_iteration, stopped_early, learn_losses,
    test_losses]`. It used to return the text alone; the four that follow
    it are what a caller needs to tell a model that STOPPED from one that
    ran out of iterations, and to plot the curve that made it stop.

    `params` is, in this exact order -- and the wrapper names the same
    order in the same words, because a silent reordering here is a wrong
    answer rather than a failure:

         0  n_rows
         1  n_features
         2  n_weights        (0 means unit weights; weights_addr unread)
         3  n_flags          (0 means all-numeric; cat_flags_addr unread)
         4  border_count
         5  n_estimators
         6  max_depth
         7  learning_rate    (float)
         8  l2_leaf_reg      (float)
         9  random_seed
        10  score_function
        11  loss_alpha       (float, -1 = unset)
        12  loss_q           (float, -1 = unset)
        13  loss_delta       (float, -1 = unset)
        14  loss_variance_power (float, -1 = unset)
        15  loss_border      (float, -1 = unset)
        16  leaf_estimation_iterations  (-1 = unset, the loss decides)
        17  leaf_estimation_method      (-1 = unset, the loss decides)
        18  bagging_temperature (float)
        19  subsample        (float, -1 = unset)
        20  n_eval_rows      (0 means no held-out set; the eval addresses
                              are unread)
        21  od_pvalue        (float, -1 = unset)
        22  od_wait          (-1 = unset)
        23  use_best_model   (-1 = unset, 0 off, 1 on)
        24  best_model_min_trees
        25  random_strength  (float)
        26  use_pointwise_searcher  (0 or 1)
        27  border_build_max_samples  (0 = every row)
        28  permutation_count            (-1 = unset)
        29  ctr_estimation_permutation_id (-1 = unset)
        30  boost_from_average (-1 = unset, 0 off, 1 on)
        31  grow_policy      (0 SymmetricTree, 1 Depthwise, 2 Lossguide --
                              their EGrowPolicy order)
        32  max_leaves       (-1 = unset: 1 << depth, or 31 under Lossguide)
        33  min_data_in_leaf
        34  n_class_weights  (0 means none)

    Optional Float64 tails after counted weights are min_split_gain,
    min_child_hessian, then feature_fraction. Missing guards default to -1
    and missing feature_fraction defaults to 1, preserving existing layouts.
    A fit with `group_id` sends all three and then two more: the address of
    a uint32 buffer of group sizes (the pool's query runs in row order) and
    the group count, 35 + n_class_weights + 5 values in all. A PairLogit fit
    with `group_id` sends three more after those: the address of a uint32
    buffer of pairs (winner row then loser row per pair), the pair count, -1
    to generate the pairs from the groups and `y` (both addresses are then
    unread), and the address of a float32 buffer of one weight per pair,
    35 + n_class_weights + 8 values in all. A MultiRMSE fit sends the three
    float tails and then its TARGET DIMENSION, 35 + n_class_weights + 4
    values in all (lane/algos-trees, 2026-09-27); `y_addr` then holds that
    many DIM-MAJOR planes of `n_rows` (`y[dim * n_rows + row]`).

    AND THEN `n_class_weights` MORE VALUES, the class weights themselves,
    at `params[35 .. 35 + n_class_weights)`. They ride in this list rather
    than at a seventh buffer address for two reasons. The arity: this
    function already takes eight arguments and
    `PythonModuleBuilder.def_function` stops inferring a signature at
    around nine. And the ROUND TRIP: a Python float reaches `Float64(py=)`
    exactly, where a float written into a string and parsed back does not
    -- `String(Float32)` on this toolchain returns a one-ULP-wrong value
    on 0.46% of float32 values, and a class weight is a user's number,
    not ours to round.

    `strs` is `[loss, bootstrap_type, od_type, nan_mode]`, optionally
    followed by `feature_border_type` (lane/catboost-parity), their
    `ELossFunction`, `EBootstrapType`, `EOverfittingDetectorType` and
    `ENanMode` spellings. An empty `bootstrap_type` means none, and an
    empty `od_type` means UNSET -- which is not the same as `None`: their
    `Load` (`overfitting_detector_options.cpp:24-32`) picks the type from
    whichever of the other two was given, so an unset type with a wait is
    `Iter`. `nan_mode` is "Min", "Max" or "Forbidden"; an empty string is
    read as Min by `nan_mode_from_name`, which is their default
    (`data_processing_options.cpp:26`).

    EVERY `-1` ABOVE IS THEIR `TOption::NotSet()`, not a magic number: the
    loss picks the leaf estimator and its iteration count through
    `set_leaves_estimation_default`, which is the implementation of
    `catboost_options.cpp:273-360`. A caller that passes explicit values
    is overriding CatBoost's own defaults and should know it.
    """
    # 35 FIXED SLOTS PLUS ONE PER CLASS WEIGHT (slot 30 grew
    # boost_from_average 2026-08-22 and the count moved to 31; slots 31-33
    # grew grow_policy / max_leaves / min_data_in_leaf 2026-08-23,
    # DEVIATION 259, and the count moved to 34 -- the count is ALWAYS the
    # last fixed slot, so a new option goes before it and bumps these
    # three numbers and the wrapper's `_params`). The count is checked
    # against its slot rather than assumed, because a wrapper that
    # appended the wrong number of weights would otherwise read whatever
    # followed them -- and for a weight that is a wrong answer, not a
    # failure.
    if len(params) < 35:
        raise Error(
            "gbdt_fit: params must hold at least 35 values, got "
            + String(len(params))
        )
    var n_class_weights = params.i(34)
    if n_class_weights < 0:
        raise Error(
            "gbdt_fit: n_class_weights must not be negative, got "
            + String(n_class_weights)
        )
    var fixed_and_weights = 35 + n_class_weights
    if len(params) != fixed_and_weights and len(params) != fixed_and_weights + 1 and len(params) != fixed_and_weights + 2 and len(params) != fixed_and_weights + 3 and len(params) != fixed_and_weights + 4 and len(params) != fixed_and_weights + 5 and len(params) != fixed_and_weights + 8:
        raise Error(
            "gbdt_fit: params must hold 35 + n_class_weights, optionally min_split_gain, min_child_hessian, then feature_fraction, then the group sizes address and group count, then the pairs address, pair count and pair weights address values ("
            + String(35 + n_class_weights)
            + ") values, got "
            + String(len(params))
        )
    if len(params.strs) != 4 and len(params.strs) != 5 and len(params.strs) != 8:
        raise Error(
            "gbdt_fit: strs must hold [loss, bootstrap_type, od_type,"
            " nan_mode], optionally feature_border_type, optionally then"
            " boosting_type, the fold_len_multiplier's float64 bits and"
            " fold_permutation_block, got " + String(len(params.strs))
        )
    # the optional fifth string, their `feature_border_type` (empty or
    # absent is GreedyLogSum); an older wrapper sends four
    var border_type_name = String("GreedyLogSum")
    if len(params.strs) >= 5:
        border_type_name = params.strs[4]
        if border_type_name == "":
            border_type_name = String("GreedyLogSum")
    # the Ordered tail (lane/catboost-parity): `boosting_type`, then
    # `fold_len_multiplier` as the DECIMAL OF ITS FLOAT64 BITS (a parsed
    # decimal float is not exact, the class-weight note above), then
    # `fold_permutation_block`
    var boosting_type_name = String("Plain")
    var fold_len_multiplier = Float64(2.0)
    var fold_permutation_block = 0
    if len(params.strs) == 8:
        boosting_type_name = params.strs[5]
        fold_len_multiplier = bitcast[DType.float64](
            UInt64(Int(params.strs[6]))
        )
        fold_permutation_block = Int(params.strs[7])
    var xp = _f32_ptr(params.x)
    var yp = _f32_ptr(params.y)
    # unread when n_weights / n_flags is 0; the wrapper passes the X
    # address rather than allocating a throwaway, as `kmeans_fit` does,
    # and `_f32_ptr` still refuses a null.
    var wp = _f32_ptr(params.weights)
    var cp = _u32_ptr(params.cat_flags)
    # unread when params[20] is 0, the same contract the weights have;
    # the wrapper passes the X address rather than allocating a throwaway
    var ep = _f32_ptr(params.eval_x)
    var eyp = _f32_ptr(params.eval_y)

    var n_rows = params.i(0)
    var n_features = params.i(1)
    var n_weights = params.i(2)
    var n_flags = params.i(3)

    # Read BEFORE the GIL is released, like every other Python value here:
    # `params` is a Python list and touching it without the GIL is a data
    # race, not a slow path.
    var class_weights = List[Float32]()
    for i in range(n_class_weights):  # small-loop(n_class_weights: one weight per class): Python parameter list, not row data
        class_weights.append(Float32(params.f(35 + i)))
    # their `EGrowPolicy` by ORDINAL (enums.h order: SymmetricTree,
    # Depthwise, Lossguide), resolved to the spelling `train` takes
    var grow_code = params.i(31)
    var grow_name = String("SymmetricTree")
    if grow_code == 1:
        grow_name = String("Depthwise")
    elif grow_code == 2:
        grow_name = String("Lossguide")
    elif grow_code != 0:
        raise Error(
            "gbdt_fit: grow_policy code must be 0 (SymmetricTree), 1"
            " (Depthwise) or 2 (Lossguide), got " + String(grow_code)
        )

    # Backward-compatible optional tail after the counted class weights.
    # The default Python wrapper emits no tail, so old callers retain their
    # exact slot layout. An old extension rejects the new longer request.
    var min_split_gain = Float64(-1)
    if len(params) >= fixed_and_weights + 1:
        min_split_gain = params.f(fixed_and_weights)

    var min_child_hessian = Float64(-1)
    if len(params) >= fixed_and_weights + 2:
        min_child_hessian = params.f(fixed_and_weights + 1)

    var feature_fraction = Float64(1)
    if len(params) >= fixed_and_weights + 3:
        feature_fraction = params.f(fixed_and_weights + 2)

    # MultiRMSE's target dimension, the one value of the +4 tail
    var target_dim = 1
    if len(params) == fixed_and_weights + 4:
        target_dim = params.i(fixed_and_weights + 3)
        if target_dim < 2:
            raise Error(
                "gbdt_fit: the target-dimension tail needs a dimension >= 2,"
                " got " + String(target_dim)
            )

    # the pool's grouping, `group_id` resolved to run lengths by the wrapper;
    # read here, with the GIL held, like every other Python value
    var group_sizes = List[UInt32]()
    if len(params) >= fixed_and_weights + 5:
        var n_groups = params.i(fixed_and_weights + 4)
        if n_groups < 1:
            raise Error(
                "gbdt_fit: the group tail needs a positive group count, got "
                + String(n_groups)
            )
        # one memcpy (no per-group host loop)
        group_sizes = list_u32(_u32_ptr(params.i(fixed_and_weights + 3)), n_groups)

    # the PairLogit pairs tail; a count of -1 means generate them
    var pair_winners = List[UInt32]()
    var pair_losers = List[UInt32]()
    var pair_weights = List[Float32]()
    if len(params) == fixed_and_weights + 8:
        var n_pairs = params.i(fixed_and_weights + 6)
        if n_pairs != -1 and n_pairs < 1:
            raise Error(
                "gbdt_fit: the pairs tail needs a positive pair count or -1,"
                " got " + String(n_pairs)
            )
        if n_pairs > 0:
            var pp = _u32_ptr(params.i(fixed_and_weights + 5))
            var pw = _f32_ptr(params.i(fixed_and_weights + 7))
            # cpu3-bindings: the interleaved pairs split on the device and
            # the weights in one memcpy (no per-pair host loop)
            device_split_pairs(process_ctx[_DEVCTX_SLOT](), pp, n_pairs, pair_winners, pair_losers)
            pair_weights = list_f32(pw, n_pairs)

    var fp = GbdtFitParams(
        params.i(4),
        params.i(5),
        params.i(6),
        Float32(params.f(7)),
        Float32(params.f(8)),
        UInt64(params.i(9)),
        params.i(10),
        params.strs[0],
        Float32(params.f(11)),
        Float32(params.f(12)),
        Float32(params.f(13)),
        Float32(params.f(14)),
        Float32(params.f(15)),
        params.i(16),
        params.i(17),
        params.strs[1],
        Float32(params.f(18)),
        Float32(params.f(19)),
        params.strs[2],
        params.f(21),
        params.i(22),
        params.i(23),
        params.i(24),
        params.strs[3],
        Float32(params.f(25)),
        params.i(26) != 0,
        params.i(27),
        params.i(28),
        params.i(29),
        params.i(30),
        class_weights^,
        grow_name,
        params.i(32),
        params.i(33),
        min_split_gain,
        min_child_hessian,
        feature_fraction,
        border_type_name,
        boosting_type_name,
        fold_len_multiplier,
        fold_permutation_block,
    )
    var n_eval_rows = params.i(20)

    var result: GbdtFitResult
    var device_plan = _gbdt_device_plan()
    var ctx = process_ctx[_DEVCTX_SLOT]()
    result = gbdt_fit(
        ctx, xp, n_rows, n_features, yp, wp, n_weights,
        cp, n_flags, ep, eyp, n_eval_rows, fp,
        group_sizes=group_sizes,
        pair_winners=pair_winners,
        pair_losers=pair_losers,
        pair_weights=pair_weights,
        target_dim=target_dim,
        x_row_major=row_major,
        device_plan=device_plan,
    )
    ctx.synchronize()

    # perf-audit-3: the model text and the two curves are moved out of the
    # finished result (swapped with empty values), not copied.
    var text = String()
    var learn = List[Float64]()
    var test = List[Float64]()
    swap(text, result.text)
    swap(learn, result.learn_losses)
    swap(test, result.test_losses)
    return GbdtFitOutcome(text^, result.best_iteration, result.stopped_early, learn^, test^)


def gbdt_fit_two_level_feature_freq_run(
    xp: MutPointer[Float32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    yp: MutPointer[Float32, MutUntrackedOrigin],
    wp: MutPointer[Float32, MutUntrackedOrigin],
    n_weights: Int,
    sp: MutPointer[UInt32, MutUntrackedOrigin],
    n_sources: Int,
    learning_rate: Float32,
    l2_leaf_reg: Float32,
    random_seed: UInt64,
    device_plan: DevicePlan,
) raises -> String:
    """The device fit of `gbdt_fit_two_level_feature_freq_binding`; the door
    calls it with the GIL released."""
    var ctx = process_ctx[_DEVCTX_SLOT]()
    var text = gbdt_fit_two_level_feature_freq(
        ctx, xp, n_rows, n_features, yp, wp, n_weights, sp, n_sources,
        learning_rate, l2_leaf_reg, random_seed,
        device_plan=device_plan,
    )
    ctx.synchronize()
    return text^


def gbdt_fit_ordered_rmse_run(
    x_address: Int,
    yp: MutPointer[Float32, MutUntrackedOrigin],
    wp: MutPointer[Float32, MutUntrackedOrigin],
    pp: MutPointer[UInt32, MutUntrackedOrigin],
    n_rows: Int,
    n_features: Int,
    n_weights: Int,
    n_estimators: Int,
    max_depth: Int,
    border_count: Int,
    learning_rate: Float32,
    l2_leaf_reg: Float32,
    device_plan: DevicePlan,
) raises -> String:
    """The device fit of `gbdt_fit_ordered_rmse_binding`; the door calls it
    with the GIL released, after its checks."""
    from gbdt.train import train_ordered_rmse_ptr
    from gbdt.models.model_text import model_text

    var text: String
    # cpu2-l6-bindings: X (n_rows x n_features) is NOT copied into a host
    # List: `train_ordered_rmse_ptr` uploads it from the caller's
    # address once and refuses non-finite cells on the device. y, the
    # permutation and the weights (O(n)) still feed
    # `fit_ordered_rmse`'s host validation and fold plan as lists.
    # cpu3-bindings: one memcpy each (no per-row host loop)
    var ys = list_f32(yp, n_rows)
    var ws = list_f32(wp, n_weights)
    var permutation = list_u32(pp, n_rows)
    with process_ctx[_DEVCTX_SLOT]() as ctx:
        var trained = train_ordered_rmse_ptr(
            ctx, x_address, ys, n_rows, n_features, permutation,
            n_estimators, max_depth, border_count, learning_rate,
            l2_leaf_reg, ws,
            device_plan=device_plan,
        )
        text = model_text(trained)
        ctx.synchronize()
    return text^
