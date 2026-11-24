# SPDX-License-Identifier: Apache-2.0
# The splitters of python/mojolearn/model_selection.py, natively (lane
# mcb2-msel): the default folds (`_native_default_folds`), every
# splitter's `split` and `get_n_splits`, `check_cv` and
# `train_test_split`. Each split makes the same helper calls the Python
# made, with the same arguments, and the same random draws in the same order
# (the x_metrics programs of `CounterRng`, `fold_rows` and
# `stratified_fold_rows`, rebuilt with the shared native `_Prog`), so every
# split holds the same rows. A splitter with one split per row or per
# combination (LeaveOneOut, LeavePOut, LeaveOneGroupOut, LeavePGroupsOut)
# returns a lazy `map` over the compiled door, one split per item, as the
# Python generator made them; every other returns its list of splits (the
# Python made them all up front, or in one device program). The public
# `split` stays a generator shell that `yield from`s the result, so its
# refusals are still raised at the first `next`.
#
# UNPACK / RUN / PACK (device wave, lane dw-msel-parallel, 2026-11-21). This
# file is the boundary: it reads the splitter's stored fields, the row
# count, the labels (as int32 codes) and the seed into typed values
# (UNPACK), hands them to bindings/msel_split_plan_run.mojo, which computes
# every fold on typed values with no Python object (RUN, the GIL released
# once per call), and returns the (train, test) Arrays (PACK,
# bindings/msel_split_unpack.mojo `msel_split_finish`: the Arrays are
# allocated from the plan's sizes and RUN writes the rows into them). What
# stays here with Python objects is boundary work only: the splitter's
# class and field admission, its constructor state, `check_cv`, the lazy
# iterators of the per-row and per-combination splitters, the label input
# step, and `train_test_split`'s gather of the caller's inputs (through the
# resident helpers of bindings/msel_common_door.mojo).
from std.os import getenv
from core.shared_slot import SharedSlot
from bindings.native_loaded_module_door import NativeModules, native_loaded_module
from bindings.native_class_admission_door import class_record, class_admit, dict_keys_admitted, dict_keys_admit
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.base_helpers_key_unpack import base_helpers_key
from bindings.base_helpers_released_unpack import base_reduce_stat_int
from bindings.py_balanced import py_call, py_type, py_get, py_item, py_setitem, py_contains
from bindings.classical_arrays_door import classical_wrap_array
from bindings.native_buffer_owner_pack import native_storage_allocate
from bindings.native_owner_lifetime_door import native_release_admit
from bindings.msel_search_state_door import msel_parameter_values
from bindings.msel_member_device_unpack import msel_own_device_admit
from bindings.family_common_unpack import family_str
from bindings.family_common_pack import family_call
from bindings.label_input_sources_unpack import labels_list_integer_storage
from bindings.label_input_sources_pack import labels_series_values
from bindings.py_optional import py_present
from bindings.msel_common_door import (
    msel_b, msel_missed, msel_catch, msel_is_bool, msel_warn,
    msel_is_integral, msel_is_int_arg, msel_is_real, msel_int, msel_op, msel_cmp, msel_materialize,
    msel_try_materialize, msel_flatten, msel_encode, msel_labels_list, msel_n_samples, msel_is_array, msel_addr, msel_admit,
    msel_numeric_vector, msel_new, msel_read_i64, msel_arange, msel_range_index, msel_index_copy,
    msel_lazy, msel_u64, msel_take_labels, msel_resident_new, msel_resident_take, msel_resident_close,
)
# RUN: every fold computation, on typed values (no Python object).
from bindings.msel_split_run import msel_fold_rows_fits
from bindings.msel_split_plan_run import (
    MSEL_LABELS_ALIGNED, MSEL_LABELS_SORTED, MSEL_LABELS_FIRST_SEEN, MselCodes, MselSeed, MselSplitRoute,
    MselSplitPlan, msel_seed_refusal, msel_code_counts, msel_plan_default_folds, msel_plan_kfold,
    msel_plan_stratified, msel_plan_group_kfold, msel_plan_strat_group, msel_plan_time_series, msel_plan_shuffle,
    msel_plan_group_shuffle, msel_plan_strat_shuffle, msel_plan_select, msel_plan_group_set, msel_plan_indexed,
    msel_leave_one, msel_repeated_check, msel_plan_repeat, msel_shuffle_sizes_run,
)
# UNPACK (the entries, the label codes, the seed) and PACK (the fold arrays).
from bindings.msel_split_unpack import (
    MselLabelPin, msel_split_route, msel_label_codes, msel_label_array, msel_split_finish, msel_size_word,
    msel_size_sum,
)


# ---------------------------------------------------------------- small pieces


def msel_cls(name: String, context: PythonObject) raises -> PythonObject:
    """A model_selection class connected at import."""
    return context["family_msel_classes"][PythonObject(name)]


def msel_is(model: PythonObject, name: String, context: PythonObject) raises -> Bool:
    if name == "_Repeated":
        return py_type(model) is msel_cls("RepeatedKFold", context) or py_type(model) is msel_cls("RepeatedStratifiedKFold", context)
    return py_type(model) is msel_cls(name, context)


comptime _SPLITTER_CLASS_SLOT = "msel_splitter_plan_class_v1"


def _splitter_plan(cls: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = msel_b()
    if py_type(cls) is not py_type(py_type(PythonObject(0))):
        raise Error("TypeError: native splitters cannot invoke Python metaclass hooks")
    var plan = py_get(context["family_msel_splitter_layouts"], cls)
    if plan is None:
        raise Error("TypeError: model selection requires a connected native splitter; Python callbacks cannot execute")
    # perf2-msel: the class rows below depend only on the class; they are
    # checked once per class state (native_class_admission type watchers:
    # any class or ancestor modification invalidates the record). One split()
    # reads 6-8 fields and each read reached this walk. Python < 3.12 keeps
    # the walk on every call.
    var mods = NativeModules()
    var held = class_record(mods, _SPLITTER_CLASS_SLOT, cls)
    if held is not None and Int(py=held) == Int(plan._obj_ptr):
        return plan
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    if len(mro) != len(plan[0]):
        raise Error("TypeError: native splitter ancestry changed")
    var names = Python.tuple("__dict__", "__getattribute__", "__getattr__", "__setattr__", "__del__")
    for i in range(len(mro)):  # driver-loop: connected storage/lifecycle metadata before reading an instance
        if mro[i] is not plan[0][i]:
            raise Error("TypeError: native splitter ancestry changed")
        var mapping = py_call(b.type.__dict__["__dict__"].__get__, mro[i])
        for j in range(len(names)):  # driver-loop: fixed native hook identity checks
            if py_get(mapping, names[j]) is not plan[2][i][j]:
                raise Error("TypeError: native splitter cannot invoke Python lifecycle or attribute hooks")
        var stored = py_call(b.list, plan[4][i])
        for j in range(len(stored)):  # driver-loop: reject properties replacing stored configuration fields
            if py_get(mapping, stored[j]) is not plan[4][i][stored[j]]:
                raise Error("TypeError: native splitter configuration descriptors changed")
    # The payload is the plan's address (layouts live in the import-time
    # context for the process), never the plan: it holds the class's MRO.
    class_admit(mods, _SPLITTER_CLASS_SLOT, cls, PythonObject(Int(plan._obj_ptr)))
    return plan


def _splitter_fields(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = msel_b()
    var cls = py_type(model)
    var plan = _splitter_plan(cls, context)
    var fields = py_call(plan[1], model, cls)
    # perf2-msel: the exact-str key admission is kept per dictionary state
    # (native_class_admission dictionary watchers; any edit forces the scan).
    if dict_keys_admitted(fields):
        return fields
    var keys = py_call(b.list, fields)
    for i in range(len(keys)):  # driver-loop: dictionary keys cannot invoke custom equality
        if py_type(keys[i]) is not b.str:
            raise Error("TypeError: native splitter fields require ordinary string keys")
    if py_type(fields) is b.dict:
        dict_keys_admit(fields)
    return fields


def msel_field(model: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    var fields = _splitter_fields(model, context)
    var defaults = context["family_msel_splitter_layouts"][py_type(model)][3]
    return _splitter_value(fields, defaults, name, context)


# N070: the splitter class as a native code, resolved once per split call
# (one identity pass over the connected classes), instead of about 25
# `msel_is` tests each building a str and reading two dictionaries.
comptime SPLITTER_OTHER = 0
comptime SPLITTER_KFOLD = 1
comptime SPLITTER_STRATIFIED_KFOLD = 2
comptime SPLITTER_GROUP_KFOLD = 3
comptime SPLITTER_STRATIFIED_GROUP_KFOLD = 4
comptime SPLITTER_TIME_SERIES = 5
comptime SPLITTER_LOO = 6
comptime SPLITTER_LPO = 7
comptime SPLITTER_LOGO = 8
comptime SPLITTER_LPGO = 9
comptime SPLITTER_PREDEFINED = 10
comptime SPLITTER_SHUFFLE = 11
comptime SPLITTER_GROUP_SHUFFLE = 12
comptime SPLITTER_STRATIFIED_SHUFFLE = 13
comptime SPLITTER_REPEATED_KFOLD = 14
comptime SPLITTER_REPEATED_STRATIFIED_KFOLD = 15
comptime SPLITTER_ITERABLE = 16


def _splitter_kind(cls: PythonObject, context: PythonObject) raises -> Int:
    """The SPLITTER_* code of an exact connected splitter class (OTHER when
    it is none of them). Same identities `msel_is` compares."""
    var classes = context["family_msel_classes"]
    var names: List[String] = [
        "KFold", "StratifiedKFold", "GroupKFold", "StratifiedGroupKFold", "TimeSeriesSplit", "LeaveOneOut",
        "LeavePOut", "LeaveOneGroupOut", "LeavePGroupsOut", "PredefinedSplit", "ShuffleSplit", "GroupShuffleSplit",
        "StratifiedShuffleSplit", "RepeatedKFold", "RepeatedStratifiedKFold", "_IterableCV",
    ]
    for i in range(len(names)):  # small-loop(16 connected splitter classes): identity compare only
        var known = py_get(classes, PythonObject(names[i]))
        if known is not None and cls is known:
            return i + 1
    return SPLITTER_OTHER


struct MselSplitter(Movable):
    """One split call's splitter (N070): its class code and its admitted
    stored fields and class defaults, read once. `field` applies the same
    per-value checks as `msel_field` without repeating the class-plan and
    dictionary admission for every field."""
    var model: PythonObject
    var kind: Int
    var fields: PythonObject
    var defaults: PythonObject

    def __init__(out self, model: PythonObject, context: PythonObject) raises:
        self.model = model
        self.fields = _splitter_fields(model, context)
        self.defaults = context["family_msel_splitter_layouts"][py_type(model)][3]
        self.kind = _splitter_kind(py_type(model), context)

    def repeated(self) -> Bool:
        return self.kind == SPLITTER_REPEATED_KFOLD or self.kind == SPLITTER_REPEATED_STRATIFIED_KFOLD

    def field(self, name: String, context: PythonObject) raises -> PythonObject:
        return _splitter_value(self.fields, self.defaults, name, context)


def _splitter_value(fields: PythonObject, defaults: PythonObject, name: String,
                    context: PythonObject) raises -> PythonObject:
    """A stored splitter field (or its class default) with the per-value
    admission `msel_field` applies."""
    if not py_contains(fields, PythonObject(name)) and not py_contains(defaults, PythonObject(name)):
        raise Error("AttributeError: splitter has no stored field " + name)
    var value = py_get(fields, PythonObject(name), py_get(defaults, PythonObject(name)))
    if name == "cv":
        if value is not msel_cls("KFold", context) and value is not msel_cls("StratifiedKFold", context):
            raise Error("TypeError: repeated model selection requires a native KFold or StratifiedKFold configuration")
    else:
        msel_admit(value, context)
        var kind = py_type(value)
        if name == "_pairs" and kind is not msel_b().list and kind is not msel_b().tuple:
            raise Error("TypeError: stored fold pairs require a native list or tuple")
        if name == "cvargs":
            if kind is not msel_b().dict:
                raise Error("TypeError: repeated splitter options require a native dictionary")
            var keys = py_call(msel_b().list, value)
            for i in range(len(keys)):  # driver-loop: option keys before native lookup or copying
                if py_type(keys[i]) is not msel_b().str:
                    raise Error("TypeError: repeated splitter option names require native strings")
        if name != "_folds" and name != "_pairs" and name != "cvargs" and name != "test_fold":
            var b = msel_b()
            if value is not None and py_type(value) is not b.str and not msel_is_real(value, context) and not msel_is_bool(value, context):
                raise Error("TypeError: splitter settings require native scalar values")
    return value


struct _MselSabotage(Defaultable, Movable):
    #: -1 until the first read, then 0 or 1.
    var state: Int

    def __init__(out self):
        self.state = -1


comptime _MSEL_SABOTAGE = SharedSlot[_MselSabotage, "MojolearnMselFoldSabotageV1"]


def msel_sabotage() -> Bool:
    """`_sabotage_requested()`: both MOJOLEARN_FOLD_ORDER_SABOTAGE and
    MOJOLEARN_HOST_ALLOW_SABOTAGE set to 1 in the process environment, read
    once per process (lane envgate-classical-deviceplan,
    plan ENV_GATES_2026-11-15 (git history); before: two getenv per split, cv and
    search call). The verify tools set both for the process they start
    (pixi `check-cross-val-folds-oracle-sabotage`, tools/identity_break.py)."""
    try:
        var slot = _MSEL_SABOTAGE.get_or_create_ptr()
        if slot[].state < 0:
            var on = (getenv("MOJOLEARN_FOLD_ORDER_SABOTAGE", "") == "1"
                      and getenv("MOJOLEARN_HOST_ALLOW_SABOTAGE", "") == "1")
            slot[].state = 1 if on else 0
        return slot[].state == 1
    except:
        return False


def msel_check_splits(n_splits: PythonObject, context: PythonObject) raises -> Int:
    """`_check_splits`."""
    if msel_is_bool(n_splits, context) or not msel_is_integral(n_splits, context):
        raise Error("The number of folds must be of Integral type.")
    if msel_cmp("le", n_splits, PythonObject(1)):
        raise Error("k-fold cross-validation requires at least one train/test split by setting "
                    + "n_splits=2 or more, got n_splits=" + family_str(n_splits) + ".")
    return msel_int(n_splits)


def msel_new_kfold(name: String, n_splits: PythonObject, shuffle: PythonObject, random_state: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    """`KFold(n_splits, shuffle=..., random_state=...)` (or StratifiedKFold):
    `_KFoldBase.__init__`'s checks, then its three attributes."""
    var k = msel_check_splits(n_splits, context)
    if not msel_is_bool(shuffle, context):
        raise Error("TypeError: shuffle must be True or False")
    if not Bool(py=family_call(msel_b().bool, shuffle)) and random_state is not None:
        raise Error("Setting a random_state has no effect since shuffle is False. You should "
                    + "leave random_state to its default (None), or set shuffle=True.")
    var cls = msel_cls(name, context)
    var plan = _splitter_plan(cls, context)
    var cv = native_storage_allocate(cls)
    var fields = py_call(plan[1], cv, cls)
    py_setitem(fields, "n_splits", PythonObject(k))
    py_setitem(fields, "shuffle", shuffle)
    py_setitem(fields, "random_state", random_state)
    return cv


def msel_splitter_initialize(model: PythonObject, args: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    # The public constructor hands over untouched arguments. Admission precedes
    # formatting, numeric conversion, metadata reads and any overwritten release.
    var fields = _splitter_fields(model, context)
    msel_admit(args, context)
    var kind = String(args[0])
    var updates = Python.dict()
    if kind == "kfold":
        var k = msel_check_splits(args[1], context)
        if not msel_is_bool(args[2], context):
            raise Error("TypeError: shuffle must be True or False")
        if not msel_truth(args[2]) and args[3] is not None:
            raise Error("Setting a random_state has no effect since shuffle is False. You should "
                        + "leave random_state to its default (None), or set shuffle=True.")
        py_setitem(updates, "n_splits", PythonObject(k))
        py_setitem(updates, "shuffle", args[2])
        py_setitem(updates, "random_state", args[3])
    elif kind == "time":
        py_setitem(updates, "n_splits", PythonObject(msel_check_splits(args[1], context)))
        py_setitem(updates, "max_train_size", args[2])
        py_setitem(updates, "test_size", args[3])
        py_setitem(updates, "gap", args[4])
    elif kind == "repeated":
        if not msel_is_int_arg(args[2], context) or msel_int(args[2]) <= 0:
            raise Error("Number of repetitions must be greater than 0.")
        var base = String("KFold") if msel_is(model, "RepeatedKFold", context) else String("StratifiedKFold")
        var cvargs = Python.dict()
        py_setitem(cvargs, "n_splits", args[1])
        py_setitem(updates, "cv", msel_cls(base, context))
        py_setitem(updates, "cvargs", cvargs)
        py_setitem(updates, "n_repeats", PythonObject(msel_int(args[2])))
        py_setitem(updates, "random_state", args[3])
    elif kind == "predefined":
        var folds = msel_index_copy(args[1], context)
        var values = Python.list()
        var data = msel_read_i64(folds)
        for i in range(len(data)):  # driver-loop: finished public fold labels, without Array.tolist callbacks
            _ = py_call(values.append, PythonObject(data[i]))
        py_setitem(updates, "_folds", folds)
        py_setitem(updates, "test_fold", values)
    elif kind == "iterable":
        py_setitem(updates, "_pairs", msel_index_pairs(args[1], context))
    elif kind != "device":
        raise Error("ValueError: unknown native splitter constructor")
    # device= (lane vdf-msel): the last item of every public constructor's
    # tuple ("device": a constructor that stores its other parameters itself).
    # Admitted where given, in the estimators' words; stored only when given,
    # so a splitter built with device=None has the fields, get_params and repr
    # it had before (its split reads an absent device as None).
    if kind != "iterable":
        var device = args[len(args) - 1]
        if device is not None:
            msel_own_device_admit(device, msel_b(), context)
            py_setitem(updates, "device", device)
    var keys = py_call(msel_b().list, updates)
    for i in range(len(keys)):  # driver-loop: refuse custom finalizers before releasing overwritten state
        if py_contains(fields, keys[i]) and fields[keys[i]] is not updates[keys[i]]:
            if String(keys[i]) == "cv":
                if fields[keys[i]] is not msel_cls("KFold", context) and fields[keys[i]] is not msel_cls("StratifiedKFold", context):
                    raise Error("TypeError: repeated splitter contains unsupported class state")
            else:
                native_release_admit(fields[keys[i]], context)
    for i in range(len(keys)):  # driver-loop: commit fully prepared constructor state through native dictionary slots
        py_setitem(fields, keys[i], updates[keys[i]])
    return PythonObject(None)


def msel_flag_truth(value: PythonObject, context: PythonObject) raises -> Bool:
    # Public CV flags have scalar truth semantics; exact built-in containers
    # also have native length truth. Refuse Python-defined __bool__ paths before
    # invoking conversion, even when their objects are valid data inputs.
    msel_admit(value, context)
    var b = msel_b()
    var cls = py_type(value)
    if value is None or msel_is_bool(value, context) or msel_is_real(value, context):
        return msel_truth(value)
    if cls is b.str or cls is b.bytes or cls is b.bytearray or cls is b.list or cls is b.tuple or cls is b.dict or cls is b.range:
        return msel_truth(value)
    raise Error("TypeError: cross-validation flags require native scalar truth values")


def msel_truth(value: PythonObject) raises -> Bool:
    return Bool(py=family_call(msel_b().bool, value))


# ---------------------------------------------------------------- seed UNPACK


def msel_seed(random_state: PythonObject, used: Bool, context: PythonObject) raises -> MselSeed:
    """UNPACK of `CounterRng(random_state)`'s seed: None draws eight bytes
    from the operating system. A splitter that will not draw (`used` False)
    reads nothing. An unusable value is carried as `bad` and refused by RUN
    where the splitter first draws, so the size refusals that came first
    still come first."""
    if not used:
        return MselSeed(0, False)
    var state = random_state
    if state is None:
        state = family_call(msel_b().int.from_bytes, py_call(native_loaded_module("os").urandom, PythonObject(8)),
                            PythonObject("little"))
    if (msel_is_bool(state, context) or not msel_is_integral(state, context)
            or msel_cmp("lt", state, PythonObject(0))):
        return MselSeed(0, True)
    return MselSeed(msel_u64(state), False)


# ---------------------------------------------------------------- label UNPACK


def msel_discrete_codes(y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_discrete_codes(y)`: (classes, int32 codes) when y is discrete, else
    None."""
    var b = msel_b()
    var arr = msel_try_materialize(y, "y", Python.tuple(b.TypeError, b.ValueError), context)
    var have = not msel_missed(arr, context)
    var flat = arr
    # The native materializer is C contiguous; reduce_stat uses size/address
    # and msel_encode flattens native storage, so no Array.reshape callback
    # or temporary metadata wrapper is needed for a column vector.
    if have:
        var dtype = String(py=arr.dtype)
        if (dtype == "<f4" or dtype == "<f8") and Int(py=flat.size) > 0:
            var integral = base_reduce_stat_int(
                base_helpers_key(context["family_msel_entries"]), msel_addr(flat),
                Int(py=context["family_msel_native_code"][PythonObject(dtype)]), Int(py=flat.size),
                Int(py=context["family_msel_reduce_integral"]))
            _ = flat  # keep the Python object alive through the native call above
            if integral == 0:
                return PythonObject(None)
            _ = flat  # keep the Python object alive through the native calls above
    return msel_encode(flat if have else y, context)


def _aligned_labels(y: PythonObject, mut pin: MselLabelPin, context: PythonObject) raises -> Bool:
    """The first half of `_first_seen_native(y)`: y as one sorted-order code
    per row, in `pin`. False (what the Python returned None for) when y is
    None, has no class, or its admission raised TypeError, ValueError or
    OverflowError; the caller then takes its other label route."""
    if y is None:
        return False
    var b = msel_b()
    var classes = Python.tuple(b.TypeError, b.ValueError, b.OverflowError)
    try:
        var n = msel_n_samples(y, context)
        pin = msel_label_codes(y, n, "y", context, True, True)
        pin.form = MSEL_LABELS_ALIGNED
        return pin.view.m >= 1
    except error:
        if msel_catch(error, classes):
            return False
        raise Error(String(error))


def _strat_labels(y: PythonObject, context: PythonObject) raises -> MselLabelPin:
    """StratifiedKFold's labels: one sorted-order code per row (RUN ranks
    the classes by first row), else the encoder's codes as they are (RUN
    renumbers them with `first_seen_i32`)."""
    var pin = MselLabelPin()
    if _aligned_labels(y, pin, context):
        return pin^
    var plain = msel_label_codes(y, 0, "y", context, False, False)
    plain.form = MSEL_LABELS_SORTED
    return plain^


def _encode_first_seen(values: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_encode_first_seen(values)`: (codes list, k) by dict.fromkeys order
    (the builtins' C loops over the label objects)."""
    var b = msel_b()
    var keys = py_call(b.list, py_call(b.dict.fromkeys, values))
    var index = py_call(b.dict, py_call(b.zip, keys, py_call(b.range, PythonObject(len(keys)))))
    var codes = family_call(b.list, py_call(b.map, py_call(b.getattr, index, PythonObject("__getitem__")), values))
    return Python.tuple(codes, PythonObject(len(index)))


def _strat_group_labels(y: PythonObject, context: PythonObject) raises -> MselLabelPin:
    """StratifiedGroupKFold's labels: one sorted-order code per row, else
    first-seen codes by the builtins' dictionary order over the label
    objects."""
    var pin = MselLabelPin()
    if _aligned_labels(y, pin, context):
        return pin^
    var got = _encode_first_seen(msel_labels_list(y, "y", context), context)
    var store = family_call(native_loaded_module("array").array, PythonObject("i"), got[0])
    var yenc = classical_wrap_array(store, Python.tuple(len(store)), "<i4", "C", context)
    var listed = MselLabelPin()
    listed.view = MselCodes(msel_addr(yenc), Int(py=yenc.size), Int(py=got[1]))
    listed.owner = yenc
    listed.form = MSEL_LABELS_FIRST_SEEN
    return listed^


# ---------------------------------------------------------------- default folds


def msel_default_folds(y: PythonObject, n_splits: PythonObject, classifier: Bool,
                       context: PythonObject) raises -> PythonObject:
    """`_native_default_folds(y, n_splits, classifier)`."""
    # UNPACK
    var n = msel_n_samples(y, context)
    if msel_is_bool(n_splits, context) or not msel_is_integral(n_splits, context) or msel_cmp("lt", n_splits, PythonObject(2)):
        raise Error("cv must specify at least two folds")
    if msel_cmp("gt", n_splits, PythonObject(n)):
        raise Error("cv cannot exceed the number of samples")
    var k = msel_int(n_splits)
    var codes = MselCodes(0, 0, 0)
    var coded = False
    # Optional, not a PythonObject(None) sentinel (modular/modular#7546).
    var held = Optional[PythonObject]()
    if classifier:
        var got = msel_discrete_codes(y, context)
        if got is not None:
            held = py_present(got[1])
            codes = MselCodes(msel_addr(held.value()), n, len(got[0]))
            coded = True
    var pin = msel_split_route(context, False, False)
    var route = pin.route
    var sabotage = msel_sabotage()
    # RUN
    var plan = MselSplitPlan()
    with GILReleased(Python()):
        plan = msel_plan_default_folds(route, n, k, codes, coded, sabotage)
    # PACK (allocation), RUN (rows)
    var pairs = msel_split_finish(pin, plan, context)
    _ = held  # the codes stay alive through the plan above
    return pairs


# ---------------------------------------------------------------- KFold family


def msel_kfold_split(x: PythonObject, k: Int, shuffle: Bool, random_state: PythonObject,
                     context: PythonObject) raises -> PythonObject:
    """`KFold.split`."""
    var n = msel_n_samples(x, context)
    var seed = msel_seed(random_state, shuffle, context)
    var pin = msel_split_route(context, False, shuffle)
    var route = pin.route
    var sabotage = msel_sabotage()
    var plan = MselSplitPlan()
    with GILReleased(Python()):
        plan = msel_plan_kfold(route, n, k, shuffle, seed, sabotage)
    return msel_split_finish(pin, plan, context)


def msel_stratified_split(y: PythonObject, k: Int, shuffle: Bool, random_state: PythonObject,
                          context: PythonObject) raises -> PythonObject:
    """`StratifiedKFold.split`."""
    var labels = _strat_labels(y, context)
    var seed = msel_seed(random_state, shuffle, context)
    var view = labels.view
    var form = labels.form
    # The base helper table is called only past the one-program bounds
    # (strat_fold_assign_i32, then the folds' bincount): its key is not
    # read for the program route.
    var pin = msel_split_route(context, not (k <= 256 and msel_fold_rows_fits(view.n, k)), True)
    var route = pin.route
    var plan = MselSplitPlan()
    if shuffle and seed.bad:
        # The refusal of an unusable random_state, in its old place: after
        # the class-size refusal and after the "least populated class"
        # warning. The plan is made without a draw (it raises the size
        # refusal and records the warning), the warning is issued, then the
        # seed is refused. Only this refusal path pays for the extra plan.
        with GILReleased(Python()):
            plan = msel_plan_stratified(route, view, form, k, False, seed)
        if plan.warned:
            msel_warn(plan.warning, msel_b().UserWarning, context)
        _ = labels  # the codes stay alive through the plan above
        raise Error(msel_seed_refusal())
    with GILReleased(Python()):
        plan = msel_plan_stratified(route, view, form, k, shuffle, seed)
    var pairs = msel_split_finish(pin, plan, context)
    _ = labels  # the codes stay alive through the plan above
    return pairs


def msel_group_kfold_split(x: PythonObject, groups: PythonObject, k: Int, shuffle: Bool,
                           random_state: PythonObject, context: PythonObject) raises -> PythonObject:
    """`GroupKFold.split`."""
    var n = msel_n_samples(x, context)
    var labels = msel_label_codes(groups, n, "groups", context, True, True)
    var seed = msel_seed(random_state, shuffle, context)
    var pin = msel_split_route(context, False, shuffle)
    var route = pin.route
    var view = labels.view
    var plan = MselSplitPlan()
    with GILReleased(Python()):
        plan = msel_plan_group_kfold(route, view, k, shuffle, seed)
    var pairs = msel_split_finish(pin, plan, context)
    _ = labels  # the codes stay alive through the plan and the fill above
    return pairs


def msel_strat_group_split(x: PythonObject, y: PythonObject, groups: PythonObject, k: Int, shuffle: Bool,
                           random_state: PythonObject, context: PythonObject) raises -> PythonObject:
    """`StratifiedGroupKFold.split`."""
    var n = msel_n_samples(x, context)
    var group_labels = msel_label_codes(groups, n, "groups", context, True, True)
    var labels = _strat_group_labels(y, context)
    var seed = msel_seed(random_state, shuffle, context)
    var pin = msel_split_route(context, True, shuffle)
    var route = pin.route
    var view = labels.view
    var form = labels.form
    var group_view = group_labels.view
    var plan = MselSplitPlan()
    with GILReleased(Python()):
        plan = msel_plan_strat_group(route, n, view, form, group_view, k, shuffle, seed)
    var pairs = msel_split_finish(pin, plan, context)
    _ = labels  # both code vectors stay alive through the plan above
    _ = group_labels
    return pairs


# ---------------------------------------------------------------- the other splitters


def msel_time_series_split(sp: MselSplitter, x: PythonObject, context: PythonObject) raises -> PythonObject:
    """`TimeSeriesSplit.split`."""
    var n = msel_n_samples(x, context)
    var k = msel_int(sp.field("n_splits", context))
    var ts_field = sp.field("test_size", context)
    var has_ts = ts_field is not None
    var ts = msel_int(ts_field) if has_ts else 0
    var gap = msel_int(sp.field("gap", context))
    var max_train_field = sp.field("max_train_size", context)
    # `max_train_size` was read at the first split, so its own refusal (a
    # value with no truth or integer form) follows the fold-count refusals
    # and is not raised at all for a splitter that makes no split. Same
    # order here: an unreadable value is set aside, the plan raises its
    # refusals, and only a plan with a split reads the value again to raise
    # its error. The usual value costs nothing more than the one read.
    var max_train = 0
    var unread = False
    try:
        max_train = msel_int(max_train_field) if msel_truth(max_train_field) else 0
    except:
        unread = True
    var pin = msel_split_route(context, False, False)
    var plan = msel_plan_time_series(n, k, ts, has_ts, gap, max_train)
    if unread and plan.splits() > 0:
        max_train = msel_int(max_train_field) if msel_truth(max_train_field) else 0
    return msel_split_finish(pin, plan, context)


def msel_loo_split(x: PythonObject, sp: MselSplitter, context: PythonObject) raises -> PythonObject:
    """`LeaveOneOut.split`: one split per row, made on demand."""
    var n = msel_n_samples(x, context)
    if n <= 1:
        raise Error("Cannot perform LeaveOneOut with n_samples=" + String(n) + ".")
    var b = msel_b()
    var items = py_call(b.zip, py_call(native_loaded_module("itertools").repeat, PythonObject(n)),
                        py_call(b.range, PythonObject(n)))
    return msel_lazy(sp.model, "family_msel_split_loo_at", items, context)


def msel_loo_at(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """Split s of LeaveOneOut over n rows: test row s, train every other."""
    var n = Int(py=args[0])
    var s = Int(py=args[1])
    var train = msel_new(n - 1, "<i8", context)
    var test = msel_new(1, "<i8", context)
    var pin = msel_split_route(context, False, False)
    var route = pin.route
    var train_at = msel_addr(train)
    var test_at = msel_addr(test)
    with GILReleased(Python()):
        msel_leave_one(route, n, s, train_at, test_at)
    _ = test  # keep the Python object alive through the native call above
    _ = train  # keep the Python object alive through the native call above
    return Python.tuple(train, test)


def msel_lpo_split(x: PythonObject, sp: MselSplitter, context: PythonObject) raises -> PythonObject:
    """`LeavePOut.split`: the combinations in lexicographic order (the
    order `next_combination_i64` stepped through), one split each, made on
    demand."""
    var n = msel_n_samples(x, context)
    var p_field = sp.field("p", context)
    if msel_cmp("le", PythonObject(n), p_field):
        raise Error("p=" + family_str(p_field) + " must be strictly less than the number of samples=" + String(n))
    var p = msel_int(p_field)
    if p < 0:
        raise Error("r must be non-negative")
    if p < 1:
        var one = Python.list()
        _ = py_call(one.append, Python.tuple(msel_arange(0, n, context), msel_new(0, "<i8", context)))
        return one
    var b = msel_b()
    var itertools = native_loaded_module("itertools")
    var combos = py_call(itertools.combinations, py_call(b.range, PythonObject(n)), PythonObject(p))
    var items = py_call(b.zip, py_call(itertools.repeat, PythonObject(n)), combos)
    return msel_lazy(sp.model, "family_msel_split_lpo_at", items, context)


def _only_pair(pairs: PythonObject) raises -> PythonObject:
    """The one (train, test) pair of a single-split plan."""
    var pair = py_item(pairs, PythonObject(0))
    return pair


def msel_lpo_at(args: PythonObject, context: PythonObject) raises -> PythonObject:
    """One LeavePOut split: the combination's rows as the test set."""
    var n = Int(py=args[0])
    var combo = family_call(native_loaded_module("array").array, PythonObject("q"), args[1])
    var k = len(combo)
    var rows = Int(py=py_call(combo.buffer_info)[0]) if k else 0
    var pin = msel_split_route(context, False, False)
    var plan = msel_plan_indexed(n, rows, k)
    var pairs = msel_split_finish(pin, plan, context)
    # The integer address does not retain its array owner. Keep the complete
    # combination alive until the native mask and both row arrays are built.
    _ = combo
    return _only_pair(pairs)


def _group_state(labels: MselLabelPin, route: MselSplitRoute, context: PythonObject) raises -> PythonObject:
    """What a lazy group splitter carries from split to split: (the int32
    group codes, n, each group's row count)."""
    var view = labels.view
    var counted = List[Int64]()
    with GILReleased(Python()):
        counted = msel_code_counts(route, view)
    var counts = Python.list()
    for g in range(view.m):  # driver-loop: over groups
        _ = py_call(counts.append, PythonObject(Int(counted[g])))
    return Python.tuple(msel_label_array(labels, context), PythonObject(view.n), py_call(msel_b().tuple, counts))


def msel_logo_split(x: PythonObject, groups: PythonObject, sp: MselSplitter,
                    context: PythonObject) raises -> PythonObject:
    """`LeaveOneGroupOut.split`: one split per group, made on demand."""
    var n = msel_n_samples(x, context)
    var labels = msel_label_codes(groups, n, "groups", context, True, True)
    if labels.view.m <= 1:
        raise Error("The groups parameter contains fewer than 2 unique groups (" + family_str(labels.classes) + "). "
                    + "LeaveOneGroupOut expects at least 2.")
    var pin = msel_split_route(context, False, False)
    var state = _group_state(labels, pin.route, context)
    var b = msel_b()
    var items = py_call(b.zip, py_call(native_loaded_module("itertools").repeat, state),
                        py_call(b.range, PythonObject(labels.view.m)))
    return msel_lazy(sp.model, "family_msel_split_logo_at", items, context)


def msel_logo_at(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var state = args[0]
    var g = Int(py=args[1])
    var codes = state[0]
    var counts = state[2]
    var view = MselCodes(msel_addr(codes), Int(py=state[1]), len(counts))
    var wants: List[Int] = [g]
    var sizes: List[Int] = [Int(py=counts[g])]
    var pin = msel_split_route(context, False, False)
    var plan = msel_plan_select(pin.route, view, wants, False, sizes)
    var pairs = msel_split_finish(pin, plan, context)
    _ = codes  # keep the Python object alive through the native call above
    return _only_pair(pairs)


def msel_lpgo_split(x: PythonObject, groups: PythonObject, sp: MselSplitter,
                    context: PythonObject) raises -> PythonObject:
    """`LeavePGroupsOut.split`: one split per combination of groups, made
    on demand."""
    var n = msel_n_samples(x, context)
    var labels = msel_label_codes(groups, n, "groups", context, True, True)
    var n_groups = sp.field("n_groups", context)
    if msel_cmp("ge", n_groups, PythonObject(labels.view.m)):
        raise Error("The groups parameter contains fewer than (or equal to) n_groups (" + family_str(n_groups)
                    + ") numbers of unique groups (" + family_str(labels.classes) + ").")
    var pin = msel_split_route(context, False, False)
    var state = _group_state(labels, pin.route, context)
    var b = msel_b()
    var itertools = native_loaded_module("itertools")
    var combos = py_call(itertools.combinations, py_call(b.range, PythonObject(labels.view.m)), n_groups)
    var items = py_call(b.zip, py_call(itertools.repeat, state), combos)
    return msel_lazy(sp.model, "family_msel_split_lpgo_at", items, context)


def msel_lpgo_at(args: PythonObject, context: PythonObject) raises -> PythonObject:
    var state = args[0]
    var combo = args[1]
    var counts = state[2]
    var chosen = List[Int]()
    var size = 0
    for i in range(len(combo)):  # driver-loop: over the chosen groups of one combination
        var g = Int(py=combo[i])
        chosen.append(g)
        size += Int(py=counts[g])
    var codes = state[0]
    var view = MselCodes(msel_addr(codes), Int(py=state[1]), len(counts))
    var pin = msel_split_route(context, False, False)
    var route = pin.route
    var plan = MselSplitPlan()
    with GILReleased(Python()):
        plan = msel_plan_group_set(route, view, chosen, size)
    var pairs = msel_split_finish(pin, plan, context)
    _ = codes  # keep the Python object alive through the native calls above
    return _only_pair(pairs)


def msel_sizes_of(n: Int, test_size: PythonObject, train_size: PythonObject, default_test: PythonObject,
                  context: PythonObject) raises -> Tuple[Int, Int]:
    """`_validate_shuffle_split`: UNPACK of the sizes (the default test size
    when both are None; each size's typed word; the real sizes' sum check),
    then the one typed body `msel_shuffle_sizes_run` (lane mf-split5: a
    member fit computes its sizes with the same body over its own rows).
    Before, the arithmetic ran on the Python values here."""
    var test = test_size
    if test_size is None and train_size is None:
        test = default_test
    var over = msel_size_sum(test, train_size, context)
    return msel_shuffle_sizes_run(n, msel_size_word(test, context), msel_size_word(train_size, context), over[0],
                                  over[1])


def msel_shuffle_split(x: PythonObject, n_splits: Int, test_size: PythonObject, train_size: PythonObject,
                       random_state: PythonObject, default_test: PythonObject, context: PythonObject) raises -> PythonObject:
    """`ShuffleSplit.split`."""
    var n = msel_n_samples(x, context)
    var sizes = msel_sizes_of(n, test_size, train_size, default_test, context)
    var n_train = sizes[0]
    var n_test = sizes[1]
    var seed = msel_seed(random_state, True, context)
    var pin = msel_split_route(context, False, True)
    var route = pin.route
    var plan = MselSplitPlan()
    with GILReleased(Python()):
        plan = msel_plan_shuffle(route, n, n_splits, n_train, n_test, seed)
    return msel_split_finish(pin, plan, context)


def msel_group_shuffle_split(x: PythonObject, groups: PythonObject, n_splits: Int, test_size: PythonObject,
                             train_size: PythonObject, random_state: PythonObject, default_test: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    """`GroupShuffleSplit.split`."""
    var labels = msel_label_codes(groups, msel_n_samples(x, context), "groups", context, True, True)
    var view = labels.view
    var sizes = msel_sizes_of(view.m, test_size, train_size, default_test, context)
    var n_train = sizes[0]
    var n_test = sizes[1]
    var seed = msel_seed(random_state, True, context)
    var pin = msel_split_route(context, False, True)
    var route = pin.route
    var plan = MselSplitPlan()
    with GILReleased(Python()):
        plan = msel_plan_group_shuffle(route, view, n_splits, n_train, n_test, seed)
    var pairs = msel_split_finish(pin, plan, context)
    _ = labels  # the codes stay alive through the plan and the fill above
    return pairs


def msel_strat_shuffle_split(y: PythonObject, n_splits: Int, test_size: PythonObject, train_size: PythonObject,
                             random_state: PythonObject, default_test: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    """`StratifiedShuffleSplit._native_split`: every split, in split order
    (the same draws in the same order the generator made)."""
    var n0 = 0 if y is None else msel_n_samples(y, context)
    var labels = msel_label_codes(y, n0, "y", context, True, True)
    var view = labels.view
    var sizes = msel_sizes_of(view.n, test_size, train_size, default_test, context)
    var n_train = sizes[0]
    var n_test = sizes[1]
    var seed = msel_seed(random_state, True, context)
    var pin = msel_split_route(context, False, True)
    var route = pin.route
    var plan = MselSplitPlan()
    with GILReleased(Python()):
        plan = msel_plan_strat_shuffle(route, view, n_splits, n_train, n_test, seed)
    var pairs = msel_split_finish(pin, plan, context)
    _ = labels  # the codes stay alive through the plan above
    return pairs


def msel_predefined_split(sp: MselSplitter, context: PythonObject) raises -> PythonObject:
    """`PredefinedSplit.split`: each fold value other than -1, in sorted
    order."""
    var folds = sp.field("_folds", context)
    var labels = msel_label_codes(folds, Int(py=folds.size), "test_fold", context, True, True)
    var view = labels.view
    var wants = List[Int]()
    for c in range(len(labels.classes)):  # driver-loop: over fold values
        if msel_cmp("ne", labels.classes[c], PythonObject(-1)):
            wants.append(c)
    var pin = msel_split_route(context, False, False)
    var route = pin.route
    var none = List[Int]()
    var plan = MselSplitPlan()
    with GILReleased(Python()):
        plan = msel_plan_select(route, view, wants, True, none)
    var pairs = msel_split_finish(pin, plan, context)
    _ = labels  # the codes stay alive through the plan and the fill above
    return pairs


def msel_repeated_split(sp: MselSplitter, x: PythonObject, y: PythonObject, groups: PythonObject,
                        context: PythonObject) raises -> PythonObject:
    """`_Repeated.split`: repeat r is a shuffled run of the splitter with
    seed `_mix_seed(base, r)`. UNPACK is done once for all repeats (the
    labels are encoded once; before, every repeat built a splitter object
    and admitted and encoded y again); each repeat is one plan and one
    fill."""
    var base = msel_seed(sp.field("random_state", context), True, context)
    msel_repeated_check(base)
    var cv = sp.field("cv", context)
    var cvargs = sp.field("cvargs", context)
    var repeats = msel_int(sp.field("n_repeats", context))
    var pairs = Python.list()
    if repeats <= 0:
        return pairs
    var stratified = cv is msel_cls("StratifiedKFold", context)
    if not stratified and cv is not msel_cls("KFold", context):
        raise Error("TypeError: repeated model selection requires a native splitter configuration")
    var k = msel_check_splits(py_get(cvargs, PythonObject("n_splits"), PythonObject(5)), context)
    var n = 0
    var labels = MselLabelPin()
    if stratified:
        labels = _strat_labels(y, context)
    else:
        n = msel_n_samples(x, context)
    var view = labels.view
    var form = labels.form
    var pin = msel_split_route(context, stratified and not (k <= 256 and msel_fold_rows_fits(view.n, k)), True)
    var route = pin.route
    for r in range(repeats):  # driver-loop: over repeats (one plan and one fill each)
        var plan = MselSplitPlan()
        with GILReleased(Python()):
            # The one body of a repeat (bindings/msel_split_plan_run.mojo
            # `msel_plan_repeat`; a member fit calls it too).
            plan = msel_plan_repeat(route, n, view, form, k, stratified, base, r)
        _ = py_call(pairs.extend, msel_split_finish(pin, plan, context))
    _ = labels  # the codes stay alive through every repeat above
    return pairs


# ---------------------------------------------------------------- dispatch


def _msel_label_source(value: PythonObject, name: String, context: PythonObject) raises -> PythonObject:
    """The label input `msel_split` hands its consumers: a numeric vector
    as is, the stored numeric vector of a numpy-backed Series, a native copy
    of an exact flat int / bool list, else the label input step's list."""
    if msel_numeric_vector(value, False, context):
        return value
    if value is not None:
        # `value` was admitted with X and the other inputs by msel_split.
        var stored = labels_series_values(value, context)
        if stored is not None and msel_numeric_vector(stored, False, context):
            return stored
        if labels_list_integer_storage(value):
            return py_call(msel_b().list, value)
    return msel_labels_list(value, name, context)


def msel_split(model: PythonObject, var x: PythonObject, var y: PythonObject, var groups: PythonObject,
               context: PythonObject) raises -> PythonObject:
    """`<splitter>.split(X, y, groups)` of a mojolearn splitter: its splits
    (a list, or a lazy iterator for the per-row and per-combination
    splitters)."""
    # N070: the class code and the stored fields once per split call.
    var sp = MselSplitter(model, context)
    msel_admit(Python.tuple(x, y, groups), context)
    var y_used = sp.kind == SPLITTER_STRATIFIED_KFOLD or sp.kind == SPLITTER_STRATIFIED_GROUP_KFOLD or sp.kind == SPLITTER_STRATIFIED_SHUFFLE or sp.kind == SPLITTER_REPEATED_STRATIFIED_KFOLD
    var groups_used = sp.kind == SPLITTER_STRATIFIED_GROUP_KFOLD or sp.kind == SPLITTER_GROUP_KFOLD or sp.kind == SPLITTER_GROUP_SHUFFLE or sp.kind == SPLITTER_LOGO or sp.kind == SPLITTER_LPGO
    var x_used = not (sp.kind == SPLITTER_STRATIFIED_KFOLD or sp.kind == SPLITTER_STRATIFIED_SHUFFLE or sp.kind == SPLITTER_REPEATED_STRATIFIED_KFOLD or sp.kind == SPLITTER_PREDEFINED or sp.kind == SPLITTER_ITERABLE)
    # Split algorithms consume only X's row count. A native range captures it
    # once, including repeated folds, without reading caller buffer metadata
    # again after any compiled helper executes. Label leaves are pinned once.
    if x_used:
        x = py_call(msel_b().range, msel_n_samples(x, context))
    # Perf (lane perf-svm-and-msel): a numeric ndarray/Array label vector
    # holds no Python leaf to pin; every consumer below encodes it through
    # msel_encode (msel_label_codes), which reads it from its
    # buffer. Flattening it here boxed every label on every split call.
    # perf3-label-doors: a numpy-backed pandas Series whose stored exact
    # ndarray is such a vector is read through it (the labels its `tolist()`
    # lists, in the same order), and an exact flat list / tuple of exact ints
    # or bools becomes one native list copy: the label input step lists the
    # same ints / bools and admits every one of them.
    if y_used:
        y = _msel_label_source(y, "y", context)
    if groups_used:
        groups = _msel_label_source(groups, "groups", context)
    if sp.kind == SPLITTER_STRATIFIED_GROUP_KFOLD:
        return msel_strat_group_split(x, y, groups, msel_int(sp.field("n_splits", context)),
                                      msel_truth(sp.field("shuffle", context)), sp.field("random_state", context), context)
    if sp.kind == SPLITTER_GROUP_KFOLD:
        return msel_group_kfold_split(x, groups, msel_int(sp.field("n_splits", context)),
                                      msel_truth(sp.field("shuffle", context)), sp.field("random_state", context), context)
    if sp.kind == SPLITTER_STRATIFIED_KFOLD:
        return msel_stratified_split(y, msel_int(sp.field("n_splits", context)), msel_truth(sp.field("shuffle", context)),
                                     sp.field("random_state", context), context)
    if sp.kind == SPLITTER_KFOLD:
        return msel_kfold_split(x, msel_int(sp.field("n_splits", context)), msel_truth(sp.field("shuffle", context)),
                                sp.field("random_state", context), context)
    if sp.kind == SPLITTER_TIME_SERIES:
        return msel_time_series_split(sp, x, context)
    if sp.kind == SPLITTER_LOO:
        return msel_loo_split(x, sp, context)
    if sp.kind == SPLITTER_LPO:
        return msel_lpo_split(x, sp, context)
    if sp.kind == SPLITTER_LOGO:
        return msel_logo_split(x, groups, sp, context)
    if sp.kind == SPLITTER_LPGO:
        return msel_lpgo_split(x, groups, sp, context)
    if sp.kind == SPLITTER_PREDEFINED:
        return msel_predefined_split(sp, context)
    if (sp.kind == SPLITTER_SHUFFLE or sp.kind == SPLITTER_GROUP_SHUFFLE
            or sp.kind == SPLITTER_STRATIFIED_SHUFFLE):
        var n_splits = msel_int(sp.field("n_splits", context))
        var default_test = sp.field("_default_test_size", context)
        if sp.kind == SPLITTER_GROUP_SHUFFLE:
            return msel_group_shuffle_split(x, groups, n_splits, sp.field("test_size", context),
                                            sp.field("train_size", context), sp.field("random_state", context),
                                            default_test, context)
        if sp.kind == SPLITTER_STRATIFIED_SHUFFLE:
            return msel_strat_shuffle_split(y, n_splits, sp.field("test_size", context), sp.field("train_size", context),
                                            sp.field("random_state", context), default_test, context)
        return msel_shuffle_split(x, n_splits, sp.field("test_size", context), sp.field("train_size", context),
                                  sp.field("random_state", context), default_test, context)
    if sp.repeated():
        return msel_repeated_split(sp, x, y, groups, context)
    if sp.kind == SPLITTER_ITERABLE:
        return sp.field("_pairs", context)
    raise Error("NotImplementedError: ")


def msel_n_splits(model: PythonObject, x: PythonObject, y: PythonObject, groups: PythonObject,
                  context: PythonObject) raises -> PythonObject:
    """`<splitter>.get_n_splits(X, y, groups)`."""
    var b = msel_b()
    var sp = MselSplitter(model, context)  # N070: class code and fields once per call
    if sp.kind == SPLITTER_LOO:
        if x is None:
            raise Error("The 'X' parameter should not be None.")
        return PythonObject(msel_n_samples(x, context))
    if sp.kind == SPLITTER_LPO:
        return _comb(PythonObject(msel_n_samples(x, context)), sp.field("p", context))
    if sp.kind == SPLITTER_LOGO:
        return _distinct_group_count(groups, context)
    if sp.kind == SPLITTER_LPGO:
        var m = _distinct_group_count(groups, context)
        return _comb(m, sp.field("n_groups", context))
    if sp.kind == SPLITTER_PREDEFINED:
        var classes = msel_encode(sp.field("_folds", context), context)[0]
        return PythonObject(len(classes) - (1 if py_contains(classes, PythonObject(-1)) else 0))
    if sp.repeated():
        var cv = sp.field("cv", context)
        var cvargs = sp.field("cvargs", context)
        var repeats = sp.field("n_repeats", context)
        var per = PythonObject(None)
        if cv is msel_cls("KFold", context) or cv is msel_cls("StratifiedKFold", context):
            per = PythonObject(msel_check_splits(py_get(cvargs, PythonObject("n_splits"), PythonObject(5)), context))
        else:
            raise Error("TypeError: repeated model selection requires a native splitter configuration")
        return family_call(msel_op("mul"), per, repeats)
    if sp.kind == SPLITTER_ITERABLE:
        return family_call(b.len, sp.field("_pairs", context))
    return sp.field("n_splits", context)


def _distinct_group_count(groups: PythonObject, context: PythonObject) raises -> PythonObject:
    """`len(set(labels_list(groups)))`. N074: an integer ndarray / Array
    vector is counted from its buffer by the shared encoder (its classes are
    the distinct ints, the same count the set of the listed Python ints
    has) instead of boxing every label into a list and a set. Float, bool,
    str and mixed labels keep the listed set (NaN and 1 == 1.0 == True
    follow Python's set equality there)."""
    if groups is not None and msel_numeric_vector(groups, False, context) and _integer_vector(groups, context):
        return PythonObject(len(msel_encode(groups, context)[0]))
    var b = msel_b()
    return family_call(b.len, py_call(b.set, msel_labels_list(groups, "groups", context)))


def _integer_vector(value: PythonObject, context: PythonObject) raises -> Bool:
    # The dtype of a vector msel_numeric_vector just admitted (an exact
    # ndarray or the native Array; no other class reaches here).
    var ndarray = py_get(context, "ndarray")
    var typestr = String(value.dtype.str) if ndarray is not None and py_type(value) is ndarray else String(py=value.dtype)
    return typestr == "<i4" or typestr == "<i8" or typestr == "<u4"


def _comb(n: PythonObject, k: PythonObject) raises -> PythonObject:
    """`_comb(n, k)`: n choose k in exact integers, 0 outside [0, n]."""
    if msel_cmp("lt", k, PythonObject(0)) or msel_cmp("gt", k, n):
        return PythonObject(0)
    return family_call(native_loaded_module("math").comb, n, k)


# ---------------------------------------------------------------- check_cv


def msel_native_discrete(y: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_native_discrete(y)`: True / False, or None when the native test
    does not apply."""
    var b = msel_b()
    var arr = y
    if not msel_is_array(y, context):
        arr = msel_try_materialize(y, "y", Python.tuple(b.TypeError, b.ValueError, b.OverflowError), context)
        if msel_missed(arr, context):
            return PythonObject(None)
    var nd = Int(py=arr.ndim)
    if nd == 2 and Int(py=arr.shape[1]) == 1:
        nd = 1
    if nd != 1 or Int(py=arr.size) < 1:
        return PythonObject(None)
    var dtype = arr.dtype
    if not py_contains(context["family_msel_native_encode"], dtype):
        return PythonObject(None)
    var d = String(py=dtype)
    var bkey = base_helpers_key(context["family_msel_entries"])
    if d != "<f4" and d != "<f8":
        return PythonObject(True)
    var c = msel_materialize(arr, "y", context)
    var _keep_ret_c_r972 = PythonObject(base_reduce_stat_int(bkey, msel_addr(c),
                                                             Int(py=context["family_msel_native_code"][dtype]),
                                                             Int(py=arr.size),
                                                             Int(py=context["family_msel_reduce_integral"])) != 0)
    _ = c  # keep the Python object alive through the native call above
    var _keep_ret_c = _keep_ret_c_r972^
    _ = c  # keep the Python object alive through the native call above
    return _keep_ret_c^


def msel_check_cv(cv_in: PythonObject, y: PythonObject, classifier: Bool, shuffle: PythonObject,
                  random_state: PythonObject, context: PythonObject) raises -> PythonObject:
    """`check_cv(cv, y, classifier=..., shuffle=..., random_state=...)`."""
    var b = msel_b()
    var cv = PythonObject(5) if cv_in is None else cv_in
    if msel_is_integral(cv, context) and not msel_is_bool(cv, context):
        var stratify = False
        var native = msel_native_discrete(y, context) if classifier and y is not None else PythonObject(None)
        if native is not None:
            stratify = msel_truth(native)
        elif classifier and y is not None:
            var labels = msel_flatten(y, context)
            var itertools = native_loaded_module("itertools")
            var all_str = msel_truth(family_call(b.all, py_call(b.map, b.isinstance, labels, py_call(itertools.repeat, b.str))))
            if all_str:
                stratify = True
            else:
                # every label Integral, or a Real whose float is integer
                # valued (float.is_integer is False for NaN and infinity)
                var all_int = True
                var all_real = True
                for i in range(len(labels)):  # driver-loop: admitted scalar identities, without Python ABC callbacks
                    all_int = all_int and msel_is_integral(labels[i], context)
                    all_real = all_real and msel_is_real(labels[i], context)
                if all_int:
                    stratify = True
                elif all_real:
                    stratify = msel_truth(family_call(b.all, py_call(b.map, b.float.is_integer, py_call(b.map, b.float, labels))))
        var rs = random_state if msel_truth(shuffle) else PythonObject(None)
        var kind = String("StratifiedKFold") if stratify else String("KFold")
        return msel_new_kfold(kind, cv, shuffle, rs, context)
    if py_type(py_type(cv)) is not py_type(py_type(PythonObject(0))):
        raise Error("TypeError: cv cannot invoke Python metaclass hooks")
    if py_contains(context["family_msel_splitter_layouts"], py_type(cv)):
        _ = _splitter_fields(cv, context)
        return cv
    if py_type(cv) is not b.list and py_type(cv) is not b.tuple:
        raise Error("TypeError: cv must be an integer, native splitter, or list/tuple of index pairs; Python protocols cannot execute")
    return msel_iterable_cv(cv, context)


def msel_index_pairs(cv: PythonObject, context: PythonObject) raises -> PythonObject:
    """Copy admitted native fold pairs once, without caller iterator protocols."""
    var b = msel_b()
    if py_type(cv) is not b.list and py_type(cv) is not b.tuple:
        raise Error("TypeError: explicit folds require a list/tuple of index pairs, without Python iterator hooks")
    msel_admit(cv, context)
    var pairs = Python.list()
    for i in range(len(cv)):  # driver-loop: native fixed fold container
        var item = msel_parameter_values(cv[i], context)
        if len(item) != 2:
            raise Error("ValueError: each explicit fold must contain exactly train and test indices")
        var train = msel_index_copy(item[0], context)
        var test = msel_index_copy(item[1], context)
        _ = py_call(pairs.append, Python.tuple(train, test))
    return pairs


def msel_iterable_cv(cv: PythonObject, context: PythonObject) raises -> PythonObject:
    """`_IterableCV(cv)`: the caller's index pairs copied once."""
    var pairs = msel_index_pairs(cv, context)
    var cls = msel_cls("_IterableCV", context)
    var plan = _splitter_plan(cls, context)
    var out = native_storage_allocate(cls)
    py_setitem(py_call(plan[1], out, cls), "_pairs", pairs)
    return out


# ---------------------------------------------------------------- train_test_split


def msel_train_test_split(arrays: PythonObject, test_size: PythonObject, train_size: PythonObject,
                          random_state: PythonObject, shuffle: PythonObject, stratify: PythonObject,
                          context: PythonObject) raises -> PythonObject:
    """`train_test_split(*arrays, ...)`."""
    var b = msel_b()
    if len(arrays) == 0:
        raise Error("At least one array required as input")
    msel_admit(Python.tuple(arrays, test_size, train_size, random_state, shuffle, stratify), context)
    var prepared = Python.list()
    for i in range(len(arrays)):  # driver-loop: each caller input is prepared once before split execution
        var value = arrays[i]
        if py_type(value) is b.list or py_type(value) is b.tuple:
            value = py_call(b.tuple, value)
        else:
            value = msel_materialize(value, "split data", context)
        _ = py_call(prepared.append, value)
    var n = msel_n_samples(prepared[0], context)
    var lengths = Python.list()
    var mismatch = False
    for i in range(len(prepared)):  # driver-loop: compare only stable prepared input metadata
        var m = msel_n_samples(prepared[i], context)
        _ = py_call(lengths.append, PythonObject(m))
        mismatch = mismatch or m != n
    if mismatch:
        raise Error("Found input variables with inconsistent numbers of samples: " + family_str(lengths))
    var sizes = msel_sizes_of(n, test_size, train_size, PythonObject(0.25), context)
    var n_train = sizes[0]
    var n_test = sizes[1]
    var train = PythonObject(None)
    var test = PythonObject(None)
    if not msel_flag_truth(shuffle, context):
        if stratify is not None:
            raise Error("Stratified train/test split is not implemented for shuffle=False")
        train = msel_range_index(0, n_train, context)
        test = msel_range_index(n_train, n_train + n_test, context)
    else:
        var split: PythonObject
        if stratify is not None:
            # A numeric label vector is encoded from its buffer (msel_encode);
            # a numeric Series or an exact int list takes the one-read source
            # (perf3-label-doors), the rest the label input step.
            var strat = _msel_label_source(stratify, "stratify", context)
            split = msel_strat_shuffle_split(strat, 1, PythonObject(n_test), PythonObject(n_train), random_state,
                                             PythonObject(0.1), context)[0]
        else:
            split = msel_shuffle_split(py_call(b.range, n), 1, PythonObject(n_test), PythonObject(n_train), random_state,
                                       PythonObject(0.1), context)[0]
        train = split[0]
        test = split[1]
    var out = Python.list()
    for i in range(len(prepared)):  # driver-loop: gather train/test from the same retained input snapshot
        var value = prepared[i]
        if py_type(value) is b.list or py_type(value) is b.tuple:
            _ = py_call(out.append, msel_take_labels(value, train, context))
            _ = py_call(out.append, msel_take_labels(value, test, context))
            continue
        # pass 3 (rank 316): the input prepared above is stored once in one
        # resident entry and both sides gathered from it (msel_take_any
        # re-materialized it and opened, uploaded and closed a resident per
        # side: three host copies and two uploads per input).
        var res = msel_resident_new(context)
        try:
            _ = py_call(out.append, msel_resident_take(res, value, train, context))
            _ = py_call(out.append, msel_resident_take(res, value, test, context))
        finally:
            msel_resident_close(res, context)
    return out
