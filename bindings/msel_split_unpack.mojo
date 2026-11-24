# SPDX-License-Identifier: Apache-2.0
"""UNPACK and PACK of the splitters (device wave, lane dw-msel-parallel,
2026-11-21). RUN is bindings/msel_split_plan_run.mojo.

UNPACK, once per split operation:
  * `msel_split_route`: the compiled entries the operation calls (the
    model-selection helper table, the base helper table and, for a splitter
    that draws or sorts, the x_metrics program runner and its two range
    switches). Before, the program runner was looked up again for every
    program a splitter ran (one per split of StratifiedShuffleSplit) and the
    base table key was read at every use.
  * `msel_label_codes`: a label or group vector as its int32 codes (the
    shared label encoder: classes in sorted order), pinned for the
    operation. Codes of another width are narrowed into native storage.
  * the seed (`msel_seed`, in the door: bindings/msel_split_door.mojo).

PACK:
  * `msel_split_allocate`: one (train, test) pair of int64 Arrays per split
    of a plan, sized by the plan, their addresses recorded in the plan for
    `msel_split_fill`.
  * `msel_split_finish`: the plan's warning, the allocation, the fill with
    the GIL released, the list of pairs.
"""
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from core.admitted_convert import admitted_convert_vector
from std.memory import unsafe_memcpy
from bindings.py_balanced import py_call, py_get, py_item
from bindings.metrics_keys_unpack import metrics_keys
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.arena_prog_native_door import metric_route_of, metric_route_entries
from bindings.msel_helpers_key_unpack import msel_helpers_key
from bindings.msel_common_door import (
    msel_b, msel_addr, msel_new, msel_encode, msel_labels_list, msel_warn,
    msel_is_bool, msel_is_integral, msel_is_real, msel_int, msel_op, msel_cmp,
)
from bindings.family_common_unpack import family_str
from bindings.family_common_pack import family_call
from bindings.msel_split_plan_run import (
    MselCodes, MselSplitRoute, MselSplitPlan, msel_split_route_none, msel_split_fill,
    MSEL_LABELS_ALIGNED, MselSizeWord, MSEL_SIZE_NONE, MSEL_SIZE_INT, MSEL_SIZE_REAL, MSEL_SIZE_BAD,
)


def msel_size_word(value: PythonObject, context: PythonObject) raises -> MselSizeWord:
    """UNPACK of one `_validate_shuffle_split` size (lane mf-split5; the
    door's `_size_kind`): None, an integer (a bool is refused), a real (a
    Python float, or a NumPy floating scalar read at its own width: its
    product with the row count is NumPy's at that width), else refused.
    `text` is the value's `str`, for the refusal words. A real scalar whose
    width is not 16, 32 or 64 bits (an extended-precision long double) is
    refused: no native width computes its product."""
    if value is None:
        return MselSizeWord(MSEL_SIZE_NONE, 0, 0.0, 64, String("None"))
    var text = family_str(value)
    if msel_is_bool(value, context):
        return MselSizeWord(MSEL_SIZE_BAD, 0, 0.0, 64, text)
    if msel_is_integral(value, context):
        return MselSizeWord(MSEL_SIZE_INT, msel_int(value), 0.0, 64, text)
    if msel_is_real(value, context):
        var bits = 64
        if py_type_is_float(value):
            bits = 64
        else:
            var width = Int(py=value.dtype.itemsize)
            if width == 4:
                bits = 32
            elif width == 2:
                bits = 16
            elif width != 8:
                return MselSizeWord(MSEL_SIZE_BAD, 0, 0.0, 64, text)
        return MselSizeWord(MSEL_SIZE_REAL, 0, Float64(py=value), bits, text)
    return MselSizeWord(MSEL_SIZE_BAD, 0, 0.0, 64, text)


def py_type_is_float(value: PythonObject) raises -> Bool:
    """`type(value) is float` (identity: no hook)."""
    return py_call(msel_b().type, value) is msel_b().float


def msel_size_sum(test: PythonObject, train: PythonObject, context: PythonObject) raises -> Tuple[Bool, String]:
    """UNPACK: for two real sizes, whether `train + test > 1` and the sum's
    `str` (the door's own arithmetic; neither depends on the row count).
    (False, "") otherwise."""
    var t = msel_size_word(test, context)
    var r = msel_size_word(train, context)
    if t.kind != MSEL_SIZE_REAL or r.kind != MSEL_SIZE_REAL:
        return (False, String(""))
    var total = family_call(msel_op("add"), train, test)
    return (msel_cmp("gt", total, PythonObject(1)), family_str(total))


struct MselRoutePin(Movable):
    """The route of one split operation and the Python owners of the
    program runner's connection (held for the operation)."""

    var route: MselSplitRoute
    var provider: PythonObject
    var entries: PythonObject

    def __init__(out self):
        self.route = msel_split_route_none()
        self.provider = PythonObject(None)
        self.entries = PythonObject(None)


def msel_split_route(context: PythonObject, program: Bool) raises -> MselRoutePin:
    """The entries of one split operation. `program`: it runs x_metrics
    programs (no splitter calls the base helper table since lane
    vdf-scope-split2, 2026-11-24, so its key is no longer read). The thread's
    resident scope is not consulted (its unread third flag went in arena
    unit E); the image's own resident store is (`route.store`, read once
    here by `metric_route_of` with the range switch: StratifiedKFold puts
    its label codes there once and both of its programs read the slot; lane
    vdf-scope-split2, 2026-11-24).

    The program runner is the default-mode x_metrics image's, connected
    once per factory: what `metric_door_run(prog, None, context)` resolved
    for every program (bindings/metrics_program_door.mojo,
    bindings/arena_prog_native_door.mojo `_metrics_run_plan`)."""
    var pin = MselRoutePin()
    pin.route.helpers = msel_helpers_key(context)
    if program:
        var nk = metrics_keys()
        var prog_context = py_item(context, nk[].k_family_metrics_prog_context)
        # The one runner UNPACK (bindings/arena_prog_native_door.mojo
        # `metric_route_of`, lane bm-meta-3: this was a copy of it).
        var got = metric_route_of(metric_route_entries(PythonObject(None), prog_context))
        pin.route.program = got.route.program
        pin.route.ranges = got.route.ranges
        pin.route.output_ranges = got.route.output_ranges
        pin.route.store = got.route.store
        pin.provider = got.provider
        pin.entries = got.entries
    return pin^


struct MselLabelPin(Movable):
    """A label or group vector as int32 codes, pinned for one operation:
    `view` is what RUN reads; `owner` (the encoder's code Array) or `narrow`
    (native storage, when the encoder's codes were another width) holds its
    words; `classes` are the encoder's classes (for messages and for
    PredefinedSplit's -1); `form` is the MSEL_LABELS_* code."""

    var view: MselCodes
    var owner: PythonObject
    var narrow: List[Int32]
    var classes: PythonObject
    var form: Int

    def __init__(out self):
        self.view = MselCodes(0, 0, 0)
        self.owner = PythonObject(None)
        self.narrow = List[Int32]()
        self.classes = PythonObject(None)
        self.form = MSEL_LABELS_ALIGNED


def msel_label_codes(values: PythonObject, n: Int, name: String, context: PythonObject, aligned: Bool,
                     refuse_none: Bool) raises -> MselLabelPin:
    """`_GroupCodes(values)`: the labels under the order rule as int32
    codes. `aligned`: one label per row is required (`n` rows), refused in
    the splitter's words otherwise; else the codes are taken as encoded
    (their own count). `refuse_none`: None is refused in the label input
    step's words before the encoder sees it."""
    var pin = MselLabelPin()
    if refuse_none and values is None:
        _ = msel_labels_list(values, name, context)
    var got = msel_encode(values, context)
    var classes = got[0]
    var codes = got[1]
    var size = Int(py=codes.size)
    if aligned and (Int(py=codes.ndim) != 1 or size != n):
        raise Error(name + " has " + String(size) + " labels for " + String(n) + " samples")
    var address = msel_addr(codes)
    var dtype = String(py=codes.dtype)
    if dtype != "<i4":
        # The label codes narrowed to int32 by the typed parallel conversion
        # (core/admitted_convert.mojo) into native storage.
        pin.narrow = List[Int32](length=max(size, 1), fill=0)
        if not admitted_convert_vector(address, dtype, Int(pin.narrow.unsafe_ptr()), "<i4", size):
            raise Error("TypeError: " + name + " label codes have dtype " + dtype + ", which no int32 conversion reads")
        _ = codes  # keep the Python object alive through the native call above
        address = Int(pin.narrow.unsafe_ptr())
    pin.owner = codes
    pin.classes = classes
    pin.view = MselCodes(address, size, len(classes))
    return pin^


def msel_label_array(pin: MselLabelPin, context: PythonObject) raises -> PythonObject:
    """The pinned codes as an int32 Array (the state a lazy group splitter
    carries from split to split): the encoder's own Array, or a copy of the
    narrowed words."""
    if len(pin.narrow) == 0:
        return pin.owner
    var copied = msel_new(pin.view.n, "<i4", context)
    if pin.view.n > 0:
        var dst = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=msel_addr(copied))
        var src = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=pin.view.address)
        unsafe_memcpy(dest=dst, src=src, count=pin.view.n)
        _ = copied  # keep the Python object alive through the native copy above
    return copied


def msel_split_allocate(mut plan: MselSplitPlan, context: PythonObject) raises -> PythonObject:
    """PACK, first half: one (train, test) pair of int64 Arrays per split,
    their addresses recorded for the fill."""
    var pairs = Python.list()
    for s in range(plan.splits()):  # driver-loop: over splits (output allocation only)
        var test = msel_new(plan.test_sizes[s], "<i8", context)
        var train = msel_new(plan.train_sizes[s], "<i8", context)
        plan.test_dst.append(msel_addr(test))
        plan.train_dst.append(msel_addr(train))
        _ = py_call(pairs.append, Python.tuple(train, test))
    return pairs


def msel_split_finish(pin: MselRoutePin, mut plan: MselSplitPlan, context: PythonObject) raises -> PythonObject:
    """The plan's warning (issued before any row is written, as before),
    the allocation, the rows with the GIL released, the list of (train,
    test) pairs."""
    if plan.warned:
        msel_warn(plan.warning, msel_b().UserWarning, context)
    var pairs = msel_split_allocate(plan, context)
    var route = pin.route
    with GILReleased(Python()):
        msel_split_fill(route, plan)
    _ = pairs  # every destination stays alive through the fill above
    _ = pin.provider
    return pairs
