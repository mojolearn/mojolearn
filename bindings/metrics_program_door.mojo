# SPDX-License-Identifier: Apache-2.0
# The one x_metrics program builder of the metric doors (lane
# metric-programs-2, 2026-11-12). It replaces the three copies that held the
# Python owners and a per-program Python opcode snapshot: `MxProg`
# (bindings/metrics_exp_common.mojo), `McProg` (bindings/mcls_common.mojo)
# and `_Prog` (bindings/metrics_reg_door.mojo).
#
# The program itself is typed (core/metrics_program_plan.mojo: the arena
# layout, the stage records through the typed op table, the declared
# outputs, program-owned constant words). The only Python here is the PINS:
# the caller buffers (admitted arrays) the program reads, held in input
# order until the shared runner has copied them into the arena. Same layout,
# same records, same declared outputs as before: the bits do not move.
from std.python import PythonObject
from core.metrics_program_plan import MetricProgramPlan
from bindings.arena_prog_run import MetricArenaOwner, METRIC_INPUT_TEMP
from bindings.arena_prog_native_door import (
    metrics_run_plan_entries, metrics_run_plan_slots, metric_route_of, metric_route_entries, MetricRoutePin,
)
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.metrics_keys_unpack import metrics_keys
from bindings.py_balanced import py_item, py_get, py_type
from bindings.x_metrics_resident_door import (
    metric_store_ref, metric_scope_word, metric_resident_of, metric_rows_gather, metric_store_release,
)
from core.entry_table import EntryRef, entry_ref_unresolved
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call
from x_metrics.descriptor_schema import MP_DEV_FIRST_NONFINITE
from std.python import Python
from std.python._cpython import GILReleased


struct MetricDoorProg(Movable):
    """A typed metric program and the door pins of its caller buffers."""
    var plan: MetricProgramPlan
    var pins: List[PythonObject]
    #: Empty unless a buffer input is a resident slot (`put_resident`); then
    #: one word per pin, -1 for a host buffer (lane vdf-xm-transport).
    var slots: List[Int]

    def __init__(out self):
        self.plan = MetricProgramPlan()
        self.pins = List[PythonObject]()
        self.slots = List[Int]()

    def alloc(mut self, n: Int) raises -> Int:
        return self.plan.alloc(n)

    def put(mut self, array: PythonObject) raises -> Int:
        """A float32 (or int32, stored as its bits) C-ordered array -> its
        offset; the array is pinned until the run."""
        var off = self.plan.input(Int(py=array.size))
        self.pins.append(array)
        if len(self.slots) > 0:
            self.slots.append(-1)
        return off

    def pin(mut self, array: PythonObject, keep: Bool):
        """Pin a buffer input a typed builder already laid out, in input
        order. `keep` False: a per-call temporary of the door (encoded codes,
        admitted weights) that the runner copies and never keeps resident in
        a `resident()` scope (`METRIC_INPUT_TEMP`); True: the caller's own
        buffer, which the runner reads from its resident copy inside a scope
        (put once per scope, found again by identity; lane vdf-xm-wire)."""
        if not keep and len(self.slots) == 0:
            self.slots = List[Int](length=len(self.pins), fill=-1)
        self.pins.append(array)
        if len(self.slots) > 0:
            self.slots.append(-1 if keep else METRIC_INPUT_TEMP)

    def put_resident(mut self, owner: PythonObject, words: Int, slot: Int) raises -> Int:
        """A buffer input already on the route image's resident store: `words`
        words of slot `slot` (put or gathered and checked by the door, which
        frees it after the run), `owner` (its DeviceRows) pinned until the
        run. The range route copies it device to device; it never reaches
        the host arena."""
        var off = self.plan.input(words)
        if len(self.slots) == 0:
            self.slots = List[Int](length=len(self.pins), fill=-1)
        self.pins.append(owner)
        self.slots.append(slot)
        return off

    def put_i32(mut self, array: PythonObject) raises -> Int:
        """int32 words (read with ldi), copied in as their bits."""
        return self.put(array)

    def put_words(mut self, var words: List[Int32]) raises -> Int:
        """Program-owned Int32 words (no Python owner)."""
        return self.plan.input_words(words^)

    def put_f64s(mut self, values: List[Float64]) raises -> Int:
        """Binary64 values as program-owned Int32 word pairs, low first."""
        return self.plan.input_f64s(values)

    def f64_words(mut self, value: Float64) raises -> Int:
        """`_f64_words`: a binary64 scalar as two Int32 words (low first), an input."""
        return self.plan.input_f64(value)

    def scratch(mut self, n: Int) raises -> Int:
        return self.plan.scratch(n)

    def stage(mut self, op: String, total: Int, params: List[Int]) raises:
        self.plan.stage_op(op, total, params)

    def want(mut self, off: Int, n: Int, count: Int = -1, mult: Int = 1) raises -> Int:
        """Declare an output range (`_Prog.want(off, n, count, mult)`)."""
        return self.plan.want(off, n, count, mult)

    def want_count(mut self, off: Int, n: Int, count: Int, mult: Int) raises -> Int:
        return self.plan.want(off, n, count, mult)

    def group(mut self, key: Int, n: Int, m: Int, values: Int, vstride: Int, weights: Int,
              width: Int) raises -> Tuple[Int, Int]:
        return self.plan.group(key, n, m, values, vstride, weights, width)

    def counts(mut self, off: Int, m: Int) raises -> Int:
        return self.plan.counts(off, m)

    def scan_flag(mut self, src: Int, n: Int, mode: Int) raises -> Int:
        return self.plan.scan_flag(src, n, mode)


def metric_plan_run(plan: MetricProgramPlan, pins: List[PythonObject], mode: PythonObject,
                    context: PythonObject) raises -> MetricArenaOwner:
    """`_run(prog, numeric_mode)`: a typed plan and its pins handed to the
    shared runner (bindings/arena_prog_native_door.mojo `_metrics_run_plan`,
    UNPACK of the input addresses + typed RUN, the GIL released).

    The image is the one `context` names: a `device="cpu"` estimator's CPU
    label scope (bindings/labels_descriptor_boundary_door.mojo `labels_scope`
    of a family context) carries the CPU x_metrics runner entries
    (`labels_metrics_entries`), which then run the plan; any other context
    names none and runs the process image of `mode`. One dictionary read
    with the interned key per program."""
    var nk = metrics_keys()
    var prog_context = py_item(context, nk[].k_family_metrics_prog_context)
    return metrics_run_plan_entries(plan, pins, _door_entries(mode, context), prog_context)


def _door_entries(mode: PythonObject, context: PythonObject) raises -> PythonObject:
    """The connected x_metrics image a door program of `context` runs on
    (`metric_plan_run`): the CPU label scope's entries, else `mode`'s."""
    var nk = metrics_keys()
    var prog_context = py_item(context, nk[].k_family_metrics_prog_context)
    var entries = py_get(context, nk[].k_labels_metrics_entries)
    if entries is None:
        return metric_route_entries(mode, prog_context)
    if py_type(entries) is py_type(PythonObject("")):
        # the CPU image did not connect at import: its recorded words
        raise Error(String(entries))
    classical_connect_numpy(prog_context)
    return entries


def metric_route_open(mode: PythonObject, context: PythonObject) raises -> MetricRoutePin:
    """UNPACK of a typed metric RUN (lane bm-meta-3, 2026-11-23): the program
    route `metric_door_run` would take for `mode` / `context`, typed
    (`pin.route`, bindings/arena_prog_run.mojo `MetricRoute`) for
    `metric_inputs_run`, its connection owners held by the pin."""
    return metric_route_of(_door_entries(mode, context))


def metric_door_run(prog: MetricDoorProg, mode: PythonObject, context: PythonObject) raises -> MetricArenaOwner:
    """`_run(prog, numeric_mode)` of a door program."""
    if len(prog.slots) > 0:
        var nk = metrics_keys()
        var prog_context = py_item(context, nk[].k_family_metrics_prog_context)
        return metrics_run_plan_slots(prog.plan, prog.pins, prog.slots, _door_entries(mode, context), prog_context)
    return metric_plan_run(prog.plan, prog.pins, mode, context)


# ---------------------------------------------------------------- resident X (lane vdf-xm-transport)


struct MetricRowsSlot(Movable):
    """A DeviceRows input of a metric door gathered on the one resident
    store (the x_prep image's): its slot (-1: not gathered; the door reads the
    host rows as before), its rows and columns, the route image's program
    entry (its scans) and the store's transport entry that owns the slot
    (`metric_slot_release`)."""
    var slot: Int
    var rows: Int
    var cols: Int
    var program: NativeDescriptorCall
    var store: EntryRef

    def __init__(out self, slot: Int, rows: Int, cols: Int, program: NativeDescriptorCall, store: EntryRef):
        self.slot = slot
        self.rows = rows
        self.cols = cols
        self.program = program
        self.store = store


def metric_door_rows_slot(value: PythonObject, mode: PythonObject, context: PythonObject,
                          mut owned: List[Int]) raises -> MetricRowsSlot:
    """UNPACK of a 2-D X that is a `DeviceRows` on a GPU image: the fold's
    rows gathered on the device out of its once-put base (the active
    `resident()` scope's x_prep cache keeps the base), into a slot the
    door owns (appended to `owned`; free with `metric_slot_release` after
    the run). Anything else (a host array, the host column, a CPU-only
    install, ranges off, a base that is not float32)
    returns slot -1 and the
    door admits X on the host, as before. One type compare for a host X."""
    var nk = metrics_keys()
    var prog_context = py_item(context, nk[].k_family_metrics_prog_context)
    var rows_class = py_get(prog_context, "maxabs_rows_class")
    var none = entry_ref_unresolved()
    if rows_class is None or py_type(value) is not rows_class:
        return MetricRowsSlot(-1, 0, 0, none.call, none)
    var pin = metric_route_open(mode, context)
    if not pin.route.store:
        return MetricRowsSlot(-1, 0, 0, none.call, none)
    var program = pin.route.program
    # an x_prep image that is not connected raises in words (lane vdf-xm-wire2)
    var store = metric_store_ref(program)
    var resident = metric_resident_of(metric_scope_word(prog_context), store)
    var shape = value.shape
    var slot = metric_rows_gather(value, store, owned, resident, prog_context)
    _ = pin^
    if slot < 0:
        return MetricRowsSlot(-1, 0, 0, none.call, none)
    return MetricRowsSlot(slot, Int(py=shape[0]), Int(py=shape[1]), program, store)


def metric_slot_first_nonfinite(rows: MetricRowsSlot) raises -> Int:
    """The first NaN or infinity among the gathered words, scanned on the
    device in place (`MP_DEV_FIRST_NONFINITE`, vdf-xm-labels'
    `metric_first_nonfinite_resident`), or -1: the host scan
    `host_all_finite_f32` of the same words, for a resident X."""
    var first = -1
    with GILReleased(Python()):
        first = native_descriptor_call(rows.program, MP_DEV_FIRST_NONFINITE, [], [rows.slot, rows.rows * rows.cols], [])
    return first


def metric_slot_release(rows: MetricRowsSlot, owned: List[Int]) -> String:
    """Free the slots a door's gather acquired (newest first); failure text,
    never raises."""
    if len(owned) == 0:
        return String()
    return metric_store_release(rows.store, owned)
