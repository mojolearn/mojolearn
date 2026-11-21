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
from bindings.arena_prog_run import MetricArenaOwner
from bindings.arena_prog_native_door import _metrics_run_plan, metrics_run_plan_entries
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.metrics_keys_unpack import metrics_keys
from bindings.py_balanced import py_item, py_get, py_type


struct MetricDoorProg(Movable):
    """A typed metric program and the door pins of its caller buffers."""
    var plan: MetricProgramPlan
    var pins: List[PythonObject]

    def __init__(out self):
        self.plan = MetricProgramPlan()
        self.pins = List[PythonObject]()

    def alloc(mut self, n: Int) raises -> Int:
        return self.plan.alloc(n)

    def put(mut self, array: PythonObject) raises -> Int:
        """A float32 (or int32, stored as its bits) C-ordered array -> its
        offset; the array is pinned until the run."""
        var off = self.plan.input(Int(py=array.size))
        self.pins.append(array)
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
    var entries = py_get(context, nk[].k_labels_metrics_entries)
    if entries is None:
        return _metrics_run_plan(plan, pins, mode, prog_context)
    if py_type(entries) is py_type(PythonObject("")):
        # the CPU image did not connect at import: its recorded words
        raise Error(String(entries))
    classical_connect_numpy(prog_context)
    return metrics_run_plan_entries(plan, pins, entries, prog_context)


def metric_door_run(prog: MetricDoorProg, mode: PythonObject, context: PythonObject) raises -> MetricArenaOwner:
    """`_run(prog, numeric_mode)` of a door program."""
    return metric_plan_run(prog.plan, prog.pins, mode, context)
