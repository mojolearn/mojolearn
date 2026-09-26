# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The `parallel_groves` forest prediction on the host, for a box with no GPU
(the CPU training lane, rf-clf-balanced-parallel and
et-reg-bootstrap-parallel, 2026-09-15).

HOST ONLY. Nothing here imports `max.gpu`, `max.gpu` or a `DeviceContext`.
`core/forest_inference.mojo` holds the device kernels and imports `max.gpu`
at module level, so its arithmetic is RESTATED here, not imported, with the
line of each routine it MIRRORS; a disagreement between the two is a finding.

WHAT IS MIRRORED. `inference_engine="parallel_groves"` predicts through the
resident snapshot (`bindings/forest_inference_binding.mojo`,
`core/forest_inference_model.mojo`), whose one launch is
`launch_forest_inference[RF_INPUT, GROVE=True]`
(`core/forest_inference.mojo:203-250`):

  * `reached_leaf` (`:47-79`, the separate-arrays layout; the packed layout
    of `MOJOLEARN_FOREST_PACKED_NODES` walks the same nodes to the same
    leaf): the finite-key compare of `finite_key` (`:39-44`, signed zeros
    equal, a subnormal kept), the RandomForest input flushed first
    (`RF_INPUT`), right child = left + 1.
  * `forest_vector_grove32_kernel` (`:148-200`, 2 to 8 outputs) and
    `forest_grove32_kernel` (`:101-133`, 1 or more than 8): per row and
    output, lane `g` of 32 adds trees `g, g+32, ...` through `forest_add`
    (`:34-36`, both operands and the result flushed) into a zero, then the
    shuffle-down reduction for step 16, 8, 4, 2, 1 (lane `< step` adds lane
    `+ step`; every lane read in a step is one no lane writes in it), then
    lane 0's sum flushed, divided by `Float32(trees)` through
    `identical_div` and flushed again. The vector kernel runs the same
    arithmetic per output; only the traversal is shared.
  * the refusals the device path raises: non-finite thresholds, leaves or
    inputs (`require_finite`, `:252-255`, and `require_finite_pointer`), a
    non-finite result, and a malformed tree (`validate_flat_forest`,
    `:258-301`, restated as the walk checks below).

THREADS (DEVIATION 2960, lane/forest-groves-cpu-and-speed, 2026-09-17).
`predict_into` fans its rows out to host threads the way DEVIATION 2900
fans the sequential walk out (`core/forest_host_predict.mojo`): the same
`MOJOLEARN_CPU_THREADS` reading, contiguous row ranges, a task never
under `FOREST_HOST_MIN_ROWS_PER_TASK` rows. Each row's arithmetic is
`_predict_row` below, the per-row body of the pre-2960 loop moved into a
function and otherwise untouched: 32 lane sums, lane `g` adding trees `g,
g+32, ...` in increasing order, the 16/8/4/2/1 fold, one division. A
thread owns whole rows and its own 32-lane scratch, so no output bit
depends on the thread count. Since this lane the shipped forest host
binding (`bindings/_mojolearn_forest_host.mojo`) runs this engine for a
`-parallel-groves-1` archive, as the rf and trees host families already
did for a GPU class on a CPU-only install.

TWO SABOTAGE ARMS, both default off. `MOJOLEARN_FOREST_HOST_SABOTAGE`
(the forest host gate's existing negative control) divides the fold by
`trees + 1` here as it does in the sequential walk, so the gate's
sabotage column moves every groves cell too. `MOJOLEARN_FOREST_GROVES_SABOTAGE`
(DEVIATION 2961) replaces the 16/8/4/2/1 fold with a left fold over the
32 lanes in lane order, the same 31 additions in another association and
nothing else, so a comparison against the GPU groves engine is watched to
FAIL on association alone; a fixture whose lane sums add exactly in both
orders reads IDENTICAL under it, and that is a fact about the fixture
(tools/forest_groves_identity.py reports it per cell).
"""
from max.algorithm import sync_parallelize
from std.memory import bitcast
from std.sys.compile import is_defined

from checks.numerics import ftz, identical_div
from core.forest_host_predict import (
    FOREST_HOST_SABOTAGE,
    host_task_count,
    host_worker_count,
)


comptime GROVE_LANES = 32

#: DEVIATION 2961: the association sabotage, read back by
#: `forest_host_groves_sabotage` and refused outside the gate by
#: `_forest_host.py`.
comptime FOREST_GROVES_SABOTAGE = is_defined["MOJOLEARN_FOREST_GROVES_SABOTAGE"]()


def _finite_key(value: Float32) -> UInt32:
    """`finite_key`, `core/forest_inference.mojo:39-44`."""
    var bits = bitcast[DType.uint32](value)
    if (bits & UInt32(0x7FFFFFFF)) == UInt32(0):
        bits = UInt32(0)
    if (bits & UInt32(0x80000000)) != UInt32(0):
        return ~bits
    return bits ^ UInt32(0x80000000)


def _forest_add(a: Float32, b: Float32) -> Float32:
    """`forest_add`, `core/forest_inference.mojo:34-36`."""
    return ftz(ftz(a) + ftz(b))


def _is_finite(value: Float32) -> Bool:
    return (bitcast[DType.uint32](value) & UInt32(0x7F800000)) != UInt32(0x7F800000)


struct HostGroveForest(Movable):
    """The resident snapshot's five arrays and dimensions, validated once at
    prepare as `ResidentForest.__init__` validates them."""

    var offsets: List[Int32]
    var columns: List[Int32]
    var thresholds: List[Float32]
    var left: List[Int32]
    var leaves: List[Float32]
    var features: Int
    var outputs: Int
    var trees: Int

    def __init__(
        out self,
        var offsets: List[Int32],
        var columns: List[Int32],
        var thresholds: List[Float32],
        var left: List[Int32],
        var leaves: List[Float32],
        features: Int,
        outputs: Int,
    ) raises:
        # `validate_flat_forest`, `core/forest_inference.mojo:258-301`.
        if features < 1 or outputs < 1 or len(offsets) < 2:
            raise Error("forest inference requires rows>=0, features/outputs/trees>=1")
        var nodes = len(columns)
        if nodes < 1 or nodes > 2147483647 // outputs:
            raise Error("forest inference leaf element count exceeds Int32 range")
        if len(thresholds) != nodes or len(left) != nodes or len(leaves) != nodes * outputs:
            raise Error("forest inference flat array shape mismatch")
        if offsets[0] != Int32(0) or Int(offsets[len(offsets) - 1]) != nodes:
            raise Error("forest inference offsets must cover all nodes")
        for i in range(nodes):
            if not _is_finite(thresholds[i]):
                raise Error("forest inference prototype requires finite Float32 values")
        for i in range(nodes * outputs):
            if not _is_finite(leaves[i]):
                raise Error("forest inference prototype requires finite Float32 values")
        for t in range(len(offsets) - 1):
            var base = Int(offsets[t])
            var end = Int(offsets[t + 1])
            if base < 0 or end <= base or end > nodes:
                raise Error("forest inference offsets must be strictly increasing")
            var seen = List[UInt8](length=end - base, fill=UInt8(0))
            var pending: List[Int] = [0]
            var visited = 0
            while len(pending) > 0:
                var local = pending.pop()
                if seen[local] != UInt8(0):
                    raise Error("forest inference requires acyclic trees without shared children")
                seen[local] = UInt8(1)
                visited += 1
                var node = base + local
                var child = Int(left[node])
                if child != -1:
                    if (
                        child < 0 or child + 1 >= end - base or columns[node] < Int32(0)
                        or Int(columns[node]) >= features
                    ):
                        raise Error("forest inference feature/child index out of bounds")
                    pending.append(child)
                    pending.append(child + 1)
            if visited != end - base:
                raise Error("forest inference tree contains unreachable nodes")
        self.trees = len(offsets) - 1
        self.offsets = offsets^
        self.columns = columns^
        self.thresholds = thresholds^
        self.left = left^
        self.leaves = leaves^
        self.features = features
        self.outputs = outputs

    def reached_leaf[RF_INPUT: Bool](
        self, x: MutPointer[Float32, MutUntrackedOrigin], tree: Int, row: Int
    ) -> Int:
        """`reached_leaf`, `core/forest_inference.mojo:47-79`."""
        var base = Int(self.offsets[tree])
        var node = base
        var child = Int(self.left[node])
        while child != -1:
            var value = x[row * self.features + Int(self.columns[node])]
            comptime if RF_INPUT:
                value = ftz(value)
            var go_left = _finite_key(value) <= _finite_key(self.thresholds[node])
            node = base + child + (0 if go_left else 1)
            child = Int(self.left[node])
        return node

    def _predict_row[RF_INPUT: Bool](
        self,
        x: MutPointer[Float32, MutUntrackedOrigin],
        output: MutPointer[Float32, MutUntrackedOrigin],
        sums: MutPointer[Float32, MutUntrackedOrigin],
        row: Int,
        outputs: Int,
        divisor: Float32,
    ):
        """One row of the grove kernels: `sums` is this thread's
        `GROVE_LANES * outputs` scratch. The body is the pre-2960 loop."""
        for lane in range(GROVE_LANES):
            for c in range(outputs):
                sums[lane * outputs + c] = Float32(0.0)
            var tree = lane
            while tree < self.trees:
                var node = self.reached_leaf[RF_INPUT](x, tree, row)
                for c in range(outputs):
                    sums[lane * outputs + c] = _forest_add(
                        sums[lane * outputs + c], self.leaves[node * outputs + c]
                    )
                tree += GROVE_LANES
        comptime if FOREST_GROVES_SABOTAGE:
            # DEVIATION 2961: a left fold over the lanes in lane order.
            for lane in range(1, GROVE_LANES):
                for c in range(outputs):
                    sums[c] = _forest_add(sums[c], sums[lane * outputs + c])
        else:
            var step = GROVE_LANES // 2
            while step > 0:
                for lane in range(step):
                    for c in range(outputs):
                        sums[lane * outputs + c] = _forest_add(
                            sums[lane * outputs + c], sums[(lane + step) * outputs + c]
                        )
                step //= 2
        for c in range(outputs):
            output[row * outputs + c] = ftz(identical_div(ftz(sums[c]), divisor))

    def predict_into[RF_INPUT: Bool](
        self,
        x: MutPointer[Float32, MutUntrackedOrigin],
        output: MutPointer[Float32, MutUntrackedOrigin],
        rows: Int,
        features: Int,
        outputs: Int,
        workers: Int = 0,
    ) raises:
        """`ResidentForest.predict_into` (`core/forest_inference_model.mojo`)
        through the grove kernels, row by row and output by output; the rows
        fan out to host threads (DEVIATION 2960), `workers` as
        `host_worker_count` reads it."""
        if features != self.features or outputs != self.outputs:
            raise Error("resident forest dimensions differ from prepared snapshot")
        if rows < 0 or rows > 2147483647 // features or rows > 2147483647 // outputs:
            raise Error("resident forest prediction dimensions exceed Int32")
        if rows == 0:
            return
        var divisor = Float32(self.trees)
        comptime if FOREST_HOST_SABOTAGE:
            divisor = Float32(self.trees + 1)
        var tasks = host_task_count(rows, host_worker_count(workers))
        var chunk = (rows + tasks - 1) // tasks
        # 1: a non-finite input in the task's rows (`require_finite_pointer`
        # on x, refused before any row of that task is walked); 2: a
        # non-finite result (`require_finite_pointer` on the output).
        var failed = List[Int](length=tasks, fill=0)
        var scratch = List[Float32](length=tasks * GROVE_LANES * outputs, fill=Float32(0.0))
        var sp = Pointer(to=self)
        var fp = failed.unsafe_ptr()
        var scp = scratch.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()

        def _rows_task(c: Int) {imm sp, imm fp, imm scp, imm x, imm output, imm chunk,
                                imm rows, imm features, imm outputs, imm divisor}:
            var lo = c * chunk
            var hi = lo + chunk
            if hi > rows:
                hi = rows
            for i in range(lo * features, hi * features):
                if not _is_finite(x[i]):
                    fp.unsafe_store(c, 1)
                    return
            var sums = scp + c * GROVE_LANES * outputs
            for row in range(lo, hi):
                sp[]._predict_row[RF_INPUT](x, output, sums, row, outputs, divisor)
            for i in range(lo * outputs, hi * outputs):
                if not _is_finite(output[i]):
                    fp.unsafe_store(c, 2)
                    return

        if tasks == 1:
            _rows_task(0)
        else:
            sync_parallelize(_rows_task, tasks)
        _ = len(scratch)
        for c in range(tasks):
            if failed[c] != 0:
                raise Error("forest inference prototype requires finite Float32 values")


struct HostGroveRegistry(Defaultable, Movable):
    """`ForestRegistry` (`core/forest_inference_model.mojo`): monotonically
    increasing handles, never pointers."""

    var entries: Dict[Int, HostGroveForest]
    var next_id: Int

    def __init__(out self):
        self.entries = Dict[Int, HostGroveForest]()
        self.next_id = 1

    def prepare(mut self, var model: HostGroveForest) raises -> Int:
        if self.next_id == 9223372036854775807:
            raise Error("resident forest handle space exhausted")
        var handle = self.next_id
        self.next_id += 1
        self.entries[handle] = model^
        return handle

    def release(mut self, handle: Int) raises:
        if handle not in self.entries:
            raise Error("unknown or released resident forest handle")
        var released = self.entries.pop(handle)
        _ = released^
