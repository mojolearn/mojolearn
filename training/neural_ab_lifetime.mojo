# SPDX-License-Identifier: Apache-2.0
"""NN62 bounded explicit lifetime arena; NN64 packed-session movement.

Source drafts, no compilation/verification. These components never infer
liveness from pointer equality. A native model owner must supply the exact
last device consumer (including backward/replay/error handling) and must retain
the arena until its final context drain. Public-model integration remains owed.
"""
from std.gpu import block_dim, block_idx, thread_idx
from std.sys.compile import is_defined, get_defined_int
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

# Arms 2 and 3 of MOJOLEARN_IDN_TRAIN_SCRATCH (training/neural_identical_experiments.mojo).
comptime NN62_LIFETIME_ARENA = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and get_defined_int["MOJOLEARN_IDN_TRAIN_SCRATCH", 0]() >= 2
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)
comptime NN64_SESSION_PACK = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and is_defined["MOJOLEARN_NN64_SESSION_PACK"]()
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)


@fieldwise_init
struct NeuralLiveRange(Copyable, Movable):
    var cells: Int
    var first_stage: Int
    var last_stage: Int


struct NeuralLifetimeArena(Movable):
    var slab: DeviceBuffer[DType.float32]
    var offsets: List[Int]
    var ranges: List[NeuralLiveRange]
    var cells: Int

    def __init__(out self, ctx: DeviceContext, ranges: List[NeuralLiveRange], byte_budget: Int) raises:
        comptime if not NN62_LIFETIME_ARENA:
            raise Error("NN62 lifetime arena is not enabled")
        if byte_budget < 4:
            raise Error("NN62 budget must hold at least one float")
        self.offsets = List[Int]()
        self.ranges = ranges.copy()
        var capacity = byte_budget // 4
        var high = 1
        # Metadata only: each range describes a model tensor, not a data row.
        # First fit is deterministic and entirely driven by live byte intervals.
        for i in range(len(ranges)):  # small-loop(ranges: model tensor live ranges): plans arena offsets from metadata
            ref item = ranges[i]
            if item.cells < 1 or item.cells > capacity or item.first_stage < 0 or item.last_stage < item.first_stage:
                raise Error("NN62 invalid live range or insufficient byte budget")
            var candidate = 0
            var moved = True
            while moved:
                moved = False
                for j in range(i):  # small-loop(i: earlier live ranges): first-fit over tensor metadata
                    ref previous = ranges[j]
                    var live_overlap = item.first_stage <= previous.last_stage and previous.first_stage <= item.last_stage
                    if live_overlap:
                        var end = self.offsets[j] + previous.cells
                        if candidate < end and self.offsets[j] < candidate + item.cells:
                            candidate = end
                            moved = True
                if candidate > capacity - item.cells:
                    raise Error("NN62 live scratch exceeds configured byte budget")
            self.offsets.append(candidate)
            high = max(high, candidate + item.cells)
        self.cells = high
        self.slab = ctx.enqueue_create_buffer[DType.float32](high)

    def view(mut self, slot: Int, stage: Int) raises -> DeviceBuffer[DType.float32]:
        if slot < 0 or slot >= len(self.ranges):
            raise Error("NN62 unknown scratch slot")
        ref live = self.ranges[slot]
        if stage < live.first_stage or stage > live.last_stage:
            raise Error("NN62 scratch accessed outside declared lifetime")
        # Producers must initialize every consumed cell. No implicit clear is
        # omitted by this allocator; its public caller's normal clears still run.
        return self.slab.create_sub_buffer[DType.float32](self.offsets[slot], live.cells)


def nn_session_move_kernel[SCATTER: Bool](
    packed: MutPointer[Float32, MutAnyOrigin],
    sessions: MutPointer[Float32, MutAnyOrigin],
    prefix: MutPointer[Int64, MutAnyOrigin],
    bases: MutPointer[Int64, MutAnyOrigin],
    n_sessions: Int32, rows: Int64, width: Int32,
):
    """Pack/scatter rows of compatible independent neural sessions exactly.

    prefix is nondecreasing session row offsets (n_sessions+1), bases gives
    each session's source row base; every destination range is disjoint.
    Descriptor admission is required before this internal kernel. Zero-length
    sessions are valid; upper-bound lookup skips them canonically.
    """
    var cell = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var w = Int(width)
    if cell >= Int(rows) * w:
        return
    var row = cell // w
    var feature = cell - row * w
    var lo = 0
    var hi = Int(n_sessions)
    while lo < hi:
        var mid = (lo + hi) // 2
        if prefix[mid + 1] <= Int64(row):
            lo = mid + 1
        else:
            hi = mid
    var source_row = Int(bases[lo]) + row - Int(prefix[lo])
    var source = source_row * w + feature
    comptime if SCATTER:
        sessions[source] = packed[cell]
    else:
        packed[cell] = sessions[source]


def nn_session_move_into[SCATTER: Bool](
    ctx: DeviceContext, mut packed: DeviceBuffer[DType.float32],
    mut sessions: DeviceBuffer[DType.float32], mut prefix: DeviceBuffer[DType.int64],
    mut bases: DeviceBuffer[DType.int64], n_sessions: Int, rows: Int, width: Int,
) raises:
    """Explicit component; does not yet batch model/cache execution.

    Owner supplies admitted compatible session descriptors and retains them
    through completion. No floating arithmetic is performed, including on
    signed zeros/subnormals. Pack/scatter cost belongs inside operation timing.
    """
    comptime if not NN64_SESSION_PACK:
        raise Error("NN64 session pack is not enabled")
    if n_sessions < 1 or n_sessions > 2147483647 or rows < 0 or width < 1 or width > 2147483647:
        raise Error("NN64 invalid session shape")
    if rows > 2147483647 // width:
        raise Error("NN64 packed size exceeds component launch bound")
    if rows == 0:
        return
    ctx.enqueue_function[nn_session_move_kernel[SCATTER]](packed.unsafe_ptr(), sessions.unsafe_ptr(), prefix.unsafe_ptr(), bases.unsafe_ptr(), Int32(n_sessions), Int64(rows), Int32(width), grid_dim=((rows * width + 127) // 128, 1, 1), block_dim=(128, 1, 1))
