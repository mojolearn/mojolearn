# SPDX-License-Identifier: Apache-2.0
"""NN62 bounded explicit lifetime arena.

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


# TOMBSTONE: MOJOLEARN_NN64_SESSION_PACK (removed: its only caller, the mlp_sessions entry, is deleted; the inference sessions run packed as one forward of the SmallMLPTrainer program) and nn_session_move_kernel / nn_session_move_into (no caller) deleted 2026-11-09 by lane/m2-neural-leftovers; code recoverable at 0ce88411d3.
# Restore: git apply experiments/removed/M2_SMALL_MLP_PRIMITIVES.patch

