# SPDX-License-Identifier: Apache-2.0
# Native operation-owned progress for a resident greedy generation read.
# Boundary code admits the model/arrays once. No Python owner, callback or
# token-selection arithmetic crosses into this descriptor or its consumers.
struct ResidentGreedyRead(Movable):
    var finished: List[Int32]
    var lengths: List[Int64]
    var eos: List[Int32]
    var pad: Int
    var wrote: Int
    var consumed: Int
    var progress_ready: Bool
    var position: Int
    var max_positions: Int
    var state_capacity: Int
    var cache_remaining: Int

    def __init__(out self, finished: List[Bool], lengths: List[Int], eos: List[Int], pad: Int,
                 position: Int, max_positions: Int, state_capacity: Int):
        self.finished = List[Int32](capacity=len(finished))
        self.lengths = List[Int64](capacity=len(lengths))
        self.eos = List[Int32](capacity=len(eos))
        for value in finished:
            self.finished.append(Int32(value))
        for value in lengths:
            self.lengths.append(Int64(value))
        for value in eos:
            self.eos.append(Int32(value))
        self.pad = pad
        self.wrote = 0
        self.consumed = 0
        self.progress_ready = False
        self.position = position
        self.max_positions = max_positions
        self.state_capacity = state_capacity
        self.cache_remaining = 0


def resident_greedy_publish(progress: ResidentGreedyRead, mut finished: List[Bool], mut lengths: List[Int]):
    for r in range(len(finished)):
        finished[r] = progress.finished[r] != 0
        lengths[r] = Int(progress.lengths[r])


def resident_greedy_outputs(source: MutPointer[Int32, MutUntrackedOrigin],
                            output: MutPointer[Int32, MutUntrackedOrigin],
                            current: MutPointer[Int32, MutUntrackedOrigin],
                            batch: Int, capacity: Int, wrote: Int):
    # Final result packing only; token decisions have already run on device.
    from std.memory import memcpy
    for r in range(batch):
        memcpy(dest=output + r * wrote, src=source + r * capacity, count=wrote)
        current[r] = source[r * capacity + wrote - 1]
