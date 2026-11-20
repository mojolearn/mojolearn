# SPDX-License-Identifier: Apache-2.0
"""Native auxiliary ownership and partial-update scalar state."""
from core.host_parallel_copy import host_parallel_copy_bytes
from core.kmeans_variants_api import VariantsSpan


@fieldwise_init
struct MiniBatchStreamState(Copyable, Movable):
    var init_mode: Int
    var init_size: Int
    var batch_size: Int
    var since_reassignment: Int
    var state_hi: Int
    var state_lo: Int


struct VariantsAuxiliary(Movable):
    var values: List[Float32]

    def __init__(out self, count: Int) raises:
        if count < 0 or count > 9223372036854775807 // 4:
            raise Error("ValueError: cluster auxiliary size exceeds native storage")
        self.values = List[Float32](length=count, fill=0)

    def copy_parts(mut self, parts: List[VariantsSpan]) raises:
        var offsets = List[Int]()
        var total = 0
        for part in parts:  # metadata: at most three admitted auxiliary spans
            if part.count < 0 or total > 9223372036854775807 - part.count:
                raise Error("ValueError: cluster auxiliary size exceeds native storage")
            offsets.append(total)
            total += part.count
        if total != len(self.values):
            raise Error("ValueError: cluster auxiliary spans differ from allocated storage")
        var destination = Int(self.values.unsafe_ptr())
        # Same one allocation and one copy of each source byte as the old
        # Array auxiliary block. Reuse the existing core-count/grain policy;
        # this metadata loop never copies individual data elements itself.
        for unit in range(len(parts)):  # driver-loop: at most three span-copy stages
            var part = parts[unit]
            if part.count:
                host_parallel_copy_bytes(destination + 4 * offsets[unit], part.address, part.count * 4)
        _ = offsets^
        _ = parts
        _ = self.values

    def span(self) -> VariantsSpan:
        return VariantsSpan(Int(self.values.unsafe_ptr()) if len(self.values) else 0, len(self.values))
