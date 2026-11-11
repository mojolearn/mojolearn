# SPDX-License-Identifier: Apache-2.0
"""Typed saved GP kernel metadata over the existing packed Float32 tables."""
from core.gp_native_api import GPKernelView


@fieldwise_init
struct GPArchiveScaleSpan(ImplicitlyCopyable, Movable):
    var start: Int
    var count: Int
    var next_offset: Int


@fieldwise_init
struct GPArchiveKernelPlan(Movable):
    var view: GPKernelView
    var parameter_count: Int
    var length_count: Int
    var expected_scales: Int

    def valid(self) -> Bool:
        var total = 0
        if self.length_count > 0:
            var lengths = Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=self.view.lengths)
            for i in range(self.length_count):  # driver-loop: saved postfix node metadata, same validation order
                total += Int(lengths[i])
        return self.parameter_count == self.view.nodes and self.length_count == self.view.nodes and total == self.expected_scales

    def kind(self, node: Int) -> Int:
        return Int(Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=self.view.kinds)[node])

    def parameter_value(self, node: Int) -> Float64:
        return Float64(Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=self.view.parameters)[node])

    def scale(self, index: Int) -> Float64:
        return Float64(Pointer[Float32, ImmUntrackedOrigin](unsafe_from_address=self.view.scales)[index])

    def span(self, node: Int, offset: Int) -> GPArchiveScaleSpan:
        var length = Int(Pointer[Int32, ImmUntrackedOrigin](unsafe_from_address=self.view.lengths)[node])
        var end = offset + length
        var size = self.view.scale_count
        # Preserve Python's step-one slice normalization, including negative
        # lengths/offsets and truncated saved tables; do not add refusals.
        var first = max(0, offset + size) if offset < 0 else min(offset, size)
        var last = max(0, end + size) if end < 0 else min(end, size)
        return GPArchiveScaleSpan(first, max(0, last - first), end)
