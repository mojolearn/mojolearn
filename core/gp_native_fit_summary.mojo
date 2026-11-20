# SPDX-License-Identifier: Apache-2.0
"""GPC fitted likelihood aggregation over the existing native reduction."""
from core.hotpath_impl import reduce_stat, HP_F64, HP_SUM


struct GPCFitSummary(Movable):
    var values: List[Float64]

    def __init__(out self, count: Int):
        self.values = List[Float64](length=count, fill=0.0)

    def set(mut self, column: Int, likelihood: Float64):
        self.values[column] = likelihood

    def likelihood(self) raises -> Float64:
        var count = len(self.values)
        # Same per-class storage order, HP_F64/HP_SUM fold and division as
        # the former public-field readback. No kernel or reduction change.
        var total = reduce_stat(Int(self.values.unsafe_ptr()), HP_F64, count, HP_SUM).f
        _ = self.values
        return total / Float64(count)
