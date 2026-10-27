# SPDX-License-Identifier: Apache-2.0
# Stable generation-layer metadata. No provider objects, ownership hooks or
# Python imports belong here. Mutable continuation values are arguments.


struct GenerationLayerWords(Copyable, Movable):
    var transformer: List[Int]
    var inner: Int
    var heads: Int
    var conv: Int
    var dt_low: Float64
    var dt_high: Float64

    def __init__(out self, var transformer: List[Int], inner: Int, heads: Int, conv: Int,
                 dt_low: Float64, dt_high: Float64):
        self.transformer = transformer^
        self.inner = inner
        self.heads = heads
        self.conv = conv
        self.dt_low = dt_low
        self.dt_high = dt_high

    def transformer_params(self, batch: Int, length: Int, capacity: Int, cached: Int) -> List[Int]:
        var result = self.transformer.copy()
        result[0] = batch
        result[1] = length
        result[7] = capacity
        result[8] = cached
        return result^

    def cache_count(self, batch: Int, capacity: Int) -> Int:
        var cap = capacity if self.transformer[9] == 0 else self.transformer[9]
        return batch * self.transformer[4] * cap * self.transformer[5]


def generation_transformer_addresses(
    x: Int, weights: List[Int], k_cache: Int, v_cache: Int, y: Int,
    optional: List[Int], packed: List[Int],
) -> List[Int]:
    var result = List[Int]()
    result.append(x)
    for j in range(len(weights)):  # driver-loop: over native weight-address slots
        result.append(weights[j])
    result.append(k_cache)
    result.append(v_cache)
    result.append(y)
    for j in range(len(optional)):  # driver-loop: over native optional-weight slots
        result.append(optional[j])
    for j in range(len(packed)):  # driver-loop: over native packed-weight slots
        result.append(packed[j])
    return result^
