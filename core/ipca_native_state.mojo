# SPDX-License-Identifier: Apache-2.0
"""IncrementalPCA output storage and single-release resident identity."""
from core.ipca_native_api import IPCANativeDriver, IPCAOutputs


def ipca_output_shape(rows: Int, cols: Int, vector: Bool = False) -> List[Int]:
    var shape = List[Int](length=1, fill=rows)
    if not vector:
        shape.append(cols)
    return shape^


struct IPCAOutputRoom(Movable):
    var values: List[List[Float32]]

    def __init__(out self, features: Int):
        # Same six zeroed rooms, capacities and final prefix copies as the
        # original driver owner. Empty vectors still have a non-null dummy.
        self.values = List[List[Float32]]()
        self.values.append(List[Float32](length=max(features * features, 1), fill=Float32(0)))
        for _ in range(5):  # driver-loop: five vector outputs
            self.values.append(List[Float32](length=max(features, 1), fill=Float32(0)))

    def outputs(self) -> IPCAOutputs:
        return IPCAOutputs(Int(self.values[0].unsafe_ptr()), Int(self.values[1].unsafe_ptr()),
            Int(self.values[2].unsafe_ptr()), Int(self.values[3].unsafe_ptr()),
            Int(self.values[4].unsafe_ptr()), Int(self.values[5].unsafe_ptr()))


@fieldwise_init
struct IPCAResidentState(Movable):
    var connection: Int
    var generation: Int
    var source: Int
    var rows: Int
    var cols: Int
    var handle: Int
    var driver: IPCANativeDriver

    def matches(self, connection: Int, source: Int) -> Bool:
        return self.handle >= 0 and self.connection == connection and self.source == source

    def accept_generation(mut self, generation: Int) -> Bool:
        # Fit adoption is pending until the first query. A later public
        # dictionary change invalidates the cache exactly as before.
        if self.generation == -1:
            self.generation = generation
        return self.generation == generation

    def take_handle(mut self) -> Int:
        var handle = self.handle
        self.handle = -1
        return handle
