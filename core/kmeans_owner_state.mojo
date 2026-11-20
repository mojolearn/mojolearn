# SPDX-License-Identifier: Apache-2.0
"""Resident-center identity and release state; storage pins live at PACK."""
from core.kmeans_native_api import KMeansNativeDriver


@fieldwise_init
struct KMeansResidentState(Movable):
    var owner: Int
    var centers: Int
    var binding: Int
    var connection: Int
    var clusters: Int
    var features: Int
    var handle: Int
    var driver: KMeansNativeDriver

    def matches(self, owner: Int, centers: Int, binding: Int, connection: Int,
                clusters: Int, features: Int) -> Bool:
        return self.handle >= 0 and self.owner == owner and self.centers == centers and self.binding == binding and self.connection == connection and self.clusters == clusters and self.features == features

    def take_handle(mut self) -> Int:
        var handle = self.handle
        self.handle = -1
        return handle
