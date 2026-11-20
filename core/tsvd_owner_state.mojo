# SPDX-License-Identifier: Apache-2.0
"""Native identity and single-release state for TSVD fitted-component leases."""
from core.tsvd_native_api import TSVDNativeDriver


@fieldwise_init
struct TSVDLeaseState(Movable):
    var connection: Int
    var generation: Int
    var components: Int
    var lease: Int
    var driver: TSVDNativeDriver

    def matches(self, connection: Int, generation: Int, components: Int) -> Bool:
        return (self.lease != 0 and self.connection == connection and self.generation == generation
                and self.components == components)

    def take_lease(mut self) -> Int:
        var lease = self.lease
        self.lease = 0
        return lease
