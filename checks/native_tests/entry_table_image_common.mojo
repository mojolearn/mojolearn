# SPDX-License-Identifier: Apache-2.0
"""Shared declarations only: the consumer must not import the provider body."""
from core.state_registry import StateRegistry

comptime IMAGE_TABLE = "MojolearnEntriesTestCrossImageIdenticalCPUV1"
comptime IMAGE_ABI = "mojolearn.test.cross-image.v1"


struct ImageState(Movable):
    var values: List[Int]

    def __init__(out self, value: Int):
        self.values = List[Int](length=3, fill=value)


comptime IMAGE_STATES = StateRegistry[ImageState, "MojolearnTestCrossImageStateV1"]
