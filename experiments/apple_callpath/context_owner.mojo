# SPDX-License-Identifier: Apache-2.0
"""Movable owner with stable context identity; no native-wrapper address tests."""
from max.gpu.host import DeviceBuffer, DeviceContext
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_FAST
from std.sys.info import has_apple_gpu_accelerator
from std.sys.compile import is_defined


struct CallpathContext(Movable):
    # Treat these as owned resources: never replace either independently.
    # Callpath methods accept this owner, not an unrelated raw DeviceContext.
    var device: DeviceContext
    var identity: DeviceBuffer[DType.int32]

    def __init__(out self) raises:
        comptime if not (GLOBAL_NUMERIC_MODE == NUMERIC_FAST
            and has_apple_gpu_accelerator()
            and is_defined["MOJOLEARN_APPLE_FAST_CALLPATH_CANDIDATES"]()):
            raise Error("callpath context requires opt-in FAST Apple")
        self.device = DeviceContext()
        # The address is identity metadata only; its numeric content is never
        # read. Slots retain aliases so allocation reuse cannot impersonate a
        # context while a slot survives. Moving the owner preserves the address.
        self.identity = self.device.enqueue_create_buffer[DType.int32](1)
        self.device.synchronize()
