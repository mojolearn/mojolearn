# SPDX-License-Identifier: Apache-2.0
# Caller-owned native ABI output. Place this value in a one-element List before
# exposing its field addresses: the foreign call needs stable writable storage,
# not an inline value whose address crosses the ABI as an integer.
struct NativeCallOutput(Movable):
    var result: Int
    var error: List[UInt8]

    def __init__(out self):
        self.result = 0
        self.error = List[UInt8]()
